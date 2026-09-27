"""独立近端策略优化（IPPO）多智能体实现。"""

from __future__ import annotations

from typing import Any

import itertools
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.multi_agents.torch import MultiAgent
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.utils import ScopedTimer

from .ippo_cfg import IPPO_CFG


def compute_gae(
    *,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    discount_factor: float = 0.99,
    lambda_coefficient: float = 0.95,
    time_limit_bootstrap: bool = False,
) -> torch.Tensor:
    """计算广义优势估计（GAE）。

    沿时间轴 T 倒序递推，环境维 E 向量化（每个环境独立一条轨迹）：

        delta_i = r_i - V(s_i) + gamma * not_done_i * (V(s_{i+1}) + lambda * A_{i+1})
        A_i = delta_i

    其中 V(s_T) 用 last_values 兜底；not_done 由终止信号取反得到，
    time_limit_bootstrap 为 True 时截断信号也视为完结（不 bootstrap）。

    :param rewards: 奖励 (T, E, 1)；T 为 rollout 步数（cfg.rollouts），
        E 为环境数，布局来自记忆存储（见 Memory.create_tensor）。
    :param terminated: 回合终止信号 (T, E, 1)，bool。
    :param truncated: 回合截断信号 (T, E, 1)，bool。
    :param values: 价值网络对每个样本的估计 (T, E, 1)。
    :param last_values: rollout 末端下一状态的价值估计 (E, 1)，
        对每个环境独立 bootstrap 序列尾部。
    :param discount_factor: 折扣因子 gamma。
    :param lambda_coefficient: TD(lambda) 系数。
    :param time_limit_bootstrap: 是否在超时截断处也停止 bootstrap；
        False 时截断样本仍向后续价值传递（时间限制引导）。

    :return: 回报 returns (T, E, 1) 与归一化后的优势 advantages (T, E, 1)，
        returns = advantages + values。
    """
    advantage = 0
    advantages = torch.zeros_like(rewards)
    not_done = ((terminated | truncated) if time_limit_bootstrap else terminated).logical_not()
    memory_size = rewards.shape[0]

    # 优势递推：沿时间轴倒序；advantage 以标量起步，首次迭代广播为 (E, 1)，
    # 逐时间步覆盖每个环境。rewards[i]/values[i]/next_values 均为 (E, 1)。
    for i in reversed(range(memory_size)):
        next_values = values[i + 1] if i < memory_size - 1 else last_values
        advantage = (
            rewards[i] - values[i] + discount_factor * not_done[i] * (next_values + lambda_coefficient * advantage)
        )
        advantages[i] = advantage
    # 回报 = 优势 + 价值（value target）
    returns = advantages + values
    # 优势标准化：整体减均值除标准差，数值稳定用 1e-8 防零
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages


class IPPO(MultiAgent):
    def __init__(
        self,
        *,
        possible_agents: list[str],
        models: dict[str, dict[str, Model]],
        memories: dict[str, Memory] | None = None,
        observation_spaces: dict[str, gymnasium.Space] | None = None,
        state_spaces: dict[str, gymnasium.Space] | None = None,
        action_spaces: dict[str, gymnasium.Space] | None = None,
        device: str | torch.device | None = None,
        cfg: IPPO_CFG | dict = {},
    ) -> None:
        """独立近端策略优化（IPPO）：每个智能体独立一份策略与价值网络。

        https://arxiv.org/abs/2011.09533

        :param possible_agents: 环境可能出现的全部智能体名字。
        :param models: 各智能体的模型，外层键为智能体身份，内层键为
            模型角色（policy、value）。
        :param memories: 各智能体的回放记忆。
        :param observation_spaces: 各智能体的观测空间。
        :param state_spaces: 各智能体的状态空间。
        :param action_spaces: 各智能体的动作空间。
        :param device: 数据分配与计算设备；未指定时使用默认设备。
        :param cfg: 多智能体配置；dict 会被展开为 IPPO_CFG。

        :raises KeyError: 配置键缺失。
        """
        self.cfg: IPPO_CFG
        super().__init__(
            possible_agents=possible_agents,
            models=models,
            memories=memories,
            observation_spaces=observation_spaces,
            state_spaces=state_spaces,
            action_spaces=action_spaces,
            device=device,
            cfg=IPPO_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )

        # 按角色取各智能体的策略与价值网络（未提供时为 None）
        self.policies = {uid: self.models[uid].get("policy", None) for uid in self.possible_agents}
        self.values = {uid: self.models[uid].get("value", None) for uid in self.possible_agents}

        # 登记进检查点模块表
        for uid in self.possible_agents:
            self.checkpoint_modules[uid]["policy"] = self.policies[uid]
            self.checkpoint_modules[uid]["value"] = self.values[uid]

        # 分布式运行时广播模型参数，保证各进程初始权重一致
        for uid in self.possible_agents:
            if config.torch.is_distributed:
                logger.info(f"Broadcasting models' parameters")
                if self.policies[uid] is not None:
                    self.policies[uid].broadcast_parameters()
                    # 策略与价值共用同一模型时只广播一次
                    if self.values[uid] is not None and self.policies[uid] is not self.values[uid]:
                        self.values[uid].broadcast_parameters()

        # 自动混合精度（AMP）梯度缩放器；按 torch 版本选择构造方式
        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self.cfg.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        # 逐智能体构建优化器与学习率调度器
        self.optimizers = {}
        self.schedulers = {}
        for uid in self.possible_agents:
            if self.policies[uid] is not None and self.values[uid] is not None:
                # - 优化器：策略与价值共用模型时只建一个；否则参数链合并
                if self.policies[uid] is self.values[uid]:
                    self.optimizers[uid] = torch.optim.Adam(
                        self.policies[uid].parameters(), lr=self.cfg.learning_rate[uid][0]
                    )
                else:
                    self.optimizers[uid] = torch.optim.Adam(
                        itertools.chain(self.policies[uid].parameters(), self.values[uid].parameters()),
                        lr=self.cfg.learning_rate[uid][0],
                    )
                self.checkpoint_modules[uid]["optimizer"] = self.optimizers[uid]
                # - 学习率调度器：展开后为 (policy 调度器, value 调度器)，这里取第 0 个
                self.schedulers[uid] = self.cfg.learning_rate_scheduler[uid][0]
                if self.schedulers[uid] is not None:
                    self.schedulers[uid] = self.cfg.learning_rate_scheduler[uid][0](
                        self.optimizers[uid], **self.cfg.learning_rate_scheduler_kwargs[uid][0]
                    )

        # 逐智能体构建观测/状态/价值预处理器；未配置时用空预处理器
        self._observation_preprocessor = {}
        self._state_preprocessor = {}
        self._value_preprocessor = {}
        for uid in self.possible_agents:
            # - 观测
            if self.cfg.observation_preprocessor[uid]:
                self._observation_preprocessor[uid] = self.cfg.observation_preprocessor[uid](
                    **self.cfg.observation_preprocessor_kwargs[uid]
                )
                self.checkpoint_modules[uid]["observation_preprocessor"] = self._observation_preprocessor[uid]
            else:
                self._observation_preprocessor[uid] = self._empty_preprocessor
            # - 状态
            if self.cfg.state_preprocessor[uid]:
                self._state_preprocessor[uid] = self.cfg.state_preprocessor[uid](
                    **self.cfg.state_preprocessor_kwargs[uid]
                )
                self.checkpoint_modules[uid]["state_preprocessor"] = self._state_preprocessor[uid]
            else:
                self._state_preprocessor[uid] = self._empty_preprocessor
            # - 价值输出
            if self.cfg.value_preprocessor[uid]:
                self._value_preprocessor[uid] = self.cfg.value_preprocessor[uid](
                    **self.cfg.value_preprocessor_kwargs[uid]
                )
                self.checkpoint_modules[uid]["value_preprocessor"] = self._value_preprocessor[uid]
            else:
                self._value_preprocessor[uid] = self._empty_preprocessor

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        """初始化智能体：先切模型为评估模式，再在记忆中建张量。

        :param trainer_cfg: 训练器配置。
        """
        super().init(trainer_cfg=trainer_cfg)
        self.enable_models_training_mode(False)

        # 在各智能体的记忆中创建存储张量；除标量字段外均按空间形状建立
        if self.memories:
            for uid in self.possible_agents:
                self.memories[uid].create_tensor(
                    name="observations", size=self.observation_spaces[uid], dtype=torch.float32
                )
                self.memories[uid].create_tensor(name="states", size=self.state_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="actions", size=self.action_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="rewards", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="terminated", size=1, dtype=torch.bool)
                self.memories[uid].create_tensor(name="truncated", size=1, dtype=torch.bool)
                self.memories[uid].create_tensor(name="log_prob", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="values", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="returns", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="advantages", size=1, dtype=torch.float32)

            # 更新阶段从记忆采样时需要的张量集合
            self._tensors_names = ["observations", "states", "actions", "log_prob", "values", "returns", "advantages"]

        # 存储与计算所需的临时变量：rollout 末端的 next 观测/状态、当步对数概率与价值
        self._current_next_observations = {}
        self._current_next_states = {}
        self._current_log_prob = {}
        self._current_values = {}
        self._rollout = 0

    def act(
        self,
        observations: dict[str, torch.Tensor],
        states: dict[str, torch.Tensor | None],
        *,
        timestep: int,
        timesteps: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """使用主策略处理环境观测/状态，给出决策（动作）。

        :param observations: 各智能体的观测 dict[agent_id, Tensor]，
            每个张量首维为环境轴 E。
        :param states: 各智能体的状态；无状态环境为 None。
        :param timestep: 当前时间步。
        :param timesteps: 总时间步数。

        :return: 各智能体的动作字典与附加输出字典（含对数概率）。
        """
        actions = {}
        log_prob = {}
        outputs = {}
        current_values = {}

        for uid in self.possible_agents:
            inputs = {
                "observations": self._observation_preprocessor[uid](observations[uid]),
                "states": self._state_preprocessor[uid](states[uid]),
            }
            # 随机动作采样
            # TODO: check for stochasticity（当前实现中该结果随后被策略采样覆盖）
            if timestep < self.cfg.random_timesteps:
                actions[uid], outputs[uid] = self.policies[uid].random_act(inputs, role="policy")

            # 策略采样（无条件执行，会覆盖上面随机动作的结果；
            # 采样是否随机由模型是否启用 DeterministicMixin 决定）
            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                actions[uid], outputs[uid] = self.policies[uid].act(inputs, role="policy")
                log_prob[uid] = outputs[uid]["log_prob"]

                # 训练模式下同步计算价值，供 GAE 使用；逆预处理还原到奖励量纲
                if self.training:
                    values, _ = self.values[uid].act(inputs, role="value")
                    current_values[uid] = self._value_preprocessor[uid](values, inverse=True)

        self._current_log_prob = log_prob
        self._current_values = current_values
        return actions, outputs

    def record_transition(
        self,
        *,
        observations: dict[str, torch.Tensor],
        states: dict[str, torch.Tensor | None],
        actions: dict[str, torch.Tensor],
        rewards: dict[str, torch.Tensor],
        next_observations: dict[str, torch.Tensor],
        next_states: dict[str, torch.Tensor],
        terminated: dict[str, torch.Tensor],
        truncated: dict[str, torch.Tensor],
        infos: dict[str, Any],
        timestep: int,
        timesteps: int,
    ) -> None:
        """记录一次环境转移：先走基类统计，再奖励整形并存入各智能体记忆。

        :param observations: 转移前观测，dict[agent_id, Tensor]。
        :param states: 转移前状态。
        :param actions: 各智能体执行的动作。
        :param rewards: 各智能体的即时奖励，通常 (E, 1)。
        :param next_observations: 转移后观测。
        :param next_states: 转移后状态。
        :param terminated: 各智能体的回合终止信号。
        :param truncated: 各智能体的回合截断信号。
        :param infos: 环境附加信息。
        :param timestep: 当前时间步。
        :param timesteps: 总时间步数。
        """
        super().record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )

        if self.training:
            # 缓存 rollout 末端的 next 观测/状态，供 update 计算 last_values
            self._current_next_observations = next_observations
            self._current_next_states = next_states

            for uid in self.possible_agents:
                # 奖励整形（如随时间衰减）
                if self.cfg.rewards_shaper is not None:
                    rewards[uid] = self.cfg.rewards_shaper(rewards[uid], timestep, timesteps)

                # 时间限制（截断）bootstrap：r += gamma * V(s') * truncated，
                # 把超时截断未被统计的未来回报并入最后一步奖励
                if self.cfg.time_limit_bootstrap[uid] and truncated[uid].any():
                    with torch.no_grad():
                        inputs = {
                            "observations": self._observation_preprocessor[uid](next_observations[uid]),
                            "states": self._state_preprocessor[uid](next_states[uid]),
                        }
                        next_values, _ = self.values[uid].act(inputs, role="value")
                        next_values = self._value_preprocessor[uid](next_values, inverse=True)

                    rewards[uid] += self.cfg.discount_factor[uid] * next_values * truncated[uid]

                # 转移写入该智能体的记忆
                self.memories[uid].add_samples(
                    observations=observations[uid],
                    states=states[uid],
                    actions=actions[uid],
                    rewards=rewards[uid],
                    terminated=terminated[uid],
                    truncated=truncated[uid],
                    log_prob=self._current_log_prob[uid],
                    values=self._current_values[uid],
                )

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        """与环境交互前调用的方法（本算法无操作）。

        :param timestep: 当前时间步。
        :param timesteps: 总时间步数。
        """
        pass

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        """与环境交互后调用：累计 rollout 步数，达到间隔后触发学习。

        :param timestep: 当前时间步。
        :param timesteps: 总时间步数。
        """
        if self.training:
            self._rollout += 1
            # 每 rollouts 步且已过 learning_starts，执行一轮逐智能体更新
            if not self._rollout % self.cfg.rollouts and timestep >= self.cfg.learning_starts:
                with ScopedTimer() as timer:
                    self.enable_models_training_mode(True)
                    for uid in self.possible_agents:
                        self.update(timestep=timestep, timesteps=timesteps, uid=uid)
                    self.enable_models_training_mode(False)
                    self.track_data("Stats / Algorithm update time (ms)", timer.elapsed_time_ms)

        # 写追踪数据与检查点（由基类处理）
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def update(self, *, timestep: int, timesteps: int, uid: str) -> None:
        """算法主更新步：GAE 重估 + PPO-clip 多轮 minibatch 梯度更新。

        :param timestep: 当前时间步。
        :param timesteps: 总时间步数。
        :param uid: 智能体身份。
        """
        policy = self.policies[uid]
        value = self.values[uid]
        memory = self.memories[uid]

        # 计算 returns 与 advantages：先取 rollout 末端 next 状态的价值作 bootstrap
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            inputs = {
                "observations": self._observation_preprocessor[uid](self._current_next_observations[uid]),
                "states": self._state_preprocessor[uid](self._current_next_states[uid]),
            }
            value.enable_training_mode(False)
            last_values, _ = value.act(inputs, role="value")
            value.enable_training_mode(True)
            last_values = self._value_preprocessor[uid](last_values, inverse=True)

        values = memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=memory.get_tensor_by_name("rewards"),
            terminated=memory.get_tensor_by_name("terminated"),
            truncated=memory.get_tensor_by_name("truncated"),
            values=values,
            last_values=last_values,
            discount_factor=self.cfg.discount_factor[uid],
            lambda_coefficient=self.cfg.gae_lambda[uid],
            time_limit_bootstrap=self.cfg.time_limit_bootstrap[uid],
        )

        # value_preprocessor 以训练模式拟合后写回记忆，供损失计算使用
        memory.set_tensor_by_name("values", self._value_preprocessor[uid](values, train=True))
        memory.set_tensor_by_name("returns", self._value_preprocessor[uid](returns, train=True))
        memory.set_tensor_by_name("advantages", advantages)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # 学习轮数
        for epoch in range(self.cfg.learning_epochs[uid]):
            kl_divergences = []

            # minibatch 循环：每次从记忆采样 (B, D)，B = len(memory) / mini_batches
            for (
                sampled_observations,
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in memory.sample(
                names=self._tensors_names, batch_size=len(memory), mini_batches=self.cfg.mini_batches[uid]
            ):

                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    # 首轮（epoch 0）用采样数据拟合 running 统计，之后冻结
                    inputs = {
                        "observations": self._observation_preprocessor[uid](sampled_observations, train=not epoch),
                        "states": self._state_preprocessor[uid](sampled_states, train=not epoch),
                    }

                    # 传入 taken_actions 重新评估旧动作在当前策略下的对数概率
                    _, outputs = policy.act({**inputs, "taken_actions": sampled_actions}, role="policy")
                    next_log_prob = outputs["log_prob"]

                    # 近似 KL 散度（k3 估计量）：E[e^x - 1 - x]，x = Δlog_prob
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # KL 超阈值提前终止本轮，防止策略更新过大
                    if self.cfg.kl_threshold[uid] and kl_divergence > self.cfg.kl_threshold[uid]:
                        break

                    # 熵损失：负的熵加权，鼓励探索
                    if self.cfg.entropy_loss_scale[uid]:
                        entropy_loss = -self.cfg.entropy_loss_scale[uid] * policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    # 策略损失（PPO-clip）：-min(ratio*A, clip(ratio)*A)
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self.cfg.ratio_clip[uid], 1.0 + self.cfg.ratio_clip[uid]
                    )

                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # 价值损失：预测值可先对旧值的偏移做限幅再算 MSE
                    predicted_values, _ = value.act(inputs, role="value")

                    if self.cfg.value_clip[uid] > 0:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values,
                            min=-self.cfg.value_clip[uid],
                            max=self.cfg.value_clip[uid],
                        )
                    value_loss = self.cfg.value_loss_scale[uid] * F.mse_loss(sampled_returns, predicted_values)

                # 优化步：三项损失合并反向，AMP 缩放
                self.optimizers[uid].zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

                # 分布式：跨进程规约梯度
                if config.torch.is_distributed:
                    policy.reduce_parameters()
                    if policy is not value:
                        value.reduce_parameters()

                # 梯度全局范数裁剪（AMP 下需先还原缩放）
                if self.cfg.grad_norm_clip[uid] > 0:
                    self.scaler.unscale_(self.optimizers[uid])
                    if policy is value:
                        nn.utils.clip_grad_norm_(policy.parameters(), self.cfg.grad_norm_clip[uid])
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(policy.parameters(), value.parameters()), self.cfg.grad_norm_clip[uid]
                        )

                self.scaler.step(self.optimizers[uid])
                self.scaler.update()

                # 累计损失供统计
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self.cfg.entropy_loss_scale[uid]:
                    cumulative_entropy_loss += entropy_loss.item()

            # 学习率调度：KLAdaptiveLR 用本轮 minibatch KL 均值调步长
            if self.schedulers[uid]:
                if isinstance(self.schedulers[uid], KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    # 分布式运行时跨进程求和再平均，保证各进程调度一致
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.schedulers[uid].step(kl.item())
                else:
                    self.schedulers[uid].step()

        # 记录统计：损失按 epoch × minibatch 数取平均
        self.track_data(
            f"Loss / Policy loss ({uid})",
            cumulative_policy_loss / (self.cfg.learning_epochs[uid] * self.cfg.mini_batches[uid]),
        )
        self.track_data(
            f"Loss / Value loss ({uid})",
            cumulative_value_loss / (self.cfg.learning_epochs[uid] * self.cfg.mini_batches[uid]),
        )
        if self.cfg.entropy_loss_scale[uid]:
            self.track_data(
                f"Loss / Entropy loss ({uid})",
                cumulative_entropy_loss / (self.cfg.learning_epochs[uid] * self.cfg.mini_batches[uid]),
            )

        self.track_data(f"Policy / Standard deviation ({uid})", policy.distribution(role="policy").stddev.mean().item())

        if self.schedulers[uid]:
            self.track_data(f"Learning / Learning rate ({uid})", self.schedulers[uid].get_last_lr()[0])
