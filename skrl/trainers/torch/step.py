"""基于 PyTorch 的单步训练器模块。

提供 StepTrainer：把训练/评估循环拆成单步，由调用方逐次调用
train()/eval() 驱动，适用于自定义调度或将强化学习嵌入外部仿真循环。
"""

from __future__ import annotations

from typing import Any

import copy
import dataclasses
import sys
import tqdm

import torch

from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import MultiAgentEnvWrapper, Wrapper
from skrl.multi_agents.torch import MultiAgent
from skrl.trainers.torch import Trainer, TrainerCfg
from skrl.utils import ScopedTimer


@dataclasses.dataclass(kw_only=True)
class StepTrainerCfg(TrainerCfg):
    """单步训练器的配置类（暂无本类特有字段，各项见 TrainerCfg）。"""


class StepTrainer(Trainer):
    def __init__(
        self,
        *,
        env: Wrapper | MultiAgentEnvWrapper,
        agents: Agent | MultiAgent | list[Agent] | list[MultiAgent],
        scopes: list[int] | None = None,
        cfg: StepTrainerCfg | dict = {},
    ) -> None:
        """单步训练器。

        与基类 Trainer 的"循环写法"不同，本类把每步交互封装成一次
        train()/eval() 调用，循环由调用方控制；环境观测与状态缓存在
        实例上跨调用保持。支持同时多智能体布置：各智能体只对自身
        scope 内的环境推理与记录。

        Args:
            env: 训练/评估用的环境：单智能体为 Wrapper，
                多智能体为 MultiAgentEnvWrapper。
            agents: 训练/评估的智能体：单个 Agent/MultiAgent，或由它们
                组成的列表（长度大于 1 时为同时多智能体布置）。
            scopes: 同时多智能体布置下各智能体负责的环境数量列表；
                为 None 或空列表时自动尽量平均分配。
            cfg: 配置：StepTrainerCfg 实例，或同结构的字典
                （字典会在传入基类前转换为配置对象）。
        """
        # 纯类型注解：供类型检查器把 self.cfg 细化为 StepTrainerCfg
        self.cfg: StepTrainerCfg
        super().__init__(
            env=env,
            agents=agents,
            scopes=scopes if scopes is not None else [],
            cfg=StepTrainerCfg(**cfg) if isinstance(cfg, dict) else cfg,
        )

        # 初始化智能体（同时多智能体布置时逐个初始化）
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.init(trainer_cfg=self.cfg)
        else:
            self.agents.init(trainer_cfg=self.cfg)

        # 内部时间步计数器：调用方未显式传入 timestep 时由此自增维护
        self._timestep = 0
        # 惰性创建的 tqdm 进度条（首次调用 train()/eval() 时创建）
        self._progress = None

        # 缓存的环境观测与集中式状态：首次调用 train()/eval() 时通过
        # reset 环境获得，之后每步更新；调用 reset() 清空后会重新 reset
        self.observations = None
        self.states = None

    def train(
        self, timestep: int | None = None, timesteps: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """执行一次（单步）训练迭代。

        每次调用执行以下步骤一遍（同时多智能体布置下，各智能体对自身
        scope 的环境依次执行）：

        - 交互前钩子（逐智能体）
        - 计算动作（逐智能体，仅对自身 scope 的环境切片推理）
        - 与环境交互
        - 渲染环境
        - 记录转移（逐智能体，仅记录自身 scope 的数据）
        - 交互后钩子（逐智能体）
        - 重置环境（单环境且全部结束时）

        Args:
            timestep: 当前时间步。为 None 时使用内部计数器自增后的值。
            timesteps: 总步数。为 None 时取训练器配置中的 timesteps
                （用于进度条总长度与传递给智能体的钩子）。

        Returns:
            元组 ``(next_observations, rewards, terminated, truncated, infos)``：
            本次环境步进的下一个观测、奖励、终止标志、截断标志与信息字典，
            沿环境维 E 组织（单智能体环境为 Tensor，多智能体环境为 dict）。
        """
        if timestep is None:
            # 调用方未指定时间步：内部计数器自增
            self._timestep += 1
            timestep = self._timestep
        timesteps = timesteps if timesteps is not None else self.cfg.timesteps

        if self._progress is None:
            # 首次调用：以总步数创建进度条，此后每次调用前进 1
            self._progress = tqdm.tqdm(total=timesteps, disable=self.cfg.disable_progressbar, file=sys.stdout)
        self._progress.update(n=1)

        # 简化后续处理的技巧：把单个智能体包成列表，统一走多智能体代码
        # 路径。注意该改写会保留在实例上（首次调用后 self.agents 变为列表）
        if not isinstance(self.agents, (tuple, list)):
            self.agents = [self.agents]

        # 设置运行模式为训练
        for agent in self.agents:
            agent.enable_training_mode(True)

        # 重置环境（仅首次调用或本类 reset() 之后）：缓存初始观测与状态
        if self.observations is None:
            self.observations, infos = self.env.reset()
            self.states = self.env.state()

        # 交互前钩子（算法子类可在此更新探索率等）
        for agent in self.agents:
            agent.pre_interaction(timestep=timestep, timesteps=timesteps)

        with torch.no_grad():
            # 计算动作：各智能体仅对自身环境区间 [scope[0], scope[1]) 的切片
            # 推理，输出形状 (scope[1]-scope[0], D_act)，暂存到列表
            _actions, _outputs = [], []
            for agent, scope in zip(self.agents, self.scopes):
                with ScopedTimer() as timer:
                    actions, outputs = agent.act(
                        self.observations[scope[0] : scope[1]],
                        self.states[scope[0] : scope[1]] if self.states is not None else None,
                        timestep=timestep,
                        timesteps=self.cfg.timesteps,
                    )
                    agent.track_data("Stats / Inference time (ms)", timer.elapsed_time_ms)
                _actions.append(actions)
                _outputs.append(outputs)
            # 沿环境轴拼接各智能体的动作：各 (scope_i, D_act) -> (E, D_act)，
            # 行顺序与 self.agents / self.scopes 一致
            actions = torch.vstack(_actions)

            # 推进环境一步，并读取下一步集中式状态；耗时对所有智能体统一记录
            with ScopedTimer() as timer:
                next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                next_states = self.env.state()
                elapsed_time_ms = timer.elapsed_time_ms
                for agent in self.agents:
                    agent.track_data("Stats / Env stepping time (ms)", elapsed_time_ms)

            # 渲染环境：非 headless 且 timestep 为渲染间隔的整数倍时执行
            # （timestep=0 也满足 0 % n == 0，即首步会渲染）
            if not self.cfg.headless and not timestep % self.cfg.render_interval:
                self.env.render()

            # 记录转移：各智能体只记录自身环境区间内的数据（infos 整份传入）
            for agent, scope in zip(self.agents, self.scopes):
                agent.record_transition(
                    observations=self.observations[scope[0] : scope[1]],
                    states=self.states[scope[0] : scope[1]] if self.states is not None else None,
                    actions=actions[scope[0] : scope[1]],
                    rewards=rewards[scope[0] : scope[1]],
                    next_observations=next_observations[scope[0] : scope[1]],
                    next_states=next_states[scope[0] : scope[1]] if next_states is not None else None,
                    terminated=terminated[scope[0] : scope[1]],
                    truncated=truncated[scope[0] : scope[1]],
                    infos=infos,
                    timestep=timestep,
                    timesteps=timesteps,
                )

            # 记录环境信息：默认读取 infos["episode"] 中数值为单元素
            # Tensor 的项；键名不含 "/" 时归入 "Info / " 分组
            if self.cfg.environment_info in infos:
                for k, v in infos[self.cfg.environment_info].items():
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        for agent in self.agents:
                            agent.track_data(k if "/" in k else f"Info / {k}", v.item())

        # 交互后钩子（算法子类通常在此触发网络更新）
        for agent in self.agents:
            agent.post_interaction(timestep=timestep, timesteps=timesteps)

        # 重置环境
        # - 并行/向量化环境（单或多智能体）：episode 边界由环境内部自动
        #   处理（步进时自动重置），直接沿用下一步观测并更新缓存
        if self.env.num_envs > 1:
            self.observations = next_observations
            self.states = next_states
        # - 单环境：显式判断是否需要 reset
        else:
            # 判断重置条件
            # - 多智能体：按包装器类型识别，活跃智能体列表为空时重置。
            if isinstance(self.env, MultiAgentEnvWrapper):
                should_reset = not self.env.agents
            # - 单智能体：任一环境终止或截断
            else:
                should_reset = terminated.any() or truncated.any()
            # 显式重置
            if should_reset:
                with torch.no_grad():
                    self.observations, infos = self.env.reset()
                    self.states = self.env.state()
            else:
                self.observations = next_observations
                self.states = next_states

        return next_observations, rewards, terminated, truncated, infos

    def eval(
        self, timestep: int | None = None, timesteps: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """执行一次（单步）评估迭代。

        单步流程与 train() 相同，差异在于：运行模式设为评估
        （enable_training_mode(False)）；stochastic_evaluation 为 False 时
        用确定性均值动作（"mean_actions"，缺失时回退采样动作）替代采样
        动作；post_interaction 调用智能体父类（Agent 基类）版本，跳过
        算法子类的覆写（其中通常包含网络更新）。

        Args:
            timestep: 当前时间步。为 None 时使用内部计数器自增后的值。
            timesteps: 总步数。为 None 时取训练器配置中的 timesteps。

        Returns:
            元组 ``(next_observations, rewards, terminated, truncated, infos)``：
            本次环境步进的下一个观测、奖励、终止标志、截断标志与信息字典，
            沿环境维 E 组织（单智能体环境为 Tensor，多智能体环境为 dict）。
        """
        if timestep is None:
            # 调用方未指定时间步：内部计数器自增
            self._timestep += 1
            timestep = self._timestep
        timesteps = timesteps if timesteps is not None else self.cfg.timesteps

        if self._progress is None:
            # 首次调用：以总步数创建进度条，此后每次调用前进 1
            self._progress = tqdm.tqdm(total=timesteps, disable=self.cfg.disable_progressbar, file=sys.stdout)
        self._progress.update(n=1)

        # 简化后续处理的技巧：把单个智能体包成列表（改写保留在实例上）
        if not isinstance(self.agents, (tuple, list)):
            self.agents = [self.agents]

        # 设置运行模式为评估
        for agent in self.agents:
            agent.enable_training_mode(False)

        # 重置环境（仅首次调用或本类 reset() 之后）：缓存初始观测与状态
        if self.observations is None:
            self.observations, infos = self.env.reset()
            self.states = self.env.state()

        # 交互前钩子
        for agent in self.agents:
            agent.pre_interaction(timestep=timestep, timesteps=timesteps)

        with torch.no_grad():
            # 计算动作：各智能体仅对自身环境区间的切片推理
            _actions, _outputs = [], []
            for agent, scope in zip(self.agents, self.scopes):
                with ScopedTimer() as timer:
                    actions, outputs = agent.act(
                        self.observations[scope[0] : scope[1]],
                        self.states[scope[0] : scope[1]] if self.states is not None else None,
                        timestep=timestep,
                        timesteps=self.cfg.timesteps,
                    )
                    agent.track_data("Stats / Inference time (ms)", timer.elapsed_time_ms)
                # 评估默认使用确定性均值动作；stochastic_evaluation 为 True
                # 时保留采样动作，缺失 "mean_actions" 键时回退采样动作
                _actions.append(actions if self.cfg.stochastic_evaluation else outputs.get("mean_actions", actions))
                _outputs.append(outputs)
            # 沿环境轴拼接：(scope_i, D_act) -> (E, D_act)，顺序与 scopes 一致
            actions = torch.vstack(_actions)

            # 推进环境一步，并读取下一步集中式状态
            with ScopedTimer() as timer:
                next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                next_states = self.env.state()
                elapsed_time_ms = timer.elapsed_time_ms
                for agent in self.agents:
                    agent.track_data("Stats / Env stepping time (ms)", elapsed_time_ms)

            # 渲染环境：条件同 train()（首步也会渲染）
            if not self.cfg.headless and not timestep % self.cfg.render_interval:
                self.env.render()

            # 记录转移：各智能体只记录自身环境区间内的数据（评估阶段
            # 主要用于日志写出）
            for agent, scope in zip(self.agents, self.scopes):
                agent.record_transition(
                    observations=self.observations[scope[0] : scope[1]],
                    states=self.states[scope[0] : scope[1]] if self.states is not None else None,
                    actions=actions[scope[0] : scope[1]],
                    rewards=rewards[scope[0] : scope[1]],
                    next_observations=next_observations[scope[0] : scope[1]],
                    next_states=next_states[scope[0] : scope[1]] if next_states is not None else None,
                    terminated=terminated[scope[0] : scope[1]],
                    truncated=truncated[scope[0] : scope[1]],
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.cfg.timesteps,
                )

            # 记录环境信息（同 train()：默认读取 infos["episode"] 中的
            # 单元素 Tensor 项，对每个智能体各记录一份）
            if self.cfg.environment_info in infos:
                for k, v in infos[self.cfg.environment_info].items():
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        for agent in self.agents:
                            agent.track_data(k if "/" in k else f"Info / {k}", v.item())

        # 调用父类（Agent 基类）的 post_interaction：只做日志写出与检查点，
        # 跳过算法子类的覆写（其中通常包含网络更新）
        for agent in self.agents:
            super(agent.__class__, agent).post_interaction(timestep=timestep, timesteps=self.cfg.timesteps)

        # 重置环境：逻辑与 train() 相同
        # - 并行/向量化环境（单或多智能体）
        if self.env.num_envs > 1:
            self.observations = next_observations
            self.states = next_states
        # - 单环境：显式判断
        else:
            # 判断重置条件
            # - 多智能体：按包装器类型识别，活跃智能体列表为空时重置。
            if isinstance(self.env, MultiAgentEnvWrapper):
                should_reset = not self.env.agents
            # - 单智能体：任一环境终止或截断
            else:
                should_reset = terminated.any() or truncated.any()
            # 显式重置
            if should_reset:
                with torch.no_grad():
                    self.observations, infos = self.env.reset()
                    self.states = self.env.state()
            else:
                self.observations = next_observations
                self.states = next_states

        return next_observations, rewards, terminated, truncated, infos

    def reset(self):
        """重置训练器的内部状态。

        清零内部时间步计数器，丢弃进度条与缓存的环境观测/状态；
        下次调用 train()/eval() 时会重新 reset 环境并新建进度条。
        不调用本方法时，train()/eval() 之间共享同一环境进度。
        """
        self._timestep = 0
        self._progress = None

        self.observations = None
        self.states = None
