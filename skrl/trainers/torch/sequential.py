"""基于 PyTorch 的串行训练器模块。

提供 SequentialTrainer：同时（simultaneous）多智能体布置下，每次环境
交互中逐个智能体串行执行推理与记录（仍在同一进程内），主循环由本类持有。
"""

from __future__ import annotations

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
class SequentialTrainerCfg(TrainerCfg):
    """串行训练器的配置类（暂无本类特有字段，各项见 TrainerCfg）。"""


class SequentialTrainer(Trainer):
    def __init__(
        self,
        *,
        env: Wrapper | MultiAgentEnvWrapper,
        agents: Agent | MultiAgent | list[Agent] | list[MultiAgent],
        scopes: list[int] | None = None,
        cfg: SequentialTrainerCfg | dict = {},
    ) -> None:
        """串行训练器。

        多个智能体共享同一环境，每次交互中按顺序逐个执行（一个接一个），
        全部在同一进程内完成；动作经拼接后一次性推进环境。与并行训练器
        （ParallelTrainer）的区别在于不引入多进程。

        Args:
            env: 训练/评估用的环境：单智能体为 Wrapper，
                多智能体为 MultiAgentEnvWrapper。
            agents: 训练/评估的智能体：单个 Agent/MultiAgent，或由它们
                组成的列表（长度大于 1 时为同时多智能体布置）。
            scopes: 同时多智能体布置下各智能体负责的环境数量列表；
                为 None 或空列表时自动尽量平均分配。
            cfg: 配置：SequentialTrainerCfg 实例，或同结构的字典
                （字典会在传入基类前转换为配置对象）。
        """
        # 纯类型注解：供类型检查器把 self.cfg 细化为 SequentialTrainerCfg
        self.cfg: SequentialTrainerCfg
        super().__init__(
            env=env,
            agents=agents,
            scopes=scopes if scopes is not None else [],
            cfg=SequentialTrainerCfg(**cfg) if isinstance(cfg, dict) else cfg,
        )

        # 初始化智能体（同时多智能体布置时逐个初始化）
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.init(trainer_cfg=self.cfg)
        else:
            self.agents.init(trainer_cfg=self.cfg)

    def train(self) -> None:
        """串行训练智能体。

        非同时布置（num_simultaneous_agents == 1）时直接委托基类
        Trainer.train()。同时多智能体布置下循环执行以下步骤
        （标注"逐智能体"的步骤按 agents 顺序依次执行）：

        - 交互前钩子（逐智能体）
        - 计算动作（逐智能体，仅对自身 scope 的环境切片推理）
        - 与环境交互
        - 渲染环境
        - 记录转移（逐智能体，仅记录自身 scope 的数据）
        - 交互后钩子（逐智能体）
        - 重置环境（向量化环境无需显式重置）

        Raises:
            RuntimeError: 环境为单环境（num_envs == 1）。
                同时多智能体的串行训练仅支持并行/向量化环境。
        """
        # 设置运行模式为训练
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.enable_training_mode(True)
        else:
            self.agents.enable_training_mode(True)

        # 非同时智能体：退化为基类的单智能体循环
        if self.num_simultaneous_agents == 1:
            super().train()
            return

        # 重置环境，获取初始观测与（可选的）集中式状态
        observations, infos = self.env.reset()
        states = self.env.state()

        for timestep in tqdm.tqdm(range(self.cfg.timesteps), disable=self.cfg.disable_progressbar, file=sys.stdout):

            # 交互前钩子（逐智能体，算法子类可在此更新探索率等）
            for agent in self.agents:
                agent.pre_interaction(timestep=timestep, timesteps=self.cfg.timesteps)

            with torch.no_grad():
                # 计算动作：各智能体仅对自身环境区间 [scope[0], scope[1]) 的
                # 切片推理，输出形状 (scope[1]-scope[0], D_act)，暂存到列表
                _actions, _outputs = [], []
                for agent, scope in zip(self.agents, self.scopes):
                    with ScopedTimer() as timer:
                        actions, outputs = agent.act(
                            observations[scope[0] : scope[1]],
                            states[scope[0] : scope[1]] if states is not None else None,
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
                        observations=observations[scope[0] : scope[1]],
                        states=states[scope[0] : scope[1]] if states is not None else None,
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

                # 记录环境信息：默认读取 infos["episode"] 中数值为单元素
                # Tensor 的项；键名不含 "/" 时归入 "Info / " 分组，
                # 并对每个智能体各记录一份
                if self.cfg.environment_info in infos:
                    for k, v in infos[self.cfg.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            for agent in self.agents:
                                agent.track_data(k if "/" in k else f"Info / {k}", v.item())

            # 交互后钩子（逐智能体，算法子类通常在此触发网络更新）
            for agent in self.agents:
                agent.post_interaction(timestep=timestep, timesteps=self.cfg.timesteps)

            # 重置环境
            # - 并行/向量化环境（单或多智能体）：episode 边界由环境内部
            #   自动处理（步进时自动重置），直接沿用下一步观测
            if self.env.num_envs > 1:
                observations = next_observations
                states = next_states
            # - 单环境：同时多智能体的串行训练不支持
            else:
                raise RuntimeError("Sequential trainer is not supported for single environment")

    def eval(self) -> None:
        """串行评估智能体。

        非同时布置（num_simultaneous_agents == 1）时直接委托基类
        Trainer.eval()。同时多智能体布置下的循环与 train() 相同，
        差异在于：运行模式设为评估；stochastic_evaluation 为 False 时
        用确定性均值动作（"mean_actions"，缺失时回退采样动作）替代采样
        动作；post_interaction 调用智能体父类（Agent 基类）版本，跳过
        算法子类的覆写（其中通常包含网络更新）。

        Raises:
            RuntimeError: 环境为单环境（num_envs == 1）。
                同时多智能体的串行评估仅支持并行/向量化环境。
        """
        # 设置运行模式为评估
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.enable_training_mode(False)
        else:
            self.agents.enable_training_mode(False)

        # 非同时智能体：退化为基类的单智能体循环
        if self.num_simultaneous_agents == 1:
            super().eval()
            return

        # 重置环境，获取初始观测与（可选的）集中式状态
        observations, infos = self.env.reset()
        states = self.env.state()

        for timestep in tqdm.tqdm(range(self.cfg.timesteps), disable=self.cfg.disable_progressbar, file=sys.stdout):

            # 交互前钩子（逐智能体）
            for agent in self.agents:
                agent.pre_interaction(timestep=timestep, timesteps=self.cfg.timesteps)

            with torch.no_grad():
                # 计算动作：各智能体仅对自身环境区间的切片推理
                _actions, _outputs = [], []
                for agent, scope in zip(self.agents, self.scopes):
                    with ScopedTimer() as timer:
                        actions, outputs = agent.act(
                            observations[scope[0] : scope[1]],
                            states[scope[0] : scope[1]] if states is not None else None,
                            timestep=timestep,
                            timesteps=self.cfg.timesteps,
                        )
                        agent.track_data("Stats / Inference time (ms)", timer.elapsed_time_ms)
                    # 评估默认使用确定性均值动作；stochastic_evaluation 为
                    # True 时保留采样动作，缺失 "mean_actions" 键时回退采样动作
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

                # 记录转移：各智能体只记录自身环境区间内的数据
                # （评估阶段主要用于日志写出）
                for agent, scope in zip(self.agents, self.scopes):
                    agent.record_transition(
                        observations=observations[scope[0] : scope[1]],
                        states=states[scope[0] : scope[1]] if states is not None else None,
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

                # 记录环境信息（同 train()：对每个智能体各记录一份）
                if self.cfg.environment_info in infos:
                    for k, v in infos[self.cfg.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            for agent in self.agents:
                                agent.track_data(k if "/" in k else f"Info / {k}", v.item())

            # 调用父类（Agent 基类）的 post_interaction：只做日志写出与
            # 检查点，跳过算法子类的覆写（其中通常包含网络更新）
            for agent in self.agents:
                super(agent.__class__, agent).post_interaction(timestep=timestep, timesteps=self.cfg.timesteps)

            # 重置环境：逻辑与 train() 相同（向量化环境直接沿用下一步观测）
            # - 并行/向量化环境（单或多智能体）
            if self.env.num_envs > 1:
                observations = next_observations
                states = next_states
            # - 单环境：同时多智能体的串行评估不支持
            else:
                raise RuntimeError("Sequential trainer is not supported for single environment")
