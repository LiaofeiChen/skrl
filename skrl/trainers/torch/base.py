"""基于 PyTorch 的训练器基类模块。

提供 Trainer 抽象基类、TrainerCfg 配置数据类与 generate_equally_spaced_scopes
工具函数。Trainer 实现单智能体（单个 Agent 或 MultiAgent）的
"与环境交互 → 记录转移 → 智能体更新"训练/评估循环；同时（simultaneous）
多智能体、单步控制等变体由 step/sequential/parallel 子类实现。
"""

from __future__ import annotations

import atexit
import dataclasses
import sys
from abc import ABC
import tqdm

import torch

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import MultiAgentEnvWrapper, Wrapper
from skrl.multi_agents.torch import MultiAgent
from skrl.utils import ScopedTimer


def generate_equally_spaced_scopes(*, num_envs: int, num_simultaneous_agents: int) -> list[int]:
    """生成供多个同时（simultaneous）智能体使用的等分环境数量列表。

    将 num_envs 个并行环境尽量平均分配给 num_simultaneous_agents 个智能体：
    每个智能体先分得 int(num_envs / num_simultaneous_agents) 个环境，
    除法余数全部补到最后一份，使总数恰为 num_envs。

    Args:
        num_envs: 并行环境总数 E。
        num_simultaneous_agents: 同时智能体数量 N。

    Returns:
        长度为 N 的环境数量列表（不是区间元组），各项之和等于 num_envs；
        除余数外各项相等，余数全部计入最后一项。

    Raises:
        ValueError: 智能体数多于环境数（每份分得 0 个环境）时抛出。
    """
    # 每个智能体先分得整数商个环境
    scopes = [int(num_envs / num_simultaneous_agents)] * num_simultaneous_agents
    if sum(scopes):
        # 把除法余数全部补到最后一份，保证总数恰为 num_envs
        scopes[-1] += num_envs - sum(scopes)
    else:
        raise ValueError(
            f"The number of simultaneous agents ({num_simultaneous_agents}) is greater than the number of environments ({num_envs})"
        )
    return scopes


@dataclasses.dataclass(kw_only=True)
class TrainerCfg(ABC):
    """训练器配置的基类（仅支持关键字参数构造的 dataclass）。

    各配置项的含义见字段下方的文档字符串。
    """

    timesteps: int = 100000
    """训练/评估的总步数。"""

    headless: bool = False
    """是否以无界面模式运行（不调用 ``env.render()``）。"""

    render_interval: int = 1
    """渲染环境的时间步间隔，仅当 ``headless`` 为 False 时生效。"""

    disable_progressbar: bool | None = False
    """是否禁用进度条。为 None 时在非终端（non-TTY）输出下自动禁用。"""

    close_environment_at_exit: bool = True
    """程序正常退出时是否关闭环境。"""

    environment_info: str = "episode"
    """从环境 infos 中读取并记录日志所用的键（Gymnasium 惯例为 ``"episode"``）。"""

    stochastic_evaluation: bool = False
    """评估时是否使用采样的随机动作，而非（确定性的）均值动作。"""

    def validate(self) -> bool:
        """校验配置合法性。

        基类恒返回 True，子类按需覆写。
        """
        return True

    def expand(self) -> None:
        """展开配置（如把 ``"auto"`` 之类的约定值换算为具体数值）。

        基类不做任何操作，子类按需覆写。
        """
        pass


class Trainer(ABC):
    def __init__(
        self,
        *,
        cfg: TrainerCfg,
        env: Wrapper | MultiAgentEnvWrapper,
        agents: Agent | MultiAgent | list[Agent] | list[MultiAgent],
        scopes: list[int] | None = None,
    ) -> None:
        """实现自定义训练器的基类。

        本类负责：解析 agents/scopes（区分单智能体与同时多智能体两种布置，
        见 _setup_agents）、展开并持有配置、按配置在程序退出时关闭环境、
        在分布式运行的非主进程禁用进度条。单智能体布置可直接使用本类的
        train()/eval() 循环；同时多智能体布置须由子类重新实现这两个方法。

        Args:
            cfg: 训练器配置（TrainerCfg 或其子类实例）。
            env: 训练/评估用的环境：单智能体为 Wrapper，
                多智能体为 MultiAgentEnvWrapper。
            agents: 训练/评估的智能体：单个 Agent/MultiAgent，或由它们组成的
                列表（列表长度大于 1 时视为同时多智能体布置）。
            scopes: 同时多智能体布置下各智能体负责的环境数量列表，与
                agents 一一对应，总和须等于环境总数；为 None 或空列表时
                自动尽量平均分配。单智能体布置下忽略。
        """
        self.env = env
        self.agents = agents
        self.scopes = scopes if scopes is not None else []

        self.cfg = cfg
        # 展开配置（如换算 "auto" 约定值）
        self.cfg.expand()

        # 解析智能体布置（确定 num_simultaneous_agents，并把 scopes 归一化为区间）
        self.num_simultaneous_agents = 1
        self._setup_agents()

        # 按配置注册程序正常退出时关闭环境
        if self.cfg.close_environment_at_exit:

            @atexit.register
            def close_env():
                logger.info("Closing environment")
                self.env.close()
                logger.info("Environment closed")

        # 分布式运行时，非 0 号（非主）进程禁用进度条，避免重复输出信息/数据
        if config.torch.is_distributed:
            if config.torch.rank:
                self.cfg.disable_progressbar = True

    def __str__(self) -> str:
        """生成训练器的字符串表示（树状结构）。

        Returns:
            多行字符串：训练器类名、可并行环境数、同时智能体数，
            以及每个智能体（同时布置时）各自负责的环境区间
            ``(start:end)`` 与区间长度。
        """
        string = f"Trainer: {type(self).__name__}"
        string += f"\n  |-- Number of parallelizable environments: {self.env.num_envs}"
        string += f"\n  |-- Number of simultaneous agents: {self.num_simultaneous_agents}"
        string += "\n  |-- Agents and scopes:"
        if self.num_simultaneous_agents > 1:
            for agent, scope in zip(self.agents, self.scopes):
                string += f"\n  |     |-- agent: {type(agent)}"
                string += f"\n  |     |     |-- scope: {scope[1] - scope[0]} environments ({scope[0]}:{scope[1]})"
        else:
            string += f"\n  |     |-- agent: {type(self.agents)}"
            string += f"\n  |     |     |-- scope: {self.env.num_envs} environment(s)"
        return string

    def _setup_agents(self) -> None:
        """解析并设置同时（simultaneous）智能体布置。

        根据 agents 的类型与数量设定 num_simultaneous_agents，并把 scopes
        归一化为与 agents 一一对应的 (start, end) 环境区间列表：

        - agents 为长度 1 的列表：拆包为单个智能体，区间为
          ``(0, env.num_envs)``，num_simultaneous_agents 为 1；
        - agents 为长度大于 1 的列表：视为 N 个同时智能体；scopes 为空时
          自动尽量平均分配（余数补到最后一份），否则要求 scopes 数量与
          智能体数相同且总和恰为环境总数，最后把数量列表换算为区间；
        - agents 为单个对象（非列表）：同长度 1 的列表处理；
        - agents 为空列表：无效布置。

        Raises:
            ValueError: 布置无效：空列表；智能体数多于环境数（平均分配后
                每份为 0）；scopes 数量与智能体数不匹配；scopes 总和不等于
                环境总数。
        """
        # 校验智能体及其 scopes
        if isinstance(self.agents, (tuple, list)):
            # 单个智能体（长度为 1 的列表）：拆包，区间覆盖全部环境
            if len(self.agents) == 1:
                self.num_simultaneous_agents = 1
                self.agents = self.agents[0]
                self.scopes = [(0, self.env.num_envs)]
            # 同时多智能体
            elif len(self.agents) > 1:
                self.num_simultaneous_agents = len(self.agents)
                # 检查 scopes
                if not len(self.scopes):
                    # 未提供 scopes：警告后尽量平均分配，余数全部补到最后一份
                    logger.warning("The agents' scopes are empty. They will be generated to be as equal as possible")
                    self.scopes = [int(self.env.num_envs / len(self.agents))] * len(self.agents)
                    if sum(self.scopes):
                        self.scopes[-1] += self.env.num_envs - sum(self.scopes)
                    else:
                        raise ValueError(
                            f"The number of simultaneous agents ({len(self.agents)}) is greater than "
                            f"the number of parallelizable environments ({self.env.num_envs})"
                        )
                elif len(self.scopes) != len(self.agents):
                    # scopes 数量与智能体数不匹配
                    raise ValueError(
                        f"The number of simultaneous agents ({len(self.agents)}) doesn't match the number of scopes ({len(self.scopes)})"
                    )
                elif sum(self.scopes) != self.env.num_envs:
                    # scopes 未恰好覆盖全部环境
                    raise ValueError(
                        f"The scopes ({sum(self.scopes)}) don't cover the number of parallelizable environments ({self.env.num_envs})"
                    )
                # 把各份环境数量换算为 (start, end) 区间：对数量做前缀和，
                # 第 i 个区间为 (前 i 份数量之和, 前 i+1 份数量之和)
                index = 0
                for i in range(len(self.scopes)):
                    index += self.scopes[i]
                    self.scopes[i] = (index - self.scopes[i], index)
            else:
                raise ValueError("A list of agents is expected")
        # 非同时智能体（单个对象）：区间覆盖全部环境
        else:
            self.num_simultaneous_agents = 1
            self.scopes = [(0, self.env.num_envs)]

    def train(self) -> None:
        """训练单个智能体（单个 Agent 或 MultiAgent）。

        循环执行以下步骤：

        - 交互前钩子（pre_interaction）
        - 计算动作（act）
        - 与环境交互（env.step）
        - 渲染环境（render）
        - 记录转移（record_transition）
        - 交互后钩子（post_interaction）
        - 重置环境

        Notes:
            各交互数据沿环境维 E（env.num_envs）组织。单智能体环境
            （Wrapper）下：observations 为 Tensor (E, D_obs)，states 为
            Tensor (E, D_state) 或 None（未启用集中式状态），actions 为
            Tensor (E, D_act)；多智能体环境（MultiAgentEnvWrapper）下，
            三者均为 dict[智能体名 -> Tensor]，键由环境维护。

        Raises:
            AssertionError: 在同时多智能体布置下调用本方法。
                应继承 Trainer 并重新实现 train() 方法。
        """
        assert self.num_simultaneous_agents == 1, (
            "This method is not allowed for simultaneous agents. "
            "Inherit from `Trainer` and reimplement the `.train()` method instead."
        )

        # 重置环境，获取初始观测与（可选的）集中式状态
        observations, infos = self.env.reset()
        states = self.env.state()

        for timestep in tqdm.tqdm(range(self.cfg.timesteps), disable=self.cfg.disable_progressbar, file=sys.stdout):

            # 交互前钩子（算法子类可在此更新探索率等）
            self.agents.pre_interaction(timestep=timestep, timesteps=self.cfg.timesteps)

            with torch.no_grad():
                # 计算动作。outputs 为算法相关的输出字典（如 "mean_actions"、
                # "log_prob" 等键）；ScopedTimer 统计推理耗时并记录到跟踪数据
                with ScopedTimer() as timer:
                    actions, outputs = self.agents.act(
                        observations, states, timestep=timestep, timesteps=self.cfg.timesteps
                    )
                    self.agents.track_data("Stats / Inference time (ms)", timer.elapsed_time_ms)

                # 推进环境一步，并读取下一步集中式状态
                with ScopedTimer() as timer:
                    next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                    next_states = self.env.state()
                    self.agents.track_data("Stats / Env stepping time (ms)", timer.elapsed_time_ms)

                # 渲染环境：非 headless 且 timestep 为渲染间隔的整数倍时执行。
                # timestep=0 也满足 0 % n == 0，即首步会渲染
                if not self.cfg.headless and not timestep % self.cfg.render_interval:
                    self.env.render()

                # 记录本步转移（写入记忆与 TensorBoard 由智能体实现决定）
                self.agents.record_transition(
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
                    timesteps=self.cfg.timesteps,
                )

                # 记录环境信息：默认读取 infos["episode"]（Gymnasium 惯例）中
                # 数值为单元素 Tensor 的项；键名不含 "/" 时归入 "Info / " 分组
                if self.cfg.environment_info in infos:
                    for k, v in infos[self.cfg.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(k if "/" in k else f"Info / {k}", v.item())

            # 交互后钩子（算法子类通常在此触发网络更新，基类负责检查点与日志写出）
            self.agents.post_interaction(timestep=timestep, timesteps=self.cfg.timesteps)

            # 重置环境
            # - 并行/向量化环境（单或多智能体）：episode 边界由环境内部自动
            #   处理（步进时自动重置），直接沿用下一步观测即可
            if self.env.num_envs > 1:
                observations = next_observations
                states = next_states
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
                        observations, infos = self.env.reset()
                        states = self.env.state()
                else:
                    observations = next_observations
                    states = next_states

    def eval(self) -> None:
        """评估单个智能体（单个 Agent 或 MultiAgent）。

        循环结构与 train() 相同，交互数据的组织也与 train() 一致
        （见其 Notes），差异有两点：

        - stochastic_evaluation 为 False（默认）时，用 outputs 中的确定性
          均值动作（"mean_actions"）替代采样动作，缺失该键时回退用采样动作；
        - post_interaction 调用智能体父类（Agent 基类）版本，跳过算法
          子类的覆写（其中通常包含网络更新），评估阶段不触发训练。

        Raises:
            AssertionError: 在同时多智能体布置下调用本方法。
                应继承 Trainer 并重新实现 eval() 方法。
        """
        assert self.num_simultaneous_agents == 1, (
            "This method is not allowed for simultaneous agents. "
            "Inherit from `Trainer` and reimplement the `.eval()` method instead."
        )

        # 重置环境，获取初始观测与（可选的）集中式状态
        observations, infos = self.env.reset()
        states = self.env.state()

        for timestep in tqdm.tqdm(range(self.cfg.timesteps), disable=self.cfg.disable_progressbar, file=sys.stdout):

            # 交互前钩子
            self.agents.pre_interaction(timestep=timestep, timesteps=self.cfg.timesteps)

            with torch.no_grad():
                # 计算动作并记录推理耗时
                with ScopedTimer() as timer:
                    actions, outputs = self.agents.act(
                        observations, states, timestep=timestep, timesteps=self.cfg.timesteps
                    )
                    self.agents.track_data("Stats / Inference time (ms)", timer.elapsed_time_ms)
                # 评估默认使用确定性均值动作；stochastic_evaluation 为 True
                # 时保留采样动作，缺失 "mean_actions" 键时回退采样动作
                actions = actions if self.cfg.stochastic_evaluation else outputs.get("mean_actions", actions)

                # 推进环境一步，并读取下一步集中式状态
                with ScopedTimer() as timer:
                    next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                    next_states = self.env.state()
                    self.agents.track_data("Stats / Env stepping time (ms)", timer.elapsed_time_ms)

                # 渲染环境：条件同 train()（首步也会渲染）
                if not self.cfg.headless and not timestep % self.cfg.render_interval:
                    self.env.render()

                # 记录本步转移（评估阶段主要用于日志写出）
                self.agents.record_transition(
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
                    timesteps=self.cfg.timesteps,
                )

                # 记录环境信息（同 train()：默认读取 infos["episode"] 中的单元素 Tensor 项）
                if self.cfg.environment_info in infos:
                    for k, v in infos[self.cfg.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(k if "/" in k else f"Info / {k}", v.item())

            # 调用父类（Agent 基类）的 post_interaction：只做日志写出与检查点，
            # 跳过算法子类的覆写（其中通常包含网络更新）
            super(self.agents.__class__, self.agents).post_interaction(timestep=timestep, timesteps=self.cfg.timesteps)

            # 重置环境：逻辑与 train() 相同
            # - 并行/向量化环境（单或多智能体）
            if self.env.num_envs > 1:
                observations = next_observations
                states = next_states
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
                        observations, infos = self.env.reset()
                        states = self.env.state()
                else:
                    observations = next_observations
                    states = next_states
