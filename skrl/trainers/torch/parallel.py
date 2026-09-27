"""基于 PyTorch 多进程的并行训练器模块。

提供 ParallelTrainer：每个同时（simultaneous）智能体在独立进程中运行
推理与更新，主进程负责环境步进与调度。进程间协作方式：

- 单向管道（主进程 -> worker）：下发任务指令（dict，含 "task" 及参数）；
- 队列（每 worker 一条，双向）：收发张量等数据（观测、动作、奖励等）；
- barrier：全体（所有 worker + 主进程）在各任务结束处同步。



MAIN PROCESS
────────────────────────────────────────

① pre_interaction command
         │
         ├─────────┬─────────┐
         ↓         ↓         ↓
       worker0   worker1   worker2
         │         │         │
         └──── Barrier ──────┘

② act command + obs/state
         │
         ├─────────┬─────────┐
         ↓         ↓         ↓
       agent0    agent1    agent2
        act       act       act
         │         │         │
         └── actions ────────┘
                  ↓
              vstack
                  ↓
③             env.step()
                  ↓
       reward / next_obs / done
                  ↓

④ record_transition command
         │
         ├─────────┬─────────┐
         ↓         ↓         ↓
      memory0    memory1    memory2
         └──── Barrier ──────┘

⑤ post_interaction command
         │
         ├─────────┬─────────┐
         ↓         ↓         ↓
       update0   update1   update2
         └──── Barrier ──────┘

⑥ observations.copy_(next_observations)

进入下一 timestep
"""

from __future__ import annotations

import copy
import dataclasses
import sys
import tqdm

import torch
import torch.multiprocessing as mp

from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import MultiAgentEnvWrapper, Wrapper
from skrl.multi_agents.torch import MultiAgent
from skrl.trainers.torch import Trainer, TrainerCfg
from skrl.utils import ScopedTimer


def fn_processor(process_index, *args):
    """同时智能体工作进程的主函数（每个智能体一个进程）。

    循环从本进程的管道接收主进程的任务指令并执行，智能体对象与各类
    张量数据经本进程的队列收发；每个任务结束时在 barrier 上与主进程
    （及其他 worker）会合同步，保证阶段边界一致。

    Args:
        process_index: 本进程/智能体的编号（从 0 起），用于索引自身的
            管道、队列与环境区间。
        args: 按位置约定的参数组：

            - ``args[0]``：各进程管道读端列表（本进程取
              ``args[0][process_index]``）；
            - ``args[1]``：各进程队列列表（本进程取 ``args[1][process_index]``）；
            - ``args[2]``：全体（所有 worker + 主进程）共用的 barrier；
            - ``args[3]``：各智能体的 (start, end) 环境区间列表；
            - ``args[4]``：训练器配置。
    """
    print(f"[INFO] Processor {process_index}: started")

    # 解包本进程专属的通信端与配置
    pipe = args[0][process_index]  # 任务指令接收端（写端由主进程持有）
    queue = args[1][process_index]  # 数据收发队列（与主进程共用）
    barrier = args[2]  # 全体会合屏障
    scope = args[3][process_index]  # 本智能体负责的 (start, end) 环境区间
    trainer_cfg = args[4]

    # 本进程内的智能体，以及最近一次 "act" 任务缓存的观测/状态/动作
    # （均已切到本智能体的环境区间），供后续 "record_transition" 任务复用
    agent = None
    _observations = None
    _states = None
    _actions = None

    # 等待主进程启动所有 worker
    barrier.wait()

    while True:
        # 阻塞等待主进程的任务指令
        msg = pipe.recv()
        task = msg["task"]

        # 终止进程：跳出循环结束本进程
        if task == "terminate":
            break

        # 初始化智能体：从队列取出主进程发来的智能体对象（spawn 模式下
        # 经 pickle 传输；已共享内存的模型/记忆传的是共享存储句柄）并初始化
        elif task == "init":
            agent = queue.get()
            agent.init(trainer_cfg=trainer_cfg)
            print(f"[INFO] Processor {process_index}: init agent {type(agent).__name__} with scope {scope}")
            barrier.wait()

        # 执行智能体的交互前钩子（算法子类可在此更新探索率等）
        elif task == "pre_interaction":
            agent.pre_interaction(timestep=msg["timestep"], timesteps=msg["timesteps"])
            barrier.wait()

        # 计算动作：按主进程 put 的顺序依次从队列取出观测与状态（FIFO），
        # 切到本智能体的环境区间后推理，动作经队列回传主进程
        elif task == "act":
            _observations = queue.get()[scope[0] : scope[1]]
            _states = queue.get()
            if _states is not None:
                _states = _states[scope[0] : scope[1]]
            with torch.no_grad():
                with ScopedTimer() as timer:
                    _actions, _outputs = agent.act(
                        _observations, _states, timestep=msg["timestep"], timesteps=msg["timesteps"]
                    )
                    agent.track_data("Stats / Inference time (ms)", timer.elapsed_time_ms)
                # 评估且未启用随机评估时，用确定性均值动作替代采样动作；
                # 训练路径的指令不含该键，缺省视为 True（保留采样动作）
                if not msg.get("stochastic_evaluation", True):
                    _actions = _outputs.get("mean_actions", _actions)
                # 非 CUDA 张量转入共享内存后再入队：跨进程传的是共享存储
                # 句柄而非数据副本，主进程回收动作零拷贝
                if not _actions.is_cuda:
                    _actions.share_memory_()
                queue.put(_actions)
                barrier.wait()

        # 记录智能体经验：按主进程 put 的顺序（rewards -> next_observations
        # -> next_states -> terminated -> truncated -> infos）取出各数据，
        # 切到本智能体的区间，与 "act" 阶段缓存的观测/状态/动作一并记录
        elif task == "record_transition":
            agent.track_data("Stats / Env stepping time (ms)", msg["env_stepping_time_ms"])
            with torch.no_grad():
                rewards = queue.get()[scope[0] : scope[1]]
                next_observations = queue.get()[scope[0] : scope[1]]
                next_states = queue.get()
                if next_states is not None:
                    next_states = next_states[scope[0] : scope[1]]
                terminated = queue.get()[scope[0] : scope[1]]
                truncated = queue.get()[scope[0] : scope[1]]
                infos = queue.get()
                agent.record_transition(
                    observations=_observations,
                    states=_states,
                    actions=_actions,
                    rewards=rewards,
                    next_observations=next_observations,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=msg["timestep"],
                    timesteps=msg["timesteps"],
                )
                barrier.wait()

        # 执行智能体的交互后钩子（算法子类通常在此触发网络更新）
        elif task == "post_interaction":
            agent.post_interaction(timestep=msg["timestep"], timesteps=msg["timesteps"])
            barrier.wait()

        # 评估专用组合任务：记录转移后调用父类（Agent 基类）的
        # post_interaction，跳过算法子类的覆写（其中通常包含网络更新），
        # 仅执行日志写出与检查点
        elif task == "eval-record_transition-post_interaction":
            agent.track_data("Stats / Env stepping time (ms)", msg["env_stepping_time_ms"])
            with torch.no_grad():
                # 队列取数顺序与 record_transition 相同
                rewards = queue.get()[scope[0] : scope[1]]
                next_observations = queue.get()[scope[0] : scope[1]]
                next_states = queue.get()
                if next_states is not None:
                    next_states = next_states[scope[0] : scope[1]]
                terminated = queue.get()[scope[0] : scope[1]]
                truncated = queue.get()[scope[0] : scope[1]]
                infos = queue.get()
                agent.record_transition(
                    observations=_observations,
                    states=_states,
                    actions=_actions,
                    rewards=rewards,
                    next_observations=next_observations,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=msg["timestep"],
                    timesteps=msg["timesteps"],
                )
                super(agent.__class__, agent).post_interaction(timestep=msg["timestep"], timesteps=msg["timesteps"])
                barrier.wait()


@dataclasses.dataclass(kw_only=True)
class ParallelTrainerCfg(TrainerCfg):
    """并行训练器的配置类（暂无本类特有字段，各项见 TrainerCfg）。"""


class ParallelTrainer(Trainer):
    def __init__(
        self,
        *,
        env: Wrapper | MultiAgentEnvWrapper,
        agents: Agent | MultiAgent | list[Agent] | list[MultiAgent],
        scopes: list[int] | None = None,
        cfg: ParallelTrainerCfg | dict = {},
    ) -> None:
        """并行训练器：使用多进程并行训练智能体。

        每个同时智能体在独立进程中执行推理与更新，主进程集中推进环境，
        协作方式见模块说明。非同时布置（num_simultaneous_agents == 1）
        不涉及多进程，train()/eval() 直接退化为基类的单智能体循环。

        Args:
            env: 训练/评估用的环境：单智能体为 Wrapper，
                多智能体为 MultiAgentEnvWrapper。
            agents: 训练/评估的智能体：单个 Agent/MultiAgent，或由它们
                组成的列表（长度大于 1 时为同时多智能体布置）。
            scopes: 同时多智能体布置下各智能体负责的环境数量列表；
                为 None 或空列表时自动尽量平均分配。
            cfg: 配置：ParallelTrainerCfg 实例，或同结构的字典
                （字典会在传入基类前转换为配置对象）。

        Notes:
            构造时把多进程启动方法强制设为 ``"spawn"``：PyTorch 在
            CUDA/自动求导场景下要求子进程以 spawn 方式启动；
            ``force=True`` 允许在同一进程中重复创建训练器。
        """
        # 纯类型注解：供类型检查器把 self.cfg 细化为 ParallelTrainerCfg
        self.cfg: ParallelTrainerCfg
        super().__init__(
            env=env,
            agents=agents,
            scopes=scopes if scopes is not None else [],
            cfg=ParallelTrainerCfg(**cfg) if isinstance(cfg, dict) else cfg,
        )

        # 强制使用 spawn 启动子进程（CUDA 多进程安全要求）
        mp.set_start_method(method="spawn", force=True)

    def train(self) -> None:
        """并行训练智能体。

        非同时布置（num_simultaneous_agents == 1）时初始化智能体后直接
        委托基类 Trainer.train()。同时多智能体布置下，主进程循环执行：

        - 交互前钩子（广播指令，各进程并行执行后在 barrier 会合）
        - 计算动作（各进程并行推理，主进程回收并沿环境轴拼接）
        - 与环境交互（主进程执行）
        - 渲染环境（主进程执行）
        - 记录转移（数据经共享内存分发，各进程并行记录）
        - 交互后钩子（广播指令，各进程并行执行）
        - 重置环境（向量化环境无需显式重置）

        Raises:
            RuntimeError: 环境为单环境（num_envs == 1）。
                并行训练仅支持并行/向量化环境。
        """
        # 设置运行模式为训练
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.enable_training_mode(True)
        else:
            self.agents.enable_training_mode(True)

        # 非同时智能体：初始化智能体后退化为基类的单智能体循环
        if self.num_simultaneous_agents == 1:
            self.agents.init(trainer_cfg=self.cfg)
            super().train()
            return

        # 初始化多进程变量：每个智能体一组"单向管道 + 队列"，
        # 外加一个全体（N 个 worker + 主进程）共用的 barrier
        queues = []
        producer_pipes = []
        consumer_pipes = []
        barrier = mp.Barrier(self.num_simultaneous_agents + 1)
        processes = []

        for i in range(self.num_simultaneous_agents):
            # duplex=False：单向管道，主进程持写端（producer），worker 持读端
            pipe_read, pipe_write = mp.Pipe(duplex=False)
            producer_pipes.append(pipe_write)
            consumer_pipes.append(pipe_read)
            queues.append(mp.Queue())

        # 把张量移入共享内存：记忆与模型参数在各进程间共享，
        # 之后经队列传输时只传共享存储句柄而非数据副本；
        # 个别模型结构不支持共享时（RuntimeError）跳过
        for agent in self.agents:
            if agent.memory is not None:
                agent.memory.share_memory()
            for model in agent.models.values():
                try:
                    model.share_memory()
                except RuntimeError:
                    pass

        # 启动所有 worker 进程并等待其就绪
        for i in range(self.num_simultaneous_agents):
            process = mp.Process(
                target=fn_processor, args=(i, consumer_pipes, queues, barrier, self.scopes, self.cfg), daemon=True
            )
            processes.append(process)
            process.start()
        barrier.wait()

        # 初始化智能体：智能体对象经各自的队列发给 worker
        # （spawn 模式下经 pickle 传输，其中已共享内存的部分传存储句柄）
        for pipe, queue, agent in zip(producer_pipes, queues, self.agents):
            pipe.send({"task": "init"})
            queue.put(agent)
        barrier.wait()

        # 重置环境。观测与状态转入共享内存后，worker 经队列收到的都是
        # 同一块共享存储：主进程之后每步用 copy_ 原地更新内容，
        # 所有进程即刻可见，无需重新分发引用
        observations, infos = self.env.reset()
        states = self.env.state()
        if not observations.is_cuda:
            observations.share_memory_()
        if states is not None and not states.is_cuda:
            states.share_memory_()

        for timestep in tqdm.tqdm(range(self.cfg.timesteps), disable=self.cfg.disable_progressbar, file=sys.stdout):

            # 交互前钩子：广播指令，各进程并行执行后在 barrier 会合
            for pipe in producer_pipes:
                pipe.send({"task": "pre_interaction", "timestep": timestep, "timesteps": self.cfg.timesteps})
            barrier.wait()

            # 计算动作
            with torch.no_grad():
                # 下发 act 指令，并把观测/状态（共享存储引用）放入各队列
                for pipe, queue in zip(producer_pipes, queues):
                    pipe.send(
                        {
                            "task": "act",
                            "timestep": timestep,
                            "timesteps": self.cfg.timesteps,
                        }
                    )
                    queue.put(observations)
                    queue.put(states)

                # 各 worker 并行推理后，按队列顺序回收动作并沿环境轴拼接：
                # [(scope_0, D_act), ..., (scope_{N-1}, D_act)] -> (E, D_act)，
                # 行顺序与 self.agents / self.scopes 一致
                barrier.wait()
                actions = torch.vstack([queue.get() for queue in queues])

                # 主进程推进环境一步，并读取下一步集中式状态
                with ScopedTimer() as timer:
                    next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                    next_states = self.env.state()

                # 渲染环境：非 headless 且 timestep 为渲染间隔的整数倍时
                # 执行（timestep=0 也满足 0 % n == 0，即首步会渲染）
                if not self.cfg.headless and not timestep % self.cfg.render_interval:
                    self.env.render()

                # 记录转移：先把各返回张量转入共享内存（传句柄零拷贝），
                # 再按 worker 的取出顺序入队：rewards -> next_observations
                # -> next_states -> terminated -> truncated -> infos
                if not rewards.is_cuda:
                    rewards.share_memory_()
                if not next_observations.is_cuda:
                    next_observations.share_memory_()
                if next_states is not None and not next_states.is_cuda:
                    next_states.share_memory_()
                if not terminated.is_cuda:
                    terminated.share_memory_()
                if not truncated.is_cuda:
                    truncated.share_memory_()

                for pipe, queue in zip(producer_pipes, queues):
                    pipe.send(
                        {
                            "task": "record_transition",
                            "timestep": timestep,
                            "timesteps": self.cfg.timesteps,
                            "env_stepping_time_ms": timer.elapsed_time_ms,
                        }
                    )
                    queue.put(rewards)
                    queue.put(next_observations)
                    queue.put(next_states)
                    queue.put(terminated)
                    queue.put(truncated)
                    queue.put(infos)
                barrier.wait()

            # 交互后钩子：广播指令，各进程并行执行（算法子类通常在此触发网络更新）
            for pipe in producer_pipes:
                pipe.send({"task": "post_interaction", "timestep": timestep, "timesteps": self.cfg.timesteps})
            barrier.wait()

            # 重置环境
            # - 并行/向量化环境（单或多智能体）：把下一步数据原地拷入
            #   共享存储的观测/状态张量，所有进程即刻可见（episode 边界
            #   由环境在步进时自动处理）
            if self.env.num_envs > 1:
                observations.copy_(next_observations)
                if states is not None:
                    states.copy_(next_states)
            # - 单环境：并行训练不支持
            else:
                raise RuntimeError("Parallel trainer is not supported for single environment")

        # 终止进程：广播 terminate 指令，worker 收到后退出任务循环
        for pipe in producer_pipes:
            pipe.send({"task": "terminate"})

        # 回收进程：等待所有 worker 退出
        for process in processes:
            process.join()

    def eval(self) -> None:
        """并行评估智能体。

        非同时布置（num_simultaneous_agents == 1）时初始化智能体后直接
        委托基类 Trainer.eval()。同时多智能体布置下的循环与 train()
        相同，差异在于：

        - 计算动作指令随附 ``stochastic_evaluation`` 标志，各 worker
          据此决定使用采样动作还是确定性均值动作；
        - 记录转移与 post_interaction 合并为一条
          "eval-record_transition-post_interaction" 指令下发，其中的
          post_interaction 调用智能体父类（Agent 基类）版本，跳过算法
          子类的覆写（其中通常包含网络更新）。

        Raises:
            RuntimeError: 环境为单环境（num_envs == 1）。
                并行评估仅支持并行/向量化环境。
        """
        # 设置运行模式为评估
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.enable_training_mode(False)
        else:
            self.agents.enable_training_mode(False)

        # 非同时智能体：初始化智能体后退化为基类的单智能体循环
        if self.num_simultaneous_agents == 1:
            self.agents.init(trainer_cfg=self.cfg)
            super().eval()
            return

        # 初始化多进程变量（结构与 train() 相同）
        queues = []
        producer_pipes = []
        consumer_pipes = []
        barrier = mp.Barrier(self.num_simultaneous_agents + 1)
        processes = []

        for i in range(self.num_simultaneous_agents):
            # duplex=False：单向管道，主进程持写端，worker 持读端
            pipe_read, pipe_write = mp.Pipe(duplex=False)
            producer_pipes.append(pipe_write)
            consumer_pipes.append(pipe_read)
            queues.append(mp.Queue())

        # 把张量移入共享内存：记忆与模型参数在各进程间共享；
        # 模型项可为 None（如某些算法没有对应网络），跳过之
        for agent in self.agents:
            if agent.memory is not None:
                agent.memory.share_memory()
            for model in agent.models.values():
                if model is not None:
                    try:
                        model.share_memory()
                    except RuntimeError:
                        pass

        # 启动所有 worker 进程并等待其就绪
        for i in range(self.num_simultaneous_agents):
            process = mp.Process(
                target=fn_processor, args=(i, consumer_pipes, queues, barrier, self.scopes, self.cfg), daemon=True
            )
            processes.append(process)
            process.start()
        barrier.wait()

        # 初始化智能体：智能体对象经各自的队列发给 worker
        for pipe, queue, agent in zip(producer_pipes, queues, self.agents):
            pipe.send({"task": "init"})
            queue.put(agent)
        barrier.wait()

        # 重置环境（观测/状态转入共享内存，各 worker 收到同一块存储）
        observations, infos = self.env.reset()
        states = self.env.state()
        if not observations.is_cuda:
            observations.share_memory_()
        if states is not None and not states.is_cuda:
            states.share_memory_()

        for timestep in tqdm.tqdm(range(self.cfg.timesteps), disable=self.cfg.disable_progressbar, file=sys.stdout):

            # 交互前钩子：广播指令并行执行
            for pipe in producer_pipes:
                pipe.send({"task": "pre_interaction", "timestep": timestep, "timesteps": self.cfg.timesteps})
            barrier.wait()

            # 计算动作
            with torch.no_grad():
                # 下发 act 指令并附上 stochastic_evaluation 标志，
                # 把观测/状态（共享存储引用）放入各队列
                for pipe, queue in zip(producer_pipes, queues):
                    pipe.send(
                        {
                            "task": "act",
                            "timestep": timestep,
                            "timesteps": self.cfg.timesteps,
                            "stochastic_evaluation": self.cfg.stochastic_evaluation,
                        }
                    )
                    queue.put(observations)
                    queue.put(states)

                # 各 worker 并行推理后按队列顺序回收动作并沿环境轴拼接：
                # (scope_i, D_act) -> (E, D_act)
                barrier.wait()
                actions = torch.vstack([queue.get() for queue in queues])

                # 主进程推进环境一步，并读取下一步集中式状态
                with ScopedTimer() as timer:
                    next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                    next_states = self.env.state()

                # 渲染环境：条件同 train()（首步也会渲染）
                if not self.cfg.headless and not timestep % self.cfg.render_interval:
                    self.env.render()

                # 记录转移：各返回张量转入共享内存（传句柄零拷贝）
                if not rewards.is_cuda:
                    rewards.share_memory_()
                if not next_observations.is_cuda:
                    next_observations.share_memory_()
                if next_states is not None and not next_states.is_cuda:
                    next_states.share_memory_()
                if not terminated.is_cuda:
                    terminated.share_memory_()
                if not truncated.is_cuda:
                    truncated.share_memory_()

            # 评估组合任务：下发记录转移 + 父类 post_interaction 指令，
            # 数据按 worker 的取出顺序入队（与 train() 的 record_transition 相同）
            for pipe, queue in zip(producer_pipes, queues):
                pipe.send(
                    {
                        "task": "eval-record_transition-post_interaction",
                        "timestep": timestep,
                        "timesteps": self.cfg.timesteps,
                        "env_stepping_time_ms": timer.elapsed_time_ms,
                    }
                )
                queue.put(rewards)
                queue.put(next_observations)
                queue.put(next_states)
                queue.put(terminated)
                queue.put(truncated)
                queue.put(infos)
            barrier.wait()

            # 重置环境：逻辑与 train() 相同
            # - 并行/向量化环境（单或多智能体）
            if self.env.num_envs > 1:
                observations.copy_(next_observations)
                if states is not None:
                    states.copy_(next_states)
            # - 单环境：并行评估不支持
            else:
                raise RuntimeError("Parallel trainer is not supported for single environment")

        # 终止进程
        for pipe in producer_pipes:
            pipe.send({"task": "terminate"})

        # 回收进程
        for process in processes:
            process.join()
