"""skrl Torch 多智能体基类：实验配置、数据追踪、检查点与生命周期接口。"""

from __future__ import annotations

from typing import Any, Literal

import collections
import copy
import dataclasses
import datetime
import os
from abc import ABC, abstractmethod
import gymnasium
from packaging import version

import numpy as np
import torch

from skrl import config, logger
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.utils.tensorboard import SummaryWriter


@dataclasses.dataclass(kw_only=True)
class ExperimentCfg:
    """实验配置（检查点保存与日志记录）。"""

    directory: str = ""
    """各次运行（实验）生成数据的存储目录路径。"""

    experiment_name: str = ""
    """实验（训练/评估运行）名称。

    未指定时使用格式 ``YY-MM-DD_HH-MM-SS-SSSSSS_{agent_name}``。
    """

    write_interval: int | Literal["auto"] = "auto"
    """向 TensorBoard 写入数据的时间步间隔。

    * 小于等于 0 的值禁用 TensorBoard 数据写入。
    * 设为 ``"auto"`` 时，间隔取整个训练/评估过程均匀采集 100 个样本
      （``timesteps / 100``）。
    """

    checkpoint_interval: int | Literal["auto"] = "auto"
    """写检查点的时间步间隔。

    * 小于等于 0 的值禁用检查点写入。
    * 设为 ``"auto"`` 时，间隔取整个过程均匀采集 10 个样本
      （``timesteps / 10``）。
    """

    store_separately: bool = False
    """是否将各模块分开保存为独立文件。

    设为 True 时，各智能体的全部模块（模型、优化器、预处理器等）分别
    保存到各自文件；默认（False）打包进一个字典存入同一文件。
    """

    wandb: bool = False
    """是否启用 Weights & Biases 进行日志记录与可视化。"""

    wandb_kwargs: dict = dataclasses.field(default_factory=dict)
    """Weights & Biases 初始化的关键字参数。

    详见 `Weights & Biases 文档 <https://docs.wandb.ai/ref/python/functions/init/>`_。
    """


@dataclasses.dataclass(kw_only=True)
class MultiAgentCfg(ABC):
    """多智能体配置的基类。"""

    experiment: ExperimentCfg = dataclasses.field(default_factory=ExperimentCfg)
    """实验设置。"""

    def validate(self) -> bool:
        """校验配置合法性。"""
        return True

    def expand(self, *, possible_agents: list[str], immutable: list[str] = []) -> None:
        """把未按智能体区分的参数展开为逐智能体字典。

        展开规则：
        * 字段值是 dict 且已包含全部 ``possible_agents`` 键：视为已经
          逐智能体指定，跳过。
        * 字段值是 dict 但只包含部分键：键集不匹配，抛出 ValueError。
        * 其余情况（标量、元组、None 等）：深拷贝后复制给每个智能体，
          即 ``{uid: copy.deepcopy(value)}``。

        Args:
            possible_agents: 环境可能出现的全部智能体名字。
            immutable: 不参与展开的字段名列表；基类固定追加 "experiment"。
                子类（如 IPPO_CFG/MAPPO_CFG）会把全局共享参数加入此列表。
        """
        immutable += ["experiment"]
        for field in dataclasses.fields(self):
            if field.name in immutable:
                continue
            value = getattr(self, field.name)
            if isinstance(value, dict):
                if set(value) >= set(possible_agents):
                    continue
                elif set(value) < set(possible_agents):
                    raise ValueError(
                        f"Specified keys ({set(value)}) do not match possible agents ({set(possible_agents)})"
                    )
            setattr(self, field.name, {uid: copy.deepcopy(value) for uid in possible_agents})


class MultiAgent(ABC):
    def __init__(
        self,
        *,
        cfg: MultiAgentCfg,
        possible_agents: list[str],
        models: dict[str, dict[str, Model]],
        memories: dict[str, Memory] | None = None,
        observation_spaces: dict[str, gymnasium.Space] | None = None,
        state_spaces: dict[str, gymnasium.Space] | None = None,
        action_spaces: dict[str, gymnasium.Space] | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """表示一个 RL 多智能体/算法的基类。

        :param cfg: 多智能体配置。
        :param possible_agents: 环境可能出现的全部智能体名字，身份顺序基准。
        :param models: 各智能体的模型，外层键为智能体身份，内层字典的
            键为模型角色（如 policy、value）。
        :param memories: 各智能体存储数据与环境转移的回放记忆。
        :param observation_spaces: 各智能体的观测空间。
        :param state_spaces: 各智能体的状态空间。
        :param action_spaces: 各智能体的动作空间。
        :param device: 数据分配与计算设备；未指定时使用默认设备。
        """
        # 默认评估模式；训练前需调用 enable_training_mode(True)。
        self.training = False

        self.possible_agents = possible_agents
        self.num_agents = len(self.possible_agents)

        self.models = models
        self.memories = memories
        self.observation_spaces = observation_spaces
        self.state_spaces = state_spaces
        self.action_spaces = action_spaces

        self.device = config.torch.parse_device(device)

        self.cfg = cfg
        # 标量参数在此展开为逐智能体字典（见 MultiAgentCfg.expand）
        self.cfg.expand(possible_agents=self.possible_agents)
        if isinstance(self.cfg.experiment, dict):  # hack：兼容嵌套 dataclass 传入 dict 的情况
            self.cfg.experiment = ExperimentCfg(**self.cfg.experiment)

        # 将各智能体的各模型迁移到各自指定的设备
        for _models in self.models.values():
            for model in _models.values():
                if model is not None:
                    model.to(model.device)

        # 数据追踪：tag -> 标量列表缓冲，等待按 write_interval 汇总写入 TensorBoard
        self.tracking_data = collections.defaultdict(list)
        self.write_interval = self.cfg.experiment.write_interval

        # 最近 100 个完结回合的滑动窗口，用于回合级奖励/步数统计
        self._track_rewards = collections.deque(maxlen=100)
        self._track_timesteps = collections.deque(maxlen=100)
        # 每环境回合累计量；形状与各智能体奖励求和后的张量一致
        self._cumulative_rewards = None
        self._cumulative_timesteps = None

        # 检查点：外层按智能体身份登记参与存取的模块
        self.checkpoint_modules = {uid: {} for uid in self.possible_agents}
        self.checkpoint_interval = self.cfg.experiment.checkpoint_interval
        # 最优检查点按 "Reward / Total reward (mean)" 判定；reward 初始为 int32 下界
        self.checkpoint_best_modules = {"timestep": 0, "reward": -(2**31), "saved": False, "modules": {}}

        # 实验目录：未配置时用 <当前目录>/runs/<时间戳_类名>
        directory = (
            self.cfg.experiment.directory if self.cfg.experiment.directory else os.path.join(os.getcwd(), "runs")
        )
        experiment_name = (
            self.cfg.experiment.experiment_name
            if self.cfg.experiment.experiment_name
            else f"{datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S-%f')}_{self.__class__.__name__}"
        )
        self.experiment_dir = os.path.join(directory, experiment_name)

    def __str__(self) -> str:
        """生成多智能体的字符串表示（类名 + 展开的配置树）。

        :return: 字符串表示。
        """
        string = f"Multi-agent: {repr(self)}"
        for k, v in dataclasses.asdict(self.cfg).items():
            if type(v) is dict:
                string += f"\n  |-- {k}"
                for k1, v1 in v.items():
                    string += f"\n  |     |-- {k1}: {v1}"
            else:
                string += f"\n  |-- {k}: {v}"
        return string

    def _empty_preprocessor(self, _input: Any, *args, **kwargs) -> Any:
        """空预处理方法：原样返回输入。

        .. note::

            定义此方法是因为 PyTorch 的 multiprocessing 无法序列化 lambda。

        :param _input: 待处理的输入。

        :return: 处理后的输入（未做任何变换）。
        """
        return _input

    def _get_internal_value(self, _module: Any) -> Any:
        """获取模块或变量的内部状态/值。

        :param _module: 模块或变量。

        :return: 有 ``state_dict`` 的对象返回其状态字典，否则原样返回。
        """
        return _module.state_dict() if hasattr(_module, "state_dict") else _module

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        """初始化多智能体。

        .. warning::

            必须使用多智能体之前调用本方法。
            它将初始化 TensorBoard writer（以及可选的 Weights & Biases）
            并创建检查点目录。

        :param trainer_cfg: 训练器配置，用于把 ``"auto"`` 间隔换算为具体步数。
        """
        trainer_cfg = {} if trainer_cfg is None else dataclasses.asdict(trainer_cfg)

        # 分布式运行时非 0 号进程关闭日志/检查点，避免多进程重复写盘
        if config.torch.is_distributed and config.torch.rank:
            self.write_interval = 0
            self.checkpoint_interval = 0
            # TODO: 禁用 wandb
        # 配置 Weights & Biases
        if self.cfg.experiment.wandb:
            # 保存实验配置（含各智能体的模型结构）
            try:
                models_cfg = {
                    uid: {k: v.net._modules for (k, v) in self.models[uid].items()} for uid in self.possible_agents
                }
            except AttributeError:
                models_cfg = {
                    uid: {k: v._modules for (k, v) in self.models[uid].items()} for uid in self.possible_agents
                }
            wandb_config = {**dataclasses.asdict(self.cfg), **trainer_cfg, **models_cfg}
            # 填充默认值
            wandb_kwargs = copy.deepcopy(self.cfg.experiment.wandb_kwargs)
            wandb_kwargs.setdefault("name", os.path.split(self.experiment_dir)[-1])
            wandb_kwargs.setdefault("sync_tensorboard", True)
            wandb_kwargs.setdefault("config", {})
            wandb_kwargs["config"].update(wandb_config)
            # 初始化 Weights & Biases
            import wandb

            wandb.init(**wandb_kwargs)

        # TensorBoard 日志主入口
        if self.write_interval == "auto":
            self.write_interval = int(trainer_cfg.get("timesteps", 0) / 100)
        if self.write_interval > 0:
            self.writer = SummaryWriter(log_dir=self.experiment_dir)

        # 创建检查点目录
        if self.checkpoint_interval == "auto":
            self.checkpoint_interval = int(trainer_cfg.get("timesteps", 0) / 10)
        if self.checkpoint_interval > 0:
            os.makedirs(os.path.join(self.experiment_dir, "checkpoints"), exist_ok=True)

    def track_data(self, tag: str, value: float) -> None:
        """向 TensorBoard 追踪一条标量数据。

        .. note::

            当前仅支持标量数据。

        :param tag: 数据标识（如 'Loss/Policy loss'）；以 ``(min)``/``(max)``
            结尾的 tag 在写入时分别取最小/最大值，否则取均值。
        :param value: 待追踪的数值。
        """
        self.tracking_data[tag].append(value)

    def write_tracking_data(self, *, timestep: int, timesteps: int) -> None:
        """把累积的追踪数据写入 TensorBoard。

        :param timestep: 当前时间步。
        :param timesteps: 总时间步数。
        """
        for k, v in self.tracking_data.items():
            if k.endswith("(min)"):
                self.writer.add_scalar(tag=k, value=np.min(v), timestep=timestep)
            elif k.endswith("(max)"):
                self.writer.add_scalar(tag=k, value=np.max(v), timestep=timestep)
            else:
                self.writer.add_scalar(tag=k, value=np.mean(v), timestep=timestep)
        # 清空各数据容器，进入下一个写入周期
        self._track_rewards.clear()
        self._track_timesteps.clear()
        self.tracking_data.clear()

    def write_checkpoint(self, *, timestep: int, timesteps: int) -> None:
        """将检查点（各模块）写入持久存储。

        .. note::

            检查点保存在实验目录的 ``checkpoints`` 子目录中。
            检查点文件名为 ``timestep`` 参数值（非 ``None`` 时），
            否则使用当前系统日期时间。

        :param timestep: 当前时间步。
        :param timesteps: 总时间步数。
        """
        tag = str(timestep if timestep is not None else datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"))
        # 分智能体、分模块独立保存（文件名含智能体身份前缀）
        if self.cfg.experiment.store_separately:
            for uid in self.possible_agents:
                for name, module in self.checkpoint_modules[uid].items():
                    torch.save(
                        self._get_internal_value(module),
                        os.path.join(self.experiment_dir, "checkpoints", f"{uid}_{name}_{tag}.pt"),
                    )
        # 整个多智能体保存为单文件（外层键为智能体身份）
        else:
            modules = {
                uid: {name: self._get_internal_value(module) for name, module in self.checkpoint_modules[uid].items()}
                for uid in self.possible_agents
            }
            torch.save(modules, os.path.join(self.experiment_dir, "checkpoints", f"agent_{tag}.pt"))

        # 最优模块：只在有更新且尚未落盘时保存一次
        if self.checkpoint_best_modules["modules"] and not self.checkpoint_best_modules["saved"]:
            # 分智能体、分模块独立保存
            if self.cfg.experiment.store_separately:
                for uid in self.possible_agents:
                    for name in self.checkpoint_modules[uid].keys():
                        torch.save(
                            self.checkpoint_best_modules["modules"][uid][name],
                            os.path.join(self.experiment_dir, "checkpoints", f"best_{uid}_{name}.pt"),
                        )
            # 整个多智能体保存为单文件
            else:
                modules = {
                    uid: {
                        name: self.checkpoint_best_modules["modules"][uid][name]
                        for name in self.checkpoint_modules[uid].keys()
                    }
                    for uid in self.possible_agents
                }
                torch.save(modules, os.path.join(self.experiment_dir, "checkpoints", "best_agent.pt"))
            self.checkpoint_best_modules["saved"] = True

    @abstractmethod
    def act(
        self,
        observations: dict[str, torch.Tensor],
        states: dict[str, torch.Tensor | None],
        *,
        timestep: int,
        timesteps: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """使用主策略处理环境观测/状态，给出决策（动作）。

        :param observations: 各智能体的环境观测 dict[agent_id, Tensor]，
            每个张量首维为环境轴 E。
        :param states: 各智能体的状态；无状态环境为 None。
        :param timestep: 当前时间步。
        :param timesteps: 总时间步数。

        :return: 多智能体输出。第一项是各智能体期望的动作/值字典；
            第二项是按模型类型约定的附加输出字典（如对数概率）。
        """
        pass

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
        """记录一次环境转移。

        .. note::

            当 ``experiment.write_interval`` 配置解析为正值时，本方法同时
            统计回合奖励（瞬时与累计）和回合步数。子类必须调用本方法
            才能记录这些信息。

        :param observations: 转移前观测，dict[agent_id, Tensor]。
        :param states: 转移前状态。
        :param actions: 各智能体执行的动作。
        :param rewards: 各智能体本次动作的即时奖励，通常 (E, 1)。
        :param next_observations: 转移后观测。
        :param next_states: 转移后状态。
        :param terminated: 各智能体的回合终止信号（达到终态）。
        :param truncated: 各智能体的回合截断信号（达步数上限等）。
        :param infos: 环境附加信息。
        :param timestep: 当前时间步。
        :param timesteps: 总时间步数。
        """
        if self.write_interval > 0:
            # 全局统计口径：各智能体奖励逐元素求和（同环境轴可相加）
            _rewards = sum(rewards.values())

            # 维护每环境的回合累计奖励与步数（首次调用按求和后形状建立）
            if self._cumulative_rewards is None:
                self._cumulative_rewards = torch.zeros_like(_rewards, dtype=torch.float32)
                self._cumulative_timesteps = torch.zeros_like(_rewards, dtype=torch.int32)

            self._cumulative_rewards.add_(_rewards)
            self._cumulative_timesteps.add_(1)

            # 找出本步完结（终止或截断）的环境索引；以第一个智能体的信号为准
            finished_episodes = (next(iter(terminated.values())) + next(iter(truncated.values()))).nonzero(
                as_tuple=True
            )[0]
            if finished_episodes.numel():

                # 收割完结环境的回合累计值进入滑动窗口
                self._track_rewards.extend(self._cumulative_rewards[finished_episodes].tolist())
                self._track_timesteps.extend(self._cumulative_timesteps[finished_episodes].tolist())

                # 对应环境清零，开始新回合的累计
                self._cumulative_rewards[finished_episodes] = 0
                self._cumulative_timesteps[finished_episodes] = 0

            # 记录即时奖励统计（求和口径）
            self.tracking_data["Reward / Instantaneous reward (max)"].append(torch.max(_rewards).item())
            self.tracking_data["Reward / Instantaneous reward (min)"].append(torch.min(_rewards).item())
            self.tracking_data["Reward / Instantaneous reward (mean)"].append(torch.mean(_rewards).item())

            # 滑动窗口非空时，记录最近完结回合的奖励与步数统计
            if len(self._track_rewards):
                track_rewards = np.array(self._track_rewards)
                track_timesteps = np.array(self._track_timesteps)

                self.tracking_data["Reward / Total reward (max)"].append(np.max(track_rewards))
                self.tracking_data["Reward / Total reward (min)"].append(np.min(track_rewards))
                self.tracking_data["Reward / Total reward (mean)"].append(np.mean(track_rewards))

                self.tracking_data["Episode / Total timesteps (max)"].append(np.max(track_timesteps))
                self.tracking_data["Episode / Total timesteps (min)"].append(np.min(track_timesteps))
                self.tracking_data["Episode / Total timesteps (mean)"].append(np.mean(track_timesteps))

    def enable_training_mode(self, enabled: bool = True, *, apply_to_models: bool = False) -> None:
        """设置多智能体的训练模式：启用（训练）或禁用（评估）。

        当前模式可通过 ``training`` 属性查询。

        :param enabled: True 启用训练模式，False 启用评估模式。
        :param apply_to_models: 是否同时把该模式应用到全部智能体的模型。
        """
        self.training = enabled
        if apply_to_models:
            self.enable_models_training_mode(enabled)

    def enable_models_training_mode(self, enabled: bool = True) -> None:
        """设置全部智能体模型的训练模式：启用（训练）或禁用（评估）。

        :param enabled: True 启用训练模式，False 启用评估模式。
        """
        for _models in self.models.values():
            for model in _models.values():
                if model is not None:
                    model.enable_training_mode(enabled)

    def save(self, path: str) -> None:
        """将多智能体保存到指定路径（外层键为智能体身份）。

        :param path: 保存路径。
        """
        modules = {
            uid: {name: self._get_internal_value(module) for name, module in self.checkpoint_modules[uid].items()}
            for uid in self.possible_agents
        }
        torch.save(modules, path)

    def load(self, path: str) -> None:
        """从指定路径加载多智能体。

        .. note::

            最终的存储设备由多智能体构造函数指定的设备决定。

        :param path: 加载路径。
        """
        if version.parse(torch.__version__) >= version.parse("1.13"):
            modules = torch.load(path, map_location=self.device, weights_only=False)  # 避免 torch:FutureWarning
        else:
            modules = torch.load(path, map_location=self.device)
        if type(modules) is dict:
            for uid in self.possible_agents:
                # 检查点里没有该智能体的数据时跳过（不视为错误）
                if uid not in modules:
                    logger.warning(f"Skipping modules for '{uid}'. The agent doesn't have such an instance")
                    continue
                for name, data in modules[uid].items():
                    module = self.checkpoint_modules[uid].get(name, None)
                    if module is not None:
                        if hasattr(module, "load_state_dict"):
                            module.load_state_dict(data)
                            if hasattr(module, "eval"):
                                module.eval()
                        else:
                            raise NotImplementedError
                    else:
                        logger.warning(f"Skipping module '{uid}:{name}'. The agent doesn't have such an instance")

    @abstractmethod
    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        """与环境交互前调用的方法。

        :param timestep: 当前时间步。
        :param timesteps: 总时间步数。
        """
        pass

    @abstractmethod
    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        """与环境交互后调用的方法。

        :param timestep: 当前时间步。
        :param timesteps: 总时间步数。
        """
        timestep += 1

        # 按检查点间隔更新最优模型并写检查点（timestep % interval == 0 时触发）
        if timestep > 1 and self.checkpoint_interval > 0 and not timestep % self.checkpoint_interval:
            # 更新最优模型：以总奖励均值为准，超过历史最优时深拷贝当前模块
            reward = np.mean(self.tracking_data.get("Reward / Total reward (mean)", -(2**31)))
            if reward > self.checkpoint_best_modules["reward"]:
                self.checkpoint_best_modules["timestep"] = timestep
                self.checkpoint_best_modules["reward"] = reward
                self.checkpoint_best_modules["saved"] = False
                self.checkpoint_best_modules["modules"] = {
                    uid: {
                        k: copy.deepcopy(self._get_internal_value(v)) for k, v in self.checkpoint_modules[uid].items()
                    }
                    for uid in self.possible_agents
                }
            # 写检查点
            self.write_checkpoint(timestep=timestep, timesteps=timesteps)

        # 按写入间隔把追踪数据写入 TensorBoard
        if timestep > 1 and self.write_interval > 0 and not timestep % self.write_interval:
            self.write_tracking_data(timestep=timestep, timesteps=timesteps)

    @abstractmethod
    def update(self, *, timestep: int, timesteps: int, uid: str) -> None:
        """算法的主更新步（按智能体逐个调用）。

        .. warning::

            不应直接调用本方法，而应由多智能体自身在需要学习时调用。

        :param timestep: 当前时间步。
        :param timesteps: 总时间步数。
        :param uid: 智能体身份（possible_agents 中的名字）。
        """
        pass
