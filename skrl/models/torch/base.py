"""skrl Torch 模型基类：空间解析、参数初始化/更新与分布式工具。
                         MAPPO Agent
                         │
             ┌───────────┴───────────┐
             ↓                       ↓
          Policy                   Value
             │                       │
             └──────── Model ─────────┘
                         │
                ┌────────┼────────┐
                ↓        ↓        ↓
              act()   compute()  parameters()
                │        │
                │        └→ 你定义的神经网络
                │
                └→ Mixin 定义 RL 行为
                         │
              ┌──────────┼──────────┐
              ↓          ↓          ↓
         Gaussian    Deterministic   ...
          Mixin         Mixin
"""

from __future__ import annotations

from typing import Any

from abc import ABC, abstractmethod
import gymnasium
from packaging import version

import torch

from skrl import config, logger
from skrl.utils.spaces.torch import compute_space_size, flatten_tensorized_space, sample_space


class Model(torch.nn.Module, ABC):
    def __init__(
        self,
        *,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """实现自定义模型的基类。

        继承 Mixin（如 GaussianMixin）时，Mixin 类必须写在 Model 之前，
        例如 ``class Policy(GaussianMixin, Model)``，否则 act 的实现
        会被 Model 的抽象版本覆盖。

        :param observation_space: 观测空间；``num_observations`` 属性
            将保存该空间折算后的元素总数。
        :param state_space: 状态空间；``num_states`` 属性同理。
        :param action_space: 动作空间；``num_actions`` 属性同理。
        :param device: 数据分配与计算设备；未指定时使用默认设备。
        """
        super(Model, self).__init__()

        self.device = config.torch.parse_device(device)

        # 空间原样保留，同时折算出各空间的展平宽度（供网络输入/输出层使用）
        self.observation_space = observation_space
        self.state_space = state_space
        self.action_space = action_space
        self.num_observations = compute_space_size(observation_space)
        self.num_states = compute_space_size(state_space)
        self.num_actions = compute_space_size(action_space)

    def init_state_dict(self, inputs: dict[str, Any] = {}, *, role: str = "") -> None:
        """初始化惰性模块（lazy module）的参数。

        .. hint::

            仅当模型包含惰性模块（如模型实例化器）时调用本方法才有意义，
            且必须在对模型参数做任何操作之前调用。

        :param inputs: 模型输入，常见键为：

            - ``"observations"``: 用于决策的环境观测。
            - ``"states"``: 用于决策的环境状态。
            - ``"taken_actions"``: 策略对给定观测/状态执行的动作。

            未指定时，将从观测、状态和动作空间随机采样填充 inputs。
        :param role: 模型扮演的角色。
        """
        if not inputs:
            inputs = {
                "observations": flatten_tensorized_space(
                    sample_space(self.observation_space, backend="native", device=self.device)
                ),
                "states": flatten_tensorized_space(
                    sample_space(self.state_space, backend="native", device=self.device)
                ),
                "taken_actions": flatten_tensorized_space(
                    sample_space(self.action_space, backend="native", device=self.device)
                ),
            }
        # 初始化参数：迁移设备并跑一次前向，触发惰性模块的形状推导
        self.to(device=self.device)
        self.compute(inputs=inputs, role=role)

    def random_act(self, inputs: dict[str, Any], *, role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        """按动作空间随机动作。

        .. warning::

            从无界动作空间采样可能导致数值不稳定。

        :param inputs: 模型输入，常见键为：

            - ``"observations"``: 用于决策的环境观测。
            - ``"states"``: 用于决策的环境状态。
            - ``"taken_actions"``: 策略对给定观测/状态执行的动作。

        :param role: 模型扮演的角色。

        :return: 第一项是随机采样的动作，批量维与 ``inputs`` 中
            ``"observations"`` 的首维一致；第二项是空字典。

        :raises ValueError: 不支持的动作空间。
        """
        sample = sample_space(
            self.action_space, batch_size=inputs["observations"].shape[0], backend="native", device=self.device
        )
        return flatten_tensorized_space(sample), {}

    def init_parameters(self, method_name: str = "normal_", *args, **kwargs) -> None:
        """按指定方法名初始化模型的全部参数。

        方法名来自 `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ 模块，
        可用 ``"uniform_"``、``"normal_"``、``"constant_"`` 等。

        :param method_name: `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ 方法名。
        :param args: 所调用方法的位置参数。
        :param kwargs: 所调用方法的关键字参数。

        Example::

            # 用增益 0.5 的正交分布初始化全部参数
            >>> model.init_parameters("orthogonal_", gain=0.5)

            # 以 0.1 的稀疏度把全部参数初始化为稀疏矩阵
            >>> model.init_parameters("sparse_", sparsity=0.1)
        """
        for parameters in self.parameters():
            exec(f"torch.nn.init.{method_name}(parameters, *args, **kwargs)")

    def init_weights(self, method_name: str = "orthogonal_", *args, **kwargs) -> None:
        """按指定方法名初始化模型的权重。

        方法名来自 `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ 模块，
        可用 ``"uniform_"``、``"normal_"``、``"constant_"`` 等。

        被初始化的层：

        - torch.nn.Linear（仅 Sequential 嵌套结构内的，递归遍历）

        :param method_name: `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ 方法名。
        :param args: 所调用方法的位置参数。
        :param kwargs: 所调用方法的关键字参数。

        Example::

            # 用 [-0.1, 0.1] 均匀分布初始化全部权重
            >>> model.init_weights(method_name="uniform_", a=-0.1, b=0.1)

            # 用均值 0、标准差 0.25 的正态分布初始化全部权重
            >>> model.init_weights(method_name="normal_", mean=0.0, std=0.25)
        """

        def _update_weights(module, method_name, args, kwargs):
            for layer in module:
                if isinstance(layer, torch.nn.Sequential):
                    _update_weights(layer, method_name, args, kwargs)
                elif isinstance(layer, torch.nn.Linear):
                    exec(f"torch.nn.init.{method_name}(layer.weight, *args, **kwargs)")

        _update_weights(self.children(), method_name, args, kwargs)

    def init_biases(self, method_name: str = "constant_", *args, **kwargs) -> None:
        """按指定方法名初始化模型的偏置。

        方法名来自 `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ 模块，
        可用 ``"uniform_"``、``"normal_"``、``"constant_"`` 等。

        被初始化的层：

        - torch.nn.Linear（仅 Sequential 嵌套结构内的，递归遍历）

        :param method_name: `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ 方法名。
        :param args: 所调用方法的位置参数。
        :param kwargs: 所调用方法的关键字参数。

        Example::

            # 用常数 0 初始化全部偏置
            >>> model.init_biases(method_name="constant_", val=0)

            # 用均值 0、标准差 0.25 的正态分布初始化全部偏置
            >>> model.init_biases(method_name="normal_", mean=0.0, std=0.25)
        """

        def _update_biases(module, method_name, args, kwargs):
            for layer in module:
                if isinstance(layer, torch.nn.Sequential):
                    _update_biases(layer, method_name, args, kwargs)
                elif isinstance(layer, torch.nn.Linear):
                    exec(f"torch.nn.init.{method_name}(layer.bias, *args, **kwargs)")

        _update_biases(self.children(), method_name, args, kwargs)

    def get_specification(self) -> dict[str, Any]:
        """返回模型的规格说明。

        智能体初始化时会读取以下键：

        - ``"rnn"``: RNN/LSTM/GRU 层的循环网络规格。

          - ``"sizes"``: 各 RNN 状态的形状列表
            (层数, 环境数, RNN 状态特征数)。
            数量必须与循环层的状态数一致，
            例如 LSTM 有 2 个状态（hidden 与 cell）。

        :return: 包含模型高级规格的字典。

        Example::

            # 含一个 LSTM 层的模型
            # - 层数: 1
            # - 环境数: 4
            # - RNN 状态特征数: 64
            >>> model.get_specification()
            {'rnn': {'sizes': [(1, 4, 64), (1, 4, 64)]}}
        """
        return {}

    def forward(self, inputs: dict[str, Any], *, role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        """模型前向传播。

        .. note::

            本方法调用 :py:meth:`act` 并返回其输出；
            仅为兼容 :py:class:`torch.nn.Module` 接口而存在。

        :param inputs: 模型输入，常见键为：

            - ``"observations"``: 用于决策的环境观测。
            - ``"states"``: 用于决策的环境状态。
            - ``"taken_actions"``: 策略对给定观测/状态执行的动作。
        :param role: 模型扮演的角色。

        :return: 模型输出。第一项是模型返回的期望动作/值；
            第二项是按模型类型约定的附加输出字典。
        """
        return self.act(inputs, role=role)

    @abstractmethod
    def compute(self, inputs: dict[str, Any], *, role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        """定义模型执行的计算（由子类实现网络前向）。

        .. warning::

            本方法是抽象方法，必须由子类实现。

        :param inputs: 模型输入，常见键为：

            - ``"observations"``: 用于决策的环境观测。
            - ``"states"``: 用于决策的环境状态。
            - ``"taken_actions"``: 策略对给定观测/状态执行的动作。
        :param role: 模型扮演的角色。

        :return: 模型执行的计算。

        :raises NotImplementedError: 必须由子类实现本方法。
        """
        raise NotImplementedError("The computation performed by the model (.compute()) is not implemented")

    @abstractmethod
    def act(self, inputs: dict[str, Any], *, role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        """按指定行为执行动作。

        智能体调用本方法，基于观测/状态获得期望的动作/值。

        .. warning::

            本方法由各 Mixin（如 :py:class:`~skrl.models.torch.gaussian.GaussianMixin`）
            实现；继承 Mixin 的类只需实现 :py:meth:`compute` 方法。

        :param inputs: 模型输入，常见键为：

            - ``"observations"``: 用于决策的环境观测。
            - ``"states"``: 用于决策的环境状态。
            - ``"taken_actions"``: 策略对给定观测/状态执行的动作。
        :param role: 模型扮演的角色。

        :return: 模型输出。第一项是模型返回的期望动作/值；
            第二项是按模型类型约定的附加输出字典。

        :raises NotImplementedError: 必须由子类（或 Mixin）实现本方法。
        """
        logger.warning("Make sure to place Mixins classes before the Model class during model definition")
        raise NotImplementedError("The action to be taken by the agent (.act()) is not implemented")

    def enable_training_mode(self, enabled: bool = True) -> None:
        """设置模型的训练模式：启用（训练）或禁用（评估）。

        :param enabled: True 启用训练模式，False 启用评估模式。
            详见 :py:meth:`torch.nn.Module.train`。
        """
        self.train(enabled)

    def save(self, path: str, *, state_dict: dict[str, Any] | None = None) -> None:
        """将模型保存到指定路径。

        :param path: 保存路径。
        :param state_dict: 要保存的状态字典；为 ``None`` 时保存模型当前的
            ``state_dict``。

        Example::

            # 把当前模型保存到指定路径
            >>> model.save("/tmp/model.pt")

            # 把模型的旧版本保存到指定路径
            >>> old_state_dict = copy.deepcopy(model.state_dict())
            >>> # ...
            >>> model.save("/tmp/model.pt", old_state_dict)
        """
        torch.save(self.state_dict() if state_dict is None else state_dict, path)

    def load(self, path: str) -> None:
        """从指定路径加载模型。

        .. note::

            最终的存储设备由模型构造函数指定的设备决定。

        :param path: 加载路径。

        Example::

            # 加载模型到 CPU
            >>> model = Model(device="cpu")
            >>> model.load("model.pt")

            # 加载模型到 GPU 1
            >>> model = Model(device="cuda:1")
            >>> model.load("model.pt")
        """
        if version.parse(torch.__version__) >= version.parse("1.13"):
            state_dict = torch.load(path, map_location=self.device, weights_only=False)  # 避免 torch:FutureWarning
        else:
            state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)
        # 加载后切换到评估模式
        self.eval()

    def freeze_parameters(self, freeze: bool = True) -> None:
        """冻结或解冻内部参数。

        - 冻结：禁用梯度计算（``parameters.requires_grad = False``）。
        - 解冻：启用梯度计算（``parameters.requires_grad = True``）。

        :param freeze: 是否冻结内部参数。

        Example::

            # 冻结模型参数
            >>> model.freeze_parameters(True)

            # 解冻模型参数
            >>> model.freeze_parameters(False)
        """
        for parameters in self.parameters():
            parameters.requires_grad = not freeze

    def update_parameters(self, model: torch.nn.Module, *, polyak: float = 1.0) -> None:
        """以硬更新或软更新（polyak 平均）刷新内部参数。

        - 硬更新：:math:`\\theta = \\theta_{net}`
        - 软更新（polyak 平均）：:math:`\\theta = (1 - \\rho) \\theta + \\rho \\theta_{net}`

        :param model: 用于刷新内部参数的源模型。
        :param polyak: 0 到 1 之间的 polyak 超参数；取 1 时执行硬更新。

        Example::

            # 硬更新（来自源模型）
            >>> model.update_parameters(source_model)

            # 软更新（来自源模型）
            >>> model.update_parameters(source_model, polyak=0.005)
        """
        with torch.no_grad():
            # 硬更新：直接复制源模型参数
            if polyak == 1:
                for parameters, model_parameters in zip(self.parameters(), model.parameters()):
                    parameters.data.copy_(model_parameters.data)
            # 软更新：用原地操作避免创建新参数
            else:
                for parameters, model_parameters in zip(self.parameters(), model.parameters()):
                    parameters.data.mul_(1 - polyak)
                    parameters.data.add_(polyak * model_parameters.data)

    def broadcast_parameters(self, *, rank: int = 0) -> None:
        """分布式运行时，把模型参数广播到整个进程组（如跨所有节点）。

        调用后各进程的模型将持有来自 ``rank`` 进程的参数。

        :param rank: 广播源的进程 rank。

        Example::

            # 从 rank 1 的进程广播模型参数
            >>> if config.torch.is_distributed:
            ...     model.broadcast_parameters(rank=1)
        """
        object_list = [self.state_dict()]
        torch.distributed.broadcast_object_list(object_list, rank)
        self.load_state_dict(object_list[0])

    def reduce_parameters(self) -> None:
        """在全部进程间规约（求和后平均）模型梯度。

        调用后各进程的模型梯度将逐位一致（等效跨进程平均梯度）。

        Example::

            # 跨所有进程规约模型梯度
            >>> if config.torch.is_distributed:
            ...     model.reduce_parameters()
        """
        # 梯度拼接成单向量后一次 all_reduce，减少通信次数
        # 参考: https://github.com/entity-neural-network/incubator/pull/220
        gradients = []
        for parameters in self.parameters():
            if parameters.grad is not None:
                gradients.append(parameters.grad.view(-1))
        gradients = torch.cat(gradients)

        torch.distributed.all_reduce(gradients, op=torch.distributed.ReduceOp.SUM)

        # 规约结果除以进程数得到平均梯度，再按原形状写回各参数
        offset = 0
        for parameters in self.parameters():
            if parameters.grad is not None:
                parameters.grad.data.copy_(
                    gradients[offset : offset + parameters.numel()].view_as(parameters.grad.data)
                    / config.torch.world_size
                )
                offset += parameters.numel()
