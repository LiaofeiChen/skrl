"""表格型模型实例化器：生成基于 TabularMixin 的 Q 表模型。"""

from __future__ import annotations

from typing import Any, Literal

import gymnasium

import torch
import torch.nn as nn  # noqa

from skrl.models.torch import TabularMixin  # noqa
from skrl.models.torch import Model


def tabular_model(
    *,
    observation_space: gymnasium.Space | None = None,
    state_space: gymnasium.Space | None = None,
    action_space: gymnasium.Space | None = None,
    device: str | torch.device | None = None,
    variant: Literal["epsilon-greedy"] = "epsilon-greedy",
    variant_kwargs: dict[str, Any] = {},
    return_source: bool = False,
) -> Model | str:
    """实例化基于 :class:`~skrl.models.torch.tabular.TabularMixin` 的表格模型。

    生成类源码并 exec 实例化（同族其他实例化器的机制）。
    生成的 Q 表为全 1 初始化且 requires_grad=False，不参与梯度，
    由算法在训练中按查表方式外部更新。

    支持的变体：

    - ``epsilon-greedy``: 以概率 epsilon 随机探索、否则取 Q 值最大
      动作的简单探索/利用平衡方法。

      .. list-table::
          :header-rows: 1

          * - 参数
            - 类型
            - 默认值
            - 说明
          * - ``epsilon``
            - ``float``
            - ``0.1``
            - 选择探索（随机动作）的概率阈值

    :param observation_space: 观测空间；``num_observations`` 属性将保存
        该空间折算后的元素总数。
    :param state_space: 状态空间；``num_states`` 属性同理。
    :param action_space: 动作空间；``num_actions`` 属性同理。
    :param device: 数据分配与计算设备；未指定时使用默认设备。
    :param variant: 模型变体（当前仅 "epsilon-greedy"）。
    :param variant_kwargs: 变体专属关键字参数（如 epsilon）。
    :param return_source: 为 True 时返回模型类源码字符串而非实例。

    :return: 表格模型实例；return_source 为 True 时返回源码。
    """

    if variant == "epsilon-greedy":
        epsilon = float(variant_kwargs.get("epsilon", 0.1))
        template = f"""class TabularModel(TabularMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device, role=""):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        TabularMixin.__init__(self, role=role)

        self.q_table = nn.Parameter(
            torch.ones((self.num_observations, self.num_actions), dtype=torch.float32, device=self.device),
            requires_grad=False,
        )

    def compute(self, inputs, role=""):
        actions = torch.argmax(self.q_table[inputs["observations"]], dim=-1, keepdim=False)

        # choose random actions for exploration according to epsilon
        indexes = (torch.rand(inputs["observations"].shape[0], device=self.device) < {epsilon}).nonzero().flatten()
        if indexes.numel():
            actions[indexes] = torch.randint(self.num_actions, (indexes.numel(), 1), device=self.device)
        return actions, {{}}
        """
    else:
        raise ValueError(f"Invalid variant: '{variant}'")

    # 返回源码
    if return_source:
        return template

    # 实例化模型：exec 生成的类源码（生成代码引用的名字
    # 由模块级 noqa 导入提供），再构造实例
    _locals = {}
    exec(template, globals(), _locals)
    return _locals["TabularModel"](
        observation_space=observation_space, state_space=state_space, action_space=action_space, device=device
    )
