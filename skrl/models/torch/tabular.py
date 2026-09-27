"""表格型策略/价值 Mixin：以可学习参数直接充当 Q 表等查表结构。"""

from __future__ import annotations

from typing import Any

import torch


class TabularMixin:
    def __init__(self, *, role: str = "") -> None:
        """表格型 Mixin 模型。

        典型用法是把 ``nn.Parameter`` 直接作为 Q 表，compute 里按
        观测索引查表返回。

        :param role: 模型扮演的角色。
        """
        pass

    def __repr__(self) -> str:
        """以 torch.nn.Module 风格生成对象的字符串表示。

        :return: 字符串表示；每个参数一行，含名字与形状。
        """
        lines = []
        for name, parameter in self.named_parameters():
            lines.append(f"({name}): {parameter.__class__.__name__}(shape={list(parameter.shape)})")

        string = self.__class__.__name__ + "("
        if lines:
            string += "\n  {}\n".format("\n  ".join(lines))
        string += ")"
        return string

    def act(self, inputs: dict[str, Any], *, role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        """对环境的观测/状态做出响应（直接透传 compute）。

        :param inputs: 模型输入，常见键为：

            - ``"observations"``: 用于决策的环境观测。
            - ``"states"``: 用于决策的环境状态。
            - ``"taken_actions"``: 策略对给定观测/状态执行的动作。
        :param role: 模型扮演的角色。

        :return: 模型输出。第一项是模型返回的期望动作/值；
            第二项是按模型类型约定的附加输出字典。
        """
        actions, outputs = self.compute(inputs, role)
        return actions, outputs

    def tables(self, *, role: str = "") -> dict[str, torch.Tensor]:
        """返回模型定义的"表格"。

        :param role: 模型扮演的角色。

        :return: 参数名到参数张量的字典，即全部可学习表格；
            返回的是参数引用，修改会直接影响模型。
        """
        return {name: param for name, param in self.named_parameters()}
