"""类别分布策略 Mixin：单一离散动作空间的随机模型。"""

from __future__ import annotations

from typing import Any

import torch
from torch.distributions import Categorical


# 关闭分布参数校验以加速分布构造
Categorical.set_default_validate_args(False)


class CategoricalMixin:
    def __init__(self, *, unnormalized_log_prob: bool = True, role: str = "") -> None:
        """类别分布 Mixin 模型（随机模型）。

        :param unnormalized_log_prob: 网络输出的解释方式。
            为 True 时解释为未归一化对数概率（logits，任意实数，
            即 softmax 前的输出）；为 False 时解释为已归一化概率
            （输出必须非负、有限且和不为零）。
        :param role: 模型扮演的角色。
        """
        self._c_unnormalized_log_prob = unnormalized_log_prob
        # 最近一次 act 构建的分布；供 get_entropy/distribution 查询
        self._c_distribution = None

    def act(self, inputs: dict[str, Any], *, role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        """对环境的观测/状态做出随机响应。

        形状链：observations (E, D_obs) -> compute -> net_output (E, n)，
        n 为类别数；采样动作与 log_prob 均为 (E, 1)。

        :param inputs: 模型输入，常见键为：

            - ``"observations"``: 用于决策的环境观测。
            - ``"states"``: 用于决策的环境状态。
            - ``"taken_actions"``: 策略对给定观测/状态执行的动作；
                提供时 log_prob 改为计算这些旧动作在当前分布下的值
                （PPO 更新所需），返回的第一项仍是本次新采样的动作。

        :param role: 模型扮演的角色。

        :return: 模型输出。第一项是采样的类别索引 (E, 1)；
            第二项是包含以下附加输出的字典：

            - ``"log_prob"``: 所选类别的对数概率 (E, 1)。
            - ``"net_output"``: 网络原始输出 (E, n)。
        """
        # compute 负责观测/状态 -> (logits 或 probs, outputs)
        net_output, outputs = self.compute(inputs, role)

        # 未归一化对数概率
        if self._c_unnormalized_log_prob:
            self._c_distribution = Categorical(logits=net_output)
        # 已归一化概率
        else:
            self._c_distribution = Categorical(probs=net_output)

        # 类别采样（离散分布无 rsample，梯度经 log_prob/直方图估计传递）
        actions = self._c_distribution.sample()
        # 展平成 1 维索引再求 log_prob，与 Categorical 接口对齐
        log_prob = self._c_distribution.log_prob(inputs.get("taken_actions", actions).view(-1))

        outputs["log_prob"] = log_prob.unsqueeze(-1)
        outputs["net_output"] = net_output
        # (E,) -> (E, 1)，保持特征轴
        return actions.unsqueeze(-1), outputs

    def get_entropy(self, *, role: str = "") -> torch.Tensor:
        """计算并返回分布的熵。

        :param role: 模型扮演的角色。

        :return: 类别分布的熵 (E, 1)；尚未 act 过时返回标量 0。
        """
        if self._c_distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._c_distribution.entropy().unsqueeze(-1)

    def distribution(self, *, role: str = "") -> torch.distributions.Categorical:
        """获取模型最近一次 act 构建的分布。

        :param role: 模型扮演的角色。

        :return: 最近一次 act 的类别分布。
        """
        return self._c_distribution
