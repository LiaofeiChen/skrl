"""多维类别分布策略 Mixin：MultiDiscrete 动作空间的随机模型。"""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch.distributions import Categorical


# 关闭分布参数校验以加速分布构造
Categorical.set_default_validate_args(False)


class MultiCategoricalMixin:
    def __init__(
        self,
        *,
        unnormalized_log_prob: bool = True,
        reduction: Literal["mean", "sum", "prod", "none"] = "sum",
        role: str = "",
    ) -> None:
        """多维类别分布 Mixin 模型（随机模型）。

        动作空间须为 gymnasium 的 MultiDiscrete：网络输出按
        ``action_space.nvec`` 切分为若干段，每段一个独立类别分布。

        :param unnormalized_log_prob: 网络输出的解释方式。
            为 True 时解释为未归一化对数概率（logits，任意实数）；
            为 False 时解释为已归一化概率（输出必须非负、有限且
            各段和不为零）。
        :param reduction: 返回对数概率密度时的归约方式。
            为 ``"none"`` 时 log_prob 形状为 ``(num_samples, num_actions)``
            而非 ``(num_samples, 1)``；常用 ``"sum"`` 得到联合对数概率。
        :param role: 模型扮演的角色。

        :raises ValueError: 归约方式不合法。
        """
        self._mc_unnormalized_log_prob = unnormalized_log_prob
        # 每个离散动作维一个独立的类别分布
        self._mc_distributions = []

        if reduction not in ["mean", "sum", "prod", "none"]:
            raise ValueError("Reduction must be one of 'mean', 'sum', 'prod' or 'none'")
        self._mc_reduction = (
            torch.mean
            if reduction == "mean"
            else torch.sum if reduction == "sum" else torch.prod if reduction == "prod" else None
        )

    def act(self, inputs: dict[str, Any], *, role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        """对环境的观测/状态做出随机响应。

        形状链：observations (E, D_obs) -> compute -> net_output
        (E, sum(nvec))；按 nvec 切分后各段独立采样，
        堆叠动作 (E, num_discrete)，归约 log_prob (E, 1)。

        :param inputs: 模型输入，常见键为：

            - ``"observations"``: 用于决策的环境观测。
            - ``"states"``: 用于决策的环境状态。
            - ``"taken_actions"``: 策略对给定观测/状态执行的动作；
                提供时 log_prob 改为计算这些旧动作在当前分布下的值
                （PPO 更新所需），返回的第一项仍是本次新采样的动作。

        :param role: 模型扮演的角色。

        :return: 模型输出。第一项是各维采样的类别索引 (E, num_discrete)；
            第二项是包含以下附加输出的字典：

            - ``"log_prob"``: 归约后的对数概率 (E, 1)。
            - ``"net_output"``: 网络原始输出 (E, sum(nvec))。
        """
        # compute 负责观测/状态 -> (拼接的 logits/probs, outputs)
        net_output, outputs = self.compute(inputs, role)

        # 未归一化对数概率：按 nvec 把末轴切成各维的 logits
        if self._mc_unnormalized_log_prob:
            self._mc_distributions = [
                Categorical(logits=logits)
                for logits in torch.split(net_output, self.action_space.nvec.tolist(), dim=-1)
            ]
        # 已归一化概率
        else:
            self._mc_distributions = [
                Categorical(probs=probs) for probs in torch.split(net_output, self.action_space.nvec.tolist(), dim=-1)
            ]

        # 各维独立采样后沿新末轴堆叠：(E,) x num_discrete -> (E, num_discrete)
        actions = torch.stack([distribution.sample() for distribution in self._mc_distributions], dim=-1)

        # 对数概率：taken_actions（或新采样）按末轴拆回各维，
        # 逐维求 log_prob 后再堆叠、归约
        log_prob = torch.stack(
            [
                distribution.log_prob(_actions.view(-1))
                for _actions, distribution in zip(
                    torch.unbind(inputs.get("taken_actions", actions), dim=-1), self._mc_distributions
                )
            ],
            dim=-1,
        )
        if self._mc_reduction is not None:
            # (E, num_discrete) -> (E,)：各维独立，联合 log 概率即逐维求和
            log_prob = self._mc_reduction(log_prob, dim=-1)
        # (E,) -> (E, 1)，保持与动作同阶
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        outputs["log_prob"] = log_prob
        outputs["net_output"] = net_output
        return actions, outputs

    def get_entropy(self, *, role: str = "") -> torch.Tensor:
        """计算并返回分布的熵。

        :param role: 模型扮演的角色。

        :return: 各维熵堆叠 (E, num_discrete) 或按 reduction 归约后的
            (E, 1)；尚未 act 过时返回标量 0。
        """
        if self._mc_distributions:
            entropy = torch.stack([distribution.entropy() for distribution in self._mc_distributions], dim=-1)
            if self._mc_reduction is not None:
                return self._mc_reduction(entropy, dim=-1).unsqueeze(-1)
            return entropy
        return torch.tensor(0.0, device=self.device)

    def distribution(self, *, role: str = "") -> torch.distributions.Categorical:
        """获取模型的第一个分布。

        :param role: 模型扮演的角色。

        :return: 第一个离散维的类别分布。
        """
        # TODO: 找到返回全部分布的方式
        # TODO: 找到把分布函数（如 stddev）集成进类的方式
        return self._mc_distributions[0]
