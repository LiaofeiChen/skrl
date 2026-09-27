"""多元高斯策略 Mixin：使用 torch MultivariateNormal 的随机模型。"""

from __future__ import annotations

from typing import Any

import torch
from torch.distributions import MultivariateNormal

from skrl.utils.spaces.torch import compute_space_limits


# 关闭分布参数校验以加速分布构造
MultivariateNormal.set_default_validate_args(False)


class MultivariateGaussianMixin:
    def __init__(
        self,
        *,
        clip_actions: bool = False,
        clip_mean_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20,
        max_log_std: float = 2,
        role: str = "",
    ) -> None:
        """多元高斯 Mixin 模型（随机模型）。

        与 GaussianMixin 的区别：使用 torch 的 MultivariateNormal
        构建分布，log_prob 直接给出联合密度（无需 reduction）。

        :param clip_actions: 是否把采样后的动作裁剪到动作空间边界。
        :param clip_mean_actions: 是否把均值动作裁剪到动作空间边界；
            为 ``True`` 时在采样前先裁剪均值。
        :param clip_log_std: 是否裁剪对数标准差。
        :param min_log_std: clip_log_std 为 True 时 log_std 的下限。
        :param max_log_std: clip_log_std 为 True 时 log_std 的上限。
        :param role: 模型扮演的角色。
        """
        self._mg_clip_actions = clip_actions
        self._mg_clip_mean_actions = clip_mean_actions
        # 动作空间上下界 (D_act,)；无界空间为 None（裁剪选项等效失效）
        self._mg_min_actions, self._mg_max_actions = compute_space_limits(
            self.action_space, device=self.device, none_if_unbounded="any"
        )

        self._mg_clip_log_std = clip_log_std
        self._mg_min_log_std = min_log_std
        self._mg_max_log_std = max_log_std

        # 最近一次 act 构建的分布；供 get_entropy/distribution 查询
        self._mg_distribution = None

    def act(self, inputs: dict[str, Any], *, role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        """对环境的观测/状态做出随机响应。

        形状链：observations (E, D_obs) -> compute -> mean_actions、
        log_std 均为 (E, D_act)；采样动作 (E, D_act)，
        联合 log_prob (E, 1)。

        :param inputs: 模型输入，常见键为：

            - ``"observations"``: 用于决策的环境观测。
            - ``"states"``: 用于决策的环境状态。
            - ``"taken_actions"``: 策略对给定观测/状态执行的动作；
                提供时 log_prob 改为计算这些旧动作在当前分布下的值
                （PPO 更新所需），返回的第一项仍是本次新采样的动作。

        :param role: 模型扮演的角色。

        :return: 模型输出。第一项是采样的动作；
            第二项是包含以下附加输出的字典：

            - ``"log_std"``: 对数标准差（裁剪后的）。
            - ``"log_prob"``: 联合对数概率密度。
            - ``"mean_actions"``: 均值动作（可选裁剪后的网络输出）。
        """
        # compute 负责观测/状态 -> (mean_actions, {"log_std": ...})
        mean_actions, outputs = self.compute(inputs, role)
        log_std = outputs["log_std"]

        # 裁剪均值动作
        if self._mg_clip_mean_actions:
            mean_actions = torch.clamp(mean_actions, min=self._mg_min_actions, max=self._mg_max_actions)

        # 裁剪对数标准差
        if self._mg_clip_log_std:
            log_std = torch.clamp(log_std, min=self._mg_min_log_std, max=self._mg_max_log_std)
            outputs["log_std"] = log_std

        # 构建分布：covariance = diag(sigma^2)，但以 scale_tril 形式传入。
        # 注意 torch 语义为 Sigma = L @ L.T，因此实际协方差是 diag(sigma^4)；
        # 若意图是各维独立的 Sigma = diag(sigma^2)，应传 diag(sigma)。
        # 此为上游实现的现状，使用前请核对该语义对任务的影响。
        covariance = torch.diag(log_std.exp() * log_std.exp())
        self._mg_distribution = MultivariateNormal(mean_actions, scale_tril=covariance)

        # 重参数化采样：梯度可流经分布参数
        actions = self._mg_distribution.rsample()

        # 裁剪采样动作
        if self._mg_clip_actions:
            actions = torch.clamp(actions, min=self._mg_min_actions, max=self._mg_max_actions)

        # 联合对数概率密度：MultivariateNormal 自带按维归约，输出 (E,)
        log_prob = self._mg_distribution.log_prob(inputs.get("taken_actions", actions))
        # (E,) -> (E, 1)，保持与动作同阶
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        outputs["log_prob"] = log_prob
        outputs["mean_actions"] = mean_actions
        return actions, outputs

    def get_entropy(self, *, role: str = "") -> torch.Tensor:
        """计算并返回分布的熵。

        :param role: 模型扮演的角色。

        :return: 联合分布的熵 (E, 1)；尚未 act 过时返回标量 0。
        """
        if self._mg_distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._mg_distribution.entropy().unsqueeze(-1)

    def distribution(self, *, role: str = "") -> torch.distributions.MultivariateNormal:
        """获取模型最近一次 act 构建的分布。

        :param role: 模型扮演的角色。

        :return: 最近一次 act 的多元正态分布。
        """
        return self._mg_distribution
