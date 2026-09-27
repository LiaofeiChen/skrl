"""对角高斯策略 Mixin：各动作维独立正态分布的随机模型。
observations
    ↓
compute()
    ↓
mean_actions = μ
log_std = log σ
    ↓
σ = exp(log_std)
    ↓
Normal(μ, σ)
    ↓
rsample()
    ↓
action
    ↓
可选 clip
    ↓
返回 action

同时：
taken_actions 或 action
    ↓
distribution.log_prob(...)
    ↓
PPO 需要的 log_prob
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch.distributions import Normal

from skrl.utils.spaces.torch import compute_space_limits


# 关闭分布参数校验以加速分布构造
Normal.set_default_validate_args(False)


class GaussianMixin:
    def __init__(
        self,
        *,
        clip_actions: bool = False,
        clip_mean_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20,
        max_log_std: float = 2,
        reduction: Literal["mean", "sum", "prod", "none"] = "sum",
        role: str = "",
    ) -> None:
        """对角高斯 Mixin 模型（随机模型）。

        各动作维使用独立的正态分布（协方差为对角阵），适用于连续
        动作空间。子类只需实现 compute，返回均值动作与 log_std。

        :param clip_actions: 是否把采样后的动作裁剪到动作空间边界。
        :param clip_mean_actions: 是否把均值动作裁剪到动作空间边界；
            为 ``True`` 时在采样前先裁剪均值。
        :param clip_log_std: 是否裁剪对数标准差。
        :param min_log_std: clip_log_std 为 True 时 log_std 的下限。
        :param max_log_std: clip_log_std 为 True 时 log_std 的上限。
        :param reduction: 返回对数概率密度时的归约方式。
            为 ``"none"`` 时 log_prob 形状为 ``(num_samples, num_actions)``
            而非 ``(num_samples, 1)``；常用 ``"sum"`` 得到联合对数概率。
        :param role: 模型扮演的角色。

        :raises ValueError: 归约方式不合法。
        """
        self._g_clip_actions = clip_actions
        self._g_clip_mean_actions = clip_mean_actions
        # 动作空间上下界 (D_act,)；无界空间为 None（裁剪选项等效失效）
        self._g_min_actions, self._g_max_actions = compute_space_limits(
            self.action_space, device=self.device, none_if_unbounded="any"
        )

        self._g_clip_log_std = clip_log_std
        self._g_min_log_std = min_log_std
        self._g_max_log_std = max_log_std

        # 最近一次 act 构建的分布；供 get_entropy/distribution 查询
        self._g_distribution = None

        if reduction not in ["mean", "sum", "prod", "none"]:
            raise ValueError("Reduction must be one of 'mean', 'sum', 'prod' or 'none'")
        self._g_reduction = (
            torch.mean
            if reduction == "mean"
            else torch.sum if reduction == "sum" else torch.prod if reduction == "prod" else None
        )

    def act(self, inputs: dict[str, Any], *, role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        """对环境的观测/状态做出随机响应。

        形状链：observations (E, D_obs) -> compute -> mean_actions、
        log_std 均为 (E, D_act)；采样动作与 log_prob 归约后为
        (E, D_act) 与 (E, 1)。

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
            - ``"log_prob"``: 对数概率密度（按 reduction 归约）。
            - ``"mean_actions"``: 均值动作（可选裁剪后的网络输出）。
        """
        # compute 负责观测/状态 -> (mean_actions, {"log_std": ...})
        mean_actions, outputs = self.compute(inputs, role)
        log_std = outputs["log_std"]

        # 裁剪均值动作
        if self._g_clip_mean_actions:
            mean_actions = torch.clamp(mean_actions, min=self._g_min_actions, max=self._g_max_actions)

        # 裁剪对数标准差，防止方差过小/过大导致数值问题
        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, min=self._g_min_log_std, max=self._g_max_log_std)
            outputs["log_std"] = log_std

        # 构建对角正态分布：各动作维独立
        self._g_distribution = Normal(mean_actions, log_std.exp())

        # 重参数化采样：a = mu + sigma * eps，梯度可流经分布参数
        actions = self._g_distribution.rsample()

        # 裁剪采样动作
        if self._g_clip_actions:
            actions = torch.clamp(actions, min=self._g_min_actions, max=self._g_max_actions)

        # 对数概率密度：默认对 taken_actions 重评估，否则用本次采样
        log_prob = self._g_distribution.log_prob(inputs.get("taken_actions", actions))
        if self._g_reduction is not None:
            # (E, D_act) -> (E,)：各维独立，联合 log 概率即逐维求和
            log_prob = self._g_reduction(log_prob, dim=-1)
        # (E,) -> (E, 1)，保持与动作同阶
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        outputs["log_prob"] = log_prob
        outputs["mean_actions"] = mean_actions
        return actions, outputs

    def get_entropy(self, *, role: str = "") -> torch.Tensor:
        """计算并返回分布的熵。

        :param role: 模型扮演的角色。

        :return: 各维独立高斯的熵沿动作轴求和，(E, 1)；
            尚未 act 过时返回标量 0。
        """
        if self._g_distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._g_distribution.entropy().sum(dim=-1).unsqueeze(-1)

    def distribution(self, *, role: str = "") -> torch.distributions.Normal:
        """获取模型最近一次 act 构建的分布。

        :param role: 模型扮演的角色。

        :return: 最近一次 act 的对角正态分布（逐元素独立）。
        """
        return self._g_distribution
