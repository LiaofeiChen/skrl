"""IPPO 多智能体配置。"""

from __future__ import annotations

from typing import Callable

import dataclasses

from skrl.multi_agents.torch import MultiAgentCfg


@dataclasses.dataclass(kw_only=True)
class IPPO_CFG(MultiAgentCfg):
    """IPPO 智能体的配置。

    未按智能体区分的字段会在 expand 时深拷贝为逐智能体字典
    （见 MultiAgentCfg.expand）。
    """

    rollouts: int = 16
    """两次更新之间执行的采集步数。"""

    learning_epochs: int | dict[str, int] = 8
    """每次更新执行的学习轮数。"""

    mini_batches: int | dict[str, int] = 2
    """更新时把记忆切分的 minibatch 数量。"""

    discount_factor: float | dict[str, float] = 0.99
    """折扣因子：权衡未来奖励（接近 1.0）与即时奖励（接近 0.0）。

    取值范围 ``[0.0, 1.0]``。
    """

    gae_lambda: float | dict[str, float] = 0.95
    """计算广义优势估计（GAE）的 TD(lambda) 系数。"""

    learning_rate: float | tuple[float, float] | dict[str, float | tuple[float, float]] = 1e-3
    """策略与价值网络的学习率。

    * 标量：两个网络使用同一学习率。
    * 元组：按顺序分别用于各网络。
    """

    learning_rate_scheduler: (
        type | tuple[type | None, type | None] | dict[str, type | tuple[type | None, type | None]] | None
    ) = None
    """策略与价值网络的学习率调度器类。

    详见 :ref:`learning_rate_schedulers`。

    * 类：两个网络使用同一调度器。
    * 元组：按顺序分别用于各网络。
    """

    learning_rate_scheduler_kwargs: dict | tuple[dict, dict] | dict[str, dict | tuple[dict, dict]] = dataclasses.field(
        default_factory=dict
    )
    """学习率调度器构造函数的关键字参数。

    详见 :ref:`learning_rate_schedulers`。

    .. warning::

        ``optimizer`` 参数会自动传给调度器构造函数，
        不能出现在关键字参数中。

    * 字典：两个网络使用同一组参数。
    * 元组：按顺序分别用于各网络。
    """

    observation_preprocessor: type | None | dict[str, type | None] = None
    """环境观测的预处理器类。

    详见 :ref:`preprocessors`。
    """

    observation_preprocessor_kwargs: dict | dict[str, dict] = dataclasses.field(default_factory=dict)
    """观测预处理器构造函数的关键字参数。

    详见 :ref:`preprocessors`。
    """

    state_preprocessor: type | None | dict[str, type | None] = None
    """环境状态的预处理器类。

    详见 :ref:`preprocessors`。
    """

    state_preprocessor_kwargs: dict | dict[str, dict] = dataclasses.field(default_factory=dict)
    """状态预处理器构造函数的关键字参数。

    详见 :ref:`preprocessors`。
    """

    value_preprocessor: type | None | dict[str, type | None] = None
    """价值网络输出的预处理器类。

    详见 :ref:`preprocessors`。
    """

    value_preprocessor_kwargs: dict | dict[str, dict] = dataclasses.field(default_factory=dict)
    """价值预处理器构造函数的关键字参数。

    详见 :ref:`preprocessors`。
    """

    grad_norm_clip: float | dict[str, float] = 0.5
    """按全局范数裁剪梯度的系数。

    小于等于 0 时不裁剪。
    """

    ratio_clip: float | dict[str, float] = 0.2
    """裁剪替代目标（clipped surrogate）的概率比裁剪系数。"""

    value_clip: float | dict[str, float] = 0.2
    """价值损失计算中对预测值的裁剪系数。

    小于等于 0 时不裁剪。
    """

    entropy_loss_scale: float | dict[str, float] = 0.0
    """熵损失的缩放系数。"""

    value_loss_scale: float | dict[str, float] = 2.5
    """价值损失的缩放系数。"""

    kl_threshold: float | dict[str, float] = 0.0
    """提前停止的 KL 散度阈值。"""

    time_limit_bootstrap: bool | dict[str, bool] = False
    """是否在超时截断（回合截断）处做时间限制 bootstrap。"""

    rewards_shaper: Callable | None = None
    """奖励整形函数。"""

    mixed_precision: bool = False
    """是否启用自动混合精度以提升性能。"""

    def expand(self, *, possible_agents: list[str]) -> None:
        """展开为逐智能体字典，并把网络类参数统一规范为元组形式。

        全局共享（不逐智能体展开）的字段：mixed_precision、
        rewards_shaper、rollouts。
        展开后 learning_rate 规范为 (policy_lr, value_lr) 元组，
        learning_rate_scheduler(_kwargs) 同理。
        """
        super().expand(
            possible_agents=possible_agents,
            immutable=["mixed_precision", "rewards_shaper", "rollouts"],
        )
        for uid in possible_agents:
            # 学习率：标量 -> (lr, lr)
            if not isinstance(self.learning_rate[uid], (tuple, list)):
                self.learning_rate[uid] = (self.learning_rate[uid], self.learning_rate[uid])
            # 学习率调度器：None -> (None, None)；类 -> (类, 类)
            if self.learning_rate_scheduler[uid] is None:
                self.learning_rate_scheduler[uid] = (None, None)
            elif not isinstance(self.learning_rate_scheduler[uid], (tuple, list)):
                self.learning_rate_scheduler[uid] = (
                    self.learning_rate_scheduler[uid],
                    self.learning_rate_scheduler[uid],
                )
            # 调度器关键字参数：同样统一为二元元组
            if not isinstance(self.learning_rate_scheduler_kwargs[uid], (tuple, list)):
                self.learning_rate_scheduler_kwargs[uid] = (
                    self.learning_rate_scheduler_kwargs[uid],
                    self.learning_rate_scheduler_kwargs[uid],
                )
