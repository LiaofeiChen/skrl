"""确定性策略/价值 Mixin：直接返回网络输出，不做随机采样。"""

from __future__ import annotations

from typing import Any

import torch

from skrl.utils.spaces.torch import compute_space_limits


class DeterministicMixin:
    def __init__(self, *, clip_actions: bool = False, role: str = "") -> None:
        """确定性 Mixin 模型（确定性模型）。

        常用于确定性策略（如 DDPG/TD3 的 actor）或价值网络
        （value/critic 的输出本来就是确定值）。

        :param clip_actions: 是否把输出动作裁剪到动作空间边界。
        :param role: 模型扮演的角色。
        """
        self._d_clip_actions = clip_actions
        # 动作空间上下界 (D_act,)；无界空间为 None（裁剪选项等效失效）
        self._d_min_actions, self._d_max_actions = compute_space_limits(
            self.action_space, device=self.device, none_if_unbounded="any"
        )

    def act(self, inputs: dict[str, Any], *, role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        """对环境的观测/状态做出确定性响应。

        形状链：observations (E, D_obs) -> compute -> 动作/值
        (E, D_act)；无随机成分，附加输出仅透传 compute 的字典。

        :param inputs: 模型输入，常见键为：

            - ``"observations"``: 用于决策的环境观测。
            - ``"states"``: 用于决策的环境状态。
            - ``"taken_actions"``: 策略对给定观测/状态执行的动作。
        :param role: 模型扮演的角色。

        :return: 模型输出。第一项是模型返回的期望动作/值；
            第二项是按模型类型约定的附加输出字典。
        """
        # compute 负责观测/状态 -> (动作/值, outputs)
        actions, outputs = self.compute(inputs, role)

        # 裁剪输出动作
        if self._d_clip_actions:
            actions = torch.clamp(actions, min=self._d_min_actions, max=self._d_max_actions)

        return actions, outputs
