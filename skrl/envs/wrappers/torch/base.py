"""skrl 环境包装器基类：单智能体与多智能体两套抽象接口。"""

from __future__ import annotations

from typing import Any

from abc import ABC, abstractmethod
import gymnasium

import torch

from skrl import config


class Wrapper(ABC):
    def __init__(self, env: Any) -> None:
        """RL 环境包装器的基类。

        :param env: 被包装的环境实例。
        """
        self._env = env
        # 逐层剥开 .unwrapped 链，拿到最内层环境；不支持该属性时退回 env 本身
        try:
            self._unwrapped = self._env.unwrapped
        except:
            self._unwrapped = env

        # 设备：内层环境有 device 属性则沿用，否则按可用性选 cuda/cpu
        if hasattr(self._unwrapped, "device"):
            self._device = config.torch.parse_device(self._unwrapped.device)
        else:
            self._device = config.torch.parse_device(None)

    def __getattr__(self, key: str) -> Any:
        """从被包装环境获取属性。

        .. note::

            ``__getattr__`` 仅在常规属性查找失败后才会被调用，
            因此这里实现的是对内层环境的透传回退链。

        :param key: 属性名。

        :return: 属性值；优先取外层 ``_env``，其次取 ``_unwrapped``。

        :raises AttributeError: 两层环境都没有该属性。
        """
        if hasattr(self._env, key):
            return getattr(self._env, key)
        if hasattr(self._unwrapped, key):
            return getattr(self._unwrapped, key)
        raise AttributeError(
            f"Wrapped environment ({self._unwrapped.__class__.__name__}) does not have attribute '{key}'"
        )

    @abstractmethod
    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """复位环境。

        :return: 观测与信息字典。观测首维为环境轴 E。
        """
        pass

    @abstractmethod
    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """在环境中执行一步。

        :param actions: 待执行的动作，首维为环境轴 E，特征维由动作空间决定。

        :return: 观测、奖励、终止信号、截断信号、信息，依次排列。
        """
        pass

    @abstractmethod
    def state(self) -> torch.Tensor | None:
        """获取环境状态。

        :return: 集中式状态；环境无状态概念时为 None。
        """
        pass

    @abstractmethod
    def render(self, *args, **kwargs) -> Any:
        """渲染环境。

        :return: 被包装环境返回的任意值。
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭环境。"""
        pass

    @property
    def device(self) -> torch.device:
        """环境使用的设备。

        被包装环境没有 ``device`` 属性时，按设备可用性取 ``"cuda"`` 或 ``"cpu"``。
        """
        return self._device

    @property
    def num_envs(self) -> int:
        """环境数量。

        被包装环境没有 ``num_envs`` 属性时按 1 处理（单环境）。
        """
        return self._unwrapped.num_envs if hasattr(self._unwrapped, "num_envs") else 1

    @property
    def num_agents(self) -> int:
        """智能体数量。

        被包装环境没有 ``num_agents`` 属性时按 1 处理（单智能体）。
        """
        return self._unwrapped.num_agents if hasattr(self._unwrapped, "num_agents") else 1

    @property
    def state_space(self) -> gymnasium.Space | None:
        """状态空间。

        被包装环境没有 ``state_space`` 属性时返回 None。
        """
        return self._unwrapped.state_space if hasattr(self._unwrapped, "state_space") else None

    @property
    def observation_space(self) -> gymnasium.Space:
        """观测空间。"""
        return self._unwrapped.observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        """动作空间。"""
        return self._unwrapped.action_space


class MultiAgentEnvWrapper(ABC):
    def __init__(self, env: Any) -> None:
        """多智能体环境包装器的基类。

        :param env: 被包装的多智能体环境实例。
        """
        self._env = env
        # 逐层剥开 .unwrapped 链，拿到最内层环境；不支持该属性时退回 env 本身
        try:
            self._unwrapped = self._env.unwrapped
        except:
            self._unwrapped = env

        # 设备：内层环境有 device 属性则沿用，否则按可用性选 cuda/cpu
        if hasattr(self._unwrapped, "device"):
            self._device = config.torch.parse_device(self._unwrapped.device)
        else:
            self._device = config.torch.parse_device(None)

    def __getattr__(self, key: str) -> Any:
        """从被包装环境获取属性。

        .. note::

            ``__getattr__`` 仅在常规属性查找失败后才会被调用，
            因此这里实现的是对内层环境的透传回退链。

        :param key: 属性名。

        :return: 属性值；优先取外层 ``_env``，其次取 ``_unwrapped``。

        :raises AttributeError: 两层环境都没有该属性。
        """
        if hasattr(self._env, key):
            return getattr(self._env, key)
        if hasattr(self._unwrapped, key):
            return getattr(self._unwrapped, key)
        raise AttributeError(
            f"Wrapped environment ({self._unwrapped.__class__.__name__}) does not have attribute '{key}'"
        )

    @abstractmethod
    def reset(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """复位环境。

        :return: 观测与信息字典。观测为 dict[agent_id, Tensor]，
            各张量首维为环境轴 E，特征维由该智能体的观测空间决定。
        """
        pass

    @abstractmethod
    def step(self, actions: dict[str, torch.Tensor]) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, Any],
    ]:
        """在环境中执行一步。

        :param actions: 待执行的动作，dict[agent_id, Tensor]，
            各张量首维为环境轴 E；不同智能体的动作维数可以不同。

        :return: 观测、奖励、终止信号、截断信号、信息，均为按智能体
            身份组织的字典；terminated/truncated 为 bool 张量。
        """
        pass

    @abstractmethod
    def state(self) -> dict[str, torch.Tensor | None]:
        """获取环境状态。

        :return: dict[agent_id, Tensor | None]，按智能体身份组织的状态。
        """
        pass

    @abstractmethod
    def render(self, *args, **kwargs) -> Any:
        """渲染环境。

        :return: 被包装环境返回的任意值。
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭环境。"""
        pass

    @property
    def device(self) -> torch.device:
        """环境使用的设备。

        被包装环境没有 ``device`` 属性时，按设备可用性取 ``"cuda"`` 或 ``"cpu"``。
        """
        return self._device

    @property
    def num_envs(self) -> int:
        """环境数量。

        被包装环境没有 ``num_envs`` 属性时按 1 处理（单环境）。
        """
        return self._unwrapped.num_envs if hasattr(self._unwrapped, "num_envs") else 1

    @property
    def num_agents(self) -> int:
        """当前存活智能体数量。

        被包装环境未定义该属性时，从 ``agents`` 属性的长度读取。
        """
        try:
            return self._unwrapped.num_agents
        except:
            return len(self.agents)

    @property
    def max_num_agents(self) -> int:
        """环境可能出现的智能体总数上限。

        被包装环境未定义该属性时，从 ``possible_agents`` 属性的长度读取。
        """
        try:
            return self._unwrapped.max_num_agents
        except:
            return len(self.possible_agents)

    @property
    def agents(self) -> list[str]:
        """当前全部智能体的名字列表。

        随环境推进可能变化（智能体可加入或退出，如阵亡/完成）。
        """
        return self._unwrapped.agents

    @property
    def possible_agents(self) -> list[str]:
        """环境可能出现的全部智能体名字列表。

        环境推进过程中不变；各张量/字典的智能体身份顺序以此为准。
        """
        return self._unwrapped.possible_agents

    @property
    def state_spaces(self) -> dict[str, gymnasium.Space | None]:
        """状态空间（按智能体组织）。

        返回值虽为字典，每个智能体的空间遵循以下规则：

        * 被包装环境有 ``state_space`` 属性（同构状态）：状态是环境的
          全局视角，所有智能体共享同一空间。
        * 被包装环境有 ``state_spaces`` 属性（异构状态）：各智能体的
          状态可以不同，空间也可以不同。
        * 两者都没有：所有智能体的空间均为 None。
        """
        if hasattr(self._unwrapped, "state_space"):
            space = self._unwrapped.state_space
            return {agent: space for agent in self.possible_agents}
        elif hasattr(self._unwrapped, "state_spaces"):
            return self._unwrapped.state_spaces
        else:
            return {agent: None for agent in self.possible_agents}

    @property
    def observation_spaces(self) -> dict[str, gymnasium.Space]:
        """各智能体的观测空间（字典）。"""
        return self._unwrapped.observation_spaces

    @property
    def action_spaces(self) -> dict[str, gymnasium.Space]:
        """各智能体的动作空间（字典）。"""
        return self._unwrapped.action_spaces

    def state_space(self, agent: str) -> gymnasium.Space | None:
        """指定智能体的状态空间。

        规则见 :py:attr:`state_spaces`。

        :param agent: 智能体名字。

        :return: 该智能体的状态空间。
        """
        return self.state_spaces[agent]

    def observation_space(self, agent: str) -> gymnasium.Space:
        """指定智能体的观测空间。

        :param agent: 智能体名字。

        :return: 该智能体的观测空间。
        """
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gymnasium.Space:
        """指定智能体的动作空间。

        :param agent: 智能体名字。

        :return: 该智能体的动作空间。
        """
        return self.action_spaces[agent]
