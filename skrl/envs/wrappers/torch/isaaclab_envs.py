"""Isaac Lab 环境的 skrl 包装器：单智能体与多智能体两套实现。"""

from __future__ import annotations

from typing import Any

import gymnasium

import torch

from skrl import config
from skrl.envs.wrappers.torch.base import MultiAgentEnvWrapper, Wrapper
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space, unflatten_tensorized_space


class IsaacLabWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Lab 环境包装器（单智能体）。

        :param env: 被包装的环境实例（ManagerBasedEnv / DirectRLEnv 等）。
        """
        super().__init__(env)

        # skrl 全局种子，仅在首次 reset 时播种一次
        self._seed = config.torch.key
        # Isaac Lab 的逐环境复位发生在 step 内部，包装器只在首次调用 reset
        self._reset_once = True
        self._observations = None
        self._states = None
        self._info = {}

    @property
    def state_space(self) -> gymnasium.Space | None:
        """状态空间：优先取 critic 观测（集中式状态）。

        Isaac Lab 把 critic 观测放在 ``single_observation_space`` 字典里；
        没有该键时回退到环境的 ``state_space``，仍没有则为 None。
        """
        try:
            return self._unwrapped.single_observation_space["critic"]
        except KeyError:
            pass
        try:
            return self._unwrapped.state_space
        except AttributeError:
            return None

    @property
    def observation_space(self) -> gymnasium.Space:
        """观测空间：取 policy 观测；读取失败时回退到整个观测空间。"""
        try:
            return self._unwrapped.single_observation_space["policy"]
        except:
            return self._unwrapped.observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        """动作空间：优先取 single_action_space（多环境共享单份空间）。"""
        try:
            return self._unwrapped.single_action_space
        except:
            return self._unwrapped.action_space

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """在环境中执行一步。

        :param actions: 展平后的动作 (E, D_act)，E 为环境数；
            先按动作空间结构还原（unflatten）再传给 Isaac Lab。

        :return: 观测、奖励、终止、截断、信息。观测为展平后的
            (E, D_obs)；奖励/终止/截断统一整形为 (E, 1)。
        """
        actions = unflatten_tensorized_space(self.action_space, actions)
        with torch.no_grad():
            observations, reward, terminated, truncated, self._info = self._env.step(actions)
        # 取 policy 观测并展平为 (E, D_obs)；critic 观测作为状态缓存
        self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations["policy"]))
        states = observations.get("critic", None)
        if states is not None:
            self._states = flatten_tensorized_space(tensorize_space(self.state_space, states))
        # reward/terminated/truncated 从 (E,) 统一为 (E, 1)，对齐 skrl 约定
        return self._observations, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), self._info

    def state(self) -> torch.Tensor | None:
        """获取环境状态（最近一次 step/reset 缓存的 critic 观测）。

        :return: 展平后的状态 (E, D_state)；环境无 critic 观测时为 None。
        """
        return self._states

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """复位环境：仅首次调用时真正 reset 并播种，之后返回缓存。

        Isaac Lab 环境的逐环境复位由其内部在 step 循环中处理，
        因此这里后续调用只回传最近一次观测与信息。

        :return: 观测与信息字典。
        """
        if self._reset_once:
            observations, self._info = self._env.reset(seed=self._seed)
            self._observations = flatten_tensorized_space(
                tensorize_space(self.observation_space, observations["policy"])
            )
            states = observations.get("critic", None)
            if states is not None:
                self._states = flatten_tensorized_space(tensorize_space(self.state_space, states))
            self._reset_once = False
            self._seed = None
        return self._observations, self._info

    def render(self, *args, **kwargs) -> None:
        """渲染环境：Isaac Lab 由自带的 Kit 视口渲染，这里不做处理。"""
        return None

    def close(self) -> None:
        """关闭环境。"""
        self._env.close()


class IsaacLabMultiAgentWrapper(MultiAgentEnvWrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Lab 多智能体环境包装器。

        :param env: 被包装的多智能体环境实例。
        """
        super().__init__(env)

        # skrl 全局种子，仅在首次 reset 时播种一次
        self._seed = config.torch.key
        self._reset_once = True
        self._observations = None
        self._info = {}

    def step(self, actions: dict[str, torch.Tensor]) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, Any],
    ]:
        """在环境中执行一步。

        :param actions: 各智能体的动作 dict[agent_id, Tensor]，
            每个张量为展平后的 (E, D_act_i)，E 为环境数；先按各智能体
            动作空间还原结构再传给 Isaac Lab。

        :return: 观测、奖励、终止、截断、信息，均为按智能体身份组织的
            字典；观测展平为 (E, D_obs_i)，奖励/终止/截断整形为 (E, 1)。
        """
        actions = {k: unflatten_tensorized_space(self.action_spaces[k], v) for k, v in actions.items()}
        with torch.no_grad():
            observations, rewards, terminated, truncated, self._info = self._env.step(actions)
        self._observations = {
            k: flatten_tensorized_space(tensorize_space(self.observation_spaces[k], v)) for k, v in observations.items()
        }
        # 各智能体的 (E,) 标量统一整形为 (E, 1)，对齐 skrl 约定
        return (
            self._observations,
            {k: v.view(-1, 1) for k, v in rewards.items()},
            {k: v.view(-1, 1) for k, v in terminated.items()},
            {k: v.view(-1, 1) for k, v in truncated.items()},
            self._info,
        )

    def reset(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """复位环境：仅首次调用时真正 reset 并播种，之后返回缓存。

        :return: 各智能体的观测字典与信息字典。
        """
        if self._reset_once:
            observations, self._info = self._env.reset(seed=self._seed)
            self._observations = {
                k: flatten_tensorized_space(tensorize_space(self.observation_spaces[k], v))
                for k, v in observations.items()
            }
            self._reset_once = False
            self._seed = None
        return self._observations, self._info

    def state(self) -> dict[str, torch.Tensor | None]:
        """获取集中式状态：同一份状态张量分发给全部智能体。

        外层包装（如 gymnasium 的 OrderEnforcing）可能不透传 ``state``，
        读取失败时退回最内层环境。展平时以第一个智能体的状态空间为模板，
        要求各智能体状态空间同构。

        :return: dict[agent_id, Tensor]，键为 possible_agents 中全部身份，
            各值共享同一份展平后的 (E, D_state) 张量。
        """
        try:
            state = self._env.state()
        except AttributeError:  # 'OrderEnforcing' 对象没有 state 属性
            state = self._unwrapped.state()
        if state is not None:
            state = flatten_tensorized_space(tensorize_space(next(iter(self.state_spaces.values())), state))
        return {uid: state for uid in self.possible_agents}

    def render(self, *args, **kwargs) -> None:
        """渲染环境：Isaac Lab 由自带的 Kit 视口渲染，这里不做处理。"""
        return None

    def close(self) -> None:
        """关闭环境。"""
        self._env.close()
