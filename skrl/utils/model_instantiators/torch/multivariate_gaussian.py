"""多元高斯模型实例化器：生成基于 MultivariateGaussianMixin 的随机策略。"""

from __future__ import annotations

from typing import Any

import textwrap
import gymnasium

import torch
import torch.nn as nn  # noqa

from skrl.models.torch import MultivariateGaussianMixin  # noqa
from skrl.models.torch import Model
from skrl.utils.model_instantiators.torch.common import one_hot_encoding  # noqa
from skrl.utils.model_instantiators.torch.common import generate_containers
from skrl.utils.spaces.torch import unflatten_tensorized_space  # noqa


def multivariate_gaussian_model(
    *,
    observation_space: gymnasium.Space | None = None,
    state_space: gymnasium.Space | None = None,
    action_space: gymnasium.Space | None = None,
    device: str | torch.device | None = None,
    clip_actions: bool = False,
    clip_mean_actions: bool = False,
    clip_log_std: bool = True,
    min_log_std: float = -20,
    max_log_std: float = 2,
    initial_log_std: float = 0,
    fixed_log_std: bool = False,
    network: list[dict[str, Any]] = [],
    output: str | list[str] = "",
    return_source: bool = False,
) -> Model | str:
    """实例化基于 :class:`~skrl.models.torch.multivariate_gaussian.MultivariateGaussianMixin` 的模型。

    :param observation_space: 观测空间；``num_observations`` 属性将保存
        该空间折算后的元素总数。
    :param state_space: 状态空间；``num_states`` 属性同理。
    :param action_space: 动作空间；``num_actions`` 属性同理。
    :param device: 数据分配与计算设备；未指定时使用默认设备。
    :param clip_actions: 是否把采样后的动作裁剪到动作空间边界。
    :param clip_mean_actions: 是否把均值动作裁剪到动作空间边界；
        为 ``True`` 时在采样前先裁剪均值。
    :param clip_log_std: 是否裁剪对数标准差。
    :param min_log_std: clip_log_std 为 True 时 log_std 的下限。
    :param max_log_std: clip_log_std 为 True 时 log_std 的上限。
    :param initial_log_std: log_std 参数的初始值（形状随输出宽度）。
    :param fixed_log_std: 是否固定 log_std 参数；固定即关闭其梯度
        （生成源码中的 log_std_parameter，requires_grad 取反）。
    :param network: 网络定义（由 generate_containers 解析）。
    :param output: 输出表达式（如 "ACTIONS"、"ONE"）。
    :param return_source: 为 True 时返回模型类源码字符串而非实例。

    :return: 多元高斯模型实例；return_source 为 True 时返回源码。
    """
    # 解析模型定义（生成容器与输出表达式）
    containers, output = generate_containers(network, output, embed_output=True, indent=1)

    # 组装网络定义与 forward 语句
    networks = []
    forward: list[str] = []
    for container in containers:
        networks.append(f'self.{container["name"]}_container = {container["sequential"]}')
        forward.append(f'{container["name"]} = self.{container["name"]}_container({container["input"]})')
    # 处理输出层与输出表达式
    if output["modules"]:
        networks.append(f'self.output_layer = {output["modules"][0]}')
        forward.append(f'output = self.output_layer({container["name"]})')
    if output["output"]:
        forward.append(f'output = {output["output"]}')
    else:
        forward[-1] = forward[-1].replace(f'{container["name"]} =', "output =", 1)

    # 缩进整理生成源码
    networks = textwrap.indent("\n".join(networks), prefix=" " * 8)[8:]
    forward = textwrap.indent("\n".join(forward), prefix=" " * 8)[8:]

    template = f"""class MultivariateGaussianModel(MultivariateGaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device=None,
        clip_actions=False,
        clip_mean_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        role="",
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        MultivariateGaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_mean_actions=clip_mean_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            role=role,
        )

        {networks}
        self.log_std_parameter = nn.Parameter(
            torch.full(size=({output["size"]},), fill_value={float(initial_log_std)}, dtype=torch.float32), requires_grad={not fixed_log_std}
        )

    def compute(self, inputs, role=""):
        observations = unflatten_tensorized_space(self.observation_space, inputs.get("observations"))
        states = unflatten_tensorized_space(self.state_space, inputs.get("states"))
        taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        {forward}
        return output, {{"log_std": self.log_std_parameter}}
    """
    # 返回源码
    if return_source:
        return template

    # 实例化模型：exec 生成的类源码（生成代码引用的名字
    # 由模块级 noqa 导入提供），再构造实例
    _locals = {}
    exec(template, globals(), _locals)
    return _locals["MultivariateGaussianModel"](
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device=device,
        clip_actions=clip_actions,
        clip_mean_actions=clip_mean_actions,
        clip_log_std=clip_log_std,
        min_log_std=min_log_std,
        max_log_std=max_log_std,
    )
