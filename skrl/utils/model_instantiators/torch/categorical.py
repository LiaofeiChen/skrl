"""类别分布模型实例化器：生成基于 CategoricalMixin 的策略。"""

from __future__ import annotations

from typing import Any

import textwrap
import gymnasium

import torch
import torch.nn as nn  # noqa

from skrl.models.torch import CategoricalMixin  # noqa
from skrl.models.torch import Model
from skrl.utils.model_instantiators.torch.common import one_hot_encoding  # noqa
from skrl.utils.model_instantiators.torch.common import generate_containers
from skrl.utils.spaces.torch import unflatten_tensorized_space  # noqa


def categorical_model(
    *,
    observation_space: gymnasium.Space | None = None,
    state_space: gymnasium.Space | None = None,
    action_space: gymnasium.Space | None = None,
    device: str | torch.device | None = None,
    unnormalized_log_prob: bool = True,
    network: list[dict[str, Any]] = [],
    output: str | list[str] = "",
    return_source: bool = False,
) -> Model | str:
    """实例化基于 :class:`~skrl.models.torch.categorical.CategoricalMixin` 的模型。

    :param observation_space: 观测空间；``num_observations`` 属性将保存
        该空间折算后的元素总数。
    :param state_space: 状态空间；``num_states`` 属性同理。
    :param action_space: 动作空间；``num_actions`` 属性同理。
    :param device: 数据分配与计算设备；未指定时使用默认设备。
    :param unnormalized_log_prob: 网络输出的解释方式。
        为 True 时解释为未归一化对数概率（logits，任意实数）；
        为 False 时解释为已归一化概率（输出必须非负、有限且和不为零）。
    :param network: 网络定义（由 generate_containers 解析）。
    :param output: 输出表达式（如 "ACTIONS"、"ONE"）。
    :param return_source: 为 True 时返回模型类源码字符串而非实例。

    :return: 类别分布模型实例；return_source 为 True 时返回源码。
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

    template = f"""class CategoricalModel(CategoricalMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device=None, unnormalized_log_prob=True, role=""):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        CategoricalMixin.__init__(self, unnormalized_log_prob=unnormalized_log_prob, role=role)

        {networks}

    def compute(self, inputs, role=""):
        observations = unflatten_tensorized_space(self.observation_space, inputs.get("observations"))
        states = unflatten_tensorized_space(self.state_space, inputs.get("states"))
        taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        {forward}
        return output, {{}}
    """
    # 返回源码
    if return_source:
        return template

    # 实例化模型：exec 生成的类源码（生成代码引用的名字
    # 由模块级 noqa 导入提供），再构造实例
    _locals = {}
    exec(template, globals(), _locals)
    return _locals["CategoricalModel"](
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device=device,
        unnormalized_log_prob=unnormalized_log_prob,
    )
