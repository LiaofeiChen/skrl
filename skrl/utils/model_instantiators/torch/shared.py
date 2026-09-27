"""共享主干模型实例化器：两个角色共用一份网络与一次前向。"""

from __future__ import annotations

from typing import Any

import textwrap
import gymnasium

import torch
import torch.nn as nn  # noqa

from skrl.models.torch import Model  # noqa
from skrl.models.torch import (  # noqa
    CategoricalMixin,
    DeterministicMixin,
    GaussianMixin,
    MultiCategoricalMixin,
    MultivariateGaussianMixin,
)
from skrl.utils.model_instantiators.torch.common import one_hot_encoding  # noqa
from skrl.utils.model_instantiators.torch.common import generate_containers
from skrl.utils.spaces.torch import unflatten_tensorized_space  # noqa


def shared_model(
    *,
    observation_space: gymnasium.Space | None = None,
    state_space: gymnasium.Space | None = None,
    action_space: gymnasium.Space | None = None,
    device: str | torch.device | None = None,
    structure: list[str] = ["GaussianMixin", "DeterministicMixin"],
    roles: list[str] = [],
    parameters: list[dict[str, Any]] = [],
    single_forward_pass: bool = True,
    return_source: bool = False,
) -> Model | str:
    """实例化共享模型：两个角色（如 policy 与 value）共用同一主干。

    生成 SharedModel(MixinA, MixinB, Model) 类源码并 exec 实例化；
    act 按角色分发到对应 Mixin，compute 按角色走各自的输出头，
    共享容器（主干）只定义一份。

    :param observation_space: 观测空间；``num_observations`` 属性将保存
        该空间折算后的元素总数。
    :param state_space: 状态空间；``num_states`` 属性同理。
    :param action_space: 动作空间；``num_actions`` 属性同理。
    :param device: 数据分配与计算设备；未指定时使用默认设备。
    :param structure: 共享模型结构（两个 Mixin 类名，顺序对应 roles）。
    :param roles: 角色名列表（如 ["policy", "value"]）。
    :param parameters: 与 roles 对齐的实例化参数列表（各项含
        network/output 及 Mixin 专属参数）。
    :param single_forward_pass: 共享层/网络是否只做一次前向。True 时
        第一个角色的 compute 把共享输出缓存到 ``self._shared_output``，
        紧随其后调用的第二个角色直接复用（用后清空）；若第二个角色
        先被调用则自行计算共享主干。
    :param return_source: 为 True 时返回模型类源码字符串而非实例。

    :return: 共享模型实例；return_source 为 True 时返回源码。
    """

    def get_init(class_name, parameter, role):
        """生成指定 Mixin 的 __init__ 调用源码行。"""

        if class_name.lower() == "categoricalmixin":
            return f'CategoricalMixin.__init__(self, unnormalized_log_prob={parameter.get("unnormalized_log_prob", True)}, role="{role}")'
        elif class_name.lower() == "multicategoricalmixin":
            return f"""MultiCategoricalMixin.__init__(
            self,
            unnormalized_log_prob={parameter.get("unnormalized_log_prob", True)},
            reduction="{parameter.get("reduction", "sum")}",
            role="{role}",
            )"""
        elif class_name.lower() == "deterministicmixin":
            return (
                f'DeterministicMixin.__init__(self, clip_actions={parameter.get("clip_actions", False)}, role="{role}")'
            )
        elif class_name.lower() == "gaussianmixin":
            return f"""GaussianMixin.__init__(
            self,
            clip_actions={parameter.get("clip_actions", False)},
            clip_mean_actions={parameter.get("clip_mean_actions", False)},
            clip_log_std={parameter.get("clip_log_std", True)},
            min_log_std={parameter.get("min_log_std", -20)},
            max_log_std={parameter.get("max_log_std", 2)},
            reduction="{parameter.get("reduction", "sum")}",
            role="{role}",
        )"""
        elif class_name.lower() == "multivariategaussianmixin":
            return f"""MultivariateGaussianMixin.__init__(
            self,
            clip_actions={parameter.get("clip_actions", False)},
            clip_mean_actions={parameter.get("clip_mean_actions", False)},
            clip_log_std={parameter.get("clip_log_std", True)},
            min_log_std={parameter.get("min_log_std", -20)},
            max_log_std={parameter.get("max_log_std", 2)},
            role="{role}",
        )"""
        raise ValueError(f"Unknown class: {class_name}")

    def get_return(class_name):
        """生成 compute 尾部的 return 语句源码（高斯族附带 log_std）。"""

        if class_name.lower() == "categoricalmixin":
            return r"output, {}"
        elif class_name.lower() == "multicategoricalmixin":
            return r"output, {}"
        elif class_name.lower() == "deterministicmixin":
            return r"output, {}"
        elif class_name.lower() == "gaussianmixin":
            return r'output, {"log_std": self.log_std_parameter}'
        elif class_name.lower() == "multivariategaussianmixin":
            return r'output, {"log_std": self.log_std_parameter}'
        raise ValueError(f"Unknown class: {class_name}")

    def get_extra(class_name, parameter, role, model):
        """生成 Mixin 附加参数的源码（高斯族的 log_std_parameter）。"""

        if class_name.lower() == "categoricalmixin":
            return ""
        elif class_name.lower() == "multicategoricalmixin":
            return ""
        elif class_name.lower() == "deterministicmixin":
            return ""
        elif class_name.lower() == "gaussianmixin":
            initial_log_std = float(parameter.get("initial_log_std", 0))
            fixed_log_std = parameter.get("fixed_log_std", False)
            return f'self.log_std_parameter = nn.Parameter(torch.full(size=({model["output"]["size"]},), fill_value={initial_log_std}, dtype=torch.float32), requires_grad={not fixed_log_std})'
        elif class_name.lower() == "multivariategaussianmixin":
            initial_log_std = float(parameter.get("initial_log_std", 0))
            fixed_log_std = parameter.get("fixed_log_std", False)
            return f'self.log_std_parameter = nn.Parameter(torch.full(size=({model["output"]["size"]},), fill_value={initial_log_std}, dtype=torch.float32), requires_grad={not fixed_log_std})'
        raise ValueError(f"Unknown class: {class_name}")

    # 校验：structure/roles/parameters 三者等长
    assert (
        len(structure) == len(roles) == len(parameters)
    ), f"Invalid configuration: structures ({len(structure)}), roles ({len(roles)}) and parameters ({len(parameters)}) have different lengths"

    models = [{"class": item} for item in structure]

    # 解析两个角色的模型定义（输出层不嵌入容器，独立成头）
    for i, model in enumerate(models):
        model["forward"] = []
        model["networks"] = []
        model["containers"], model["output"] = generate_containers(
            parameters[i]["network"], parameters[i]["output"], embed_output=False, indent=1
        )

    # 共享主干只取第一个角色配置生成，两角色共用
    networks_common = []
    forward_common = []
    for container in models[0]["containers"]:
        networks_common.append(f'self.{container["name"]}_container = {container["sequential"]}')
        forward_common.append(f'{container["name"]} = self.{container["name"]}_container({container["input"]})')
    forward_common.insert(
        0, 'taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))'
    )
    forward_common.insert(0, 'states = unflatten_tensorized_space(self.state_space, inputs.get("states"))')
    forward_common.insert(
        0, 'observations = unflatten_tensorized_space(self.observation_space, inputs.get("observations"))'
    )

    # 处理两个角色各自的输出层与输出表达式（以各自角色名命名）
    if models[0]["output"]["modules"]:
        models[0]["networks"].append(f'self.{roles[0]}_layer = {models[0]["output"]["modules"][0]}')
        models[0]["forward"].append(f'output = self.{roles[0]}_layer({container["name"]})')
    if models[0]["output"]["output"]:
        models[0]["forward"].append(f'output = {models[0]["output"]["output"]}')
    else:
        models[0]["forward"][-1] = models[0]["forward"][-1].replace(f'{container["name"]} =', "output =", 1)

    if models[1]["output"]["modules"]:
        models[1]["networks"].append(f'self.{roles[1]}_layer = {models[1]["output"]["modules"][0]}')
        models[1]["forward"].append(
            f'output = self.{roles[1]}_layer({"shared_output" if single_forward_pass else container["name"]})'
        )
    if models[1]["output"]["output"]:
        models[1]["forward"].append(f'output = {models[1]["output"]["output"]}')
    else:
        models[1]["forward"][-1] = models[1]["forward"][-1].replace(f'{container["name"]} =', "output =", 1)

    # 缩进整理生成源码
    networks_common = textwrap.indent("\n".join(networks_common), prefix=" " * 8)[8:]
    models[0]["networks"] = textwrap.indent("\n".join(models[0]["networks"]), prefix=" " * 8)[8:]
    extra = get_extra(structure[0], parameters[0], roles[0], models[0])
    if extra:
        models[0]["networks"] += "\n" + textwrap.indent(extra, prefix=" " * 8)
    models[1]["networks"] = textwrap.indent("\n".join(models[1]["networks"]), prefix=" " * 8)[8:]
    extra = get_extra(structure[1], parameters[1], roles[1], models[1])
    if extra:
        models[1]["networks"] += "\n" + textwrap.indent(extra, prefix=" " * 8)

    # 单次前向：role0 的 compute 末尾缓存共享输出 self._shared_output；
    # 紧随其后的 role1 复用缓存（读取后立即置 None），
    # 若 role1 先于 role0 被调用则完整计算一遍共享主干。
    if single_forward_pass:
        models[1]["forward"] = (
            [
                "if self._shared_output is None:",
            ]
            + ["    " + item for item in forward_common]
            + [
                f'    shared_output = {container["name"]}',
                "else:",
                "    shared_output = self._shared_output",
                "self._shared_output = None",
            ]
            + models[1]["forward"]
        )
        forward_common.append(f'self._shared_output = {container["name"]}')
        forward_common = textwrap.indent("\n".join(forward_common), prefix=" " * 12)[12:]
    else:
        forward_common = textwrap.indent("\n".join(forward_common), prefix=" " * 8)[8:]

    models[0]["forward"] = textwrap.indent("\n".join(models[0]["forward"]), prefix=" " * 12)[12:]
    models[1]["forward"] = textwrap.indent("\n".join(models[1]["forward"]), prefix=" " * 12)[12:]

    template = f"""class SharedModel({",".join(structure)}, Model):
    def __init__(self, observation_space, state_space, action_space, device=None):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        {get_init(structure[0], parameters[0], roles[0])}
        {get_init(structure[1], parameters[1], roles[1])}

        {networks_common}
        {models[0]["networks"]}
        {models[1]["networks"]}

    def act(self, inputs, role=""):
        if role == "{roles[0]}":
            return {structure[0]}.act(self, inputs, role=role)
        elif role == "{roles[1]}":
            return {structure[1]}.act(self, inputs, role=role)
    """
    if single_forward_pass:
        template += f"""
    def compute(self, inputs, role=""):
        if role == "{roles[0]}":
            {forward_common}
            {models[0]["forward"]}
            return {get_return(structure[0])}
        elif role == "{roles[1]}":
            {models[1]["forward"]}
            return {get_return(structure[1])}
    """
    else:
        template += f"""
    def compute(self, inputs, role=""):
        {forward_common}
        if role == "{roles[0]}":
            {models[0]["forward"]}
            return {get_return(structure[0])}
        elif role == "{roles[1]}":
            {models[1]["forward"]}
            return {get_return(structure[1])}
    """
    # 返回源码
    if return_source:
        return template

    # 实例化模型：exec 生成的类源码（生成代码引用的名字
    # 由模块级 noqa 导入提供），再构造实例
    _locals = {}
    exec(template, globals(), _locals)
    return _locals["SharedModel"](
        observation_space=observation_space, state_space=state_space, action_space=action_space, device=device
    )
