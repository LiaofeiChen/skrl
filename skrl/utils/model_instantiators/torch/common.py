"""模型实例化器公共层：输入/输出表达式解析、层定义到模块源码的生成。"""

from __future__ import annotations

from typing import Any

import ast
from gymnasium import spaces

import torch
import torch.nn as nn


def one_hot_encoding(space: spaces.Space, x: torch.Tensor) -> torch.Tensor:
    """对张量化的 Discrete 或 MultiDiscrete 空间做 one-hot 编码。

    :param space: gymnasium 空间。
    :param x: 该空间的张量化样本/值：Discrete 为 (N, 1)，
        MultiDiscrete 为 (N, len(nvec))。

    :return: one-hot 编码张量：Discrete 为 (N, n)；
        MultiDiscrete 为 (N, sum(nvec))，各维 one-hot 沿特征维拼接。
    """
    if isinstance(space, spaces.Discrete):
        return nn.functional.one_hot(x[:, 0].long(), num_classes=space.n).float()
    elif isinstance(space, spaces.MultiDiscrete):
        return torch.cat(
            [nn.functional.one_hot(x[:, i].long(), num_classes=space.nvec[i]) for i in range(space.nvec.shape[0])],
            dim=1,
        ).float()
    else:
        raise ValueError(f"Unsupported space ({space})")


def _get_activation_function(activation: str | None, as_module: bool = True) -> str | None:
    """按名字取激活函数的源码表示。

    支持的激活函数：

    - "elu"、"leaky_relu"、"relu"、"selu"、"sigmoid"、
      "softmax"、"softplus"、"softsign"、"tanh"

    :param activation: 激活函数名（大小写不敏感）。
    :param as_module: True 返回模块形式源码（如 "nn.ReLU()"），
        False 返回函数式形式（如 "functional.relu"）。

    :return: 激活函数的源码字符串；不支持时返回 None。
    """
    activations = {
        "elu": "nn.ELU()" if as_module else "functional.elu",
        "leaky_relu": "nn.LeakyReLU()" if as_module else "functional.leaky_relu",
        "relu": "nn.ReLU()" if as_module else "functional.relu",
        "selu": "nn.SELU()" if as_module else "functional.selu",
        "sigmoid": "nn.Sigmoid()" if as_module else "functional.sigmoid",
        "softmax": "nn.Softmax()" if as_module else "functional.softmax",
        "softplus": "nn.Softplus()" if as_module else "functional.softplus",
        "softsign": "nn.Softsign()" if as_module else "functional.softsign",
        "tanh": "nn.Tanh()" if as_module else "functional.tanh",
    }
    return activations.get(activation.lower() if type(activation) is str else activation, None)


def _parse_input(source: str) -> str:
    """解析网络输入表达式：替换占位符并应用操作。

    输入表达式是写在配置里的小 DSL，例如
    ``"concatenate(OBSERVATIONS, STATES)"``；本函数把它改写成
    可嵌入生成源码的 Python 表达式（如
    ``torch.cat((observations, states), dim=1)``）。

    :param source: 输入表达式字符串。

    :return: 解析后的输入表达式。
    """

    class NodeTransformer(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call):
            if isinstance(node.func, ast.Name):
                # 操作：concatenate 映射为 torch.cat（固定 dim=1）
                if node.func.id == "concatenate":
                    node.func = ast.Attribute(value=ast.Name("torch"), attr="cat")
                    node.keywords = [ast.keyword(arg="dim", value=ast.Constant(value=1))]
                # 操作：permute 映射为 torch.permute
                elif node.func.id == "permute":
                    node.func = ast.Attribute(value=ast.Name("torch"), attr="permute")
            return node

    # 通过改写源码语法树（AST）应用操作，再还原为源码文本
    tree = ast.parse(source)
    NodeTransformer().visit(tree)
    source = ast.unparse(tree)
    # 空间/数据占位符替换为大写常量对应的实例属性或变量名
    source = source.replace("OBSERVATION_SPACE", "self.observation_space")
    source = source.replace("STATE_SPACE", "self.state_space")
    source = source.replace("ACTION_SPACE", "self.action_space")
    source = source.replace("OBSERVATIONS", "observations")
    source = source.replace("STATES", "states")
    source = source.replace("ACTIONS", "taken_actions")
    return source


def _parse_output(source: str | list[str]) -> tuple[str | list[str], list[str], int]:
    """解析网络输出表达式：替换占位符、识别操作与激活，并生成输出层。

    输出表达式中的 ACTIONS/ONE 会被替换为 PLACEHOLDER，并按其
    数量自动追加一个 LazyLinear 输出层（进入返回的 modules）。

    :param source: 输出表达式字符串。

    :return: 三元组（解析后的输出表达式、生成的输出层模块源码列表、
        输出宽度/形状对应的源码表示）。
    """

    class NodeTransformer(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call):
            if isinstance(node.func, ast.Name):
                # 操作：concatenate 映射为 torch.cat（固定 dim=1）
                if node.func.id == "concatenate":
                    node.func = ast.Attribute(value=ast.Name("torch"), attr="cat")
                    node.keywords = [ast.keyword(arg="dim", value=ast.Constant(value=1))]
                else:
                    # 激活函数：识别为支持的激活名后改写为 nn.<fn> 调用
                    activation = _get_activation_function(node.func.id, as_module=False)
                    if activation:
                        node.func = ast.Attribute(value=ast.Name("nn"), attr=activation)
            return node

    size = get_num_units("ACTIONS")
    modules = []
    if type(source) is str:
        # 占位符替换：ACTIONS/ONE -> PLACEHOLDER，并确定输出层宽度
        token = "ACTIONS" if "ACTIONS" in source else None
        token = "ONE" if "ONE" in source else token
        if token:
            size = get_num_units(token)
            modules = [f"nn.LazyLinear(out_features={get_num_units(token)})"]
            source = source.replace(token, "PLACEHOLDER")
        # 通过改写源码语法树（AST）应用操作，再还原为源码文本
        tree = ast.parse(source)
        NodeTransformer().visit(tree)
        source = ast.unparse(tree)
    elif type(source) in [list, tuple]:
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid or unsupported network output definition: {source}")
    return source, modules, size


def _generate_modules(layers: list[str], activations: list[str] | str) -> list[str]:
    """把层定义列表转换为模块源码列表。

    层定义支持三种形式：数字（线性层宽度）、字符串 "flatten"
    （展平层）、或字典（带参数的层，键为 linear/conv2d/flatten）。

    :param layers: 层定义列表。
    :param activations: 各层后的激活函数；单个字符串或单元素列表会
        复制到每一层，flatten 层后不追加激活。

    :return: 生成的模块源码列表（如 "nn.LazyLinear(out_features=32)"）。
    """
    # 展开激活配置到与层数等长（空列表补空串、单元素复制）
    if type(activations) is str:
        activations = [activations] * len(layers)
    elif type(activations) is list:
        if not len(activations):
            activations = [""] * len(layers)
        elif len(activations) == 1:
            activations = activations * len(layers)
        elif len(activations) == len(layers):
            pass
        else:
            # TODO: check the length of activations
            raise NotImplementedError(f"Activations length ({len(activations)}) don't match layers ({len(layers)})")

    modules = []
    for layer, activation in zip(layers, activations):
        # 单值简写形式
        # 数字即线性层宽度
        if type(layer) in [int, float]:  # TODO: support token, e.g.: - ACTIONS??
            layer = {"linear": layer}
        # 字符串 "flatten" 即展平层
        elif type(layer) is str:
            layer = {"flatten": {}}

        # 解析层定义
        if type(layer) is dict:
            layer_type = next(iter(layer.keys())).lower()
            # 线性层：缺省用 LazyLinear；显式给 in_features 时退化为普通 Linear
            if layer_type == "linear":
                cls = "nn.LazyLinear"
                kwargs = layer[layer_type]
                if type(kwargs) in [int, float]:
                    kwargs = {"out_features": int(kwargs)}
                elif type(kwargs) is list:
                    kwargs = {k: v for k, v in zip(["out_features", "bias"][: len(kwargs)], kwargs)}
                elif type(kwargs) is dict:
                    mapping = {
                        "features": "out_features",
                        "use_bias": "bias",
                    }
                    kwargs = {mapping.get(k, k): v for k, v in kwargs.items()}
                    kwargs["out_features"] = get_num_units(kwargs["out_features"])
                    # 非 Lazy 模块：显式给出输入通道时使用非惰性版本
                    if "in_features" in kwargs:
                        cls = "nn.Linear"
                        kwargs["in_features"] = get_num_units(kwargs["in_features"])
                else:
                    raise ValueError(f"Invalid or unsupported 'linear' layer definition: {kwargs}")
            # 二维卷积：缺省用 LazyConv2d；显式给 in_channels 时退化为普通 Conv2d
            elif layer_type == "conv2d":
                cls = "nn.LazyConv2d"
                kwargs = layer[layer_type]
                if type(kwargs) is list:
                    kwargs = {
                        k: v
                        for k, v in zip(
                            ["out_channels", "kernel_size", "stride", "padding", "bias"][: len(kwargs)], kwargs
                        )
                    }
                elif type(kwargs) is dict:
                    mapping = {
                        "features": "out_channels",
                        "strides": "stride",
                        "use_bias": "bias",
                    }
                    kwargs = {mapping.get(k, k): f'"{v.lower()}"' if type(v) is str else v for k, v in kwargs.items()}
                    # 非 Lazy 模块：显式给出输入通道时使用非惰性版本
                    if "in_channels" in kwargs:
                        cls = "nn.Conv2d"
                else:
                    raise ValueError(f"Invalid or unsupported 'conv2d' layer definition: {kwargs}")
            # 展平层：其后不追加激活函数
            elif layer_type == "flatten":
                cls = "nn.Flatten"
                activation = ""  # don't add activation after flatten layer
                kwargs = layer[layer_type]
                if type(kwargs) is list:
                    kwargs = {k: v for k, v in zip(["start_dim", "end_dim"][: len(kwargs)], kwargs)}
                elif type(kwargs) is dict:
                    pass
                else:
                    raise ValueError(f"Invalid or unsupported 'flatten' layer definition: {kwargs}")
            else:
                raise ValueError(f"Invalid or unsupported layer: {layer_type}")
        else:
            raise ValueError(f"Invalid or unsupported layer definition: {layer}")
        # 拼接层与激活函数的构造源码
        kwargs = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        modules.append(f"{cls}({kwargs})")
        activation = _get_activation_function(activation)
        if activation:
            modules.append(activation)
    return modules


def get_num_units(token: str | Any) -> str | Any:
    """把占位符 token 转换为单元数/特征数的源码表示。

    :param token: 占位符（如 "ACTIONS"、"NUM_OBSERVATIONS"）。

    :return: 对应的源码表达式（如 "self.num_actions"）；
        未知 token 原样返回。
    """
    num_units = {
        "ONE": "1",
        "NUM_OBSERVATIONS": "self.num_observations",
        "NUM_STATES": "self.num_states",
        "NUM_ACTIONS": "self.num_actions",
        "OBSERVATIONS": "self.num_observations",
        "STATES": "self.num_states",
        "ACTIONS": "self.num_actions",
    }
    token_as_str = str(token)
    if token_as_str in num_units:
        return num_units[token_as_str]
    return token


def generate_containers(
    network: list[dict[str, Any]], output: str | list[str], embed_output: bool = True, indent: int = -1
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """把网络定义生成为容器（Sequential）与输出描述。

    :param network: 网络定义列表；每项含 name/input/layers（及可选
        activations），对应生成一个 Sequential 容器。
    :param output: 网络的输出表达式。
    :param embed_output: 是否把输出层模块嵌入容器定义。
        True 时输出层追加到最后一个容器的模块末尾。
    :param indent: 生成 Sequential 源码的缩进层级；负数表示单行紧凑格式。

    :return: 二元组（容器列表、输出描述字典 {"output", "modules", "size"}）。
    """
    # 解析输出表达式
    output, output_modules, output_size = _parse_output(output)
    # 逐项构建容器
    containers = []
    for i, item in enumerate(network):
        container = {}
        container["name"] = item["name"]
        container["input"] = _parse_input(item["input"])
        container["modules"] = _generate_modules(item["layers"], item.get("activations", []))
        # 把输出层嵌入（追加到）最后一个容器的模块列表
        if embed_output and i == len(network) - 1:
            container["modules"] += output_modules
            output_modules = []
        # 生成 Sequential 容器源码（单行或多行缩进两种格式）
        if indent < 0:
            container["sequential"] = f'nn.Sequential({", ".join(container["modules"])})'
        else:
            container["sequential"] = "nn.Sequential("
            for item in container["modules"]:
                container["sequential"] += f"\n{' ' * 4 * indent}{item},"
            container["sequential"] += f"\n{' ' * 4 * (indent - 1)})"
        containers.append(container)
    # 组合输出描述
    if type(output) is str:
        # 输出即占位符本身或末容器名时无需额外输出表达式
        if output == "PLACEHOLDER" or output == container["name"]:
            output = None
        # 输出表达式中的占位符替换为容器名（或 output）
        else:
            output = output.replace("PLACEHOLDER", container["name"] if embed_output else "output")
    output = {"output": output, "modules": output_modules, "size": output_size}
    return containers, output
