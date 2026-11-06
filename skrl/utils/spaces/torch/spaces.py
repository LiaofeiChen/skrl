"""gymnasium 空间与 PyTorch 张量之间的转换工具模块。

围绕 gymnasium 空间（Box/Discrete/MultiDiscrete 及组合空间 Tuple/Dict）
提供一组无状态工具函数：旧版 gym 空间的转换、样本的张量化与还原、
张量化样本的展平与还原、空间尺寸与上下限的计算，以及随机采样。

张量化（tensorize_space）后各基本空间的样本形状约定：

- Box：``(B, *data_shape)``，data_shape 为空间本身的 shape；
- Discrete：``(B, 1)``，单个整数索引（不是 one-hot）；
- MultiDiscrete：``(B, N)``，N 为 nvec 的分量个数。

B 为批量维长度；组合空间（Tuple/Dict）递归套用上述约定。
"""

from __future__ import annotations

from typing import Any, Literal

import gymnasium
from gymnasium import spaces

import numpy as np
import torch

from skrl import config


def convert_gym_space(space: "gym.Space" | None, *, squeeze_batch_dimension: bool = False) -> gymnasium.Space | None:
    """把旧版 gym 空间转换为 gymnasium 空间。

    支持基本空间 Box/Discrete/MultiDiscrete 与组合空间 Tuple/Dict，
    组合空间递归转换其子空间。

    Args:
        space: 待转换的 gym 空间；为 None 时返回 None。
        squeeze_batch_dimension: 是否去掉基本空间样本的第 0 维（批量维）。
            目前仅对 Box 空间生效：取 ``low[0]``/``high[0]`` 作为边界，
            并把 shape 去掉第 0 维。用于 gym 空间自带批量维的场景。

    Returns:
        转换得到的 gymnasium 空间；入参为 None 时返回 None。

    Raises:
        ValueError: 空间类型不受支持。

    Notes:
        gym 在函数体内惰性导入（import gym），仅在使用本函数时才要求
        安装 gym 库。
    """
    import gym

    if space is None:
        return None
    # 基本空间
    # - Box：用 low/high/shape/dtype 重建；squeeze 时取每组边界的第 0 个
    #   值并去掉 shape 的第 0 维，即 (B, *data_shape) -> (*data_shape)
    elif isinstance(space, gym.spaces.Box):
        if squeeze_batch_dimension:
            return spaces.Box(low=space.low[0], high=space.high[0], shape=space.shape[1:], dtype=space.dtype)
        return spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
    # - Discrete：保留取值个数 n 和起始值 start，旧 Gym 未提供时从 0 开始。
    elif isinstance(space, gym.spaces.Discrete):
        return spaces.Discrete(n=space.n, start=getattr(space, "start", 0))
    # - MultiDiscrete：保留各分量取值个数 nvec 和整数 dtype。
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return spaces.MultiDiscrete(nvec=space.nvec, dtype=space.dtype)
    # 组合空间
    # - Tuple：递归转换各子空间，保持原有顺序。
    elif isinstance(space, gym.spaces.Tuple):
        return spaces.Tuple(
            spaces=tuple([convert_gym_space(s, squeeze_batch_dimension=squeeze_batch_dimension) for s in space.spaces])
        )
    # - Dict：递归转换各子空间，保持原有键
    elif isinstance(space, gym.spaces.Dict):
        return spaces.Dict(
            spaces={
                k: convert_gym_space(v, squeeze_batch_dimension=squeeze_batch_dimension)
                for k, v in space.spaces.items()
            }
        )
    raise ValueError(f"Unsupported space ({space})")


def tensorize_space(space: spaces.Space | None, x: Any, *, device: str | torch.device | None = None) -> Any:
    """把给定 gymnasium 空间的一个样本/值转换为 PyTorch 张量。

    按空间类型补齐批量维并规整形状（见模块说明的形状约定）：

    - Box：``(B, *data_shape)``；输入元素总数须能被 prod(data_shape) 整除，
      B 为商；
    - Discrete：``(B, 1)``，dtype 为 int32；
    - MultiDiscrete：``(B, N)``，dtype 为 int32；
    - Tuple/Dict：递归转换，分别返回 tuple/dict，结构与原样本一致。

    Args:
        space: gymnasium 空间，决定输出的形状与 dtype。
        x: 待张量化的样本/值：可为 torch.Tensor、numpy 数组；
            Discrete 额外接受 Python int/float 或 numpy 标量，
            MultiDiscrete 额外接受 list/tuple。
        device: 新建张量所在的设备。仅在输入不是 Tensor 而需要从数组/
            数字构造张量时使用；输入已是 Tensor 时只做 reshape，
            不改设备与 dtype。

    Returns:
        张量化后的样本：基本空间为 Tensor，组合空间为嵌套的
        tuple/dict（元素为 Tensor）。入参 space 或 x 为 None 时返回 None。

    Raises:
        ValueError: 空间类型不支持，或样本类型与空间不匹配。
    """
    if space is None or x is None:
        return None
    device = config.torch.parse_device(device)
    # 基本空间
    # - Box：reshape 为 (B, *data_shape)，批量维 B 自动推断
    if isinstance(space, spaces.Box):
        if isinstance(x, torch.Tensor):
            return x.reshape(-1, *space.shape)
        elif isinstance(x, np.ndarray):
            # 数组先转 float32 张量再规整形状
            return torch.tensor(x, device=device, dtype=torch.float32).reshape(-1, *space.shape)
        else:
            raise ValueError(f"Unsupported type ({type(x)}) for the given space ({space})")
    # - Discrete：reshape 为 (B, 1)
    elif isinstance(space, spaces.Discrete):
        if isinstance(x, torch.Tensor):
            return x.reshape(-1, 1)
        elif isinstance(x, np.ndarray):
            return torch.tensor(x, device=device, dtype=torch.int32).reshape(-1, 1)
        elif isinstance(x, np.number) or type(x) in [int, float]:
            # 标量：先包成长度 1 的一维张量，再规整为 (1, 1)
            return torch.tensor([x], device=device, dtype=torch.int32).reshape(-1, 1)
        else:
            raise ValueError(f"Unsupported type ({type(x)}) for the given space ({space})")
    # - MultiDiscrete：reshape 为 (B, N)，N 为 nvec 分量个数
    elif isinstance(space, spaces.MultiDiscrete):
        if isinstance(x, torch.Tensor):
            return x.reshape(-1, *space.shape)
        elif isinstance(x, np.ndarray):
            return torch.tensor(x, device=device, dtype=torch.int32).reshape(-1, *space.shape)
        elif type(x) in [list, tuple]:
            # 序列：先包出批量维再规整为 (1, N)
            return torch.tensor([x], device=device, dtype=torch.int32).reshape(-1, *space.shape)
        else:
            raise ValueError(f"Unsupported type ({type(x)}) for the given space ({space})")
    # 组合空间
    # - Tuple：按位置 zip 子空间与子样本，递归转换
    elif isinstance(space, spaces.Tuple):
        return tuple([tensorize_space(s, _x, device=device) for s, _x in zip(space, x)])
    # - Dict：按键取子样本，递归转换
    elif isinstance(space, spaces.Dict):
        return {k: tensorize_space(s, x[k], device=device) for k, s in space.items()}
    raise ValueError(f"Unsupported space ({space})")


def untensorize_space(space: spaces.Space | None, x: Any, *, squeeze_batch_dimension: bool = True) -> Any:
    """把张量化的样本还原为 gymnasium 空间期望的样本类型（numpy）。

    tensorize_space 的逆操作：Tensor 转为 numpy 数组（dtype 转为空间
    的 dtype），并按需去掉批量维：

    - Box：``(B, *data_shape)``；squeeze 且 B == 1 时返回 (*data_shape)；
    - Discrete：``(B, 1)``；squeeze 且 B == 1 时返回 Python 标量
      （``array.item()``）；
    - MultiDiscrete：``(B, N)``；squeeze 且 B == 1 时返回 (N,)；
    - Tuple/Dict：递归还原，分别返回 tuple/dict。

    Args:
        space: gymnasium 空间，决定输出的形状与 dtype。
        x: 张量化的样本：基本空间为 Tensor，组合空间为嵌套 tuple/dict。
        squeeze_batch_dimension: 是否去掉批量维。为 True 时仅影响
            批量维长度为 1 的样本；B > 1 的样本始终保留批量维。

    Returns:
        还原后的样本（numpy 数组或标量，组合空间为嵌套容器）。
        入参 space 或 x 为 None 时返回 None。

    Raises:
        ValueError: 空间类型不支持，或样本类型不是 Tensor。
    """
    if space is None or x is None:
        return None
    # 基本空间
    # - Box：转为 dtype 与空间一致的 numpy 数组
    elif isinstance(space, spaces.Box):
        if isinstance(x, torch.Tensor):
            # 避免 TypeError: Got unsupported ScalarType BFloat16
            # （numpy 不支持 bfloat16，需先转 float32 再转 numpy）
            if x.dtype == torch.bfloat16:
                array = np.array(x.to(dtype=torch.float32).cpu().numpy(), dtype=space.dtype)
            else:
                array = np.array(x.cpu().numpy(), dtype=space.dtype)
            # squeeze 仅在批量维长度为 1 时生效：(1, *data_shape) -> (*data_shape)
            if squeeze_batch_dimension and array.shape[0] == 1:
                return array.reshape(space.shape)
            return array.reshape(-1, *space.shape)
        raise ValueError(f"Unsupported type ({type(x)}) for the given space ({space})")
    # - Discrete：转为 numpy 标量或 (B, 1) 数组
    elif isinstance(space, spaces.Discrete):
        if isinstance(x, torch.Tensor):
            # 避免 TypeError: Got unsupported ScalarType BFloat16
            if x.dtype == torch.bfloat16:
                array = np.array(x.to(dtype=torch.float32).cpu().numpy(), dtype=space.dtype)
            else:
                array = np.array(x.cpu().numpy(), dtype=space.dtype)
            # squeeze 且 B == 1 时返回 Python 标量
            if squeeze_batch_dimension and array.shape[0] == 1:
                return array.item()
            return array.reshape(-1, 1)
        raise ValueError(f"Unsupported type ({type(x)}) for the given space ({space})")
    # - MultiDiscrete：转为 (N,) 或 (B, N) 数组
    elif isinstance(space, spaces.MultiDiscrete):
        if isinstance(x, torch.Tensor):
            # 避免 TypeError: Got unsupported ScalarType BFloat16
            if x.dtype == torch.bfloat16:
                array = np.array(x.to(dtype=torch.float32).cpu().numpy(), dtype=space.dtype)
            else:
                array = np.array(x.cpu().numpy(), dtype=space.dtype)
            # squeeze 且 B == 1 时去掉批量维：(1, N) -> (N,)
            if squeeze_batch_dimension and array.shape[0] == 1:
                return array.reshape(space.nvec.shape)
            return array.reshape(-1, *space.nvec.shape)
        raise ValueError(f"Unsupported type ({type(x)}) for the given space ({space})")
    # 组合空间
    # - Tuple：按位置递归还原
    elif isinstance(space, spaces.Tuple):
        return tuple(
            [untensorize_space(s, _x, squeeze_batch_dimension=squeeze_batch_dimension) for s, _x in zip(space, x)]
        )
    # - Dict：按键递归还原
    elif isinstance(space, spaces.Dict):
        return {
            k: untensorize_space(s, x[k], squeeze_batch_dimension=squeeze_batch_dimension) for k, s in space.items()
        }
    raise ValueError(f"Unsupported space ({space})")


def flatten_tensorized_space(x: Any) -> torch.Tensor | None:
    """把张量化的样本沿特征轴展平为单个张量。

    基本空间的张量 reshape 为 ``(B, D)``；组合空间先递归展平各子样本，
    再沿最后一维（特征轴）拼接，D 为整个空间各子部分展平宽度之和。
    Dict 按键名字典序拼接，与 unflatten_tensorized_space 的切分顺序
    一一对应，保证二者互逆。

    Args:
        x: 张量化的样本：基本空间为 Tensor，组合空间为嵌套的
            list/tuple/dict（元素为 Tensor 或更深的嵌套）。

    Returns:
        展平后的张量，形状 ``(B, D)``；一维张量（无批量维）视为
        B = 1，返回 ``(1, D)``。入参为 None 时返回 None。

    Raises:
        ValueError: 样本类型不支持。
    """
    if x is None:
        return None
    # 基本空间（Box、Discrete 与 MultiDiscrete 的张量化结果）
    # - 已有批量维（ndim > 1）：(B, ...) -> (B, D)；
    # - 无批量维（ndim == 1）：(D,) -> (1, D)
    elif isinstance(x, torch.Tensor):
        return x.reshape(x.shape[0], -1) if x.ndim > 1 else x.reshape(1, -1)
    # 组合空间
    # - Tuple/list：递归展平后沿特征轴拼接（dim=-1）
    elif type(x) in [list, tuple]:
        return torch.cat([flatten_tensorized_space(_x) for _x in x], dim=-1)
    # - Dict：按键名字典序递归展平后拼接，顺序与切分端保持一致
    elif isinstance(x, dict):
        return torch.cat([flatten_tensorized_space(x[k]) for k in sorted(x.keys())], dim=-1)
    raise ValueError(f"Unsupported sample/value type ({type(x)})")


def unflatten_tensorized_space(space: spaces.Space | None, x: torch.Tensor | None) -> Any:
    """把展平的张量按空间结构还原为张量化的样本。

    flatten_tensorized_space 的逆操作：按空间结构把 ``(B, D)`` 的张量
    切分并 reshape 回各子空间的张量化形状（见模块说明的形状约定）。

    Args:
        space: gymnasium 空间，决定切分方式与各部分的形状。
        x: 展平的张量，形状 ``(B, D)``，D 须等于空间各子部分
            （occupied_size 口径）宽度之和。

    Returns:
        张量化的样本（与 tensorize_space 的输出同构）：基本空间为
        Tensor——Box 为 ``(B, *data_shape)``、Discrete 为 ``(B, 1)``、
        MultiDiscrete 为 ``(B, N)``；组合空间为 list（Tuple）或
        dict（Dict）。入参 space 或 x 为 None 时返回 None。

    Raises:
        ValueError: 空间类型不支持。
    """
    if space is None or x is None:
        return None
    # 基本空间
    # - Box：整体 reshape 为 (B, *data_shape)
    elif isinstance(space, spaces.Box):
        return x.reshape(-1, *space.shape)
    # - Discrete：占 1 列，reshape 为 (B, 1)
    elif isinstance(space, spaces.Discrete):
        return x.reshape(-1, 1)
    # - MultiDiscrete：占 N 列，reshape 为 (B, N)
    elif isinstance(space, spaces.MultiDiscrete):
        return x.reshape(-1, *space.shape)
    # 组合空间
    # - Tuple：按子空间顺序切列，每个子空间的宽度按
    #   compute_space_size(occupied_size=True) 计算，起点逐步后移
    elif isinstance(space, spaces.Tuple):
        start = 0
        output = []
        for s in space:
            end = start + compute_space_size(s, occupied_size=True)
            # x[:, start:end]：(B, D) -> (B, end-start)，再递归还原
            output.append(unflatten_tensorized_space(s, x[:, start:end]))
            start = end
        return output
    # - Dict：按键名字典序切列（与 flatten 的拼接顺序一致），起点逐步后移
    elif isinstance(space, spaces.Dict):
        start = 0
        output = {}
        for k in sorted(space.keys()):
            end = start + compute_space_size(space[k], occupied_size=True)
            output[k] = unflatten_tensorized_space(space[k], x[:, start:end])
            start = end
        return output
    raise ValueError(f"Unsupported space ({space})")


def compute_space_size(space: spaces.Space | list[int] | int | None, *, occupied_size: bool = False) -> int:
    """计算空间的尺寸（元素个数）。

    有两种口径，由 occupied_size 决定：

    - occupied_size=False（默认）：按 ``gymnasium.spaces.flatdim`` 计算，
      即 Box 为 prod(shape)、Discrete 为取值个数 n（one-hot 宽度）、
      MultiDiscrete 为 sum(nvec)；
    - occupied_size=True：按张量表示实际占用的元素个数计算
      （与 tensorize_space 的输出形状一致），仅影响 Discrete（占 1，
      单个整数索引）与 MultiDiscrete（占 nvec.shape[0]，即分量个数 N）；
      Box 不受影响，仍为 prod(shape)。

    组合空间 Tuple/Dict 在两种口径下均为各子空间尺寸之和（occupied_size
    口径下递归沿用子空间的口径）。此外也接受非空间输入：int/float 按
    其数值返回，tuple/list 返回各元素乘积（转为 int）。

    Args:
        space: gymnasium 空间，或直接给出尺寸的 int、int 序列。
        occupied_size: 是否按"张量表示实际占用元素个数"的口径计算。

    Returns:
        空间尺寸（元素个数）；入参为 None 时返回 0。
    """
    if space is None:
        return 0
    if occupied_size:
        # 基本空间
        # - Discrete：张量表示中占 1 个元素
        if isinstance(space, spaces.Discrete):
            return 1
        # - MultiDiscrete：张量表示中占 nvec 分量个数 N 个元素
        elif isinstance(space, spaces.MultiDiscrete):
            return space.nvec.shape[0]
        # 组合空间：递归求各子空间尺寸之和
        # - Tuple
        elif isinstance(space, spaces.Tuple):
            return sum([compute_space_size(s, occupied_size=occupied_size) for s in space])
        # - Dict
        elif isinstance(space, spaces.Dict):
            return sum([compute_space_size(s, occupied_size=occupied_size) for s in space.values()])
    # 非 gymnasium 空间：直接给定的尺寸
    if type(space) in [int, float]:
        return space
    elif type(space) in [tuple, list]:
        # 各元素之积，如 (C, H, W) -> C*H*W
        return int(np.prod(space))
    # gymnasium 计算：flatdim（Box 为 prod(shape)，Discrete 为 n，
    # MultiDiscrete 为 sum(nvec)，组合空间为其展开之和）
    return gymnasium.spaces.flatdim(space)


def compute_space_limits(
    space: spaces.Space | None,
    *,
    occupied_size: bool = False,
    device: str | torch.device | None = None,
    none_if_unbounded: Literal["both", "below", "above", "any"] | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """计算空间的上下限（展平为一维的下界/上界张量）。

    Notes:
        只有 Box 空间具有实际的上下限；其余空间视为无界
        （下界 -inf，上界 +inf）。

    Args:
        space: gymnasium 空间。
        occupied_size: 限值数组的长度口径，含义同 compute_space_size：
            仅影响 Discrete（占 1 个元素）与 MultiDiscrete（占 N 个元素）。
        device: 返回张量所在的设备。
        none_if_unbounded: 空间无界时是否返回 None 代替 ±inf：

            - ``"both"``：仅当所有元素双向均无界时，low 与 high
              一起置为 None；
            - ``"below"``：所有元素下界均无界时，low 置为 None；
            - ``"above"``：所有元素上界均无界时，high 置为 None；
            - ``"any"``：与 "below"/"above" 分别独立判断，任一侧满足
              即把该侧置为 None；
            - 未指定（None）：不替换，无界位置保持 -inf/+inf。

    Returns:
        元组 ``(low, high)``：长度均为空间尺寸（按 occupied_size 口径）
        的一维张量，按空间的展平顺序排列（组合空间 Tuple 按子空间顺序、
        Dict 按键名字典序）。入参 space 为 None，或按 none_if_unbounded
        规则判定为无界的一侧，返回 None。
    """

    def _compute_limits(space: spaces.Space, *, low: np.ndarray, high: np.ndarray, index: int, occupied_size: bool):
        """递归地把各 Box 子空间的上下限填入展平限值数组。

        只处理 Box（按展平顺序写入 [index, index+size) 区间）与组合空间；
        Discrete/MultiDiscrete 等非 Box 基本空间不写入，对应位置保持
        调用方初始化的 ±inf。index 在组合空间内随子空间尺寸逐步后移。

        Args:
            space: 当前层的（子）空间。
            low/high: 展平限值数组（原地写入），长度为整空间尺寸。
            index: 当前子空间在展平数组中的起始位置。
            occupied_size: 尺寸口径，同外层函数。
        """
        # 基本空间
        # - Box：把 low/high 展平后填入 [index, index+size) 区间
        if isinstance(space, spaces.Box):
            size = compute_space_size(space, occupied_size=occupied_size)
            low[index : index + size] = space.low.flatten()
            high[index : index + size] = space.high.flatten()
        # 组合空间
        # - Tuple：按子空间顺序递归，index 随子空间尺寸后移
        elif isinstance(space, spaces.Tuple):
            for s in space:
                _compute_limits(s, low=low, high=high, index=index, occupied_size=occupied_size)
                index += compute_space_size(s, occupied_size=occupied_size)
        # - Dict：按键名字典序递归，index 随子空间尺寸后移
        elif isinstance(space, spaces.Dict):
            for k in sorted(space.keys()):
                _compute_limits(space[k], low=low, high=high, index=index, occupied_size=occupied_size)
                index += compute_space_size(space[k], occupied_size=occupied_size)

    if space is None:
        return None, None
    # 先用 ±inf 填满长度为空间尺寸的数组，Box 部分再被覆写为实际限值
    size = compute_space_size(space, occupied_size=occupied_size)
    low = np.full((size,), -float("inf"), dtype=np.float32)
    high = np.full((size,), float("inf"), dtype=np.float32)
    _compute_limits(space, low=low, high=high, index=0, occupied_size=occupied_size)
    # 检查无界空间，按 none_if_unbounded 规则把对应一侧替换为 None
    if none_if_unbounded == "both":
        # 仅当所有元素 low 与 high 同时无界时，两侧一起置 None
        if (np.isinf(low) & np.isinf(high)).all():
            low, high = None, None
    elif none_if_unbounded == "below":
        # 所有元素下界均无界时，low 置 None
        if np.isinf(low).all():
            low = None
    elif none_if_unbounded == "above":
        # 所有元素上界均无界时，high 置 None
        if np.isinf(high).all():
            high = None
    elif none_if_unbounded == "any":
        # 两侧独立判断：任一侧全部无界即把该侧置 None
        if np.isinf(low).all():
            low = None
        if np.isinf(high).all():
            high = None
    # 转为张量（仍为 numpy 数组的一侧）
    device = config.torch.parse_device(device)
    if low is not None:
        low = torch.tensor(low, device=device)
    if high is not None:
        high = torch.tensor(high, device=device)
    return low, high


def sample_space(
    space: spaces.Space | None,
    *,
    batch_size: int = 1,
    backend: Literal["numpy", "native"] = "numpy",
    device: str | torch.device | None = None,
) -> Any:
    """从指定空间生成一个随机批量样本。

    借助 ``gymnasium.vector.utils.batch_space`` 给空间加一维批量后采样，
    再规整为模块说明中的形状约定：

    - Box：``(batch_size, *data_shape)``；
    - Discrete：``(batch_size, 1)``（reshape 到 ``(batch_size, -1)``）；
    - MultiDiscrete：``(batch_size, N)``；
    - Tuple/Dict：递归采样，分别返回 tuple/dict。

    Args:
        space: gymnasium 空间。
        batch_size: 采样的批量维长度 B。
        backend: 构造基本空间样本所用的后端："numpy" 返回 numpy 数组，
            "native" 返回 PyTorch 张量（并使用 device）。
        device: 张量所在的设备；仅在 backend 为 "native" 时使用。

    Returns:
        随机样本：基本空间为 ndarray（"numpy"）或 Tensor（"native"），
        组合空间为嵌套的 tuple/dict。入参 space 为 None 时返回 None。

    Raises:
        ValueError: 空间类型或 backend 不支持。
    """
    if space is None:
        return None
    device = config.torch.parse_device(device)
    # 基本空间
    # - Box：批量空间采样后规整为 (batch_size, *data_shape)
    if isinstance(space, spaces.Box):
        sample = gymnasium.vector.utils.batch_space(space, batch_size).sample()
        if backend == "numpy":
            return np.array(sample).reshape(batch_size, *space.shape)
        elif backend == "native":
            return torch.tensor(sample, device=device).reshape(batch_size, *space.shape)
        else:
            raise ValueError(f"Unsupported backend type ({backend})")
    # - Discrete：规整为 (batch_size, 1)
    elif isinstance(space, spaces.Discrete):
        sample = gymnasium.vector.utils.batch_space(space, batch_size).sample()
        if backend == "numpy":
            return np.array(sample).reshape(batch_size, -1)
        elif backend == "native":
            return torch.tensor(sample, device=device).reshape(batch_size, -1)
        else:
            raise ValueError(f"Unsupported backend type ({backend})")
    # - MultiDiscrete：规整为 (batch_size, N)
    elif isinstance(space, spaces.MultiDiscrete):
        sample = gymnasium.vector.utils.batch_space(space, batch_size).sample()
        if backend == "numpy":
            return np.array(sample).reshape(batch_size, *space.nvec.shape)
        elif backend == "native":
            return torch.tensor(sample, device=device).reshape(batch_size, *space.nvec.shape)
        else:
            raise ValueError(f"Unsupported backend type ({backend})")
    # 组合空间
    # - Tuple：递归采样各子空间，保持原有顺序
    elif isinstance(space, spaces.Tuple):
        return tuple([sample_space(s, batch_size=batch_size, backend=backend, device=device) for s in space])
    # - Dict：递归采样各子空间，保持原有键
    elif isinstance(space, spaces.Dict):
        return {k: sample_space(s, batch_size=batch_size, backend=backend, device=device) for k, s in space.items()}
    raise ValueError(f"Unsupported space ({space})")
