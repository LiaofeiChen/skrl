"""基于 PyTorch 的 gymnasium 空间工具模块。

从 spaces 导入空间转换/尺寸与限值计算/随机采样等工具函数，
经本包命名空间对外提供，具体说明见 spaces 模块。
"""

from skrl.utils.spaces.torch.spaces import (
    compute_space_limits,
    compute_space_size,
    convert_gym_space,
    flatten_tensorized_space,
    sample_space,
    tensorize_space,
    unflatten_tensorized_space,
    untensorize_space,
)
