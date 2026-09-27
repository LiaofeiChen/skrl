"""基于 PyTorch 的记忆（经验回放）基类模块。

提供 Memory 抽象基类：以 ``(memory_size, num_envs, data_size)`` 的环形缓冲
张量存储强化学习交互数据，支持按名称创建/访问张量、按索引采样以及
文件的导入导出。具体的采样策略（如随机采样）由子类实现。
"""

from __future__ import annotations

from typing import Literal

import csv
import datetime
import functools
import operator
import os
from abc import ABC, abstractmethod
import gymnasium

import numpy as np
import torch

from skrl import config
from skrl.utils.spaces.torch import compute_space_size


class Memory(ABC):
    def __init__(
        self,
        *,
        memory_size: int,
        num_envs: int = 1,
        device: str | torch.device | None = None,
        export: bool = False,
        export_format: Literal["pt", "npz", "csv"] = "pt",
        export_directory: str = "",
    ) -> None:
        """基于环形缓冲张量的记忆基类。

        内部所有缓冲张量的形状为 ``(memory_size, num_envs, data_size)``，
        记作 (T, E, D)：T 为 memory_size（每个环境最多存储的样本步数），
        E 为 num_envs（并行环境数），D 为单条样本的数据宽度
        （以 keep_dimensions=True 创建时为 size 各轴展开的多个尾部数据轴）。

        环形缓冲由两个整数索引实现：``memory_index`` 指向维度 0（时间步槽位），
        ``env_index`` 指向维度 1（环境槽位）；写满 T 行后回到行 0 覆盖最旧数据。

        Args:
            memory_size: 每个张量第 0 维（时间步槽位）的最大长度 T。
            num_envs: 并行环境数 E，即第 1 维长度。
            device: 数据分配与计算设备；未指定时使用全局默认设备。
            export: 是否在记忆写满时导出到文件。为 True 时，记忆一旦写满、
                在环形缓冲开始覆盖最旧数据之前导出一次。
            export_format: 导出文件格式，
                支持 PyTorch (``"pt"``)、NumPy (``"npz"``) 或逗号分隔值 (``"csv"``)。
            export_directory: 导出目录；未指定时使用智能体的实验目录。

        Raises:
            ValueError: export_format 不是支持的格式。
        """
        self.memory_size = memory_size
        self.num_envs = num_envs
        self.device = config.torch.parse_device(device)

        # 内部状态变量
        self.filled = False  # 缓冲是否已写满并开始回绕覆盖
        # - 索引
        self.env_index = 0  # 当前写入的环境槽位（对应张量维度 1）
        self.memory_index = 0  # 当前写入的时间步槽位（对应张量维度 0）
        # - 存储分配
        self.tensors = {}  # 张量名 -> 存储张量，形状 (T, E, *data_shape)
        self.tensors_view = {}  # 张量名 -> 展平视图，形状 (T*E, *data_shape)，与存储张量共享内存
        # - 采样
        self.sampling_indexes = None  # 最近一次采样使用的展平行索引，初始为 None
        # 序列采样用的全量索引。展平视图中行号 b = t*E + e，第 i 个环境（0 <= i < E）
        # 的全部时间步对应行号 [i, i+E, ..., i+(T-1)*E]；将 E 个环境的序列依次拼接
        # 得到长度 T*E 的索引，即"按环境分组、组内时间步递增"的顺序，
        # 供 sample_all 的 sequence_length > 1 分支使用
        self.all_sequence_indexes = np.concatenate(
            [np.arange(i, memory_size * num_envs + i, num_envs) for i in range(num_envs)]
        )

        # 数据导出配置
        self.export = export
        self.export_format = export_format
        self.export_directory = export_directory
        if self.export_format not in ["pt", "npz", "csv"]:
            raise ValueError(f"Unsupported export format: '{self.export_format}'")

    def __len__(self) -> int:
        """计算并返回记忆当前的有效样本数。

        有效样本数按下式计算：

        - 已写满（``filled`` 为 True）：``memory_size * num_envs``，即 T*E；
        - 未写满：``memory_index * num_envs + env_index``，即已写满的整行数乘 E，
          再加当前行已写入的环境槽位数。

        Returns:
            有效样本数。顺序写入时等于展平视图 (T*E, *data_shape) 中
            从行 0 开始连续有效的行数，也是随机采样的有效范围上限。
        """
        return self.memory_size * self.num_envs if self.filled else self.memory_index * self.num_envs + self.env_index

    def share_memory(self) -> None:
        """将内部张量设为跨进程共享。

        仅对 CPU 张量调用 ``share_memory_()``；CUDA 张量跳过
        （该操作对 CUDA 张量本就不会共享底层存储）。
        """
        for tensor in self.tensors.values():
            if not tensor.is_cuda:
                tensor.share_memory_()

    def get_tensor_names(self) -> list[str]:
        """获取内部张量名称列表，按字母序排序。

        Returns:
            张量名称列表（``self.tensors`` 的键；键本身不含属性名使用的
            ``_tensor_`` 前缀，前缀只出现在实例属性名中）。
        """
        return sorted(self.tensors.keys())

    def get_tensor_by_name(self, name: str) -> torch.Tensor:
        """按名称获取内部张量。

        Args:
            name: 要获取的张量名称。

        Returns:
            形状为 (T, E, *data_shape) 的存储张量本身（引用而非副本），
            对其原地写入会直接修改记忆内容。

        Raises:
            KeyError: 名称不存在（由字典访问抛出）。
        """
        return self.tensors[name]

    def set_tensor_by_name(self, name: str, tensor: torch.Tensor) -> None:
        """按名称设置（写入）内部张量。

        Args:
            name: 要写入的张量名称。
            tensor: 源张量，形状须与目标存储张量一致或可与之广播；
                数据经 ``copy_()`` 原地复制进记忆，源张量本身不被修改。

        Raises:
            KeyError: 名称不存在（由字典访问抛出）。

        Notes:
            写入在 ``torch.no_grad()`` 下执行，不记录自动求导图。
        """
        with torch.no_grad():
            self.tensors[name].copy_(tensor)

    def create_tensor(
        self,
        name: str,
        *,
        size: int | list[int] | gymnasium.Space | None,
        dtype: torch.dtype | None = None,
        keep_dimensions: bool = False,
    ) -> bool:
        """在记忆中创建一个新的内部张量。

        张量形状为 ``(memory_size, num_envs, data_size)``，即 (T, E, *data_shape)；
        内部以属性 ``_tensor_<name>`` 持有，并登记到 tensors / tensors_view 字典。

        Args:
            name: 张量名称（须符合 PEP 8 命名风格，会拼入属性名）。
            size: 数据轴尺寸，可为整数、整数序列或 gymnasium 空间。
                传入空间时按空间占用的元素数量折算为单个整数宽度。
            dtype: 张量数据类型；为 None 时使用 PyTorch 全局默认类型
                （``torch.zeros`` 不指定 dtype 时的默认值）。
            keep_dimensions: 是否保留 size 的原始各维。

                - False：将 size 折算为单个整数 D，张量形状为 (T, E, D)；
                - True：仅支持整数序列作为 size，张量形状为 (T, E, *size)，
                  例如 size=(C, H, W) 时形状为 (T, E, C, H, W)。

        Returns:
            张量已创建返回 True；size 为 None 时不创建，返回 False；
            同名张量已存在且末轴尺寸与 dtype 均一致时视为已创建，返回 False。

        Raises:
            ValueError: keep_dimensions=True 但 size 不是整数序列；
                或同名张量已存在，但其末轴尺寸或 dtype 与本次请求不一致。

        Notes:
            每次成功创建后，所有已登记的浮点张量都会被填充 NaN：
            数据写入前若被采样到，可尽早通过 NaN 暴露误用。
        """
        # size 为 None 时不创建张量
        if size is None:
            return False
        if keep_dimensions:
            # keep_dimensions=True 时仅支持整数序列作为 size
            if not isinstance(size, (tuple, list)):
                raise ValueError("Only sequences of integers are supported as `size` when `keep_dimensions` is enabled")
        else:
            # 将 gymnasium 空间（或给定尺寸）折算为占用的元素总数，得到单个整数宽度
            size = compute_space_size(size, occupied_size=True)
        # 同名张量已存在时校验末轴尺寸与 dtype
        if name in self.tensors:
            tensor = self.tensors[name]
            if tensor.shape[-1] != size:
                raise ValueError(f"Tensor size ({size}) doesn't match the existing one ({tensor.shape[-1]}): '{name}'")
            if dtype is not None and tensor.dtype != dtype:
                raise ValueError(f"Tensor dtype ({dtype}) doesn't match the existing one ({tensor.dtype}): '{name}'")
            return False
        # 创建张量（属性名 _tensor_<name>）并登记到内部存储。
        # 目标形状：keep_dimensions=False 时 size 为整数 D，形状 (T, E, D)；
        # True 时 size 为序列，形状 (T, E, *size)
        shape = (self.memory_size, self.num_envs, *(size if keep_dimensions else [size]))
        setattr(self, f"_tensor_{name}", torch.zeros(shape, device=self.device, dtype=dtype))
        # 更新内部登记：tensors 持有存储张量；tensors_view 将 (T, E, *data_shape)
        # 展平为 (T*E, *data_shape) 的视图，与存储张量共享内存，行号 b = t*E + e，
        # 供按索引采样使用
        self.tensors[name] = getattr(self, f"_tensor_{name}")
        self.tensors_view[name] = self.tensors[name].view((-1, *shape[2:]))
        # 浮点张量填充 NaN，用于数据写入前的早期误用检测
        for tensor in self.tensors.values():
            if torch.is_floating_point(tensor):
                tensor.fill_(float("nan"))
        return True

    def reset(self) -> None:
        """重置记忆：清零内部索引与标志。

        重置后内部索引与标志恢复默认值：

        - ``filled``: False
        - ``env_index``: 0
        - ``memory_index``: 0

        Notes:
            旧数据仍保留在张量中，直到被新写入覆盖。重置后 ``__len__`` 归零，
            随机采样只在新写入的范围内进行；但 sample_all 等全量接口
            会读出未覆盖的旧数据，有效性需由调用方自行保证。
        """
        self.filled = False
        self.env_index = 0
        self.memory_index = 0

    def add_samples(self, **tensors: dict[str, torch.Tensor]) -> None:
        """向记忆中添加（存储）样本。

        Important:
            各张量须具有相同的二维形状 ``(current_num_envs, data_size)``；
            一维张量视为 current_num_envs=1 的单样本。除入口处的非空检查和
            末尾的形状不匹配报错外，不检查各张量间形状的一致性，
            也不检查写入是否越过有效范围。

        根据传入张量第 0 维（当前环境数）与构造时 num_envs 的关系，
        执行以下分支：

        - ``current_num_envs == num_envs``：写入第 memory_index 行的整行，
          memory_index 加 1；
        - ``current_num_envs < num_envs``：写入第 memory_index 行中
          ``[env_index, env_index + current_num_envs)`` 的环境切片，
          env_index 加 current_num_envs；
        - ``current_num_envs > num_envs`` 且 ``num_envs == 1``：单环境多样本，
          一批样本沿维度 0 写入连续多个时间步；样本数超过剩余容量时
          回绕到缓冲区开头，覆盖最旧数据；
        - 一维张量（隐含 current_num_envs 为 1）：写入单个元素
          ``[memory_index, env_index]``，env_index 加 1。

        Args:
            tensors: 以"张量名 -> 样本张量"传入的关键字参数；
                不在记忆中的名称与值为 None 的项会被跳过。

        Raises:
            ValueError: 未传入任何张量；或张量形状不满足上述分支条件
                （例如二维但第 0 维大于 num_envs 且 num_envs > 1，或维度不是 1/2）。
        """
        if not tensors:
            raise ValueError(
                "There are no samples. Provide samples as key-value arguments, where keys are the tensor names"
            )

        # 各张量的维度与形状（假定所有张量与代表张量具有相同维度和形状）。
        # 优先取 observations（若存在）作为代表张量
        tmp = tensors.get("observations", tensors[next(iter(tensors))])
        dim, shape = tmp.ndim, tmp.shape

        # 多环境整行写入（current_num_envs == num_envs）
        if dim == 2 and shape[0] == self.num_envs:
            for name, tensor in tensors.items():
                if name in self.tensors and tensor is not None:
                    # 写入当前时间步整行：目标切片 (E, *data_shape) 与输入形状一致
                    self.tensors[name][self.memory_index].copy_(tensor)
            self.memory_index += 1
        # 多环境部分写入（current_num_envs < num_envs）
        elif dim == 2 and shape[0] < self.num_envs:
            for name, tensor in tensors.items():
                if name in self.tensors and tensor is not None:
                    # 只写当前行的环境切片 [env_index, env_index+shape[0])，
                    # 目标切片 (shape[0], *data_shape) 与输入形状一致
                    self.tensors[name][self.memory_index, self.env_index : self.env_index + shape[0]].copy_(tensor)
            self.env_index += shape[0]
        # 单环境多样本（num_envs == 1 且 current_num_envs > 1）：一批样本占多个时间步
        elif dim == 2 and self.num_envs == 1:
            for name, tensor in tensors.items():
                if name in self.tensors and tensor is not None:
                    # 本次可连续写入的样本数：不超过样本总数，也不越过缓冲末尾
                    num_samples = min(shape[0], self.memory_size - self.memory_index)
                    # 写入前 num_samples 个样本：tensor[:num_samples] 为
                    # (num_samples, *data_shape)，unsqueeze(1) 补回长度为 1 的环境轴后
                    # 写入 (num_samples, 1, *data_shape) 的连续切片
                    self.tensors[name][self.memory_index : self.memory_index + num_samples].copy_(
                        tensor[:num_samples].unsqueeze(dim=1)
                    )
                    # 剩余样本回绕到缓冲区开头，覆盖最旧数据
                    remaining_samples = shape[0] - num_samples
                    if remaining_samples > 0:
                        # 写入开头 [0, remaining_samples) 的切片，
                        # memory_index 指向剩余样本之后的位置
                        self.tensors[name][:remaining_samples].copy_(tensor[num_samples:].unsqueeze(dim=1))
                        self.memory_index = remaining_samples
                    else:
                        self.memory_index += num_samples
        # 单环境单样本（隐含 current_num_envs == 1）
        elif dim == 1:
            for name, tensor in tensors.items():
                if name in self.tensors and tensor is not None:
                    # 写入 [memory_index, env_index] 处的单个样本 (*data_shape)
                    self.tensors[name][self.memory_index, self.env_index].copy_(tensor)
            self.env_index += 1
        else:
            raise ValueError(
                f"Expected shape (current_num_envs, data_size) where current_num_envs <= {self.num_envs}, got {shape}"
            )

        # 维护索引与标志：当前行的环境槽位写满则推进时间步槽位；
        # 时间步槽位越过末尾则回绕到 0 并置 filled
        if self.env_index >= self.num_envs:
            self.env_index = 0
            self.memory_index += 1
        if self.memory_index >= self.memory_size:
            self.memory_index = 0
            self.filled = True

            # 记忆写满时（覆盖最旧数据前）导出张量到文件
            if self.export:
                self.save(directory=self.export_directory, format=self.export_format)

    @abstractmethod
    def sample(
        self, names: list[str], *, batch_size: int, mini_batches: int = 1, sequence_length: int = 1
    ) -> list[list[torch.Tensor]]:
        """数据采样方法，由子类实现。

        Args:
            names: 参与采样的张量名称列表。
            batch_size: 采样的元素（样本）数量。
            mini_batches: 切分出的 mini-batch 数量。
            sequence_length: 每个序列的长度（用于 RNN 等时序训练）。

        Returns:
            采样结果：外层列表长度为 mini_batches，内层列表按 names
            中的顺序排列；每个张量形状为 ``(batch_size, data_size)``
            （以 keep_dimensions=True 创建时为 ``(batch_size, *data_shape)``）。
        """
        pass

    def sample_by_index(
        self, names: list[str], *, indexes: list | np.ndarray | torch.Tensor, mini_batches: int = 1
    ) -> list[list[torch.Tensor]]:
        """按索引从记忆中采样数据。

        Args:
            names: 参与采样的张量名称列表；不在记忆中的名称对应位置返回 None。
            indexes: 展平视图中的行索引（行号 b = t*E + e），
                可为列表、numpy 数组或整型张量。
            mini_batches: 将索引切分成的 mini-batch 数量。

        Returns:
            外层列表长度为 mini_batches（mini_batches=1 时为 1），
            每个 mini-batch 是按 names 顺序排列的张量列表；
            每个张量形状为 ``(该 mini-batch 的索引数, *data_shape)``，
            由高级索引收集得到的新张量，不与存储张量共享内存。

        Notes:
            mini_batches > 1 时用 ``np.array_split`` 切分索引，
            保持原顺序，各块大小相差不超过 1。
        """
        if mini_batches > 1:
            # 切分为 mini_batches 组索引，块大小相差不超过 1，保持顺序
            batches = np.array_split(indexes, mini_batches)
            return [
                [self.tensors_view[name][batch] if name in self.tensors else None for name in names]
                for batch in batches
            ]
        # 不切分：索引整体作为单个批次返回
        return [[self.tensors_view[name][indexes] if name in self.tensors else None for name in names]]

    def sample_all(
        self, names: list[str], *, mini_batches: int = 1, sequence_length: int = 1
    ) -> list[list[torch.Tensor]]:
        """采样记忆中的全部数据。

        根据 sequence_length 选择取数顺序：

        - ``sequence_length > 1``（序列顺序）：按 ``all_sequence_indexes``
          的顺序取数，即"按环境分组、组内时间步递增"，便于调用方
          按环境重组连续序列；
        - ``sequence_length == 1``（默认顺序）：按展平顺序
          （行号 b = t*E + e，时间步优先）读取全部 T*E 行。

        Args:
            names: 参与采样的张量名称列表；不在记忆中的名称对应位置返回 None。
            mini_batches: 切分出的 mini-batch 数量。
            sequence_length: 每个序列的长度，仅用于选择取数顺序。

        Returns:
            外层列表长度为 mini_batches（mini_batches=1 时为 1），
            内层列表按 names 顺序排列。每个张量的行数：

            - 默认顺序且不切分：T*E 行，形状 (T*E, *data_shape)；
            - 默认顺序且切分：每个 mini-batch 为连续的 batch_size 行，
              batch_size = T*E // mini_batches；不整除时尾部
              T*E % mini_batches 行不属于任何 mini-batch（被丢弃）；
            - 序列顺序：总行数仍为 T*E，按 ``np.array_split`` 切分，
              各块大小相差不超过 1，不丢弃样本。

        Notes:
            本方法读取整个存储张量（T*E 行），不按 ``__len__`` 截断；
            记忆未写满时可能读出 NaN（新建时填充）或重置前的旧数据。
        """
        # 序列顺序：按"环境分组、组内时间步递增"的全量索引取数
        if sequence_length > 1:
            if mini_batches > 1:
                # 切分为 mini_batches 组索引，块大小相差不超过 1，保持顺序
                batches = np.array_split(self.all_sequence_indexes, mini_batches)
                return [
                    [self.tensors_view[name][batch] if name in self.tensors else None for name in names]
                    for batch in batches
                ]
            return [
                [self.tensors_view[name][self.all_sequence_indexes] if name in self.tensors else None for name in names]
            ]
        # 默认顺序：按展平顺序（b = t*E + e，时间步优先）读取
        if mini_batches > 1:
            # 每个 mini-batch 为连续的 batch_size 行；batch_size = T*E // mini_batches，
            # 不整除时尾部余数行不属于任何切片（被丢弃）
            batch_size = (self.memory_size * self.num_envs) // mini_batches
            batches = [(batch_size * i, batch_size * (i + 1)) for i in range(mini_batches)]
            return [
                [self.tensors_view[name][batch[0] : batch[1]] if name in self.tensors else None for name in names]
                for batch in batches
            ]
        # 不切分：直接返回整个展平视图 (T*E, *data_shape)，
        # 与存储张量共享内存（非拷贝），对其原地写入会修改记忆内容
        return [[self.tensors_view[name] if name in self.tensors else None for name in names]]

    def get_sampling_indexes(self) -> list | np.ndarray | torch.Tensor:
        """获取最近一次采样使用的索引。

        Returns:
            最近一次采样的行索引（展平视图行号）；尚未采样过时为
            构造时初始化的 None。
        """
        return self.sampling_indexes

    def save(self, directory: str = "", *, format: Literal["pt", "npz", "csv"] = "pt") -> None:
        """将记忆保存到文件。

        文件写入 ``<directory>/memories/`` 目录，文件名由时间戳与本实例
        的十六进制 id 拼成，后缀由 format 决定。

        Args:
            directory: 保存目录；为空字符串时使用构造时指定的 export_directory。
            format: 文件格式，支持 PyTorch (``"pt"``)、NumPy (``"npz"``)
                或逗号分隔值 (``"csv"``)。

        Raises:
            ValueError: format 不是支持的格式。

        Notes:
            - ``"pt"``：保存 {张量名: 存储张量} 字典，保留 (T, E, *data_shape) 形状；
            - ``"npz"``：各张量转到 CPU 后保存为 npz；
            - ``"csv"``：以追加模式打开文件，先写表头（每列 ``name.i``，
              i 为数据末轴分量下标），再逐行写出展平视图中的样本，
              各张量的分量按张量名顺序拼接。
        """
        if not directory:
            directory = self.export_directory
        os.makedirs(os.path.join(directory, "memories"), exist_ok=True)
        # 导出路径：<directory>/memories/<时间戳>_memory_<实例id十六进制>.<格式>
        memory_path = os.path.join(
            directory,
            "memories",
            f"{datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S-%f')}_memory_{hex(id(self))}.{format}",
        )

        # PyTorch 格式
        if format == "pt":
            torch.save({name: self.tensors[name] for name in self.get_tensor_names()}, memory_path)
        # NumPy 格式
        elif format == "npz":
            np.savez(memory_path, **{name: self.tensors[name].cpu().numpy() for name in self.get_tensor_names()})
        # 逗号分隔值格式
        elif format == "csv":
            with open(memory_path, "a") as file:
                writer = csv.writer(file)
                names = self.get_tensor_names()
                # 写表头：每列为 "name.i"，i 为该张量数据末轴分量的下标
                headers = [[f"{name}.{i}" for i in range(self.tensors_view[name].shape[-1])] for name in names]
                writer.writerow([item for sublist in headers for item in sublist])
                # 写数据行：每行对应展平视图第 i 行样本（共 len(self) 个有效行），
                # 各张量的分量按 names 顺序拼接成一维列表
                for i in range(len(self)):
                    writer.writerow(
                        functools.reduce(operator.iconcat, [self.tensors_view[name][i].tolist() for name in names], [])
                    )
        # 不支持的格式
        else:
            raise ValueError(f"Unsupported export format: '{format}'")

    def load(self, path: str) -> None:
        """从文件加载记忆。

        支持的格式：PyTorch (``"pt"``)、NumPy (``"npz"``)；
        逗号分隔值 (``"csv"``) 不支持加载。

        Args:
            path: 要加载的文件路径，按后缀名识别格式。

        Raises:
            NotImplementedError: csv 文件不支持加载。
            ValueError: 后缀不是支持的格式。

        Notes:
            当前实现通过 ``setattr`` 用加载数据替换各 ``_tensor_<name>`` 属性，
            ``tensors`` 与 ``tensors_view`` 字典中的引用不会随之更新。
            ``"npz"`` 分支在 CPU 上新建张量，不迁移到 ``self.device``。
        """
        # TODO: 支持写入已有张量，而不是新建张量替换属性
        # PyTorch 格式
        if path.endswith(".pt"):
            data = torch.load(path)
            for name in self.get_tensor_names():
                setattr(self, f"_tensor_{name}", data[name])
        # NumPy 格式
        elif path.endswith(".npz"):
            data = np.load(path)
            for name in data:
                setattr(self, f"_tensor_{name}", torch.tensor(data[name]))
        # 逗号分隔值格式
        elif path.endswith(".csv"):
            raise NotImplementedError("Loading from csv is not supported yet")
        # 不支持的格式
        else:
            raise ValueError(f"File at path '{path}' has an unsupported format")
