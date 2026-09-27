"""基于 PyTorch 的随机采样记忆模块。

提供 RandomMemory：在基类 Memory 的环形缓冲存储上实现随机采样策略。
"""

from __future__ import annotations

from typing import Literal

import torch

from skrl.memories.torch import Memory


class RandomMemory(Memory):
    def __init__(
        self,
        *,
        memory_size: int,
        num_envs: int = 1,
        device: str | torch.device | None = None,
        export: bool = False,
        export_format: Literal["pt", "npz", "csv"] = "pt",
        export_directory: str = "",
        replacement: bool = False,
    ) -> None:
        """随机采样记忆（从记忆中随机采样一个批次）。

        存储布局与基类 Memory 相同：(T, E, D) 的环形缓冲张量，
        T 为 memory_size、E 为 num_envs、D 为单条样本的数据宽度；
        采样时在展平视图 (T*E, *data_shape) 的前 ``len(self)`` 个有效行中
        随机抽取索引。

        Args:
            memory_size: 每个张量第 0 维（时间步槽位）的最大长度 T。
            num_envs: 并行环境数 E，即第 1 维长度。
            device: 数据分配与计算设备；未指定时使用全局默认设备。
            export: 是否在记忆写满时导出到文件。为 True 时，记忆一旦写满、
                在环形缓冲开始覆盖最旧数据之前导出一次。
            export_format: 导出文件格式，
                支持 PyTorch (``"pt"``)、NumPy (``"npz"``) 或逗号分隔值 (``"csv"``)。
            export_directory: 导出目录；未指定时使用智能体的实验目录。
            replacement: 是否有放回采样。
                True：同一条目可被多次抽中，批次大小恒等于 batch_size；
                False：每条样本至多抽中一次；有效样本数小于 batch_size 时，
                实际返回的批次大小为 ``len(self)``（可采到的最大数量）。

        Raises:
            ValueError: export_format 不是支持的格式。
        """
        super().__init__(
            memory_size=memory_size,
            num_envs=num_envs,
            device=device,
            export=export,
            export_format=export_format,
            export_directory=export_directory,
        )

        self._replacement = replacement

    def sample(
        self, names: list[str], *, batch_size: int, mini_batches: int = 1, sequence_length: int = 1
    ) -> list[list[torch.Tensor]]:
        """从记忆中随机采样一个批次。

        采样范围为展平视图前 ``len(self)`` 个有效行（行号 b = t*E + e）。
        sequence_length > 1 时进行序列采样：先随机抽取 batch_size 个起点，
        再为每个起点取同一环境内连续 sequence_length 个时间步的展平行
        （步长为 E）；有效起点范围相应缩小 (sequence_length-1)*E，
        保证序列尾部不越过有效数据。

        Args:
            names: 参与采样的张量名称列表；不在记忆中的名称对应位置返回 None。
            batch_size: 采样的索引数量（序列采样时为序列起点数）。
            mini_batches: 切分出的 mini-batch 数量。
            sequence_length: 每个序列的长度；大于 1 时启用序列采样。

        Returns:
            外层列表长度为 mini_batches（mini_batches=1 时为 1），
            内层列表按 names 顺序排列。每个张量的形状：

            - sequence_length == 1：``(batch_size, *data_shape)``；
            - sequence_length > 1：``(batch_size * sequence_length, *data_shape)``，
              每个起点的 sequence_length 个连续时间步相邻排列，序列间按采样顺序。

        Notes:
            replacement=False 且有效样本数不足 batch_size 时，
            返回的批次只包含全部有效样本（见构造参数 replacement 说明）。
        """
        # 有效样本数：展平视图前 len(self) 行
        size = len(self)
        if sequence_length > 1:
            # 记 L = sequence_length。同一环境内连续 L 个时间步在展平视图中的
            # 行号偏移为 [0, E, 2E, ..., (L-1)*E]（步长为 E）
            sequence_indexes = torch.arange(0, self.num_envs * sequence_length, self.num_envs)
            # 有效起点上界缩小 (L-1)*E，保证起点加最大偏移后仍落在有效行内
            size -= sequence_indexes[-1].item()
        # 生成随机索引
        if self._replacement:
            # 有放回：在 [0, size) 均匀抽取 batch_size 个，索引可重复
            indexes = torch.randint(0, size, (batch_size,))
        else:
            # 无放回：随机排列取前 batch_size 个；size < batch_size 时只能取到 size 个。
            # 随机采样性能的讨论见：
            # https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/19
            indexes = torch.randperm(size, dtype=torch.long)[:batch_size]
        # 生成序列索引
        if sequence_length > 1:
            # 偏移 (L,) 经 repeat 得 (B, L)，起点 (B,) 变形为 (B, 1)；
            # 广播相加得 (B, L)，每行 [b, b+E, ..., b+(L-1)*E] 为同一环境内连续 L 步；
            # 再展平为 (B*L,)，序列内部保持连续，序列之间按采样顺序排列
            indexes = (sequence_indexes.repeat(indexes.shape[0], 1) + indexes.view(-1, 1)).view(-1)
        # 记录本次采样索引，并按索引采样
        self.sampling_indexes = indexes
        return self.sample_by_index(names=names, indexes=indexes, mini_batches=mini_batches)
