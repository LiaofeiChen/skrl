"""运行时标准化预处理器：增量维护均值/方差并做 z-score 变换。"""

from __future__ import annotations

import gymnasium

import torch
import torch.nn as nn

from skrl import config
from skrl.utils.spaces.torch import compute_space_size


class RunningStandardScaler(nn.Module):
    def __init__(
        self,
        size: int | list[int] | gymnasium.Space,
        *,
        epsilon: float = 1e-8,
        clip_threshold: float = 5.0,
        device: str | torch.device | None = None,
    ) -> None:
        """按"去均值、除标准差"标准化输入数据。

        统计量（均值/方差/计数）以 buffer 形式增量维护，随模型一起
        保存/加载。常用作 IPPO/MAPPO 的 observation/state/value
        预处理器：观测走标准化正向；value 走 inverse=True 把归一化
        输出还原回奖励量纲。

        :param size: 输入空间的尺寸（末轴宽度 D，可传 gymnasium 空间）。
        :param epsilon: 防除零的小量。
        :param clip_threshold: 标准化结果的裁剪阈值（对称 ±）。
        :param device: 数据分配与计算设备；未指定时使用默认设备。

        Example::

            >>> running_standard_scaler = RunningStandardScaler(size=2)
            >>> data = torch.rand(3, 2)  # (N, 2)
            >>> running_standard_scaler(data)
            tensor([[0.1954, 0.3356],
                    [0.9719, 0.4163],
                    [0.8540, 0.1982]])
        """
        super().__init__()

        self.epsilon = epsilon
        self.clip_threshold = clip_threshold

        self.device = config.torch.parse_device(device)

        # 折算为末轴宽度 D（标量字段为 1）
        size = compute_space_size(size, occupied_size=True)

        # 统计量用 float64 累积保证精度；注册为 buffer 会进入
        # state_dict，随模型保存/加载、随 .to() 迁移设备。
        # current_count 初始为 1：与初始方差 1 对应的伪计数，
        # 避免首批数据以过小分母主导统计。
        self.register_buffer("running_mean", torch.zeros(size, dtype=torch.float64, device=self.device))
        self.register_buffer("running_variance", torch.ones(size, dtype=torch.float64, device=self.device))
        self.register_buffer("current_count", torch.ones((), dtype=torch.float64, device=self.device))

    def _parallel_variance(self, input_mean: torch.Tensor, input_var: torch.Tensor, input_count: int) -> None:
        """用并行方差算法把本批统计量合并进运行统计量。

        合并公式（Chan et al. 的并行算法）：

            delta = mu_in - mu_run
            M2 = var_run * N_run + var_in * N_in + delta^2 * N_run * N_in / N_total
            mu_run' = mu_run + delta * N_in / N_total
            var_run' = M2 / N_total

        只需本批的 (均值, 方差, 数量) 即可合并，不保留历史数据。

        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param input_mean: 本批数据的均值 (D,)。
        :param input_var: 本批数据的方差 (D,)。
        :param input_count: 本批样本数。
        """
        delta = input_mean - self.running_mean
        total_count = self.current_count + input_count
        # M2 是合并后的平方偏差之和
        M2 = (
            (self.running_variance * self.current_count)
            + (input_var * input_count)
            + delta**2 * self.current_count * input_count / total_count
        )

        # 更新运行统计量
        self.running_mean = self.running_mean + delta * input_count / total_count
        self.running_variance = M2 / total_count
        self.current_count = total_count

    def _compute(self, x: torch.Tensor, *, train: bool = False, inverse: bool = False) -> torch.Tensor:
        """执行标准化（或逆变换）计算。

        :param x: 输入张量 (N, D)；训练统计也接受 (T, E, D)，
            此时沿前两维归约（与记忆存储布局一致），计数为 T*E。
        :param train: 是否用本批数据更新运行统计量（先更新再变换）。
        :param inverse: 是否做逆变换，把标准化数据还原回原量纲。

        :return: 标准化（或还原）后的张量，形状与输入相同。
        """
        if train:
            # 批统计量：mean/var 沿批量维归约到 (D,)
            if x.dim() == 3:
                self._parallel_variance(torch.mean(x, dim=(0, 1)), torch.var(x, dim=(0, 1)), x.shape[0] * x.shape[1])
            else:
                self._parallel_variance(torch.mean(x, dim=0), torch.var(x, dim=0), x.shape[0])
            # 注：torch.var 默认无偏估计（N-1 分母），而合并公式按
            # 总体方差（N 分母）推导，input_var * input_count 略有偏差，
            # 量级 O(1/N)；此为上游实现现状。

        # 逆变换：y * sigma + mu，把 clamp 后的标准化值还原回原表示
        if inverse:
            return (
                torch.sqrt(self.running_variance.float())
                * torch.clamp(x, min=-self.clip_threshold, max=self.clip_threshold)
                + self.running_mean.float()
            )
        # 正向标准化：(x - mu) / (sigma + eps)，再对称裁剪到阈值内
        return torch.clamp(
            (x - self.running_mean.float()) / (torch.sqrt(self.running_variance.float()) + self.epsilon),
            min=-self.clip_threshold,
            max=self.clip_threshold,
        )

    def forward(
        self, x: torch.Tensor | None, *, train: bool = False, inverse: bool = False, no_grad: bool = True
    ) -> torch.Tensor | None:
        """标准化器的前向传播。

        :param x: 输入张量；None 直接原样返回 None（省去调用方判空）。
        :param train: 是否用本批数据更新运行统计量。
        :param inverse: 是否做逆变换还原量纲。
        :param no_grad: 是否禁用梯度计算（默认 True：预处理通常
            不参与反向传播）。

        :return: 标准化（或还原）后的张量。

        Example::

            >>> x = torch.rand(3, 2, device="cuda:0")
            >>> running_standard_scaler(x)
            tensor([[0.6933, 0.1905],
                    [0.3806, 0.3162],
                    [0.1140, 0.0272]], device='cuda:0')

            >>> running_standard_scaler(x, train=True)
            tensor([[ 0.8681, -0.6731],
                    [ 0.0560, -0.3684],
                    [-0.6360, -1.0690]], device='cuda:0')

            >>> running_standard_scaler(x, inverse=True)
            tensor([[0.6260, 0.5468],
                    [0.5056, 0.5987],
                    [0.4029, 0.4795]], device='cuda:0')
        """
        if x is None:
            return None
        if no_grad:
            with torch.no_grad():
                return self._compute(x, train=train, inverse=inverse)
        return self._compute(x, train=train, inverse=inverse)
