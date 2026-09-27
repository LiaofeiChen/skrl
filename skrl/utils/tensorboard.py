"""轻量 TensorBoard 事件写入器：直接包装 tensorboard 原生 EventFileWriter。"""

from __future__ import annotations

import time
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


class SummaryWriter:
    def __init__(self, log_dir: str, *, queue_size: int = 10, flush_seconds: int = 120):
        """TensorBoard 标量摘要写入器。

        不经 torch.utils.tensorboard，直接用 tensorboard 包的
        EventFileWriter 手写事件协议（Event/Summary proto），
        避免对 torch 的 TensorBoard 集成的依赖。智能体的
        track_data -> write_tracking_data 路径最终落到这里的
        add_scalar。

        :param log_dir: 事件文件的保存目录（即 TensorBoard 的
            --logdir 指向的实验目录）。
        :param queue_size: 事件队列上限；积满该数量即强制落盘。
        :param flush_seconds: 待写事件的最长滞留秒数，到时刷新落盘。
        """
        self._event_file_writer = EventFileWriter(log_dir, max_queue_size=queue_size, flush_secs=flush_seconds)

    def add_scalar(self, *, tag: str, value: float, timestep: int) -> None:
        """向摘要添加一个标量值。

        与 torch 的同名方法不同，本方法为纯关键字参数。

        :param tag: 标量记录名（如 'Loss / Policy loss'）。
        :param value: 标量值。
        :param timestep: 记录对应的全局步数（x 轴）。
        """
        # 手工构造事件协议：Summary.Value 承载标量，step 定 x 轴，
        # wall_time 记录墙钟时间；实际写入由后台线程异步完成。
        event = Event(summary=Summary(value=[Summary.Value(tag=tag, simple_value=value)]))
        event.step = timestep
        event.wall_time = time.time()
        self._event_file_writer.add_event(event)

    def flush(self) -> None:
        """把待写事件立即刷新到持久存储。"""
        self._event_file_writer.flush()

    def close(self) -> None:
        """关闭写入器（关闭前会处理剩余队列）。"""
        self._event_file_writer.close()
