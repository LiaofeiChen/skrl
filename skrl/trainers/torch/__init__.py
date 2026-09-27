"""基于 PyTorch 的训练器模块。

统一从 base 导入 Trainer/TrainerCfg/generate_equally_spaced_scopes，
从 step/sequential/parallel 导入对应训练器及其配置，经本包命名空间
对外提供。三种训练器的区别：

- StepTrainer：单步驱动，循环由调用方控制；
- SequentialTrainer：整循环内置，多智能体同进程串行执行；
- ParallelTrainer：整循环内置，每个同时智能体独立进程并行执行。
"""

from skrl.trainers.torch.base import Trainer, TrainerCfg, generate_equally_spaced_scopes  # isort:skip

from skrl.trainers.torch.parallel import ParallelTrainer, ParallelTrainerCfg
from skrl.trainers.torch.sequential import SequentialTrainer, SequentialTrainerCfg
from skrl.trainers.torch.step import StepTrainer, StepTrainerCfg
