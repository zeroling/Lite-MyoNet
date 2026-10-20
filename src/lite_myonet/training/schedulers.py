from torch.optim.lr_scheduler import _LRScheduler
import logging
import sys
import os
import math
from datetime import datetime

class WarmupCosineLR(_LRScheduler):
    """Two stages; the decay is identical regardless of warmup shape/index."""

    def __init__(self, optimizer, total_epochs=300, peak_lr=1e-3,
                 warmup_epochs=30, warmup_init_lr=5e-5, warmup_gamma=0.5,
                 min_lr=1e-6, legacy_warmup_index=False, last_epoch=-1):
        self.total_epochs = int(total_epochs)
        self.peak_lr = float(peak_lr)
        self.warmup_epochs = int(warmup_epochs)
        self.warmup_init_lr = float(warmup_init_lr)
        self.warmup_gamma = float(warmup_gamma)
        self.min_lr = float(min_lr)
        self.legacy_warmup_index = bool(legacy_warmup_index)
        if not 1 < self.warmup_epochs < self.total_epochs:
            raise ValueError("两阶段调度需要至少2轮预热及1轮衰减。")
        for group in optimizer.param_groups:
            group['lr'] = self.warmup_init_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = min(self.last_epoch + 1, self.total_epochs)
        if epoch <= self.warmup_epochs:
            progress = (epoch / self.warmup_epochs if self.legacy_warmup_index
                        else (epoch - 1) / (self.warmup_epochs - 1))
            lr = self.warmup_init_lr + (self.peak_lr - self.warmup_init_lr) * progress ** self.warmup_gamma
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.peak_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
        return [lr for _ in self.base_lrs]

class WarmupHoldCosineLR(_LRScheduler):
    """Teacher-only warmup, hold and cosine decay with a positive final-epoch LR."""

    def __init__(self, optimizer, total_epochs=300, peak_lr=1e-3,
                 warmup_epochs=30, hold_epochs=60, warmup_init_lr=5e-5,
                 warmup_gamma=0.5, min_lr=1e-6, last_epoch=-1):
        if not 1 <= warmup_epochs < total_epochs or not 0 <= hold_epochs < total_epochs - warmup_epochs:
            raise ValueError("预热和保持阶段必须给余弦衰减留下轮次。")
        if not 0 < min_lr <= peak_lr or not 0 <= warmup_init_lr <= peak_lr:
            raise ValueError("学习率上下限配置不正确。")
        self.total_epochs = int(total_epochs)
        self.peak_lr = float(peak_lr)
        self.warmup_epochs = int(warmup_epochs)
        self.hold_epochs = int(hold_epochs)
        self.warmup_init_lr = float(warmup_init_lr)
        self.warmup_gamma = float(warmup_gamma)
        self.min_lr = float(min_lr)
        for group in optimizer.param_groups:
            group['lr'] = self.warmup_init_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = min(self.last_epoch + 1, self.total_epochs)
        hold_end = self.warmup_epochs + self.hold_epochs
        if epoch <= self.warmup_epochs:
            progress = (epoch - 1) / max(1, self.warmup_epochs - 1)
            lr = self.warmup_init_lr + (self.peak_lr - self.warmup_init_lr) * progress ** self.warmup_gamma
        elif epoch <= hold_end:
            lr = self.peak_lr
        else:
            progress = (epoch - hold_end) / (self.total_epochs - hold_end)
            lr = self.min_lr + (self.peak_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
        return [lr for _ in self.base_lrs]


class TwoStageLR(_LRScheduler):
    def __init__(self, optimizer, total_epochs=500, peak_lr=1e-3,
                 warmup_ratio=0.2, warmup_init_lr=5e-5, warmup_gamma=2.0, decay_gamma=2.0, last_epoch=-1):
        """
        Two-Stage Learning Rate Scheduler
        - warmup 阶段: 从 warmup_init_lr 按幂次曲线上升到 peak_lr
        - decay 阶段: 使用旧代码 ``peak_lr*(1-progress**decay_gamma)``。
        - 轮次索引沿用旧 ``last_epoch+1``，包括初始化后的首轮学习率。
        """
        if total_epochs <= 0:
            raise ValueError("total_epochs 必须大于 0。")
        if not 0.0 <= warmup_ratio < 1.0:
            raise ValueError("warmup_ratio 必须位于 [0, 1) 区间。")
        if not 0.0 <= warmup_init_lr <= peak_lr:
            raise ValueError("warmup_init_lr 必须位于 [0, peak_lr] 区间。")

        self.total_epochs = int(total_epochs)
        self.peak_lr = float(peak_lr)
        self.warmup_ratio = warmup_ratio
        self.warmup_init_lr = float(warmup_init_lr)
        self.warmup_gamma = warmup_gamma
        self.decay_gamma = decay_gamma

        self.warmup_epochs = int(self.total_epochs * warmup_ratio)
        self.decay_epochs = self.total_epochs - self.warmup_epochs

        # 初始化优化器学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.warmup_init_lr

        super(TwoStageLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1

        if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
            progress = epoch / self.warmup_epochs
            lr = self.warmup_init_lr + (self.peak_lr - self.warmup_init_lr) * (progress ** self.warmup_gamma)
        elif epoch <= self.total_epochs:
            decay_progress = (epoch - self.warmup_epochs) / self.decay_epochs
            lr = self.peak_lr * (1.0 - decay_progress ** self.decay_gamma)
        else:
            lr = 0.0

        return [lr for _ in self.base_lrs]

def setup_logging(log_dir, experiment_name=""):
    """
    设置日志记录器，将输出同时写入 log 文件和控制台。
    支持多次训练/消融实验，不会重复打印，也不会覆盖已有日志。

    参数:
        log_dir (str): 日志目录
        experiment_name (str): 可选，用于区分不同实验
    返回:
        logger: logging.Logger 对象
        csv_filename: CSV 文件路径
    """
    os.makedirs(log_dir, exist_ok=True)

    # 使用 timestamp 防止覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{experiment_name}_{timestamp}" if experiment_name else timestamp

    log_filename = os.path.join(log_dir, f"{base_name}_train.log")
    csv_filename = os.path.join(log_dir, f"{base_name}_train.csv")

    # 获取 logger 并清空已有 handler，防止重复打印
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    # 文件 handler
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, csv_filename
