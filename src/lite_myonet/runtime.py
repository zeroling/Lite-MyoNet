"""训练进程的轻量运行时约束。"""

import os

import torch


def resume_readable_only():
    """Report the user-selected resume policy without hash verification gates."""
    return True


def _positive_environment_integer(name, fallback=None):
    value = os.environ.get(name)
    if value is None:
        return fallback
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须为正整数。")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须为正整数。") from error
    if result <= 0:
        raise ValueError(f"{name} 必须为正整数。")
    return result


def configure_torch_cpu_threads_from_environment():
    """在任何模型或数据并行工作开始前设置 PyTorch CPU 线程池。"""
    num_threads = _positive_environment_integer("TORCH_NUM_THREADS")
    num_interop_threads = _positive_environment_integer(
        "TORCH_NUM_INTEROP_THREADS", fallback=num_threads
    )
    if num_threads is None:
        return {
            "num_threads": int(torch.get_num_threads()),
            "num_interop_threads": int(torch.get_num_interop_threads()),
        }
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(num_interop_threads)
    except RuntimeError:
        if int(torch.get_num_interop_threads()) != int(num_interop_threads):
            raise
    return {
        "num_threads": int(num_threads),
        "num_interop_threads": int(num_interop_threads),
    }


__all__ = ["configure_torch_cpu_threads_from_environment", "resume_readable_only"]
