"""Runtime controls shared by training entry points."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def _positive_environment_integer(name: str, fallback: int | None = None) -> int | None:
    """Read an optional positive integer from the environment."""
    value = os.environ.get(name)
    if value is None:
        return fallback
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer.") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def configure_torch_cpu_threads_from_environment() -> dict[str, int]:
    """Configure PyTorch CPU thread pools from optional environment variables."""
    num_threads = _positive_environment_integer("TORCH_NUM_THREADS")
    num_interop = _positive_environment_integer(
        "TORCH_NUM_INTEROP_THREADS", fallback=num_threads
    )
    if num_threads is None:
        return {
            "num_threads": int(torch.get_num_threads()),
            "num_interop_threads": int(torch.get_num_interop_threads()),
        }
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(int(num_interop))
    except RuntimeError:
        if int(torch.get_num_interop_threads()) != int(num_interop):
            raise
    return {
        "num_threads": int(num_threads),
        "num_interop_threads": int(num_interop),
    }


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = torch.cuda.is_available()


def choose_device(requested: str) -> torch.device:
    """Resolve an explicit or automatic PyTorch device selection."""
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(normalized)
