"""Estimate evaluation BatchNorm buffers without changing the training state."""
from contextlib import contextmanager

import torch

from .progress import create_progress_bar


@contextmanager
def recalibrated_batchnorm(model, loader, device, seed, description):
    """Expose training-data-calibrated buffers, then restore the caller's state.

    The supplied loader is independent of the shuffled optimizer-training
    loader. Dropout stays disabled; only tracked BatchNorm layers enter train
    mode, with cumulative means after resetting their running statistics.
    """
    modules = [m for m in model.modules()
               if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
               and m.track_running_stats]
    modes = [(m, m.training) for m in model.modules()]
    originals = [(m, m.momentum, m.running_mean.clone(), m.running_var.clone(),
                  m.num_batches_tracked.clone()) for m in modules]
    generator = loader.generator
    generator_state = generator.get_state()
    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    metadata = {"source_split": "train", "seed": int(seed),
                "batch_size": loader.batch_size, "sample_count": 0,
                "batchnorm_modules": len(modules), "dropout_enabled": False,
                "statistics": "cumulative_mean", "optimizer_updates": 0}
    try:
        with torch.random.fork_rng(devices=devices):
            generator.manual_seed(int(seed))
            model.eval()
            for module in modules:
                module.reset_running_stats()
                module.momentum = None
                module.train()
            with torch.no_grad():
                for joints, bones, _ in create_progress_bar(loader, description):
                    joints = joints.to(device, non_blocking=device.type == "cuda")
                    bones = bones.to(device, non_blocking=device.type == "cuda")
                    model(joints, bones)
                    metadata["sample_count"] += int(joints.shape[0])
            model.eval()
            yield metadata
    finally:
        with torch.no_grad():
            for module, momentum, mean, variance, batches in originals:
                module.momentum = momentum
                module.running_mean.copy_(mean)
                module.running_var.copy_(variance)
                module.num_batches_tracked.copy_(batches)
        for module, training in modes:
            module.training = training
        generator.set_state(generator_state)


__all__ = ["recalibrated_batchnorm"]
