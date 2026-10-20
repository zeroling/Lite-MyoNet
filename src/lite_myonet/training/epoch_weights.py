"""Compact inference snapshots; the last checkpoint owns resume state."""
import os
from pathlib import Path
import torch


def save_epoch_weights(run_dir, epoch, payload):
    directory = Path(run_dir) / "epoch_weights"
    directory.mkdir(exist_ok=True)
    destination = directory / f"epoch_{epoch:03d}.pth"
    temporary = destination.with_suffix(".pth.tmp")
    torch.save(dict(payload, epoch=int(epoch), precision="fp32", checkpoint_kind="epoch_weights"), temporary)
    os.replace(temporary, destination)
