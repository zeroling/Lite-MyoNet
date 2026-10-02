"""Create a tiny NCTV dataset for installation and loader smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def create_split(path: Path, sample_count: int, seed: int) -> None:
    """Write one deterministic joints-only NPZ split."""
    generator = np.random.default_rng(seed)
    joints = generator.normal(size=(sample_count, 3, 120, 23)).astype(np.float32)
    labels = np.arange(sample_count, dtype=np.int64) % 2
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, joints=joints, labels=labels)


def main() -> None:
    """Create train, validation, and test splits in a destination directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/dummy"))
    args = parser.parse_args()
    create_split(args.output / "train.npz", 8, 1)
    create_split(args.output / "val.npz", 4, 2)
    create_split(args.output / "test.npz", 4, 3)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
