"""Public full-teacher training entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ..runtime import choose_device
from .engine import _run_one_training_seed, load_training_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "teacher.yaml"


def train_teacher(
    config_path: str | Path,
    device: str = "auto",
    resume: str | Path | None = None,
    seed: int | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Train one complete three-stream teacher and evaluate its selected weights once."""
    config, _ = load_training_config(config_path, "teacher")
    selected_seed = int(config["seeds"][0] if seed is None else seed)
    if selected_seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    device_object = choose_device(device)
    return _run_one_training_seed(
        config,
        "teacher",
        seed=selected_seed,
        device=device_object,
        resume=resume,
        output_root=output_root,
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the standalone teacher-training argument parser."""
    parser = argparse.ArgumentParser(description="Train a full Lite-MyoNet teacher.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Training YAML configuration.")
    parser.add_argument("--device", default="auto", help="Training device: auto, cpu, or cuda.")
    parser.add_argument(
        "--resume", nargs="?", const="auto", default=None,
        help="Resume the existing run; optional value is auto or a checkpoint/directory path.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the first configured seed.")
    parser.add_argument("--output-root", default=None, help="Override the YAML output root.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run teacher training from command-line arguments and print its result."""
    args = build_parser().parse_args(argv)
    result = train_teacher(
        args.config,
        device=args.device,
        resume=args.resume,
        seed=args.seed,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


__all__ = ["build_parser", "main", "train_teacher"]
