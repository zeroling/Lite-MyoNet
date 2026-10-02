"""Public full-student training entry point with three-term distillation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ..runtime import choose_device
from .engine import (
    _run_one_training_seed,
    load_skeleton_and_labels,
    load_training_config,
    restore_model,
    skeleton_mapping,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "student.yaml"


def _load_teacher_checkpoint(
    config: dict[str, Any],
    teacher_checkpoint: str | Path | None,
) -> tuple[Path, torch.nn.Module, dict[str, Any]]:
    """Load the first fully restorable teacher from the requested path and its backup."""
    if teacher_checkpoint is not None:
        path = Path(teacher_checkpoint).expanduser().resolve()
    else:
        teacher_config = config["teacher"]
        if not isinstance(teacher_config, dict) or set(teacher_config) != {"checkpoint"}:
            raise ValueError("teacher must contain exactly a checkpoint path.")
        value = teacher_config["checkpoint"]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("teacher.checkpoint must be a non-empty path string.")
        from ..config import resolve_config_path

        path = resolve_config_path(config, value)
    if path.is_dir():
        candidates = [path / "best_teacher.pth", path / "best_teacher.pth.bak"]
    else:
        candidates = [path]
        if path.name == "best_teacher.pth":
            candidates.append(path.with_name("best_teacher.pth.bak"))

    errors: list[str] = []
    for candidate in candidates:
        try:
            model, payload = restore_model(candidate, device="cpu")
        except Exception as error:
            errors.append(f"{candidate}: {type(error).__name__}: {error}")
            continue
        if payload.get("model_type") != "teacher":
            errors.append(
                f"{candidate}: checkpoint model_type is {payload.get('model_type')!r}, expected 'teacher'."
            )
            continue
        return candidate.resolve(), model, payload

    detail = "; ".join(errors) if errors else "no checkpoint candidates were found"
    raise RuntimeError(
        f"No readable teacher checkpoint could be restored from {path}; {detail}."
    )


def train_student(
    config_path: str | Path,
    device: str = "auto",
    resume: str | Path | None = None,
    seed: int | None = None,
    teacher_checkpoint: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Train one complete student with CE and all three teacher-distillation losses."""
    config, _ = load_training_config(config_path, "student")
    selected_seed = int(config["seeds"][0] if seed is None else seed)
    if selected_seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    device_object = choose_device(device)
    skeleton, label_map = load_skeleton_and_labels(config)
    config["_num_classes"] = len(label_map)
    teacher_path, teacher_model, teacher_payload = _load_teacher_checkpoint(
        config, teacher_checkpoint
    )
    if getattr(teacher_model, "num_classes", None) != len(label_map):
        raise ValueError("Teacher and student label maps imply different class counts.")
    teacher_label_map = teacher_payload.get("label_map")
    if teacher_label_map is not None and dict(teacher_label_map) != dict(label_map):
        raise ValueError("Teacher and student label-name-to-index mappings differ.")
    if skeleton_mapping(teacher_model.skeleton) != skeleton_mapping(skeleton):
        raise ValueError("Teacher checkpoint skeleton does not match the student input skeleton.")
    teacher_model.to(device_object).eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)
    return _run_one_training_seed(
        config,
        "student",
        seed=selected_seed,
        device=device_object,
        resume=resume,
        output_root=output_root,
        teacher_model=teacher_model,
        teacher_checkpoint=teacher_path,
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the standalone student-training argument parser."""
    parser = argparse.ArgumentParser(description="Train a full Lite-MyoNet student.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Training YAML configuration.")
    parser.add_argument("--device", default="auto", help="Training device: auto, cpu, or cuda.")
    parser.add_argument(
        "--resume", nargs="?", const="auto", default=None,
        help="Resume the existing run; optional value is auto or a checkpoint/directory path.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the first configured seed.")
    parser.add_argument("--teacher-checkpoint", default=None, help="Override the configured teacher checkpoint.")
    parser.add_argument("--output-root", default=None, help="Override the YAML output root.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run student training from command-line arguments and print its result."""
    args = build_parser().parse_args(argv)
    if args.seed is not None:
        results = [train_student(
            args.config,
            device=args.device,
            resume=args.resume,
            seed=args.seed,
            teacher_checkpoint=args.teacher_checkpoint,
            output_root=args.output_root,
        )]
    else:
        config, _ = load_training_config(args.config, "student")
        seeds = [int(value) for value in config["seeds"]]
        if args.resume not in (None, "auto") and len(seeds) != 1:
            raise ValueError("An explicit checkpoint or directory resume requires --seed when multiple seeds are configured.")
        results = [
            train_student(
                args.config,
                device=args.device,
                resume=args.resume,
                seed=seed,
                teacher_checkpoint=args.teacher_checkpoint,
                output_root=args.output_root,
            )
            for seed in seeds
        ]
    print(json.dumps({"runs": results}, ensure_ascii=False, indent=2))
    return 0


__all__ = ["build_parser", "main", "train_student"]
