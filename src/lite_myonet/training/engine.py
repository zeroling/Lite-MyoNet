"""Shared model construction, checkpoint I/O, and training utilities."""

from __future__ import annotations

import csv
import json
import math
import platform
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
import sklearn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, TensorDataset

from ..config import load_label_map, load_yaml, resolve_config_path
from ..data.nctv_dataset import load_nctv_arrays
from ..evaluation.metrics import classification_metrics
from ..models.skeleton import SkeletonDefinition, load_skeleton, skeleton_from_mapping
from ..models.lite_myonet import LiteMyoNet
from ..models.myoformer import MyoFormer
from ..runtime import choose_device, seed_everything


HISTORY_FIELDS = (
    "epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "val_balanced_accuracy",
    "val_macro_f1",
    "learning_rate",
    "train_seconds",
    "val_seconds",
    "checkpoint_save_seconds",
    "epoch_seconds",
)
MODEL_TYPES = {"teacher", "student"}


def json_compatible(value: Any) -> Any:
    """Convert tensors, NumPy values, paths, and containers to JSON values."""
    if isinstance(value, Mapping):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Cannot serialize value of type {type(value).__name__} to JSON.")


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Write JSON through a flushed same-directory temporary file and replace."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(json_compatible(payload), stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def atomic_write_yaml(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write YAML through a flushed same-directory temporary file and replace."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            yaml.safe_dump(json_compatible(payload), stream, sort_keys=False, allow_unicode=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _fsync_directory(directory: Path) -> None:
    """Best-effort flush directory metadata after atomic replacement."""
    if os.name == "nt":
        return
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    """Load a trusted local PyTorch checkpoint across supported torch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def is_full_training_checkpoint(payload: Any) -> bool:
    """Return whether a payload contains the full portable training state."""
    if not isinstance(payload, dict):
        return False
    required = {
        "schema_version",
        "model_type",
        "model_config",
        "skeleton_config",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "epoch",
        "best_epoch",
        "seed",
    }
    return required.issubset(payload) and payload.get("model_type") in MODEL_TYPES


def atomic_save_checkpoint(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically save a complete checkpoint and preserve the prior readable one."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = Path(str(destination) + ".bak")
    if destination.exists():
        existing_is_readable = False
        try:
            old_payload = _torch_load(destination)
            if not is_full_training_checkpoint(old_payload):
                raise ValueError("Existing checkpoint does not contain full resume state.")
            existing_is_readable = True
        except Exception:
            existing_is_readable = False
        if existing_is_readable:
            backup_temp = backup.with_name(f".{backup.name}.{os.getpid()}-{time.time_ns()}.tmp")
            try:
                with destination.open("rb") as source, backup_temp.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(backup_temp, backup)
                _fsync_directory(destination.parent)
            except Exception:
                if backup_temp.exists():
                    backup_temp.unlink()
                raise RuntimeError(
                    f"Could not preserve prior checkpoint as {backup}; original remains intact."
                )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Validate the temporary archive before it can replace the current file.
        saved = _torch_load(temporary)
        if not is_full_training_checkpoint(saved):
            raise RuntimeError("Refusing to install an incomplete checkpoint payload.")
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def checkpoint_candidates(run_dir: str | Path, model_type: str) -> list[Path]:
    """List last and best checkpoint generations in deterministic priority order."""
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {sorted(MODEL_TYPES)}.")
    root = Path(run_dir)
    best_name = f"best_{model_type}.pth"
    names = (
        "last_checkpoint.pth",
        "last_checkpoint.pth.bak",
        best_name,
        f"{best_name}.bak",
    )
    return [root / name for name in names if (root / name).is_file()]


def readable_checkpoint_candidates(
    run_dir: str | Path, model_type: str
) -> list[tuple[Path, dict[str, Any]]]:
    """Load complete checkpoints and sort candidates by completed epoch."""
    candidates: list[tuple[Path, dict[str, Any]]] = []
    errors: list[str] = []
    for path in checkpoint_candidates(run_dir, model_type):
        try:
            payload = _torch_load(path)
            if not is_full_training_checkpoint(payload):
                raise ValueError("Checkpoint lacks full training-state fields.")
            if payload["model_type"] != model_type:
                raise ValueError(f"Checkpoint model_type is {payload['model_type']!r}.")
            candidates.append((path, payload))
        except Exception as error:
            errors.append(f"{path.name}: {type(error).__name__}: {error}")
    candidates.sort(key=lambda pair: int(pair[1].get("epoch", -1)), reverse=True)
    if not candidates:
        detail = "; ".join(errors) if errors else "no candidate files were found"
        raise RuntimeError(
            f"No readable complete {model_type} checkpoint exists in {run_dir}; {detail}. "
            "Refusing to start a fresh run implicitly."
        )
    return candidates


def skeleton_mapping(skeleton: SkeletonDefinition) -> dict[str, Any]:
    """Serialize a validated skeleton definition as a portable version-1 mapping."""
    return {
        "schema_version": 1,
        "skeleton_id": skeleton.skeleton_id,
        "point_count": skeleton.point_count,
        "root_index": skeleton.root_index,
        "nodes": [
            {"id": node_id, "index": index}
            for index, node_id in enumerate(skeleton.node_ids)
        ],
        "model_graph_edges": [list(edge) for edge in skeleton.model_graph_edges],
        "bone_vector_edges": [list(edge) for edge in skeleton.bone_vector_edges],
        "symmetry_pairs": [list(pair) for pair in skeleton.symmetry_pairs],
        "channels": list(skeleton.channels),
        "frame_count": skeleton.frame_count,
    }


def build_model(
    model_type: str,
    skeleton: SkeletonDefinition,
    model_config: Mapping[str, Any],
) -> torch.nn.Module:
    """Build a full teacher or student from its explicit portable constructor config."""
    configuration = dict(model_config)
    if model_type == "teacher":
        return MyoFormer(skeleton=skeleton, **configuration)
    if model_type == "student":
        return LiteMyoNet(skeleton=skeleton, **configuration)
    raise ValueError(f"Unknown model_type {model_type!r}.")


def restore_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Restore a full model using only the checkpoint-embedded architecture schema."""
    source = Path(checkpoint_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {source}")
    payload = _torch_load(source, map_location=device)
    if not isinstance(payload, dict) or "model_type" not in payload:
        raise ValueError("Checkpoint does not include a portable model_type.")
    model_type = payload["model_type"]
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported checkpoint model_type {model_type!r}.")
    if not isinstance(payload.get("model_config"), dict):
        raise ValueError("Checkpoint is missing model_config.")
    if not isinstance(payload.get("skeleton_config"), dict):
        raise ValueError("Checkpoint is missing embedded skeleton_config.")
    skeleton = skeleton_from_mapping(payload["skeleton_config"])
    model = build_model(model_type, skeleton, payload["model_config"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    return model, payload


class TwoStageLR:
    """Apply the original power-curve warmup and polynomial decay schedule."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_epochs: int,
        warmup_ratio: float = 0.1,
        warmup_init_lr: float = 1.0e-4,
        warmup_gamma: float = 1.0,
        decay_gamma: float = 2.0,
    ) -> None:
        """Initialize the optimizer schedule and immediately set epoch-zero LR."""
        if total_epochs <= 0:
            raise ValueError("total_epochs must be positive.")
        if not 0.0 <= float(warmup_ratio) < 1.0:
            raise ValueError("warmup_ratio must lie in [0, 1).")
        if warmup_gamma <= 0.0 or decay_gamma <= 0.0:
            raise ValueError("warmup_gamma and decay_gamma must be positive.")
        self.optimizer = optimizer
        self.total_epochs = int(total_epochs)
        self.warmup_ratio = float(warmup_ratio)
        self.warmup_init_lr = float(warmup_init_lr)
        self.warmup_gamma = float(warmup_gamma)
        self.decay_gamma = float(decay_gamma)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.warmup_epochs = int(self.total_epochs * self.warmup_ratio)
        self.decay_epochs = max(1, self.total_epochs - self.warmup_epochs)
        self.last_epoch = 0
        self._apply()

    def _lr_at(self, base_lr: float, epoch: int) -> float:
        """Compute one parameter group's rate for a zero-based training epoch."""
        if self.warmup_epochs and epoch < self.warmup_epochs:
            warmup_progress = epoch / max(1, self.warmup_epochs - 1)
            warmup_progress = min(max(warmup_progress, 0.0), 1.0)
            fraction = warmup_progress ** self.warmup_gamma
            return self.warmup_init_lr + fraction * (base_lr - self.warmup_init_lr)
        decay_progress = (epoch - self.warmup_epochs) / self.decay_epochs
        decay_progress = min(max(decay_progress, 0.0), 1.0)
        return base_lr * ((1.0 - decay_progress) ** self.decay_gamma)

    def _apply(self) -> None:
        """Set current optimizer learning rates for the scheduled epoch."""
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self._lr_at(base_lr, self.last_epoch)

    def step(self) -> None:
        """Advance the schedule to the next epoch."""
        self.last_epoch += 1
        self._apply()

    def state_dict(self) -> dict[str, Any]:
        """Return all scheduler state required for exact continuation."""
        return {
            "total_epochs": self.total_epochs,
            "warmup_ratio": self.warmup_ratio,
            "warmup_init_lr": self.warmup_init_lr,
            "warmup_gamma": self.warmup_gamma,
            "decay_gamma": self.decay_gamma,
            "base_lrs": list(self.base_lrs),
            "warmup_epochs": self.warmup_epochs,
            "decay_epochs": self.decay_epochs,
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore scheduler position and settings, rejecting group-count changes."""
        if len(state.get("base_lrs", [])) != len(self.optimizer.param_groups):
            raise ValueError("Scheduler parameter-group count differs from the checkpoint.")
        self.total_epochs = int(state["total_epochs"])
        self.warmup_ratio = float(state["warmup_ratio"])
        self.warmup_init_lr = float(state["warmup_init_lr"])
        self.warmup_gamma = float(state.get("warmup_gamma", 1.0))
        self.decay_gamma = float(state["decay_gamma"])
        self.base_lrs = [float(value) for value in state["base_lrs"]]
        self.warmup_epochs = int(state["warmup_epochs"])
        self.decay_epochs = int(state.get("decay_epochs", max(1, self.total_epochs - self.warmup_epochs)))
        self.last_epoch = int(state["last_epoch"])
        self._apply()


def configure_cpu_threads(count: int) -> None:
    """Constrain PyTorch CPU thread pools to the requested positive count."""
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("cpu_threads must be a positive integer.")
    torch.set_num_threads(count)
    try:
        torch.set_num_interop_threads(count)
    except RuntimeError:
        if torch.get_num_interop_threads() != count:
            raise


def load_training_config(config_path: str | Path, expected_kind: str) -> tuple[dict[str, Any], Path]:
    """Load common training YAML and validate the shared public contract."""
    if expected_kind not in MODEL_TYPES:
        raise ValueError(f"expected_kind must be one of {sorted(MODEL_TYPES)}.")
    config = load_yaml(config_path)
    common = {
        "schema_version", "skeleton_config", "label_map", "data", "output_root",
        "seeds", "epochs", "batch_size", "num_workers", "cpu_threads",
        "precision", "deterministic", "selection_metric", "model", "loss", "optimizer", "scheduler",
    }
    expected = common | ({"teacher", "distillation"} if expected_kind == "student" else set())
    actual = set(config) - {"_config_path"}
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(f"Training config fields mismatch: missing={missing}, unknown={unknown}.")
    if config["schema_version"] != 1:
        raise ValueError("Training schema_version must be 1.")
    for field in ("skeleton_config", "label_map", "output_root"):
        if not isinstance(config[field], str) or not config[field].strip():
            raise ValueError(f"{field} must be a non-empty path string.")
    if isinstance(config["epochs"], bool) or not isinstance(config["epochs"], int) or config["epochs"] <= 0:
        raise ValueError("epochs must be a positive integer.")
    if isinstance(config["batch_size"], bool) or not isinstance(config["batch_size"], int) or config["batch_size"] <= 0:
        raise ValueError("batch_size must be a positive integer.")
    for field in ("num_workers", "cpu_threads"):
        if isinstance(config[field], bool) or not isinstance(config[field], int) or config[field] < (1 if field == "cpu_threads" else 0):
            raise ValueError(f"{field} has an invalid value.")
    if config["precision"] not in {"fp32", "bf16", "fp16"}:
        raise ValueError("precision must be fp32, bf16, or fp16.")
    if config["selection_metric"] != "macro_f1":
        raise ValueError("The public protocol selects checkpoints by validation macro_f1.")
    if not isinstance(config["deterministic"], bool):
        raise ValueError("deterministic must be a boolean.")
    for field in ("model", "loss", "optimizer", "scheduler"):
        if not isinstance(config[field], dict):
            raise ValueError(f"{field} must be a mapping.")
    seeds = config["seeds"]
    if not isinstance(seeds, list) or not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("seeds must be a non-empty list of non-negative integers.")
    skeleton_path = resolve_config_path(config, config["skeleton_config"])
    label_path = resolve_config_path(config, config["label_map"])
    config["_skeleton_path"] = str(skeleton_path)
    config["_label_map_path"] = str(label_path)
    return config, skeleton_path


def resolve_data_entry(config: Mapping[str, Any], entry: Any) -> tuple[Path, Path | None]:
    """Resolve a dataset path and optional labels path relative to its YAML."""
    if isinstance(entry, str):
        return resolve_config_path(config, entry), None
    if not isinstance(entry, Mapping) or set(entry) - {"path", "labels_path"} or "path" not in entry:
        raise ValueError("Each data split must be a path or a {path, labels_path} mapping.")
    data_path = resolve_config_path(config, entry["path"])
    labels_path = resolve_config_path(config, entry["labels_path"]) if entry.get("labels_path") else None
    return data_path, labels_path


def create_data_loaders(
    config: Mapping[str, Any], skeleton: SkeletonDefinition, seed: int
) -> tuple[dict[str, DataLoader], torch.Generator]:
    """Load all declared splits and create deterministic minibatch loaders."""
    data_config = config["data"]
    if not isinstance(data_config, Mapping) or set(data_config) != {"train", "val", "test"}:
        raise ValueError("data must contain exactly train, val, and test splits.")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        path, labels_path = resolve_data_entry(config, data_config[split])
        joints, labels = load_nctv_arrays(path, skeleton, labels_path=labels_path)
        if labels.numel() and int(labels.max()) >= int(config["_num_classes"]):
            raise ValueError(f"{split} contains a label outside label_map class range.")
        dataset = TensorDataset(joints, labels)
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=split == "train",
            num_workers=int(config["num_workers"]),
            generator=generator if split == "train" else None,
            pin_memory=False,
            drop_last=False,
        )
    return loaders, generator


def load_skeleton_and_labels(config: dict[str, Any]) -> tuple[SkeletonDefinition, dict[str, int]]:
    """Load the external skeleton and label mapping once for a run."""
    skeleton = load_skeleton(config["_skeleton_path"])
    labels = load_label_map(config["_label_map_path"])
    config["_num_classes"] = len(labels)
    if len(skeleton.channels) <= 0:
        raise ValueError("Skeleton must provide at least one input channel.")
    return skeleton, labels


def capture_rng_state(loader_generator: torch.Generator) -> dict[str, Any]:
    """Capture all host and available accelerator random-generator states."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "loader": loader_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any], loader_generator: torch.Generator) -> None:
    """Restore Python, NumPy, CPU/CUDA, and data-loader random-generator states."""
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "loader" in state:
        loader_generator.set_state(state["loader"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def append_history(path: Path, record: Mapping[str, Any], *, jsonl_path: Path) -> None:
    """Append one epoch record by atomically rewriting CSV and JSONL histories."""
    row = {field: record.get(field) for field in HISTORY_FIELDS}
    existing: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as stream:
            existing.extend(dict(item) for item in csv.DictReader(stream))
    epoch = int(row["epoch"])
    if existing and int(existing[-1]["epoch"]) >= epoch:
        existing = [item for item in existing if int(item["epoch"]) < epoch]
    existing.append(row)
    _atomic_write_history(path, existing)
    _atomic_write_jsonl(jsonl_path, existing)


def _atomic_write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write complete CSV history with a stable column order and atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write complete JSONL history atomically, one compact object per epoch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(json_compatible(row), ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def reconcile_history(
    run_dir: Path,
    checkpoint_epoch: int,
    checkpoint_record: Mapping[str, Any] | None = None,
) -> None:
    """Repair available logs without making them a prerequisite for checkpoint resume."""
    csv_path = run_dir / "history.csv"
    jsonl_path = run_dir / "history.jsonl"
    retained: dict[int, dict[str, Any]] = {}
    errors: list[str] = []
    invalid_rows = 0
    checkpoint_record_used = False

    def retain_row(row: Any) -> None:
        """Keep a complete finite epoch row only at or before the restored epoch."""
        nonlocal invalid_rows
        try:
            if not isinstance(row, Mapping) or not set(HISTORY_FIELDS).issubset(row):
                raise ValueError("History row has missing columns.")
            epoch = int(row["epoch"])
            if epoch <= 0:
                raise ValueError("History epoch must be positive.")
            normalized = {"epoch": epoch}
            for field in HISTORY_FIELDS[1:]:
                value = float(row[field])
                if not math.isfinite(value):
                    raise ValueError("History row contains a non-finite metric.")
                normalized[field] = value
            if epoch <= checkpoint_epoch:
                retained.setdefault(epoch, normalized)
        except (KeyError, TypeError, ValueError, OverflowError):
            invalid_rows += 1

    if csv_path.exists():
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    retain_row(row)
        except (OSError, UnicodeError, csv.Error) as error:
            errors.append(f"history.csv: {type(error).__name__}: {error}")
    if jsonl_path.exists():
        try:
            with jsonl_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    try:
                        retain_row(json.loads(line))
                    except json.JSONDecodeError:
                        invalid_rows += 1
        except (OSError, UnicodeError) as error:
            errors.append(f"history.jsonl: {type(error).__name__}: {error}")
    if isinstance(checkpoint_record, Mapping) and checkpoint_epoch not in retained:
        if checkpoint_record.get("epoch") == checkpoint_epoch:
            retain_row(checkpoint_record)
            checkpoint_record_used = checkpoint_epoch in retained
    rows = [retained[epoch] for epoch in sorted(retained)]
    missing_count = max(0, checkpoint_epoch - len(rows))
    _atomic_write_history(csv_path, rows)
    _atomic_write_jsonl(jsonl_path, rows)
    if missing_count or invalid_rows or errors or checkpoint_record_used:
        atomic_write_json(run_dir / "history_recovery.json", {
            "schema_version": 1,
            "checkpoint_epoch": checkpoint_epoch,
            "retained_epochs": sorted(retained),
            "missing_epoch_count": missing_count,
            "invalid_rows_discarded": invalid_rows,
            "read_errors": errors,
            "checkpoint_record_used": checkpoint_record_used,
            "checkpoint_record_timing": "pre-save snapshot; lost final save timing is not reconstructed",
            "historical_metrics_fabricated": False,
            "resume_uses_checkpoint_state": True,
        })
        print(
            "History logs were incomplete; continuing from the readable checkpoint "
            "without inventing missing metrics.（历史日志不完整，继续加载可读检查点，不补造缺失指标。）"
        )


def autocast_context(device: torch.device, precision: str):
    """Return a device-appropriate autocast context for the configured precision."""
    if precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True)
    if precision == "fp16":
        return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")
    raise ValueError(f"Unsupported precision {precision!r}.")


def restore_rng_from_checkpoint(payload: Mapping[str, Any], loader_generator: torch.Generator) -> None:
    """Restore RNG snapshots when present, keeping legacy-readable checkpoints usable."""
    if isinstance(payload.get("rng_state"), Mapping):
        restore_rng_state(payload["rng_state"], loader_generator)


def build_checkpoint_payload(
    *,
    model_type: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: TwoStageLR,
    scaler: Any,
    skeleton: SkeletonDefinition,
    model_config: Mapping[str, Any],
    label_map: Mapping[str, int],
    seed: int,
    epoch: int,
    best_epoch: int,
    best_value: float,
    loader_generator: torch.Generator,
    test_count: int,
    history_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a portable architecture description and complete train state."""
    scaler_state = scaler.state_dict() if scaler is not None else {}
    return {
        "schema_version": 1,
        "model_type": model_type,
        "model_config": json_compatible(model_config),
        "label_map": json_compatible(label_map) if label_map is not None else None,
        "skeleton_config": skeleton_mapping(skeleton),
        "num_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "seed": int(seed),
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_value": float(best_value),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler_state,
        "rng_state": capture_rng_state(loader_generator),
        "test_count": int(test_count),
        "test_status": "not_started",
        "history_record": json_compatible(history_record) if history_record is not None else None,
    }


def load_training_checkpoint(
    run_dir: Path,
    model_type: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: TwoStageLR,
    scaler: Any,
    *,
    seed: int,
    configured_epochs: int,
    loader_generator: torch.Generator,
    checkpoint_path: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Restore the newest fully loadable checkpoint and its optimizer trajectory."""
    if checkpoint_path is None:
        candidates = readable_checkpoint_candidates(run_dir, model_type)
    else:
        explicit_path = Path(checkpoint_path).expanduser().resolve()
        if not explicit_path.is_file():
            raise FileNotFoundError(f"Requested resume checkpoint does not exist: {explicit_path}")
        try:
            explicit_payload = _torch_load(explicit_path)
        except Exception as error:
            raise RuntimeError(f"Requested resume checkpoint is unreadable: {explicit_path}: {error}") from error
        if not is_full_training_checkpoint(explicit_payload) or explicit_payload.get("model_type") != model_type:
            raise RuntimeError(f"Requested checkpoint is not a complete {model_type} resume checkpoint.")
        candidates = [(explicit_path, explicit_payload)]
    failures: list[str] = []
    for path, payload in candidates:
        try:
            epoch = int(payload["epoch"])
            if epoch < 0 or epoch > configured_epochs:
                raise ValueError(f"epoch {epoch} is outside [0, {configured_epochs}].")
            if int(payload["seed"]) != int(seed):
                raise ValueError(f"checkpoint seed {payload['seed']} does not match requested {seed}.")
            model.load_state_dict(payload["model_state_dict"], strict=True)
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            scheduler.load_state_dict(payload["scheduler_state_dict"])
            if scaler is not None and payload.get("scaler_state_dict"):
                scaler.load_state_dict(payload["scaler_state_dict"])
            restore_rng_from_checkpoint(payload, loader_generator)
            reconcile_history(run_dir, epoch, payload.get("history_record"))
            return path, payload
        except Exception as error:
            failures.append(f"{path.name}: {type(error).__name__}: {error}")
    raise RuntimeError(
        f"No {model_type} checkpoint in {run_dir} could restore model, optimizer, scheduler, "
        f"and epoch state: {'; '.join(failures)}. Refusing to start fresh."
    )


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Run deterministic inference and return average loss, labels, logits, predictions."""
    model.eval()
    targets: list[np.ndarray] = []
    logits_rows: list[np.ndarray] = []
    total_loss = 0.0
    total_count = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for joints, labels in loader:
            joints = joints.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast_context(device, precision):
                logits = model(joints)
                loss = criterion(logits.float(), labels)
                if not bool(torch.isfinite(logits).all().item()):
                    raise FloatingPointError("Evaluation logits contain NaN or Inf.")
            loss_value = float(loss.item())
            if not math.isfinite(loss_value):
                raise FloatingPointError("Evaluation loss contains NaN or Inf.")
            total_loss += loss_value * labels.shape[0]
            total_count += labels.shape[0]
            targets.append(labels.detach().cpu().numpy().astype(np.int64, copy=False))
            logits_rows.append(logits.detach().float().cpu().numpy())
    if not total_count:
        raise RuntimeError("Evaluation loader is empty.")
    target_array = np.concatenate(targets)
    logits_array = np.concatenate(logits_rows)
    prediction_array = logits_array.argmax(axis=1).astype(np.int64)
    return total_loss / total_count, target_array, logits_array, prediction_array


def calculate_metrics(targets: np.ndarray, predictions: np.ndarray, num_classes: int) -> dict[str, Any]:
    """Compute the package's aggregate and per-class classification metrics."""
    return classification_metrics(targets, predictions, num_classes)


def create_progress_bar(loader: DataLoader, description: str):
    """Create the shared progress renderer, with a red tqdm fallback."""
    try:
        from .progress import create_progress_bar as shared_progress_bar
    except ImportError:
        return tqdm(loader, desc=description, colour="red", dynamic_ncols=True, leave=False)
    return shared_progress_bar(loader, description)


def _model_constructor_config(
    kind: str,
    raw_model: Mapping[str, Any],
    *,
    num_classes: int,
    in_channels: int,
) -> dict[str, Any]:
    """Normalize YAML model values into explicit full-model constructor kwargs."""
    config = dict(raw_model)
    config.pop("name", None)
    config.pop("type", None)
    if "width_ratio" in config:
        raise ValueError("The public trainer uses the full model and does not accept width_ratio.")
    config["num_classes"] = int(num_classes)
    config["in_channels"] = int(in_channels)
    if kind == "teacher":
        aliases = {
            "spatial_attention": "spatial_attention_config",
            "symmetry_attention": "symmetry_attention_config",
            "scale_attention": "scale_attention_config",
            "head": "head_config",
        }
        for alias, canonical in aliases.items():
            if alias in config and canonical not in config:
                config[canonical] = config.pop(alias)
        allowed = {
            "num_classes", "in_channels", "dropout", "spatial_attention_config",
            "symmetry_attention_config", "scale_attention_config", "head_config",
        }
        unknown = sorted(set(config) - allowed)
        if unknown:
            raise ValueError(f"Unknown teacher model fields: {unknown}.")
        defaults = {
            "dropout": 0.05,
            "spatial_attention_config": {"num_heads": 8},
            "symmetry_attention_config": {"num_heads": 8},
            "scale_attention_config": {"num_heads": 8},
            "head_config": {
                "dropout": 0.30,
                "scale_init": 10.0,
                "scale_min": 1.0,
                "scale_max": 20.0,
                "norm": "layernorm",
                "num_mc_samples": 0,
                "mc_dropout_eval": False,
            },
        }
    elif kind == "student":
        allowed = {"num_classes", "in_channels", "channels", "dropout", "fourier_degree"}
        unknown = sorted(set(config) - allowed)
        if unknown:
            raise ValueError(f"Unknown student model fields: {unknown}.")
        defaults = {"channels": [32, 64, 128], "dropout": 0.05, "fourier_degree": 2}
    else:
        raise ValueError(f"Unknown model kind {kind!r}.")
    return {**defaults, **config}


def _make_optimizer(model: torch.nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    """Build the configured AdamW optimizer with reproducible parameter groups."""
    name = str(config.get("type", config.get("name", "AdamW"))).lower()
    if name != "adamw":
        raise ValueError("Only the protocol's AdamW optimizer is supported.")
    lr = float(config.get("lr", config.get("peak_lr", 5.0e-4)))
    betas = config.get("betas", [0.9, 0.999])
    if isinstance(betas, tuple):
        betas = list(betas)
    if not isinstance(betas, list) or len(betas) != 2:
        raise ValueError("optimizer.betas must contain two coefficients.")
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(float(betas[0]), float(betas[1])),
        eps=float(config.get("eps", 1.0e-8)),
        weight_decay=float(config.get("weight_decay", 1.0e-4)),
    )


def _write_run_readme(
    run_dir: Path,
    kind: str,
    seed: int,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    """Write a bilingual run note with the recorded protocol and artifacts."""
    lines = [
        f"# Lite-MyoNet {kind.title()} Run",
        "",
        f"- Seed: `{seed}`",
        f"- Epochs: `{config['epochs']}`",
        f"- Batch size: `{config['batch_size']}`",
        f"- Selection: validation `macro_f1`",
        f"- Precision: `{config['precision']}`",
        f"- Device: `{manifest['device']['name']}` (`{manifest['device']['type']}`)",
        f"- Trainable parameters: `{manifest['model']['num_parameters']}`",
        "- Input: float32 joint coordinates in NCTV layout; bone vectors are derived by the model.",
        "- The test split is evaluated once after training using the validation-selected checkpoint.",
        "",
        "## 运行记录（中文）",
        "",
        f"- 随机种子：`{seed}`；总轮数：`{config['epochs']}`；批大小：`{config['batch_size']}`。",
        f"- 设备：`{manifest['device']['name']}`；有效精度：`{manifest['effective_precision']}`。",
        f"- 可训练参数量：`{manifest['model']['num_parameters']}`；模型按验证集 Macro-F1 选优。",
        "- 输入为 float32、NCTV 排列的 joints；bones 在模型内部从骨架定义动态生成。",
        "- test 仅在训练结束后基于最佳验证权重评估；已提交结果后恢复不会再次评估。",
        "- Python/PyTorch/NumPy/scikit-learn 版本、每个 split 的样本数/形状/类别支持数，以及完整模型和骨架配置记录在 `run_manifest.json`。",
        "",
        "## Files",
        "",
        "- `last_checkpoint.pth`: latest complete resumable state.",
        "- `last_checkpoint.pth.bak`: previous complete resumable state.",
        f"- `best_{kind}.pth`: validation-macro-F1-selected full checkpoint.",
        "- `history.csv` and `history.jsonl`: epoch metrics and wall-clock timing.",
        "- `test_results.json`: one-time test metrics and array artifact names.",
        "",
    ]
    (run_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _build_run_manifest(
    *,
    kind: str,
    seed: int,
    config: Mapping[str, Any],
    device: torch.device,
    skeleton: SkeletonDefinition,
    label_map: Mapping[str, int],
    model: torch.nn.Module,
    model_config: Mapping[str, Any],
    loaders: Mapping[str, DataLoader],
    teacher_checkpoint: str | Path | None,
) -> dict[str, Any]:
    """Collect environment and protocol details without computing content hashes."""
    device_name = str(device)
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    split_records = {}
    class_count = len(label_map)
    for split, loader in loaders.items():
        joints, labels = loader.dataset.tensors
        supports = torch.bincount(labels.to(dtype=torch.long), minlength=class_count)
        split_records[split] = {
            "samples": int(labels.shape[0]),
            "joints_shape": [int(value) for value in joints.shape],
            "joints_dtype": str(joints.dtype),
            "labels_shape": [int(value) for value in labels.shape],
            "class_support": [int(value) for value in supports.tolist()],
        }
    requested_precision = str(config["precision"])
    effective_precision = requested_precision
    if requested_precision == "fp16" and device.type != "cuda":
        effective_precision = "fp32 (fp16 disabled outside CUDA)"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "model_type": kind,
        "seed": int(seed),
        "training": {
            "epochs": int(config["epochs"]),
            "batch_size": int(config["batch_size"]),
            "num_workers": int(config["num_workers"]),
            "cpu_threads": int(config["cpu_threads"]),
            "deterministic": bool(config["deterministic"]),
            "selection_metric": str(config["selection_metric"]),
            "precision_requested": requested_precision,
        },
        "effective_precision": effective_precision,
        "device": {
            "type": device.type,
            "name": device_name,
            "index": device.index,
        },
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "scikit_learn": str(sklearn.__version__),
        },
        "model": {
            "constructor": kind,
            "constructor_config": json_compatible(model_config),
            "num_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        },
        "skeleton": skeleton_mapping(skeleton),
        "label_map": dict(label_map),
        "splits": split_records,
        "loss": json_compatible(config["loss"]),
        "optimizer": json_compatible(config["optimizer"]),
        "scheduler": json_compatible(config["scheduler"]),
        "teacher_checkpoint": str(teacher_checkpoint) if teacher_checkpoint is not None else None,
        "resume_hash_checks": False,
    }
    if kind == "student":
        manifest["distillation"] = json_compatible(config["distillation"])
    return manifest


def _clean_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove internal resolved-path fields from a public config snapshot."""
    return {key: value for key, value in config.items() if not str(key).startswith("_")}


def _resolve_run_directory(
    config: Mapping[str, Any],
    kind: str,
    seed: int,
    output_root: str | Path | None,
) -> Path:
    """Resolve the stable one-seed run directory from YAML or API override."""
    if output_root is None:
        root = resolve_config_path(config, config["output_root"])
    else:
        root = Path(output_root).expanduser().resolve()
    return root / f"{kind}_seed_{seed}"


def _read_test_result(
    run_dir: Path,
    model_type: str,
    seed: int,
    expected_parameters: int,
    num_classes: int,
    expected_test_samples: int,
) -> dict[str, Any] | None:
    """Validate and return an already completed one-time test result, if present."""
    result_path = run_dir / "test_results.json"
    if not result_path.is_file():
        return None
    try:
        with result_path.open("r", encoding="utf-8") as stream:
            result = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Existing test result is unreadable: {result_path}") from error
    if (
        not isinstance(result, dict)
        or result.get("model_type") != model_type
        or result.get("seed") != seed
        or result.get("test_count") != 1
    ):
        raise RuntimeError("Existing test result does not prove exactly one evaluation for this seed.")
    if result.get("num_parameters") != expected_parameters or not isinstance(result.get("metrics"), dict):
        raise RuntimeError("Existing test result has incompatible model metadata.")
    for key in ("targets_file", "predictions_file", "probabilities_file"):
        array_path = run_dir / str(result.get(key, ""))
        if not array_path.is_file():
            raise RuntimeError(f"Existing test result references missing array: {array_path}")
        try:
            loaded = np.load(array_path, allow_pickle=False)
            if key == "targets_file":
                targets = loaded
            elif key == "predictions_file":
                predictions = loaded
            else:
                probabilities = loaded
        except Exception as error:
            raise RuntimeError(f"Existing test array is unreadable: {array_path}") from error
    if targets.ndim != 1 or predictions.shape != targets.shape:
        raise RuntimeError("Existing test targets and predictions have incompatible shapes.")
    if targets.shape[0] != expected_test_samples:
        raise RuntimeError("Existing test array length does not match the configured test split.")
    if probabilities.ndim != 2 or probabilities.shape[0] != targets.shape[0]:
        raise RuntimeError("Existing test probabilities have an incompatible shape.")
    if probabilities.shape[1] != num_classes or not np.isfinite(probabilities).all():
        raise RuntimeError("Existing test probabilities are empty or non-finite.")
    if targets.size and (targets.min() < 0 or predictions.min() < 0 or targets.max() >= probabilities.shape[1] or predictions.max() >= probabilities.shape[1]):
        raise RuntimeError("Existing test labels fall outside the probability class range.")
    if not isinstance(result.get("best_epoch"), int) or result["best_epoch"] <= 0:
        raise RuntimeError("Existing test result does not contain a valid best epoch.")
    return result


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    """Write one NumPy array with flush and atomic same-directory replacement."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, values, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _resolve_best_for_test(run_dir: Path, kind: str) -> tuple[Path, dict[str, Any]]:
    """Load the exact best checkpoint recorded by the newest readable training state."""
    readable = readable_checkpoint_candidates(run_dir, kind)
    expected_best_epoch = int(readable[0][1]["best_epoch"])
    best_name = f"best_{kind}.pth"
    errors: list[str] = []
    found = []
    for path in (run_dir / best_name, run_dir / f"{best_name}.bak"):
        if not path.is_file():
            continue
        try:
            payload = _torch_load(path)
            if not is_full_training_checkpoint(payload) or payload.get("model_type") != kind:
                raise ValueError("Checkpoint is not a complete checkpoint of this model type.")
            if int(payload["epoch"]) != expected_best_epoch or int(payload["best_epoch"]) != expected_best_epoch:
                raise ValueError(
                    f"best epoch {payload['epoch']} does not match recorded best epoch {expected_best_epoch}."
                )
            found.append((path, payload))
        except Exception as error:
            errors.append(f"{path.name}: {type(error).__name__}: {error}")
    if not found:
        raise RuntimeError(
            f"No readable best_{kind} checkpoint matches recorded best epoch {expected_best_epoch}; "
            f"{' ; '.join(errors) if errors else 'no best files exist'}. Refusing to test last weights."
        )
    return found[0]


def _start_or_retry_test(
    run_dir: Path,
    *,
    seed: int,
    best_epoch: int,
    best_path: Path,
) -> int:
    """Record a one-time test attempt or retry only the same selected checkpoint."""
    marker_path = run_dir / "test_evaluation_started.json"
    attempts = 0
    if marker_path.exists():
        try:
            with marker_path.open("r", encoding="utf-8") as stream:
                marker = json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("Existing test-attempt marker is unreadable.") from error
        if not isinstance(marker, dict) or int(marker.get("seed", -1)) != seed:
            raise RuntimeError("Existing test-attempt marker belongs to a different run.")
        if int(marker.get("best_epoch", -1)) != best_epoch:
            raise RuntimeError("A prior test attempt used a different best epoch; refusing to mix results.")
        if marker.get("status") == "completed":
            raise RuntimeError("The test marker says evaluation completed but no committed result exists.")
        attempts = int(marker.get("attempts", 1))
    attempts += 1
    atomic_write_json(marker_path, {
        "schema_version": 1,
        "seed": seed,
        "test_count": 0,
        "status": "started" if attempts == 1 else "retrying",
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_path),
        "attempts": attempts,
        "attempt_started_at_unix": time.time(),
    })
    return attempts


def _run_one_training_seed(
    config: dict[str, Any],
    kind: str,
    *,
    seed: int,
    device: torch.device,
    resume: str | Path | None,
    output_root: str | Path | None,
    teacher_model: torch.nn.Module | None = None,
    teacher_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Train one full model seed, select by validation macro-F1, then test once."""
    configure_cpu_threads(int(config["cpu_threads"]))
    seed_everything(int(seed), deterministic=bool(config["deterministic"]))
    skeleton, label_map = load_skeleton_and_labels(config)
    config["_num_classes"] = len(label_map)
    model_config = _model_constructor_config(
        kind,
        config["model"],
        num_classes=len(label_map),
        in_channels=len(skeleton.channels),
    )
    model = build_model(kind, skeleton, model_config).to(device)
    num_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if teacher_model is not None:
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)
    loaders, loader_generator = create_data_loaders(config, skeleton, seed)
    optimizer = _make_optimizer(model, config["optimizer"])
    scheduler_config = config["scheduler"]
    scheduler = TwoStageLR(
        optimizer,
        total_epochs=int(config["epochs"]),
        warmup_ratio=float(scheduler_config.get("warmup_ratio", 0.1)),
        warmup_init_lr=float(scheduler_config.get("warmup_init_lr", 1.0e-4)),
        warmup_gamma=float(scheduler_config.get("warmup_gamma", 1.0)),
        decay_gamma=float(scheduler_config.get("decay_gamma", 2.0)),
    )
    scaler_enabled = device.type == "cuda" and config["precision"] == "fp16"
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    run_dir = _resolve_run_directory(config, kind, seed, output_root)
    explicit_resume_path: Path | None = None
    if resume is None:
        if run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {run_dir}; pass --resume auto to continue.")
        run_dir.mkdir(parents=True, exist_ok=False)
        atomic_write_yaml(run_dir / "resolved_config.yaml", _clean_config(config))
        atomic_write_json(run_dir / "skeleton.json", skeleton_mapping(skeleton))
        atomic_write_json(run_dir / "run_status.json", {
            "schema_version": 1, "model_type": kind, "seed": seed, "status": "running",
            "epoch": 0, "best_epoch": 0, "test_count": 0,
        })
        (run_dir / "training.log").write_text("", encoding="utf-8")
    else:
        if isinstance(resume, str) and resume.strip().lower() == "auto":
            if not run_dir.is_dir():
                raise FileNotFoundError(f"Cannot resume missing run directory: {run_dir}")
        else:
            requested = Path(resume).expanduser().resolve()
            if not requested.exists():
                raise FileNotFoundError(f"Requested resume checkpoint does not exist: {requested}")
            if requested.is_dir():
                run_dir = requested
            else:
                explicit_resume_path = requested
                run_dir = requested.parent
    run_manifest = _build_run_manifest(
        kind=kind,
        seed=seed,
        config=config,
        device=device,
        skeleton=skeleton,
        label_map=label_map,
        model=model,
        model_config=model_config,
        loaders=loaders,
        teacher_checkpoint=teacher_checkpoint,
    )
    run_manifest["runtime"] = {
        "torch_cpu_threads": int(torch.get_num_threads()),
        "torch_interop_threads": int(torch.get_num_interop_threads()),
    }
    atomic_write_json(run_dir / "run_manifest.json", run_manifest)
    _write_run_readme(run_dir, kind, seed, config, run_manifest)
    try:
        existing_test = _read_test_result(
            run_dir,
            kind,
            seed,
            num_parameters,
            len(label_map),
            len(loaders["test"].dataset),
        )
        if existing_test is not None:
            best_path, best_payload = _resolve_best_for_test(run_dir, kind)
            if int(existing_test["best_epoch"]) != int(best_payload["epoch"]):
                raise RuntimeError("Committed test result best_epoch does not match the saved best checkpoint.")
            run_manifest["selected_checkpoint"] = str(best_path)
            run_manifest["best_epoch"] = int(existing_test["best_epoch"])
            run_manifest["best_validation_macro_f1"] = float(existing_test.get("best_validation_macro_f1", best_payload.get("best_value", 0.0)))
            run_manifest["test_metrics"] = existing_test["metrics"]
            run_manifest["test_count"] = 1
            run_manifest["test_attempts"] = int(existing_test.get("test_attempts", 1))
            atomic_write_json(run_dir / "run_manifest.json", run_manifest)
            atomic_write_json(run_dir / "test_evaluation_started.json", {
                "schema_version": 1, "seed": seed, "test_count": 1,
                "status": "completed", "best_epoch": int(existing_test["best_epoch"]),
                "best_checkpoint": str(best_path),
                "attempts": int(existing_test.get("test_attempts", 1)),
                "reconciled_at_unix": time.time(),
            })
            atomic_write_json(run_dir / "run_status.json", {
                "schema_version": 1, "model_type": kind, "seed": seed,
                "status": "completed", "epoch": int(config["epochs"]),
                "best_epoch": int(existing_test["best_epoch"]),
                "best_value": run_manifest["best_validation_macro_f1"],
                "test_count": 1, "best_checkpoint": str(best_path),
                "test_results": "test_results.json",
            })
            if not (run_dir / "README.md").is_file():
                _write_run_readme(run_dir, kind, seed, config, run_manifest)
            return {
                "run_dir": str(run_dir), "best_checkpoint": str(best_path), "seed": seed,
                "test_metrics": existing_test["metrics"], "num_parameters": num_parameters,
            }
        if resume is not None:
            checkpoint_path, checkpoint = load_training_checkpoint(
                run_dir,
                kind,
                model,
                optimizer,
                scheduler,
                scaler,
                seed=seed,
                configured_epochs=int(config["epochs"]),
                loader_generator=loader_generator,
                checkpoint_path=explicit_resume_path,
            )
            start_epoch = int(checkpoint["epoch"]) + 1
            best_epoch = int(checkpoint["best_epoch"])
            best_value = float(checkpoint.get("best_value", float("-inf")))
            atomic_write_json(run_dir / "run_status.json", {
                "schema_version": 1, "model_type": kind, "seed": seed, "status": "running",
                "epoch": int(checkpoint["epoch"]), "best_epoch": best_epoch, "test_count": 0,
                "resumed_from": str(checkpoint_path),
            })
        else:
            start_epoch = 1
            best_epoch = 0
            best_value = float("-inf")
        criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=float(config["loss"].get("label_smoothing", 0.0))
        )
        distillation = None
        if kind == "student":
            from .distillation import DistillationLoss

            distillation_config = dict(config["distillation"])
            distillation = DistillationLoss(**distillation_config)
            weights = (
                distillation.kd_weight,
                distillation.static_adjacency_weight,
                distillation.dynamic_similarity_weight,
            )
            if not all(math.isfinite(weight) and weight > 0.0 for weight in weights):
                raise ValueError("All three distillation loss weights must be finite and positive.")
            if teacher_model is None:
                raise ValueError("Student training requires a restored teacher model.")
        for epoch in range(start_epoch, int(config["epochs"]) + 1):
            epoch_started = time.perf_counter()
            train_started = time.perf_counter()
            model.train()
            if teacher_model is not None:
                teacher_model.eval()
            train_loss_sum = 0.0
            train_correct = 0
            train_count = 0
            progress = create_progress_bar(loaders["train"], f"{kind} seed {seed} epoch {epoch}/{config['epochs']}")
            for joints, labels in progress:
                joints = joints.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, str(config["precision"])):
                    if kind == "teacher":
                        logits = model(joints)
                        total_loss = criterion(logits.float(), labels)
                    else:
                        student_logits, student_features = model(joints, return_features=True)
                        with torch.no_grad():
                            teacher_logits, teacher_features = teacher_model(joints, return_features=True)
                        components = distillation(
                            student_logits,
                            teacher_logits,
                            student_features,
                            teacher_features,
                            physical_adjacency=model.physical_adjacency,
                        )
                        ce_loss = criterion(student_logits.float(), labels)
                        total_loss = ce_loss + components["loss_total"]
                        logits = student_logits
                if not bool(torch.isfinite(logits).all().item()):
                    raise FloatingPointError("Training logits contain NaN or Inf.")
                batch_loss_value = float(total_loss.detach().float().item())
                if not math.isfinite(batch_loss_value):
                    raise FloatingPointError("Training loss contains NaN or Inf.")
                if scaler_enabled:
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()
                batch_count = int(labels.shape[0])
                train_loss_sum += batch_loss_value * batch_count
                train_correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
                train_count += batch_count
                if hasattr(progress, "set_postfix"):
                    progress.set_postfix(loss=f"{batch_loss_value:.4f}")
            train_seconds = time.perf_counter() - train_started
            if train_count <= 0:
                raise RuntimeError("Training loader produced no examples.")
            train_loss = train_loss_sum / train_count
            train_accuracy = train_correct / train_count
            val_started = time.perf_counter()
            val_loss, val_targets, val_logits, val_predictions = collect_predictions(
                model, loaders["val"], device=device, precision=str(config["precision"])
            )
            val_metrics = calculate_metrics(val_targets, val_predictions, len(label_map))
            val_seconds = time.perf_counter() - val_started
            score = float(val_metrics["macro_f1"])
            improved = score > best_value or best_epoch == 0
            if improved:
                best_epoch = epoch
                best_value = score
            learning_rate = float(optimizer.param_groups[0]["lr"])
            scheduler.step()
            history_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_macro_f1": score,
                "learning_rate": learning_rate,
                "train_seconds": train_seconds,
                "val_seconds": val_seconds,
                "checkpoint_save_seconds": 0.0,
                "epoch_seconds": train_seconds + val_seconds,
            }
            append_history(run_dir / "history.csv", history_record, jsonl_path=run_dir / "history.jsonl")
            payload = build_checkpoint_payload(
                model_type=kind,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                skeleton=skeleton,
                model_config=model_config,
                seed=seed,
                epoch=epoch,
                best_epoch=best_epoch,
                best_value=best_value,
                loader_generator=loader_generator,
                test_count=0,
                history_record=history_record,
                label_map=label_map,
            )
            save_started = time.perf_counter()
            if improved:
                atomic_save_checkpoint(payload, run_dir / f"best_{kind}.pth")
            atomic_save_checkpoint(payload, run_dir / "last_checkpoint.pth")
            checkpoint_save_seconds = time.perf_counter() - save_started
            history_record["checkpoint_save_seconds"] = checkpoint_save_seconds
            history_record["epoch_seconds"] = train_seconds + val_seconds + checkpoint_save_seconds
            append_history(run_dir / "history.csv", history_record, jsonl_path=run_dir / "history.jsonl")
            log_line = (
                f"epoch={epoch} train_loss={train_loss:.6f} train_accuracy={train_accuracy:.6f} "
                f"val_loss={val_loss:.6f} val_accuracy={val_metrics['accuracy']:.6f} "
                f"val_macro_f1={score:.6f} best_epoch={best_epoch} "
                f"train_seconds={train_seconds:.3f} val_seconds={val_seconds:.3f} "
                f"save_seconds={checkpoint_save_seconds:.3f} "
                f"epoch_seconds={time.perf_counter() - epoch_started:.3f}\n"
            )
            with (run_dir / "training.log").open("a", encoding="utf-8") as stream:
                stream.write(log_line)
                stream.flush()
                os.fsync(stream.fileno())
            atomic_write_json(run_dir / "run_status.json", {
                "schema_version": 1, "model_type": kind, "seed": seed, "status": "running",
                "epoch": epoch, "best_epoch": best_epoch, "best_value": best_value,
                "test_count": 0, "latest_checkpoint": "last_checkpoint.pth",
            })
            print(log_line.rstrip())
        best_path, best_payload = _resolve_best_for_test(run_dir, kind)
        model.load_state_dict(best_payload["model_state_dict"], strict=True)
        model.to(device)
        marker_path = run_dir / "test_evaluation_started.json"
        test_attempt = _start_or_retry_test(
            run_dir,
            seed=seed,
            best_epoch=int(best_payload["epoch"]),
            best_path=best_path,
        )
        _, test_targets, test_logits, test_predictions = collect_predictions(
            model, loaders["test"], device=device, precision=str(config["precision"])
        )
        test_metrics = calculate_metrics(test_targets, test_predictions, len(label_map))
        probabilities = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy().astype(np.float32)
        _atomic_save_npy(run_dir / "test_targets.npy", test_targets.astype(np.int64, copy=False))
        _atomic_save_npy(run_dir / "test_predictions.npy", test_predictions.astype(np.int64, copy=False))
        _atomic_save_npy(run_dir / "test_probabilities.npy", probabilities)
        test_result = {
            "schema_version": 1, "model_type": kind, "seed": seed, "test_count": 1,
            "best_epoch": best_epoch, "best_validation_macro_f1": best_value,
            "test_attempts": test_attempt,
            "best_checkpoint": str(best_path), "num_parameters": num_parameters,
            "metrics": test_metrics, "targets_file": "test_targets.npy",
            "predictions_file": "test_predictions.npy", "probabilities_file": "test_probabilities.npy",
            "num_test_samples": int(test_targets.shape[0]),
            "completed_at_unix": time.time(),
        }
        atomic_write_json(run_dir / "test_results.json", test_result)
        run_manifest["selected_checkpoint"] = str(best_path)
        run_manifest["best_epoch"] = best_epoch
        run_manifest["best_validation_macro_f1"] = best_value
        run_manifest["test_metrics"] = test_metrics
        run_manifest["test_count"] = 1
        atomic_write_json(run_dir / "run_manifest.json", run_manifest)
        atomic_write_json(marker_path, {
            "schema_version": 1, "seed": seed, "test_count": 1, "status": "completed",
            "best_epoch": best_epoch, "best_checkpoint": str(best_path),
            "attempts": test_attempt,
            "completed_at_unix": test_result["completed_at_unix"],
        })
        atomic_write_json(run_dir / "run_status.json", {
            "schema_version": 1, "model_type": kind, "seed": seed, "status": "completed",
            "epoch": int(config["epochs"]), "best_epoch": best_epoch, "best_value": best_value,
            "test_count": 1, "best_checkpoint": str(best_path), "test_results": "test_results.json",
        })
        return {
            "run_dir": str(run_dir), "best_checkpoint": str(best_path), "seed": seed,
            "test_metrics": test_metrics, "num_parameters": num_parameters,
        }
    except Exception:
        if run_dir.exists():
            status_path = run_dir / "run_status.json"
            if status_path.exists():
                try:
                    with status_path.open("r", encoding="utf-8") as stream:
                        status = json.load(stream)
                    if status.get("status") != "completed":
                        status["status"] = "failed"
                        status["error_recorded_at_unix"] = time.time()
                        atomic_write_json(status_path, status)
                except Exception:
                    pass
        raise


__all__ = [
    "TwoStageLR", "append_history", "atomic_save_checkpoint", "atomic_write_json",
    "atomic_write_yaml", "autocast_context", "build_checkpoint_payload", "build_model",
    "calculate_metrics", "capture_rng_state", "checkpoint_candidates", "choose_device",
    "collect_predictions", "configure_cpu_threads", "create_data_loaders",
    "is_full_training_checkpoint", "json_compatible", "load_skeleton_and_labels",
    "load_training_checkpoint", "load_training_config", "readable_checkpoint_candidates",
    "restore_model", "restore_rng_from_checkpoint", "resolve_config_path", "resolve_data_entry",
    "seed_everything", "skeleton_mapping",
]
