import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import tempfile
import time
import uuid
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from torch.utils.data import DataLoader
import tqdm as tqdm_package
from tqdm import tqdm

from ..data.manifest import verify_manifest
from ..data.packed import InMemoryNpzSkeletonDataset, verify_dataset_archive
from ..data.synthetic import SkeletonDataset
from ..models.teacher import MyoFormerTeacher
from ..runtime import configure_torch_cpu_threads_from_environment, resume_readable_only
from .checkpoint_io import load_latest_checkpoint, save_checkpoint_atomic as _durable_save_checkpoint_atomic
from .schedulers import TwoStageLR, WarmupHoldCosineLR, WarmupCosineLR
from .epoch_stats import EpochStatistics
from .batchnorm_calibration import recalibrated_batchnorm


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TEACHER_RESUME_REQUIRED_KEYS = {
    "epoch", "seed", "target_epochs", "config_sha256", "code_tree_sha256",
    "data_sha256", "label_map_sha256", "model_state_dict", "optimizer_state_dict",
    "scheduler_state_dict", "scaler_state_dict", "rng_state", "best_value",
    "precision", "effective_precision",
}
TEACHER_READABLE_RESUME_REQUIRED_KEYS = {
    "epoch", "seed", "best_value", "model_state_dict", "optimizer_state_dict",
    "scheduler_state_dict",
}
TEACHER_BEST_REQUIRED_KEYS = {
    "epoch", "seed", "selection_metric", "selection_value", "val_metrics",
    "model_config", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
    "scaler_state_dict", "precision", "effective_precision",
}


def set_global_seed(seed, deterministic=True):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def capture_rng_state(train_generator=None):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    if train_generator is not None:
        state["train_loader_generator"] = train_generator.get_state()
    return state


def restore_rng_state(state, train_generator=None):
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError("断点缺少完整 RNG 状态，拒绝非等价续训。")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch.load(..., map_location="cuda") 会把 RNG 字节张量也搬到 CUDA，
    # 但 PyTorch 的 RNG 恢复接口只接受位于 CPU 的 ByteTensor。显式搬回
    # CPU，使旧断点和不同 map_location 的调用都能可靠恢复。
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("断点包含 CUDA RNG 状态，但当前 CUDA 不可用。")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])
    if train_generator is not None:
        if "train_loader_generator" not in state:
            raise ValueError("断点缺少训练加载器生成器状态，拒绝非等价续训。")
        train_generator.set_state(state["train_loader_generator"].cpu())


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_tree_sha256(project_root):
    project_root = Path(project_root).resolve()
    paths = []
    for relative_pattern in ("src/**/*.py", "scripts/**/*.py", "tools/**/*.py"):
        paths.extend(project_root.glob(relative_pattern))
    for file_name in ("pyproject.toml",):
        path = project_root / file_name
        if path.is_file():
            paths.append(path)

    digest = hashlib.sha256()
    for path in sorted(set(paths), key=lambda item: item.relative_to(project_root).as_posix()):
        relative_path = path.relative_to(project_root).as_posix().encode("utf-8")
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        with path.open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def resolve_path(config_dir, value):
    expanded = os.path.expandvars(str(value))
    if "${" in expanded:
        raise ValueError(f"路径中的环境变量未设置：{value}")
    path = Path(expanded).expanduser()
    return path.resolve() if path.is_absolute() else (config_dir / path).resolve()


def load_config(path):
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件不是字典结构：{path}")
    return config, path


def load_label_map(config, config_path):
    data_config = config.get("data", {})
    if not data_config.get("label_map"):
        raise ValueError("配置中缺少 data.label_map。")
    label_map_path = resolve_path(config_path.parent, data_config["label_map"])
    with label_map_path.open("r", encoding="utf-8") as label_file:
        label_map = json.load(label_file)
    if not isinstance(label_map, dict) or not label_map:
        raise ValueError("标签映射必须是非空字典。")
    normalized = {str(name): int(index) for name, index in label_map.items()}
    expected_indices = list(range(len(normalized)))
    actual_indices = sorted(normalized.values())
    if actual_indices != expected_indices:
        raise ValueError(
            f"标签编号必须连续覆盖 0 到 {len(normalized) - 1}，实际为 {actual_indices}。"
        )
    return normalized, label_map_path


def validate_experiment_config(config):
    for key in ("epochs", "batch_size"):
        if int(config.get(key, 0)) <= 0:
            raise ValueError(f"{key} 必须大于 0。")

    workers = int(config.get("num_workers", 0))
    if workers < 0:
        raise ValueError("num_workers 不能为负数。")
    if workers > 0 and int(config.get("prefetch_factor", 2)) <= 0:
        raise ValueError("prefetch_factor 必须大于 0。")
    if config.get("data", {}).get("archive") and workers != 0:
        raise ValueError(
            "单文件 NPZ 数据包会把所选划分解压到内存，num_workers 必须为 0，"
            "避免 Windows 工作进程复制整块数组。"
        )
    if config.get("precision", "float32") not in ("float32", "fp32", "amp", "bf16"):
        raise ValueError("precision 必须是 'float32'、'fp32'、'amp' 或 'bf16'。")

    loss_config = config.get("loss", {})
    for key in ("train_temperature", "eval_temperature"):
        if float(loss_config.get(key, 1.0)) <= 0:
            raise ValueError(f"loss.{key} 必须大于 0。")
    if float(loss_config.get("label_smoothing", 0.0)) not in (0.0,) and not (
        0.0 < float(loss_config["label_smoothing"]) < 1.0
    ):
        raise ValueError("loss.label_smoothing 必须位于 [0, 1) 区间。")
    if float(loss_config.get("confidence_penalty", 0.0)) < 0:
        raise ValueError("loss.confidence_penalty 不能为负数。")
    if float(loss_config.get("topk_gap_weight", 0.0)) < 0:
        raise ValueError("loss.topk_gap_weight 不能为负数。")
    if int(loss_config.get("topk_gap_k", 3)) < 2:
        raise ValueError("loss.topk_gap_k 必须至少为 2。")

    seeds = [int(seed) for seed in config.get("seeds", [])]
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds 中存在重复值。")


def effective_precision(device, requested_precision):
    if requested_precision not in ("float32", "fp32", "amp", "bf16"):
        raise ValueError("precision 必须是 'float32'、'fp32'、'amp' 或 'bf16'。")
    if requested_precision in ("float32", "fp32"):
        return "fp32"
    if device.type != "cuda":
        if requested_precision == "bf16":
            raise ValueError("precision=bf16 要求支持 BF16 的 CUDA 设备。")
        raise ValueError("precision=amp 只允许在 CUDA 设备上启用。")
    if requested_precision == "bf16":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(f"CUDA 设备 {device} 不支持 BF16 autocast。")
    return requested_precision


def verify_configured_data(config, config_path):
    data_config = config.get("data", {})
    if data_config.get("archive"):
        archive_path = resolve_path(config_path.parent, data_config["archive"])
        return verify_dataset_archive(
            archive_path=archive_path,
            expected_sha256=data_config.get("archive_sha256"),
        )
    if not data_config.get("manifest"):
        raise ValueError("正式实验必须通过 data.archive 或 data.manifest 固定数据划分。")
    manifest_path = resolve_path(config_path.parent, data_config["manifest"])
    dataset_root = resolve_path(config_path.parent, data_config["root"])
    return verify_manifest(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        verify_hashes=(
            not resume_readable_only()
            and bool(data_config.get("verify_hashes_before_run", False))
        ),
    )


def make_dataset(config, config_path, split, label_map):
    data_config = config.get("data", {})
    if data_config.get("archive"):
        archive_path = resolve_path(config_path.parent, data_config["archive"])
        dataset = InMemoryNpzSkeletonDataset(
            archive_path=archive_path,
            split=split,
            expected_frames=int(data_config.get("frames", 120)),
            num_points=int(data_config.get("num_points", 23)),
        )
        labels = dataset.labels
        if labels.size == 0:
            raise ValueError(f"数据包 {split} 划分为空。")
        if labels.min() < 0 or labels.max() >= len(label_map):
            raise ValueError(f"数据包 {split} 标签超出 label_map 范围。")
        return dataset
    if data_config.get("manifest"):
        dataset_root = resolve_path(config_path.parent, data_config["root"])
        manifest_path = resolve_path(config_path.parent, data_config["manifest"])
        return SkeletonDataset(
            data_dir=dataset_root,
            muscle_to_label=label_map,
            manifest_path=manifest_path,
            split=split,
            expected_frames=int(data_config.get("frames", 120)),
            num_points=int(data_config.get("num_points", 23)),
        )

    legacy_key = f"{split}_data"
    if legacy_key not in config:
        raise KeyError(f"配置中缺少 data.manifest 或 {legacy_key}。")
    return SkeletonDataset(
        data_dir=resolve_path(config_path.parent, config[legacy_key]),
        muscle_to_label=label_map,
        expected_frames=int(data_config.get("frames", 120)),
        num_points=int(data_config.get("num_points", 23)),
    )


def make_loader(dataset, config, seed, shuffle, device):
    workers = int(config.get("num_workers", 0))
    if getattr(dataset, "requires_single_process_loader", False) and workers != 0:
        raise ValueError("内存 NPZ 数据集只能使用 num_workers=0。")
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": int(config["batch_size"]),
        "shuffle": shuffle,
        "num_workers": workers,
        "persistent_workers": workers > 0,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    return DataLoader(**loader_kwargs)


def create_train_val_loaders(config, config_path, seed, device, label_map):
    train_dataset = make_dataset(config, config_path, "train", label_map)
    val_dataset = make_dataset(config, config_path, "val", label_map)
    return (
        make_loader(train_dataset, config, seed, True, device),
        make_loader(val_dataset, config, seed + 1, False, device),
    )


def create_test_loader(config, config_path, seed, device, label_map):
    test_dataset = make_dataset(config, config_path, "test", label_map)
    return make_loader(test_dataset, config, seed + 2, False, device)


def create_model_and_training_objects(config, device, num_classes):
    model_config = config.get("model", {})
    model = MyoFormerTeacher(
        num_classes=num_classes,
        num_points=int(model_config.get("num_points", 23)),
        in_channels=int(model_config.get("in_channels", 3)),
        dropout=float(model_config.get("dropout", 0.05)),
        use_spatial_attention=bool(model_config.get("use_spatial_attention", True)),
        use_symmetry_attention=bool(model_config.get("use_symmetry_attention", True)),
        use_scale_attention=bool(model_config.get("use_scale_attention", True)),
        use_joint_spatial_stream=bool(model_config.get("use_joint_spatial_stream", True)),
        use_joint_temporal_stream=bool(model_config.get("use_joint_temporal_stream", True)),
        use_bone_vector_stream=bool(model_config.get("use_bone_vector_stream", True)),
        width_ratio=float(model_config.get("width_ratio", 1.0)),
        spatial_attention_config=model_config.get("spatial_attention", {}),
        symmetry_attention_config=model_config.get("symmetry_attention", {}),
        skeleton_edges=model_config.get("skeleton_edges"),
        symmetry_pairs=model_config.get("symmetry_pairs"),
        root_index=int(model_config.get("root_index", 0)),
        input_scheme=model_config.get("input_scheme", "X1/X2/X2"),
        scale_attention_config=model_config.get("scale_attention", {}),
        head_config=model_config.get("head", {}),
    ).to(device).float()

    loss_config = config.get("loss", {})
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(loss_config.get("label_smoothing", 0.0))
    )

    optimizer_config = config.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config.get("peak_lr", 5e-4)),
        weight_decay=float(optimizer_config.get("weight_decay", 1e-4)),
        betas=tuple(optimizer_config.get("betas", [0.9, 0.999])),
        eps=float(optimizer_config.get("eps", 1e-8)),
    )

    scheduler_config = config.get("scheduler", {})
    if scheduler_config.get("type") == "warmup_cosine":
        scheduler = WarmupCosineLR(
            optimizer, total_epochs=int(config["epochs"]),
            peak_lr=float(optimizer_config.get("peak_lr", 1e-3)),
            warmup_epochs=int(scheduler_config.get("warmup_epochs", 30)),
            warmup_init_lr=float(scheduler_config.get("warmup_init_lr", 5e-5)),
            warmup_gamma=float(scheduler_config.get("warmup_gamma", 0.5)),
            min_lr=float(scheduler_config.get("min_lr", 1e-6)),
        )
    elif scheduler_config.get("type", "two_stage") == "warmup_hold_cosine":
        scheduler = WarmupHoldCosineLR(
            optimizer=optimizer, total_epochs=int(config["epochs"]),
            peak_lr=float(optimizer_config.get("peak_lr", 1e-3)),
            warmup_epochs=int(scheduler_config.get("warmup_epochs", 30)),
            hold_epochs=int(scheduler_config.get("hold_epochs", 60)),
            warmup_init_lr=float(scheduler_config.get("warmup_init_lr", 5e-5)),
            warmup_gamma=float(scheduler_config.get("warmup_gamma", 0.5)),
            min_lr=float(scheduler_config.get("min_lr", 1e-6)),
        )
    else:
        scheduler = TwoStageLR(
            optimizer=optimizer,
            total_epochs=int(config["epochs"]),
            peak_lr=float(optimizer_config.get("peak_lr", 5e-4)),
            warmup_ratio=float(scheduler_config.get("warmup_ratio", 0.1)),
            warmup_init_lr=float(scheduler_config.get("warmup_init_lr", 1e-4)),
            warmup_gamma=float(scheduler_config.get("warmup_gamma", 1.0)),
            decay_gamma=float(scheduler_config.get("decay_gamma", 2.0)),
        )
    return model, criterion, optimizer, scheduler


def calculate_metrics(targets, predictions, top3_hits):
    sample_count = len(targets)
    accuracy = float(np.mean(np.asarray(targets) == np.asarray(predictions)))
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, predictions, average="weighted", zero_division=0)),
        "top3_accuracy": float(top3_hits / sample_count),
        "sample_count": int(sample_count),
    }


def create_progress_bar(loader, description):
    """Delegate terminal ownership to the controller during parallel training."""
    from .progress import create_progress_bar as progress_bar
    return progress_bar(loader, description)


def format_teacher_epoch_result(seed, epoch, learning_rate, train_metrics, val_metrics):
    """生成控制台和实验日志共用的教师轮次汇总。"""
    return (
        f"轮次结果 教师 种子={seed} 轮次={epoch:03d} "
        f"学习率={learning_rate:.8f} "
        f"训练损失={train_metrics['loss']:.4f} "
        f"训练准确率={train_metrics['accuracy']:.4f} "
        f"验证损失={val_metrics['loss']:.4f} "
        f"验证准确率={val_metrics['accuracy']:.4f} "
        f"验证平衡准确率={val_metrics['balanced_accuracy']:.4f} "
        f"验证宏平均F1={val_metrics['macro_f1']:.4f}"
    )


def run_one_epoch(
    model,
    loader,
    criterion,
    device,
    split,
    optimizer=None,
    loss_config=None,
    scaler=None,
    mixed_precision=False,
    autocast_dtype=torch.float16,
    progress_description=None,
):
    is_train = optimizer is not None
    if mixed_precision and device.type != "cuda":
        raise ValueError("混合精度只允许在 CUDA 设备上启用。")
    if is_train and mixed_precision and scaler is None:
        raise ValueError("混合精度训练必须提供 GradScaler。")
    model.train(is_train)
    loss_config = loss_config or {}
    train_temperature = float(loss_config.get("train_temperature", 1.0))
    eval_temperature = float(loss_config.get("eval_temperature", 1.0))
    confidence_penalty = float(loss_config.get("confidence_penalty", 0.0))
    topk_gap_weight = float(loss_config.get("topk_gap_weight", 0.0))
    topk_gap_k = int(loss_config.get("topk_gap_k", 3))

    total_samples = 0
    statistics = EpochStatistics(keep_probabilities=split == "test")
    last_progress_update = 0.0
    trainable_parameters = tuple(model.parameters()) if is_train else ()

    # tqdm 直接写继承的 stderr，仅在 PyCharm 控制台原地刷新；矩阵主控
    # 只读取 stdout，因此批次进度不会进入 matrix.log。
    progress = create_progress_bar(loader, progress_description or split)
    context = torch.enable_grad() if is_train else torch.inference_mode()
    with context:
        for joints, bones, targets in progress:
            joints = joints.to(device, non_blocking=device.type == "cuda")
            bones = bones.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, dtype=torch.long, non_blocking=device.type == "cuda")

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=bool(mixed_precision),
            ):
                logits = model(joints, bones)
                # 旧入口所有划分的损失均用训练温度；评估温度只校准输出概率。
                logits_for_loss = logits / max(train_temperature, 1e-6)
                classification_loss = criterion(logits_for_loss, targets)
                if confidence_penalty or (topk_gap_weight and topk_gap_k >= 2):
                    probabilities_for_loss = F.softmax(logits_for_loss, dim=1)
                regularization_loss = logits.new_zeros(())

                if confidence_penalty:
                    negative_entropy = torch.sum(
                        probabilities_for_loss * torch.log(probabilities_for_loss + 1e-12),
                        dim=1,
                    ).mean()
                    regularization_loss = regularization_loss + confidence_penalty * negative_entropy

                if topk_gap_weight and topk_gap_k >= 2:
                    top_values = torch.topk(
                        probabilities_for_loss,
                        k=min(topk_gap_k, probabilities_for_loss.size(1)),
                        dim=1,
                    ).values
                    regularization_loss = regularization_loss + topk_gap_weight * (
                        top_values[:, 0] - top_values.mean(dim=1)
                    ).mean()

                loss = classification_loss + regularization_loss
            if is_train:
                scaler_enabled = scaler is not None and scaler.is_enabled()
                scaler.scale(loss).backward() if scaler_enabled else loss.backward()
                if scaler_enabled:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
                if scaler_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

            batch_size = targets.size(0)
            total_samples += batch_size

            probabilities = F.softmax(logits.detach() / max(eval_temperature, 1e-6), dim=1)
            predictions = logits.argmax(dim=1)
            top3 = torch.topk(probabilities, k=min(3, probabilities.size(1)), dim=1).indices
            statistics.add(
                (loss, classification_loss, regularization_loss), targets,
                predictions, top3.eq(targets.unsqueeze(1)).any(dim=1), probabilities,
            )
            now = time.monotonic()
            if now - last_progress_update >= 1.0:
                progress.set_postfix(loss=f"{loss.item():.4f}")
                last_progress_update = now

    totals, targets_all, predictions_all, top3_hits, probabilities_array = statistics.finish(model.num_classes)
    total_loss, total_classification_loss, total_regularization_loss = totals
    metrics = calculate_metrics(targets_all, predictions_all, top3_hits)
    metrics.update(
        {
            "loss": total_loss / total_samples,
            "classification_loss": total_classification_loss / total_samples,
            "regularization_loss": total_regularization_loss / total_samples,
        }
    )
    return metrics, np.asarray(targets_all), np.asarray(predictions_all), probabilities_array


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)


def save_checkpoint_atomic(path, value):
    path = Path(path)
    if path.name.startswith("last_teacher.pth"):
        required_keys = (
            TEACHER_READABLE_RESUME_REQUIRED_KEYS
            if resume_readable_only() else TEACHER_RESUME_REQUIRED_KEYS
        )
    elif path.name.startswith("best_teacher.pth"):
        required_keys = TEACHER_BEST_REQUIRED_KEYS
    else:
        required_keys = {"epoch"}
    return _durable_save_checkpoint_atomic(path, value, required_keys=required_keys)


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    os.replace(temporary_path, path)


def _powershell_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def build_teacher_resume_command(config_path, run_dir, data_archive_path, device):
    # 矩阵生成的配置位于 configs/matrix_resolved；不能再从配置层级反推项目根目录。
    project_root = PROJECT_ROOT
    script_path = project_root / "scripts" / "train_teacher.py"
    return (
        f"$env:LITE_MYONET_DATA_ARCHIVE={_powershell_quote(data_archive_path)}; "
        f"$env:PYTHONPATH={_powershell_quote(project_root / 'src')}; "
        f"& {_powershell_quote(sys.executable)} {_powershell_quote(script_path)} "
        f"--config {_powershell_quote(config_path)} --device {device.type} "
        f"--resume {_powershell_quote(run_dir)}"
    )


def render_teacher_run_readme(status):
    latest = status.get("latest_val_metrics") or {}
    test = status.get("test_metrics") or {}
    lines = [
        "# Lite-MyoNet 教师训练运行说明",
        "",
        "本目录是一项可独立审计和断点续训的正式教师训练运行。教师仅训练一个固定随机种子，随后作为学生蒸馏源。",
        "",
        "## 运行状态",
        "",
        f"- 状态：`{status['status']}`",
        f"- 随机种子：`{status['seed']}`",
        f"- 当前轮次：`{status['current_epoch']} / {status['target_epochs']}`",
        f"- 最佳验证指标：`{status.get('selection_metric')}` = `{status.get('best_value')}`",
        f"- 设备：`{status['device']}`",
        f"- 可训练参数：`{status['trainable_parameters']}`",
        "",
        "## 可复现身份",
        "",
        f"- 配置：`{status['config_path']}`",
        f"- 配置 SHA-256（安全哈希算法 256 位）：`{status['config_sha256']}`",
        f"- 代码树 SHA-256（安全哈希算法 256 位）：`{status['code_tree_sha256']}`",
        f"- 数据 SHA-256（安全哈希算法 256 位）：`{status['data_sha256']}`",
        f"- 标签映射 SHA-256（安全哈希算法 256 位）：`{status['label_map_sha256']}`",
        "",
        "## 断点续训",
        "",
        "```powershell",
        status["resume_command"],
        "```",
        "",
        "`last_teacher.pth` 保存最近一轮及随机数状态；`best_teacher.pth` 只由验证集指标选择。训练期不读取测试集，全部训练完成后测试一次。",
        "",
        "## 最近验证指标",
        "",
    ]
    if latest:
        lines.extend(f"- {key}: `{value}`" for key, value in sorted(latest.items()))
    else:
        lines.append("- 尚未完成第一轮验证。")
    lines.extend(["", "## 最终测试指标", ""])
    if test:
        lines.extend(f"- {key}: `{value}`" for key, value in sorted(test.items()))
    else:
        lines.append("- 尚未执行测试；测试只会在验证集选模完成后执行一次。")
    lines.extend([
        "",
        "## 主要产物",
        "",
        "- `resolved_config.yaml`：运行时完整配置。",
        "- `provenance.json`：软件、硬件、数据和代码来源。",
        "- `run_status.json`：机器可读进度与恢复信息。",
        "- `history.csv`：逐轮训练与验证指标。",
        "- `last_teacher.pth`：严格断点续训权重。",
        "- `best_teacher.pth`：验证集最优教师权重。",
        "- `test_results.json` 与测试预测数组：最终一次测试的完整输出。",
        "",
    ])
    return "\n".join(lines)


def write_teacher_run_status(run_dir, status):
    write_json_atomic(Path(run_dir) / "run_status.json", status)
    readme_path = Path(run_dir) / "README.md"
    temporary_path = readme_path.with_name(".README.md.tmp")
    temporary_path.write_text(render_teacher_run_readme(status), encoding="utf-8")
    os.replace(temporary_path, readme_path)


def aggregate_seed_results(results):
    if not results:
        raise ValueError("无法汇总空的实验结果。")

    summary = {
        "num_seeds": len(results),
        "seeds": [int(result["seed"]) for result in results],
        "runs": [result["run_dir"] for result in results],
    }
    for split_name in ("val_metrics", "test_metrics"):
        split_summary = {}
        metric_names = sorted(results[0][split_name])
        for metric_name in metric_names:
            values = [float(result[split_name][metric_name]) for result in results]
            split_summary[metric_name] = {
                "values": values,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        summary[split_name] = split_summary
    return summary


def check_experiment_setup(config, config_path, device, label_map):
    datasets = {
        split: make_dataset(config, config_path, split, label_map)
        for split in ("train", "val", "test")
    }
    model, criterion, _, _ = create_model_and_training_objects(
        config, device, len(label_map)
    )
    model.eval()

    joints, bones, target = datasets["train"][0]
    with torch.inference_mode():
        logits = model(
            joints.unsqueeze(0).to(device),
            bones.unsqueeze(0).to(device),
        )
        loss = criterion(logits, torch.tensor([target], device=device))

    report = {
        "device": str(device),
        "dataset_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "trainable_parameters": model.count_params(),
        "logits_shape": list(logits.shape),
        "finite_logits": bool(torch.isfinite(logits).all().item()),
        "finite_loss": bool(torch.isfinite(loss).item()),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def prepare_run_directory(config, config_path, seed):
    output_root = resolve_path(config_path.parent, config.get("output_root", "runs"))
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"teacher_seed_{seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def resolve_resume_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        last_path = path / "last_teacher.pth"
        candidates = (last_path, last_path.with_name(f"{last_path.name}.bak"))
    else:
        if path.name not in {"last_teacher.pth", "last_teacher.pth.bak"}:
            if not path.is_file():
                raise FileNotFoundError(f"续训断点不存在：{path}")
            raise ValueError("教师续训只接受 last_teacher.pth 或 last_teacher.pth.bak。")
        last_path = path.parent / "last_teacher.pth"
        candidates = (last_path, last_path.with_name(f"{last_path.name}.bak"))
    resolved, _ = load_latest_checkpoint(
        candidates,
        required_keys=(
            TEACHER_READABLE_RESUME_REQUIRED_KEYS
            if resume_readable_only() else TEACHER_RESUME_REQUIRED_KEYS
        ),
        map_location="cpu",
    )
    return resolved


def _backup_teacher_artifact(path):
    path = Path(path)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = path.with_name(f"{path.name}.pre_resume-{stamp}-{uuid.uuid4().hex[:8]}.bak")
    with path.open("rb") as source, backup.open("xb") as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)
        destination.flush()
        os.fsync(destination.fileno())
    return backup


def _reconcile_teacher_resume_progress(run_dir, checkpoint, run_status):
    checkpoint_epoch = int(checkpoint["epoch"])
    history_path = Path(run_dir) / "history.csv"
    if not history_path.is_file():
        raise FileNotFoundError(f"续训目录缺少 history.csv：{history_path}")
    with history_path.open("r", newline="", encoding="utf-8-sig") as history_file:
        reader = csv.DictReader(history_file)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or "epoch" not in fieldnames:
        raise ValueError(f"无法识别训练历史格式：{history_path}")
    try:
        epochs = [int(row["epoch"]) for row in rows]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"训练历史包含无效 epoch：{history_path}") from error
    expected_epochs = list(range(1, checkpoint_epoch + 1))
    kept = [(row, epoch) for row, epoch in zip(rows, epochs) if epoch <= checkpoint_epoch]
    if [epoch for _, epoch in kept] != expected_epochs:
        raise ValueError(
            f"history.csv 的 epoch 与选中 checkpoint epoch={checkpoint_epoch} 不连续；"
            "拒绝跳过或伪造训练日志。"
        )
    if len(kept) != len(rows):
        _backup_teacher_artifact(history_path)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{history_path.name}.", suffix=".tmp", dir=str(history_path.parent)
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", newline="", encoding="utf-8-sig") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(row for row, _ in kept)
                output_file.flush()
                os.fsync(output_file.fileno())
            os.replace(temporary, history_path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    status_epoch = int(run_status.get("current_epoch", 0))
    if status_epoch > checkpoint_epoch:
        _backup_teacher_artifact(Path(run_dir) / "run_status.json")
        run_status["current_epoch"] = checkpoint_epoch


def verify_resume_code_compatibility(run_dir, checkpoint_path, saved_code_sha256, current_code_sha256):
    """验证一次明确登记、且绑定到具体断点文件的非数值兼容迁移。"""
    if saved_code_sha256 == current_code_sha256:
        return None

    compatibility_path = Path(run_dir) / "resume_compatibility.json"
    if not compatibility_path.is_file():
        raise ValueError("续训断点的代码树哈希与当前代码不一致。")
    try:
        compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"续训兼容记录无法读取：{compatibility_path}。") from error

    expected = {
        "schema_version": 1,
        "checkpoint_name": Path(checkpoint_path).name,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "saved_code_tree_sha256": saved_code_sha256,
        "current_code_tree_sha256": current_code_sha256,
    }
    mismatches = {
        key: {"expected": value, "actual": compatibility.get(key)}
        for key, value in expected.items()
        if compatibility.get(key) != value
    }
    if mismatches:
        raise ValueError(f"续训兼容记录与当前断点或代码不一致：{mismatches}")
    if not compatibility.get("reason"):
        raise ValueError("续训兼容记录缺少 reason，拒绝跨代码哈希续训。")
    return {
        "record_path": str(compatibility_path),
        "saved_code_tree_sha256": saved_code_sha256,
        "current_code_tree_sha256": current_code_sha256,
        "reason": str(compatibility["reason"]),
    }


def write_run_provenance(
    run_dir,
    config,
    config_path,
    seed,
    device,
    model,
    data_verification_report,
    label_map_path,
):
    data_config = config.get("data", {})
    manifest_path = (
        resolve_path(config_path.parent, data_config["manifest"])
        if data_config.get("manifest")
        else None
    )
    archive_path = (
        resolve_path(config_path.parent, data_config["archive"])
        if data_config.get("archive")
        else None
    )
    provenance = {
        "seed": seed,
        "device": str(device),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "pyyaml": yaml.__version__,
        "tqdm": tqdm_package.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "trainable_parameters": model.count_params(),
        "code_tree_sha256": code_tree_sha256(PROJECT_ROOT),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "label_map_path": str(label_map_path),
        "label_map_sha256": file_sha256(label_map_path),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest_sha256": file_sha256(manifest_path) if manifest_path else None,
        "archive_path": str(archive_path) if archive_path else None,
        "archive_sha256": file_sha256(archive_path) if archive_path else None,
        "data_verification": data_verification_report,
        "precision": config.get("precision", "float32"),
        "effective_precision": effective_precision(device, config.get("precision", "float32")),
    }
    write_json(run_dir / "provenance.json", provenance)
    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file, allow_unicode=True, sort_keys=False)


def train_one_seed(
    config,
    config_path,
    seed,
    device,
    data_verification_report,
    label_map,
    label_map_path,
    resume_path=None,
):
    set_global_seed(seed, deterministic=bool(config.get("deterministic", True)))
    resume_checkpoint = resolve_resume_checkpoint(resume_path) if resume_path else None
    run_dir = resume_checkpoint.parent if resume_checkpoint else prepare_run_directory(
        config, config_path, seed
    )
    completed_results_path = run_dir / "test_results.json"
    if resume_checkpoint and completed_results_path.is_file():
        with completed_results_path.open("r", encoding="utf-8") as stream:
            completed = json.load(stream)
        if int(completed.get("seed", -1)) != int(seed):
            raise ValueError("已完成结果的 seed 与请求续训的 seed 不一致。")
        print(f"seed={seed} 已完成测试，续训命令不会重复读取测试集：{run_dir}")
        return completed
    train_loader, val_loader = create_train_val_loaders(
        config, config_path, seed, device, label_map
    )
    calibration_config = config.get("evaluation", {}).get("batch_norm_recalibration", {})
    calibration_loader = None
    if calibration_config.get("enabled", False):
        calibration_loader = DataLoader(
            train_loader.dataset,
            batch_size=int(calibration_config.get("batch_size", 4)),
            shuffle=True, num_workers=0, pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(seed),
        )

    def calibration_context(description, for_selection=False):
        if calibration_loader is None or (for_selection and calibration_config.get("mode") == "final_only"):
            return nullcontext(None)
        return recalibrated_batchnorm(model, calibration_loader, device, seed, description)

    model, criterion, optimizer, scheduler = create_model_and_training_objects(
        config, device, len(label_map)
    )
    requested_precision = config.get("precision", "float32")
    actual_precision = effective_precision(device, requested_precision)
    use_amp = actual_precision == "amp"
    mixed_precision = actual_precision in {"amp", "bf16"}
    autocast_dtype = torch.bfloat16 if actual_precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if not resume_checkpoint:
        write_run_provenance(
            run_dir,
            config,
            config_path,
            seed,
            device,
            model,
            data_verification_report,
            label_map_path,
        )

    selection_metric = config.get("selection_metric", "macro_f1")
    allowed_metrics = {"accuracy", "balanced_accuracy", "macro_f1"}
    if selection_metric not in allowed_metrics:
        raise ValueError(f"selection_metric 必须属于 {sorted(allowed_metrics)}。")

    best_value = float("-inf")
    best_path = run_dir / "best_teacher.pth"
    best_backup_path = best_path.with_name(f"{best_path.name}.bak")
    last_path = run_dir / "last_teacher.pth"
    history_path = run_dir / "history.csv"
    start_epoch = 1
    best_val_metrics = {}
    resume_compatibility = None
    if resume_checkpoint:
        readable_only = resume_readable_only()
        # 先在 CPU 上反序列化，避免 CPU/CUDA RNG 状态被 map_location 一并
        # 移到显卡；load_state_dict 会把模型和优化器状态放到各自参数设备。
        checkpoint = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        required_keys = (
            TEACHER_READABLE_RESUME_REQUIRED_KEYS
            if readable_only else TEACHER_RESUME_REQUIRED_KEYS
        )
        missing_keys = (
            sorted(required_keys - set(checkpoint))
            if isinstance(checkpoint, dict)
            else sorted(required_keys)
        )
        if missing_keys:
            raise ValueError(f"续训断点缺少键：{missing_keys}")
        if int(checkpoint["seed"]) != int(seed):
            raise ValueError("续训断点的 seed 与命令不一致。")
        if int(checkpoint["epoch"]) < 0 or int(checkpoint["epoch"]) > int(config["epochs"]):
            raise ValueError(
                f"续训断点 epoch={checkpoint['epoch']} 超出可续训范围 "
                f"[0, {int(config['epochs'])}]。"
            )
        if not readable_only:
            if int(checkpoint["target_epochs"]) != int(config["epochs"]):
                raise ValueError("续训断点的总轮数与当前配置不一致。")
            if checkpoint["precision"] != requested_precision:
                raise ValueError("precision 配置与 checkpoint 不一致，拒绝续训。")
            if checkpoint["effective_precision"] != actual_precision:
                raise ValueError("当前设备的有效精度与 checkpoint 不一致，拒绝续训。")
            if checkpoint["config_sha256"] != file_sha256(config_path):
                raise ValueError("续训断点的配置哈希与当前配置不一致。")
            current_code_sha256 = code_tree_sha256(PROJECT_ROOT)
            resume_compatibility = verify_resume_code_compatibility(
                run_dir,
                resume_checkpoint,
                checkpoint["code_tree_sha256"],
                current_code_sha256,
            )
            current_data_sha256 = data_verification_report.get("archive_sha256")
            if current_data_sha256 is None:
                current_data_sha256 = data_verification_report.get("manifest_sha256")
            if checkpoint["data_sha256"] != current_data_sha256:
                raise ValueError("续训断点的数据哈希与当前数据不一致。")
            if checkpoint["label_map_sha256"] != file_sha256(label_map_path):
                raise ValueError("续训断点的标签映射哈希与当前文件不一致。")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if not readable_only:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            train_generator = getattr(train_loader, "generator", None)
            restore_rng_state(checkpoint["rng_state"], train_generator=train_generator)
        elif "scaler_state_dict" in checkpoint:
            try:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            except (RuntimeError, ValueError, KeyError):
                pass
        if readable_only and 'rng_state' in checkpoint:
            restore_rng_state(checkpoint['rng_state'], train_generator=getattr(train_loader,'generator',None))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_value = float(checkpoint["best_value"])
        best_val_metrics = checkpoint.get("best_val_metrics", {})
        if not history_path.is_file():
            raise FileNotFoundError(f"续训目录缺少 history.csv：{history_path}")
    data_sha256 = data_verification_report.get(
        "archive_sha256", data_verification_report.get("manifest_sha256")
    )
    data_archive_path = data_verification_report.get(
        "archive_path", data_verification_report.get("manifest_path", "")
    )
    status_path = run_dir / "run_status.json"
    if resume_checkpoint and status_path.is_file():
        with status_path.open("r", encoding="utf-8") as stream:
            run_status = json.load(stream)
        _reconcile_teacher_resume_progress(run_dir, checkpoint, run_status)
        run_status.update({
            "status": "running",
            "resumed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "current_epoch": start_epoch - 1,
            "best_value": best_value,
            "code_tree_sha256": code_tree_sha256(PROJECT_ROOT),
        })
        if resume_compatibility is not None:
            run_status.setdefault("resume_compatibility_events", []).append(
                {
                    **resume_compatibility,
                    "accepted_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                }
            )
    else:
        run_status = {
            "schema_version": 1,
            "run_kind": "teacher",
            "status": "initialized",
            "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "seed": int(seed),
            "current_epoch": 0,
            "target_epochs": int(config["epochs"]),
            "selection_metric": selection_metric,
            "best_value": None,
            "device": str(device),
            "precision": requested_precision,
            "effective_precision": actual_precision,
            "trainable_parameters": model.count_params(),
            "run_dir": str(run_dir),
            "config_path": str(config_path),
            "config_sha256": file_sha256(config_path),
            "code_tree_sha256": code_tree_sha256(PROJECT_ROOT),
            "data_sha256": data_sha256,
            "label_map_sha256": file_sha256(label_map_path),
            "resume_checkpoint": str(last_path),
            "best_checkpoint": str(best_path),
            "resume_command": build_teacher_resume_command(
                config_path, run_dir, data_archive_path, device
            ),
            "latest_train_metrics": None,
            "latest_val_metrics": None,
            "best_val_metrics": None,
            "test_metrics": None,
        }
    write_teacher_run_status(run_dir, run_status)
    fieldnames = [
        "epoch",
        "learning_rate",
        *[f"train_{name}" for name in (
            "loss", "classification_loss", "regularization_loss", "accuracy",
            "balanced_accuracy", "macro_f1", "weighted_f1", "top3_accuracy",
        )],
        *[f"val_{name}" for name in (
            "loss", "classification_loss", "regularization_loss", "accuracy",
            "balanced_accuracy", "macro_f1", "weighted_f1", "top3_accuracy",
        )],
    ]

    # Provenance is constant during this run; do not rescan the source tree every epoch.
    run_config_sha = file_sha256(config_path)
    run_code_sha = code_tree_sha256(PROJECT_ROOT)
    run_label_sha = file_sha256(label_map_path)
    history_mode = "a" if resume_checkpoint else "w"
    with history_path.open(history_mode, newline="", encoding="utf-8-sig") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=fieldnames)
        if not resume_checkpoint:
            writer.writeheader()
            history_file.flush()
            train_generator = getattr(train_loader, "generator", None)
            save_checkpoint_atomic(
                last_path,
                {
                    "status": "initialized",
                    "epoch": 0,
                    "seed": seed,
                    "target_epochs": int(config["epochs"]),
                    "selection_metric": selection_metric,
                    "best_value": best_value,
                    "best_val_metrics": best_val_metrics,
                    "config_sha256": run_config_sha,
                    "code_tree_sha256": run_code_sha,
                    "data_sha256": data_verification_report.get(
                        "archive_sha256", data_verification_report.get("manifest_sha256")
                    ),
                    "label_map_sha256": run_label_sha,
                    "model_config": config.get("model", {}),
                    "loss_config": config.get("loss", {}),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "precision": requested_precision,
                    "effective_precision": actual_precision,
                    "rng_state": capture_rng_state(train_generator=train_generator),
                },
            )

        for epoch in range(start_epoch, int(config["epochs"]) + 1):
            learning_rate = optimizer.param_groups[0]["lr"]
            train_metrics, _, _, _ = run_one_epoch(
                model,
                train_loader,
                criterion,
                device,
                split="train",
                optimizer=optimizer,
                loss_config=config.get("loss", {}),
                scaler=scaler,
                mixed_precision=mixed_precision,
                autocast_dtype=autocast_dtype,
                progress_description=f"教师 种子{seed} 第{epoch}/{config['epochs']}轮 训练",
            )
            with calibration_context(f"教师 种子{seed} 第{epoch}轮 训练集统计校准", for_selection=True) as calibration:
                val_metrics, _, _, _ = run_one_epoch(
                    model,
                    val_loader,
                    criterion,
                    device,
                    split="val",
                    optimizer=None,
                    loss_config=config.get("loss", {}),
                    scaler=None,
                    mixed_precision=mixed_precision,
                    autocast_dtype=autocast_dtype,
                    progress_description=f"教师 种子{seed} 第{epoch}/{config['epochs']}轮 验证",
                )

                row = {"epoch": epoch, "learning_rate": learning_rate}
                row.update({f"train_{key}": value for key, value in train_metrics.items() if key != "sample_count"})
                row.update({f"val_{key}": value for key, value in val_metrics.items() if key != "sample_count"})
                writer.writerow(row)
                history_file.flush()

                current_value = val_metrics[selection_metric]
                if current_value > best_value:
                    best_value = current_value
                    best_val_metrics = val_metrics
                    save_checkpoint_atomic(
                        best_path,
                        {
                            "epoch": epoch,
                            "seed": seed,
                            "selection_metric": selection_metric,
                            "selection_value": current_value,
                            "val_metrics": val_metrics,
                            "model_config": config.get("model", {}),
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "scaler_state_dict": scaler.state_dict(),
                            "precision": requested_precision,
                            "effective_precision": actual_precision,
                            "batch_norm_recalibration": calibration,
                        },
                    )
                if config.get("save_epoch_weights", False):
                    from .epoch_weights import save_epoch_weights
                    save_epoch_weights(run_dir, epoch, {
                        "model_state_dict": model.state_dict(), "model_config": config.get("model", {}),
                        "seed": int(seed), "learning_rate": learning_rate,
                        "train_metrics": train_metrics, "val_metrics": val_metrics,
                        "batch_norm_recalibration": calibration,
                        "selection_metric": selection_metric,
                    })
            # Evaluation buffers have been restored; last_teacher remains a training checkpoint.
            scheduler.step()
            train_generator = getattr(train_loader, "generator", None)
            save_checkpoint_atomic(
                last_path,
                {
                    "status": "training",
                    "epoch": epoch,
                    "seed": seed,
                    "target_epochs": int(config["epochs"]),
                    "selection_metric": selection_metric,
                    "best_value": best_value,
                    "best_val_metrics": best_val_metrics,
                    "config_sha256": run_config_sha,
                    "code_tree_sha256": run_code_sha,
                    "data_sha256": data_verification_report.get(
                        "archive_sha256", data_verification_report.get("manifest_sha256")
                    ),
                    "label_map_sha256": run_label_sha,
                    "model_config": config.get("model", {}),
                    "loss_config": config.get("loss", {}),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "precision": requested_precision,
                    "effective_precision": actual_precision,
                    "rng_state": capture_rng_state(train_generator=train_generator),
                },
            )
            run_status.update({
                "status": "running",
                "current_epoch": epoch,
                "best_value": best_value,
                "latest_train_metrics": train_metrics,
                "latest_val_metrics": val_metrics,
                "best_val_metrics": best_val_metrics,
                "precision": requested_precision,
                "effective_precision": actual_precision,
                "latest_batch_norm_recalibration": calibration,
                "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            })
            write_teacher_run_status(run_dir, run_status)

            print(
                format_teacher_epoch_result(
                    seed, epoch, learning_rate, train_metrics, val_metrics
                ),
                flush=True,
            )

    # 只有验证集选模结束后才构造并读取测试集。
    selected_best_path, selected_best_cpu = load_latest_checkpoint(
        (best_path, best_backup_path),
        required_keys=TEACHER_BEST_REQUIRED_KEYS,
        map_location="cpu",
    )
    if (
        int(selected_best_cpu["seed"]) != int(seed)
        or selected_best_cpu["selection_metric"] != selection_metric
        or float(selected_best_cpu["selection_value"]) != float(best_value)
    ):
        raise ValueError(
            "可读的最佳教师权重与续训状态记录的 best epoch/指标不一致；拒绝替换测试模型。"
        )
    run_status["best_checkpoint"] = str(selected_best_path)
    checkpoint = torch.load(selected_best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    selected_val_metrics = checkpoint["val_metrics"]
    calibrated_val_metrics = selected_val_metrics
    with calibration_context(f"教师 种子{seed} 最佳权重训练集统计校准") as test_calibration:
        if test_calibration is not None:
            calibrated_val_metrics, _, _, _ = run_one_epoch(
                model, val_loader, criterion, device, split="val", optimizer=None,
                loss_config=config.get("loss", {}), mixed_precision=mixed_precision,
                autocast_dtype=autocast_dtype, progress_description=f"教师 种子{seed} 最佳权重校准后验证",
            )
            checkpoint["model_state_dict"] = model.state_dict()
            checkpoint["batch_norm_recalibration"] = test_calibration
            checkpoint["calibrated_val_metrics"] = calibrated_val_metrics
            # Cache generation must load exactly the calibrated model used for the test.
            save_checkpoint_atomic(best_path, checkpoint)
            run_status["best_checkpoint"] = str(best_path)
        test_loader = create_test_loader(config, config_path, seed, device, label_map)
        test_metrics, targets, predictions, probabilities = run_one_epoch(
            model,
            test_loader,
            criterion,
            device,
            split="test",
            optimizer=None,
            loss_config=config.get("loss", {}),
            scaler=None,
            mixed_precision=mixed_precision,
            autocast_dtype=autocast_dtype,
            progress_description=f"教师 种子{seed} 测试",
        )
    del train_loader, val_loader, calibration_loader
    gc.collect()

    class_names = [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]
    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    result = {
        "seed": seed,
        "run_dir": str(run_dir),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "selection_metric": checkpoint["selection_metric"],
        "selection_value": float(checkpoint["selection_value"]),
        "val_metrics": calibrated_val_metrics,
        "selection_val_metrics": selected_val_metrics,
        "calibrated_val_metrics": calibrated_val_metrics,
        "epoch_selection_statistics": "training_running_bn" if calibration_config.get("mode") == "final_only" else "recalibrated_bn",
        "test_metrics": test_metrics,
        "classification_report": report,
        "batch_norm_recalibration": test_calibration,
    }
    write_json(run_dir / "test_results.json", result)
    np.save(run_dir / "test_targets.npy", targets)
    np.save(run_dir / "test_predictions.npy", predictions)
    np.save(run_dir / "test_probabilities.npy", probabilities)
    last_checkpoint_path, _ = load_latest_checkpoint(
        (last_path, last_path.with_name(f"{last_path.name}.bak")),
        required_keys=(
            TEACHER_READABLE_RESUME_REQUIRED_KEYS
            if resume_readable_only() else TEACHER_RESUME_REQUIRED_KEYS
        ),
        map_location="cpu",
    )
    final_state = torch.load(last_checkpoint_path, map_location="cpu", weights_only=False)
    final_state["status"] = "complete"
    final_state["test_results"] = str(completed_results_path)
    save_checkpoint_atomic(last_path, final_state)
    run_status.update({
        "status": "completed",
        "current_epoch": int(config["epochs"]),
        "best_value": float(checkpoint["selection_value"]),
        "best_val_metrics": checkpoint["val_metrics"],
        "calibrated_val_metrics": calibrated_val_metrics,
        "test_metrics": test_metrics,
        "test_results": str(completed_results_path),
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    })
    write_teacher_run_status(run_dir, run_status)
    print(f"完成 seed={seed}，结果目录：{run_dir}")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/teacher.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument(
        "--resume",
        help="从 run 目录或 last_teacher.pth 恢复单个 seed，默认不检验哈希；已完成时不重复测试。",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="仅检查配置、数据清单和一次前向传播，不启动训练。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.resume:
        os.environ["LITE_MYONET_RESUME_READABLE_ONLY"] = "1"
    configure_torch_cpu_threads_from_environment()
    config, config_path = load_config(args.config)
    validate_experiment_config(config)
    label_map, label_map_path = load_label_map(config, config_path)
    data_verification_report = verify_configured_data(config, config_path)
    print(json.dumps(data_verification_report, ensure_ascii=False, indent=2))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前进程不可用。")

    if args.check_only:
        if args.resume:
            raise ValueError("--check-only 与 --resume 不能同时使用。")
        check_experiment_setup(config, config_path, device, label_map)
        return

    if args.resume:
        resume_checkpoint = resolve_resume_checkpoint(args.resume)
        resume_metadata = torch.load(
            resume_checkpoint, map_location="cpu", weights_only=False
        )
        if not isinstance(resume_metadata, dict) or "seed" not in resume_metadata:
            raise ValueError("续训断点缺少 seed。")
        resume_seed = int(resume_metadata["seed"])
        if args.seeds and args.seeds != [resume_seed]:
            raise ValueError("--resume 只能恢复断点对应的单个 seed。")
        seeds = [resume_seed]
    else:
        seeds = args.seeds if args.seeds else list(config.get("seeds", []))
    if not seeds:
        raise ValueError("至少需要提供一个随机种子。")

    if len(seeds) != len(set(seeds)):
        raise ValueError("命令行提供的随机种子存在重复值。")

    results = [
        train_one_seed(
            config,
            config_path,
            int(seed),
            device,
            data_verification_report,
            label_map,
            label_map_path,
            resume_path=args.resume,
        )
        for seed in seeds
    ]
    summary = aggregate_seed_results(results)
    output_root = resolve_path(config_path.parent, config.get("output_root", "runs"))
    summary_path = output_root / (
        f"teacher_multiseed_summary_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"多种子汇总：{summary_path}")


if __name__ == "__main__":
    main()
