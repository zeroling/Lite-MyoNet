"""训练双流学生模型并严格记录师生蒸馏实验。"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import shlex
import shutil
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping
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

from ..data.packed import InMemoryNpzSkeletonDataset, load_archive_metadata
from ..models import topology
from ..models.student import MyoFormerStudent
from ..models.teacher import MyoFormerTeacher
from ..runtime import configure_torch_cpu_threads_from_environment, resume_readable_only
from .distillation import DistillationLoss, STUDENT_FEATURE_NAMES
from .epoch_stats import EpochStatistics
from .schedulers import TwoStageLR, WarmupCosineLR
from .teacher_trainer import (
    create_progress_bar,
    file_sha256,
    code_tree_sha256,
    load_config,
    load_label_map,
    make_loader,
    resolve_path,
    seed_worker,
    set_global_seed,
    verify_resume_code_compatibility,
    write_json,
)
from .checkpoint_io import (
    CheckpointRecoveryError,
    load_latest_checkpoint,
    save_checkpoint_atomic as _durable_save_checkpoint_atomic,
)


STUDENT_RESUME_REQUIRED_KEYS = {
    "student_state_dict", "student_config", "seed", "epoch", "selection_metric",
    "optimizer_state_dict", "scheduler_state_dict", "loss_config",
    "teacher_checkpoint_path", "teacher_checkpoint_sha256", "rng_state", "val_metrics",
    "config_sha256", "data_archive_sha256", "optimizer_config", "scheduler_config",
    "best_value", "best_epoch", "train_loader_generator_state", "training_device_type",
    "precision", "effective_precision", "scaler_state", "teacher_cache_manifest_sha256",
    "test_evaluation_count", "checkpoint_kind",
}
STUDENT_READABLE_RESUME_REQUIRED_KEYS = {
    "student_state_dict", "optimizer_state_dict", "scheduler_state_dict",
    "seed", "epoch", "selection_metric", "best_value", "best_epoch",
    "checkpoint_kind",
}


_ROOT_KEYS = {
    "data", "teacher", "output_root", "seeds", "deterministic",
    "selection_metric", "epochs", "batch_size", "num_workers",
    "prefetch_factor", "model", "loss", "distillation", "optimizer",
    "scheduler", "precision", "teacher_cache", "save_epoch_weights",
}
_DATA_KEYS = {"archive", "archive_sha256", "label_map", "frames", "num_points"}
_TEACHER_KEYS = {"checkpoint", "sha256"}
_STUDENT_KEYS = {
    "num_points", "in_channels", "channels", "dropout", "temporal_basis",
    "fourier_degree", "spline_grid_size", "use_static_graph",
    "use_dynamic_graph", "classifier_head", "skeleton_edges", "use_learnable_bias", "use_fixed_prior",
}
_LOSS_KEYS = {"label_smoothing"}
_DISTILLATION_KEYS = {
    "kd_temperature", "topology_temperature", "kd_weight",
    "static_adjacency_weight", "dynamic_similarity_weight",
}
_OPTIMIZER_KEYS = {"peak_lr", "weight_decay", "betas", "eps"}
_SCHEDULER_KEYS = {"warmup_ratio", "warmup_init_lr", "warmup_gamma", "decay_gamma", "type", "min_lr"}
_SELECTION_METRICS = {"accuracy", "balanced_accuracy", "macro_f1"}
_STUDENT_FEATURE_RENAMES = {
    "joint_stage1": "joint.stage1",
    "bone_vector_stage1": "bone.stage1",
    "joint_stage2": "joint.stage2",
    "bone_vector_stage2": "bone.stage2",
    "joint_stage3": "joint.stage3",
    "bone_vector_stage3": "bone.stage3",
}


def _canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_mapping(config, name):
    if not isinstance(config, Mapping):
        raise ValueError(f"配置项 {name} 必须是映射结构。")
    return config


def _reject_unknown_keys(config, allowed, name):
    unknown = sorted(set(config) - set(allowed))
    if unknown:
        raise ValueError(f"配置项 {name} 含有未知键 {unknown}；请修正键名，禁止静默忽略。")


def _finite_number(value, name, minimum=None, maximum=None, minimum_open=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是有限数值。")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须是有限数值。") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} 必须是有限数值。")
    if minimum is not None and (result <= minimum if minimum_open else result < minimum):
        operator = ">" if minimum_open else ">="
        raise ValueError(f"{name} 必须 {operator} {minimum}。")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} 必须不大于 {maximum}。")
    return result


def validate_student_config(config):
    """严格检查配置结构、必要字段和参数范围。"""
    _require_mapping(config, "根")
    _reject_unknown_keys(config, _ROOT_KEYS, "根")
    required_root = {
        "data", "teacher", "output_root", "seeds", "selection_metric",
        "epochs", "batch_size", "num_workers", "model", "loss",
        "distillation", "optimizer", "scheduler", "precision",
    }
    missing = sorted(required_root - set(config))
    if missing:
        raise ValueError(f"学生训练配置缺少必需项：{missing}。")

    data = _require_mapping(config["data"], "data")
    _reject_unknown_keys(data, _DATA_KEYS, "data")
    for key in ("archive", "archive_sha256", "label_map", "frames", "num_points"):
        if key not in data or data[key] in (None, ""):
            raise ValueError(f"配置中必须明确提供 data.{key}。")
    if not resume_readable_only():
        if len(str(data["archive_sha256"])) != 64:
            raise ValueError("data.archive_sha256 必须是 64 位 SHA-256。")
        try:
            int(str(data["archive_sha256"]), 16)
        except ValueError as error:
            raise ValueError("data.archive_sha256 必须是十六进制 SHA-256。") from error
    if int(data["frames"]) <= 0 or int(data["num_points"]) <= 0:
        raise ValueError("data.frames 和 data.num_points 必须为正整数。")

    teacher = _require_mapping(config["teacher"], "teacher")
    _reject_unknown_keys(teacher, _TEACHER_KEYS, "teacher")
    if not teacher.get("checkpoint"):
        raise ValueError("配置中必须明确提供 teacher.checkpoint。")
    if not resume_readable_only() and (
        not teacher.get("sha256") or len(str(teacher["sha256"])) != 64
    ):
        raise ValueError("配置中必须提供 64 位 teacher.sha256。")

    model = _require_mapping(config["model"], "model")
    _reject_unknown_keys(model, _STUDENT_KEYS, "model")
    required_model = {
        "num_points", "in_channels", "channels", "dropout", "temporal_basis",
        "fourier_degree", "spline_grid_size", "use_static_graph",
        "use_dynamic_graph", "classifier_head",
    }
    missing_model = sorted(required_model - set(model))
    if missing_model:
        raise ValueError(f"配置中必须明确提供完整学生架构参数：model.{missing_model}。")
    if int(model["num_points"]) != int(data["num_points"]):
        raise ValueError("model.num_points 必须与 data.num_points 相同。")
    if len(model["channels"]) != 3 or any(int(width) <= 0 for width in model["channels"]):
        raise ValueError("model.channels 必须恰含三个正整数。")
    if model["temporal_basis"] not in {"standard", "fourier", "bspline"}:
        raise ValueError("model.temporal_basis 必须为 standard、fourier 或 bspline。")
    if int(model["fourier_degree"]) not in (1, 2, 3):
        raise ValueError("model.fourier_degree 必须为 1、2 或 3。")
    if int(model["spline_grid_size"]) <= 0:
        raise ValueError("model.spline_grid_size 必须为正整数。")
    if not isinstance(model["use_static_graph"], bool) or not isinstance(model["use_dynamic_graph"], bool):
        raise ValueError("model.use_static_graph 和 model.use_dynamic_graph 必须是布尔值。")
    if not model["use_static_graph"] and not model["use_dynamic_graph"]:
        raise ValueError("静态图和动态时序图不能同时关闭。")
    if model["classifier_head"] not in {"cosine", "linear"}:
        raise ValueError("model.classifier_head 仅支持 cosine 或 linear。")
    _finite_number(model["dropout"], "model.dropout", minimum=0.0, maximum=1.0)
    if float(model["dropout"]) == 1.0:
        raise ValueError("model.dropout 必须小于 1。")

    loss = _require_mapping(config["loss"], "loss")
    _reject_unknown_keys(loss, _LOSS_KEYS, "loss")
    if set(loss) != _LOSS_KEYS:
        raise ValueError("loss 必须显式设置 label_smoothing。")
    _finite_number(loss["label_smoothing"], "loss.label_smoothing", minimum=0.0, maximum=1.0)
    if float(loss["label_smoothing"]) == 1.0:
        raise ValueError("loss.label_smoothing 必须位于 [0, 1) 区间。")

    distillation = _require_mapping(config["distillation"], "distillation")
    _reject_unknown_keys(distillation, _DISTILLATION_KEYS, "distillation")
    missing_distillation = sorted(_DISTILLATION_KEYS - set(distillation))
    if missing_distillation:
        raise ValueError(f"蒸馏配置缺少必需项：{missing_distillation}。")
    for key in ("kd_temperature", "topology_temperature"):
        _finite_number(distillation[key], f"distillation.{key}", minimum=0.0, minimum_open=True)
    for key in ("kd_weight", "static_adjacency_weight", "dynamic_similarity_weight"):
        _finite_number(distillation[key], f"distillation.{key}", minimum=0.0)

    optimizer = _require_mapping(config["optimizer"], "optimizer")
    _reject_unknown_keys(optimizer, _OPTIMIZER_KEYS, "optimizer")
    if set(optimizer) != _OPTIMIZER_KEYS:
        raise ValueError("optimizer 必须显式设置 peak_lr、weight_decay、betas 和 eps。")
    _finite_number(optimizer["peak_lr"], "optimizer.peak_lr", minimum=0.0, minimum_open=True)
    _finite_number(optimizer["weight_decay"], "optimizer.weight_decay", minimum=0.0)
    _finite_number(optimizer["eps"], "optimizer.eps", minimum=0.0, minimum_open=True)
    if not isinstance(optimizer["betas"], (list, tuple)) or len(optimizer["betas"]) != 2:
        raise ValueError("optimizer.betas 必须包含两个 [0, 1) 数值。")
    for index, beta in enumerate(optimizer["betas"]):
        _finite_number(beta, f"optimizer.betas[{index}]", minimum=0.0, maximum=1.0)
        if float(beta) == 1.0:
            raise ValueError("optimizer.betas 的每个值必须小于 1。")

    scheduler = _require_mapping(config["scheduler"], "scheduler")
    _reject_unknown_keys(scheduler, _SCHEDULER_KEYS, "scheduler")
    if not {"warmup_ratio", "warmup_init_lr", "warmup_gamma", "decay_gamma"} <= set(scheduler):
        raise ValueError("scheduler 必须显式设置所有四个 TwoStageLR 参数。")
    _finite_number(scheduler["warmup_ratio"], "scheduler.warmup_ratio", minimum=0.0, maximum=1.0)
    if float(scheduler["warmup_ratio"]) == 1.0:
        raise ValueError("scheduler.warmup_ratio 必须小于 1。")
    _finite_number(scheduler["warmup_init_lr"], "scheduler.warmup_init_lr", minimum=0.0)
    _finite_number(scheduler["warmup_gamma"], "scheduler.warmup_gamma", minimum=0.0)
    _finite_number(scheduler["decay_gamma"], "scheduler.decay_gamma", minimum=0.0)
    if float(scheduler["warmup_init_lr"]) > float(optimizer["peak_lr"]):
        raise ValueError("scheduler.warmup_init_lr 不能大于 optimizer.peak_lr。")

    for name in ("epochs", "batch_size"):
        if isinstance(config[name], bool) or int(config[name]) <= 0:
            raise ValueError(f"{name} 必须为正整数。")
    workers = config["num_workers"]
    if isinstance(workers, bool) or int(workers) != 0:
        raise ValueError("内存 NPZ 数据包训练要求 num_workers=0。")
    if "prefetch_factor" in config and int(config["prefetch_factor"]) <= 0:
        raise ValueError("prefetch_factor 必须为正整数。")
    seeds = config["seeds"]
    if not isinstance(seeds, (tuple, list)) or not seeds:
        raise ValueError("seeds 必须为非空整数列表。")
    if any(isinstance(seed, bool) for seed in seeds):
        raise ValueError("seeds 只能包含整数。")
    normalized_seeds = [int(seed) for seed in seeds]
    if len(normalized_seeds) != len(set(normalized_seeds)):
        raise ValueError("seeds 中存在重复值。")
    if config["selection_metric"] not in _SELECTION_METRICS:
        raise ValueError(f"selection_metric 必须属于 {sorted(_SELECTION_METRICS)}。")
    if config["precision"] not in {"amp", "fp32", "bf16"}:
        raise ValueError("precision 必须为 amp、bf16 或 fp32。")
    teacher_cache = config.get("teacher_cache")
    if teacher_cache is not None:
        _require_mapping(teacher_cache, "teacher_cache")
        _reject_unknown_keys(teacher_cache, {"directory", "manifest_sha256"}, "teacher_cache")
        if set(teacher_cache) != {"directory", "manifest_sha256"}:
            raise ValueError("teacher_cache 必须显式包含 directory 和 manifest_sha256。")
        if not isinstance(teacher_cache["directory"], (str, os.PathLike)) or not str(teacher_cache["directory"]):
            raise ValueError("teacher_cache.directory 必须是非空路径。")
        cache_manifest_hash = str(teacher_cache["manifest_sha256"])
        if not resume_readable_only():
            if len(cache_manifest_hash) != 64:
                raise ValueError("teacher_cache.manifest_sha256 必须是 64 位 SHA-256。")
            try:
                int(cache_manifest_hash, 16)
            except ValueError as error:
                raise ValueError("teacher_cache.manifest_sha256 必须是十六进制 SHA-256。") from error
    if not isinstance(config["output_root"], (str, os.PathLike)) or not str(config["output_root"]):
        raise ValueError("output_root 必须是非空路径。")
    return normalized_seeds


def effective_precision(device, requested_precision):
    if requested_precision not in {"amp", "fp32", "bf16"}:
        raise ValueError("precision 必须为 amp、bf16 或 fp32。")
    if requested_precision == "bf16":
        if device.type != "cuda":
            raise ValueError("precision=bf16 要求支持 BF16 的 CUDA 设备。")
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(f"CUDA 设备 {device} 不支持 BF16 autocast。")
        return "bf16"
    return "amp" if device.type == "cuda" and requested_precision == "amp" else "fp32"


def create_grad_scaler(device, requested_precision):
    enabled = effective_precision(device, requested_precision) == "amp"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def open_teacher_target_cache(config, config_path, teacher_report, data_report, label_map_path, device):
    cache_config = config.get("teacher_cache")
    if cache_config is None:
        return None, None
    from .teacher_target_cache import TeacherTargetCache

    cache_dir = resolve_path(config_path.parent, cache_config["directory"])
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("教师目标缓存缺少 manifest.json：{}".format(manifest_path))
    actual_manifest_hash = file_sha256(manifest_path)
    if not resume_readable_only() and actual_manifest_hash.lower() != str(cache_config["manifest_sha256"]).lower():
        raise ValueError("teacher_cache.manifest_sha256 与实际 manifest 不一致。")
    if not teacher_report.get("resolved_config_path"):
        raise ValueError("无法验证缓存所用教师架构：teacher_report 缺少 resolved_config_path。")
    teacher_config, _ = load_config(teacher_report["resolved_config_path"])
    cache_loss_config = {
        "kd_temperature": float(config["distillation"]["kd_temperature"]),
        "topology_temperature": float(config["distillation"]["topology_temperature"]),
    }
    effective = effective_precision(device, str(config["precision"]))
    from .teacher_target_cache import code_tree_sha256 as cache_code_tree_sha256
    from .teacher_target_cache import make_cache_identity

    expected_hashes = make_cache_identity(
        teacher_checkpoint_sha256=teacher_report["checkpoint_sha256"],
        archive_sha256=data_report["archive_sha256"],
        label_map_sha256=file_sha256(label_map_path),
        requested_precision=str(config["precision"]),
        effective_precision=effective,
        teacher_model_config=teacher_config["model"],
        loss_config=cache_loss_config,
        code_sha256=cache_code_tree_sha256(Path(__file__).resolve().parents[3]),
        generation_batch_size=int(config["batch_size"]),
    )
    # The resolved config pins this exact manifest SHA. The manifest retains the
    # code hash from cache generation as provenance; only that provenance may
    # differ after an unrelated code fix. All semantic identity fields and all
    # cache payload hashes remain strictly verified by the reader.
    cache = TeacherTargetCache(
        cache_dir,
        expected_hashes=expected_hashes,
        allow_code_provenance_mismatch=True,
    )
    if int(cache.manifest.get("generation_batch_size", -1)) != int(config["batch_size"]):
        cache.close()
        raise ValueError(
            "教师目标缓存 generation_batch_size 与学生配置 batch_size 不一致。"
        )
    manifest_identity = dict(cache.manifest["identity"])
    pinned_identity = dict(expected_hashes)
    pinned_identity["code_sha256"] = manifest_identity["code_sha256"]
    code_provenance_mismatch = (
        expected_hashes["code_sha256"] != manifest_identity["code_sha256"]
    )
    from .distillation import FEATURE_MATCHES, TEACHER_FEATURE_NAMES
    from .teacher_target_cache import physical_edge_indices

    representation = cache.manifest.get("target_representation", {})
    expected_edges = list(range(int(config["data"]["num_points"]) ** 2))
    expected_matches = [
        {"student": match.student, "teacher": match.teacher, "weight": match.weight}
        for match in FEATURE_MATCHES
    ]
    if representation.get("edge_indices_flat") != expected_edges:
        raise ValueError("教师目标缓存静态物理边索引与学生骨架配置不一致。")
    if representation.get("feature_match_order") != expected_matches:
        raise ValueError("教师目标缓存 feature match 顺序或权重与蒸馏损失不一致。")
    if representation.get("teacher_feature_order") != list(TEACHER_FEATURE_NAMES):
        raise ValueError("教师目标缓存教师特征顺序与蒸馏损失契约不一致。")
    if float(representation.get("topology_temperature", -1.0)) != cache_loss_config["topology_temperature"]:
        raise ValueError("教师目标缓存 topology_temperature 与学生配置不一致。")
    if cache.manifest.get("loss_config") != cache_loss_config:
        raise ValueError("教师目标缓存的目标生成损失配置与当前学生配置不一致。")
    split_counts = data_report.get("split_counts", {})
    num_classes = len(load_label_map(config, config_path)[0])
    num_points = int(config["data"]["num_points"])
    for split in ("train", "val"):
        split_manifest = cache.manifest["splits"][split]
        count = int(split_counts.get(split, -1))
        expected_shapes = {
            "teacher_logits": [count, num_classes],
            "static_targets": [count, len(FEATURE_MATCHES), len(expected_edges)],
            "dynamic_targets": [count, len(FEATURE_MATCHES), num_points, num_points],
            "row_indices": [count],
        }
        if count <= 0 or int(split_manifest.get("count", -1)) != count:
            raise ValueError("教师目标缓存 {} 样本数与数据包 metadata 不一致。".format(split))
        if split_manifest.get("shapes") != expected_shapes:
            raise ValueError("教师目标缓存 {} 数组形状不符合当前训练协议。".format(split))
    report = {
        "directory": str(cache_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": actual_manifest_hash,
        "expected_hashes": expected_hashes,
        "manifest_identity": manifest_identity,
        "pinned_identity": pinned_identity,
        "code_provenance_mismatch": code_provenance_mismatch,
        "splits": {name: int(len(cache.splits[name].logits)) for name in ("train", "val")},
        "test_cached": False,
    }
    return cache, report


def _teacher_cache_identity_matches(saved_report, current_report):
    """Keep cache identity strict while allowing its source-tree provenance to age."""
    if saved_report is None or current_report is None:
        return saved_report is None and current_report is None
    if not isinstance(saved_report, Mapping) or not isinstance(current_report, Mapping):
        return False

    def normalized(report):
        result = dict(report)
        expected = result.get("expected_hashes")
        if isinstance(expected, Mapping):
            expected = dict(expected)
            expected.pop("code_sha256", None)
            result["expected_hashes"] = expected
        result.pop("code_provenance_mismatch", None)
        return result

    return normalized(saved_report) == normalized(current_report)


def verify_student_archive(config, config_path):
    """只读取包文件哈希和 metadata；训练前不实例化或读取 test 划分。"""
    data = config["data"]
    archive_path = resolve_path(config_path.parent, data["archive"])
    if not archive_path.is_file():
        raise FileNotFoundError(f"数据包不存在：{archive_path}")
    actual_hash = file_sha256(archive_path)
    if not resume_readable_only() and actual_hash.lower() != str(data["archive_sha256"]).lower():
        raise ValueError(
            f"数据包 SHA-256 不匹配：期望 {data['archive_sha256']}，实际 {actual_hash}。"
        )
    metadata = load_archive_metadata(archive_path)
    expected_shape = [3, int(data["frames"]), int(data["num_points"])]
    if list(metadata.get("sample_shape", [])) != expected_shape:
        raise ValueError(
            f"数据包样本形状为 {metadata.get('sample_shape')}，期望 {expected_shape}。"
        )
    split_counts = metadata.get("split_counts")
    if not isinstance(split_counts, dict) or set(split_counts) != {"train", "val", "test"}:
        raise ValueError("数据包 metadata.split_counts 必须明确列出 train、val、test。")
    if any(int(split_counts[split]) <= 0 for split in ("train", "val", "test")):
        raise ValueError("数据包的 train、val、test 划分都必须非空。")
    if sum(int(value) for value in split_counts.values()) != int(metadata["total_sequences"]):
        raise ValueError("数据包划分数量之和与 total_sequences 不一致。")
    return {
        "archive_path": str(archive_path),
        "archive_sha256": actual_hash,
        "source_manifest_sha256": metadata.get("source_manifest_sha256"),
        "split_counts": {key: int(value) for key, value in split_counts.items()},
        "sample_shape": expected_shape,
        "dtype": metadata.get("array_dtype"),
        "total_sequences": int(metadata["total_sequences"]),
        "test_arrays_constructed": False,
    }


def create_student_config(model_config, num_classes):
    """返回可直接传给 MyoFormerStudent 的、完整而可序列化的架构配置。"""
    return {
        "num_classes": int(num_classes),
        "num_points": int(model_config["num_points"]),
        "in_channels": int(model_config["in_channels"]),
        "channels": [int(value) for value in model_config["channels"]],
        "dropout": float(model_config["dropout"]),
        "temporal_basis": str(model_config["temporal_basis"]),
        "fourier_degree": int(model_config["fourier_degree"]),
        "spline_grid_size": int(model_config["spline_grid_size"]),
        "use_static_graph": bool(model_config["use_static_graph"]),
        "use_dynamic_graph": bool(model_config["use_dynamic_graph"]),
        "use_learnable_bias": bool(model_config.get("use_learnable_bias", True)),
        "use_fixed_prior": bool(model_config.get("use_fixed_prior", True)),
        "classifier_head": str(model_config["classifier_head"]),
        "skeleton_edges": [
            [int(source), int(target)]
            for source, target in model_config.get("skeleton_edges", topology.bones)
        ],
    }


def rename_student_features(features):
    """把模型公开的 underscore 阶段名映射到蒸馏损失的点分隔命名契约。"""
    if not isinstance(features, Mapping):
        raise TypeError("学生 return_features 必须是具名特征映射。")
    expected = set(_STUDENT_FEATURE_RENAMES)
    actual = set(features)
    if actual != expected:
        raise ValueError(
            "学生模型公开特征键不符合训练契约；缺少={}，多余={}。".format(
                sorted(expected - actual), sorted(actual - expected)
            )
        )
    renamed = {
        target: features[source]
        for source, target in _STUDENT_FEATURE_RENAMES.items()
    }
    if tuple(renamed) != STUDENT_FEATURE_NAMES:
        # 明确排序，避免未来修改映射字典顺序后造成意外的序列语义。
        renamed = {name: renamed[name] for name in STUDENT_FEATURE_NAMES}
    return renamed


def _create_teacher_model(model_config, num_classes):
    if not isinstance(model_config, Mapping):
        raise ValueError("教师源配置缺少 model 映射。")
    allowed = {
        "num_points", "in_channels", "dropout", "use_spatial_attention",
        "use_symmetry_attention", "use_scale_attention", "use_joint_spatial_stream",
        "use_joint_temporal_stream", "use_bone_vector_stream", "spatial_attention",
        "symmetry_attention", "scale_attention", "head", "width_ratio",
        "skeleton_edges", "symmetry_pairs", "root_index", "input_scheme",
    }
    _reject_unknown_keys(model_config, allowed, "teacher.model")
    for key in (
        "use_joint_spatial_stream", "use_joint_temporal_stream", "use_bone_vector_stream"
    ):
        if model_config.get(key) is not True:
            raise ValueError(f"教师蒸馏要求教师三条流全部启用：teacher.model.{key}=true。")
    return MyoFormerTeacher(
        num_classes=int(num_classes),
        num_points=int(model_config["num_points"]),
        in_channels=int(model_config["in_channels"]),
        dropout=float(model_config["dropout"]),
        width_ratio=float(model_config.get("width_ratio", 1.0)),
        use_spatial_attention=bool(model_config["use_spatial_attention"]),
        use_symmetry_attention=bool(model_config["use_symmetry_attention"]),
        use_scale_attention=bool(model_config["use_scale_attention"]),
        use_joint_spatial_stream=True,
        use_joint_temporal_stream=True,
        use_bone_vector_stream=True,
        spatial_attention_config=model_config["spatial_attention"],
        symmetry_attention_config=model_config["symmetry_attention"],
        skeleton_edges=model_config.get("skeleton_edges"),
        symmetry_pairs=model_config.get("symmetry_pairs"),
        root_index=int(model_config.get("root_index", 0)),
        input_scheme=model_config.get("input_scheme", "X1/X2/X2"),
        scale_attention_config=model_config["scale_attention"],
        head_config=model_config["head"],
    )


def _torch_load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch 2.0 compatibility.
        return torch.load(path, map_location=map_location)


def load_strict_teacher(config, config_path, label_map, label_map_path, data_report, device):
    """仅接收当前训练器生成且带完整运行旁证的教师最佳 checkpoint。"""
    teacher_cfg = config["teacher"]
    readable_only = resume_readable_only()
    checkpoint_path = resolve_path(config_path.parent, teacher_cfg["checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"教师 checkpoint 不存在：{checkpoint_path}")
    checkpoint_sha256 = file_sha256(checkpoint_path)
    if not readable_only and checkpoint_sha256.lower() != str(teacher_cfg["sha256"]).lower():
        raise ValueError(
            f"教师 checkpoint SHA-256 不匹配：期望 {teacher_cfg['sha256']}，实际 {checkpoint_sha256}。"
        )

    run_dir = checkpoint_path.parent
    resolved_config_path = run_dir / "resolved_config.yaml"
    provenance_path = run_dir / "provenance.json"
    if not resolved_config_path.is_file() or (not readable_only and not provenance_path.is_file()):
        raise ValueError(
            "无法读取教师架构/运行来源文件：需要 resolved_config.yaml，严格模式还需要 provenance.json。"
        )
    teacher_config, _ = load_config(resolved_config_path)
    teacher_provenance = {}
    if provenance_path.is_file():
        try:
            with provenance_path.open("r", encoding="utf-8") as provenance_file:
                teacher_provenance = json.load(provenance_file)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            if not readable_only:
                raise
    if not readable_only:
        source_config_path = Path(teacher_provenance.get("config_path", "")).expanduser()
        if not source_config_path.is_file():
            raise ValueError("教师原始配置文件不存在，不能验证该 checkpoint 的来源。")
        if file_sha256(source_config_path) != teacher_provenance.get("config_sha256"):
            raise ValueError("教师原始配置 SHA-256 与 provenance.json 不一致。")
        source_config, _ = load_config(source_config_path)
        if source_config != teacher_config:
            raise ValueError("教师 resolved_config.yaml 与经过哈希验证的原始配置内容不一致。")
        if teacher_provenance.get("label_map_sha256") != file_sha256(label_map_path):
            raise ValueError("教师与学生使用的标签映射文件 SHA-256 不一致。")
        if teacher_provenance.get("archive_sha256") != data_report["archive_sha256"]:
            raise ValueError("教师训练使用的数据包 SHA-256 与学生配置不一致。")
        if teacher_config.get("data", {}).get("archive_sha256") != data_report["archive_sha256"]:
            raise ValueError("教师 resolved_config.yaml 指向的数据包哈希与学生配置不一致。")
        for key in ("frames", "num_points"):
            if int(teacher_config.get("data", {}).get(key, -1)) != int(config["data"][key]):
                raise ValueError(f"教师与学生的 data.{key} 配置不一致。")
        if teacher_provenance.get("data_verification", {}).get("archive_sha256") != data_report["archive_sha256"]:
            raise ValueError("教师 provenance 的数据校验记录与学生数据包哈希不一致。")

    required_checkpoint_keys = {
        "epoch", "seed", "selection_metric", "selection_value", "val_metrics",
        "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
    }
    checkpoint = _torch_load_checkpoint(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("教师 checkpoint 必须是当前格式的字典结构。")
    if readable_only:
        required_checkpoint_keys = {"epoch", "seed", "model_state_dict"}
    missing = sorted(required_checkpoint_keys - set(checkpoint))
    if missing:
        raise ValueError(f"教师 checkpoint 缺少当前格式字段 {missing}；拒绝旧权重静默兼容。")
    if int(checkpoint["epoch"]) <= 0:
        raise ValueError("教师 checkpoint 的 epoch 必须为正数。")
    if not readable_only and int(checkpoint["seed"]) != int(teacher_provenance["seed"]):
        raise ValueError("教师 checkpoint 的 seed 与运行旁证不一致。")
    if not readable_only and checkpoint["selection_metric"] not in _SELECTION_METRICS:
        raise ValueError("教师 checkpoint 的验证集 selection_metric 无效。")

    model_config = teacher_config.get("model")
    if not isinstance(model_config, Mapping):
        raise ValueError("教师 resolved_config.yaml 缺少 model 架构描述。")
    if int(model_config.get("num_points", -1)) != int(config["data"]["num_points"]):
        raise ValueError("教师架构的 num_points 与学生输入骨架不一致。")
    if int(model_config.get("in_channels", -1)) != 3 or int(config["model"]["in_channels"]) != 3:
        raise ValueError("教师和学生输入必须均为三维骨架坐标/向量。")
    teacher = _create_teacher_model(model_config, num_classes=len(label_map)).to(device).float()
    expected_state = teacher.state_dict()
    incoming_state = checkpoint["model_state_dict"]
    if not isinstance(incoming_state, Mapping):
        raise ValueError("教师 model_state_dict 必须是张量映射。")
    expected_keys = set(expected_state)
    actual_keys = set(incoming_state)
    if expected_keys != actual_keys:
        raise ValueError(
            "教师 checkpoint 张量键与当前架构不一致；缺少={}，多余={}。".format(
                sorted(expected_keys - actual_keys), sorted(actual_keys - expected_keys)
            )
        )
    for key, expected in expected_state.items():
        actual = incoming_state[key]
        if not isinstance(actual, torch.Tensor):
            raise TypeError(f"教师 checkpoint 的 {key} 不是张量。")
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(
                f"教师 checkpoint 的 {key} 形状或 dtype 与当前架构不一致："
                f"{tuple(actual.shape)}/{actual.dtype} vs {tuple(expected.shape)}/{expected.dtype}。"
            )
        if actual.is_floating_point() and not bool(torch.isfinite(actual).all()):
            raise ValueError(f"教师 checkpoint 的 {key} 含 NaN 或 Inf。")
    try:
        teacher.load_state_dict(incoming_state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"教师 checkpoint strict=True 加载失败：{error}") from error
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("教师参数冻结失败。")
    if teacher.training:
        raise RuntimeError("教师 eval 模式设置失败。")
    return teacher, {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "epoch": int(checkpoint["epoch"]),
        "seed": int(checkpoint["seed"]),
        "selection_metric": checkpoint.get("selection_metric"),
        "selection_value": float(checkpoint.get("selection_value", float("nan"))),
        "resolved_config_path": str(resolved_config_path),
        "provenance_path": str(provenance_path),
    }


def create_student_training_objects(config, device, num_classes):
    student_config = create_student_config(config["model"], num_classes)
    student = MyoFormerStudent(**student_config).to(device).float()
    loss_config = {
        "cross_entropy": {"label_smoothing": float(config["loss"]["label_smoothing"])},
        "distillation": {key: float(value) for key, value in config["distillation"].items()},
    }
    criterion = nn.CrossEntropyLoss(
        label_smoothing=loss_config["cross_entropy"]["label_smoothing"]
    )
    distillation = DistillationLoss(**loss_config["distillation"]).to(device)
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(distillation.parameters()),
        lr=float(optimizer_config["peak_lr"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["eps"]),
    )
    scheduler_config = config["scheduler"]
    scheduler_class = WarmupCosineLR if scheduler_config.get("type") == "warmup_cosine" else TwoStageLR
    schedule_options = (
        {"warmup_epochs": int(config["epochs"] * scheduler_config["warmup_ratio"]),
         "min_lr": float(scheduler_config.get("min_lr", 1e-6)), "legacy_warmup_index": True}
        if scheduler_class is WarmupCosineLR else
        {"warmup_ratio": float(scheduler_config["warmup_ratio"]), "decay_gamma": float(scheduler_config["decay_gamma"])}
    )
    scheduler = scheduler_class(
        optimizer=optimizer,
        total_epochs=int(config["epochs"]),
        peak_lr=float(optimizer_config["peak_lr"]),
        warmup_init_lr=float(scheduler_config["warmup_init_lr"]),
        warmup_gamma=float(scheduler_config["warmup_gamma"]),
        **schedule_options,
    )
    return student, criterion, distillation, optimizer, scheduler, student_config, loss_config


def calculate_student_metrics(targets, predictions, probabilities):
    if len(targets) == 0:
        raise ValueError("不能对空划分计算指标。")
    if probabilities.ndim != 2 or probabilities.shape[0] != len(targets):
        raise ValueError("概率数组形状与标签数量不匹配。")
    top_k = min(3, probabilities.shape[1])
    top_indices = np.argpartition(probabilities, -top_k, axis=1)[:, -top_k:]
    top3_accuracy = float(np.mean(np.any(top_indices == targets[:, None], axis=1)))
    return {
        "accuracy": float(np.mean(targets == predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, predictions, average="weighted", zero_division=0)),
        "top3_accuracy": top3_accuracy,
        "sample_count": int(len(targets)),
    }


def format_student_epoch_result(seed, epoch, learning_rate, train_metrics, val_metrics):
    """生成控制台和实验日志共用的学生轮次汇总。"""
    return (
        f"轮次结果 学生 种子={seed} 轮次={epoch:03d} "
        f"学习率={learning_rate:.8f} "
        f"训练损失={train_metrics['loss']:.4f} "
        f"训练准确率={train_metrics['accuracy']:.4f} "
        f"验证损失={val_metrics['loss']:.4f} "
        f"验证准确率={val_metrics['accuracy']:.4f} "
        f"验证平衡准确率={val_metrics['balanced_accuracy']:.4f} "
        f"验证宏平均F1={val_metrics['macro_f1']:.4f}"
    )


def run_student_epoch(
    student, teacher, loader, criterion, distillation, device, split,
    optimizer=None, keep_probabilities=False, precision="fp32", scaler=None,
    teacher_cache=None,
    progress_description=None,
):
    training = optimizer is not None
    student.train(training)
    teacher.eval()
    scaler = scaler if scaler is not None else create_grad_scaler(device, precision)
    effective = effective_precision(device, precision)
    amp_enabled = effective in {"amp", "bf16"}
    autocast_dtype = torch.bfloat16 if effective == "bf16" else torch.float16
    sample_count = 0
    statistics = EpochStatistics(keep_probabilities=keep_probabilities)
    last_progress_update = 0.0
    needs_topology = (
        distillation.static_adjacency_weight > 0.0
        or distillation.dynamic_similarity_weight > 0.0
    )
    needs_teacher = distillation.kd_weight > 0.0 or needs_topology
    required_fields = tuple(name for name, enabled in (
        ("teacher_logits", distillation.kd_weight > 0.0),
        ("static_targets", distillation.static_adjacency_weight > 0.0),
        ("dynamic_targets", distillation.dynamic_similarity_weight > 0.0),
    ) if enabled)
    context = torch.enable_grad() if training else torch.inference_mode()
    progress = create_progress_bar(loader, progress_description or split)
    with context:
        use_teacher_cache = teacher_cache is not None and split in {"train", "val"}
        for batch in progress:
            if len(batch) == 4:
                joints, bones, labels, row_indices = batch
            else:
                joints, bones, labels = batch
                row_indices = None
            joints = joints.to(device, non_blocking=device.type == "cuda")
            bones = bones.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, dtype=torch.long, non_blocking=device.type == "cuda")
            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda" if amp_enabled else "cpu",
                dtype=autocast_dtype,
                enabled=amp_enabled,
            ):
                teacher_logits = teacher_features = None
                cached_teacher_targets = None
                if use_teacher_cache and needs_teacher:
                    if row_indices is None:
                        raise ValueError("启用教师目标缓存时 DataLoader 必须返回 split-local row index。")
                    cached_teacher_targets = teacher_cache.get(
                        split, row_indices, device, fields=required_fields
                    )
                elif needs_teacher:
                    with torch.inference_mode():
                        teacher_result = teacher(
                            joints, bones, return_features=needs_topology
                        )
                    if needs_topology:
                        teacher_logits, teacher_features = teacher_result
                    else:
                        teacher_logits = teacher_result
                student_result = student(joints, bones, return_features=needs_topology)
                if needs_topology:
                    student_logits, student_features = student_result
                    student_features = rename_student_features(student_features)
                else:
                    student_logits, student_features = student_result, None
                ce_loss = criterion(student_logits, labels)
                distillation_components = distillation(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    student_features=student_features,
                    teacher_features=teacher_features,
                    cached_teacher_targets=cached_teacher_targets,
                    physical_adjacency=(
                        student.physical_adjacency
                        if distillation.static_adjacency_weight > 0.0 else None
                    ),
                    check_finite=False,
                )
                loss = ce_loss + distillation_components["loss_total"]
            if training:
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                # 旧学生入口直接更新，不做教师所用的梯度范数裁剪。
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

            batch_size = int(labels.shape[0])
            sample_count += batch_size
            probabilities = F.softmax(student_logits.detach().float(), dim=1)
            predictions = student_logits.detach().argmax(dim=1)
            top3_indices = torch.topk(
                probabilities, k=min(3, probabilities.shape[1]), dim=1
            ).indices
            statistics.add(
                (loss, ce_loss, distillation_components["loss_kd"],
                 distillation_components["loss_static_adjacency"],
                 distillation_components["loss_dynamic_similarity"]),
                labels, predictions,
                top3_indices.eq(labels.unsqueeze(1)).any(dim=1), probabilities,
            )
            now = time.monotonic()
            if now - last_progress_update >= 1.0:
                progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
                last_progress_update = now

    if sample_count == 0:
        raise ValueError(f"{split} DataLoader 为空。")
    totals, targets, predictions, top3_hits, probabilities = statistics.finish(student.num_classes)
    if not all(math.isfinite(value) for value in totals):
        raise FloatingPointError(f"{split} 总损失出现 NaN 或 Inf。")
    total_loss, total_ce, total_kd, total_static, total_dynamic = totals
    if keep_probabilities:
        metrics = calculate_student_metrics(targets, predictions, probabilities)
    else:
        metrics = {
            "accuracy": float(np.mean(targets == predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
            "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(targets, predictions, average="weighted", zero_division=0)),
            "top3_accuracy": float(top3_hits / sample_count),
            "sample_count": int(sample_count),
        }
    metrics.update({
        "loss": total_loss / sample_count,
        "cross_entropy_loss": total_ce / sample_count,
        "logit_kd_loss": total_kd / sample_count,
        "static_topology_loss": total_static / sample_count,
        "dynamic_topology_loss": total_dynamic / sample_count,
    })
    return metrics, targets, predictions, probabilities


def capture_rng_state():
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "state": numpy_state[1].astype(np.uint32, copy=False).tolist(),
            "position": int(numpy_state[2]),
            "has_gaussian": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().cpu(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [value.cpu() for value in torch.cuda.get_rng_state_all()]
    else:
        state["torch_cuda"] = []
    return state


def restore_rng_state(state, device):
    """恢复训练 checkpoint 中只含基础类型和张量的 RNG 状态。"""
    if not isinstance(state, Mapping) or not {"python", "numpy", "torch_cpu", "torch_cuda"}.issubset(state):
        raise ValueError("checkpoint.rng_state 字段不完整。")
    try:
        random.setstate(tuple(state["python"]))
        numpy_state = state["numpy"]
        restored_numpy = (
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_state["state"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gaussian"]),
            float(numpy_state["cached_gaussian"]),
        )
        np.random.set_state(restored_numpy)
        torch.set_rng_state(state["torch_cpu"].cpu())
        if device.type == "cuda":
            cuda_states = state["torch_cuda"]
            if not cuda_states:
                raise ValueError("checkpoint 没有 CUDA RNG 状态，不能在 CUDA 上精确续训。")
            torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"无法恢复 checkpoint RNG 状态：{error}") from error


def resolve_resume_target(value):
    """选择指定 run 中 epoch 最新的完整 last/backup/best 断点。"""
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        run_dir = path
    else:
        accepted_names = {
            "last_student.pth", "last_student.pth.bak",
            "best_student.pth", "best_student.pth.bak",
        }
        if path.name not in accepted_names:
            if not path.exists():
                raise FileNotFoundError(f"续训 checkpoint 不存在：{path}")
            raise ValueError(
                "--resume checkpoint 文件名必须是 last_student.pth、last_student.pth.bak、"
                "best_student.pth 或 best_student.pth.bak。"
            )
        run_dir = path.parent

    last_path = run_dir / "last_student.pth"
    best_path = run_dir / "best_student.pth"
    candidates = (
        last_path,
        last_path.with_name(f"{last_path.name}.bak"),
        best_path,
        best_path.with_name(f"{best_path.name}.bak"),
    )
    checkpoint_path, _ = load_latest_checkpoint(
        candidates,
        required_keys=(
            STUDENT_READABLE_RESUME_REQUIRED_KEYS
            if resume_readable_only() else STUDENT_RESUME_REQUIRED_KEYS
        ),
        map_location="cpu",
    )
    return run_dir, checkpoint_path


def _backup_resume_artifact(path):
    path = Path(path)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = path.with_name(f"{path.name}.pre_resume-{stamp}-{uuid.uuid4().hex[:8]}.bak")
    with path.open("rb") as source, backup.open("xb") as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)
        destination.flush()
        os.fsync(destination.fileno())
    return backup


def _write_reconciled_history(path, rows, fieldnames, encoding):
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{Path(path).name}.", suffix=".tmp", dir=str(Path(path).parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding=encoding) as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _reconcile_csv_history(path, checkpoint_epoch):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"续训 run 缺少 history.csv：{path}")
    with path.open("r", newline="", encoding="utf-8-sig") as history_file:
        reader = csv.DictReader(history_file)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or "epoch" not in fieldnames:
        raise ValueError(f"无法识别训练历史格式：{path}")
    try:
        epochs = [int(row["epoch"]) for row in rows]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"训练历史包含无效 epoch：{path}") from error
    expected_epochs = list(range(1, int(checkpoint_epoch) + 1))
    kept_rows = [row for row, epoch in zip(rows, epochs) if epoch <= int(checkpoint_epoch)]
    kept_epochs = [epoch for epoch in epochs if epoch <= int(checkpoint_epoch)]
    if kept_epochs != expected_epochs:
        raise ValueError(
            f"history.csv 的 epoch 与选中 checkpoint epoch={checkpoint_epoch} 不连续；"
            "拒绝跳过或伪造训练日志。"
        )
    if len(kept_rows) != len(rows):
        _backup_resume_artifact(path)
        _write_reconciled_history(path, kept_rows, fieldnames, "utf-8-sig")


def _reconcile_jsonl_history(path, checkpoint_epoch):
    path = Path(path)
    with path.open("r", encoding="utf-8") as history_file:
        lines = history_file.readlines()
    records = []
    epochs = []
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
            epoch = record["epoch"]
            epoch = int(epoch)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"history.jsonl 第 {line_number} 行没有有效 epoch，拒绝自动修剪。"
            ) from error
        records.append((line, epoch))
        epochs.append(epoch)
    expected_epochs = list(range(1, int(checkpoint_epoch) + 1))
    kept = [(line, epoch) for line, epoch in records if epoch <= int(checkpoint_epoch)]
    kept_epochs = [epoch for _, epoch in kept]
    if kept_epochs != expected_epochs:
        raise ValueError(
            f"history.jsonl 的 epoch 与选中 checkpoint epoch={checkpoint_epoch} 不连续；"
            "拒绝跳过或伪造训练日志。"
        )
    if len(kept) != len(lines):
        _backup_resume_artifact(path)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output_file:
                output_file.writelines(line for line, _ in kept)
                output_file.flush()
                os.fsync(output_file.fileno())
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


def _reconcile_resume_progress(run_dir, checkpoint, saved_status):
    checkpoint_epoch = int(checkpoint["epoch"])
    checkpoint_best_epoch = int(checkpoint["best_epoch"])
    _reconcile_csv_history(Path(run_dir) / "history.csv", checkpoint_epoch)
    jsonl_history = Path(run_dir) / "history.jsonl"
    if jsonl_history.is_file():
        _reconcile_jsonl_history(jsonl_history, checkpoint_epoch)

    current_epoch = int(saved_status.get("current_epoch", 0))
    best_epoch = int(saved_status.get("best_epoch", 0))
    if current_epoch > checkpoint_epoch or best_epoch > checkpoint_best_epoch:
        _backup_resume_artifact(Path(run_dir) / "run_status.json")
        saved_status["current_epoch"] = checkpoint_epoch
        saved_status["best_epoch"] = checkpoint_best_epoch
        saved_status["best_val_value"] = (
            float(checkpoint["best_value"])
            if math.isfinite(float(checkpoint["best_value"])) else None
        )
        saved_status["status"] = "running"


def load_resume_checkpoint(
    checkpoint_path, config, config_path, seed, student_config, loss_config, teacher_report,
    data_report, device, scaler, teacher_cache_report=None,
):
    checkpoint = _torch_load_checkpoint(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("续训 checkpoint 必须是当前格式的字典。")
    readable_only = resume_readable_only()
    required_keys = (
        STUDENT_READABLE_RESUME_REQUIRED_KEYS
        if readable_only else STUDENT_RESUME_REQUIRED_KEYS
    )
    missing = sorted(required_keys - set(checkpoint))
    if missing:
        raise ValueError(f"续训 checkpoint 缺少当前格式字段 {missing}。")
    if int(checkpoint["seed"]) != int(seed):
        raise ValueError(
            f"checkpoint 种子 {checkpoint['seed']} 与续训种子 {seed} 不一致。"
        )
    if not readable_only:
        if checkpoint["student_config"] != student_config:
            raise ValueError("学生架构配置与 checkpoint 不一致，拒绝续训。")
        if checkpoint["loss_config"] != loss_config:
            raise ValueError("损失配置与 checkpoint 不一致，拒绝续训。")
        if checkpoint["config_sha256"] != file_sha256(config_path):
            raise ValueError("训练配置文件 SHA-256 与 checkpoint 不一致，拒绝续训。")
        if checkpoint["data_archive_sha256"] != data_report["archive_sha256"]:
            raise ValueError("数据包 SHA-256 与 checkpoint 不一致，拒绝续训。")
        expected_cache_hash = teacher_cache_report["manifest_sha256"] if teacher_cache_report else None
        if checkpoint["teacher_cache_manifest_sha256"] != expected_cache_hash:
            raise ValueError("教师目标缓存 manifest SHA-256 与 checkpoint 不一致，拒绝续训。")
        if (
            checkpoint["teacher_checkpoint_path"] != teacher_report["checkpoint_path"]
            or checkpoint["teacher_checkpoint_sha256"] != teacher_report["checkpoint_sha256"]
        ):
            raise ValueError("教师 checkpoint 路径或 SHA-256 与续训配置不一致。")
        if checkpoint["optimizer_config"] != dict(config["optimizer"]):
            raise ValueError("optimizer 配置与 checkpoint 不一致，拒绝续训。")
        if checkpoint["scheduler_config"] != dict(config["scheduler"]):
            raise ValueError("scheduler 配置与 checkpoint 不一致，拒绝续训。")
        if checkpoint["selection_metric"] != config["selection_metric"]:
            raise ValueError("验证集选模指标与 checkpoint 不一致，拒绝续训。")
        if checkpoint["training_device_type"] != device.type:
            raise ValueError("续训设备类型与 checkpoint 不一致，不能保证 RNG 状态可复现。")
        expected_precision = str(config["precision"])
        expected_effective_precision = effective_precision(device, expected_precision)
        if checkpoint["precision"] != expected_precision:
            raise ValueError("precision 配置与 checkpoint 不一致，拒绝续训。")
        if checkpoint["effective_precision"] != expected_effective_precision:
            raise ValueError("当前设备的有效精度与 checkpoint 不一致，拒绝续训。")
        if not isinstance(checkpoint["scaler_state"], Mapping):
            raise ValueError("checkpoint.scaler_state 必须是字典。")
        try:
            scaler.load_state_dict(checkpoint["scaler_state"])
        except (RuntimeError, ValueError, KeyError) as error:
            raise ValueError(f"GradScaler 状态与当前精度不兼容：{error}") from error
    if int(checkpoint["epoch"]) < 0 or int(checkpoint["epoch"]) > int(config["epochs"]):
        raise ValueError(
            f"checkpoint epoch={checkpoint['epoch']} 超出总训练轮数；"
            f"当前总 epochs={config['epochs']}。"
        )
    return checkpoint


def _write_json_atomic(path, value):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


def _save_checkpoint_atomic(path, checkpoint):
    return _durable_save_checkpoint_atomic(
        path,
        checkpoint,
        required_keys=(
            STUDENT_READABLE_RESUME_REQUIRED_KEYS
            if resume_readable_only() else STUDENT_RESUME_REQUIRED_KEYS
        ),
    )


def _load_best_student_checkpoint(run_dir, resume_checkpoint, map_location="cpu"):
    expected_epoch = int(resume_checkpoint["best_epoch"])
    best_path = Path(run_dir) / "best_student.pth"
    last_path = Path(run_dir) / "last_student.pth"
    candidates = (
        (best_path, "best"),
        (best_path.with_name(f"{best_path.name}.bak"), "best"),
        (last_path, "last"),
        (last_path.with_name(f"{last_path.name}.bak"), "last"),
    )
    reasons = []
    for candidate_path, expected_kind in candidates:
        try:
            loaded_path, candidate = load_latest_checkpoint(
                [candidate_path],
                required_keys=(
                    STUDENT_READABLE_RESUME_REQUIRED_KEYS
                    if resume_readable_only() else STUDENT_RESUME_REQUIRED_KEYS
                ),
                map_location=map_location,
            )
        except (CheckpointRecoveryError, FileNotFoundError) as error:
            reasons.append(f"{candidate_path.name}: {error}")
            continue
        if int(candidate["epoch"]) != expected_epoch:
            reasons.append(
                f"{candidate_path.name}: epoch={candidate['epoch']} but best_epoch={expected_epoch}"
            )
            continue
        if int(candidate["best_epoch"]) != expected_epoch:
            reasons.append(f"{candidate_path.name}: best_epoch metadata does not match its epoch")
            continue
        if candidate.get("checkpoint_kind") != expected_kind:
            reasons.append(
                f"{candidate_path.name}: checkpoint_kind={candidate.get('checkpoint_kind')!r}"
            )
            continue
        if not resume_readable_only():
            for key in (
                "seed", "student_config", "config_sha256", "data_archive_sha256",
                "teacher_checkpoint_sha256", "teacher_cache_manifest_sha256", "selection_metric",
            ):
                if candidate.get(key) != resume_checkpoint.get(key):
                    raise ValueError(
                        f"验证集最佳 checkpoint 的 {key} 与续训 checkpoint 不一致。"
                    )
        if float(candidate["best_value"]) != float(resume_checkpoint["best_value"]):
            raise ValueError("验证集最佳 checkpoint 的 best_value 与续训 checkpoint 不一致。")
        return loaded_path, candidate

    raise CheckpointRecoveryError(
        "没有找到与续训状态 best_epoch 完全对应的验证集最佳权重；拒绝用 last 权重替代。 "
        + "; ".join(reasons)
    )


def prepare_experiment_directory(config, config_path):
    output_root = resolve_path(config_path.parent, config["output_root"])
    while True:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        experiment_dir = output_root / f"student_{timestamp}_{uuid.uuid4().hex}"
        try:
            experiment_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            if experiment_dir.exists():
                continue
            raise
        return experiment_dir


def write_student_provenance(
    run_dir, config, config_path, seed, device, data_report, teacher_report,
    label_map_path, student_config, loss_config, precision, effective, scaler_state,
    teacher_cache_report=None,
):
    provenance = {
        "seed": int(seed),
        "device": str(device),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "label_map_path": str(label_map_path),
        "label_map_sha256": file_sha256(label_map_path),
        "data_verification": data_report,
        "teacher": teacher_report,
        "teacher_cache": teacher_cache_report,
        "student_config": student_config,
        "loss_config": loss_config,
        "precision": precision,
        "effective_precision": effective,
        "scaler_state": scaler_state,
    }
    write_json(run_dir / "provenance.json", provenance)
    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file, allow_unicode=True, sort_keys=False)


def _command_line(parts):
    if os.name == "nt":
        import subprocess
        return subprocess.list2cmdline([str(part) for part in parts])
    return shlex.join([str(part) for part in parts])


def _build_run_status(
    run_dir, config, config_path, seed, device, data_report, teacher_report,
    student_config, loss_config, precision, effective, scaler_state, previous=None,
    teacher_cache_report=None,
):
    project_root = Path(__file__).resolve().parents[3]
    script_path = project_root / "scripts" / "train_student.py"
    common = [sys.executable, script_path, "--config", config_path, "--device", device]
    if previous is None:
        launch_command = _command_line(common + ["--seeds", seed])
        resume_count = 0
        started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    else:
        launch_command = previous.get("launch_command") or _command_line(common + ["--seeds", seed])
        resume_count = int(previous.get("resume_count", 0)) + 1
        started_at = previous.get("started_at") or dt.datetime.now(dt.timezone.utc).isoformat()
    resume_command = _command_line(common + ["--resume", run_dir])
    return {
        "status": "running",
        "seed": int(seed),
        "current_epoch": int(previous.get("current_epoch", 0)) if previous else 0,
        "total_epochs": int(config["epochs"]),
        "best_epoch": int(previous.get("best_epoch", 0)) if previous else 0,
        "selection_metric": config["selection_metric"],
        "best_val_value": previous.get("best_val_value") if previous else None,
        "started_at": started_at,
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "resume_count": resume_count,
        "device": str(device),
        "precision": precision,
        "effective_precision": effective,
        "scaler_state": scaler_state,
        "launch_command": launch_command,
        "resume_command": resume_command,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "code_sha256": code_tree_sha256(project_root),
        "data": data_report,
        "teacher": teacher_report,
        "teacher_cache": teacher_cache_report,
        "student_config": student_config,
        "loss_config": loss_config,
        "test_attempt_count": int(previous.get("test_attempt_count", 0)) if previous else 0,
        "test_evaluation_count": int(previous.get("test_evaluation_count", 0)) if previous else 0,
        "test_metrics": previous.get("test_metrics") if previous else None,
        "output_files": [
            "README.md", "run_status.json", "resolved_config.yaml", "provenance.json",
            "history.csv", "last_student.pth", "best_student.pth",
            "test_results.json", "test_targets.npy", "test_predictions.npy",
            "test_probabilities.npy",
        ],
    }


def write_run_readme(run_dir, status):
    test_metrics = status.get("test_metrics")
    if isinstance(test_metrics, Mapping):
        test_summary = "\n".join(
            f"- {label}（{key}）：{float(test_metrics[key]):.4f}"
            for key, label in (
                ("accuracy", "准确率"),
                ("balanced_accuracy", "平衡准确率"),
                ("macro_f1", "宏平均 F1"),
                ("weighted_f1", "加权 F1"),
                ("top3_accuracy", "Top-3 准确率"),
            )
        )
    else:
        test_summary = "尚未完成测试集评估。"
    content = f"""# 学生模型运行记录

运行状态：{status['status']}
随机种子：{status['seed']}
训练进度：{status['current_epoch']}/{status['total_epochs']}
最佳轮次：{status['best_epoch']}
验证集选模指标：{status['selection_metric']} = {status.get('best_val_value')}
设备：{status['device']}
精度模式：{status['precision']}（本机实际使用：{status['effective_precision']}）
GradScaler 状态：`{json.dumps(status['scaler_state'], ensure_ascii=False)}`

## 模型与损失

学生模型完整配置：

```json
{json.dumps(status['student_config'], ensure_ascii=False, indent=2)}
```

交叉熵与蒸馏损失配置：

```json
{json.dumps(status['loss_config'], ensure_ascii=False, indent=2)}
```

## 数据、教师与代码身份

- 配置文件：`{status['config']['path']}`
- 配置 SHA-256：`{status['config']['sha256']}`
- 数据包：`{status['data']['archive_path']}`
- 数据包 SHA-256：`{status['data']['archive_sha256']}`
- 教师 checkpoint：`{status['teacher']['checkpoint_path']}`
- 教师 SHA-256：`{status['teacher']['checkpoint_sha256']}`
- 源代码树 SHA-256：`{status['code_sha256']}`

## 启动与续训

本次启动命令：

```text
{status['launch_command']}
```

可复制的续训命令（`--resume` 表示从最近一轮 checkpoint 继续）：

```text
{status['resume_command']}
```

续训不检验哈希，直接读取可用检查点并恢复模型、优化器、学习率调度器与蒸馏带宽参数。若测试结果已经完整保存，续训命令只返回已有结果，不会再次评估测试集。

## 结果与文件

当前测试指标：

{test_summary}

- `best_student.pth`：验证集选出的最佳学生权重及训练状态。
- `last_student.pth`：最近完成 epoch 的断点，供续训使用。
- `history.csv`：每轮训练集与验证集损失和分类指标。
- `test_results.json`：测试集总体指标和逐类别分类报告。
- `test_targets.npy`、`test_predictions.npy`、`test_probabilities.npy`：测试标签、预测类别和类别概率。
- `provenance.json` 与 `run_status.json`：运行配置、数据/教师/代码哈希和进度。
"""
    (Path(run_dir) / "README.md").write_text(content, encoding="utf-8")


def update_run_status(run_dir, status_data, **updates):
    status_data.update(updates)
    status_data["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _write_json_atomic(Path(run_dir) / "run_status.json", status_data)
    write_run_readme(run_dir, status_data)
    return status_data


def _classification_report(targets, predictions, class_names):
    return classification_report(
        targets,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )


def train_one_student_seed(
    config, config_path, seed, device, data_report, label_map, label_map_path,
    teacher, teacher_report, run_dir, resume_checkpoint_path=None,
):
    set_global_seed(seed, deterministic=bool(config.get("deterministic", True)))
    teacher_cache, teacher_cache_report = open_teacher_target_cache(
        config, config_path, teacher_report, data_report, label_map_path, device
    )
    dataset_options = {}
    if teacher_cache is not None:
        dataset_options["return_index"] = True
    train_dataset = InMemoryNpzSkeletonDataset(
        archive_path=data_report["archive_path"], split="train",
        expected_frames=int(config["data"]["frames"]),
        num_points=int(config["data"]["num_points"]),
        **dataset_options,
    )
    val_dataset = InMemoryNpzSkeletonDataset(
        archive_path=data_report["archive_path"], split="val",
        expected_frames=int(config["data"]["frames"]),
        num_points=int(config["data"]["num_points"]),
        **dataset_options,
    )
    for split_name, dataset in (("train", train_dataset), ("val", val_dataset)):
        if int(dataset.labels.min()) < 0 or int(dataset.labels.max()) >= len(label_map):
            raise ValueError(f"{split_name} 划分的标签超出标签映射范围。")
    train_loader = make_loader(train_dataset, config, int(seed), True, device)
    val_loader = make_loader(val_dataset, config, int(seed) + 1, False, device)
    student, criterion, distillation, optimizer, scheduler, student_config, loss_config = (
        create_student_training_objects(config, device, len(label_map))
    )
    requested_precision = str(config["precision"])
    actual_precision = effective_precision(device, requested_precision)
    scaler = create_grad_scaler(device, requested_precision)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    selection_metric = config["selection_metric"]
    history_path = run_dir / "history.csv"
    checkpoint_path = run_dir / "best_student.pth"
    start_epoch = 1
    best_value = float("-inf")
    best_epoch = 0
    if resume_checkpoint_path is not None:
        readable_only = resume_readable_only()
        checkpoint = load_resume_checkpoint(
            resume_checkpoint_path, config, config_path, seed, student_config,
            loss_config, teacher_report, data_report, device, scaler,
            teacher_cache_report=teacher_cache_report,
        )
        student.load_state_dict(checkpoint["student_state_dict"], strict=True)
        if "distillation_state_dict" in checkpoint:
            distillation.load_state_dict(checkpoint["distillation_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "train_loader_generator_state" in checkpoint:
            try:
                train_loader.generator.set_state(checkpoint["train_loader_generator_state"].cpu())
            except (AttributeError, RuntimeError, TypeError) as error:
                raise ValueError(f"无法恢复训练 DataLoader 的随机状态：{error}") from error
        elif not readable_only:
            raise ValueError("断点缺少训练 DataLoader 随机状态。")
        if "rng_state" in checkpoint:
            restore_rng_state(checkpoint["rng_state"], device)
        elif not readable_only:
            raise ValueError("断点缺少 RNG 状态。")
        start_epoch = int(checkpoint["epoch"]) + 1
        best_value = float(checkpoint["best_value"])
        best_epoch = int(checkpoint["best_epoch"])
        if not history_path.is_file():
            raise FileNotFoundError(f"续训 run 缺少 history.csv：{history_path}")
        status_path = run_dir / "run_status.json"
        resume_compatibility = None
        if not readable_only:
            provenance_path = run_dir / "provenance.json"
            if not provenance_path.is_file():
                raise FileNotFoundError(f"续训 run 缺少 provenance.json：{provenance_path}")
            with provenance_path.open("r", encoding="utf-8") as provenance_file:
                saved_provenance = json.load(provenance_file)
            if (
                saved_provenance.get("config_sha256") != file_sha256(config_path)
                or saved_provenance.get("teacher", {}).get("checkpoint_sha256")
                != teacher_report["checkpoint_sha256"]
                or not _teacher_cache_identity_matches(
                    saved_provenance.get("teacher_cache"), teacher_cache_report
                )
            ):
                raise ValueError("续训 run 的 provenance 与当前配置、教师 checkpoint 或缓存不一致。")
            if not status_path.is_file():
                raise FileNotFoundError(f"续训 run 缺少 run_status.json：{status_path}")
        saved_status = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.is_file() else {}
        )
        if not readable_only:
            current_code_hash = code_tree_sha256(Path(__file__).resolve().parents[3])
            if (
                saved_status.get("config", {}).get("sha256") != file_sha256(config_path)
                or saved_status.get("teacher", {}).get("checkpoint_sha256")
                != teacher_report["checkpoint_sha256"]
                or saved_status.get("data", {}).get("archive_sha256")
                != data_report["archive_sha256"]
                or not _teacher_cache_identity_matches(
                    saved_status.get("teacher_cache"), teacher_cache_report
                )
            ):
                raise ValueError("续训 run_status.json 与当前配置、数据包、教师 checkpoint 或缓存不一致。")
            resume_compatibility = verify_resume_code_compatibility(
                run_dir,
                resume_checkpoint_path,
                saved_status.get("code_sha256"),
                current_code_hash,
            )
        _reconcile_resume_progress(run_dir, checkpoint, saved_status)
        run_status = _build_run_status(
            run_dir, config, config_path, seed, device, data_report, teacher_report,
            student_config, loss_config, requested_precision, actual_precision,
            scaler.state_dict(), previous=saved_status,
            teacher_cache_report=teacher_cache_report,
        )
        run_status["current_epoch"] = int(checkpoint["epoch"])
        run_status["best_epoch"] = int(checkpoint["best_epoch"])
        run_status["best_val_value"] = (
            float(checkpoint["best_value"])
            if math.isfinite(float(checkpoint["best_value"])) else None
        )
        compatibility_events = list(saved_status.get("resume_compatibility_events", []))
        if resume_compatibility is not None:
            compatibility_events.append({
                **resume_compatibility,
                "accepted_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            })
        if compatibility_events:
            run_status["resume_compatibility_events"] = compatibility_events
        update_run_status(run_dir, run_status)
    else:
        write_student_provenance(
            run_dir, config, config_path, seed, device, data_report, teacher_report,
            label_map_path, student_config, loss_config, requested_precision,
            actual_precision, scaler.state_dict(),
            teacher_cache_report=teacher_cache_report,
        )
        run_status = _build_run_status(
            run_dir, config, config_path, seed, device, data_report, teacher_report,
            student_config, loss_config, requested_precision, actual_precision,
            scaler.state_dict(), teacher_cache_report=teacher_cache_report,
        )
        update_run_status(run_dir, run_status)

    metric_columns = (
        "loss", "cross_entropy_loss", "logit_kd_loss", "static_topology_loss",
        "dynamic_topology_loss", "accuracy", "balanced_accuracy", "macro_f1",
        "weighted_f1", "top3_accuracy",
    )
    columns = ["epoch", "learning_rate"] + [f"train_{key}" for key in metric_columns] + [
        f"val_{key}" for key in metric_columns
    ]
    history_mode = "a" if resume_checkpoint_path is not None else "w"
    with history_path.open(history_mode, newline="", encoding="utf-8-sig") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=columns)
        if resume_checkpoint_path is None:
            writer.writeheader()
            history_file.flush()
            initial_checkpoint = {
                "student_state_dict": student.state_dict(),
                "distillation_state_dict": distillation.state_dict(),
                "student_config": student_config,
                "seed": int(seed),
                "epoch": 0,
                "selection_metric": selection_metric,
                "selection_value": float("-inf"),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss_config": loss_config,
                "teacher_checkpoint_path": teacher_report["checkpoint_path"],
                "teacher_checkpoint_sha256": teacher_report["checkpoint_sha256"],
                "rng_state": capture_rng_state(),
                "val_metrics": {},
                "optimizer_config": dict(config["optimizer"]),
                "scheduler_config": dict(config["scheduler"]),
                "config_sha256": file_sha256(config_path),
                "data_archive_sha256": data_report["archive_sha256"],
                "teacher_cache_manifest_sha256": (
                    teacher_cache_report["manifest_sha256"] if teacher_cache_report else None
                ),
                "best_value": float("-inf"),
                "best_epoch": 0,
                "train_loader_generator_state": train_loader.generator.get_state().cpu(),
                "training_device_type": device.type,
                "precision": requested_precision,
                "effective_precision": actual_precision,
                "scaler_state": scaler.state_dict(),
                "test_evaluation_count": 0,
                "checkpoint_kind": "last",
            }
            _save_checkpoint_atomic(run_dir / "last_student.pth", initial_checkpoint)
        for epoch in range(start_epoch, int(config["epochs"]) + 1):
            learning_rate = float(optimizer.param_groups[0]["lr"])
            train_metrics, _, _, _ = run_student_epoch(
                student, teacher, train_loader, criterion, distillation, device,
                split="train", optimizer=optimizer, keep_probabilities=False,
                precision=requested_precision, scaler=scaler,
                teacher_cache=teacher_cache,
                progress_description=f"学生 种子{seed} 第{epoch}/{config['epochs']}轮 训练",
            )
            val_metrics, _, _, _ = run_student_epoch(
                student, teacher, val_loader, criterion, distillation, device,
                split="val", optimizer=None, keep_probabilities=False,
                precision=requested_precision, scaler=scaler,
                teacher_cache=teacher_cache,
                progress_description=f"学生 种子{seed} 第{epoch}/{config['epochs']}轮 验证",
            )
            writer.writerow({
                "epoch": epoch,
                "learning_rate": learning_rate,
                **{f"train_{key}": value for key, value in train_metrics.items() if key in metric_columns},
                **{f"val_{key}": value for key, value in val_metrics.items() if key in metric_columns},
            })
            history_file.flush()

            current_value = float(val_metrics[selection_metric])
            improved = current_value > best_value
            if improved:
                best_value = current_value
                best_epoch = epoch
            scheduler.step()
            checkpoint = {
                "student_state_dict": student.state_dict(),
                "distillation_state_dict": distillation.state_dict(),
                "student_config": student_config,
                "seed": int(seed),
                "epoch": int(epoch),
                "selection_metric": selection_metric,
                "selection_value": current_value,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss_config": loss_config,
                "teacher_checkpoint_path": teacher_report["checkpoint_path"],
                "teacher_checkpoint_sha256": teacher_report["checkpoint_sha256"],
                "rng_state": capture_rng_state(),
                "val_metrics": val_metrics,
                "optimizer_config": dict(config["optimizer"]),
                "scheduler_config": dict(config["scheduler"]),
                "config_sha256": file_sha256(config_path),
                "data_archive_sha256": data_report["archive_sha256"],
                "teacher_cache_manifest_sha256": (
                    teacher_cache_report["manifest_sha256"] if teacher_cache_report else None
                ),
                "best_value": best_value,
                "best_epoch": best_epoch,
                "train_loader_generator_state": train_loader.generator.get_state().cpu(),
                "training_device_type": device.type,
                "precision": requested_precision,
                "effective_precision": actual_precision,
                "scaler_state": scaler.state_dict(),
                "test_evaluation_count": 0,
            }
            if improved:
                best_checkpoint = dict(checkpoint)
                best_checkpoint["checkpoint_kind"] = "best"
                _save_checkpoint_atomic(checkpoint_path, best_checkpoint)
            last_checkpoint = dict(checkpoint)
            last_checkpoint["checkpoint_kind"] = "last"
            _save_checkpoint_atomic(run_dir / "last_student.pth", last_checkpoint)
            if config.get("save_epoch_weights", False):
                from .epoch_weights import save_epoch_weights
                save_epoch_weights(run_dir, epoch, {
                    "student_state_dict": student.state_dict(), "student_config": student_config,
                    "distillation_state_dict": distillation.state_dict(), "seed": int(seed),
                    "learning_rate": learning_rate, "train_metrics": train_metrics,
                    "val_metrics": val_metrics, "selection_metric": selection_metric,
                })
            update_run_status(
                run_dir,
                run_status,
                status="running",
                current_epoch=epoch,
                best_epoch=best_epoch,
                best_val_value=best_value,
                precision=requested_precision,
                effective_precision=actual_precision,
                scaler_state=scaler.state_dict(),
            )
            print(
                format_student_epoch_result(
                    seed, epoch, learning_rate, train_metrics, val_metrics
                ),
                flush=True,
            )
    # 最佳模型只能来自其对应 epoch 的完整 best/backup 断点；不得用
    # 更晚的 last 权重冒充验证集选出的模型。
    _, checkpoint = _load_best_student_checkpoint(
        run_dir, checkpoint, map_location=device
    )
    expected_cache_manifest_sha = (
        teacher_cache_report["manifest_sha256"] if teacher_cache_report else None
    )
    if (
        not resume_readable_only()
        and checkpoint["teacher_cache_manifest_sha256"] != expected_cache_manifest_sha
    ):
        raise ValueError("最佳学生 checkpoint 的缓存 manifest SHA-256 与当前缓存不一致。")
    student.load_state_dict(checkpoint["student_state_dict"], strict=True)
    if "distillation_state_dict" in checkpoint:
        distillation.load_state_dict(checkpoint["distillation_state_dict"], strict=True)
    student.eval()
    del train_loader, val_loader, train_dataset, val_dataset
    if teacher_cache is not None:
        teacher_cache.close()
        teacher_cache = None

    test_results_path = run_dir / "test_results.json"
    test_started_path = run_dir / "test_evaluation_started.json"
    if test_results_path.is_file():
        existing = json.loads(test_results_path.read_text(encoding="utf-8"))
        if (
            int(existing.get("test_evaluation_count", 0)) != 1
            or int(existing.get("seed", -1)) != int(seed)
            or not all(
                (run_dir / name).is_file()
                for name in ("test_targets.npy", "test_predictions.npy", "test_probabilities.npy")
            )
        ):
            raise ValueError("既有 test_results.json 的身份校验失败；不会重复读取 test。")
        if not resume_readable_only() and (
            existing.get("config_sha256") != file_sha256(config_path)
            or existing.get("teacher_checkpoint_sha256") != teacher_report["checkpoint_sha256"]
            or existing.get("teacher_cache_manifest_sha256")
            != (teacher_cache_report["manifest_sha256"] if teacher_cache_report else None)
        ):
            raise ValueError("既有 test_results.json 的 SHA-256 校验失败；不会重复读取 test。")
        if test_started_path.is_file():
            marker = json.loads(test_started_path.read_text(encoding="utf-8"))
            marker["status"] = "complete"
            marker["test_evaluation_count"] = 1
            _write_json_atomic(test_started_path, marker)
        update_run_status(
            run_dir,
            run_status,
            status="complete",
            current_epoch=int(run_status["total_epochs"]),
            best_epoch=int(existing["checkpoint_epoch"]),
            best_val_value=float(existing["selection_value"]),
            test_attempt_count=int(existing.get("test_attempt_count", 1)),
            test_evaluation_count=1,
            test_metrics=existing["test_metrics"],
            output_files=sorted(
                item.name for item in Path(run_dir).iterdir() if item.is_file()
            ),
        )
        return existing
    if int(checkpoint.get("test_evaluation_count", 0)) == 1:
        raise RuntimeError(
            "checkpoint 记录 test 已完成一次但 test_results.json 缺失；为避免二次评估，拒绝重测。"
        )
    test_attempt_count = 1
    if test_started_path.exists():
        marker = json.loads(test_started_path.read_text(encoding="utf-8"))
        if (
            int(marker.get("seed", -1)) != int(seed)
            or (
                not resume_readable_only()
                and (
                    marker.get("config_sha256") != file_sha256(config_path)
                    or marker.get("teacher_checkpoint_sha256")
                    != teacher_report["checkpoint_sha256"]
                )
            )
        ):
            raise ValueError("未完成 test 标记与当前 run 身份不一致；拒绝重用该目录。")
        if marker.get("status") == "complete" or int(marker.get("test_evaluation_count", 0)) == 1:
            raise RuntimeError(
                "test 标记显示评估已完成但 test_results.json 缺失；为避免第二次测试，拒绝重跑。"
            )
        test_attempt_count = int(marker.get("test_attempt_count", 1)) + 1
    _write_json_atomic(test_started_path, {
        "seed": int(seed),
        "epoch": int(checkpoint["epoch"]),
        "config_sha256": file_sha256(config_path),
        "teacher_checkpoint_sha256": teacher_report["checkpoint_sha256"],
        "test_attempt_count": test_attempt_count,
        "test_evaluation_count": 0,
        "status": "in_progress",
    })
    update_run_status(
        run_dir,
        run_status,
        status="test_in_progress",
        current_epoch=int(config["epochs"]),
        best_epoch=int(checkpoint["epoch"]),
        best_val_value=float(checkpoint["selection_value"]),
        test_attempt_count=test_attempt_count,
        test_evaluation_count=0,
    )

    test_dataset = InMemoryNpzSkeletonDataset(
        archive_path=data_report["archive_path"], split="test",
        expected_frames=int(config["data"]["frames"]),
        num_points=int(config["data"]["num_points"]),
    )
    if int(test_dataset.labels.min()) < 0 or int(test_dataset.labels.max()) >= len(label_map):
        raise ValueError("test 划分的标签超出标签映射范围。")
    test_loader = make_loader(test_dataset, config, int(seed) + 2, False, device)
    test_metrics, targets, predictions, probabilities = run_student_epoch(
        student, teacher, test_loader, criterion, distillation, device,
        split="test", optimizer=None, keep_probabilities=True,
        precision=requested_precision, scaler=scaler,
        progress_description=f"学生 种子{seed} 测试",
    )
    class_names = [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]
    report = _classification_report(targets, predictions, class_names)
    result = {
        "seed": int(seed),
        "run_dir": str(run_dir),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "selection_metric": checkpoint["selection_metric"],
        "selection_value": float(checkpoint["selection_value"]),
        "val_metrics": checkpoint["val_metrics"],
        "test_metrics": test_metrics,
        "per_class": report,
        "test_attempt_count": test_attempt_count,
        "test_evaluation_count": 1,
        "config_sha256": file_sha256(config_path),
        "teacher_checkpoint_sha256": teacher_report["checkpoint_sha256"],
        "teacher_cache_manifest_sha256": (
            teacher_cache_report["manifest_sha256"] if teacher_cache_report else None
        ),
    }
    np.save(run_dir / "test_targets.npy", targets)
    np.save(run_dir / "test_predictions.npy", predictions)
    np.save(run_dir / "test_probabilities.npy", probabilities)
    _write_json_atomic(test_results_path, result)
    completion_marker = json.loads(test_started_path.read_text(encoding="utf-8"))
    completion_marker["status"] = "complete"
    completion_marker["test_evaluation_count"] = 1
    completion_marker["test_metrics"] = test_metrics
    _write_json_atomic(test_started_path, completion_marker)
    for marker_checkpoint_path in (checkpoint_path, run_dir / "last_student.pth"):
        if marker_checkpoint_path.is_file():
            marker_checkpoint = _torch_load_checkpoint(marker_checkpoint_path, map_location="cpu")
            marker_checkpoint["test_evaluation_count"] = 1
            _save_checkpoint_atomic(marker_checkpoint_path, marker_checkpoint)
    update_run_status(
        run_dir,
        run_status,
        status="complete",
        current_epoch=int(config["epochs"]),
        best_epoch=int(checkpoint["epoch"]),
        best_val_value=float(checkpoint["selection_value"]),
        test_attempt_count=test_attempt_count,
        test_evaluation_count=1,
        test_metrics=test_metrics,
        output_files=sorted(
            item.name for item in Path(run_dir).iterdir() if item.is_file()
        ),
    )
    print(f"完成学生模型 seed={seed}，结果目录：{run_dir}")
    return result


def aggregate_seed_results(results):
    if not results:
        raise ValueError("不能汇总空的种子结果。")
    names = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "top3_accuracy")
    summary = {
        "num_seeds": len(results),
        "seeds": [int(item["seed"]) for item in results],
        "runs": [item["run_dir"] for item in results],
        "test_evaluation_count_per_seed": [int(item["test_evaluation_count"]) for item in results],
        "test_metrics": {},
    }
    for name in names:
        values = [float(item["test_metrics"][name]) for item in results]
        summary["test_metrics"][name] = {
            "values": values,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary


def check_student_setup(config, config_path, device, label_map, label_map_path, data_report):
    teacher, teacher_report = load_strict_teacher(
        config, config_path, label_map, label_map_path, data_report, device
    )
    train_dataset = InMemoryNpzSkeletonDataset(
        archive_path=data_report["archive_path"], split="train",
        expected_frames=int(config["data"]["frames"]),
        num_points=int(config["data"]["num_points"]),
    )
    val_dataset = InMemoryNpzSkeletonDataset(
        archive_path=data_report["archive_path"], split="val",
        expected_frames=int(config["data"]["frames"]),
        num_points=int(config["data"]["num_points"]),
    )
    student, criterion, distillation, _, _, student_config, _ = create_student_training_objects(
        config, device, len(label_map)
    )
    joints, bones, target = train_dataset[0]
    joints = joints.unsqueeze(0).to(device)
    bones = bones.unsqueeze(0).to(device)
    student.eval()
    teacher.eval()
    with torch.inference_mode():
        teacher_logits, teacher_features = teacher(joints, bones, return_features=True)
        student_logits, student_features = student(joints, bones, return_features=True)
        student_features = rename_student_features(student_features)
        ce = criterion(student_logits, torch.tensor([target], dtype=torch.long, device=device))
        components = distillation(
            student_logits, teacher_logits, student_features, teacher_features,
            physical_adjacency=student.physical_adjacency,
        )
    report = {
        "device": str(device),
        "precision": config["precision"],
        "effective_precision": effective_precision(device, config["precision"]),
        "scaler_state": create_grad_scaler(device, config["precision"]).state_dict(),
        "train_count": len(train_dataset),
        "val_count": len(val_dataset),
        "test_arrays_constructed": False,
        "student_config": student_config,
        "student_trainable_parameters": student.count_params(),
        "teacher_trainable_parameters": sum(p.numel() for p in teacher.parameters() if p.requires_grad),
        "student_logits_shape": list(student_logits.shape),
        "teacher_logits_shape": list(teacher_logits.shape),
        "finite_cross_entropy": bool(torch.isfinite(ce).item()),
        "finite_distillation": all(bool(torch.isfinite(value).all()) for value in components.values()),
        "teacher_checkpoint": teacher_report,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="训练 Lite-MyoNet 蒸馏学生模型。")
    parser.add_argument("--config", default="configs/student.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument(
        "--resume", metavar="CHECKPOINT_OR_RUN_DIR",
        help="从 last_student.pth 或其 run 目录恢复；同一 run 的测试集不会重复评估。",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="只检查教师、训练/验证数据和一次蒸馏前向，不读取 test 划分。",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.resume:
        os.environ["LITE_MYONET_RESUME_READABLE_ONLY"] = "1"
    configure_torch_cpu_threads_from_environment()
    config, config_path = load_config(args.config)
    seeds_from_config = validate_student_config(config)
    label_map, label_map_path = load_label_map(config, config_path)
    if len(label_map) < 3:
        raise ValueError("Top-3 指标要求至少包含三个类别。")
    if len(label_map) != 33:
        raise ValueError(f"合成数据正式协议要求 33 类，实际为 {len(label_map)} 类。")
    data_report = verify_student_archive(config, config_path)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 设备，但当前进程不可用。")
    teacher, teacher_report = load_strict_teacher(
        config, config_path, label_map, label_map_path, data_report, device
    )
    if args.check_only:
        check_student_setup(config, config_path, device, label_map, label_map_path, data_report)
        return

    if args.resume and args.seeds:
        raise ValueError("--resume 已指定一个固定种子 run，不能同时传入 --seeds。")
    seeds = [int(seed) for seed in args.seeds] if args.seeds else seeds_from_config
    if not seeds:
        raise ValueError("至少需要一个随机种子。")
    if len(seeds) != len(set(seeds)):
        raise ValueError("命令行随机种子存在重复值。")
    resume_run_dir = None
    resume_checkpoint_path = None
    if args.resume:
        resume_run_dir, resume_checkpoint_path = resolve_resume_target(args.resume)
        checkpoint_probe = _torch_load_checkpoint(resume_checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint_probe, Mapping) or "seed" not in checkpoint_probe:
            raise ValueError("续训 checkpoint 不含有效 seed。")
        seeds = [int(checkpoint_probe["seed"])]
        if (resume_run_dir / "test_results.json").is_file():
            existing = json.loads((resume_run_dir / "test_results.json").read_text(encoding="utf-8"))
            if (
                int(existing.get("test_evaluation_count", 0)) != 1
                or int(existing.get("seed", -1)) != int(seeds[0])
                or not all(
                    (resume_run_dir / name).is_file()
                    for name in ("test_targets.npy", "test_predictions.npy", "test_probabilities.npy")
                )
            ):
                raise ValueError("已完成 run 的测试结果身份校验失败。")
            if not resume_readable_only() and (
                existing.get("config_sha256") != file_sha256(config_path)
                or existing.get("teacher_checkpoint_sha256") != teacher_report["checkpoint_sha256"]
            ):
                raise ValueError("已完成 run 的测试结果 SHA-256 校验失败。")
            marker_path = resume_run_dir / "test_evaluation_started.json"
            if marker_path.is_file():
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                if (
                    int(marker.get("seed", -1)) != int(seeds[0])
                    or (
                        not resume_readable_only()
                        and (
                            marker.get("config_sha256") != file_sha256(config_path)
                            or marker.get("teacher_checkpoint_sha256")
                            != teacher_report["checkpoint_sha256"]
                        )
                    )
                ):
                    raise ValueError("test 标记与已保存测试结果身份不一致。")
                marker["status"] = "complete"
                marker["test_evaluation_count"] = 1
                marker["test_metrics"] = existing["test_metrics"]
                _write_json_atomic(marker_path, marker)
            status_path = resume_run_dir / "run_status.json"
            if status_path.is_file():
                saved_status = json.loads(status_path.read_text(encoding="utf-8"))
                update_run_status(
                    resume_run_dir,
                    saved_status,
                    status="complete",
                    current_epoch=int(saved_status.get("total_epochs", existing["checkpoint_epoch"])),
                    best_epoch=int(existing["checkpoint_epoch"]),
                    best_val_value=float(existing["selection_value"]),
                    test_attempt_count=int(existing.get("test_attempt_count", 1)),
                    test_evaluation_count=1,
                    test_metrics=existing["test_metrics"],
                    output_files=sorted(
                        item.name for item in resume_run_dir.iterdir() if item.is_file()
                    ),
                )
            print("该 run 已完成唯一一次 test 评估，直接返回已保存结果，不会重复测试。")
            print(json.dumps(existing, ensure_ascii=False, indent=2))
            return
        marker_path = resume_run_dir / "test_evaluation_started.json"
        status_path = resume_run_dir / "run_status.json"
        saved_status = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.is_file() else {}
        )
        checkpoint_test_count = int(checkpoint_probe.get("test_evaluation_count", 0))
        status_test_count = int(saved_status.get("test_evaluation_count", 0))
        marker = None
        if marker_path.exists():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if (
                int(marker.get("seed", -1)) != int(seeds[0])
                or (
                    not resume_readable_only()
                    and (
                        marker.get("config_sha256") != file_sha256(config_path)
                        or marker.get("teacher_checkpoint_sha256")
                        != teacher_report["checkpoint_sha256"]
                    )
                )
            ):
                raise ValueError("未完成 test 标记与当前 run 身份不一致；拒绝续训。")
        marker_test_count = int(marker.get("test_evaluation_count", 0)) if marker else 0
        if any(count not in (0, 1) for count in (
            checkpoint_test_count, status_test_count, marker_test_count
        )):
            raise ValueError("checkpoint、run_status 与 test 标记中的 test_evaluation_count 无效。")
        if (
            checkpoint_test_count == 1
            or status_test_count == 1
            or marker_test_count == 1
            or (marker and marker.get("status") == "complete")
        ):
            raise RuntimeError(
                "checkpoint、run_status 或 test 标记显示评估已完成，但 test_results.json 缺失；"
                "为避免重复测试，拒绝续训。"
            )
        experiment_dir = resume_run_dir.parent
    else:
        experiment_dir = prepare_experiment_directory(config, config_path)
    results = []
    for seed in seeds:
        run_dir = resume_run_dir if resume_run_dir is not None else experiment_dir / f"seed_{int(seed)}"
        if resume_run_dir is None:
            run_dir.mkdir(parents=True, exist_ok=False)
        results.append(
            train_one_student_seed(
                config, config_path, int(seed), device, data_report, label_map,
                label_map_path, teacher, teacher_report, run_dir,
                resume_checkpoint_path=resume_checkpoint_path,
            )
        )
    summary = aggregate_seed_results(results)
    summary_path = experiment_dir / "multiseed_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"多种子汇总：{summary_path}")


if __name__ == "__main__":
    main()


__all__ = [
    "aggregate_seed_results",
    "calculate_student_metrics",
    "capture_rng_state",
    "create_student_config",
    "create_student_training_objects",
    "load_strict_teacher",
    "rename_student_features",
    "run_student_epoch",
    "train_one_student_seed",
    "validate_student_config",
    "verify_student_archive",
]
