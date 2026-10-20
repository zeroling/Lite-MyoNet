"""Strict, row-indexed float32 cache of teacher distillation targets.

Only the train and validation splits are supported. Test data is deliberately
not opened by this module.
"""

import hashlib
import json
import os
import shutil
import uuid
import datetime as dt
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..data.manifest import file_sha256
from ..data.packed import InMemoryNpzSkeletonDataset, load_archive_metadata
from ..models import topology
from ..runtime import resume_readable_only
from .distillation import FEATURE_MATCHES, TEACHER_FEATURE_NAMES


CACHE_SCHEMA_VERSION = 3
SPLITS = ("train", "val")
HASH_FIELDS = (
    "teacher_checkpoint_sha256",
    "archive_sha256",
    "label_map_sha256",
    "requested_precision",
    "effective_precision",
    "teacher_model_config_sha256",
    "loss_config_sha256",
    "code_sha256",
)
IDENTITY_FIELDS = HASH_FIELDS + ("generation_batch_size",)
ARRAY_NAMES = ("teacher_logits", "static_targets", "dynamic_targets", "row_indices")


def canonical_json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def object_sha256(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def cache_loss_config(loss_config):
    """Return the loss parameters that alter cached target contents."""
    if not isinstance(loss_config, Mapping):
        raise TypeError("loss_config 必须是映射。")
    required = ("kd_temperature", "topology_temperature")
    missing = [key for key in required if key not in loss_config]
    if missing:
        raise ValueError("缓存目标损失配置缺少字段：{}。".format(missing))
    return {key: float(loss_config[key]) for key in required}


def code_tree_sha256(project_root):
    """Use the same code-tree hash contract as training provenance."""
    from .teacher_trainer import code_tree_sha256 as project_code_hash

    return project_code_hash(project_root)


def physical_edge_indices(num_points, edges=None):
    """Return row-major indices matching DistillationLoss._physical_edge_mask."""
    edges = topology.bones if edges is None else edges
    mask = np.zeros((int(num_points), int(num_points)), dtype=np.bool_)
    for source, target in edges:
        mask[int(source), int(target)] = True
        mask[int(target), int(source)] = True
    np.fill_diagonal(mask, False)
    return np.flatnonzero(mask.reshape(-1)).astype(np.int64)


def _normalize_hashes(values):
    if not isinstance(values, Mapping):
        raise TypeError("expected_hashes 必须是映射。")
    missing = [name for name in IDENTITY_FIELDS if name not in values]
    if missing:
        raise ValueError("缓存身份缺少字段：{}。".format(missing))
    result = {}
    for name in HASH_FIELDS:
        value = str(values[name]).lower()
        if name.endswith("_sha256") and (
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("{} 必须是 64 位 SHA-256。".format(name))
        result[name] = value
    batch_size = values["generation_batch_size"]
    if (isinstance(batch_size, bool)
            or not isinstance(batch_size, (int, np.integer))
            or int(batch_size) <= 0):
        raise ValueError("generation_batch_size 必须是正整数。")
    result["generation_batch_size"] = int(batch_size)
    return result


def make_cache_identity(
    *, teacher_checkpoint_sha256, archive_sha256, label_map_sha256,
    requested_precision, effective_precision, teacher_model_config,
    loss_config, code_sha256, generation_batch_size,
):
    """Create the exact identity contract consumed by builder and reader."""
    if (isinstance(generation_batch_size, bool)
            or not isinstance(generation_batch_size, (int, np.integer))
            or int(generation_batch_size) <= 0):
        raise ValueError("generation_batch_size 必须为正整数。")
    return {
        "teacher_checkpoint_sha256": str(teacher_checkpoint_sha256).lower(),
        "archive_sha256": str(archive_sha256).lower(),
        "label_map_sha256": str(label_map_sha256).lower(),
        "requested_precision": str(requested_precision),
        "effective_precision": str(effective_precision),
        "teacher_model_config_sha256": object_sha256(teacher_model_config),
        "loss_config_sha256": object_sha256(cache_loss_config(loss_config)),
        "code_sha256": str(code_sha256).lower(),
        "generation_batch_size": int(generation_batch_size),
    }


def _feature_tensor(features, name):
    value = features[name] if isinstance(features, Mapping) else features[TEACHER_FEATURE_NAMES.index(name)]
    if value.ndim != 4 or not value.is_floating_point():
        raise ValueError("教师特征 {} 必须是浮点 [B,C,T,V] 张量。".format(name))
    return value.detach().to(dtype=torch.float32)


def compress_teacher_targets(
    teacher_logits, teacher_features, *, topology_temperature=0.2,
    num_points=23, edges=None,
):
    """Compress one teacher batch into the exact values used by the loss.

    Static values are the row-major physical-edge cosine affinities. Dynamic
    values are the teacher softmax distributions after the configured
    temperature. All outputs are float32 and detached.
    """
    temperature = float(topology_temperature)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("topology_temperature 必须是有限正数。")
    if teacher_logits.ndim != 2 or not teacher_logits.is_floating_point():
        raise ValueError("teacher_logits 必须是浮点 [B,K] 张量。")
    logits = teacher_logits.detach().to(dtype=torch.float32)
    from .distillation import topology_squared_distances
    static_values=[]
    dynamic_values=[]
    for match in FEATURE_MATCHES:
        feature=_feature_tensor(teacher_features,match.teacher)
        static_values.append(topology_squared_distances(feature).flatten(1))
        dynamic_values.append(topology_squared_distances(feature,dynamic=True))

    return {
        "teacher_logits": logits.contiguous(),
        "static_targets": torch.stack(static_values, dim=1).contiguous(),
        "dynamic_targets": torch.stack(dynamic_values, dim=1).contiguous(),
    }


def _sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _source_sample_ids(archive_path, split):
    with np.load(archive_path, allow_pickle=False) as archive:
        key = "{}_sample_ids".format(split)
        if key not in archive:
            raise ValueError("数据包缺少行对齐字段 {}。".format(key))
        ids = [str(value) for value in archive[key].tolist()]
    return ids


class IndexedDataset(Dataset):
    """Add the source split row index to each sample without changing its order."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        joints, bones, label = self.dataset[int(index)]
        return joints, bones, label, int(index)


class IndexedTargetDataset(Dataset):
    """Index-bearing dataset wrapper for training with shuffled DataLoaders."""

    def __init__(self, dataset, cache_split):
        if len(dataset) != cache_split.count:
            raise ValueError("数据集长度与教师目标缓存行数不一致。")
        self.dataset = dataset
        self.cache_split = cache_split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        joints, bones, label = self.dataset[int(index)]
        return joints, bones, label, int(index)


class _SplitCache:
    def __init__(self, directory, split_metadata, verify_hashes=True):
        if not isinstance(split_metadata, Mapping):
            raise ValueError("缓存划分 metadata 必须是映射。")
        self.directory = Path(directory)
        raw_count = split_metadata.get("count")
        if (isinstance(raw_count, bool)
                or not isinstance(raw_count, (int, np.integer))
                or int(raw_count) <= 0):
            raise ValueError("缓存划分 count 必须是正整数。")
        self.count = int(raw_count)
        expected_shapes = split_metadata["shapes"]
        files_metadata = split_metadata["files"]
        if not isinstance(expected_shapes, Mapping) or not isinstance(files_metadata, Mapping):
            raise ValueError("缓存划分 shapes/files metadata 必须是映射。")
        if not set(ARRAY_NAMES).issubset(expected_shapes) or not set(ARRAY_NAMES).issubset(files_metadata):
            raise ValueError("缓存划分缺少必需数组的 shape/files metadata。")
        readable_only = resume_readable_only()
        self.arrays = {}
        for name in ARRAY_NAMES:
            path = self.directory / (name + ".npy")
            info = files_metadata[name]
            if not isinstance(info, Mapping) or "bytes" not in info or "sha256" not in info:
                raise ValueError("缓存数组 {} 的 files metadata 不完整。".format(name))
            if (isinstance(info["bytes"], bool)
                    or not isinstance(info["bytes"], (int, np.integer))
                    or int(info["bytes"]) <= 0
                    or not isinstance(info["sha256"], str)):
                raise ValueError("缓存数组 {} 的 files metadata 类型错误。".format(name))
            shape = expected_shapes[name]
            if (not isinstance(shape, (list, tuple)) or not shape
                    or any(isinstance(value, bool)
                           or not isinstance(value, (int, np.integer))
                           or int(value) <= 0 for value in shape)):
                raise ValueError("缓存数组 {} 的 shape metadata 不合法。".format(name))
            if not path.is_file():
                raise ValueError("缓存数组缺失：{}。".format(path))
            if not readable_only and path.stat().st_size != int(info["bytes"]):
                raise ValueError("缓存数组缺失或大小不符：{}。".format(path))
            if verify_hashes and not readable_only and _sha256_file(path) != info["sha256"]:
                raise ValueError("缓存数组 SHA-256 不匹配：{}。".format(path))
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            expected_dtype = np.int64 if name == "row_indices" else np.float32
            if list(array.shape) != list(shape) or array.dtype != expected_dtype:
                raise ValueError("缓存数组 shape/dtype 不匹配：{}。".format(path))
            if array.shape[0] != self.count:
                raise ValueError("缓存数组首维与划分 count 不一致：{}。".format(path))
            if name == "teacher_logits" and (array.ndim != 2 or array.shape[1] <= 0):
                raise ValueError("缓存 teacher_logits 必须是 [N,K] 且 K>0。")
            if name == "static_targets" and (
                array.ndim != 3 or array.shape[1] != len(FEATURE_MATCHES) or array.shape[2] <= 0
            ):
                raise ValueError("缓存 static_targets 的形状不合理。")
            if name == "dynamic_targets" and (
                array.ndim != 4
                or array.shape[1] != len(FEATURE_MATCHES)
                or array.shape[2] <= 0
                or array.shape[2] != array.shape[3]
            ):
                raise ValueError("缓存 dynamic_targets 的形状不合理。")
            if name == "row_indices" and array.shape != (self.count,):
                raise ValueError("缓存 row_indices 长度与 count 不一致。")
            self.arrays[name] = array
        if not np.array_equal(self.arrays["row_indices"], np.arange(self.count, dtype=np.int64)):
            raise ValueError("缓存 row_indices 不是严格的本地数据行序。")

    @property
    def logits(self):
        return self.arrays["teacher_logits"]

    @property
    def static(self):
        return self.arrays["static_targets"]

    @property
    def dynamic(self):
        return self.arrays["dynamic_targets"]

    def get(self, row_indices, device=None, fields=None):
        indices = np.asarray(row_indices, dtype=np.int64).reshape(-1)
        if len(indices) == 0 or int(indices.min()) < 0 or int(indices.max()) >= self.count:
            raise IndexError("请求的缓存行索引越界或为空。")
        target_device = torch.device("cpu" if device is None else device)
        names = ("teacher_logits", "static_targets", "dynamic_targets") if fields is None else tuple(fields)
        unknown = sorted(set(names) - {"teacher_logits", "static_targets", "dynamic_targets"})
        if unknown:
            raise ValueError("未知缓存字段：{}。".format(unknown))
        arrays = {
            "teacher_logits": self.logits,
            "static_targets": self.static,
            "dynamic_targets": self.dynamic,
        }
        # NumPy integer-array indexing already creates an independent writable
        # batch. A second np.array(copy=True) only duplicated that allocation.
        result = {key: torch.from_numpy(arrays[key][indices]) for key in names}
        if target_device.type != "cpu":
            result = {key: value.to(target_device, non_blocking=True) for key, value in result.items()}
        return result

    def close(self):
        for array in self.arrays.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()


class TeacherTargetCache:
    """Validated memory-mapped reader; opened read-only for concurrent workers."""

    def __init__(
        self, cache_dir, expected_hashes=None, verify_hashes=True,
        allow_code_provenance_mismatch=False,
    ):
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        manifest_path = self.cache_dir / "manifest.json"
        status_path = self.cache_dir / "status.json"
        if not manifest_path.is_file() or not status_path.is_file():
            raise FileNotFoundError("教师目标缓存缺少 manifest.json 或 status.json。")
        readable_only = resume_readable_only()
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.status = json.loads(status_path.read_text(encoding="utf-8"))
        if self.status.get("status") != "complete" or self.manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError("缓存不是已完成的受支持版本。")
        if (
            not readable_only
            and self.status.get("manifest_sha256") != _sha256_file(manifest_path)
        ):
            raise ValueError("缓存 manifest SHA-256 与 status 不匹配。")

        # The identity's shape and hash-string formats are part of the cache
        # metadata schema. Readable-only resume skips comparing that identity
        # with today's inputs, but still rejects a structurally unusable cache.
        actual_identity = _normalize_hashes(self.manifest.get("identity", {}))
        if expected_hashes is not None:
            if not readable_only:
                expected = _normalize_hashes(expected_hashes)
                compared_fields = (
                    tuple(field for field in IDENTITY_FIELDS if field != "code_sha256")
                    if allow_code_provenance_mismatch else IDENTITY_FIELDS
                )
                mismatches = {
                    key: {"expected": expected[key], "actual": actual_identity[key]}
                    for key in compared_fields if expected[key] != actual_identity[key]
                }
                if mismatches:
                    raise ValueError("教师目标缓存身份不匹配：{}。".format(mismatches))
        manifest_batch_size = self.manifest.get("generation_batch_size")
        identity_batch_size = self.manifest.get("identity", {}).get("generation_batch_size")
        status_batch_size = self.status.get("generation_batch_size")
        if (isinstance(manifest_batch_size, bool)
                or not isinstance(manifest_batch_size, int)
                or manifest_batch_size <= 0
                or manifest_batch_size != identity_batch_size
                or status_batch_size != manifest_batch_size):
            raise ValueError("缓存 manifest/status generation_batch_size 缺失或与 identity 不一致。")
        if tuple(self.manifest.get("splits", {}).keys()) != SPLITS:
            raise ValueError("缓存必须严格只包含 train 和 val 划分。")
        self.splits = {
            split: _SplitCache(
                self.cache_dir / split,
                self.manifest["splits"][split],
                verify_hashes=verify_hashes and not readable_only,
            ) for split in SPLITS
        }

    def get(self, split, row_indices, device=None, fields=None):
        if split not in SPLITS:
            raise ValueError("教师目标缓存禁止读取 {} 划分。".format(split))
        return self.splits[split].get(row_indices, device, fields)

    def close(self):
        for split_cache in self.splits.values():
            split_cache.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def build_teacher_target_cache(
    output_dir, archive_path, teacher_checkpoint_path, label_map_path, teacher, device,
    *, teacher_config, loss_config, requested_precision, effective_precision,
    code_sha256, generation_batch_size, expected_hashes=None,
):
    """Build once from train/val and atomically publish a strict cache.

    If output_dir already exists, it is accepted only when every identity field
    matches and all cache payload hashes validate; an incompatible cache raises.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    archive_path = Path(archive_path).expanduser().resolve()
    teacher_checkpoint_path = Path(teacher_checkpoint_path).expanduser().resolve()
    label_map_path = Path(label_map_path).expanduser().resolve()
    if (isinstance(generation_batch_size, bool)
            or not isinstance(generation_batch_size, (int, np.integer))
            or int(generation_batch_size) <= 0):
        raise ValueError("generation_batch_size 必须为正整数。")
    generation_batch_size = int(generation_batch_size)
    archive_sha = _sha256_file(archive_path)
    teacher_sha = _sha256_file(teacher_checkpoint_path)
    label_sha = _sha256_file(label_map_path)
    code_sha = str(code_sha256).lower()
    cached_loss_config = cache_loss_config(loss_config)
    identity = make_cache_identity(
        teacher_checkpoint_sha256=teacher_sha,
        archive_sha256=archive_sha,
        label_map_sha256=label_sha,
        requested_precision=requested_precision,
        effective_precision=effective_precision,
        teacher_model_config=teacher_config,
        loss_config=loss_config,
        code_sha256=code_sha,
        generation_batch_size=generation_batch_size,
    )
    if expected_hashes is not None and _normalize_hashes(expected_hashes) != identity:
        raise ValueError("传入 expected_hashes 与当前源文件/config/code 哈希不一致。")
    if output_dir.exists():
        existing = TeacherTargetCache(output_dir, expected_hashes=identity, verify_hashes=True)
        return existing.manifest

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir.with_name(".{}.building-{}".format(output_dir.name, uuid.uuid4().hex))
    work_dir.mkdir()
    metadata = load_archive_metadata(archive_path)
    sample_shape = tuple(int(value) for value in metadata["sample_shape"])
    num_points = int(sample_shape[-1])
    edges = teacher_config.get("skeleton_edges", topology.bones)
    edge_indices = np.arange(num_points * num_points, dtype=np.int64)
    if len(edge_indices) == 0:
        raise ValueError("缓存目标要求至少一条物理骨架边。")
    temperature = float(loss_config.get("topology_temperature", 0.2))
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "identity": identity,
        "generation_batch_size": generation_batch_size,
        "requested_precision": str(requested_precision),
        "effective_precision": str(effective_precision),
        "teacher_model_config": teacher_config,
        "loss_config": cached_loss_config,
        "target_representation": {
            "teacher_logits": "raw logits, float32",
            "static_targets": "full pairwise squared normalized-feature distances, flattened row-major, float32",
            "dynamic_targets": "full pairwise squared normalized-motion distances, float32",
            "feature_match_order": [
                {"student": match.student, "teacher": match.teacher, "weight": match.weight}
                for match in FEATURE_MATCHES
            ],
            "teacher_feature_order": list(TEACHER_FEATURE_NAMES),
            "topology_temperature": temperature,
            "edge_indices_flat": edge_indices.tolist(),
            "formulation": "full_rbf_squared_distances_learned_bandwidth",
            "edge_indices_sha256": hashlib.sha256(edge_indices.astype("<i8", copy=False).tobytes()).hexdigest(),
        },
        "splits": {},
    }
    _write_json(work_dir / "status.json", {
        "status": "building", "generation_batch_size": generation_batch_size,
        "completed_splits": [], "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()
    })
    teacher.eval()
    for parameter in teacher.parameters():
        if parameter.requires_grad:
            parameter.requires_grad_(False)
    try:
        for split in SPLITS:
            dataset = InMemoryNpzSkeletonDataset(
                archive_path, split, expected_frames=sample_shape[1], num_points=num_points
            )
            ids = _source_sample_ids(archive_path, split)
            if len(ids) != len(dataset):
                raise ValueError("{} sample_ids 与数据集行数不一致。".format(split))
            split_dir = work_dir / split
            split_dir.mkdir()
            count = len(dataset)
            num_classes = len(metadata["label_map"])
            shapes = {
                "teacher_logits": [count, num_classes],
                "static_targets": [count, len(FEATURE_MATCHES), len(edge_indices)],
                "dynamic_targets": [count, len(FEATURE_MATCHES), num_points, num_points],
                "row_indices": [count],
            }
            arrays = {
                "teacher_logits": np.lib.format.open_memmap(split_dir / "teacher_logits.npy", mode="w+", dtype=np.float32, shape=tuple(shapes["teacher_logits"])),
                "static_targets": np.lib.format.open_memmap(split_dir / "static_targets.npy", mode="w+", dtype=np.float32, shape=tuple(shapes["static_targets"])),
                "dynamic_targets": np.lib.format.open_memmap(split_dir / "dynamic_targets.npy", mode="w+", dtype=np.float32, shape=tuple(shapes["dynamic_targets"])),
                "row_indices": np.lib.format.open_memmap(split_dir / "row_indices.npy", mode="w+", dtype=np.int64, shape=(count,)),
            }
            loader = DataLoader(
                IndexedDataset(dataset), batch_size=generation_batch_size, shuffle=False,
                num_workers=0, pin_memory=torch.device(device).type == "cuda",
            )
            cursor = 0
            autocast_enabled = str(effective_precision) in {"bf16", "amp"}
            autocast_dtype = torch.bfloat16 if str(effective_precision) == "bf16" else torch.float16
            with torch.inference_mode():
                for joints, bones, _, row_indices in loader:
                    joints = joints.to(device, non_blocking=torch.device(device).type == "cuda")
                    bones = bones.to(device, non_blocking=torch.device(device).type == "cuda")
                    with torch.autocast(
                        device_type="cuda" if torch.device(device).type == "cuda" else "cpu",
                        dtype=autocast_dtype, enabled=autocast_enabled,
                    ):
                        logits, features = teacher(joints, bones, return_features=True)
                    targets = compress_teacher_targets(
                        logits, features, topology_temperature=temperature,
                        num_points=num_points, edges=edges,
                    )
                    rows = row_indices.numpy().astype(np.int64, copy=False)
                    if not np.array_equal(rows, np.arange(cursor, cursor + len(rows), dtype=np.int64)):
                        raise RuntimeError("预计算 DataLoader 行序异常，拒绝写入错位目标。")
                    end = cursor + len(rows)
                    arrays["teacher_logits"][rows] = targets["teacher_logits"].cpu().numpy()
                    arrays["static_targets"][rows] = targets["static_targets"].cpu().numpy()
                    arrays["dynamic_targets"][rows] = targets["dynamic_targets"].cpu().numpy()
                    arrays["row_indices"][rows] = rows
                    cursor = end
            if cursor != count:
                raise RuntimeError("{} 缓存只写入 {}/{} 行。".format(split, cursor, count))
            split_files = {}
            for name, array in arrays.items():
                array.flush()
                mmap = getattr(array, "_mmap", None)
                if mmap is not None:
                    mmap.close()
                path = split_dir / (name + ".npy")
                split_files[name] = {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            manifest["splits"][split] = {
                "count": count,
                "sample_ids_sha256": hashlib.sha256(canonical_json_bytes(ids)).hexdigest(),
                "shapes": shapes,
                "files": split_files,
                "completed_rows": cursor,
            }
            _write_json(work_dir / "manifest.json", manifest)
            _write_json(work_dir / "status.json", {
                "status": "building", "generation_batch_size": generation_batch_size,
                "completed_splits": list(manifest["splits"]),
                "completed_rows": {key: value["completed_rows"] for key, value in manifest["splits"].items()},
            })
            del arrays, loader, dataset

        total_rows = sum(item["count"] for item in manifest["splits"].values())
        sample_split = manifest["splits"]["train"]
        logits_shape = sample_split["shapes"]["teacher_logits"]
        static_shape = sample_split["shapes"]["static_targets"]
        dynamic_shape = sample_split["shapes"]["dynamic_targets"]
        target_names = ("teacher_logits", "static_targets", "dynamic_targets")
        target_payload_bytes = sum(
            int(np.prod(split_info["shapes"][name], dtype=np.int64)) * 4
            for split_info in manifest["splits"].values()
            for name in target_names
        )
        row_index_bytes = sum(
            int(np.prod(split_info["shapes"]["row_indices"], dtype=np.int64)) * 8
            for split_info in manifest["splits"].values()
        )
        target_payload_mib = target_payload_bytes / (1024.0 ** 2)
        readme = (
            "# 教师蒸馏目标缓存\n\n"
            "缓存仅包含训练集与验证集的 float32 NPY memmap，不会构造或读取测试集。"
            "缓存行使用 packed archive 中各划分的本地 row index；DataLoader 打乱后必须按该索引读取目标。\n\n"
            "本次缓存共 {} 行；每行 logits shape={}、static shape={}、dynamic shape={}。"
            "目标数组原始 payload 为 {:,} 字节（{:.2f} MiB），仅统计 logits/static/dynamic，明确排除 row_indices。"
            "row_indices 另占 {:,} 字节。上述容量均不含 NPY 文件头、manifest 和文件系统开销。\n\n"
            "目标生成 batch size={}（与学生配置保持一致）；生成 batch size 也属于缓存身份。"
            "来源、shape、代码/模型/损失与数据哈希，以及每个数组 SHA-256 见 manifest.json。\n"
        ).format(
            total_rows, logits_shape[1:], static_shape[1:], dynamic_shape[1:],
            target_payload_bytes, target_payload_mib, row_index_bytes, generation_batch_size,
        )
        (work_dir / "README.md").write_text(readme, encoding="utf-8")
        manifest_path = work_dir / "manifest.json"
        _write_json(work_dir / "status.json", {
            "status": "complete", "generation_batch_size": generation_batch_size,
            "completed_splits": list(SPLITS),
            "manifest_sha256": _sha256_file(manifest_path),
            "total_rows": sum(value["count"] for value in manifest["splits"].values()),
        })
        os.replace(str(work_dir), str(output_dir))
        TeacherTargetCache(output_dir, expected_hashes=identity, verify_hashes=True)
        return manifest
    except BaseException:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        raise


__all__ = [
    "CACHE_SCHEMA_VERSION", "HASH_FIELDS", "IndexedDataset", "IndexedTargetDataset",
    "TeacherTargetCache", "build_teacher_target_cache", "code_tree_sha256",
    "cache_loss_config", "compress_teacher_targets", "make_cache_identity", "object_sha256",
    "physical_edge_indices",
]
