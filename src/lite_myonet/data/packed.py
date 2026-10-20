import gc
import hashlib
import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import Dataset

from .manifest import REQUIRED_SPLITS, file_sha256, verify_manifest
from ..runtime import resume_readable_only


ARCHIVE_SCHEMA_VERSION = 1
ARRAY_LAYOUT = "NCTV"


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as source_file:
        return json.load(source_file)


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")


def _decode_json_array(array, field_name):
    if array.dtype != np.uint8:
        raise ValueError(f"{field_name} 必须保存为 uint8 UTF-8 字节数组。")
    try:
        return json.loads(array.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field_name} 不是有效的 UTF-8 JSON。") from error


def _fixed_unicode(values):
    values = [str(value) for value in values]
    width = max(1, max((len(value) for value in values), default=1))
    return np.asarray(values, dtype=f"<U{width}")


def _validate_source_array(array, path, expected_shape):
    if tuple(array.shape) != tuple(expected_shape):
        raise ValueError(
            f"源数组形状错误：{path} 为 {array.shape}，期望 {tuple(expected_shape)}。"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"源数组不是数值类型：{path} 为 {array.dtype}。")
    if not np.isfinite(array).all():
        raise ValueError(f"源数组包含 NaN 或 Inf：{path}。")


def _safe_remove_work_dir(work_dir, output_path):
    work_dir = Path(work_dir).resolve()
    output_path = Path(output_path).resolve()
    expected_prefix = f".{output_path.stem}.packing-"
    if work_dir.parent != output_path.parent or not work_dir.name.startswith(
        expected_prefix
    ):
        raise RuntimeError(f"拒绝清理未验证的临时目录：{work_dir}")
    if work_dir.exists():
        shutil.rmtree(work_dir)


def _close_memmaps(arrays):
    for array in arrays:
        memory_map = getattr(array, "_mmap", None)
        if memory_map is not None:
            memory_map.close()


def _derive_label_map(records):
    label_map = {}
    for record in records:
        label_name = str(record["label_name"])
        label = int(record["label"])
        previous = label_map.get(label_name)
        if previous is not None and previous != label:
            raise ValueError(f"类别 {label_name} 同时映射到 {previous} 和 {label}。")
        label_map[label_name] = label
    return dict(sorted(label_map.items(), key=lambda item: item[1]))


def build_dataset_archive(
    manifest_path,
    dataset_root,
    output_path,
    expected_shape=(3, 120, 23),
):
    """把清单选定的小 NPY 文件写成一个 float32 压缩 NPZ 数据包。"""

    manifest_path = Path(manifest_path).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path.suffix.lower() != ".npz":
        raise ValueError(f"输出文件必须使用 .npz 扩展名：{output_path}")
    if output_path.exists():
        raise FileExistsError(f"输出文件已经存在，未覆盖：{output_path}")

    expected_shape = tuple(int(dimension) for dimension in expected_shape)
    if len(expected_shape) != 3 or any(dimension <= 0 for dimension in expected_shape):
        raise ValueError(f"expected_shape 必须是三个正整数：{expected_shape}。")

    manifest_report = verify_manifest(
        manifest_path, dataset_root, verify_hashes=True
    )
    manifest = _load_json(manifest_path)
    all_records = [
        record
        for split_name in REQUIRED_SPLITS
        for record in manifest["splits"][split_name]
    ]
    label_map = _derive_label_map(all_records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output_path.parent / f".{output_path.stem}.packing-{uuid.uuid4().hex}"
    temporary_archive = output_path.with_name(
        f".{output_path.name}.part-{uuid.uuid4().hex}"
    )
    work_dir.mkdir()
    memmaps = []
    try:
        archive_arrays = {}
        metadata = {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "description": (
                f"Lite-MyoNet {manifest_report['total_sequences']:,}-sequence "
                "synthetic dataset packed as one float32 NPZ archive for "
                "in-memory loading."
            ),
            "array_layout": ARRAY_LAYOUT,
            "array_dtype": "float32",
            "sample_shape": list(expected_shape),
            "total_sequences": int(manifest_report["total_sequences"]),
            "split_counts": manifest_report["split_counts"],
            "class_counts": manifest_report["class_counts"],
            "label_map": label_map,
            "source_manifest_sha256": file_sha256(manifest_path),
            "conversion": "numpy.astype(float32)",
            "simulation_parameter_storage": (
                "integer thousandths; -1 marks healthy samples without an "
                "impairment parameter"
            ),
        }
        archive_arrays["metadata_json_utf8"] = np.frombuffer(
            _json_bytes(metadata), dtype=np.uint8
        )
        archive_arrays["source_manifest_json_utf8"] = np.frombuffer(
            manifest_path.read_bytes(), dtype=np.uint8
        )

        for split_name in REQUIRED_SPLITS:
            records = manifest["splits"][split_name]
            split_shape = (len(records),) + expected_shape
            joints = open_memmap(
                work_dir / f"{split_name}_joints.npy",
                mode="w+",
                dtype=np.float32,
                shape=split_shape,
            )
            bones = open_memmap(
                work_dir / f"{split_name}_bones.npy",
                mode="w+",
                dtype=np.float32,
                shape=split_shape,
            )
            memmaps.extend((joints, bones))

            labels = np.empty(len(records), dtype=np.int16)
            parameter_milli = np.full(len(records), -1, dtype=np.int16)
            sample_ids = []
            label_names = []
            split_group_ids = []
            source_joint_paths = []
            source_bone_paths = []

            for index, record in enumerate(records):
                joint_path = dataset_root / record["joints"]
                bone_path = dataset_root / record["bones"]
                source_joints = np.load(joint_path, mmap_mode="r", allow_pickle=False)
                source_bones = np.load(bone_path, mmap_mode="r", allow_pickle=False)
                _validate_source_array(source_joints, joint_path, expected_shape)
                _validate_source_array(source_bones, bone_path, expected_shape)
                joints[index] = source_joints.astype(np.float32, copy=False)
                bones[index] = source_bones.astype(np.float32, copy=False)

                labels[index] = int(record["label"])
                raw_milli = record.get("simulation_parameter_milli")
                if raw_milli is not None:
                    parameter_milli[index] = int(raw_milli)
                sample_ids.append(
                    record.get("source_sample_id", Path(record["joints"]).stem)
                )
                label_names.append(record["label_name"])
                split_group_ids.append(record["split_group_id"])
                source_joint_paths.append(record["joints"])
                source_bone_paths.append(record["bones"])

            joints.flush()
            bones.flush()
            archive_arrays[f"{split_name}_joints"] = joints
            archive_arrays[f"{split_name}_bones"] = bones
            archive_arrays[f"{split_name}_labels"] = labels
            archive_arrays[f"{split_name}_simulation_parameter_milli"] = parameter_milli
            archive_arrays[f"{split_name}_sample_ids"] = _fixed_unicode(sample_ids)
            archive_arrays[f"{split_name}_label_names"] = _fixed_unicode(label_names)
            archive_arrays[f"{split_name}_split_group_ids"] = _fixed_unicode(
                split_group_ids
            )
            archive_arrays[f"{split_name}_source_joint_paths"] = _fixed_unicode(
                source_joint_paths
            )
            archive_arrays[f"{split_name}_source_bone_paths"] = _fixed_unicode(
                source_bone_paths
            )

        with temporary_archive.open("wb") as archive_file:
            np.savez_compressed(archive_file, **archive_arrays)

        _close_memmaps(memmaps)
        memmaps.clear()
        archive_arrays.clear()
        gc.collect()

        verification = verify_dataset_archive(
            temporary_archive,
            dataset_root=dataset_root,
            compare_sources=True,
        )
        temporary_archive.replace(output_path)
        verification["archive_path"] = str(output_path)
        verification["archive_sha256"] = file_sha256(output_path)
        verification["archive_bytes"] = output_path.stat().st_size
        return verification
    except Exception:
        _close_memmaps(memmaps)
        memmaps.clear()
        gc.collect()
        if temporary_archive.exists():
            temporary_archive.unlink()
        raise
    finally:
        _safe_remove_work_dir(work_dir, output_path)


def verify_dataset_archive(
    archive_path,
    expected_sha256=None,
    dataset_root=None,
    compare_sources=False,
):
    """校验单文件 NPZ 数据包，并可逐条比对原始 float32 模型输入。"""

    archive_path = Path(archive_path).expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"数据包不存在：{archive_path}")
    archive_hash = file_sha256(archive_path)
    if (
        expected_sha256
        and not resume_readable_only()
        and archive_hash.lower() != str(expected_sha256).lower()
    ):
        raise ValueError(
            f"数据包 SHA-256 不匹配：期望 {expected_sha256}，实际 {archive_hash}。"
        )

    with np.load(archive_path, allow_pickle=False) as archive:
        if "metadata_json_utf8" not in archive or "source_manifest_json_utf8" not in archive:
            raise ValueError("数据包缺少 metadata 或源清单。")
        metadata = _decode_json_array(archive["metadata_json_utf8"], "metadata_json_utf8")
        manifest_bytes = archive["source_manifest_json_utf8"].tobytes()
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("数据包中的源清单不是有效的 UTF-8 JSON。") from error

        if int(metadata.get("schema_version", -1)) != ARCHIVE_SCHEMA_VERSION:
            raise ValueError(f"不支持的数据包 schema：{metadata.get('schema_version')}。")
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        if (
            not resume_readable_only()
            and manifest_hash != metadata.get("source_manifest_sha256")
        ):
            raise ValueError("数据包中的源清单哈希与 metadata 不一致。")

        if compare_sources and dataset_root is None:
            raise ValueError("compare_sources=True 时必须提供 dataset_root。")
        resolved_root = (
            Path(dataset_root).expanduser().resolve() if dataset_root is not None else None
        )

        split_counts = {}
        for split_name in REQUIRED_SPLITS:
            records = manifest["splits"][split_name]
            count = len(records)
            split_counts[split_name] = count
            required_keys = [
                f"{split_name}_joints",
                f"{split_name}_bones",
                f"{split_name}_labels",
                f"{split_name}_simulation_parameter_milli",
                f"{split_name}_sample_ids",
                f"{split_name}_label_names",
                f"{split_name}_split_group_ids",
                f"{split_name}_source_joint_paths",
                f"{split_name}_source_bone_paths",
            ]
            missing = [key for key in required_keys if key not in archive]
            if missing:
                raise ValueError(f"数据包缺少 {split_name} 字段：{missing}。")

            expected_shape = (count, *tuple(metadata["sample_shape"]))
            for modality, source_key in (("joints", "joints"), ("bones", "bones")):
                key = f"{split_name}_{modality}"
                values = archive[key]
                if values.shape != expected_shape or values.dtype != np.float32:
                    raise ValueError(
                        f"{key} 为 shape={values.shape}, dtype={values.dtype}，"
                        f"期望 shape={expected_shape}, dtype=float32。"
                    )
                if not np.isfinite(values).all():
                    raise ValueError(f"{key} 包含 NaN 或 Inf。")
                if compare_sources:
                    for index, record in enumerate(records):
                        source = np.load(
                            resolved_root / record[source_key],
                            mmap_mode="r",
                            allow_pickle=False,
                        )
                        if not np.array_equal(values[index], source.astype(np.float32)):
                            raise ValueError(
                                f"{key}[{index}] 与源文件 {record[source_key]} 不一致。"
                            )
                del values

            labels = archive[f"{split_name}_labels"]
            parameter_milli = archive[f"{split_name}_simulation_parameter_milli"]
            if labels.shape != (count,) or labels.dtype != np.int16:
                raise ValueError(f"{split_name}_labels 的形状或类型错误。")
            if parameter_milli.shape != (count,) or parameter_milli.dtype != np.int16:
                raise ValueError(
                    f"{split_name}_simulation_parameter_milli 的形状或类型错误。"
                )
            for index, record in enumerate(records):
                expected_label = int(record["label"])
                expected_parameter = record.get("simulation_parameter_milli")
                expected_parameter = -1 if expected_parameter is None else int(expected_parameter)
                if int(labels[index]) != expected_label:
                    raise ValueError(f"{split_name} 第 {index} 条标签与源清单不一致。")
                if int(parameter_milli[index]) != expected_parameter:
                    raise ValueError(
                        f"{split_name} 第 {index} 条仿真参数与源清单不一致。"
                    )

    if split_counts != metadata["split_counts"]:
        raise ValueError(
            f"数据包划分数量 {split_counts} 与 metadata {metadata['split_counts']} 不一致。"
        )
    if sum(split_counts.values()) != int(metadata["total_sequences"]):
        raise ValueError("数据包总样本数与 metadata 不一致。")
    return {
        "archive_path": str(archive_path),
        "archive_sha256": archive_hash,
        "archive_bytes": archive_path.stat().st_size,
        "source_manifest_sha256": metadata["source_manifest_sha256"],
        "compare_sources": bool(compare_sources),
        "total_sequences": int(metadata["total_sequences"]),
        "split_counts": split_counts,
        "sample_shape": list(metadata["sample_shape"]),
        "dtype": metadata["array_dtype"],
    }


def load_archive_metadata(archive_path):
    archive_path = Path(archive_path).expanduser().resolve()
    with np.load(archive_path, allow_pickle=False) as archive:
        return _decode_json_array(archive["metadata_json_utf8"], "metadata_json_utf8")


class InMemoryNpzSkeletonDataset(Dataset):
    """从单个压缩 NPZ 中解压一个划分并驻留在内存。"""

    requires_single_process_loader = True

    def __init__(
        self,
        archive_path,
        split,
        transform=None,
        expected_frames=120,
        num_points=23,
        return_index=False,
    ):
        if split not in REQUIRED_SPLITS:
            raise ValueError(f"未知划分：{split!r}。")
        self.archive_path = Path(archive_path).expanduser().resolve()
        self.transform = transform
        self.return_index = bool(return_index)
        with np.load(self.archive_path, allow_pickle=False) as archive:
            metadata = _decode_json_array(
                archive["metadata_json_utf8"], "metadata_json_utf8"
            )
            expected_shape = (3, int(expected_frames), int(num_points))
            if tuple(metadata["sample_shape"]) != expected_shape:
                raise ValueError(
                    f"数据包样本形状为 {tuple(metadata['sample_shape'])}，"
                    f"期望 {expected_shape}。"
                )
            self.joints = np.ascontiguousarray(archive[f"{split}_joints"])
            self.bones = np.ascontiguousarray(archive[f"{split}_bones"])
            self.labels = archive[f"{split}_labels"].astype(np.int64, copy=True)

        if len(self.joints) == 0:
            raise RuntimeError(f"数据包中的 {split} 划分为空。")
        if len(self.joints) != len(self.bones) or len(self.joints) != len(self.labels):
            raise ValueError(f"数据包中的 {split} 数组长度不一致。")

    @property
    def memory_bytes(self):
        return self.joints.nbytes + self.bones.nbytes + self.labels.nbytes

    def __len__(self):
        return len(self.labels)

    def __getitems__(self, indices):
        """Batch slicing keeps the same fresh storage while reducing Python copies."""
        if self.transform is not None:
            return [self[index] for index in indices]
        joints = torch.from_numpy(np.ascontiguousarray(self.joints[indices], dtype=np.float32))
        bones = torch.from_numpy(np.ascontiguousarray(self.bones[indices], dtype=np.float32))
        if self.return_index:
            return [(joints[row], bones[row], int(self.labels[index]), int(index))
                    for row, index in enumerate(indices)]
        return [(joints[row], bones[row], int(self.labels[index])) for row, index in enumerate(indices)]

    def __getitem__(self, index):
        joints_array = np.array(
            self.joints[index], dtype=np.float32, order="C", copy=True
        )
        bones_array = np.array(
            self.bones[index], dtype=np.float32, order="C", copy=True
        )
        joints = torch.from_numpy(joints_array)
        bones = torch.from_numpy(bones_array)
        if self.transform is not None:
            joints, bones = self.transform(joints, bones)
        result = (joints, bones, int(self.labels[index]))
        if self.return_index:
            return result + (int(index),)
        return result
