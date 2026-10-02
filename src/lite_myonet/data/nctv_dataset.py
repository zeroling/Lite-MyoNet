"""Strict NCTV dataset readers for joint arrays and integer labels."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import yaml

from ..models.skeleton import SkeletonDefinition, load_skeleton


PathLike = Union[str, Path]


def _validate_arrays(joints, labels, skeleton):
    """Validate float32 NCTV joints and integer labels against a skeleton."""
    if not isinstance(skeleton, SkeletonDefinition):
        raise TypeError("skeleton must be a SkeletonDefinition.")
    if not isinstance(joints, np.ndarray) or joints.ndim != 4:
        shape = getattr(joints, "shape", None)
        raise ValueError(f"joints must have NCTV shape, got {shape}.")
    if joints.dtype != np.float32:
        raise TypeError(f"joints must use float32, got {joints.dtype}.")
    if not np.isfinite(joints).all():
        raise ValueError("joints contains NaN or Inf.")
    if joints.shape[0] <= 0:
        raise ValueError("joints must contain at least one sequence.")
    if joints.shape[1] != len(skeleton.channels):
        raise ValueError(
            f"joints has C={joints.shape[1]}, expected {len(skeleton.channels)} channels."
        )
    if joints.shape[2] <= 0:
        raise ValueError("joints must contain at least one frame.")
    if skeleton.frame_count is not None and joints.shape[2] != skeleton.frame_count:
        raise ValueError(
            f"joints has T={joints.shape[2]}, expected frame_count={skeleton.frame_count}."
        )
    if joints.shape[3] != skeleton.point_count:
        raise ValueError(
            f"joints has V={joints.shape[3]}, expected point_count={skeleton.point_count}."
        )
    if not isinstance(labels, np.ndarray) or labels.ndim != 1:
        shape = getattr(labels, "shape", None)
        raise ValueError(f"labels must have shape [N], got {shape}.")
    if labels.dtype.kind not in "iu":
        raise TypeError(f"labels must use an integer dtype, got {labels.dtype}.")
    if labels.shape[0] != joints.shape[0]:
        raise ValueError("labels length must match the N dimension of joints.")
    if labels.size and int(labels.min()) < 0:
        raise ValueError("labels must be non-negative integers.")
    if labels.size and int(labels.max()) > np.iinfo(np.int64).max:
        raise ValueError("labels exceed the supported int64 range.")


def load_nctv_arrays(
    path: PathLike,
    skeleton: SkeletonDefinition,
    *,
    labels_path: Optional[PathLike] = None,
    joints_key: str = "joints",
    labels_key: str = "labels",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load float32 NCTV joints and integer labels from NPY or NPZ files.

    NPZ files must contain exactly the configured joints and labels arrays.
    NPY files store the joints array and require a separate labels NPY file.
    """
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {source_path}")
    if not isinstance(joints_key, str) or not joints_key:
        raise ValueError("joints_key must be a non-empty string.")
    if not isinstance(labels_key, str) or not labels_key:
        raise ValueError("labels_key must be a non-empty string.")
    if joints_key == labels_key:
        raise ValueError("joints_key and labels_key must differ.")

    suffix = source_path.suffix.lower()
    if suffix == ".npz":
        if labels_path is not None:
            raise ValueError("labels_path must not be supplied for an NPZ dataset.")
        with np.load(source_path, allow_pickle=False) as archive:
            expected_keys = {joints_key, labels_key}
            actual_keys = set(archive.files)
            if actual_keys != expected_keys:
                raise ValueError(
                    f"NPZ arrays must be exactly {sorted(expected_keys)}, "
                    f"got {sorted(actual_keys)}."
                )
            joints = np.array(archive[joints_key], copy=True)
            labels = np.array(archive[labels_key], copy=True)
    elif suffix == ".npy":
        if labels_path is None:
            raise ValueError("labels_path is required when loading NPY joints.")
        label_file = Path(labels_path).expanduser().resolve()
        if not label_file.is_file():
            raise FileNotFoundError(f"Labels NPY file does not exist: {label_file}")
        if label_file.suffix.lower() != ".npy":
            raise ValueError("labels_path must point to a .npy file.")
        joints = np.load(source_path, allow_pickle=False)
        labels = np.load(label_file, allow_pickle=False)
    else:
        raise ValueError("Dataset files must use the .npy or .npz extension.")

    _validate_arrays(joints, labels, skeleton)
    joints_tensor = torch.from_numpy(np.ascontiguousarray(joints, dtype=np.float32))
    labels_tensor = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64))
    return joints_tensor, labels_tensor


def load_nctv_arrays_from_config(
    path: PathLike,
    config_path: PathLike,
    *,
    labels_path: Optional[PathLike] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load an NCTV dataset using a version-1 YAML data configuration."""
    data_path = Path(path).expanduser().resolve()
    config_file = Path(config_path).expanduser().resolve()
    if not config_file.is_file():
        raise FileNotFoundError(f"Data configuration does not exist: {config_file}")
    try:
        with config_file.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid data YAML {config_file}: {exc}") from exc
    if not isinstance(config, dict) or any(not isinstance(key, str) for key in config):
        raise ValueError("Data configuration must be a YAML mapping with string keys.")

    expected_fields = {
        "schema_version",
        "format",
        "skeleton_config",
        "joints_key",
        "labels_key",
        "array_layout",
        "dtype",
        "labels_dtype",
        "persist_bones",
    }
    if set(config) != expected_fields:
        missing = sorted(expected_fields - set(config))
        unknown = sorted(set(config) - expected_fields)
        raise ValueError(
            f"Data configuration fields mismatch: missing={missing}, unknown={unknown}."
        )
    if (
        isinstance(config["schema_version"], bool)
        or not isinstance(config["schema_version"], int)
        or config["schema_version"] != 1
    ):
        raise ValueError("Data schema_version must be 1.")
    if config["format"] not in {"npy", "npz"}:
        raise ValueError("Data format must be npy or npz.")
    if data_path.suffix.lower() != f".{config['format']}":
        raise ValueError(
            f"Data configuration format={config['format']} does not match {data_path.suffix}."
        )
    if config["array_layout"] != "NCTV":
        raise ValueError("array_layout must be NCTV; other layouts require offline conversion.")
    if config["dtype"] != "float32":
        raise ValueError("dtype must be float32.")
    if config["labels_dtype"] != "integer":
        raise ValueError("labels_dtype must be integer.")
    if config["persist_bones"] is not False:
        raise ValueError("persist_bones must be false; bones are derived at model input time.")
    if not isinstance(config["skeleton_config"], str) or not config["skeleton_config"].strip():
        raise ValueError("skeleton_config must be a non-empty path.")

    skeleton_path = (config_file.parent / config["skeleton_config"]).resolve()
    skeleton = load_skeleton(skeleton_path)
    return load_nctv_arrays(
        data_path,
        skeleton,
        labels_path=labels_path,
        joints_key=config["joints_key"],
        labels_key=config["labels_key"],
    )
