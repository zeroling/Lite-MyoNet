from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lite_myonet.data.nctv_dataset import load_nctv_arrays, load_nctv_arrays_from_config
from lite_myonet.models.skeleton import load_skeleton


def _load_tiny_skeleton(tmp_path, frame_count=4):
    """Create and load a small skeleton with an explicit frame count."""
    config = {
        "schema_version": 1,
        "skeleton_id": "tiny-3",
        "point_count": 3,
        "root_index": 1,
        "nodes": [
            {"id": "left", "index": 0},
            {"id": "root", "index": 1},
            {"id": "right", "index": 2},
        ],
        "model_graph_edges": [[0, 1], [1, 2]],
        "bone_vector_edges": [[0, 1], [2, 0]],
        "channels": ["x", "y", "z"],
        "frame_count": frame_count,
    }
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return load_skeleton(path)


def test_npz_loads_only_nctv_joints_and_labels(tmp_path):
    """Load only joint arrays and labels from an NPZ archive."""
    skeleton = _load_tiny_skeleton(tmp_path)
    joints = np.arange(72, dtype=np.float32).reshape(2, 3, 4, 3)
    labels = np.asarray([1, 2], dtype=np.int16)
    archive_path = tmp_path / "train.npz"
    np.savez(archive_path, joints=joints, labels=labels)

    loaded_joints, loaded_labels = load_nctv_arrays(archive_path, skeleton)

    assert tuple(loaded_joints.shape) == (2, 3, 4, 3)
    assert loaded_joints.dtype == torch.float32
    assert tuple(loaded_labels.shape) == (2,)
    assert loaded_labels.dtype == torch.int64
    torch.testing.assert_close(loaded_joints, torch.from_numpy(joints))
    torch.testing.assert_close(loaded_labels, torch.tensor([1, 2]))


def test_npy_requires_explicit_labels_file(tmp_path):
    """Require an explicit labels file when loading NPY joints."""
    skeleton = _load_tiny_skeleton(tmp_path)
    joints = np.zeros((2, 3, 4, 3), dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int64)
    joints_path = tmp_path / "joints.npy"
    labels_path = tmp_path / "labels.npy"
    np.save(joints_path, joints)
    np.save(labels_path, labels)

    loaded_joints, loaded_labels = load_nctv_arrays(
        joints_path,
        skeleton,
        labels_path=labels_path,
    )

    assert tuple(loaded_joints.shape) == tuple(joints.shape)
    assert tuple(loaded_labels.shape) == (2,)


@pytest.mark.parametrize(
    "joints, labels, message",
    [
        (np.zeros((2, 4, 3, 3), dtype=np.float32), np.asarray([0, 1]), "C=4"),
        (np.zeros((2, 3, 4, 4), dtype=np.float32), np.asarray([0, 1]), "V=4"),
        (np.zeros((2, 3, 5, 3), dtype=np.float32), np.asarray([0, 1]), "T=5"),
        (np.zeros((2, 3, 4, 3), dtype=np.float64), np.asarray([0, 1]), "float32"),
        (np.full((2, 3, 4, 3), np.nan, dtype=np.float32), np.asarray([0, 1]), "NaN or Inf"),
        (np.zeros((2, 3, 4, 3), dtype=np.float32), np.asarray([0]), "length"),
        (np.zeros((2, 3, 4, 3), dtype=np.float32), np.asarray([0.0, 1.0]), "integer dtype"),
    ],
)
def test_npz_shape_dtype_finiteness_and_label_contract_are_checked(
    tmp_path, joints, labels, message
):
    """Reject arrays that violate NCTV shape, dtype, finiteness, or label requirements."""
    skeleton = _load_tiny_skeleton(tmp_path)
    archive_path = tmp_path / "bad.npz"
    np.savez(archive_path, joints=joints, labels=labels)

    with pytest.raises((TypeError, ValueError), match=message):
        load_nctv_arrays(archive_path, skeleton)


def test_npz_rejects_persisted_bone_vectors_and_unknown_arrays(tmp_path):
    """Reject archives with persisted bones or unknown array keys."""
    skeleton = _load_tiny_skeleton(tmp_path)
    joints = np.zeros((1, 3, 4, 3), dtype=np.float32)
    labels = np.asarray([0], dtype=np.int64)
    archive_path = tmp_path / "with_bones.npz"
    np.savez(archive_path, joints=joints, labels=labels, bones=joints.copy())

    with pytest.raises(ValueError, match="must be exactly"):
        load_nctv_arrays(archive_path, skeleton)


def test_non_npy_npz_formats_and_layouts_are_not_guessed(tmp_path):
    """Reject unsupported formats rather than inferring array layouts."""
    skeleton = _load_tiny_skeleton(tmp_path)
    invalid_path = tmp_path / "joints.csv"
    invalid_path.write_text("1,2,3\n", encoding="utf-8")

    with pytest.raises(ValueError, match=".npy or .npz"):
        load_nctv_arrays(invalid_path, skeleton)


def test_npy_ctv_single_sample_is_not_implicitly_promoted_to_nctv(tmp_path):
    """Reject CTV arrays instead of guessing a missing batch dimension."""
    skeleton = _load_tiny_skeleton(tmp_path)
    joints_path = tmp_path / "single_sample.npy"
    labels_path = tmp_path / "single_label.npy"
    np.save(joints_path, np.zeros((3, 4, 3), dtype=np.float32))
    np.save(labels_path, np.asarray([0], dtype=np.int64))

    with pytest.raises(ValueError, match="NCTV shape"):
        load_nctv_arrays(joints_path, skeleton, labels_path=labels_path)


def test_data_yaml_resolves_skeleton_and_loads_npz_arrays(tmp_path):
    """Resolve the skeleton config and load arrays under the data YAML contract."""
    config_root = tmp_path / "configs"
    skeleton_dir = config_root / "skeletons"
    data_dir = config_root / "data"
    skeleton_dir.mkdir(parents=True)
    data_dir.mkdir()
    skeleton = {
        "schema_version": 1,
        "skeleton_id": "tiny-3",
        "point_count": 3,
        "root_index": 1,
        "nodes": [
            {"id": "left", "index": 0},
            {"id": "root", "index": 1},
            {"id": "right", "index": 2},
        ],
        "model_graph_edges": [[0, 1], [1, 2]],
        "bone_vector_edges": [[0, 1], [2, 0]],
        "channels": ["x", "y", "z"],
        "frame_count": 4,
    }
    skeleton_path = skeleton_dir / "tiny.yaml"
    skeleton_path.write_text(yaml.safe_dump(skeleton), encoding="utf-8")
    data_config = {
        "schema_version": 1,
        "format": "npz",
        "skeleton_config": "../skeletons/tiny.yaml",
        "joints_key": "joints",
        "labels_key": "labels",
        "array_layout": "NCTV",
        "dtype": "float32",
        "labels_dtype": "integer",
        "persist_bones": False,
    }
    data_config_path = data_dir / "nctv.yaml"
    data_config_path.write_text(yaml.safe_dump(data_config), encoding="utf-8")
    archive_path = tmp_path / "data.npz"
    joints = np.zeros((1, 3, 4, 3), dtype=np.float32)
    labels = np.asarray([0], dtype=np.int64)
    np.savez(archive_path, joints=joints, labels=labels)

    loaded_joints, loaded_labels = load_nctv_arrays_from_config(
        archive_path,
        data_config_path,
    )

    assert tuple(loaded_joints.shape) == (1, 3, 4, 3)
    assert tuple(loaded_labels.shape) == (1,)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"array_layout": "TVC"}, "array_layout must be NCTV"),
        ({"persist_bones": True}, "persist_bones must be false"),
        ({"dtype": "float64"}, "dtype must be float32"),
    ],
)
def test_data_yaml_rejects_noncanonical_contract(tmp_path, changes, message):
    """Reject noncanonical data YAML fields before reading the archive."""
    config_root = tmp_path / "configs"
    skeleton_dir = config_root / "skeletons"
    data_dir = config_root / "data"
    skeleton_dir.mkdir(parents=True)
    data_dir.mkdir()
    skeleton_path = skeleton_dir / "tiny.yaml"
    skeleton_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "skeleton_id": "tiny-3",
                "point_count": 3,
                "root_index": 1,
                "nodes": [
                    {"id": "left", "index": 0},
                    {"id": "root", "index": 1},
                    {"id": "right", "index": 2},
                ],
                "model_graph_edges": [[0, 1], [1, 2]],
                "bone_vector_edges": [[0, 1], [2, 0]],
                "channels": ["x", "y", "z"],
                "frame_count": 4,
            }
        ),
        encoding="utf-8",
    )
    data_config = {
        "schema_version": 1,
        "format": "npz",
        "skeleton_config": "../skeletons/tiny.yaml",
        "joints_key": "joints",
        "labels_key": "labels",
        "array_layout": "NCTV",
        "dtype": "float32",
        "labels_dtype": "integer",
        "persist_bones": False,
    }
    data_config.update(changes)
    data_config_path = data_dir / "nctv.yaml"
    data_config_path.write_text(yaml.safe_dump(data_config), encoding="utf-8")
    archive_path = tmp_path / "data.npz"
    archive_path.write_bytes(b"not read before config validation")

    with pytest.raises(ValueError, match=message):
        load_nctv_arrays_from_config(archive_path, data_config_path)
