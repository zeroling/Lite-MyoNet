from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lite_myonet.data.bones import joints_to_bones
from lite_myonet.models.skeleton import load_skeleton


def _load_tiny_skeleton(tmp_path):
    """Create a small skeleton with a nonzero root index."""
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
        "frame_count": 2,
    }
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return load_skeleton(path)


def test_torch_bone_vectors_follow_configured_direction_and_zero_root(tmp_path):
    """Check configured bone directions and zero-valued root vectors."""
    skeleton = _load_tiny_skeleton(tmp_path)
    joints = torch.tensor(
        [[[[1.0, 10.0, 4.0], [2.0, 20.0, 7.0]]]],
        dtype=torch.float32,
    ).expand(1, 3, 2, 3).clone()
    joints[:, 1] *= 2
    joints[:, 2] *= 3

    bones = joints_to_bones(joints, skeleton)

    assert bones.shape == joints.shape
    torch.testing.assert_close(bones[..., 0], joints[..., 0] - joints[..., 1])
    torch.testing.assert_close(bones[..., 1], torch.zeros_like(bones[..., 1]))
    torch.testing.assert_close(bones[..., 2], joints[..., 2] - joints[..., 0])


def test_torch_bone_vectors_preserve_autograd(tmp_path):
    """Verify bone construction preserves gradients to joint coordinates."""
    skeleton = _load_tiny_skeleton(tmp_path)
    joints = torch.randn(2, 3, 2, 3, dtype=torch.float32, requires_grad=True)

    bones = joints_to_bones(joints, skeleton)
    loss = bones.square().sum()
    loss.backward()

    assert joints.grad is not None
    assert torch.isfinite(joints.grad).all()
    assert torch.count_nonzero(joints.grad).item() > 0


def test_numpy_bone_vectors_are_supported(tmp_path):
    """Check NumPy joint arrays are accepted and retain their shape."""
    skeleton = _load_tiny_skeleton(tmp_path)
    joints = np.arange(36, dtype=np.float32).reshape(2, 3, 2, 3)

    bones = joints_to_bones(joints, skeleton)

    assert bones.shape == joints.shape
    np.testing.assert_array_equal(bones[..., 1], np.zeros_like(bones[..., 1]))
    np.testing.assert_array_equal(bones[..., 0], joints[..., 0] - joints[..., 1])
    np.testing.assert_array_equal(bones[..., 2], joints[..., 2] - joints[..., 0])


@pytest.mark.parametrize(
    "joints, error_type, message",
    [
        (torch.zeros(3, 2, 3), ValueError, "NCTV shape"),
        (torch.zeros(1, 4, 2, 3), ValueError, "C dimension"),
        (torch.zeros(1, 3, 2, 4), ValueError, "V dimension"),
        (torch.zeros(1, 3, 2, 3, dtype=torch.int64), TypeError, "floating-point"),
    ],
)
def test_invalid_bone_inputs_are_rejected(tmp_path, joints, error_type, message):
    """Reject unsupported joint arrays and incompatible skeleton inputs."""
    skeleton = _load_tiny_skeleton(tmp_path)

    with pytest.raises(error_type, match=message):
        joints_to_bones(joints, skeleton)
