from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lite_myonet.models.skeleton import load_skeleton
from lite_myonet.models.lite_myonet import LiteMyoNet, STUDENT_FEATURE_KEYS


def _write_small_skeleton(tmp_path: Path):
    """Write and load a three-node skeleton with a nonzero root index."""
    config = {
        "schema_version": 1,
        "skeleton_id": "student-test-3",
        "point_count": 3,
        "root_index": 1,
        "nodes": [
            {"id": "left", "index": 0},
            {"id": "root", "index": 1},
            {"id": "right", "index": 2},
        ],
        "model_graph_edges": [[0, 1], [1, 2]],
        "bone_vector_edges": [[0, 1], [2, 1]],
        "channels": ["x", "y", "z"],
        "frame_count": 8,
    }
    config_path = tmp_path / "student-test-skeleton.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return load_skeleton(config_path)


def test_main_d2_student_has_the_reference_parameter_budget_and_skeleton():
    """Check default student size and full-body skeleton settings."""
    skeleton = load_skeleton(PROJECT_ROOT / "configs" / "skeletons" / "fullbody23.yaml")
    model = LiteMyoNet(skeleton=skeleton).cpu()

    assert model.channels == (32, 64, 128)
    assert model.dropout_rate == 0.05
    assert model.fourier_degree == 2
    assert model.count_params() == 307_927
    assert len(model.state_dict()) == 216
    assert model.num_points == 23


def test_student_generates_bones_from_nctv_input_using_yaml_topology(tmp_path):
    """Verify the student derives bones from NCTV joints and YAML topology."""
    skeleton = _write_small_skeleton(tmp_path)
    model = LiteMyoNet(
        skeleton=skeleton,
        num_classes=4,
        channels=(8, 16, 32),
        dropout=0.0,
    ).cpu().eval()
    joints = torch.randn(2, 3, 8, 3, requires_grad=True)
    captured_bones = {}

    def capture_bone_input(_module, inputs):
        """Capture the bone tensor passed to the first bone-stream block."""
        captured_bones["value"] = inputs[0].detach().clone()

    handle = model.bone_vector_stream[0].register_forward_pre_hook(capture_bone_input)
    try:
        with torch.enable_grad():
            logits, features = model(joints, return_features=True)
            logits.square().mean().backward()
    finally:
        handle.remove()

    expected_bones = torch.zeros_like(joints.detach())
    expected_bones[..., 0] = joints.detach()[..., 0] - joints.detach()[..., 1]
    expected_bones[..., 2] = joints.detach()[..., 2] - joints.detach()[..., 1]
    torch.testing.assert_close(captured_bones["value"], expected_bones)
    assert tuple(logits.shape) == (2, 4)
    assert tuple(features) == STUDENT_FEATURE_KEYS
    assert [features[name].shape[1] for name in STUDENT_FEATURE_KEYS] == [8, 8, 16, 16, 32, 32]
    assert joints.grad is not None
    assert torch.isfinite(joints.grad).all()
