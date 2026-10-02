"""CPU numerical tests for joints-only dynamic-batch ONNX export."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import torch
import yaml


from lite_myonet.export.onnx_export import export_checkpoint_to_onnx
from lite_myonet.models.skeleton import load_skeleton
from lite_myonet.models.lite_myonet import LiteMyoNet


def _write_student_checkpoint(
    tmp_path: Path, *, include_frame_count: bool = True
) -> tuple[Path, tuple[int, int | None, int]]:
    """Build a full Fourier student checkpoint without performing training."""
    root = Path(__file__).resolve().parents[1]
    skeleton_path = root / "configs" / "skeletons" / "utd_mhad20.yaml"
    skeleton_mapping = yaml.safe_load(skeleton_path.read_text(encoding="utf-8"))
    if not include_frame_count:
        skeleton_mapping.pop("frame_count", None)
    skeleton_config = tmp_path / "skeleton.yaml"
    skeleton_config.write_text(yaml.safe_dump(skeleton_mapping, sort_keys=False), encoding="utf-8")
    skeleton = load_skeleton(skeleton_config)
    model_config = {
        "num_classes": 27,
        "in_channels": len(skeleton.channels),
        "channels": [32, 64, 128],
        "dropout": 0.05,
        "fourier_degree": 2,
    }
    torch.manual_seed(11)
    model = LiteMyoNet(
        skeleton=skeleton,
        num_classes=model_config["num_classes"],
        in_channels=model_config["in_channels"],
        channels=model_config["channels"],
        dropout=model_config["dropout"],
        fourier_degree=model_config["fourier_degree"],
    ).eval()
    checkpoint = tmp_path / "student.pth"
    torch.save({
        "model_type": "student",
        "model_config": model_config,
        "skeleton_config": skeleton_mapping,
        "model_state_dict": model.state_dict(),
    }, checkpoint)
    return checkpoint, (len(skeleton.channels), skeleton.frame_count, skeleton.point_count)


def test_export_has_one_dynamic_joints_input_and_matches_pytorch(tmp_path):
    """Export dynamic-batch ONNX and compare runtime logits with PyTorch."""
    pytest.importorskip("onnx")
    onnxruntime = pytest.importorskip("onnxruntime")
    torch.set_num_threads(1)
    checkpoint, dimensions = _write_student_checkpoint(tmp_path)
    destination = tmp_path / "student.onnx"

    result = export_checkpoint_to_onnx(
        checkpoint, destination, batch_sizes=(1, 2, 3), atol=1e-4, rtol=1e-3
    )

    assert destination.is_file()
    assert result["input_name"] == "joints"
    assert result["output_name"] == "logits"
    assert result["input_shape"] == [None, *dimensions]
    assert result["fourier_degree"] == 2
    assert result["validated"] is True
    assert result["batch_sizes"] == [1, 2, 3]
    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        str(destination), sess_options=session_options, providers=["CPUExecutionProvider"]
    )
    assert [item.name for item in session.get_inputs()] == ["joints"]
    assert [item.name for item in session.get_outputs()] == ["logits"]
    assert session.get_inputs()[0].shape[0] is None or isinstance(
        session.get_inputs()[0].shape[0], str
    )


def test_export_refuses_to_overwrite_without_explicit_flag(tmp_path):
    """Preserve an existing ONNX destination without overwrite authorization."""
    checkpoint = tmp_path / "not_loaded.pth"
    checkpoint.write_bytes(b"fixture")
    destination = tmp_path / "existing.onnx"
    destination.write_bytes(b"preserve")

    with pytest.raises(FileExistsError, match="already exists"):
        export_checkpoint_to_onnx(checkpoint, destination)
    assert destination.read_bytes() == b"preserve"


def test_export_requires_checkpoint_skeleton_frame_count(tmp_path):
    """Require frame_count in the checkpoint skeleton before export."""
    pytest.importorskip("onnx")
    checkpoint, _ = _write_student_checkpoint(tmp_path, include_frame_count=False)
    destination = tmp_path / "student.onnx"

    with pytest.raises(ValueError, match="Checkpoint skeleton must define frame_count"):
        export_checkpoint_to_onnx(checkpoint, destination, validate=False)

    assert not destination.exists()


def test_export_requires_checkpoint_and_onnx_suffix(tmp_path):
    """Require an existing checkpoint and an ONNX destination suffix."""
    missing = tmp_path / "missing.pth"
    with pytest.raises(FileNotFoundError, match="Checkpoint does not exist"):
        export_checkpoint_to_onnx(missing, tmp_path / "model.onnx")

    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"fixture")
    with pytest.raises(ValueError, match=".onnx extension"):
        export_checkpoint_to_onnx(checkpoint, tmp_path / "model.pt")
