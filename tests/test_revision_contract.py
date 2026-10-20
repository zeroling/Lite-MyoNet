"""Small CPU checks for published model/config/topology contracts."""
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lite_myonet.models import LiteMyoNet, MyoFormerStudent, MyoFormerTeacher
from lite_myonet.runtime import resume_readable_only
from lite_myonet.training.distillation import DistillationLoss, FEATURE_MATCHES
from lite_myonet.training.student_trainer import validate_student_config
from lite_myonet.visualization.topology import affinity


def test_default_student_and_readable_resume(monkeypatch):
    cfg = yaml.safe_load((ROOT / "configs/student.yaml").read_text(encoding="utf-8"))
    validate_student_config(cfg)
    assert cfg["model"]["fourier_degree"] == 2
    assert cfg["seeds"] == [11, 21, 31, 41, 51]
    assert cfg["precision"] == "fp32" and cfg["batch_size"] == 16
    assert LiteMyoNet is MyoFormerStudent
    monkeypatch.setenv("LITE_MYONET_RESUME_READABLE_ONLY", "0")
    assert resume_readable_only()


def test_student_forward_and_actual_topology_route():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = MyoFormerStudent(fourier_degree=2).eval()
    joints = torch.randn(1, 3, 120, 23)
    bones = torch.zeros_like(joints)
    from lite_myonet.models.topology import bones as edges
    child, parent = zip(*edges)
    bones[..., child] = joints[..., child] - joints[..., parent]
    with torch.inference_mode():
        logits, features = model(joints, bones, return_features=True)
    assert logits.shape == (1, 33) and torch.isfinite(logits).all()
    assert features["joint_stage3"].shape == (1, 128, 30, 23)
    route = [(m.teacher, m.weight) for m in FEATURE_MATCHES if m.student == "joint.stage3"]
    assert route == [("joint_spatial.stage3", .5), ("joint_temporal.stage3", .5)]


def test_fixed_visual_affinity_matches_training_kl():
    torch.manual_seed(3)
    student, sd = affinity(torch.randn(2, 8, 5, 23))
    teacher, td = affinity(torch.randn(2, 16, 3, 23))
    expected = torch.nn.functional.kl_div((student + 1e-8).log(), teacher, reduction="batchmean")
    actual = DistillationLoss._rbf_kl(sd, td, torch.zeros(1))
    assert torch.allclose(student.sum(-1), torch.ones(2, 23), atol=1e-6)
    assert torch.allclose(actual, expected, atol=1e-6)
