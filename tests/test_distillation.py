from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lite_myonet.models.lite_myonet import STUDENT_FEATURE_KEYS
from lite_myonet.training.distillation import (
    FEATURE_MATCHES,
    STUDENT_FEATURE_NAMES,
    TEACHER_FEATURE_NAMES,
    DistillationLoss,
)


def _random_features(names, channel_base, *, requires_grad=False):
    """Create named random stage features with varied channel and time sizes."""
    features = {}
    for index, name in enumerate(names):
        channels = channel_base + index % 3
        frames = 3 + index % 3
        features[name] = torch.randn(
            2,
            channels,
            frames,
            4,
            dtype=torch.float64,
            requires_grad=requires_grad,
        )
    return features


def _manual_reference_components(student_logits, teacher_logits, students, teachers, adjacency):
    """Compute the original three distillation terms directly from their equations."""
    temperature = 4.0
    topology_temperature = 0.2
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)

    graph = adjacency.amax(dim=0) if adjacency.ndim == 3 else adjacency
    edge_mask = (graph > 0) | (graph.transpose(0, 1) > 0)
    edge_mask = edge_mask.clone()
    edge_mask.fill_diagonal_(False)

    student_order = tuple(STUDENT_FEATURE_NAMES)
    teacher_order = tuple(TEACHER_FEATURE_NAMES)
    student_indices = {name: index for index, name in enumerate(student_order)}
    teacher_indices = {name: index for index, name in enumerate(teacher_order)}
    static_terms = []
    dynamic_terms = []
    weight_sum = sum(match.weight for match in FEATURE_MATCHES)
    for match in FEATURE_MATCHES:
        student = students[student_order[student_indices[match.student]]]
        teacher = teachers[teacher_order[teacher_indices[match.teacher]]].detach()

        student_nodes = F.normalize(student.mean(dim=2).transpose(1, 2), p=2, dim=-1, eps=1e-8)
        teacher_nodes = F.normalize(teacher.mean(dim=2).transpose(1, 2), p=2, dim=-1, eps=1e-8)
        student_affinity = torch.bmm(student_nodes, student_nodes.transpose(1, 2))
        teacher_affinity = torch.bmm(teacher_nodes, teacher_nodes.transpose(1, 2))
        static = (student_affinity[:, edge_mask] - teacher_affinity[:, edge_mask]).square().mean()
        static_terms.append(match.weight * static)

        student_motion = (student[:, :, 1:, :] - student[:, :, :-1, :]).abs().mean(dim=2)
        teacher_motion = (teacher[:, :, 1:, :] - teacher[:, :, :-1, :]).abs().mean(dim=2)
        student_nodes = F.normalize(student_motion.transpose(1, 2), p=2, dim=-1, eps=1e-8)
        teacher_nodes = F.normalize(teacher_motion.transpose(1, 2), p=2, dim=-1, eps=1e-8)
        student_similarity = torch.bmm(student_nodes, student_nodes.transpose(1, 2))
        teacher_similarity = torch.bmm(teacher_nodes, teacher_nodes.transpose(1, 2))
        dynamic = F.kl_div(
            F.log_softmax(student_similarity / topology_temperature, dim=-1),
            F.softmax(teacher_similarity / topology_temperature, dim=-1).detach(),
            reduction="none",
        ).sum(dim=-1).mean().clamp_min(0.0)
        dynamic_terms.append(match.weight * dynamic)

    static_loss = sum(static_terms) / weight_sum
    dynamic_loss = sum(dynamic_terms) / weight_sum
    return {
        "loss_total": kd_loss + static_loss + dynamic_loss,
        "loss_kd": kd_loss,
        "loss_static_adjacency": static_loss,
        "loss_dynamic_similarity": dynamic_loss,
    }


def test_feature_names_and_default_terms_match_the_reference_equations():
    """Check the OSS model key/order adapters and numeric loss equivalence."""
    assert STUDENT_FEATURE_NAMES == STUDENT_FEATURE_KEYS
    assert TEACHER_FEATURE_NAMES == (
        "joint_spatial_stage1",
        "joint_spatial_stage2",
        "joint_spatial_stage3",
        "joint_temporal_stage1",
        "joint_temporal_stage2",
        "joint_temporal_stage3",
        "bone_vector_stage1",
        "bone_vector_stage2",
        "bone_vector_stage3",
    )

    torch.manual_seed(702)
    student_logits = torch.randn(2, 5, dtype=torch.float64)
    teacher_logits = torch.randn(2, 5, dtype=torch.float64)
    students = _random_features(STUDENT_FEATURE_NAMES, 3)
    teachers = _random_features(TEACHER_FEATURE_NAMES, 5)
    student_sequence = tuple(students[name] for name in STUDENT_FEATURE_NAMES)
    teacher_sequence = tuple(teachers[name] for name in TEACHER_FEATURE_NAMES)
    adjacency = torch.zeros(2, 4, 4, dtype=torch.float64)
    adjacency[0, 0, 1] = adjacency[0, 1, 0] = 1.0
    adjacency[1, 1, 2] = adjacency[1, 2, 1] = 1.0
    adjacency[1, 2, 3] = adjacency[1, 3, 2] = 1.0
    adjacency.diagonal(dim1=-2, dim2=-1).fill_(1.0)

    actual = DistillationLoss()(
        student_logits,
        teacher_logits,
        student_sequence,
        teacher_sequence,
        adjacency,
    )
    expected = _manual_reference_components(
        student_logits,
        teacher_logits,
        students,
        teachers,
        adjacency,
    )
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-12, atol=1e-12)


def test_distillation_gradients_reach_student_only():
    """Verify detached teacher outputs do not receive gradients from distillation."""
    torch.manual_seed(51)
    student_logits = torch.randn(2, 4, requires_grad=True)
    teacher_logits = torch.randn(2, 4, requires_grad=True)
    students = _random_features(STUDENT_FEATURE_NAMES, 4, requires_grad=True)
    teachers = _random_features(TEACHER_FEATURE_NAMES, 6, requires_grad=True)
    adjacency = torch.eye(4)
    adjacency[0, 1] = adjacency[1, 0] = 1.0
    adjacency[1, 2] = adjacency[2, 1] = 1.0
    adjacency[2, 3] = adjacency[3, 2] = 1.0

    losses = DistillationLoss()(
        student_logits,
        teacher_logits,
        students,
        tuple(teachers[name] for name in TEACHER_FEATURE_NAMES),
        adjacency,
    )
    losses["loss_total"].backward()

    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()
    for feature in students.values():
        assert feature.grad is not None
        assert torch.isfinite(feature.grad).all()
    assert teacher_logits.grad is None
    for feature in teachers.values():
        assert feature.grad is None
