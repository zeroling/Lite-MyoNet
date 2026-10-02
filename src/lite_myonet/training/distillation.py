"""Logit and topology distillation losses for the Lite-MyoNet models."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


STUDENT_FEATURE_NAMES: Tuple[str, ...] = (
    "joint_stage1",
    "bone_vector_stage1",
    "joint_stage2",
    "bone_vector_stage2",
    "joint_stage3",
    "bone_vector_stage3",
)

TEACHER_FEATURE_NAMES: Tuple[str, ...] = (
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


@dataclass(frozen=True)
class FeatureMatch:
    """Describe one named student-to-teacher feature correspondence."""

    student: str
    teacher: str
    weight: float = 1.0


FEATURE_MATCHES: Tuple[FeatureMatch, ...] = (
    FeatureMatch("joint_stage1", "joint_spatial_stage1", 0.5),
    FeatureMatch("joint_stage1", "joint_temporal_stage1", 0.5),
    FeatureMatch("joint_stage2", "joint_spatial_stage2", 0.5),
    FeatureMatch("joint_stage2", "joint_temporal_stage2", 0.5),
    FeatureMatch("joint_stage3", "joint_spatial_stage3", 0.5),
    FeatureMatch("joint_stage3", "joint_temporal_stage3", 0.5),
    FeatureMatch("bone_vector_stage1", "bone_vector_stage1", 1.0),
    FeatureMatch("bone_vector_stage2", "bone_vector_stage2", 1.0),
    FeatureMatch("bone_vector_stage3", "bone_vector_stage3", 1.0),
)

FeatureInput = Union[Sequence[torch.Tensor], Mapping[str, torch.Tensor]]


def _finite_positive(value: float, name: str) -> float:
    """Return a finite positive scalar or raise a descriptive error."""
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a finite positive number.") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive; got {value!r}.")
    return result


def _finite_nonnegative(value: float, name: str) -> float:
    """Return a finite nonnegative scalar or raise a descriptive error."""
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a finite nonnegative number.") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative; got {value!r}.")
    return result


def _work_dtype(*tensors: torch.Tensor) -> torch.dtype:
    """Choose a stable promoted floating-point dtype for loss calculations."""
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _validate_logits(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> None:
    """Validate the shape, dtype, and device contract for logit distillation."""
    for name, logits in (("student_logits", student_logits), ("teacher_logits", teacher_logits)):
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if logits.ndim != 2 or logits.shape[0] <= 0 or logits.shape[1] <= 0:
            raise ValueError(f"{name} must be a nonempty [B,K] tensor; got {tuple(logits.shape)}.")
        if not logits.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype.")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "Student and teacher logits must have equal shapes; "
            f"got {tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}."
        )
    if student_logits.device != teacher_logits.device:
        raise ValueError("Student and teacher logits must be on the same device.")


def compute_logit_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """Compute temperature-scaled KL logit distillation with the T-squared factor."""
    temperature = _finite_positive(temperature, "temperature")
    _validate_logits(student_logits, teacher_logits)
    dtype = _work_dtype(student_logits, teacher_logits)
    student = student_logits.to(dtype=dtype)
    teacher = teacher_logits.detach().to(dtype=dtype)
    student_log_probability = F.log_softmax(student / temperature, dim=1)
    teacher_probability = F.softmax(teacher / temperature, dim=1)
    return F.kl_div(
        student_log_probability,
        teacher_probability,
        reduction="batchmean",
    ) * (temperature ** 2)


def _as_named_features(
    features: FeatureInput,
    expected_names: Tuple[str, ...],
    label: str,
) -> Dict[str, torch.Tensor]:
    """Convert an ordered feature sequence or exact-key mapping to a dictionary."""
    if isinstance(features, Mapping):
        actual = set(features.keys())
        expected = set(expected_names)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            raise ValueError(f"{label} feature keys differ from the contract: missing={missing}, extra={extra}.")
        ordered = {name: features[name] for name in expected_names}
    else:
        if isinstance(features, (str, bytes)) or not isinstance(features, Sequence):
            raise TypeError(f"{label} features must be a sequence or named mapping.")
        if len(features) != len(expected_names):
            raise ValueError(
                f"{label} must provide exactly {len(expected_names)} stage features; got {len(features)}."
            )
        ordered = dict(zip(expected_names, features))

    common_batch = None
    common_nodes = None
    for name, tensor in ordered.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{label} feature {name} must be a torch.Tensor.")
        if tensor.ndim != 4:
            raise ValueError(f"{label} feature {name} must be [B,C,T,V]; got {tuple(tensor.shape)}.")
        if any(int(size) <= 0 for size in tensor.shape):
            raise ValueError(f"Every dimension of {label} feature {name} must be positive.")
        if not tensor.is_floating_point():
            raise TypeError(f"{label} feature {name} must use a floating-point dtype.")
        if common_batch is None:
            common_batch, common_nodes = tensor.shape[0], tensor.shape[3]
        elif tensor.shape[0] != common_batch:
            raise ValueError(f"All {label} stages must have the same batch size.")
        elif tensor.shape[3] != common_nodes:
            raise ValueError(f"All {label} stages must have the same number of nodes V.")
    return ordered


def _validate_feature_contract(
    student_logits: torch.Tensor,
    student_features: FeatureInput,
    teacher_features: FeatureInput,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Validate named stage features and the fixed six-to-nine feature mapping."""
    students = _as_named_features(student_features, STUDENT_FEATURE_NAMES, "Student")
    teachers = _as_named_features(teacher_features, TEACHER_FEATURE_NAMES, "Teacher")
    for label, features in (("Student", students), ("Teacher", teachers)):
        for name, feature in features.items():
            if feature.shape[0] != student_logits.shape[0]:
                raise ValueError(f"{label} feature {name} batch size does not match the logits.")
            if feature.device != student_logits.device:
                raise ValueError(f"{label} feature {name} must be on the logits device.")
    for match in FEATURE_MATCHES:
        student = students[match.student]
        teacher = teachers[match.teacher]
        if student.shape[3] != teacher.shape[3]:
            raise ValueError(
                f"Node count differs for feature match {match.student} <- {match.teacher}: "
                f"{student.shape[3]} vs {teacher.shape[3]}."
            )
    return students, teachers


def _physical_edge_mask(
    adjacency: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Build an undirected, non-self edge mask from a physical adjacency tensor."""
    if not isinstance(adjacency, torch.Tensor):
        raise TypeError("physical_adjacency must be a torch.Tensor.")
    if adjacency.device != device:
        raise ValueError("physical_adjacency must be on the same device as the features.")
    if not adjacency.is_floating_point():
        raise TypeError("physical_adjacency must use a floating-point dtype.")
    if adjacency.ndim == 2:
        if tuple(adjacency.shape) != (num_nodes, num_nodes):
            raise ValueError(f"physical_adjacency must be [{num_nodes},{num_nodes}].")
        graph = adjacency
    elif adjacency.ndim == 3:
        if adjacency.shape[0] <= 0 or tuple(adjacency.shape[1:]) != (num_nodes, num_nodes):
            raise ValueError(f"Partitioned physical_adjacency must be [K,{num_nodes},{num_nodes}].")
        graph = adjacency.amax(dim=0)
    else:
        raise ValueError("physical_adjacency must be [V,V] or [K,V,V].")
    edges = (graph > 0) | (graph.transpose(0, 1) > 0)
    edges = edges.clone()
    edges.fill_diagonal_(False)
    return edges


def _static_edge_affinity_loss(
    student_feature: torch.Tensor,
    teacher_feature: torch.Tensor,
    edge_mask: torch.Tensor,
) -> torch.Tensor:
    """Match student and detached teacher cosine affinities on physical edges."""
    dtype = _work_dtype(student_feature, teacher_feature)
    student = student_feature.to(dtype=dtype).mean(dim=2).transpose(1, 2)
    teacher = teacher_feature.detach().to(dtype=dtype).mean(dim=2).transpose(1, 2)
    student = F.normalize(student, p=2, dim=-1, eps=1e-8)
    teacher = F.normalize(teacher, p=2, dim=-1, eps=1e-8)
    student_affinity = torch.bmm(student, student.transpose(1, 2))
    teacher_affinity = torch.bmm(teacher, teacher.transpose(1, 2))
    return (student_affinity[:, edge_mask] - teacher_affinity[:, edge_mask]).square().mean()


def _dynamic_similarity_loss(
    student_feature: torch.Tensor,
    teacher_feature: torch.Tensor,
    temperature: float,
    match_name: str,
) -> torch.Tensor:
    """Match temporal-motion similarity distributions with a teacher-target KL loss."""
    if student_feature.shape[2] < 2 or teacher_feature.shape[2] < 2:
        raise ValueError(
            f"Dynamic topology requires at least two frames for {match_name}; "
            f"got {student_feature.shape[2]} and {teacher_feature.shape[2]}."
        )
    dtype = _work_dtype(student_feature, teacher_feature)
    student_delta = student_feature.to(dtype=dtype)
    teacher_delta = teacher_feature.detach().to(dtype=dtype)
    student_motion = (student_delta[:, :, 1:, :] - student_delta[:, :, :-1, :]).abs().mean(dim=2)
    teacher_motion = (teacher_delta[:, :, 1:, :] - teacher_delta[:, :, :-1, :]).abs().mean(dim=2)
    student_nodes = F.normalize(student_motion.transpose(1, 2), p=2, dim=-1, eps=1e-8)
    teacher_nodes = F.normalize(teacher_motion.transpose(1, 2), p=2, dim=-1, eps=1e-8)
    student_similarity = torch.bmm(student_nodes, student_nodes.transpose(1, 2))
    teacher_similarity = torch.bmm(teacher_nodes, teacher_nodes.transpose(1, 2))
    student_log_probability = F.log_softmax(student_similarity / temperature, dim=-1)
    teacher_probability = F.softmax(teacher_similarity / temperature, dim=-1).detach()
    loss = F.kl_div(student_log_probability, teacher_probability, reduction="none")
    return loss.sum(dim=-1).mean().clamp_min(0.0)


class DistillationLoss(nn.Module):
    """Combine logit KL with physical-edge and dynamic-topology feature losses.

    Student features follow the open-source model's six-key mapping and teacher
    features follow its nine-item stream order. All feature maps use [B,C,T,V].
    Teacher targets are detached inside each loss; the training loop should also
    keep the teacher model in evaluation mode with gradients disabled.
    """

    def __init__(
        self,
        kd_temperature: float = 4.0,
        topology_temperature: float = 0.2,
        kd_weight: float = 1.0,
        static_adjacency_weight: float = 1.0,
        dynamic_similarity_weight: float = 1.0,
    ) -> None:
        """Store validated temperatures and weights for the three loss terms."""
        super().__init__()
        self.kd_temperature = _finite_positive(kd_temperature, "kd_temperature")
        self.topology_temperature = _finite_positive(topology_temperature, "topology_temperature")
        self.kd_weight = _finite_nonnegative(kd_weight, "kd_weight")
        self.static_adjacency_weight = _finite_nonnegative(
            static_adjacency_weight, "static_adjacency_weight"
        )
        self.dynamic_similarity_weight = _finite_nonnegative(
            dynamic_similarity_weight, "dynamic_similarity_weight"
        )

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor | None = None,
        student_features: FeatureInput | None = None,
        teacher_features: FeatureInput | None = None,
        physical_adjacency: torch.Tensor | None = None,
        check_finite: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Return total and unweighted component losses for one training batch."""
        if not isinstance(student_logits, torch.Tensor):
            raise TypeError("student_logits must be a torch.Tensor.")
        if student_logits.ndim != 2 or student_logits.shape[0] <= 0 or student_logits.shape[1] <= 0:
            raise ValueError("student_logits must be a nonempty [B,K] tensor.")
        if not student_logits.is_floating_point():
            raise TypeError("student_logits must use a floating-point dtype.")

        need_topology = self.static_adjacency_weight > 0.0 or self.dynamic_similarity_weight > 0.0
        if self.kd_weight > 0.0:
            if teacher_logits is None:
                raise ValueError("teacher_logits must be provided when kd_weight is positive.")
            _validate_logits(student_logits, teacher_logits)

        students: Dict[str, torch.Tensor] = {}
        teachers: Dict[str, torch.Tensor] = {}
        if need_topology:
            if student_features is None:
                raise ValueError("student_features must be provided when topology losses are enabled.")
            if teacher_features is None:
                raise ValueError("teacher_features must be provided when topology losses are enabled.")
            students, teachers = _validate_feature_contract(
                student_logits,
                student_features,
                teacher_features,
            )

        edge_mask = None
        if self.static_adjacency_weight > 0.0:
            if physical_adjacency is None:
                raise ValueError(
                    "physical_adjacency must be provided when static adjacency loss is enabled."
                )
            num_nodes = next(iter(students.values())).shape[3]
            edge_mask = _physical_edge_mask(
                physical_adjacency,
                num_nodes,
                student_logits.device,
            )

        zero = student_logits.to(dtype=_work_dtype(student_logits)).sum() * 0.0
        static_terms = []
        dynamic_terms = []
        weight_sum = sum(match.weight for match in FEATURE_MATCHES)
        for match in FEATURE_MATCHES if need_topology else ():
            student_feature = students[match.student]
            if self.static_adjacency_weight > 0.0:
                static_terms.append(
                    match.weight
                    * _static_edge_affinity_loss(
                        student_feature,
                        teachers[match.teacher],
                        edge_mask,
                    )
                )
            if self.dynamic_similarity_weight > 0.0:
                dynamic_terms.append(
                    match.weight
                    * _dynamic_similarity_loss(
                        student_feature,
                        teachers[match.teacher],
                        self.topology_temperature,
                        f"{match.student} <- {match.teacher}",
                    )
                )

        static_loss = sum(static_terms) / weight_sum if static_terms else zero
        dynamic_loss = sum(dynamic_terms) / weight_sum if dynamic_terms else zero
        kd_loss = (
            compute_logit_kd(student_logits, teacher_logits, self.kd_temperature)
            if self.kd_weight > 0.0
            else zero
        )
        total = (
            self.kd_weight * kd_loss
            + self.static_adjacency_weight * static_loss
            + self.dynamic_similarity_weight * dynamic_loss
        )
        components = {
            "loss_total": total,
            "loss_kd": kd_loss,
            "loss_static_adjacency": static_loss,
            "loss_dynamic_similarity": dynamic_loss,
        }
        if check_finite and not bool(torch.isfinite(total).all().item()):
            raise FloatingPointError("Distillation total loss contains NaN or Inf.")
        return components


__all__ = [
    "DistillationLoss",
    "FEATURE_MATCHES",
    "FeatureMatch",
    "STUDENT_FEATURE_NAMES",
    "TEACHER_FEATURE_NAMES",
    "compute_logit_kd",
]
