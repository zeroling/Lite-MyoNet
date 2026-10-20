"""Stable logit and named-stage topology distillation losses.

The feature ordering is an explicit contract with ``MyoFormerTeacher``:
teacher features are grouped by stream (three stages per stream), while student
features are interleaved by stage (joint, bone).  Two independent RBF log bandwidths are learned through the student loss.
"""

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


STUDENT_FEATURE_NAMES: Tuple[str, ...] = (
    "joint.stage1",
    "bone.stage1",
    "joint.stage2",
    "bone.stage2",
    "joint.stage3",
    "bone.stage3",
)

TEACHER_FEATURE_NAMES: Tuple[str, ...] = (
    "joint_spatial.stage1",
    "joint_spatial.stage2",
    "joint_spatial.stage3",
    "joint_temporal.stage1",
    "joint_temporal.stage2",
    "joint_temporal.stage3",
    "bone_vector.stage1",
    "bone_vector.stage2",
    "bone_vector.stage3",
)


@dataclass(frozen=True)
class FeatureMatch:
    """One named student-to-teacher feature correspondence."""

    student: str
    teacher: str
    weight: float = 1.0


# Joint stages distill equally from the spatial and temporal joint streams;
# bone stages distill one-to-one from the teacher bone-vector stream.
FEATURE_MATCHES: Tuple[FeatureMatch, ...] = (
    FeatureMatch("joint.stage1", "joint_spatial.stage1", 0.5),
    FeatureMatch("joint.stage1", "joint_temporal.stage1", 0.5),
    FeatureMatch("joint.stage2", "joint_spatial.stage2", 0.5),
    FeatureMatch("joint.stage2", "joint_temporal.stage2", 0.5),
    FeatureMatch("joint.stage3", "joint_spatial.stage3", 0.5),
    FeatureMatch("joint.stage3", "joint_temporal.stage3", 0.5),
    FeatureMatch("bone.stage1", "bone_vector.stage1", 1.0),
    FeatureMatch("bone.stage2", "bone_vector.stage2", 1.0),
    FeatureMatch("bone.stage3", "bone_vector.stage3", 1.0),
)

FeatureInput = Union[Sequence[torch.Tensor], Mapping[str, torch.Tensor]]


def _finite_positive(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise TypeError("{} 必须是有限正数。".format(name))
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("{} 必须是有限正数，实际为 {!r}。".format(name, value))
    return result


def _finite_nonnegative(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise TypeError("{} 必须是有限非负数。".format(name))
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("{} 必须是有限非负数，实际为 {!r}。".format(name, value))
    return result


def _work_dtype(*tensors: torch.Tensor) -> torch.dtype:
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _validate_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> None:
    for name, logits in (("student_logits", student_logits), ("teacher_logits", teacher_logits)):
        if not isinstance(logits, torch.Tensor):
            raise TypeError("{} 必须是 torch.Tensor。".format(name))
        if logits.ndim != 2 or logits.shape[0] <= 0 or logits.shape[1] <= 0:
            raise ValueError(
                "{} 必须为非空 [B,K] 张量，实际形状为 {}。".format(name, tuple(logits.shape))
            )
        if not logits.is_floating_point():
            raise TypeError("{} 必须使用浮点 dtype。".format(name))
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "学生和教师 logits 形状必须一致，实际分别为 {} 和 {}。".format(
                tuple(student_logits.shape), tuple(teacher_logits.shape)
            )
        )
    if student_logits.device != teacher_logits.device:
        raise ValueError("学生和教师 logits 必须位于同一设备。")


def compute_logit_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """计算 KL(logit KD)，并按标准知识蒸馏约定乘以 ``T ** 2``。"""

    temperature = _finite_positive(temperature, "temperature")
    _validate_logits(student_logits, teacher_logits)
    dtype = _work_dtype(student_logits, teacher_logits)
    student = student_logits.to(dtype=dtype)
    teacher = teacher_logits.detach().to(dtype=dtype)
    log_student = F.log_softmax(student / temperature, dim=1)
    probability_teacher = F.softmax(teacher / temperature, dim=1)
    loss = F.kl_div(log_student, probability_teacher, reduction="batchmean")
    loss = loss * (temperature ** 2)
    return loss


def _as_named_features(
    features: FeatureInput,
    expected_names: Tuple[str, ...],
    label: str,
) -> Dict[str, torch.Tensor]:
    if isinstance(features, Mapping):
        actual = set(features.keys())
        expected = set(expected_names)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            raise ValueError(
                "{}特征名不符合契约；缺少={}，多余={}。".format(label, missing, extra)
            )
        ordered = {name: features[name] for name in expected_names}
    else:
        if isinstance(features, (str, bytes)) or not isinstance(features, Sequence):
            raise TypeError("{} features 必须是特征序列或具名映射。".format(label))
        if len(features) != len(expected_names):
            raise ValueError(
                "{}必须严格提供 {} 个阶段特征，实际为 {}。".format(
                    label, len(expected_names), len(features)
                )
            )
        ordered = dict(zip(expected_names, features))

    common_batch = None
    common_nodes = None
    for name, tensor in ordered.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("{} 特征 {} 必须是 torch.Tensor。".format(label, name))
        if tensor.ndim != 4:
            raise ValueError(
                "{} 特征 {} 必须为 [B,C,T,V]，实际形状为 {}。".format(
                    label, name, tuple(tensor.shape)
                )
            )
        if any(int(size) <= 0 for size in tensor.shape):
            raise ValueError("{} 特征 {} 的每一维都必须大于 0。".format(label, name))
        if not tensor.is_floating_point():
            raise TypeError("{} 特征 {} 必须使用浮点 dtype。".format(label, name))
        if common_batch is None:
            common_batch = tensor.shape[0]
            common_nodes = tensor.shape[3]
        elif tensor.shape[0] != common_batch:
            raise ValueError("{} 的所有阶段必须具有相同 batch 大小。".format(label))
        elif tensor.shape[3] != common_nodes:
            raise ValueError("{} 的所有阶段必须具有相同关节数 V。".format(label))
    return ordered


def _validate_feature_contract(
    student_logits: torch.Tensor,
    student_features: FeatureInput,
    teacher_features: FeatureInput,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    students = _as_named_features(student_features, STUDENT_FEATURE_NAMES, "学生")
    teachers = _as_named_features(teacher_features, TEACHER_FEATURE_NAMES, "教师")
    for label, features in (("学生", students), ("教师", teachers)):
        for name, feature in features.items():
            if feature.shape[0] != student_logits.shape[0]:
                raise ValueError(
                    "{} 特征 {} 的 batch 大小 {} 与 logits 的 {} 不一致。".format(
                        label, name, feature.shape[0], student_logits.shape[0]
                    )
                )
            if feature.device != student_logits.device:
                raise ValueError("{} 特征 {} 与 logits 必须位于同一设备。".format(label, name))
    for match in FEATURE_MATCHES:
        student = students[match.student]
        teacher = teachers[match.teacher]
        if student.shape[3] != teacher.shape[3]:
            raise ValueError(
                "特征映射 {} <- {} 的关节数不一致：{} 与 {}。".format(
                    match.student, match.teacher, student.shape[3], teacher.shape[3]
                )
            )
    return students, teachers


def _physical_edge_mask(
    adjacency: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(adjacency, torch.Tensor):
        raise TypeError("physical_adjacency 必须是 torch.Tensor。")
    if adjacency.device != device:
        raise ValueError("physical_adjacency 必须与输入特征位于同一设备。")
    if not adjacency.is_floating_point():
        raise TypeError("physical_adjacency 必须使用浮点 dtype。")
    if adjacency.ndim == 2:
        if tuple(adjacency.shape) != (num_nodes, num_nodes):
            raise ValueError(
                "physical_adjacency 必须为 [{0},{0}]，实际形状为 {1}。".format(
                    num_nodes, tuple(adjacency.shape)
                )
            )
        graph = adjacency
    elif adjacency.ndim == 3:
        if adjacency.shape[0] <= 0 or tuple(adjacency.shape[1:]) != (num_nodes, num_nodes):
            raise ValueError(
                "分区 physical_adjacency 必须为 [K,{0},{0}] 且 K>0，实际形状为 {1}。".format(
                    num_nodes, tuple(adjacency.shape)
                )
            )
        graph = adjacency.amax(dim=0)
    else:
        raise ValueError("physical_adjacency 必须是 [V,V] 或 [K,V,V] 张量。")
    edges = (graph > 0) | (graph.transpose(0, 1) > 0)
    edges = edges.clone()
    edges.fill_diagonal_(False)
    return edges


def _static_edge_affinity_loss(
    student_feature: torch.Tensor,
    teacher_feature: torch.Tensor,
    edge_mask: torch.Tensor,
) -> torch.Tensor:
    dtype = _work_dtype(student_feature, teacher_feature)
    student = student_feature.to(dtype=dtype).mean(dim=2).transpose(1, 2)
    teacher = teacher_feature.detach().to(dtype=dtype).mean(dim=2).transpose(1, 2)
    student = F.normalize(student, p=2, dim=-1, eps=1e-8)
    teacher = F.normalize(teacher, p=2, dim=-1, eps=1e-8)
    student_affinity = torch.bmm(student, student.transpose(1, 2))
    teacher_affinity = torch.bmm(teacher, teacher.transpose(1, 2))
    return (student_affinity[:, edge_mask] - teacher_affinity[:, edge_mask]).square().mean()


def _static_edge_affinity_loss_cached(
    student_feature: torch.Tensor,
    teacher_edge_affinity: torch.Tensor,
    edge_mask: torch.Tensor,
) -> torch.Tensor:
    dtype = _work_dtype(student_feature, teacher_edge_affinity)
    student = student_feature.to(dtype=dtype).mean(dim=2).transpose(1, 2)
    student = F.normalize(student, p=2, dim=-1, eps=1e-8)
    student_affinity = torch.bmm(student, student.transpose(1, 2))
    student_edges = student_affinity[:, edge_mask]
    teacher_edge_affinity = teacher_edge_affinity.detach().to(dtype=dtype)
    if student_edges.shape != teacher_edge_affinity.shape:
        raise ValueError(
            "静态缓存目标 shape {} 与学生物理边 shape {} 不匹配。".format(
                tuple(teacher_edge_affinity.shape), tuple(student_edges.shape)
            )
        )
    return (student_edges - teacher_edge_affinity).square().mean()


def _dynamic_similarity_loss(
    student_feature: torch.Tensor,
    teacher_feature: torch.Tensor,
    temperature: float,
    match_name: str,
) -> torch.Tensor:
    if student_feature.shape[2] < 2 or teacher_feature.shape[2] < 2:
        raise ValueError(
            "动态拓扑要求映射 {} 的学生和教师 T 都至少为 2，实际为 {} 和 {}。".format(
                match_name, student_feature.shape[2], teacher_feature.shape[2]
            )
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
    student_log_prob = F.log_softmax(student_similarity / temperature, dim=-1)
    teacher_prob = F.softmax(teacher_similarity / temperature, dim=-1).detach()
    # 按 batch 和节点对取均值，避免 KL 数值随 V^2 线性放大。
    loss = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=-1).mean()
    return loss.clamp_min(0.0)


def _dynamic_similarity_loss_cached(
    student_feature: torch.Tensor,
    teacher_similarity: torch.Tensor,
    temperature: float,
    match_name: str,
) -> torch.Tensor:
    if student_feature.shape[2] < 2:
        raise ValueError(
            "动态拓扑要求映射 {} 的学生 T 至少为 2，实际为 {}。".format(
                match_name, student_feature.shape[2]
            )
        )
    dtype = _work_dtype(student_feature, teacher_similarity)
    student_delta = student_feature.to(dtype=dtype)
    student_motion = (student_delta[:, :, 1:, :] - student_delta[:, :, :-1, :]).abs().mean(dim=2)
    student_nodes = F.normalize(student_motion.transpose(1, 2), p=2, dim=-1, eps=1e-8)
    student_similarity = torch.bmm(student_nodes, student_nodes.transpose(1, 2))
    student_log_prob = F.log_softmax(student_similarity / temperature, dim=-1)
    teacher_prob = teacher_similarity.detach().to(dtype=dtype)
    loss = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=-1).mean()
    return loss.clamp_min(0.0)


def topology_squared_distances(feature, dynamic=False):
    """Full V x V RBF sufficient statistics; bandwidth remains trainable."""
    feature = feature.to(dtype=_work_dtype(feature))
    if dynamic:
        if feature.shape[2] < 2:
            raise ValueError("动态拓扑要求至少两个时间帧。")
        nodes = (feature[:, :, 1:] - feature[:, :, :-1]).abs().mean(dim=2)
    else:
        nodes = feature.mean(dim=2)
    nodes = F.normalize(nodes, p=2, dim=1)
    return torch.cdist(nodes.transpose(1, 2), nodes.transpose(1, 2), p=2).square()


class DistillationLoss(nn.Module):
    """Original ST-GTP: full RBF topology, batchmean KL, stage sums.

    static/dynamic weights include the outer beta (main: 0.1 each).
    topology_temperature is retained as a metadata-compatible field; RBF uses
    the two learnable log_sigma scalars, both initialized to zero.
    """
    def __init__(self, kd_temperature=4.0, topology_temperature=0.2,
                 kd_weight=1.0, static_adjacency_weight=0.1,
                 dynamic_similarity_weight=0.1):
        super().__init__()
        self.kd_temperature=_finite_positive(kd_temperature,'kd_temperature')
        self.topology_temperature=_finite_positive(topology_temperature,'topology_temperature')
        self.kd_weight=_finite_nonnegative(kd_weight,'kd_weight')
        self.static_adjacency_weight=_finite_nonnegative(static_adjacency_weight,'static_adjacency_weight')
        self.dynamic_similarity_weight=_finite_nonnegative(dynamic_similarity_weight,'dynamic_similarity_weight')
        self.log_sigma_static=nn.Parameter(torch.zeros(1))
        self.log_sigma_dyn=nn.Parameter(torch.zeros(1))

    @staticmethod
    def _rbf_kl(student_distance, teacher_distance, rho):
        # Cached teacher distances are detached, but teacher probabilities
        # must be rebuilt using the current bandwidth on every update.
        denominator=rho.exp()+1e-8
        ps=F.softmax(-student_distance/denominator,dim=-1)
        with torch.no_grad():
            pt=F.softmax(-teacher_distance.detach()/denominator,dim=-1)
        return F.kl_div(torch.log(ps+1e-8),pt,reduction='batchmean')

    def forward(self, student_logits, teacher_logits=None, student_features=None,
                teacher_features=None, physical_adjacency=None,
                cached_teacher_targets=None, check_finite=True):
        if not isinstance(student_logits,torch.Tensor) or student_logits.ndim!=2:
            raise ValueError("student_logits 必须为 [B,K] 张量。")
        if teacher_features is not None and cached_teacher_targets is not None:
            raise ValueError("teacher_features 与 cached_teacher_targets 不能同时提供。")
        zero=student_logits.float().sum()*0.0
        kd=zero
        if self.kd_weight:
            if cached_teacher_targets is not None:
                teacher_logits=cached_teacher_targets.get('teacher_logits')
            if teacher_logits is None:
                raise ValueError("启用 KD 时需要教师 logits。")
            kd=compute_logit_kd(student_logits,teacher_logits,self.kd_temperature)
        stat=dyn=zero
        if self.static_adjacency_weight or self.dynamic_similarity_weight:
            if student_features is None:
                raise ValueError("启用拓扑蒸馏时需要学生阶段特征。")
            students=_as_named_features(student_features,STUDENT_FEATURE_NAMES,'学生')
            if cached_teacher_targets is None:
                if teacher_features is None:
                    raise ValueError("启用拓扑蒸馏时需要教师特征或缓存。")
                students,teachers=_validate_feature_contract(student_logits,student_features,teacher_features)
            # Compute each student stage only once; repeated mappings retain
            # the original weights 0.5/0.5 for joint and 1.0 for bone.
            sd={};dd={}
            for i,m in enumerate(FEATURE_MATCHES):
                f=students[m.student]
                for enabled,dynamic,key,rho in (
                    (self.static_adjacency_weight,False,'static_targets',self.log_sigma_static),
                    (self.dynamic_similarity_weight,True,'dynamic_targets',self.log_sigma_dyn)):
                    if not enabled: continue
                    pool=dd if dynamic else sd
                    if m.student not in pool:
                        pool[m.student]=topology_squared_distances(f,dynamic)
                    if cached_teacher_targets is None:
                        td=topology_squared_distances(teachers[m.teacher].detach(),dynamic)
                    else:
                        if key not in cached_teacher_targets:
                            raise ValueError("缓存缺少 "+key)
                        td=cached_teacher_targets[key][:,i].reshape(f.shape[0],f.shape[3],f.shape[3]).float()
                    if td.shape!=pool[m.student].shape:
                        raise ValueError("缓存或教师拓扑 shape 与学生不一致。")
                    term=m.weight*self._rbf_kl(pool[m.student],td,rho)
                    if dynamic: dyn=dyn+term
                    else: stat=stat+term
        total=self.kd_weight*kd+self.static_adjacency_weight*stat+self.dynamic_similarity_weight*dyn
        if check_finite and not bool(torch.isfinite(total).all()):
            raise FloatingPointError("蒸馏总损失出现 NaN 或 Inf。")
        return dict(loss_total=total,loss_kd=kd,loss_static_adjacency=stat,loss_dynamic_similarity=dyn)


__all__ = [
    "DistillationLoss",
    "FEATURE_MATCHES",
    "FeatureMatch",
    "STUDENT_FEATURE_NAMES",
    "TEACHER_FEATURE_NAMES",
    "compute_logit_kd",
]
