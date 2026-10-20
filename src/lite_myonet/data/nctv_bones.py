"""Differentiable derivation of bone vectors from NCTV joint coordinates."""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

from ..models.skeleton import SkeletonDefinition


def joints_to_bones(
    joints: Union[torch.Tensor, np.ndarray],
    skeleton: SkeletonDefinition,
) -> Union[torch.Tensor, np.ndarray]:
    """Create output-minus-reference bone vectors from an NCTV tensor or array."""
    if not isinstance(skeleton, SkeletonDefinition):
        raise TypeError("skeleton must be a SkeletonDefinition.")
    if isinstance(joints, torch.Tensor):
        if joints.ndim != 4:
            raise ValueError(f"joints must have NCTV shape, got {tuple(joints.shape)}.")
        if joints.shape[0] <= 0 or joints.shape[1] != len(skeleton.channels):
            raise ValueError("joints N and C dimensions do not match the skeleton contract.")
        if joints.shape[2] <= 0 or joints.shape[3] != skeleton.point_count:
            raise ValueError("joints T and V dimensions do not match the skeleton contract.")
        if skeleton.frame_count is not None and joints.shape[2] != skeleton.frame_count:
            raise ValueError(
                f"joints has T={joints.shape[2]}, expected frame_count={skeleton.frame_count}."
            )
        if not torch.is_floating_point(joints):
            raise TypeError("joints must use a floating-point dtype.")

        if not skeleton.bone_vector_edges:
            return torch.zeros_like(joints)
        output_indices = torch.tensor(
            [output for output, _ in skeleton.bone_vector_edges],
            dtype=torch.long,
            device=joints.device,
        )
        reference_indices = torch.tensor(
            [reference for _, reference in skeleton.bone_vector_edges],
            dtype=torch.long,
            device=joints.device,
        )
        vectors = joints.index_select(-1, output_indices) - joints.index_select(
            -1, reference_indices
        )
        return torch.zeros_like(joints).index_copy(-1, output_indices, vectors)

    if isinstance(joints, np.ndarray):
        if joints.ndim != 4:
            raise ValueError(f"joints must have NCTV shape, got {joints.shape}.")
        if joints.shape[0] <= 0 or joints.shape[1] != len(skeleton.channels):
            raise ValueError("joints N and C dimensions do not match the skeleton contract.")
        if joints.shape[2] <= 0 or joints.shape[3] != skeleton.point_count:
            raise ValueError("joints T and V dimensions do not match the skeleton contract.")
        if skeleton.frame_count is not None and joints.shape[2] != skeleton.frame_count:
            raise ValueError(
                f"joints has T={joints.shape[2]}, expected frame_count={skeleton.frame_count}."
            )
        if not np.issubdtype(joints.dtype, np.floating):
            raise TypeError("joints must use a floating-point dtype.")
        result = np.zeros_like(joints)
        for output_index, reference_index in skeleton.bone_vector_edges:
            result[..., output_index] = (
                joints[..., output_index] - joints[..., reference_index]
            )
        return result

    raise TypeError("joints must be a torch.Tensor or numpy.ndarray.")
