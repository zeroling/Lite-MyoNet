import argparse
import json
from pathlib import Path

import numpy as np

from ..models.topology import bones as PROJECT_TOPOLOGY_EDGES


def _normalise_axis(axis, ndim):
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"节点轴 {axis} 超出 {ndim} 维数组的范围。")
    return axis


def _validate_edges(edges, num_points):
    normalised = [(int(output), int(reference)) for output, reference in edges]
    for output, reference in normalised:
        if not 0 <= output < num_points or not 0 <= reference < num_points:
            raise ValueError(f"拓扑边 {(output, reference)} 超出节点范围。")
        if output == reference:
            raise ValueError(f"拓扑边 {(output, reference)} 不能自连接。")
    output_indices = [output for output, _ in normalised]
    expected_outputs = list(range(1, num_points))
    if sorted(output_indices) != expected_outputs:
        raise ValueError(
            "项目骨向量拓扑必须对除 0 以外的每个输出节点恰好赋值一次。"
        )
    return normalised


def compute_project_bone_vectors(
    joint_coordinates,
    *,
    point_axis=-1,
    edges=PROJECT_TOPOLOGY_EDGES,
    num_points=23,
):
    """按 Lite-MyoNet 项目拓扑由关节坐标计算骨向量。

    每条边表示 ``(输出节点, 参考节点)``，结果为前者坐标减去后者
    坐标，并存入输出节点索引。节点 0 的结果保持为零。
    """
    joints = np.asarray(joint_coordinates)
    if joints.ndim < 2:
        raise ValueError("关节坐标至少需要两个维度。")
    point_axis = _normalise_axis(point_axis, joints.ndim)
    if joints.shape[point_axis] != int(num_points):
        raise ValueError(
            f"节点轴长度为 {joints.shape[point_axis]}，期望 {num_points}。"
        )
    if not np.issubdtype(joints.dtype, np.number):
        raise TypeError(f"关节坐标必须是数值数组，实际为 {joints.dtype}。")
    if not np.isfinite(joints).all():
        raise ValueError("关节坐标包含 NaN 或 Inf。")

    normalised_edges = _validate_edges(edges, int(num_points))
    result = np.zeros_like(joints)
    for output_index, reference_index in normalised_edges:
        output_slice = [slice(None)] * joints.ndim
        reference_slice = [slice(None)] * joints.ndim
        output_slice[point_axis] = output_index
        reference_slice[point_axis] = reference_index
        result[tuple(output_slice)] = (
            joints[tuple(output_slice)] - joints[tuple(reference_slice)]
        )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="按 Lite-MyoNet 项目 23 节点拓扑生成骨向量数组。"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point-axis", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"输出已存在：{args.output}。如需覆盖，请显式传入 --overwrite。"
        )
    joints = np.load(args.input, allow_pickle=False)
    bone_vectors = compute_project_bone_vectors(
        joints,
        point_axis=args.point_axis,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if args.overwrite else "xb"
    with args.output.open(mode) as output_file:
        np.save(output_file, bone_vectors, allow_pickle=False)
    report = {
        "input": args.input.as_posix(),
        "output": args.output.as_posix(),
        "shape": list(bone_vectors.shape),
        "dtype": str(bone_vectors.dtype),
        "point_axis": args.point_axis,
        "edge_count": len(PROJECT_TOPOLOGY_EDGES),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


# Compatibility for generic NCTV conversion helpers.
from .nctv_bones import joints_to_bones
