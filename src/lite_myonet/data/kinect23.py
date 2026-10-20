"""Kinect（微软体感相机）21/25 点到项目 23 点的历史兼容映射。

输入为 (..., 21, 3) 或 (..., 25, 3)，最后一维保持输入坐标轴顺序。
输出为 (..., 23, 3)，float32（32 位浮点）。本模块只做节点映射，
不做坐标反射、身高缩放、根节点平移或时间重采样。

数学依据为 Old/Same.py 与 Old/Trans.py，完整解释见：
docs/CLINICAL_MAPPING.md。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# Microsoft（微软）Kinect v2 的零基节点顺序。
KINECT_JOINT_NAMES = (
    "SpineBase", "SpineMid", "Neck", "Head",
    "ShoulderLeft", "ElbowLeft", "WristLeft", "HandLeft",
    "ShoulderRight", "ElbowRight", "WristRight", "HandRight",
    "HipLeft", "KneeLeft", "AnkleLeft", "FootLeft",
    "HipRight", "KneeRight", "AnkleRight", "FootRight",
    "SpineShoulder", "HandTipLeft", "ThumbLeft", "HandTipRight", "ThumbRight",
)

# Old/Same.py 的 source -> target（中间节点 -> 最终槽位）字典。
SOURCE_TO_TARGET = {
    1: 14, 14: 13, 19: 15, 17: 16, 2: 17,
    3: 18, 20: 19, 18: 20, 4: 21, 5: 22,
    15: 12, 0: 11, 16: 0, 10: 6, 6: 1, 7: 2,
    8: 3, 21: 4, 9: 5, 11: 7, 12: 8, 22: 9, 13: 10,
}
TARGET_FROM_INTERMEDIATE = tuple(
    source for source, target in sorted(SOURCE_TO_TARGET.items(), key=lambda item: item[1])
)


def process_points(points: np.ndarray) -> np.ndarray:
    """构造旧算法的 23 个中间节点，供 reorder_array（重排函数）使用。

    用原始 Kinect 索引直接取点，不使用会移动索引的连续删除操作。
    中点使用 float64（64 位浮点）计算后转 float32，以对应旧 Trans.py
    的列表运算精度和保存顺序。支持整段序列一次转换。
    """
    k = np.asarray(points, dtype=np.float64)
    if k.ndim < 2 or k.shape[-2:] != (21, 3):
        raise ValueError("process_points 的输入应为 (...,21,3)。")

    torso_midpoint = (k[..., 20, :] + k[..., 1, :]) / 2
    pelvis_midpoint = (k[..., 1, :] + k[..., 0, :]) / 2
    upper_arm_left = (k[..., 4, :] + k[..., 5, :]) / 2
    upper_arm_right = (k[..., 8, :] + k[..., 9, :]) / 2
    shoulder_left = (k[..., 20, :] + upper_arm_left) / 2
    shoulder_right = (k[..., 20, :] + upper_arm_right) / 2
    foot_midpoint_left = (k[..., 14, :] + k[..., 15, :]) / 2
    foot_midpoint_right = (k[..., 18, :] + k[..., 19, :]) / 2

    intermediate = [
        k[..., 1, :], k[..., 2, :], k[..., 5, :], k[..., 6, :],
        k[..., 9, :], k[..., 10, :], k[..., 12, :], k[..., 13, :],
        k[..., 14, :], k[..., 15, :], k[..., 16, :], k[..., 17, :],
        k[..., 18, :], k[..., 19, :], k[..., 20, :],
        torso_midpoint, pelvis_midpoint, upper_arm_left, upper_arm_right,
        shoulder_left, shoulder_right, foot_midpoint_left, foot_midpoint_right,
    ]
    return np.stack(intermediate, axis=-2).astype(np.float32)


def reorder_array(intermediate: np.ndarray) -> np.ndarray:
    """将 23 个中间节点放入与旧版相同的项目输入槽位。"""
    return np.take(intermediate, TARGET_FROM_INTERMEDIATE, axis=-2)


def map_kinect_to_synthetic23(points: np.ndarray) -> np.ndarray:
    """21/25 点 -> 23 点；25 点输入忽略末尾四个手指节点。

    25 点输入支持是本次整理新增的接口。旧 CSV（逗号分隔文本）解析器
    和已核对的原始数组本身只有 21 点，不据此声称找回了采集端导出代码。
    保留旧映射的左右槽位约定，标签侧别修正在后续评估中另行完成。
    """
    points = np.asarray(points)
    if points.ndim < 2 or points.shape[-1] != 3 or points.shape[-2] not in (21, 25):
        raise ValueError("输入应为 (...,21,3) 或 (...,25,3)，采用 Kinect v2 零基顺序。")
    return reorder_array(process_points(points[..., :21, :]))


def main():
    parser = argparse.ArgumentParser(description="按旧映射将 Kinect 21/25 点 NPY（数组文件）转换为 23 点。")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mapped = map_kinect_to_synthetic23(np.load(args.input, allow_pickle=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.save(stream, mapped, allow_pickle=False)
    print(f"转换完成：{mapped.shape}，{mapped.dtype}，保存至 {args.output}")


if __name__ == "__main__":
    main()
