import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SkeletonDataset(Dataset):
    """读取配对的关节坐标与骨向量，将二者转为 ``[C,T,V]`` 并返回标签。"""

    def __init__(
        self,
        data_dir,
        muscle_to_label,
        transform=None,
        manifest_path=None,
        split=None,
        expected_frames=120,
        num_points=23,
    ):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.muscle_to_label = dict(muscle_to_label)
        self.transform = transform
        self.expected_frames = int(expected_frames)
        self.num_points = int(num_points)

        if manifest_path is None:
            self.data_pairs = self._scan_directory(self.data_dir)
        else:
            if split is None:
                raise ValueError("使用 manifest_path 时必须显式提供 split。")
            self.data_pairs = self._read_manifest(Path(manifest_path), split)

        if not self.data_pairs:
            raise RuntimeError(f"没有找到可用的数据对：{self.data_dir}")

    def _scan_directory(self, directory):
        if not directory.is_dir():
            raise FileNotFoundError(f"数据目录不存在：{directory}")

        file_names = {path.name for path in directory.glob("*.npy")}
        pairs = []
        for joint_path in sorted(directory.glob("*.npy"), key=lambda path: path.name):
            if joint_path.name.startswith("skeleton23_"):
                continue

            muscle_name = joint_path.stem.rsplit("_", 1)[0]
            if muscle_name not in self.muscle_to_label:
                raise ValueError(f"无法识别文件中的类别：{joint_path.name}")

            bone_name = f"skeleton23_{joint_path.name}"
            if bone_name not in file_names:
                raise FileNotFoundError(f"缺少配对骨向量文件：{directory / bone_name}")

            pairs.append(
                (joint_path, directory / bone_name, self.muscle_to_label[muscle_name])
            )
        return pairs

    def _read_manifest(self, manifest_path, split):
        manifest_path = manifest_path.expanduser().resolve()
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)

        records = manifest.get("splits", {}).get(split)
        if records is None:
            raise KeyError(f"manifest 中不存在 split={split!r}。")

        root = Path(manifest.get("dataset_root", self.data_dir))
        if not root.is_absolute():
            root = (manifest_path.parent / root).resolve()

        pairs = []
        for record in records:
            joint_path = (root / record["joints"]).resolve()
            bone_path = (root / record["bones"]).resolve()
            label_name = record["label_name"]
            expected_label = self.muscle_to_label[label_name]
            if int(record["label"]) != expected_label:
                raise ValueError(
                    f"manifest 标签不一致：{joint_path.name}，"
                    f"记录为 {record['label']}，映射应为 {expected_label}。"
                )
            if not joint_path.is_file() or not bone_path.is_file():
                raise FileNotFoundError(f"manifest 指向缺失文件：{joint_path} / {bone_path}")
            pairs.append((joint_path, bone_path, expected_label))
        return pairs

    def __len__(self):
        return len(self.data_pairs)

    def _to_ctv(self, array, source_path):
        if array.shape == (3, self.expected_frames, self.num_points):
            converted = array
        elif array.shape == (self.expected_frames, self.num_points, 3):
            converted = np.transpose(array, (2, 0, 1))
        else:
            raise ValueError(
                f"数据形状错误：{source_path} 的形状为 {array.shape}，"
                f"期望 (3,{self.expected_frames},{self.num_points}) 或 "
                f"({self.expected_frames},{self.num_points},3)。"
            )
        if not np.isfinite(converted).all():
            raise ValueError(f"数据包含 NaN 或 Inf：{source_path}")
        return torch.from_numpy(np.ascontiguousarray(converted)).float()

    def __getitem__(self, index):
        joint_path, bone_path, label = self.data_pairs[index]
        joints = self._to_ctv(np.load(joint_path, allow_pickle=False), joint_path)
        bones = self._to_ctv(np.load(bone_path, allow_pickle=False), bone_path)

        if self.transform is not None:
            joints, bones = self.transform(joints, bones)

        return joints, bones, int(label)
