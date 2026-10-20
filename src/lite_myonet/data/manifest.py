import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .simulation_parameters import canonicalize_simulation_parameter


REQUIRED_SPLITS = ("train", "val", "test")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(manifest_path, dataset_root, verify_hashes=False):
    """校验清单结构、分组隔离、文件存在性及可选的文件内容哈希。"""
    manifest_path = Path(manifest_path).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("manifest 缺少 splits 字典。")
    missing_splits = [name for name in REQUIRED_SPLITS if name not in splits]
    if missing_splits:
        raise ValueError(f"manifest 缺少数据划分：{missing_splits}。")

    group_to_split = {}
    joint_path_to_split = {}
    bone_path_to_split = {}
    joint_hash_to_split = {}
    bone_hash_to_split = {}
    canonical_raw_tokens = {}
    impaired_group_labels = {}
    split_counts = {}
    class_counts = {}
    schema_version = int(manifest.get("schema_version", 1))

    for split_name in REQUIRED_SPLITS:
        records = splits[split_name]
        if not isinstance(records, list) or not records:
            raise ValueError(f"manifest 的 {split_name} 划分为空或不是列表。")
        split_counts[split_name] = len(records)
        class_counts[split_name] = dict(Counter(record["label_name"] for record in records))

        for record in records:
            group_id = record.get("split_group_id")
            if not group_id:
                raise ValueError(f"记录缺少 split_group_id：{record.get('joints')}。")
            previous_split = group_to_split.get(group_id)
            if previous_split is not None and previous_split != split_name:
                raise ValueError(
                    f"参数组跨划分泄漏：{group_id} 同时出现在 {previous_split} 和 {split_name}。"
                )
            group_to_split[group_id] = split_name

            if schema_version >= 2:
                label_name = record.get("label_name")
                if label_name == "normal":
                    for field_name in (
                        "simulation_parameter_raw",
                        "simulation_parameter_milli",
                        "simulation_parameter",
                        "simulation_parameter_residual",
                    ):
                        if record.get(field_name) is not None:
                            raise ValueError(
                                f"健康样本的 {field_name} 必须为 null：{record.get('joints')}。"
                            )
                    if not group_id.startswith("normal:sample:"):
                        raise ValueError(f"健康样本组 ID 格式错误：{group_id}。")
                else:
                    raw_parameter = record.get("simulation_parameter_raw")
                    if raw_parameter is None:
                        raise ValueError(
                            f"病理样本缺少 simulation_parameter_raw：{record.get('joints')}。"
                        )
                    parameter = canonicalize_simulation_parameter(raw_parameter)
                    if int(record.get("simulation_parameter_milli", -1)) != parameter.milli:
                        raise ValueError(
                            f"病理样本规范参数编号不一致：{record.get('joints')}。"
                        )
                    if record.get("simulation_parameter") != parameter.canonical:
                        raise ValueError(
                            f"病理样本规范参数字符串不一致：{record.get('joints')}。"
                        )
                    if record.get("simulation_parameter_residual") != parameter.residual:
                        raise ValueError(
                            f"病理样本参数残差不一致：{record.get('joints')}。"
                        )
                    expected_group_id = f"impaired:param_milli:{parameter.milli:03d}"
                    if group_id != expected_group_id:
                        raise ValueError(
                            f"病理样本组 ID {group_id} 与规范参数 {expected_group_id} 不一致。"
                        )
                    previous_raw = canonical_raw_tokens.setdefault(
                        parameter.milli, parameter.raw
                    )
                    if previous_raw != parameter.raw:
                        raise ValueError(
                            f"原始参数 {previous_raw} 和 {parameter.raw} 映射到同一规范编号 "
                            f"{parameter.milli}，拒绝静默合并。"
                        )
                    impaired_group_labels.setdefault(group_id, set()).add(label_name)

            joint_relative = record["joints"]
            bone_relative = record["bones"]
            for relative_path, path_registry, kind in (
                (joint_relative, joint_path_to_split, "关节"),
                (bone_relative, bone_path_to_split, "骨向量"),
            ):
                previous_path_split = path_registry.get(relative_path)
                if previous_path_split is not None:
                    raise ValueError(
                        f"{kind}文件被重复引用：{relative_path}，出现在 "
                        f"{previous_path_split} 和 {split_name}。"
                    )
                path_registry[relative_path] = split_name

            joint_path = (dataset_root / joint_relative).resolve()
            bone_path = (dataset_root / bone_relative).resolve()
            if not joint_path.is_file() or not bone_path.is_file():
                raise FileNotFoundError(f"manifest 指向缺失文件：{joint_path} / {bone_path}")

            for hash_key, hash_registry, path in (
                ("joints_sha256", joint_hash_to_split, joint_path),
                ("bones_sha256", bone_hash_to_split, bone_path),
            ):
                stored_hash = record.get(hash_key)
                if not stored_hash:
                    raise ValueError(f"记录缺少 {hash_key}：{joint_relative}。")
                previous_hash_split = hash_registry.get(stored_hash)
                if previous_hash_split is not None:
                    raise ValueError(
                        f"相同内容被重复使用：{path.name}，位于 "
                        f"{previous_hash_split} 和 {split_name}。"
                    )
                hash_registry[stored_hash] = split_name
                if verify_hashes:
                    actual_hash = file_sha256(path)
                    if actual_hash != stored_hash:
                        raise ValueError(
                            f"文件内容哈希不匹配：{path}，记录为 {stored_hash}，"
                            f"实际为 {actual_hash}。"
                        )

    total = sum(split_counts.values())
    if int(manifest.get("total_sequences", -1)) != total:
        raise ValueError(
            f"manifest 总数不一致：记录为 {manifest.get('total_sequences')}，实际为 {total}。"
        )
    stored_class_counts = manifest.get("class_counts")
    if stored_class_counts is not None and stored_class_counts != class_counts:
        raise ValueError("manifest 中记录的 class_counts 与 splits 实际内容不一致。")

    if schema_version >= 2 and impaired_group_labels:
        expected_impaired_labels = {
            label_name
            for counts in class_counts.values()
            for label_name in counts
            if label_name != "normal"
        }
        actual_incomplete_groups = {
            group_id: sorted(expected_impaired_labels - labels)
            for group_id, labels in impaired_group_labels.items()
            if labels != expected_impaired_labels
        }
        if actual_incomplete_groups:
            raise ValueError(
                "正式 manifest 不得包含不完整病理参数组："
                f"{actual_incomplete_groups}。"
            )

        complete_group_labels = {
            group_id: labels
            for group_id, labels in impaired_group_labels.items()
            if labels == expected_impaired_labels
        }
        core = manifest.get("balanced_pathological_core")
        if not isinstance(core, dict):
            raise ValueError("manifest 缺少 balanced_pathological_core 声明。")
        actual_core_sequences = sum(len(labels) for labels in complete_group_labels.values())
        if int(core.get("parameter_group_count", -1)) != len(complete_group_labels):
            raise ValueError("manifest 的平衡病理核心参数组数与实际不一致。")
        if int(core.get("sequence_count", -1)) != actual_core_sequences:
            raise ValueError("manifest 的平衡病理核心序列数与实际不一致。")
        if int(core.get("class_count", -1)) != len(expected_impaired_labels):
            raise ValueError("manifest 的平衡病理核心类别数与实际不一致。")

    return {
        "manifest_path": str(manifest_path),
        "dataset_root": str(dataset_root),
        "manifest_sha256": file_sha256(manifest_path),
        "verify_hashes": bool(verify_hashes),
        "total_sequences": total,
        "split_counts": split_counts,
        "unique_group_count": len(group_to_split),
        "class_counts": class_counts,
        "schema_version": schema_version,
        "canonical_simulation_parameter_count": len(canonical_raw_tokens),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()

    report = verify_manifest(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        verify_hashes=args.verify_hashes,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
