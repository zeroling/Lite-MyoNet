import argparse
import json
import random
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from .manifest import file_sha256
from .simulation_parameters import canonicalize_simulation_parameter


PARAMETER_GROUP_COUNTS_BY_SPLIT = {"train": 175, "val": 87, "test": 88}
NORMAL_COUNTS_BY_SPLIT = {"train": 800, "val": 400, "test": 400}
NORMAL_LABEL = "normal"
EXPECTED_PARAMETER_MILLI = tuple(range(600, 951))
EXPECTED_TOTAL_SEQUENCES = 12800


def discover_pairs(dataset_root, known_labels):
    dataset_root = Path(dataset_root).expanduser().resolve()
    records_by_label = defaultdict(list)
    canonical_raw_tokens = defaultdict(set)

    for source_split in ("train", "val", "test"):
        split_dir = dataset_root / source_split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"缺少源数据目录：{split_dir}")

        for joint_path in sorted(split_dir.glob("*.npy"), key=lambda path: path.name):
            if joint_path.name.startswith("skeleton23_"):
                continue
            label_name, source_sample_id = joint_path.stem.rsplit("_", 1)
            if label_name not in known_labels:
                raise ValueError(f"未知类别文件：{joint_path}")
            bone_path = joint_path.with_name(f"skeleton23_{joint_path.name}")
            if not bone_path.is_file():
                raise FileNotFoundError(f"缺少骨向量配对文件：{bone_path}")

            record = {
                "joints": joint_path.relative_to(dataset_root).as_posix(),
                "bones": bone_path.relative_to(dataset_root).as_posix(),
                "source_split": source_split,
                "source_sample_id": source_sample_id,
                "label_name": label_name,
                "label": int(known_labels[label_name]),
            }
            if label_name == NORMAL_LABEL:
                record.update(
                    {
                        "simulation_parameter_raw": None,
                        "simulation_parameter_milli": None,
                        "simulation_parameter": None,
                        "simulation_parameter_residual": None,
                    }
                )
            else:
                parameter = canonicalize_simulation_parameter(source_sample_id)
                canonical_raw_tokens[parameter.milli].add(parameter.raw)
                record.update(
                    {
                        "simulation_parameter_raw": parameter.raw,
                        "simulation_parameter_milli": parameter.milli,
                        "simulation_parameter": parameter.canonical,
                        "simulation_parameter_residual": parameter.residual,
                    }
                )
            records_by_label[label_name].append(record)

    collisions = {
        milli: sorted(raw_tokens)
        for milli, raw_tokens in canonical_raw_tokens.items()
        if len(raw_tokens) != 1
    }
    if collisions:
        raise RuntimeError(
            "多个原始仿真参数落入同一千分位网格，拒绝静默合并："
            f"{collisions}"
        )
    observed_grid = tuple(sorted(canonical_raw_tokens))
    if observed_grid != EXPECTED_PARAMETER_MILLI:
        missing = sorted(set(EXPECTED_PARAMETER_MILLI) - set(observed_grid))
        extra = sorted(set(observed_grid) - set(EXPECTED_PARAMETER_MILLI))
        raise RuntimeError(
            "仿真参数网格不是预期的 0.600–0.950、步长 0.001："
            f"missing={missing}, extra={extra}。"
        )
    return records_by_label


def build_manifest(dataset_root, labels, seed):
    dataset_root = Path(dataset_root).expanduser().resolve()
    records_by_label = discover_pairs(dataset_root, labels)
    missing_labels = sorted(set(labels) - set(records_by_label))
    if missing_labels:
        raise RuntimeError(f"缺少类别：{missing_labels}")
    if NORMAL_LABEL not in labels:
        raise RuntimeError(f"标签映射中缺少健康类别 {NORMAL_LABEL!r}。")

    required_groups = sum(PARAMETER_GROUP_COUNTS_BY_SPLIT.values())
    rng = random.Random(seed)
    splits = {name: [] for name in PARAMETER_GROUP_COUNTS_BY_SPLIT}
    source_counts = {
        label_name: len(records_by_label[label_name])
        for label_name in sorted(labels, key=labels.get)
    }
    impaired_labels = [
        label_name
        for label_name in sorted(labels, key=labels.get)
        if label_name != NORMAL_LABEL
    ]

    records_by_parameter = defaultdict(dict)
    raw_by_parameter = {}
    residual_by_parameter = {}
    for label_name in impaired_labels:
        for record in records_by_label[label_name]:
            parameter_milli = int(record["simulation_parameter_milli"])
            if label_name in records_by_parameter[parameter_milli]:
                raise RuntimeError(
                    f"类别 {label_name} 的规范仿真参数 {parameter_milli} 出现重复记录。"
                )
            records_by_parameter[parameter_milli][label_name] = record
            raw_by_parameter.setdefault(
                parameter_milli, record["simulation_parameter_raw"]
            )
            residual_by_parameter.setdefault(
                parameter_milli, record["simulation_parameter_residual"]
            )

    expected_impaired_labels = set(impaired_labels)
    complete_parameter_milli = sorted(
        parameter_milli
        for parameter_milli, records in records_by_parameter.items()
        if set(records) == expected_impaired_labels
    )
    incomplete_details = []
    for parameter_milli, records in sorted(records_by_parameter.items()):
        if set(records) != expected_impaired_labels:
            incomplete_details.append(
                {
                    "simulation_parameter_milli": parameter_milli,
                    "simulation_parameter": f"{Decimal(parameter_milli) / 1000:.3f}",
                    "simulation_parameter_raw": raw_by_parameter[parameter_milli],
                    "available_label_count": len(records),
                    "missing_labels": sorted(expected_impaired_labels - set(records)),
                }
            )
    if len(complete_parameter_milli) != required_groups:
        raise RuntimeError(
            f"发现 {len(complete_parameter_milli)} 个完整仿真参数组，"
            f"正式 11,200 条病理协议要求 {required_groups} 个。"
        )

    selected_parameter_milli = list(complete_parameter_milli)
    rng.shuffle(selected_parameter_milli)
    parameter_splits = {}
    offset = 0
    for split_name, group_count in PARAMETER_GROUP_COUNTS_BY_SPLIT.items():
        split_parameter_milli = selected_parameter_milli[offset:offset + group_count]
        parameter_splits[split_name] = sorted(split_parameter_milli)
        offset += group_count
        for parameter_milli in split_parameter_milli:
            for label_name in impaired_labels:
                record = records_by_parameter[parameter_milli].get(label_name)
                if record is None:
                    continue
                record["split_group_id"] = f"impaired:param_milli:{parameter_milli:03d}"
                splits[split_name].append(record)

    normal_candidates = sorted(
        records_by_label[NORMAL_LABEL],
        key=lambda record: (record["source_split"], record["joints"]),
    )
    if len(normal_candidates) != source_counts[NORMAL_LABEL]:
        raise RuntimeError("健康样本发现结果与 source_counts 不一致。")
    pathological_counts_by_split = {
        split_name: len(records) for split_name, records in splits.items()
    }
    normal_counts_by_split = dict(NORMAL_COUNTS_BY_SPLIT)
    if sum(normal_counts_by_split.values()) != len(normal_candidates):
        raise RuntimeError(
            f"目标划分需要 {sum(normal_counts_by_split.values())} 条健康样本，"
            f"但源数据有 {len(normal_candidates)} 条。"
        )
    rng.shuffle(normal_candidates)
    offset = 0
    for split_name, split_count in normal_counts_by_split.items():
        for record in normal_candidates[offset:offset + split_count]:
            record["split_group_id"] = f"normal:sample:{record['source_sample_id']}"
            splits[split_name].append(record)
        offset += split_count

    seen_joint_hashes = {}
    seen_bone_hashes = {}
    group_to_split = {}
    for split_name, records in splits.items():
        for record in records:
            joint_path = dataset_root / record["joints"]
            bone_path = dataset_root / record["bones"]
            record["joints_sha256"] = file_sha256(joint_path)
            record["bones_sha256"] = file_sha256(bone_path)
            for hash_key, registry, kind in (
                ("joints_sha256", seen_joint_hashes, "关节"),
                ("bones_sha256", seen_bone_hashes, "骨向量"),
            ):
                previous_split = registry.get(record[hash_key])
                if previous_split is not None:
                    raise RuntimeError(
                        f"检测到完全重复的{kind}序列：{record['joints']}，"
                        f"同时出现在 {previous_split} 与 {split_name}。"
                    )
                registry[record[hash_key]] = split_name

            previous_group_split = group_to_split.get(record["split_group_id"])
            if previous_group_split is not None and previous_group_split != split_name:
                raise RuntimeError(
                    f"仿真参数组跨 split 泄漏：{record['split_group_id']} 同时出现在 "
                    f"{previous_group_split} 与 {split_name}。"
                )
            group_to_split[record["split_group_id"]] = split_name
        records.sort(key=lambda record: (record["label"], record["joints"]))

    class_counts = {
        split_name: dict(sorted(Counter(r["label_name"] for r in records).items()))
        for split_name, records in splits.items()
    }
    complete_parameter_set = set(complete_parameter_milli)
    balanced_core_group_counts_by_split = {
        split_name: sum(
            parameter_milli in complete_parameter_set
            for parameter_milli in split_parameter_milli
        )
        for split_name, split_parameter_milli in parameter_splits.items()
    }
    balanced_core_counts_by_split = {
        split_name: group_count * len(impaired_labels)
        for split_name, group_count in balanced_core_group_counts_by_split.items()
    }
    total_sequences = sum(len(records) for records in splits.values())
    if total_sequences != EXPECTED_TOTAL_SEQUENCES:
        raise RuntimeError(
            f"正式协议总数应为 {EXPECTED_TOTAL_SEQUENCES}，"
            f"实际为 {total_sequences}。"
        )
    target_split_totals = {
        split_name: len(records) for split_name, records in splits.items()
    }
    max_residual = max(
        abs(Decimal(residual))
        for residual in residual_by_parameter.values()
    )
    return {
        "schema_version": 2,
        "description": (
            "Lite-MyoNet 12,800-sequence protocol: 11,200 pathological sequences "
            "from 350 complete parameter groups x 32 muscle labels, plus all "
            "1,600 normal sequences. The incomplete 0.870 source group is "
            "excluded in full. Pathological samples are re-split by a "
            "collision-checked integer index on the filename-encoded 0.001 "
            "simulation-parameter grid; original filename tokens are retained."
        ),
        "selection_seed": seed,
        "split_unit": "canonical_filename_parameter_milli",
        "source_sample_policy": (
            "all_complete_pathological_parameter_groups_plus_all_normal_records"
        ),
        "preserves_original_source_split": False,
        "simulation_parameter_grid": {
            "canonical_min": "0.600",
            "canonical_max": "0.950",
            "step": "0.001",
            "index_min_milli": 600,
            "index_max_milli": 950,
            "observed_unique_raw_token_count": len(raw_by_parameter),
            "observed_unique_canonical_index_count": len(records_by_parameter),
            "raw_to_canonical_collision_count": 0,
            "maximum_absolute_residual": format(max_residual, "f"),
            "semantic_scope": "filename-encoded simulation parameter identifier",
        },
        "parameter_group_counts_by_split": PARAMETER_GROUP_COUNTS_BY_SPLIT,
        "pathological_counts_by_split": pathological_counts_by_split,
        "normal_counts_by_split": normal_counts_by_split,
        "target_split_totals": target_split_totals,
        "total_sequences": total_sequences,
        "source_counts": source_counts,
        "balanced_pathological_core": {
            "sequence_count": sum(balanced_core_counts_by_split.values()),
            "parameter_group_count": len(complete_parameter_milli),
            "class_count": len(impaired_labels),
            "group_counts_by_split": balanced_core_group_counts_by_split,
            "sequence_counts_by_split": balanced_core_counts_by_split,
        },
        "excluded_incomplete_source_parameter_groups": incomplete_details,
        "excluded_source_sequence_count": sum(
            item["available_label_count"] for item in incomplete_details
        ),
        "impaired_parameter_milli_by_split": parameter_splits,
        "class_counts": class_counts,
        "splits": splits,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--label-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()

    with args.label_map.open("r", encoding="utf-8") as label_file:
        labels = json.load(label_file)
    manifest = build_manifest(args.dataset_root.resolve(), labels, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, ensure_ascii=False, indent=2)
    print(f"已写入：{args.output.resolve()}")
    print(
        "split 数量："
        + ", ".join(f"{name}={len(records)}" for name, records in manifest["splits"].items())
    )


if __name__ == "__main__":
    main()
