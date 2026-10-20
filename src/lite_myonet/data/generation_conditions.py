import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


SPLIT_ORDER = ("train", "val", "test")


def _inclusive_grid(start, stop, step):
    if step <= 0:
        raise ValueError("参数网格步长必须为正整数。")
    if stop < start:
        raise ValueError("参数网格终点不能小于起点。")
    values = list(range(int(start), int(stop) + 1, int(step)))
    if not values or values[-1] != int(stop):
        raise ValueError("参数网格终点与步长不对齐。")
    return values


def _validate_label_map(labels, label_map):
    expected = [
        name
        for name, label_id in sorted(label_map.items(), key=lambda item: item[1])
        if name != "normal"
    ]
    if label_map.get("normal") != 0:
        raise ValueError("标签映射必须将 normal 固定为 0。")
    if labels != expected:
        raise ValueError("生成协议的 32 个病理标签与 label_map.json 不一致。")


def _assign_group_splits(group_values, counts, rng):
    if tuple(counts) != SPLIT_ORDER:
        raise ValueError("划分顺序必须为 train、val、test。")
    if sum(int(counts[name]) for name in SPLIT_ORDER) != len(group_values):
        raise ValueError("划分组数之和与可用条件组数不一致。")
    shuffled = list(group_values)
    rng.shuffle(shuffled)
    assignment = {}
    offset = 0
    for split_name in SPLIT_ORDER:
        split_count = int(counts[split_name])
        for value in shuffled[offset:offset + split_count]:
            assignment[value] = split_name
        offset += split_count
    return assignment


def build_generation_condition_table(protocol, label_map):
    """由机器可读协议构建确定性仿真条件表。

    该函数只固定条件和划分，不运行物理仿真，也不生成关节序列。
    """
    condition_model = protocol["target_condition_model"]
    pathology = condition_model["pathology"]
    normal = condition_model["normal"]
    labels = list(condition_model["pathology_labels"])
    _validate_label_map(labels, label_map)

    pathological_levels = _inclusive_grid(
        pathology["start_milli"],
        pathology["stop_milli_inclusive"],
        pathology["step_milli"],
    )
    excluded = {int(value) for value in pathology.get("excluded_milli", [])}
    pathological_levels = [
        value for value in pathological_levels if value not in excluded
    ]
    if len(pathological_levels) != int(pathology["parameter_level_count"]):
        raise ValueError("病理参数网格数量与协议声明不一致。")

    split_protocol = protocol["split_protocol"]
    rng = random.Random(int(split_protocol["selection_seed"]))
    pathological_assignment = _assign_group_splits(
        pathological_levels,
        split_protocol["pathological_group_counts"],
        rng,
    )

    conditions = []
    for parameter_milli in pathological_levels:
        split_name = pathological_assignment[parameter_milli]
        for label_name in labels:
            conditions.append(
                {
                    "condition_id": (
                        f"pathological:{parameter_milli:04d}:{label_name}"
                    ),
                    "split": split_name,
                    "label_name": label_name,
                    "label": int(label_map[label_name]),
                    "condition_type": "pathological",
                    "target_parameter_family": label_name,
                    "target_force_scale": parameter_milli / 1000.0,
                    "target_force_scale_milli": parameter_milli,
                    "non_target_force_scale": 1.0,
                    "changed_target_family_count": 1,
                    "evidence_status": "target_reconstruction",
                }
            )

    normal_levels = _inclusive_grid(
        normal["start_milli"],
        normal["stop_milli_inclusive"],
        normal["step_milli"],
    )
    if len(normal_levels) != int(normal["parameter_level_count_per_target"]):
        raise ValueError("正常参数网格数量与协议声明不一致。")
    normal_condition_keys = [
        (label_name, parameter_milli)
        for label_name in labels
        for parameter_milli in normal_levels
    ]
    normal_assignment = _assign_group_splits(
        normal_condition_keys,
        split_protocol["normal_counts"],
        rng,
    )
    for label_name, parameter_milli in normal_condition_keys:
        conditions.append(
            {
                "condition_id": f"normal:{label_name}:{parameter_milli:04d}",
                "split": normal_assignment[(label_name, parameter_milli)],
                "label_name": "normal",
                "label": int(label_map["normal"]),
                "condition_type": "normal",
                "target_parameter_family": label_name,
                "target_force_scale": parameter_milli / 1000.0,
                "target_force_scale_milli": parameter_milli,
                "non_target_force_scale": 1.0,
                "changed_target_family_count": 1 if parameter_milli < 1000 else 0,
                "evidence_status": "target_reconstruction",
            }
        )

    condition_ids = [record["condition_id"] for record in conditions]
    if len(condition_ids) != len(set(condition_ids)):
        raise RuntimeError("条件表中出现重复 condition_id。")
    expected_pathological = int(pathology["sequence_count"])
    expected_normal = int(normal["sequence_count"])
    type_counts = Counter(record["condition_type"] for record in conditions)
    if type_counts != Counter(
        {"pathological": expected_pathological, "normal": expected_normal}
    ):
        raise RuntimeError("生成条件数与协议声明不一致。")

    split_counts = Counter(record["split"] for record in conditions)
    expected_split_counts = {
        split_name: int(
            protocol["current_archive"]["splits"][split_name]["total"]
        )
        for split_name in SPLIT_ORDER
    }
    if dict(split_counts) != expected_split_counts:
        raise RuntimeError(
            f"条件表划分数量不一致：{dict(split_counts)} != "
            f"{expected_split_counts}"
        )

    return {
        "schema_version": 1,
        "description": (
            "Deterministic target reconstruction conditions; this table does not "
            "claim to recover the historical simulator state."
        ),
        "selection_seed": int(split_protocol["selection_seed"]),
        "summary": {
            "pathological": type_counts["pathological"],
            "normal": type_counts["normal"],
            "total": len(conditions),
            "splits": expected_split_counts,
        },
        "conditions": conditions,
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="构建 Lite-MyoNet 合成数据目标重建条件表。"
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--label-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    label_map = json.loads(args.label_map.read_text(encoding="utf-8"))
    table = build_generation_condition_table(protocol, label_map)
    table["source_protocol"] = {
        "path": args.protocol.as_posix(),
        "sha256": file_sha256(args.protocol),
    }
    table["source_label_map"] = {
        "path": args.label_map.as_posix(),
        "sha256": file_sha256(args.label_map),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(table, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(table["summary"], ensure_ascii=False, indent=2))
    print(f"已写入：{args.output.resolve()}")


if __name__ == "__main__":
    main()
