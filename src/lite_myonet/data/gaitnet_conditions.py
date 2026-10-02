"""Build explicit target schedules without pretending to execute a simulator."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from lite_myonet.config import load_label_map, load_yaml, resolve_config_path, write_json
from lite_myonet.data.generative_gaitnet import UPSTREAM_COMMIT, UPSTREAM_REPOSITORY


SPLITS = ("train", "val", "test")


def integer_grid(spec: dict) -> list[int]:
    """Build an inclusive thousandth-scale grid without floating-point filenames."""
    keys = ("start_milli", "stop_milli", "step_milli")
    if any(isinstance(spec.get(key), bool) or not isinstance(spec.get(key), int) for key in keys):
        raise ValueError("Grid bounds and step must be integers in thousandths.")
    start, stop, step = (spec[key] for key in keys)
    if not 0 <= start <= stop <= 1000 or step <= 0 or (stop - start) % step:
        raise ValueError("The inclusive force-scale grid is outside [0,1000] or misaligned.")
    excluded = spec.get("excluded_milli", [])
    if any(isinstance(value, bool) or not isinstance(value, int) for value in excluded):
        raise ValueError("Excluded force levels must be integer thousandths.")
    levels = list(range(start, stop + 1, step))
    if len(set(excluded)) != len(excluded) or not set(excluded).issubset(levels):
        raise ValueError("Excluded force levels must be unique members of the original grid.")
    levels = [level for level in levels if level not in set(excluded)]
    if not levels:
        raise ValueError("The force-scale grid is empty.")
    return levels


def split_assignment(keys: list, counts: dict, generator: random.Random) -> dict:
    """Assign shuffled whole conditions or whole force-level groups to fixed splits."""
    if set(counts) != set(SPLITS):
        raise ValueError("Split counts must define train, val, and test.")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in counts.values()):
        raise ValueError("Every split count must be a positive integer.")
    if sum(counts.values()) != len(keys):
        raise ValueError("Split counts do not match the number of available keys.")
    ordered = list(keys)
    generator.shuffle(ordered)
    result = {}
    offset = 0
    for split in SPLITS:
        for key in ordered[offset:offset + counts[split]]:
            result[key] = split
        offset += counts[split]
    return result


def build_condition_table(config: dict, labels: dict[str, int]) -> dict:
    """Create single-target force-scaling conditions and leakage-safe group splits."""
    if config.get("schema_version") != 1 or labels.get("normal") != 0:
        raise ValueError("Use schema_version=1 and label normal=0.")
    if sorted(labels.values()) != list(range(len(labels))):
        raise ValueError("Label identifiers must be contiguous from zero.")
    targets = [name for name, index in sorted(labels.items(), key=lambda item: item[1]) if index != 0]
    if len(targets) != 32:
        raise ValueError("The supplied synthetic protocol requires 32 pathological target families.")
    pathological_levels = integer_grid(config["pathological"])
    normal_levels = integer_grid(config["normal"])
    if max(pathological_levels) > 950 or min(normal_levels) <= 950:
        raise ValueError("Normal levels must exceed 0.950 and pathological levels must not.")
    generator = random.Random(int(config["selection_seed"]))
    pathology_splits = split_assignment(
        pathological_levels, config["pathological"]["split_group_counts"], generator,
    )
    normal_keys = [(target, level) for target in targets for level in normal_levels]
    normal_splits = split_assignment(normal_keys, config["normal"]["split_counts"], generator)
    conditions = []
    for normal, keys in ((False, [(target, level) for level in pathological_levels for target in targets]), (True, normal_keys)):
        for target, level in keys:
            condition_type = "normal" if normal else "pathological"
            conditions.append({
                "condition_id": f"{condition_type}:{target}:{level:04d}",
                "condition_type": condition_type,
                "label_name": "normal" if normal else target,
                "label": 0 if normal else labels[target],
                "target_parameter_family": target,
                "target_force_scale_milli": level,
                "target_force_scale": level / 1000.0,
                "other_force_scales": 1.0,
                "split": normal_splits[(target, level)] if normal else pathology_splits[level],
                "simulation_seed": int(config["simulation_seed_base"]) + len(conditions),
            })
    return {
        "schema_version": 1,
        "protocol_status": "target_generation_schedule_not_historical_simulator_state",
        "upstream": {"repository": UPSTREAM_REPOSITORY, "commit": UPSTREAM_COMMIT},
        "summary": {
            "total": len(conditions),
            "types": dict(Counter(item["condition_type"] for item in conditions)),
            "splits": dict(Counter(item["split"] for item in conditions)),
            "pathological_parameter_levels": len(pathological_levels),
            "normal_parameter_levels": len(normal_levels),
        },
        "conditions": conditions,
    }


def main(argv: list[str] | None = None) -> int:
    """Write a deterministic target condition schedule for an external GaitNet rollout."""
    parser = argparse.ArgumentParser(description="Build a Generative GaitNet target-condition schedule.")
    parser.add_argument("--config", type=Path, default=Path("configs/conversion/generative_gaitnet_conditions.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    config = load_yaml(arguments.config)
    labels = load_label_map(resolve_config_path(config, config["label_map"]))
    result = build_condition_table(config, labels)
    write_json(arguments.output, result)
    print(result["summary"])
    return 0
