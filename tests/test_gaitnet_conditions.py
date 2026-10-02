"""Check force-level integerization and whole-group split invariants."""

from pathlib import Path

import pytest

from lite_myonet.config import load_label_map, load_yaml, resolve_config_path
from lite_myonet.data.gaitnet_conditions import build_condition_table, integer_grid


def test_target_schedule_has_complete_groups_and_fixed_counts():
    """Verify all retained pathological levels have every class in a single split."""
    root = Path(__file__).resolve().parents[1]
    config = load_yaml(root / "configs/conversion/generative_gaitnet_conditions.yaml")
    labels = load_label_map(resolve_config_path(config, config["label_map"]))
    table = build_condition_table(config, labels)
    assert table["summary"] == {
        "total": 12800,
        "types": {"pathological": 11200, "normal": 1600},
        "splits": {"train": 6400, "val": 3184, "test": 3216},
        "pathological_parameter_levels": 350,
        "normal_parameter_levels": 50,
    }
    groups = {}
    for condition in table["conditions"]:
        if condition["condition_type"] == "pathological":
            groups.setdefault(condition["target_force_scale_milli"], []).append(condition)
        else:
            assert condition["target_force_scale_milli"] > 950
            assert condition["label"] == 0
    assert 870 not in groups
    assert all(len(values) == 32 and len({item["split"] for item in values}) == 1 for values in groups.values())
    assert table == build_condition_table(config, labels)
    assert len({item["simulation_seed"] for item in table["conditions"]}) == 12800


@pytest.mark.parametrize("spec", [
    {"start_milli": 600, "stop_milli": 950, "step_milli": 3},
    {"start_milli": 600, "stop_milli": 950, "step_milli": 1, "excluded_milli": [1001]},
    {"start_milli": True, "stop_milli": 950, "step_milli": 1},
])
def test_invalid_integer_grids_are_rejected(spec):
    """Reject misaligned grids and invalid exclusion values instead of rounding silently."""
    with pytest.raises(ValueError):
        integer_grid(spec)
