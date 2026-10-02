from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lite_myonet.models.skeleton import SkeletonDefinition, load_skeleton


def _small_config():
    """Return a valid three-node skeleton config with a nonzero root."""
    return {
        "schema_version": 1,
        "skeleton_id": "tiny-3",
        "point_count": 3,
        "root_index": 1,
        "nodes": [
            {"id": "left", "index": 0},
            {"id": "root", "index": 1},
            {"id": "right", "index": 2},
        ],
        "model_graph_edges": [[0, 1], [1, 2]],
        "bone_vector_edges": [[0, 1], [2, 0]],
        "symmetry_pairs": [[0, 2]],
        "channels": ["x", "y", "z"],
        "frame_count": 4,
    }


def _write_config(tmp_path, config, name="skeleton.yaml"):
    """Write a skeleton mapping to YAML and return the resulting path."""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_load_skeleton_returns_validated_definition_and_hash(tmp_path):
    """Load a validated skeleton definition and its canonical hash."""
    definition = load_skeleton(_write_config(tmp_path, _small_config()))

    assert isinstance(definition, SkeletonDefinition)
    assert definition.skeleton_id == "tiny-3"
    assert definition.point_count == 3
    assert definition.root_index == 1
    assert definition.model_graph_edges == ((0, 1), (1, 2))
    assert definition.bone_vector_edges == ((0, 1), (2, 0))
    assert definition.symmetry_pairs == ((0, 2),)
    assert definition.channels == ("x", "y", "z")
    assert definition.frame_count == 4
    assert definition.node_ids == ("left", "root", "right")
    assert len(definition.config_hash) == 64


def test_hash_is_stable_for_equivalent_edge_and_mapping_order(tmp_path):
    """Ensure equivalent mapping and edge orders yield the same hash."""
    first = _small_config()
    second = _small_config()
    second = {key: first[key] for key in reversed(first)}
    second["model_graph_edges"] = [[2, 1], [1, 0]]
    second["bone_vector_edges"] = [[2, 0], [0, 1]]
    second["symmetry_pairs"] = [[2, 0]]

    first_hash = load_skeleton(_write_config(tmp_path, first, "first.yaml")).config_hash
    second_hash = load_skeleton(_write_config(tmp_path, second, "second.yaml")).config_hash

    assert first_hash == second_hash


def test_fullbody_23_configuration_uses_explicit_project_topology():
    """Check that the full-body config defines the expected 23-node topology."""
    config_path = PROJECT_ROOT / "configs" / "skeletons" / "fullbody23.yaml"
    definition = load_skeleton(config_path)

    assert definition.point_count == 23
    assert definition.root_index == 0
    assert len(definition.model_graph_edges) == 22
    assert len(definition.bone_vector_edges) == 22
    assert len(definition.symmetry_pairs) == 9
    assert definition.frame_count == 120


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda config: config.update(point_count=4), "exactly point_count"),
        (lambda config: config["nodes"][2].update(index=1), "unique and contiguous"),
        (lambda config: config.update(model_graph_edges=[[0, 0], [1, 2]]), "self-edge"),
        (lambda config: config.update(model_graph_edges=[[0, 1]]), "connect every declared node"),
        (lambda config: config.update(bone_vector_edges=[[0, 2], [2, 0]]), "cycle"),
        (lambda config: config.update(symmetry_pairs=[[0, 2], [2, 0]]), "duplicate edge"),
        (lambda config: config.update(symmetry_pairs=[[0, 1]]), "root_index"),
        (lambda config: config.update(channels=["x", "x", "z"]), "channels must be unique"),
        (lambda config: config.update(frame_count=True), "positive integer"),
        (lambda config: config.update(schema_version=True), "schema_version must be 1"),
        (lambda config: config.update(typo_option=True), "Unknown skeleton configuration fields"),
    ],
)
def test_invalid_topology_or_fields_are_rejected(tmp_path, mutate, message):
    """Reject skeleton mutations that violate topology or schema requirements."""
    config = _small_config()
    mutate(config)

    with pytest.raises(ValueError, match=message):
        load_skeleton(_write_config(tmp_path, config))


def test_duplicate_yaml_keys_are_rejected(tmp_path):
    """Reject repeated keys in a skeleton YAML document."""
    path = tmp_path / "duplicate.yaml"
    path.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate YAML key"):
        load_skeleton(path)
