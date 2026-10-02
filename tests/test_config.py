"""Tests for shared configuration and provenance utilities."""

from __future__ import annotations

import json

import pytest

from lite_myonet.config import canonical_sha256, load_label_map, load_yaml


def test_canonical_hash_is_independent_of_key_order() -> None:
    """Verify that provenance hashes use canonical mapping order."""
    assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256({"b": 2, "a": 1})


def test_yaml_root_must_be_mapping(tmp_path) -> None:
    """Reject YAML files whose root value is not a mapping."""
    path = tmp_path / "invalid.yaml"
    path.write_text("- item\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a mapping"):
        load_yaml(path)


def test_label_map_must_be_contiguous(tmp_path) -> None:
    """Reject label mappings with missing class indices."""
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"a": 0, "b": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="contiguous"):
        load_label_map(path)
