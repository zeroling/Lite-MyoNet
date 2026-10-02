"""Check that failed configuration writes preserve the prior complete file."""

import json

import pytest

from lite_myonet import config


def test_failed_json_write_keeps_previous_document(tmp_path, monkeypatch):
    """Inject a partial serialization failure without replacing existing state."""
    destination = tmp_path / "state.json"
    config.write_json(destination, {"status": "completed"})

    def fail_after_partial_write(payload, stream, **kwargs):
        """Simulate a power-loss-adjacent failure during temporary serialization."""
        stream.write("{partial")
        raise OSError("injected serialization failure")

    monkeypatch.setattr(config.json, "dump", fail_after_partial_write)
    with pytest.raises(OSError, match="injected"):
        config.write_json(destination, {"status": "running"})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"status": "completed"}
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_yaml_write_keeps_previous_document(tmp_path, monkeypatch):
    """Preserve the old YAML document and remove temporary data after failure."""
    destination = tmp_path / "config.yaml"
    config.write_yaml(destination, {"epochs": 300})
    original = destination.read_bytes()

    def fail_after_partial_write(payload, stream, **kwargs):
        """Raise while the new YAML is still confined to its temporary file."""
        stream.write("epochs:")
        raise OSError("injected YAML failure")

    monkeypatch.setattr(config.yaml, "safe_dump", fail_after_partial_write)
    with pytest.raises(OSError, match="injected"):
        config.write_yaml(destination, {"epochs": 1})
    assert destination.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
