"""Configuration and provenance utilities shared by public entry points."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and attach the resolved source path."""
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {resolved}")
    payload["_config_path"] = str(resolved)
    return payload


def resolve_config_path(config: Mapping[str, Any], value: str | Path) -> Path:
    """Resolve a path relative to the YAML file that declared it."""
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    source = config.get("_config_path")
    base = Path(str(source)).resolve().parent if source else Path.cwd()
    return (base / expanded).resolve()


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a mapping after deterministic JSON serialization."""
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def write_json(path: str | Path, payload: Any) -> Path:
    """Flush a JSON document and atomically replace its destination file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def write_yaml(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Flush a stable-order YAML mapping and atomically replace its destination."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(dict(payload), stream, sort_keys=False, allow_unicode=True)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_label_map(path: str | Path) -> dict[str, int]:
    """Load and validate a contiguous name-to-index label mapping."""
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Label map must be a non-empty JSON object.")
    labels = {str(name): int(index) for name, index in payload.items()}
    expected = list(range(len(labels)))
    if sorted(labels.values()) != expected:
        raise ValueError("Label indices must be unique and contiguous from zero.")
    return labels
