"""Write named body-position captures in the converter's neutral NPZ input format."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def write_capture_archive(
    output_path: str | Path,
    positions: np.ndarray,
    timestamps: np.ndarray,
    body_names: list[str] | tuple[str, ...],
    label_names: list[str] | tuple[str, ...],
    sample_metadata: list[dict[str, Any]],
    *,
    overwrite: bool = False,
) -> Path:
    """Store simulator positions and labels without changing coordinates or frame order."""

    path = Path(output_path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Capture output must use the .npz extension.")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Capture output already exists: {path}")

    values = np.asarray(positions)
    if values.ndim == 3:
        values = values[np.newaxis, ...]
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError("positions must have shape [F,B,3] or [N,F,B,3].")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("positions must be finite numeric coordinates.")
    names = [str(name) for name in body_names]
    labels = [str(name) for name in label_names]
    if len(names) != values.shape[2] or len(set(names)) != len(names):
        raise ValueError("body_names must uniquely name the body axis.")
    if len(labels) != values.shape[0]:
        raise ValueError("label_names must contain one class name per sample.")
    time_values = np.asarray(timestamps, dtype=np.float64)
    if time_values.ndim == 1:
        if time_values.shape != (values.shape[1],):
            raise ValueError("timestamps must match the frame count.")
    elif time_values.shape != (values.shape[0], values.shape[1]):
        raise ValueError("timestamps must have shape [F] or [N,F].")
    if not np.isfinite(time_values).all() or np.any(np.diff(time_values, axis=-1) <= 0):
        raise ValueError("timestamps must be finite and strictly increasing.")
    if len(sample_metadata) != values.shape[0] or any(
        not isinstance(record, dict) for record in sample_metadata
    ):
        raise ValueError("sample_metadata must contain one JSON object per sample.")

    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps({"samples": sample_metadata}, ensure_ascii=False).encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(
                stream,
                positions=values,
                timestamps=time_values,
                body_names=np.asarray(names, dtype=f"<U{max(map(len, names))}"),
                label_names=np.asarray(labels, dtype=f"<U{max(map(len, labels))}"),
                metadata_json_utf8=np.frombuffer(metadata, dtype=np.uint8),
            )
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path
