"""Convert named Generative GaitNet body-position captures to Lite-MyoNet NCTV arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml


UPSTREAM_REPOSITORY = "https://github.com/namjohn10/GenerativeGaitNet"
UPSTREAM_COMMIT = "eaf3e256925c8b0acc3e95b539535beba105b511"
OUTPUT_LAYOUT = "NCTV"
OUTPUT_DTYPE = "float32"
class ConversionError(ValueError):
    """Raised when a capture or conversion configuration is invalid."""


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest for a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Return the SHA-256 digest of a canonical JSON-compatible value."""

    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_conversion_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML conversion configuration and referenced maps."""

    config_path = Path(path)
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConversionError(f"Could not read conversion configuration: {error}") from error
    if not isinstance(config, dict):
        raise ConversionError("Conversion configuration must contain a mapping at its root.")
    skeleton_config_path = config.get("skeleton_config_path")
    if skeleton_config_path is not None:
        resolved_skeleton = (config_path.parent / str(skeleton_config_path)).resolve()
        try:
            skeleton_config = yaml.safe_load(resolved_skeleton.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ConversionError(f"Could not read skeleton configuration: {error}") from error
        nodes = skeleton_config.get("nodes") if isinstance(skeleton_config, dict) else None
        channels = skeleton_config.get("channels") if isinstance(skeleton_config, dict) else None
        if not isinstance(nodes, list) or not isinstance(channels, list):
            raise ConversionError("Skeleton configuration must provide nodes and channels.")
        ordered_nodes = sorted(nodes, key=lambda item: int(item["index"]))
        if [int(item["index"]) for item in ordered_nodes] != list(range(len(ordered_nodes))):
            raise ConversionError("Skeleton node indices must be contiguous from zero.")
        config["nodes"] = [str(item["id"]) for item in ordered_nodes]
        config["channels"] = [str(channel) for channel in channels]
        config["skeleton_frame_count"] = skeleton_config.get("frame_count")
        point_count = skeleton_config.get("point_count")
        if point_count is not None and int(point_count) != len(ordered_nodes):
            raise ConversionError("Skeleton point_count does not match its node list.")
        config["_skeleton_config_path"] = str(resolved_skeleton)
        config["_skeleton_config_sha256"] = sha256_file(resolved_skeleton)
    label_map_path = config.get("label_map_path")
    if label_map_path is not None:
        resolved_label_map = (config_path.parent / str(label_map_path)).resolve()
        try:
            config["label_map"] = json.loads(resolved_label_map.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ConversionError(f"Could not read label map: {error}") from error
        config["_label_map_path"] = str(resolved_label_map)
        config["_label_map_sha256"] = sha256_file(resolved_label_map)
    _validate_config(config)
    config["_config_path"] = str(config_path.resolve())
    config["_config_sha256"] = sha256_file(config_path)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    """Validate the named-body conversion schema and coordinate settings."""
    if not isinstance(config, dict) or int(config.get("schema_version", -1)) != 1:
        raise ConversionError("Conversion configuration schema_version must be 1.")
    required_sections = ("input_keys", "nodes", "temporal", "coordinates", "channels", "label_map")
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ConversionError(f"Missing configuration sections: {', '.join(missing)}")
    nodes = config["nodes"]
    if not isinstance(nodes, list) or not nodes or len(set(nodes)) != len(nodes):
        raise ConversionError("nodes must contain unique body names in target order.")
    channels = config["channels"]
    if (
        not isinstance(channels, list)
        or len(channels) != 3
        or {str(channel).lower() for channel in channels} != {"x", "y", "z"}
    ):
        raise ConversionError("channels must be a permutation of x, y, and z.")
    temporal = config["temporal"]
    target_frames = int(temporal.get("target_frames", 0))
    target_rate_hz = float(temporal.get("target_rate_hz", 0.0))
    if target_frames <= 0 or not np.isfinite(target_rate_hz) or target_rate_hz <= 0:
        raise ConversionError("temporal.target_frames and target_rate_hz must be positive.")
    skeleton_frame_count = config.get("skeleton_frame_count")
    if skeleton_frame_count is not None and int(skeleton_frame_count) != target_frames:
        raise ConversionError("The conversion frame count must match the skeleton config.")
    coordinates = config["coordinates"]
    scale = float(coordinates.get("scale_to_output_units", 0.0))
    if not np.isfinite(scale) or scale <= 0:
        raise ConversionError("coordinates.scale_to_output_units must be positive and finite.")
    matrix = np.asarray(coordinates.get("rotation_matrix"), dtype=np.float64)
    offset = np.asarray(coordinates.get("translation"), dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ConversionError("coordinates.rotation_matrix must be a finite 3 by 3 matrix.")
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ConversionError("coordinates.translation must contain three finite values.")
    if np.linalg.matrix_rank(matrix) != 3:
        raise ConversionError("coordinates.rotation_matrix must be invertible.")
    if coordinates.get("origin_mode", "world") not in ("world", "root_relative_per_frame"):
        raise ConversionError(
            "coordinates.origin_mode must be 'world' or 'root_relative_per_frame'."
        )
    label_map = config["label_map"]
    if not isinstance(label_map, dict) or not label_map:
        raise ConversionError("label_map must be a non-empty mapping of names to ids.")
    label_ids = [int(value) for value in label_map.values()]
    if len(label_ids) != len(set(label_ids)) or min(label_ids) < 0:
        raise ConversionError("label_map ids must be unique non-negative integers.")
    if sorted(label_ids) != list(range(len(label_ids))):
        raise ConversionError("label_map ids must be contiguous and start at zero.")
    input_keys = config["input_keys"]
    for key in ("positions", "timestamps", "body_names", "label_names"):
        if not isinstance(input_keys.get(key), str) or not input_keys[key]:
            raise ConversionError(f"input_keys.{key} must be a non-empty key name.")


def _decode_text_values(values: np.ndarray, field_name: str) -> list[str]:
    """Decode a one-dimensional string array without loading Python objects."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ConversionError(f"{field_name} must be one-dimensional.")
    output = []
    for value in array.tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        output.append(str(value))
    return output


def _decode_metadata(value: Any, count: int) -> list[dict[str, Any]]:
    """Decode per-trajectory metadata from UTF-8 JSON."""
    if value is None:
        return [{} for _ in range(count)]
    if isinstance(value, np.ndarray) and value.dtype == np.uint8:
        payload = value.tobytes().decode("utf-8")
    elif isinstance(value, np.ndarray) and value.ndim == 0:
        payload = str(value.item())
    elif isinstance(value, bytes):
        payload = value.decode("utf-8")
    elif isinstance(value, str):
        payload = value
    elif isinstance(value, (dict, list)):
        parsed = value
        return _metadata_records(parsed, count)
    else:
        raise ConversionError("Metadata must be UTF-8 JSON, a dictionary, or a list.")
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ConversionError(f"metadata_json is invalid JSON: {error}") from error
    return _metadata_records(parsed, count)


def _metadata_records(value: Any, count: int) -> list[dict[str, Any]]:
    """Validate and copy one metadata object per captured trajectory."""
    if isinstance(value, dict):
        records = value.get("samples", [{} for _ in range(count)])
    else:
        records = value
    if not isinstance(records, list) or len(records) != count:
        raise ConversionError("Metadata must contain one sample record per input sequence.")
    if any(not isinstance(record, dict) for record in records):
        raise ConversionError("Every metadata sample record must be a JSON object.")
    return [dict(record) for record in records]


def _normalise_positions(positions: np.ndarray) -> tuple[np.ndarray, bool]:
    """Validate captured body positions and add a missing sample dimension."""
    values = np.asarray(positions)
    squeezed = values.ndim == 3
    if squeezed:
        values = values[np.newaxis, ...]
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ConversionError("positions must have shape [F,B,3] or [N,F,B,3].")
    if any(size <= 0 for size in values.shape):
        raise ConversionError("positions cannot contain an empty dimension.")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ConversionError("positions must be finite numeric coordinates.")
    return values.astype(np.float64, copy=False), squeezed


def _normalise_timestamps(timestamps: np.ndarray, sample_count: int, frame_count: int) -> np.ndarray:
    """Validate and broadcast strictly increasing per-trajectory timestamps."""
    values = np.asarray(timestamps, dtype=np.float64)
    if values.ndim == 1:
        if values.shape[0] != frame_count:
            raise ConversionError("One-dimensional timestamps must match the frame count.")
        values = np.broadcast_to(values[np.newaxis, :], (sample_count, frame_count))
    if values.shape != (sample_count, frame_count) or not np.isfinite(values).all():
        raise ConversionError("timestamps must have shape [F] or [N,F] and be finite.")
    if np.any(np.diff(values, axis=1) <= 0):
        raise ConversionError("timestamps must be strictly increasing for every sample.")
    return values


def _resolve_window_starts(
    sample_metadata: list[dict[str, Any]],
    config: dict[str, Any],
    override: float | None,
) -> np.ndarray:
    """Resolve an explicit conversion window start for each trajectory."""
    if override is not None:
        start = float(override)
        starts = np.full(len(sample_metadata), start, dtype=np.float64)
    elif config["temporal"].get("window_start_seconds") is not None:
        start = float(config["temporal"]["window_start_seconds"])
        starts = np.full(len(sample_metadata), start, dtype=np.float64)
    else:
        values = [record.get("window_start_seconds") for record in sample_metadata]
        if any(value is None for value in values):
            raise ConversionError(
                "Set temporal.window_start_seconds, pass the CLI override, or store "
                "window_start_seconds in every sample metadata record."
            )
        starts = np.asarray(values, dtype=np.float64)
    if not np.isfinite(starts).all() or np.any(starts < 0):
        raise ConversionError("Window start times must be finite and non-negative.")
    return starts


def convert_positions(
    positions: np.ndarray,
    timestamps: np.ndarray,
    body_names: np.ndarray | list[str] | tuple[str, ...],
    label_names: np.ndarray | list[str] | tuple[str, ...],
    metadata: Any,
    config: dict[str, Any],
    *,
    window_start_seconds: float | None = None,
    source_sha256: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Convert named body positions to NCTV float32 arrays and sample labels."""

    _validate_config(config)
    positions, _ = _normalise_positions(positions)
    sample_count, source_frames, source_bodies, _ = positions.shape
    names = _decode_text_values(np.asarray(body_names), "body_names")
    if len(names) != source_bodies or len(set(names)) != len(names):
        raise ConversionError("body_names must uniquely name every input body axis.")
    name_to_index = {name: index for index, name in enumerate(names)}
    missing_nodes = [name for name in config["nodes"] if name not in name_to_index]
    if missing_nodes:
        raise ConversionError(f"Input capture is missing required bodies: {missing_nodes}")
    if np.asarray(label_names).reshape(-1).shape != (sample_count,):
        raise ConversionError("label_names must contain one class name per sample.")
    names_of_labels = _decode_text_values(np.asarray(label_names).reshape(-1), "label_names")
    label_map = {str(name): int(value) for name, value in config["label_map"].items()}
    unknown_labels = sorted(set(names_of_labels) - set(label_map))
    if unknown_labels:
        raise ConversionError(f"Unknown label names: {unknown_labels}")

    sample_metadata = _decode_metadata(metadata, sample_count)
    starts = _resolve_window_starts(sample_metadata, config, window_start_seconds)
    time_values = _normalise_timestamps(timestamps, sample_count, source_frames)
    temporal = config["temporal"]
    output_frames = int(temporal["target_frames"])
    output_rate = float(temporal["target_rate_hz"])
    relative_times = np.arange(output_frames, dtype=np.float64) / output_rate
    body_indices = [name_to_index[name] for name in config["nodes"]]
    selected = positions[:, :, body_indices, :]

    coordinates = config["coordinates"]
    coordinate_indices = ["xyz".index(str(channel).lower()) for channel in config["channels"]]
    rotation = np.asarray(coordinates["rotation_matrix"], dtype=np.float64)
    scale = float(coordinates["scale_to_output_units"])
    translation = np.asarray(coordinates["translation"], dtype=np.float64)
    transformed = np.einsum("ij,nfbj->nfbi", rotation, selected) * scale
    transformed += translation.reshape(1, 1, 1, 3)
    if coordinates.get("origin_mode", "world") == "root_relative_per_frame":
        transformed -= transformed[:, :, :1, :]

    resampled = np.empty((sample_count, output_frames, len(body_indices), 3), dtype=np.float32)
    updated_metadata = []
    for sample_index in range(sample_count):
        target_times = starts[sample_index] + relative_times
        source_times = time_values[sample_index]
        if target_times[0] < source_times[0] or target_times[-1] > source_times[-1]:
            raise ConversionError(
                f"Sample {sample_index} does not cover requested time window "
                f"[{target_times[0]:.6f}, {target_times[-1]:.6f}] seconds."
            )
        for body_index in range(len(body_indices)):
            for coordinate_index in range(3):
                resampled[sample_index, :, body_index, coordinate_index] = np.interp(
                    target_times,
                    source_times,
                    transformed[sample_index, :, body_index, coordinate_index],
                ).astype(np.float32)
        record = dict(sample_metadata[sample_index])
        record.update(
            {
                "sample_index": sample_index,
                "label_name": names_of_labels[sample_index],
                "label": label_map[names_of_labels[sample_index]],
                "source_frame_count": int(source_frames),
                "source_body_count": int(source_bodies),
                "window_start_seconds": float(starts[sample_index]),
                "window_end_seconds": float(target_times[-1]),
                "target_frame_count": output_frames,
                "target_rate_hz": output_rate,
                "resampling": "linear_interpolation",
            }
        )
        updated_metadata.append(record)

    resampled = resampled[..., coordinate_indices]
    joints_nctv = np.ascontiguousarray(np.transpose(resampled, (0, 3, 1, 2)), dtype=np.float32)
    labels = np.asarray([label_map[name] for name in names_of_labels], dtype=np.int64)
    output_metadata = {
        "schema_version": 1,
        "upstream": {
            "repository": config.get("upstream", {}).get("repository", UPSTREAM_REPOSITORY),
            "commit": config.get("upstream", {}).get("commit", UPSTREAM_COMMIT),
            "license": "Apache-2.0",
        },
        "array_layout": OUTPUT_LAYOUT,
        "array_dtype": OUTPUT_DTYPE,
        "sample_shape": list(joints_nctv.shape[1:]),
        "sample_count": int(sample_count),
        "body_order": list(config["nodes"]),
        "label_map": label_map,
        "label_map_path": config.get("_label_map_path"),
        "label_map_sha256": config.get("_label_map_sha256"),
        "channels": list(config["channels"]),
        "coordinate_transform": coordinates,
        "temporal_config": temporal,
        "source_sha256": source_sha256,
        "config_sha256": config.get("_config_sha256"),
        "skeleton_config_path": config.get("_skeleton_config_path"),
        "skeleton_config_sha256": config.get("_skeleton_config_sha256"),
        "bones_included": False,
        "samples": updated_metadata,
    }
    if not np.isfinite(joints_nctv).all():
        raise ConversionError("Converted NCTV array contains NaN or Inf.")
    return joints_nctv, labels, np.asarray(names_of_labels, dtype=f"<U{max(map(len, names_of_labels))}"), output_metadata


def _write_atomic_npy(path: Path, array: np.ndarray, overwrite: bool) -> None:
    """Save a NumPy array through a temporary sibling and atomic replacement."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as stream:
            temporary = Path(stream.name)
            np.save(stream, array, allow_pickle=False)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _write_atomic_json(path: Path, value: dict[str, Any], overwrite: bool) -> None:
    """Save conversion metadata through a temporary sibling and atomic replacement."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=path.suffix, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_conversion_output(
    output_path: str | Path,
    joints: np.ndarray,
    labels: np.ndarray,
    label_names: np.ndarray,
    metadata: dict[str, Any],
    expected_sample_shape: tuple[int, int, int],
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Write NCTV joints with labels and metadata to NPZ or NPY plus sidecars."""

    path = Path(output_path)
    if path.suffix.lower() not in (".npz", ".npy"):
        raise ConversionError("Output extension must be .npz or .npy.")
    if joints.ndim != 4 or joints.shape[1:] != tuple(expected_sample_shape):
        raise ConversionError(
            f"Output joints must have NCTV shape [N,{','.join(map(str, expected_sample_shape))}]."
        )
    if joints.dtype != np.float32 or labels.shape != (joints.shape[0],):
        raise ConversionError("Output joints must be float32 and labels must have shape [N].")
    if label_names.shape != (joints.shape[0],):
        raise ConversionError("Output label_names must have shape [N].")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npz":
        metadata_path = path.with_name(f"{path.stem}.metadata.json")
        targets = (path, metadata_path)
        existing = [str(target) for target in targets if target.exists()]
        if existing and not overwrite:
            raise FileExistsError(f"Output already exists: {existing}")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(
                    stream,
                    joints=joints,
                    labels=labels,
                )
            temporary.replace(path)
            _write_atomic_json(metadata_path, metadata, overwrite)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return [path, metadata_path]

    label_path = path.with_name(f"{path.stem}.labels.npy")
    metadata_path = path.with_name(f"{path.stem}.metadata.json")
    targets = (path, label_path, metadata_path)
    existing = [str(target) for target in targets if target.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Output already exists: {existing}")
    _write_atomic_npy(path, joints, overwrite)
    _write_atomic_npy(label_path, labels, overwrite)
    _write_atomic_json(metadata_path, metadata, overwrite)
    return list(targets)


def convert_gaitnet_archive(
    input_path: str | Path,
    output_path: str | Path,
    config_path: str | Path,
    *,
    window_start_seconds: float | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert a named-body-position NPZ capture to an NCTV training archive."""

    source_path = Path(input_path)
    if source_path.suffix.lower() != ".npz":
        raise ConversionError("Raw capture input must be an .npz file.")
    config = load_conversion_config(config_path)
    keys = config["input_keys"]
    try:
        with np.load(source_path, allow_pickle=False) as archive:
            required = ("positions", "timestamps", "body_names", "label_names")
            missing = [key_name for key_name in required if keys[key_name] not in archive]
            if missing:
                raise ConversionError(f"Capture is missing configured arrays: {missing}")
            metadata_value = (
                archive[keys["metadata_json_utf8"]]
                if keys.get("metadata_json_utf8") in archive
                else None
            )
            joints, labels, label_names, metadata = convert_positions(
                archive[keys["positions"]],
                archive[keys["timestamps"]],
                archive[keys["body_names"]],
                archive[keys["label_names"]],
                metadata_value,
                config,
                window_start_seconds=window_start_seconds,
                source_sha256=sha256_file(source_path),
            )
    except OSError as error:
        raise ConversionError(f"Could not read capture archive: {error}") from error
    output_paths = write_conversion_output(
        output_path,
        joints,
        labels,
        label_names,
        metadata,
        expected_sample_shape=(
            len(config["channels"]),
            int(config["temporal"]["target_frames"]),
            len(config["nodes"]),
        ),
        overwrite=overwrite,
    )
    return {
        "input": str(source_path.resolve()),
        "input_sha256": sha256_file(source_path),
        "outputs": [str(path.resolve()) for path in output_paths],
        "array_layout": OUTPUT_LAYOUT,
        "array_dtype": OUTPUT_DTYPE,
        "joints_shape": list(joints.shape),
        "labels_shape": list(labels.shape),
        "bones_written": False,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the installed command-line entry point for capture conversion."""

    parser = argparse.ArgumentParser(
        description="Convert named body-position captures to float32 NCTV arrays."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "configs"
        / "conversion"
        / "generative_gaitnet.yaml",
    )
    parser.add_argument("--window-start-seconds", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = convert_gaitnet_archive(
            args.input,
            args.output,
            args.config,
            window_start_seconds=args.window_start_seconds,
            overwrite=args.overwrite,
        )
    except (ConversionError, FileExistsError, OSError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": "ok", **report}, ensure_ascii=False, indent=2))
    return 0
