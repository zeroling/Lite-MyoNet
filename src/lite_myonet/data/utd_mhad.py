"""Prepare UTD-MHAD skeleton sequences as deterministic NCTV datasets."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import stat
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
from sklearn.model_selection import train_test_split


OFFICIAL_PAGE = "https://personal.utdallas.edu/~kehtar/UTD-MHAD.html"
ARCHIVE_URLS = {
    "Skeleton.zip": "https://personal.utdallas.edu/~kehtar/UTD-MAD/Skeleton.zip",
    "Sample_Code.zip": "https://personal.utdallas.edu/~kehtar/UTD-MAD/Sample_Code.zip",
}
ACTION_NAMES = (
    "swipt_left",
    "swipt_right",
    "wave",
    "clap",
    "throw",
    "arm_cross",
    "basketball_shoot",
    "draw_x",
    "draw_circle_CW",
    "draw_circle_CCW",
    "draw_triangle",
    "bowling",
    "boxing",
    "baseball_swing",
    "tennis_swing",
    "arm_curl",
    "tennis_serve",
    "push",
    "knock",
    "catch",
    "pickup_throw",
    "jog",
    "walk",
    "sit2stand",
    "stand2sit",
    "lunge",
    "squat",
)
FILENAME_PATTERN = re.compile(
    r"^a(?P<action>\d+)_s(?P<subject>\d+)_t(?P<trial>\d+)_skeleton\.mat$",
    re.IGNORECASE,
)
EXPECTED_COMBINATION_COUNT = 27 * 8 * 4


class UTDMHADDataError(ValueError):
    """Raised when a UTD-MHAD source or conversion contract is invalid."""


@dataclass(frozen=True)
class UTDRecord:
    """Hold one validated and temporally resampled UTD-MHAD sequence."""

    action_id: int
    subject_id: int
    trial_id: int
    source_file: str
    source_frame_count: int
    joints: np.ndarray


def parse_skeleton_filename(filename: str | Path) -> tuple[int, int, int]:
    """Parse an official action, subject, and trial identifier from a filename."""
    match = FILENAME_PATTERN.fullmatch(Path(filename).name)
    if match is None:
        raise UTDMHADDataError(f"Invalid UTD-MHAD skeleton filename: {filename}")
    action, subject, trial = (int(match.group(key)) for key in ("action", "subject", "trial"))
    if not 1 <= action <= 27 or not 1 <= subject <= 8 or not 1 <= trial <= 4:
        raise UTDMHADDataError(
            f"Filename identifiers are outside the official range: {filename}"
        )
    return action, subject, trial


def convert_d_skel(
    d_skel: np.ndarray,
    *,
    target_frames: int = 120,
    root_index: int = 3,
) -> np.ndarray:
    """Convert one ``[20, 3, T]`` MAT array to root-centered ``[3, T, 20]`` data."""
    values = np.asarray(d_skel)
    if values.ndim != 3 or values.shape[:2] != (20, 3) or values.shape[2] <= 0:
        raise UTDMHADDataError(
            f"d_skel must have shape [20,3,T] with T>0, got {values.shape}."
        )
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise UTDMHADDataError("d_skel must contain finite numeric coordinates.")
    if isinstance(target_frames, bool) or not isinstance(target_frames, int) or target_frames <= 0:
        raise UTDMHADDataError("target_frames must be a positive integer.")
    if isinstance(root_index, bool) or not isinstance(root_index, int) or not 0 <= root_index < 20:
        raise UTDMHADDataError("root_index must be an index in [0,20).")

    time_major = np.asarray(values, dtype=np.float64).transpose(2, 0, 1)
    time_major -= time_major[:, root_index : root_index + 1, :]
    source_positions = np.arange(time_major.shape[0], dtype=np.float64)
    target_positions = np.linspace(
        0.0, float(time_major.shape[0] - 1), target_frames, dtype=np.float64
    )
    resampled = np.empty((target_frames, 20, 3), dtype=np.float32)
    for node_index in range(20):
        for channel_index in range(3):
            resampled[:, node_index, channel_index] = np.interp(
                target_positions,
                source_positions,
                time_major[:, node_index, channel_index],
            )
    joints = np.ascontiguousarray(resampled.transpose(2, 0, 1), dtype=np.float32)
    if not np.isfinite(joints).all():
        raise UTDMHADDataError("Converted sequence contains NaN or Inf.")
    return joints


def _validate_zip(path: Path) -> tuple[int, int]:
    """Validate ZIP readability and return its member count and total uncompressed size."""
    if not path.is_file() or not zipfile.is_zipfile(path):
        raise UTDMHADDataError(f"Archive is not a readable ZIP file: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            broken_member = archive.testzip()
            if broken_member is not None:
                raise UTDMHADDataError(
                    f"ZIP member failed its CRC check: {broken_member} in {path.name}"
                )
            infos = archive.infolist()
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise UTDMHADDataError(f"Could not read ZIP archive {path}: {error}") from error
    return len(infos), sum(item.file_size for item in infos)


def download_archive(url: str, destination: str | Path, *, timeout: int = 60) -> Path:
    """Download or resume one archive and accept it only if it is a readable ZIP."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        _validate_zip(target)
        return target

    partial = target.with_name(target.name + ".part")
    if partial.exists():
        try:
            _validate_zip(partial)
        except UTDMHADDataError:
            pass
        else:
            partial.replace(target)
            return target
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        if not offset:
            raise UTDMHADDataError(f"Could not download {url}: {error}") from error
        request = urllib.request.Request(url)
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.URLError as restart_error:
            raise UTDMHADDataError(
                f"Could not restart download for {url}: {restart_error}"
            ) from restart_error
    except urllib.error.URLError as error:
        raise UTDMHADDataError(f"Could not download {url}: {error}") from error

    append = False
    if offset and getattr(response, "status", response.getcode()) == 206:
        content_range = response.headers.get("Content-Range", "")
        append = content_range.lower().startswith(f"bytes {offset}-")
    if offset and not append:
        response.close()
        request = urllib.request.Request(url)
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.URLError as error:
            raise UTDMHADDataError(f"Could not restart download for {url}: {error}") from error
    try:
        with response, partial.open("ab" if append else "wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    except OSError as error:
        raise UTDMHADDataError(f"Could not save downloaded archive {partial}: {error}") from error
    _validate_zip(partial)
    partial.replace(target)
    return target


def ensure_official_archives(raw_dir: str | Path) -> dict[str, Path]:
    """Ensure both official UTD-MHAD source archives are available and readable."""
    directory = Path(raw_dir)
    return {
        name: download_archive(url, directory / name)
        for name, url in ARCHIVE_URLS.items()
    }


def safe_extract_zip(archive_path: str | Path, destination: str | Path) -> list[Path]:
    """Extract a ZIP after rejecting traversal, drive paths, and symbolic links."""
    archive_path = Path(archive_path)
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _validate_zip(archive_path)
    extracted: list[Path] = []
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        validated: list[tuple[zipfile.ZipInfo, Path]] = []
        for item in entries:
            member = item.filename
            member_path = PurePosixPath(member)
            if (
                not member
                or member_path.is_absolute()
                or ".." in member_path.parts
                or "\\" in member
                or any(":" in part for part in member_path.parts)
            ):
                raise UTDMHADDataError(f"Unsafe ZIP member path: {member!r}")
            mode = item.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise UTDMHADDataError(f"Symbolic links are not allowed in ZIP archives: {member}")
            target = (root / Path(*member_path.parts)).resolve()
            if not target.is_relative_to(root):
                raise UTDMHADDataError(f"ZIP member escapes extraction directory: {member}")
            validated.append((item, target))

        for item, target in validated:
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            extracted.append(target)
    return extracted


def _load_label_map(path: str | Path) -> dict[str, int]:
    """Load the official action-name mapping and validate contiguous zero-based IDs."""
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UTDMHADDataError(f"Could not read UTD-MHAD label map {source}: {error}") from error
    if not isinstance(raw, dict) or set(raw) != set(ACTION_NAMES):
        raise UTDMHADDataError("Label map keys must exactly match the 27 official action names.")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw.values()):
        raise UTDMHADDataError("Label map values must be integer class IDs.")
    if sorted(raw.values()) != list(range(len(ACTION_NAMES))):
        raise UTDMHADDataError("Label map IDs must be unique and contiguous from zero.")
    return {str(name): int(label) for name, label in raw.items()}


def _load_skeleton_records(
    skeleton_directory: Path,
    *,
    root_index: int,
    target_frames: int,
) -> tuple[list[UTDRecord], list[dict[str, str]], list[tuple[int, int, int]]]:
    """Load all readable skeleton files and record malformed or missing combinations."""
    try:
        from scipy.io import loadmat
    except ImportError as error:
        raise UTDMHADDataError(
            "UTD-MHAD conversion requires scipy.io.loadmat; install the package's data-preparation extra."
        ) from error

    paths = sorted(skeleton_directory.rglob("*_skeleton.mat"))
    records: list[UTDRecord] = []
    anomalies: list[dict[str, str]] = []
    seen: set[tuple[int, int, int]] = set()
    for path in paths:
        try:
            action, subject, trial = parse_skeleton_filename(path.name)
            identity = (action, subject, trial)
            if identity in seen:
                anomalies.append({"file": str(path.relative_to(skeleton_directory)), "reason": "duplicate combination"})
                continue
            values = loadmat(path, variable_names=["d_skel"]).get("d_skel")
            if values is None:
                raise UTDMHADDataError("missing d_skel variable")
            joints = convert_d_skel(values, target_frames=target_frames, root_index=root_index)
        except (OSError, ValueError, TypeError, KeyError) as error:
            anomalies.append(
                {"file": str(path.relative_to(skeleton_directory)), "reason": str(error)}
            )
            continue
        seen.add(identity)
        records.append(
            UTDRecord(
                action_id=action,
                subject_id=subject,
                trial_id=trial,
                source_file=str(path.relative_to(skeleton_directory)).replace("\\", "/"),
                source_frame_count=int(values.shape[2]),
                joints=joints,
            )
        )
    records.sort(key=lambda item: (item.action_id, item.subject_id, item.trial_id))
    expected = {
        (action, subject, trial)
        for action in range(1, 28)
        for subject in range(1, 9)
        for trial in range(1, 5)
    }
    missing = sorted(expected - seen)
    return records, anomalies, missing


def create_split_assignments(
    records: Iterable[UTDRecord],
    labels: np.ndarray,
    *,
    validation_size: float = 0.2,
    random_state: int = 2026,
) -> list[str]:
    """Assign odd subjects to train/validation and even subjects to held-out test."""
    rows = list(records)
    labels = np.asarray(labels)
    if labels.shape != (len(rows),) or labels.dtype.kind not in "iu":
        raise UTDMHADDataError("labels must be an integer vector aligned with the records.")
    if not math.isfinite(validation_size) or not 0.0 < validation_size < 1.0:
        raise UTDMHADDataError("validation_size must be finite and strictly between zero and one.")
    development = [index for index, row in enumerate(rows) if row.subject_id % 2 == 1]
    test = [index for index, row in enumerate(rows) if row.subject_id % 2 == 0]
    if not development or not test:
        raise UTDMHADDataError("Both odd-subject development and even-subject test data are required.")
    try:
        train, validation = train_test_split(
            development,
            test_size=validation_size,
            random_state=random_state,
            stratify=labels[development],
        )
    except ValueError as error:
        raise UTDMHADDataError(f"Could not create the stratified validation split: {error}") from error
    assignments = [""] * len(rows)
    for indices, split in ((train, "train"), (validation, "val"), (test, "test")):
        for index in indices:
            assignments[index] = split
    if any(not split for split in assignments):
        raise UTDMHADDataError("Every valid record must belong to exactly one split.")
    return assignments


def _write_npz(path: Path, joints: np.ndarray, labels: np.ndarray, *, overwrite: bool) -> None:
    """Write a strict two-array NPZ with NCTV float32 joints and int64 labels."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                joints=np.ascontiguousarray(joints, dtype=np.float32),
                labels=np.ascontiguousarray(labels, dtype=np.int64),
            )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _support_by_class(
    labels: np.ndarray,
    assignments: list[str],
    label_map: dict[str, int],
) -> dict[str, dict[str, int]]:
    """Count total and per-split sample support for every action class."""
    result: dict[str, dict[str, int]] = {}
    for name, label in label_map.items():
        result[name] = {
            split: int(sum(int(labels[index]) == label and assignments[index] == split for index in range(len(labels))))
            for split in ("train", "val", "test")
        }
        result[name]["total"] = sum(result[name][split] for split in ("train", "val", "test"))
    return result


def _audit_subject_split(
    records: list[UTDRecord], assignments: list[str]
) -> dict[str, Any]:
    """Verify one-time sample assignment and the fixed odd/even subject boundary."""
    if len(records) != len(assignments) or any(
        value not in {"train", "val", "test"} for value in assignments
    ):
        raise UTDMHADDataError("Every sample must be assigned exactly once to a known split.")
    identities = [
        (record.action_id, record.subject_id, record.trial_id) for record in records
    ]
    if len(identities) != len(set(identities)):
        raise UTDMHADDataError("Duplicate action/subject/trial identities remain after inspection.")
    split_subjects = {
        split: sorted(
            {record.subject_id for record, assigned in zip(records, assignments) if assigned == split}
        )
        for split in ("train", "val", "test")
    }
    train_val_subjects = set(split_subjects["train"]) | set(split_subjects["val"])
    test_subjects = set(split_subjects["test"])
    test_development_overlap = sorted(test_subjects & train_val_subjects)
    invalid_development_subjects = sorted(
        subject for subject in train_val_subjects if subject % 2 == 0
    )
    invalid_test_subjects = sorted(subject for subject in test_subjects if subject % 2 == 1)
    if test_development_overlap or invalid_development_subjects or invalid_test_subjects:
        raise UTDMHADDataError(
            "Subject split leakage detected across the fixed odd/even train-test boundary."
        )
    return {
        "sample_identity_unique": True,
        "all_samples_assigned_once": True,
        "subjects_by_split": split_subjects,
        "train_val_subject_overlap": sorted(
            set(split_subjects["train"]) & set(split_subjects["val"])
        ),
        "test_train_subject_overlap": sorted(
            set(split_subjects["test"]) & set(split_subjects["train"])
        ),
        "test_val_subject_overlap": sorted(
            set(split_subjects["test"]) & set(split_subjects["val"])
        ),
        "test_subject_disjoint_from_development": True,
        "validation_is_sample_level_within_development": True,
    }


def prepare_utd_mhad(
    skeleton_zip: str | Path,
    sample_code_zip: str | Path,
    output_directory: str | Path,
    skeleton_config: str | Path,
    label_map_path: str | Path,
    *,
    validation_size: float = 0.2,
    random_state: int = 2026,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert official MAT sequences, create deterministic splits, and write artifacts."""
    from ..models.skeleton import load_skeleton

    skeleton_archive = Path(skeleton_zip)
    sample_archive = Path(sample_code_zip)
    skeleton_members, skeleton_uncompressed = _validate_zip(skeleton_archive)
    sample_members, sample_uncompressed = _validate_zip(sample_archive)
    skeleton = load_skeleton(skeleton_config)
    if skeleton.point_count != 20 or skeleton.root_index != 3 or skeleton.frame_count != 120:
        raise UTDMHADDataError(
            "UTD-MHAD skeleton config must define 20 nodes, root index 3, and 120 frames."
        )
    if skeleton.channels != ("x", "y", "z"):
        raise UTDMHADDataError("UTD-MHAD coordinate channels must be [x,y,z].")
    label_map = _load_label_map(label_map_path)

    extracted_root = skeleton_archive.parent / "extracted"
    safe_extract_zip(skeleton_archive, extracted_root)
    safe_extract_zip(sample_archive, extracted_root)
    joint_order_file = extracted_root / "Skeleton" / "Skeleton_joint_order.txt"
    if not joint_order_file.is_file():
        raise UTDMHADDataError("Official Skeleton_joint_order.txt is missing from Skeleton.zip.")
    joint_order_text = joint_order_file.read_text(encoding="utf-8", errors="replace").lower()
    if "hip_center" not in joint_order_text or "20 x 3 x num_frame" not in joint_order_text:
        raise UTDMHADDataError("Official joint-order documentation does not match the expected format.")

    records, anomalies, missing = _load_skeleton_records(
        extracted_root / "Skeleton",
        root_index=skeleton.root_index,
        target_frames=skeleton.frame_count,
    )
    if not records:
        raise UTDMHADDataError("No valid skeleton records were found.")
    labels = np.asarray(
        [label_map[ACTION_NAMES[record.action_id - 1]] for record in records], dtype=np.int64
    )
    assignments = create_split_assignments(
        records,
        labels,
        validation_size=validation_size,
        random_state=random_state,
    )
    split_audit = _audit_subject_split(records, assignments)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    target_files = [
        *(output_directory / f"{split}.npz" for split in ("train", "val", "test")),
        output_directory / "split.csv",
        output_directory / "metadata.json",
    ]
    existing = [str(path) for path in target_files if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Output already exists: {existing}")
    for split in ("train", "val", "test"):
        indices = [index for index, value in enumerate(assignments) if value == split]
        joints = np.stack([records[index].joints for index in indices]).astype(np.float32, copy=False)
        split_labels = labels[indices]
        _write_npz(output_directory / f"{split}.npz", joints, split_labels, overwrite=overwrite)

    csv_path = output_directory / "split.csv"
    if csv_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {csv_path}")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "sample_id",
                "action_id",
                "action_name",
                "label",
                "subject_id",
                "trial_id",
                "split",
                "source_file",
                "source_frame_count",
            ),
        )
        writer.writeheader()
        for record, label, split in zip(records, labels.tolist(), assignments):
            writer.writerow(
                {
                    "sample_id": f"a{record.action_id}_s{record.subject_id}_t{record.trial_id}",
                    "action_id": record.action_id,
                    "action_name": ACTION_NAMES[record.action_id - 1],
                    "label": label,
                    "subject_id": record.subject_id,
                    "trial_id": record.trial_id,
                    "split": split,
                    "source_file": record.source_file,
                    "source_frame_count": record.source_frame_count,
                }
            )

    split_counts = {split: assignments.count(split) for split in ("train", "val", "test")}
    class_support = _support_by_class(labels, assignments, label_map)
    frame_counts = [record.source_frame_count for record in records]
    metadata = {
        "schema_version": 1,
        "dataset": "UTD-MHAD",
        "official_page": OFFICIAL_PAGE,
        "official_archive_urls": ARCHIVE_URLS,
        "archives": {
            skeleton_archive.name: {
                "size_bytes": skeleton_archive.stat().st_size,
                "member_count": skeleton_members,
                "uncompressed_bytes": skeleton_uncompressed,
                "valid_zip": True,
            },
            sample_archive.name: {
                "size_bytes": sample_archive.stat().st_size,
                "member_count": sample_members,
                "uncompressed_bytes": sample_uncompressed,
                "valid_zip": True,
            },
        },
        "raw_inspection": {
            "expected_action_subject_trial_combinations": EXPECTED_COMBINATION_COUNT,
            "skeleton_mat_file_count": len(records) + len(anomalies),
            "valid_sequence_count": len(records),
            "missing_combinations": [
                {"action_id": action, "subject_id": subject, "trial_id": trial}
                for action, subject, trial in missing
            ],
            "duplicate_or_malformed_anomalies": anomalies,
            "anomaly_count": len(anomalies),
            "raw_shape": "[20,3,T]",
            "raw_dtype": "float64",
            "frame_count_min": min(frame_counts),
            "frame_count_max": max(frame_counts),
            "finite_values": True,
        },
        "protocol": {
            "development_subjects": [1, 3, 5, 7],
            "test_subjects": [2, 4, 6, 8],
            "validation_fraction_of_development": validation_size,
            "validation_random_state": random_state,
            "validation_stratification": "global stratified split by action label",
            "description": (
                "Project protocol: odd subjects form development data and even subjects form fixed test data; "
                "a deterministic stratified validation subset is drawn from odd subjects. This is not the "
                "official full-training odd-subject result."
            ),
            "split_leakage_audit": split_audit,
        },
        "conversion": {
            "joint_order_source": "Skeleton/Skeleton_joint_order.txt in the official archive",
            "joint_order": [
                "head",
                "shoulder_center",
                "spine",
                "hip_center",
                "left_shoulder",
                "left_elbow",
                "left_wrist",
                "left_hand",
                "right_shoulder",
                "right_elbow",
                "right_wrist",
                "right_hand",
                "left_hip",
                "left_knee",
                "left_ankle",
                "left_foot",
                "right_hip",
                "right_knee",
                "right_ankle",
                "right_foot",
            ],
            "root_index": 3,
            "root_centering": "subtract hip_center coordinates at each frame",
            "temporal_resampling": "linear interpolation over the full source sequence",
            "target_frames": 120,
            "coordinate_scale": "preserved from source meters; no unit conversion or axis flip",
            "array_layout": "NCTV",
            "joints_dtype": "float32",
            "labels_dtype": "int64",
            "npz_keys": ["joints", "labels"],
            "persist_bones": False,
        },
        "artifacts": {
            "split_csv": csv_path.name,
            "splits": {split: f"{split}.npz" for split in ("train", "val", "test")},
            "split_counts": split_counts,
            "class_support": class_support,
        },
    }
    metadata_path = output_directory / "metadata.json"
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {metadata_path}")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    """Download, validate, convert, and report the official UTD-MHAD skeleton data."""
    import argparse

    project_root = Path(__file__).resolve().parents[3]
    default_raw = project_root / "data" / "utd_mhad" / "raw"
    default_processed = project_root / "data" / "utd_mhad" / "processed"
    parser = argparse.ArgumentParser(
        description="Prepare UTD-MHAD skeleton MAT sequences as train/val/test NCTV archives."
    )
    parser.add_argument("--raw-dir", type=Path, default=default_raw)
    parser.add_argument("--output-dir", type=Path, default=default_processed)
    parser.add_argument(
        "--skeleton-config",
        type=Path,
        default=project_root / "configs" / "skeletons" / "utd_mhad20.yaml",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=project_root / "configs" / "label_maps" / "utd_mhad27.json",
    )
    parser.add_argument("--skeleton-zip", type=Path)
    parser.add_argument("--sample-code-zip", type=Path)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.skeleton_zip is None and args.sample_code_zip is None:
            archives = ensure_official_archives(args.raw_dir)
            skeleton_zip = archives["Skeleton.zip"]
            sample_code_zip = archives["Sample_Code.zip"]
        elif args.skeleton_zip is not None and args.sample_code_zip is not None:
            skeleton_zip = args.skeleton_zip
            sample_code_zip = args.sample_code_zip
        else:
            raise UTDMHADDataError(
                "Provide both --skeleton-zip and --sample-code-zip, or neither to download both."
            )
        metadata = prepare_utd_mhad(
            skeleton_zip,
            sample_code_zip,
            args.output_dir,
            args.skeleton_config,
            args.label_map,
            validation_size=args.validation_size,
            random_state=args.random_state,
            overwrite=args.overwrite,
        )
    except (UTDMHADDataError, OSError, zipfile.BadZipFile) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": "ok", **metadata}, ensure_ascii=False, indent=2))
    return 0
