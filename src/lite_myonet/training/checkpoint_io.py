"""Durable checkpoint writes and conservative recovery of training state."""

import os
import pickle
import shutil
import tempfile
import uuid
import zipfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import torch


class CheckpointRecoveryError(RuntimeError):
    """Raised when checkpoint files exist but none is a complete loadable state."""


_CHECKPOINT_READ_ERRORS = (
    OSError,
    EOFError,
    RuntimeError,
    ValueError,
    TypeError,
    IndexError,
    zipfile.BadZipFile,
    pickle.UnpicklingError,
)


def _load_checkpoint(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch 2.0 compatibility.
        return torch.load(path, map_location=map_location)


def _validate_checkpoint(value, required_keys):
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    missing = sorted(set(required_keys) - set(value))
    if missing:
        raise ValueError(f"checkpoint is missing required training state: {missing}")
    epoch = value.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"checkpoint has an invalid epoch: {epoch!r}")


def load_latest_checkpoint(candidates, required_keys, map_location="cpu"):
    """Return the highest-epoch readable, structurally complete candidate.

    Candidate order breaks epoch ties, so callers should list the preferred
    primary checkpoint before its backup and any best checkpoint.
    """
    required_keys = set(required_keys)
    existing = []
    valid = []
    errors = []
    seen = set()

    for order, raw_path in enumerate(candidates):
        path = Path(raw_path).expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            continue
        existing.append(path)
        try:
            value = _load_checkpoint(path, map_location=map_location)
            _validate_checkpoint(value, required_keys)
        except _CHECKPOINT_READ_ERRORS as error:
            errors.append(f"{path.name}: {type(error).__name__}: {error}")
            continue
        valid.append((int(value["epoch"]), -order, path, value))

    if not existing:
        candidate_names = ", ".join(str(Path(path)) for path in candidates)
        raise FileNotFoundError(f"no checkpoint candidate exists: {candidate_names}")
    if not valid:
        details = "; ".join(errors) or "no complete candidates"
        raise CheckpointRecoveryError(
            "no complete readable training checkpoint was found; refusing to start from scratch. "
            f"Candidates: {details}"
        )

    _, _, selected_path, selected_value = max(valid, key=lambda item: (item[0], item[1]))
    return selected_path, selected_value


def _fsync_parent(path):
    if os.name == "nt":
        return
    try:
        descriptor = os.open(str(path.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _temporary_path(parent, name):
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{name}.", suffix=".tmp", dir=str(parent)
    )
    return descriptor, Path(raw_path)


def _copy_atomic(source, destination):
    descriptor, temporary = _temporary_path(destination.parent, destination.name)
    try:
        with os.fdopen(descriptor, "wb") as output_file, Path(source).open("rb") as source_file:
            shutil.copyfileobj(source_file, output_file, length=1024 * 1024)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, destination)
        _fsync_parent(destination)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise


def _preserve_corrupt_file(path):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = path.with_name(
        f"{path.name}.corrupt-{stamp}-{uuid.uuid4().hex[:8]}"
    )
    _copy_atomic(path, destination)
    return destination


def save_checkpoint_atomic(path, value, required_keys=(), backup_path=None):
    """Durably save a checkpoint while retaining the previous usable state.

    The previous target is copied to ``<name>.bak`` only if it passes the same
    structural completeness check. A damaged target is archived under a unique
    ``.corrupt-*`` name and is never used to replace an existing backup.
    """
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    required_keys = set(required_keys)
    _validate_checkpoint(value, required_keys)
    backup_path = (
        Path(backup_path).expanduser().resolve()
        if backup_path is not None
        else path.with_name(f"{path.name}.bak")
    )

    descriptor, temporary = _temporary_path(path.parent, path.name)
    try:
        with os.fdopen(descriptor, "wb") as output_file:
            torch.save(value, output_file)
            output_file.flush()
            os.fsync(output_file.fileno())

        if path.is_file():
            try:
                previous = _load_checkpoint(path, map_location="cpu")
                _validate_checkpoint(previous, required_keys)
            except _CHECKPOINT_READ_ERRORS:
                _preserve_corrupt_file(path)
            else:
                _copy_atomic(path, backup_path)

        os.replace(temporary, path)
        _fsync_parent(path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise

    return path
