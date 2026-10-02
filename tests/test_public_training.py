"""CPU integration coverage for portable public teacher/student training."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from lite_myonet.models.skeleton import load_skeleton
from lite_myonet.training import engine as common
from lite_myonet.training.engine import (
    TwoStageLR,
    _make_optimizer,
    atomic_save_checkpoint,
    build_model,
    load_training_checkpoint,
    restore_model,
    skeleton_mapping,
)
from lite_myonet.training.student_trainer import train_student
from lite_myonet.training import student_trainer as student_cli
from lite_myonet.training.teacher_trainer import train_teacher


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    """Create small deterministic NCTV splits, schema files, and CPU configs."""
    root.mkdir(parents=True, exist_ok=True)
    skeleton = {
        "schema_version": 1,
        "skeleton_id": "cpu_fixture_v1",
        "point_count": 5,
        "root_index": 0,
        "nodes": [{"id": f"node_{index}", "index": index} for index in range(5)],
        "model_graph_edges": [[0, 1], [1, 2], [0, 3], [3, 4]],
        "bone_vector_edges": [[1, 0], [2, 1], [3, 0], [4, 3]],
        "symmetry_pairs": [[1, 3], [2, 4]],
        "channels": ["x", "y", "z"],
        "frame_count": 32,
    }
    skeleton_path = root / "skeleton.yaml"
    skeleton_path.write_text(yaml.safe_dump(skeleton, sort_keys=False), encoding="utf-8")
    labels = {"class_zero": 0, "class_one": 1}
    label_path = root / "labels.json"
    label_path.write_text(json.dumps(labels), encoding="utf-8")
    rng = np.random.default_rng(418)
    data_paths = []
    for split, count in (("train", 4), ("val", 2), ("test", 2)):
        split_root = root / split
        split_root.mkdir()
        joints = rng.normal(size=(count, 3, 32, 5)).astype(np.float32)
        class_labels = np.arange(count, dtype=np.int64) % 2
        np.savez(split_root / f"{split}.npz", joints=joints, labels=class_labels)
        data_paths.append(split_root / f"{split}.npz")
    shared = {
        "schema_version": 1,
        "skeleton_config": "skeleton.yaml",
        "label_map": "labels.json",
        "data": {
            "train": str(data_paths[0].relative_to(root)),
            "val": str(data_paths[1].relative_to(root)),
            "test": str(data_paths[2].relative_to(root)),
        },
        "output_root": "runs",
        "seeds": [731],
        "epochs": 2,
        "batch_size": 2,
        "num_workers": 0,
        "cpu_threads": 1,
        "precision": "fp32",
        "deterministic": True,
        "selection_metric": "macro_f1",
        "loss": {"label_smoothing": 0.05},
        "optimizer": {
            "type": "AdamW",
            "peak_lr": 0.0005,
            "weight_decay": 0.0001,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
        },
        "scheduler": {
            "warmup_ratio": 0.1,
            "warmup_init_lr": 0.0001,
            "warmup_gamma": 1.0,
            "decay_gamma": 2.0,
        },
    }
    teacher = {
        **shared,
        "model": {
            "dropout": 0.05,
            "spatial_attention_config": {"num_heads": 8, "cdiv_win": 5},
            "symmetry_attention_config": {"num_heads": 8, "use_contra": True},
            "scale_attention_config": {"num_heads": 8},
            "head_config": {"dropout": 0.1, "norm": "layernorm"},
        },
    }
    teacher_path = root / "teacher.yaml"
    teacher_path.write_text(yaml.safe_dump(teacher, sort_keys=False), encoding="utf-8")
    student = {
        **shared,
        "model": {"channels": [32, 64, 128], "dropout": 0.05, "fourier_degree": 2},
        "teacher": {"checkpoint": "runs/teacher_seed_731/best_teacher.pth"},
        "distillation": {
            "kd_temperature": 4.0,
            "topology_temperature": 0.2,
            "kd_weight": 1.0,
            "static_adjacency_weight": 1.0,
            "dynamic_similarity_weight": 1.0,
        },
    }
    student_path = root / "student.yaml"
    student_path.write_text(yaml.safe_dump(student, sort_keys=False), encoding="utf-8")
    return teacher_path, student_path, label_path


def _write_portable_model_checkpoint(path: Path, skeleton_path: Path, model_type: str = "teacher") -> None:
    """Write a small portable teacher or student model payload for path-resolution tests."""
    skeleton = load_skeleton(skeleton_path)
    if model_type == "teacher":
        model_config = {
            "num_classes": 2,
            "in_channels": 3,
            "dropout": 0.05,
            "spatial_attention_config": {"num_heads": 8, "cdiv_win": 5},
            "symmetry_attention_config": {"num_heads": 8, "use_contra": True},
            "scale_attention_config": {"num_heads": 8},
            "head_config": {"dropout": 0.1, "norm": "layernorm"},
        }
    else:
        model_config = {
            "num_classes": 2,
            "in_channels": 3,
            "channels": [32, 64, 128],
            "dropout": 0.05,
            "fourier_degree": 2,
        }
    model = build_model(model_type, skeleton, model_config)
    payload = {
        "model_type": model_type,
        "model_config": model_config,
        "skeleton_config": skeleton_mapping(skeleton),
        "model_state_dict": model.state_dict(),
        "label_map": {"class_zero": 0, "class_one": 1},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _train_student_with_teacher_path(
    monkeypatch: pytest.MonkeyPatch,
    config_path: Path,
    teacher_path: Path,
) -> Path:
    """Run student checkpoint validation while replacing the actual training loop."""
    observed: dict[str, Any] = {}

    def fake_training_loop(config, model_type, **kwargs):
        """Record checkpoint-validation inputs without running optimization."""
        observed["model_type"] = model_type
        observed["teacher_checkpoint"] = Path(kwargs["teacher_checkpoint"])
        return {"run_dir": "unused", "seed": 731}

    monkeypatch.setattr(student_cli, "_run_one_training_seed", fake_training_loop)
    train_student(
        config_path,
        device="cpu",
        seed=731,
        teacher_checkpoint=teacher_path,
    )
    assert observed["model_type"] == "student"
    return observed["teacher_checkpoint"]


@pytest.fixture(scope="module")
def portable_teacher_source(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Provide one genuine portable teacher checkpoint for candidate fallback tests."""
    root = tmp_path_factory.mktemp("teacher_checkpoint_source") / "fixture"
    _, student_config, _ = _write_fixture(root)
    checkpoint = root / "source" / "best_teacher.pth"
    _write_portable_model_checkpoint(checkpoint, root / "skeleton.yaml")
    return student_config, checkpoint


@pytest.mark.parametrize("location", ["explicit_file", "directory"])
@pytest.mark.parametrize("primary_state", ["corrupt", "missing"])
def test_student_teacher_checkpoint_falls_back_to_readable_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    portable_teacher_source: tuple[Path, Path],
    location: str,
    primary_state: str,
):
    """Use a genuinely loadable sibling backup when the teacher primary is unusable."""
    student_config, source_checkpoint = portable_teacher_source
    run_dir = tmp_path / "teacher_run"
    run_dir.mkdir()
    primary = run_dir / "best_teacher.pth"
    backup = run_dir / "best_teacher.pth.bak"
    shutil.copyfile(source_checkpoint, backup)
    if primary_state == "corrupt":
        primary.write_bytes(b"truncated checkpoint")
    requested_path = primary if location == "explicit_file" else run_dir

    selected = _train_student_with_teacher_path(monkeypatch, student_config, requested_path)

    assert selected == backup.resolve()


@pytest.mark.parametrize("location", ["explicit_file", "directory"])
def test_student_teacher_checkpoint_reports_when_all_candidates_are_unreadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    portable_teacher_source: tuple[Path, Path],
    location: str,
):
    """Reject teacher recovery clearly when both the requested checkpoint and backup are bad."""
    student_config, _ = portable_teacher_source
    run_dir = tmp_path / "teacher_run"
    run_dir.mkdir()
    primary = run_dir / "best_teacher.pth"
    primary.write_bytes(b"corrupt primary")
    (run_dir / "best_teacher.pth.bak").write_bytes(b"corrupt backup")
    requested_path = primary if location == "explicit_file" else run_dir

    with pytest.raises(RuntimeError, match="No readable teacher checkpoint"):
        _train_student_with_teacher_path(monkeypatch, student_config, requested_path)


def test_student_teacher_checkpoint_prefers_valid_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    portable_teacher_source: tuple[Path, Path],
):
    """Keep the valid primary preferred when both teacher generations restore."""
    student_config, source_checkpoint = portable_teacher_source
    run_dir = tmp_path / "teacher_run"
    run_dir.mkdir()
    primary = run_dir / "best_teacher.pth"
    backup = run_dir / "best_teacher.pth.bak"
    shutil.copyfile(source_checkpoint, primary)
    shutil.copyfile(source_checkpoint, backup)

    selected = _train_student_with_teacher_path(monkeypatch, student_config, primary)

    assert selected == primary.resolve()


def test_student_teacher_checkpoint_skips_loadable_non_teacher_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    portable_teacher_source: tuple[Path, Path],
):
    """Continue to a teacher backup when the primary restores the wrong model type."""
    student_config, source_checkpoint = portable_teacher_source
    skeleton_path = portable_teacher_source[0].parent / "skeleton.yaml"
    run_dir = tmp_path / "teacher_run"
    run_dir.mkdir()
    primary = run_dir / "best_teacher.pth"
    backup = run_dir / "best_teacher.pth.bak"
    _write_portable_model_checkpoint(primary, skeleton_path, model_type="student")
    shutil.copyfile(source_checkpoint, backup)

    selected = _train_student_with_teacher_path(monkeypatch, student_config, primary)

    assert selected == backup.resolve()


def test_two_stage_lr_matches_reference_formula_for_all_300_epochs():
    """Verify every formal-default LR value against the original scheduler equation."""
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    scheduler = TwoStageLR(
        optimizer,
        total_epochs=300,
        warmup_ratio=0.1,
        warmup_init_lr=0.0001,
        warmup_gamma=1.0,
        decay_gamma=2.0,
    )
    observed = []
    for _ in range(300):
        observed.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    warmup_epochs = int(300 * 0.1)
    decay_epochs = 300 - warmup_epochs
    expected = []
    for epoch in range(300):
        if epoch < warmup_epochs:
            progress = epoch / max(1, warmup_epochs - 1)
            expected.append(0.0001 + progress ** 1.0 * (0.0005 - 0.0001))
        else:
            progress = (epoch - warmup_epochs) / max(1, decay_epochs)
            expected.append(0.0005 * (1.0 - progress) ** 2.0)
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-15)


def test_student_cli_defaults_to_all_seeds_and_explicit_seed_is_single(tmp_path: Path, monkeypatch, capsys):
    """Verify CLI seed fan-out and require --seed for explicit multi-seed resume paths."""
    _, student_config, _ = _write_fixture(tmp_path / "fixture")
    payload = yaml.safe_load(student_config.read_text(encoding="utf-8"))
    payload["seeds"] = [11, 12, 13, 14, 15]
    student_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    calls = []

    def fake_train(config_path, *, device, resume, seed, teacher_checkpoint, output_root):
        """Record a trainer call and return a normalized minimal result."""
        calls.append(seed)
        return {
            "run_dir": f"run-{seed}",
            "best_checkpoint": f"best-{seed}.pth",
            "seed": seed,
            "test_metrics": {"accuracy": 0.5},
            "num_parameters": 10,
        }

    monkeypatch.setattr(student_cli, "train_student", fake_train)
    assert student_cli.main(["--config", str(student_config)]) == 0
    assert calls == [11, 12, 13, 14, 15]
    assert len(json.loads(capsys.readouterr().out)["runs"]) == 5

    calls.clear()
    assert student_cli.main(["--config", str(student_config), "--seed", "99"]) == 0
    assert calls == [99]
    capsys.readouterr()

    with pytest.raises(ValueError, match="requires --seed"):
        student_cli.main(["--config", str(student_config), "--resume", "some-checkpoint.pth"])


def test_checkpoint_backup_copy_failure_preserves_previous_primary(tmp_path: Path, monkeypatch):
    """Ensure an I/O failure while refreshing the backup cannot displace a good primary."""
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = TwoStageLR(optimizer, total_epochs=2)
    skeleton = load_skeleton(Path(__file__).parent.parent / "configs" / "skeletons" / "fullbody23.yaml")
    generator = torch.Generator().manual_seed(5)
    payload = {
        "schema_version": 1,
        "model_type": "student",
        "model_config": {"num_classes": 2, "in_channels": 3, "channels": [32, 64, 128], "dropout": 0.05, "fourier_degree": 2},
        "skeleton_config": skeleton_mapping(skeleton),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": 1,
        "best_epoch": 1,
        "seed": 5,
    }
    destination = tmp_path / "last_checkpoint.pth"
    atomic_save_checkpoint(payload, destination)
    before = destination.read_bytes()

    def fail_copy(*args, **kwargs):
        """Simulate an I/O failure while copying the previous checkpoint."""
        raise OSError("injected backup copy failure")

    monkeypatch.setattr(common.shutil, "copyfileobj", fail_copy)
    newer = {**payload, "epoch": 2, "best_epoch": 1}
    with pytest.raises(RuntimeError, match="prior checkpoint"):
        atomic_save_checkpoint(newer, destination)
    assert destination.read_bytes() == before


def test_teacher_student_training_resume_fallback_test_retry_and_commit_repair(tmp_path: Path, monkeypatch):
    """Exercise full CPU teacher/student training and all supported recovery windows."""
    teacher_config, student_config, label_path = _write_fixture(tmp_path / "fixture")
    monkeypatch.setattr(common, "create_progress_bar", lambda loader, description: loader)
    teacher_run = tmp_path / "fixture" / "runs" / "teacher_seed_731"
    actual_save = common.atomic_save_checkpoint
    injected_training_failure = {"done": False}

    def fail_after_first_last_save(payload, path):
        """Interrupt after persisting the first last-checkpoint save."""
        saved_path = actual_save(payload, path)
        if Path(path).name == "last_checkpoint.pth" and payload["epoch"] == 1 and not injected_training_failure["done"]:
            injected_training_failure["done"] = True
            raise RuntimeError("injected power loss after first last checkpoint")
        return saved_path

    monkeypatch.setattr(common, "atomic_save_checkpoint", fail_after_first_last_save)
    with pytest.raises(RuntimeError, match="injected power loss"):
        train_teacher(teacher_config, device="cpu", seed=731)
    assert (teacher_run / "last_checkpoint.pth").is_file()
    monkeypatch.setattr(common, "atomic_save_checkpoint", actual_save)

    actual_collect = common.collect_predictions
    test_attempt_failure = {"calls": 0}

    def fail_first_teacher_test(model, loader, **kwargs):
        """Interrupt the second teacher test-evaluation attempt."""
        test_attempt_failure["calls"] += 1
        if test_attempt_failure["calls"] == 2:
            raise RuntimeError("injected interrupted test evaluation")
        return actual_collect(model, loader, **kwargs)

    monkeypatch.setattr(common, "collect_predictions", fail_first_teacher_test)
    with pytest.raises(RuntimeError, match="interrupted test"):
        train_teacher(teacher_config, device="cpu", resume="auto", seed=731)
    marker_path = teacher_run / "test_evaluation_started.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["status"] == "started"
    assert marker["attempts"] == 1
    assert not (teacher_run / "test_results.json").exists()
    monkeypatch.setattr(common, "collect_predictions", actual_collect)

    real_atomic_write = common.atomic_write_json
    manifest_failure = {"done": False}

    def fail_after_committed_test_result(path, payload):
        """Fail manifest persistence after the test result is committed."""
        destination = Path(path)
        if (
            destination.name == "run_manifest.json"
            and (destination.parent / "test_results.json").exists()
            and not manifest_failure["done"]
        ):
            manifest_failure["done"] = True
            raise OSError("injected failure after durable test result")
        return real_atomic_write(path, payload)

    monkeypatch.setattr(common, "atomic_write_json", fail_after_committed_test_result)
    with pytest.raises(OSError, match="durable test result"):
        train_teacher(teacher_config, device="cpu", resume="auto", seed=731)
    committed = json.loads((teacher_run / "test_results.json").read_text(encoding="utf-8"))
    assert committed["test_count"] == 1
    assert committed["test_attempts"] == 2
    assert json.loads((teacher_run / "run_status.json").read_text(encoding="utf-8"))["status"] == "failed"
    monkeypatch.setattr(common, "atomic_write_json", real_atomic_write)
    monkeypatch.setattr(
        common,
        "collect_predictions",
        lambda *args, **kwargs: pytest.fail("a committed test result must not be reevaluated"),
    )
    repaired_teacher = train_teacher(teacher_config, device="cpu", resume="auto", seed=731)
    monkeypatch.setattr(common, "collect_predictions", actual_collect)
    assert Path(repaired_teacher["best_checkpoint"]).is_file()
    assert json.loads((teacher_run / "run_status.json").read_text(encoding="utf-8"))["status"] == "completed"
    assert json.loads((teacher_run / "test_evaluation_started.json").read_text(encoding="utf-8"))["attempts"] == 2
    history_rows = list(csv.DictReader((teacher_run / "history.csv").open("r", encoding="utf-8")))
    assert [int(row["epoch"]) for row in history_rows] == [1, 2]
    manifest = json.loads((teacher_run / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["software"]["torch"] == str(torch.__version__)
    assert manifest["splits"]["train"]["joints_shape"] == [4, 3, 32, 5]
    assert manifest["model"]["num_parameters"] == repaired_teacher["num_parameters"]

    restored_teacher, teacher_payload = restore_model(repaired_teacher["best_checkpoint"], device="cpu")
    assert teacher_payload["label_map"] == {"class_zero": 0, "class_one": 1}
    assert restored_teacher.num_classes == 2

    # A swapped semantic class mapping must not silently distill teacher targets.
    swapped_labels = tmp_path / "fixture" / "swapped_labels.json"
    swapped_labels.write_text(json.dumps({"class_one": 0, "class_zero": 1}), encoding="utf-8")
    student_payload = yaml.safe_load(student_config.read_text(encoding="utf-8"))
    student_payload["label_map"] = str(swapped_labels)
    swapped_student_config = tmp_path / "fixture" / "student_swapped.yaml"
    swapped_student_config.write_text(yaml.safe_dump(student_payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="label-name-to-index"):
        train_student(
            swapped_student_config,
            device="cpu",
            seed=731,
            teacher_checkpoint=repaired_teacher["best_checkpoint"],
        )

    student_run = tmp_path / "fixture" / "runs" / "student_seed_731"
    injected_student_failure = {"done": False}

    def fail_after_student_first_last_save(payload, path):
        """Interrupt after persisting the student's first last checkpoint."""
        saved_path = actual_save(payload, path)
        if Path(path).name == "last_checkpoint.pth" and payload["epoch"] == 1 and not injected_student_failure["done"]:
            injected_student_failure["done"] = True
            raise RuntimeError("injected student interruption")
        return saved_path

    monkeypatch.setattr(common, "atomic_save_checkpoint", fail_after_student_first_last_save)
    with pytest.raises(RuntimeError, match="student interruption"):
        train_student(
            student_config,
            device="cpu",
            seed=731,
            teacher_checkpoint=repaired_teacher["best_checkpoint"],
        )
    monkeypatch.setattr(common, "atomic_save_checkpoint", actual_save)

    # Corrupt only a copied primary checkpoint, then prove recovery loads the backup epoch.
    backup_run = tmp_path / "backup_recovery"
    shutil.copytree(teacher_run, backup_run)
    (backup_run / "last_checkpoint.pth").write_bytes(b"truncated ZIP archive")
    (backup_run / "best_teacher.pth").unlink(missing_ok=True)
    (backup_run / "best_teacher.pth.bak").unlink(missing_ok=True)
    with pytest.raises(RuntimeError, match="Refusing to test last weights"):
        common._resolve_best_for_test(backup_run, "teacher")
    skeleton = load_skeleton(teacher_payload["skeleton_config"] if isinstance(teacher_payload["skeleton_config"], str) else teacher_config.parent / "skeleton.yaml")
    training_model = build_model("teacher", skeleton, teacher_payload["model_config"])
    resume_optimizer = _make_optimizer(training_model, {"peak_lr": 0.0005, "weight_decay": 0.0001})
    resume_scheduler = TwoStageLR(resume_optimizer, total_epochs=2)
    resume_scaler = torch.cuda.amp.GradScaler(enabled=False)
    generator = torch.Generator().manual_seed(731)
    resolved_path, recovered_payload = load_training_checkpoint(
        backup_run,
        "teacher",
        training_model,
        resume_optimizer,
        resume_scheduler,
        resume_scaler,
        seed=731,
        configured_epochs=2,
        loader_generator=generator,
    )
    assert resolved_path.name == "last_checkpoint.pth.bak"
    assert recovered_payload["epoch"] == 1

    student_result = train_student(
        student_config,
        device="cpu",
        resume="auto",
        seed=731,
        teacher_checkpoint=repaired_teacher["best_checkpoint"],
    )
    assert student_result["num_parameters"] > 0
    assert student_result["test_metrics"]["macro_f1"] >= 0.0
    _, student_checkpoint = restore_model(student_result["best_checkpoint"], device="cpu")
    assert student_checkpoint["model_type"] == "student"
    assert student_checkpoint["label_map"] == teacher_payload["label_map"]
    assert student_run.is_dir()
