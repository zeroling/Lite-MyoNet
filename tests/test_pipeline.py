"""CPU tests for ordered pipeline execution and checkpoint-aware resumption."""

from __future__ import annotations

import importlib
import json
import os
import sys
import types

import numpy as np
import pytest
import yaml

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    if torch.get_num_interop_threads() != 1:
        raise


pipeline = importlib.import_module("lite_myonet.training.pipeline")


def _write_config(tmp_path):
    """Write a minimal pipeline config with teacher and student references."""
    teacher_config = tmp_path / "teacher.yaml"
    student_config = tmp_path / "student.yaml"
    teacher_config.write_text("schema_version: 1\n", encoding="utf-8")
    student_config.write_text("schema_version: 1\n", encoding="utf-8")
    pipeline_config = tmp_path / "pipeline.yaml"
    pipeline_config.write_text(yaml.safe_dump({
        "schema_version": 1,
        "teacher_config": "teacher.yaml",
        "student_config": "student.yaml",
        "output_root": "runs/utd_mhad",
        "teacher_seed": 2026,
        "student_seeds": [2026, 2027, 2028, 2029, 2030],
    }, sort_keys=False), encoding="utf-8")
    return pipeline_config


def _install_fake_trainers(monkeypatch, calls, *, fail_once_seed=None):
    """Install fake trainers that record calls and can simulate one failure."""
    teacher_module = types.ModuleType("lite_myonet.training.teacher_trainer")
    student_module = types.ModuleType("lite_myonet.training.student_trainer")
    failed_seeds = set()

    def train_teacher(config_path, *, device, resume, seed, output_root):
        """Record a teacher call and create a completed fake stage."""
        calls.append(("teacher", seed, resume, str(config_path)))
        return _complete_stage(output_root, "teacher", seed, accuracy=0.5)

    def train_student(config_path, *, device, resume, seed, teacher_checkpoint, output_root):
        """Record a student call and optionally simulate one interrupted run."""
        calls.append(("student", seed, resume, str(config_path)))
        if fail_once_seed == seed and seed not in failed_seeds:
            failed_seeds.add(seed)
            run_dir = output_root / f"student_seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "last.pth").write_bytes(b"partial checkpoint")
            raise RuntimeError("simulated interruption")
        return _complete_stage(
            output_root, "student", seed, accuracy=0.6 + 0.1 * (seed - 2026)
        )

    teacher_module.train_teacher = train_teacher
    student_module.train_student = train_student
    monkeypatch.setitem(sys.modules, "lite_myonet.training.teacher_trainer", teacher_module)
    monkeypatch.setitem(sys.modules, "lite_myonet.training.student_trainer", student_module)


def _complete_stage(output_root, kind, seed, accuracy):
    """Create checkpoint and result files for a completed pipeline stage."""
    run_dir = output_root / f"{kind}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best.pth"
    torch.save({
        "model_type": kind,
        "model_config": {},
        "skeleton_config": {},
        "model_state_dict": {},
        "seed": seed,
    }, checkpoint)
    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": accuracy - 0.02,
        "macro_f1": accuracy - 0.03,
        "per_class_recall": [accuracy, accuracy - 0.1],
    }
    (run_dir / "test_results.json").write_text(
        json.dumps({
            "schema_version": 1,
            "model_type": kind,
            "seed": seed,
            "test_count": 1,
            "metrics": metrics,
        }), encoding="utf-8"
    )
    return {
        "run_dir": run_dir,
        "best_checkpoint": checkpoint,
        "seed": seed,
        "test_metrics": metrics,
        "num_parameters": 12345,
    }


def _completed_stage(result):
    """Wrap a normalized trainer result as a completed pipeline stage."""
    return {
        "status": "completed",
        "result": {
            **result,
            "run_dir": str(result["run_dir"]),
            "best_checkpoint": str(result["best_checkpoint"]),
        },
    }


def test_pipeline_runs_teacher_then_five_students_and_writes_summary(tmp_path, monkeypatch):
    """Run fake trainers in order and verify the persisted pipeline summary."""
    config = _write_config(tmp_path)
    calls = []
    _install_fake_trainers(monkeypatch, calls)

    summary = pipeline.run_pipeline(config, device="cpu")

    assert calls[0][:3] == ("teacher", 2026, None)
    assert [call[1] for call in calls[1:]] == [2026, 2027, 2028, 2029, 2030]
    assert all(call[2] is None for call in calls)
    assert [item["seed"] for item in summary["students"]] == [2026, 2027, 2028, 2029, 2030]
    assert summary["aggregates"]["accuracy"]["mean"] == pytest.approx(0.8)
    assert summary["aggregates"]["accuracy"]["sample_sd"] == pytest.approx(0.158113883)
    assert summary["aggregates"]["accuracy"]["student_teacher_ratio"] == pytest.approx(1.6)
    assert summary["aggregates"]["student_teacher_accuracy_ratio"] == pytest.approx(1.6)
    output_root = tmp_path / "runs" / "utd_mhad"
    assert json.loads((output_root / "state.json").read_text(encoding="utf-8"))["status"] == "completed"
    assert json.loads((output_root / "summary.json").read_text(encoding="utf-8"))["status"] == "completed"
    readme = (output_root / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Lite-MyoNet Teacher–Student Training Summary")
    assert "Student aggregates" in readme


def test_resume_skips_completed_stages_and_auto_resumes_interrupted_seed(tmp_path, monkeypatch):
    """Resume an interrupted stage while skipping stages already completed."""
    config = _write_config(tmp_path)
    calls = []
    _install_fake_trainers(monkeypatch, calls, fail_once_seed=2028)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        pipeline.run_pipeline(config, device="cpu")

    calls.clear()
    summary = pipeline.run_pipeline(config, device="cpu", resume=True)

    assert all(call[0] != "teacher" for call in calls)
    student_calls = [call for call in calls if call[0] == "student"]
    assert [call[1] for call in student_calls] == [2028, 2029, 2030]
    assert [call[2] for call in student_calls] == ["auto", None, None]
    assert len(summary["students"]) == 5


def test_dry_run_does_not_create_output_or_import_trainers(tmp_path, monkeypatch):
    """Verify dry-run planning creates no output and invokes no trainer."""
    config = _write_config(tmp_path)
    calls = []
    _install_fake_trainers(monkeypatch, calls)

    plan = pipeline.run_pipeline(config, dry_run=True)

    assert plan["dry_run"] is True
    assert plan["execution_order"] == [
        "teacher", "student_seed_2026", "student_seed_2027", "student_seed_2028",
        "student_seed_2029", "student_seed_2030",
    ]
    assert not (tmp_path / "runs" / "utd_mhad").exists()
    assert calls == []


def test_existing_output_without_resume_is_not_overwritten(tmp_path, monkeypatch):
    """Preserve existing output when a new run lacks resume authorization."""
    config = _write_config(tmp_path)
    output_root = tmp_path / "runs" / "utd_mhad"
    output_root.mkdir(parents=True)
    (output_root / "untracked.txt").write_text("preserve", encoding="utf-8")
    calls = []
    _install_fake_trainers(monkeypatch, calls)

    with pytest.raises(FileExistsError, match="not empty"):
        pipeline.run_pipeline(config)
    assert (output_root / "untracked.txt").read_text(encoding="utf-8") == "preserve"
    assert calls == []


def test_resume_requires_a_pipeline_state_and_does_not_start_training(tmp_path, monkeypatch):
    """Require saved pipeline state before resuming any training stage."""
    config = _write_config(tmp_path)
    calls = []
    _install_fake_trainers(monkeypatch, calls)

    with pytest.raises(FileNotFoundError, match="without pipeline state"):
        pipeline.run_pipeline(config, resume=True)
    assert calls == []
    assert not (tmp_path / "runs" / "utd_mhad").exists()


def test_completed_stage_missing_checkpoint_fails_instead_of_silently_skipping(tmp_path):
    """Reject a completed stage whose best checkpoint is missing."""
    result = _complete_stage(tmp_path, "student", 2026, 0.7)
    result["best_checkpoint"].unlink()
    stage = _completed_stage(result)

    with pytest.raises(RuntimeError, match="missing its best checkpoint"):
        pipeline._validate_completed_stage(stage, 2026, "student")


def test_completed_stage_with_unreadable_checkpoint_fails_without_hash_validation(tmp_path):
    """Reject an unreadable checkpoint without relying on hash validation."""
    result = _complete_stage(tmp_path, "student", 2026, 0.7)
    result["best_checkpoint"].write_bytes(b"not a torch checkpoint")
    stage = _completed_stage(result)

    with pytest.raises(RuntimeError, match="cannot be loaded"):
        pipeline._validate_completed_stage(stage, 2026, "student")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_type", "teacher", "model_type"),
        ("seed", 2027, "seed"),
    ],
)
def test_completed_stage_rejects_wrong_checkpoint_identity(tmp_path, field, value, message):
    """Reject completed checkpoints with a mismatched model type or seed."""
    result = _complete_stage(tmp_path, "student", 2026, 0.7)
    payload = torch.load(result["best_checkpoint"], map_location="cpu", weights_only=False)
    payload[field] = value
    torch.save(payload, result["best_checkpoint"])

    with pytest.raises(RuntimeError, match=message):
        pipeline._validate_completed_stage(_completed_stage(result), 2026, "student")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("empty", "model_type, seed, test_count, and metrics"),
        ("model_type", "model_type"),
        ("seed", "seed"),
        ("test_count", "test_count"),
        ("metrics", "do not match the saved pipeline result"),
    ],
)
def test_completed_stage_rejects_invalid_test_result_json(tmp_path, mutation, message):
    """Reject test-result JSON with missing or inconsistent stage data."""
    result = _complete_stage(tmp_path, "student", 2026, 0.7)
    result_path = result["run_dir"] / "test_results.json"
    if mutation == "empty":
        result_path.write_text("{}", encoding="utf-8")
    else:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if mutation == "model_type":
            payload["model_type"] = "teacher"
        elif mutation == "seed":
            payload["seed"] = 2027
        elif mutation == "test_count":
            payload["test_count"] = 0
        else:
            payload["metrics"]["accuracy"] = 0.99
        result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        pipeline._validate_completed_stage(_completed_stage(result), 2026, "student")


def _write_tiny_training_configs(tmp_path):
    """Create a short-frame five-node CPU fixture for all six public training runs."""
    skeleton = {
        "schema_version": 1,
        "skeleton_id": "pipeline_cpu_fixture_v1",
        "point_count": 5,
        "root_index": 0,
        "nodes": [{"id": f"joint_{index}", "index": index} for index in range(5)],
        "model_graph_edges": [[0, 1], [0, 2], [0, 3], [0, 4]],
        "bone_vector_edges": [[1, 0], [2, 0], [3, 0], [4, 0]],
        "symmetry_pairs": [[1, 2], [3, 4]],
        "channels": ["x", "y", "z"],
        "frame_count": 32,
    }
    (tmp_path / "skeleton.yaml").write_text(
        yaml.safe_dump(skeleton, sort_keys=False), encoding="utf-8"
    )
    (tmp_path / "labels.json").write_text(
        json.dumps({"class_a": 0, "class_b": 1}), encoding="utf-8"
    )
    generator = np.random.default_rng(2026)
    joints = generator.normal(size=(2, 3, 32, 5)).astype(np.float32)
    labels = np.asarray([0, 1], dtype=np.int64)
    for split in ("train", "val", "test"):
        np.savez(tmp_path / f"{split}.npz", joints=joints, labels=labels)

    shared = {
        "schema_version": 1,
        "skeleton_config": "skeleton.yaml",
        "label_map": "labels.json",
        "data": {split: f"{split}.npz" for split in ("train", "val", "test")},
        "output_root": "runs",
        "epochs": 1,
        "batch_size": 2,
        "num_workers": 0,
        "cpu_threads": 1,
        "precision": "fp32",
        "deterministic": True,
        "selection_metric": "macro_f1",
        "loss": {"label_smoothing": 0.0},
        "optimizer": {
            "type": "AdamW", "peak_lr": 1.0e-4, "weight_decay": 0.0,
            "betas": [0.9, 0.999], "eps": 1.0e-8,
        },
        "scheduler": {
            "warmup_ratio": 0.0, "warmup_init_lr": 1.0e-4,
            "warmup_gamma": 1.0, "decay_gamma": 1.0,
        },
    }
    teacher_config = dict(shared)
    teacher_config.update({
        "seeds": [2026],
        "model": {
            "dropout": 0.0,
            "spatial_attention_config": {"num_heads": 8, "cdiv_win": 5},
            "symmetry_attention_config": {"num_heads": 8, "use_contra": True},
            "scale_attention_config": {"num_heads": 8},
            "head_config": {
                "dropout": 0.0, "norm": "layernorm", "scale_init": 10.0,
                "scale_min": 1.0, "scale_max": 20.0,
                "num_mc_samples": 0, "mc_dropout_eval": False,
            },
        },
    })
    student_config = dict(shared)
    student_config.update({
        "seeds": [2026, 2027, 2028, 2029, 2030],
        "model": {"channels": [32, 64, 128], "dropout": 0.0, "fourier_degree": 2},
        "teacher": {"checkpoint": "runs/teacher_seed_2026/best_teacher.pth"},
        "distillation": {
            "kd_temperature": 4.0,
            "topology_temperature": 0.2,
            "kd_weight": 1.0,
            "static_adjacency_weight": 1.0,
            "dynamic_similarity_weight": 1.0,
        },
    })
    (tmp_path / "teacher.yaml").write_text(
        yaml.safe_dump(teacher_config, sort_keys=False), encoding="utf-8"
    )
    (tmp_path / "student.yaml").write_text(
        yaml.safe_dump(student_config, sort_keys=False), encoding="utf-8"
    )
    pipeline_config = {
        "schema_version": 1,
        "teacher_config": "teacher.yaml",
        "student_config": "student.yaml",
        "output_root": "runs",
        "teacher_seed": 2026,
        "student_seeds": [2026, 2027, 2028, 2029, 2030],
    }
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(
        yaml.safe_dump(pipeline_config, sort_keys=False), encoding="utf-8"
    )
    return pipeline_path


def test_pipeline_real_cpu_teacher_five_students_and_resume_skip(tmp_path, monkeypatch):
    """Run the CPU teacher-student pipeline and verify resume skips completed stages."""
    from lite_myonet.training import engine as common
    from lite_myonet.training import student_trainer as student
    from lite_myonet.training import teacher_trainer as teacher

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        assert torch.get_num_interop_threads() == 1

    class SilentProgress:
        def __init__(self, iterable):
            """Store the iterable wrapped by the silent progress bar."""
            self.iterable = iterable

        def __iter__(self):
            """Yield wrapped items without displaying progress."""
            return iter(self.iterable)

        def set_postfix(self, *args, **kwargs):
            """Accept progress metadata without producing output."""
            return None

    monkeypatch.setattr(common, "create_progress_bar", lambda loader, description: SilentProgress(loader))
    config_path = _write_tiny_training_configs(tmp_path)

    first = pipeline.run_pipeline(config_path, device="cpu")

    assert [item["seed"] for item in first["students"]] == [2026, 2027, 2028, 2029, 2030]
    assert first["teacher"]["num_parameters"] > 0
    assert all(item["num_parameters"] > 0 for item in first["students"])
    run_dirs = [tmp_path / "runs" / "teacher_seed_2026"] + [
        tmp_path / "runs" / f"student_seed_{seed}" for seed in range(2026, 2031)
    ]
    for run_dir in run_dirs:
        status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
        test_result = json.loads((run_dir / "test_results.json").read_text(encoding="utf-8"))
        assert status["status"] == "completed"
        assert status["test_count"] == 1
        assert test_result["test_count"] == 1

    def unexpected_training_call(*args, **kwargs):
        """Fail if resume attempts to retrain a completed stage."""
        raise AssertionError("A completed stage was retrained during --resume.")

    monkeypatch.setattr(teacher, "train_teacher", unexpected_training_call)
    monkeypatch.setattr(student, "train_student", unexpected_training_call)
    resumed = pipeline.run_pipeline(config_path, device="cpu", resume=True)
    assert resumed == first
    for run_dir in run_dirs:
        test_result = json.loads((run_dir / "test_results.json").read_text(encoding="utf-8"))
        assert test_result["test_count"] == 1
