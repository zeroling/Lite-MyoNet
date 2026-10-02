"""Run one teacher and five seeded students with explicit resume state."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import load_yaml, resolve_config_path, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_all.yaml"
STATE_FILENAME = "state.json"


def _positive_seed(value: Any, name: str) -> int:
    """Validate a seed represented as a positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _load_pipeline_config(config_path: str | Path) -> tuple[dict[str, Any], Path, Path, Path]:
    """Load and validate the public pipeline configuration and its paths."""
    config = load_yaml(config_path)
    allowed = {
        "schema_version", "teacher_config", "student_config", "output_root",
        "teacher_seed", "student_seeds",
    }
    actual = set(config) - {"_config_path"}
    if actual != allowed:
        missing = sorted(allowed - actual)
        unknown = sorted(actual - allowed)
        raise ValueError(f"Pipeline fields mismatch: missing={missing}, unknown={unknown}.")
    if (
        isinstance(config["schema_version"], bool)
        or not isinstance(config["schema_version"], int)
        or config["schema_version"] != 1
    ):
        raise ValueError("Pipeline schema_version must be 1.")
    for field in ("teacher_config", "student_config", "output_root"):
        if not isinstance(config[field], str) or not config[field].strip():
            raise ValueError(f"{field} must be a non-empty path string.")

    teacher_seed = _positive_seed(config["teacher_seed"], "teacher_seed")
    student_seeds = config["student_seeds"]
    if not isinstance(student_seeds, list) or len(student_seeds) != 5:
        raise ValueError("student_seeds must contain exactly five seeds.")
    parsed_seeds = [_positive_seed(seed, f"student_seeds[{index}]")
                    for index, seed in enumerate(student_seeds)]
    if len(set(parsed_seeds)) != 5:
        raise ValueError("student_seeds must be unique.")

    teacher_config = resolve_config_path(config, config["teacher_config"])
    student_config = resolve_config_path(config, config["student_config"])
    output_root = resolve_config_path(config, config["output_root"])
    if not teacher_config.is_file():
        raise FileNotFoundError(f"Teacher configuration does not exist: {teacher_config}")
    if not student_config.is_file():
        raise FileNotFoundError(f"Student configuration does not exist: {student_config}")
    config["teacher_seed"] = teacher_seed
    config["student_seeds"] = parsed_seeds
    return config, teacher_config, student_config, output_root


def _new_state(config: Mapping[str, Any], teacher_config: Path, student_config: Path,
               output_root: Path) -> dict[str, Any]:
    """Create a serializable state document before any training starts."""
    teacher_seed = int(config["teacher_seed"])
    student_seeds = [int(seed) for seed in config["student_seeds"]]
    return {
        "schema_version": 1,
        "status": "running",
        "teacher_config": str(teacher_config),
        "student_config": str(student_config),
        "output_root": str(output_root),
        "teacher_seed": teacher_seed,
        "student_seeds": student_seeds,
        "teacher": {
            "status": "pending", "seed": teacher_seed,
            "run_dir": str(output_root / f"teacher_seed_{teacher_seed}"),
        },
        "students": {
            str(seed): {
                "status": "pending", "seed": seed,
                "run_dir": str(output_root / f"student_seed_{seed}"),
            }
            for seed in student_seeds
        },
        "summary": None,
    }


def _read_state(state_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Read a prior state and ensure its fixed stage identities match this plan."""
    try:
        with state_path.open("r", encoding="utf-8") as stream:
            state = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to read pipeline state {state_path}: {exc}") from exc
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise ValueError(f"Unsupported pipeline state schema in {state_path}.")
    if state.get("teacher_seed") != config["teacher_seed"]:
        raise ValueError("Saved teacher seed differs from the current pipeline configuration.")
    if state.get("student_seeds") != config["student_seeds"]:
        raise ValueError("Saved student seeds differ from the current pipeline configuration.")
    if not isinstance(state.get("teacher"), dict) or not isinstance(state.get("students"), dict):
        raise ValueError("Pipeline state is missing teacher or student stage records.")
    for seed in config["student_seeds"]:
        if str(seed) not in state["students"]:
            raise ValueError(f"Pipeline state is missing student seed {seed}.")
    return state


def _normalize_result(result: Mapping[str, Any], expected_seed: int) -> dict[str, Any]:
    """Validate and normalize the stable public result returned by a trainer."""
    required = {"run_dir", "best_checkpoint", "seed", "test_metrics", "num_parameters"}
    if not isinstance(result, Mapping):
        raise TypeError("Training functions must return a mapping.")
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f"Training result is missing required fields: {missing}.")
    seed = _positive_seed(result["seed"], "result.seed")
    if seed != expected_seed:
        raise ValueError(f"Training returned seed {seed}; expected {expected_seed}.")
    run_dir = Path(result["run_dir"]).expanduser().resolve()
    best_checkpoint = Path(result["best_checkpoint"]).expanduser().resolve()
    metrics = result["test_metrics"]
    if not isinstance(metrics, Mapping) or "accuracy" not in metrics:
        raise ValueError("Training result test_metrics must include accuracy.")
    accuracy = metrics["accuracy"]
    if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)):
        raise ValueError("test_metrics.accuracy must be numeric.")
    if not math.isfinite(float(accuracy)) or not 0.0 <= float(accuracy) <= 1.0:
        raise ValueError("test_metrics.accuracy must be finite and in [0, 1].")
    parameters = result["num_parameters"]
    if isinstance(parameters, bool) or not isinstance(parameters, int) or parameters <= 0:
        raise ValueError("num_parameters must be a positive integer.")
    return {
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_checkpoint),
        "seed": seed,
        "test_metrics": _json_compatible(dict(metrics)),
        "num_parameters": int(parameters),
    }


def _json_compatible(value: Any) -> Any:
    """Convert common numeric container values into JSON-compatible objects."""
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return _json_compatible(value.tolist())
    if hasattr(value, "item"):
        return _json_compatible(value.item())
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable.")


def _validate_completed_stage(
    stage: Mapping[str, Any], seed: int, model_type: str
) -> dict[str, Any]:
    """Load a completed stage only when its result artifacts remain readable."""
    if stage.get("status") != "completed":
        raise ValueError("Only completed stages can be restored without training.")
    if model_type not in {"teacher", "student"}:
        raise ValueError("model_type must be 'teacher' or 'student'.")
    result = stage.get("result")
    if not isinstance(result, Mapping):
        raise RuntimeError(f"Completed seed {seed} is missing its stored result.")
    normalized = _normalize_result(result, seed)
    if not Path(normalized["best_checkpoint"]).is_file():
        raise RuntimeError(
            f"Completed seed {seed} is missing its best checkpoint; refusing to skip it."
        )
    _validate_checkpoint_readable(
        Path(normalized["best_checkpoint"]), model_type, int(seed)
    )
    _validate_test_result_file(
        Path(normalized["run_dir"]), model_type, int(seed), normalized["test_metrics"]
    )
    return normalized


def _validate_checkpoint_readable(
    checkpoint_path: Path, model_type: str, seed: int
) -> None:
    """Load a local checkpoint and check only its minimal stage identity."""
    import torch

    try:
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Completed checkpoint cannot be loaded: {checkpoint_path}: {exc}") from exc
    if not isinstance(payload, dict) or not {
        "model_type", "model_config", "skeleton_config", "model_state_dict", "seed"
    }.issubset(payload):
        raise RuntimeError(f"Completed checkpoint has no readable portable model state: {checkpoint_path}")
    if payload["model_type"] != model_type:
        raise RuntimeError(
            f"Completed checkpoint model_type {payload['model_type']!r} does not match {model_type!r}."
        )
    checkpoint_seed = payload["seed"]
    if (
        isinstance(checkpoint_seed, bool)
        or not isinstance(checkpoint_seed, int)
        or checkpoint_seed != seed
    ):
        raise RuntimeError(
            f"Completed checkpoint seed {checkpoint_seed!r} does not match expected seed {seed}."
        )


def _validate_test_result_file(
    run_dir: Path,
    model_type: str,
    seed: int,
    expected_metrics: Mapping[str, Any],
) -> None:
    """Require test results to match the stage identity and persisted metrics."""
    result_path = run_dir / "test_results.json"
    if not result_path.is_file():
        raise RuntimeError(
            f"Completed seed {seed} is missing {result_path}; refusing to skip it."
        )
    try:
        with result_path.open("r", encoding="utf-8") as stream:
            test_payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Completed test results are unreadable at {result_path}: {exc}") from exc
    if not isinstance(test_payload, dict):
        raise RuntimeError(f"Completed test results are not a JSON object: {result_path}")
    required = {"model_type", "seed", "test_count", "metrics"}
    missing = sorted(required - set(test_payload))
    if missing:
        raise RuntimeError(
            f"Completed test results must include model_type, seed, test_count, and metrics; "
            f"missing={missing}: {result_path}"
        )
    if test_payload["model_type"] != model_type:
        raise RuntimeError(
            f"Completed test result model_type {test_payload['model_type']!r} "
            f"does not match {model_type!r}: {result_path}"
        )
    if (
        isinstance(test_payload["seed"], bool)
        or not isinstance(test_payload["seed"], int)
        or test_payload["seed"] != seed
    ):
        raise RuntimeError(
            f"Completed test result seed {test_payload['seed']!r} does not match "
            f"expected seed {seed}: {result_path}"
        )
    if (
        isinstance(test_payload["test_count"], bool)
        or not isinstance(test_payload["test_count"], int)
        or test_payload["test_count"] != 1
    ):
        raise RuntimeError(
            f"Completed test result must record test_count=1: {result_path}"
        )
    if not isinstance(test_payload["metrics"], dict):
        raise RuntimeError(f"Completed test result metrics must be an object: {result_path}")
    stored_metrics = _json_compatible(test_payload["metrics"])
    normalized_expected = _json_compatible(dict(expected_metrics))
    if stored_metrics != normalized_expected:
        raise RuntimeError(
            f"Completed test result metrics do not match the saved pipeline result: {result_path}"
        )


def _stage_resume_mode(stage: Mapping[str, Any]) -> str | None:
    """Return auto-resume only after a stage had begun in a prior invocation."""
    return "auto" if stage.get("status") in {"running", "failed", "interrupted"} else None


def _metric_aggregates(student_results: Sequence[Mapping[str, Any]],
                       teacher_metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Compute means and sample standard deviations for shared scalar metrics."""
    common_keys = set(teacher_metrics)
    for result in student_results:
        common_keys &= set(result["test_metrics"])
    aggregates: dict[str, Any] = {}
    for key in sorted(common_keys):
        values = [result["test_metrics"][key] for result in student_results]
        teacher_value = teacher_metrics[key]
        if (
            isinstance(teacher_value, bool)
            or not isinstance(teacher_value, (int, float))
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values)
            or not all(math.isfinite(float(value)) for value in [teacher_value, *values])
        ):
            continue
        numeric_values = [float(value) for value in values]
        metric_mean = statistics.mean(numeric_values)
        teacher_numeric = float(teacher_value)
        aggregates[key] = {
            "mean": metric_mean,
            "sample_sd": statistics.stdev(numeric_values),
            "teacher": teacher_numeric,
            "student_teacher_ratio": (
                metric_mean / teacher_numeric if teacher_numeric > 0.0 else None
            ),
        }
    teacher_accuracy = float(teacher_metrics["accuracy"])
    aggregates["student_teacher_accuracy_ratio"] = (
        statistics.mean(float(item["test_metrics"]["accuracy"]) for item in student_results)
        / teacher_accuracy
        if teacher_accuracy > 0.0 else None
    )
    return aggregates


def _write_readme(path: Path, summary: Mapping[str, Any]) -> None:
    """Write a detailed per-pipeline report beside the machine-readable summary."""
    teacher = summary["teacher"]
    students = summary["students"]
    aggregates = summary["aggregates"]
    rows = [
        "# Lite-MyoNet Teacher–Student Training Summary",
        "",
        "This report records one teacher run and the five sequential student runs.",
        "The metrics below are test-set results returned by the training entry points.",
        "",
        "## Teacher",
        "",
        f"- Seed: `{teacher['seed']}`",
        f"- Accuracy: `{teacher['test_metrics']['accuracy']:.6f}`",
        f"- Balanced accuracy: `{float(teacher['test_metrics'].get('balanced_accuracy', float('nan'))):.6f}`",
        f"- Macro-F1: `{float(teacher['test_metrics'].get('macro_f1', float('nan'))):.6f}`",
        f"- Parameters: `{teacher['num_parameters']}`",
        f"- Best checkpoint: `{teacher['best_checkpoint']}`",
        "",
        "## Students",
        "",
        "| Seed | Accuracy | Balanced accuracy | Macro-F1 | Parameters | Best checkpoint |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for item in students:
        metrics = item["test_metrics"]
        rows.append(
            "| {seed} | {accuracy:.6f} | {balanced:.6f} | {macro_f1:.6f} | {parameters} | `{checkpoint}` |".format(
                seed=item["seed"],
                accuracy=float(metrics["accuracy"]),
                balanced=float(metrics.get("balanced_accuracy", float("nan"))),
                macro_f1=float(metrics.get("macro_f1", float("nan"))),
                parameters=item["num_parameters"],
                checkpoint=item["best_checkpoint"],
            )
        )
    rows.extend(["", "## Student aggregates", ""])
    for key, value in aggregates.items():
        if isinstance(value, Mapping) and "mean" in value:
            rows.append(
                f"- `{key}`: mean `{value['mean']:.6f}`, sample SD `{value['sample_sd']:.6f}`, "
                f"teacher `{value['teacher']:.6f}`, student/teacher ratio "
                f"`{value['student_teacher_ratio']}`"
            )
        else:
            rows.append(f"- `{key}`: `{value}`")
    rows.extend([
        "",
        "Detailed per-class metrics, confusion matrices, seeds, parameter counts, and paths "
        "are available in `summary.json`.",
        "",
    ])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(rows))
    temporary.replace(path)


def _build_summary(teacher: Mapping[str, Any],
                   students: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the machine-readable run report and the required student aggregates."""
    return {
        "schema_version": 1,
        "status": "completed",
        "teacher": dict(teacher),
        "students": [dict(result) for result in students],
        "aggregates": _metric_aggregates(students, teacher["test_metrics"]),
    }


def _dry_run_plan(config: Mapping[str, Any], teacher_config: Path,
                  student_config: Path, output_root: Path,
                  *, device: str, resume: bool) -> dict[str, Any]:
    """Describe the ordered stages without creating files or initializing training."""
    teacher_seed = int(config["teacher_seed"])
    return {
        "schema_version": 1,
        "dry_run": True,
        "device": device,
        "resume_requested": resume,
        "teacher": {
            "config": str(teacher_config),
            "seed": teacher_seed,
            "output_root": str(output_root),
            "run_dir": str(output_root / f"teacher_seed_{teacher_seed}"),
        },
        "students": [
            {
                "config": str(student_config),
                "seed": int(seed),
                "output_root": str(output_root),
                "run_dir": str(output_root / f"student_seed_{seed}"),
            }
            for seed in config["student_seeds"]
        ],
        "execution_order": ["teacher", *[f"student_seed_{seed}" for seed in config["student_seeds"]]],
    }


def run_pipeline(config_path: str | Path = DEFAULT_CONFIG, *, device: str = "auto",
                 resume: bool = False, dry_run: bool = False) -> dict[str, Any]:
    """Run the teacher first, then five students, with explicit checkpoint recovery."""
    config, teacher_config, student_config, output_root = _load_pipeline_config(config_path)
    if not isinstance(resume, bool) or not isinstance(dry_run, bool):
        raise TypeError("resume and dry_run must be booleans.")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty string.")
    if dry_run:
        return _dry_run_plan(
            config, teacher_config, student_config, output_root, device=device, resume=resume
        )

    state_path = output_root / STATE_FILENAME
    if resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"Cannot resume without pipeline state: {state_path}")
        state = _read_state(state_path, config)
    else:
        if state_path.exists():
            raise FileExistsError(f"Pipeline state already exists; use --resume: {state_path}")
        if output_root.exists() and any(output_root.iterdir()):
            raise FileExistsError(
                f"Output root is not empty and has no pipeline state: {output_root}"
            )
        output_root.mkdir(parents=True, exist_ok=True)
        state = _new_state(config, teacher_config, student_config, output_root)
        write_json(state_path, state)

    from .teacher_trainer import train_teacher
    from .student_trainer import train_student

    teacher_stage = state["teacher"]
    if teacher_stage.get("status") == "completed":
        teacher_result = _validate_completed_stage(
            teacher_stage, int(config["teacher_seed"]), "teacher"
        )
    else:
        resume_mode = _stage_resume_mode(teacher_stage) if resume else None
        teacher_stage["status"] = "running"
        teacher_stage["error"] = None
        teacher_stage["run_dir"] = str(output_root / f"teacher_seed_{config['teacher_seed']}")
        write_json(state_path, state)
        try:
            raw_teacher = train_teacher(
                teacher_config,
                device=device,
                resume=resume_mode,
                seed=int(config["teacher_seed"]),
                output_root=output_root,
            )
            teacher_result = _normalize_result(raw_teacher, int(config["teacher_seed"]))
            _validate_result_files(teacher_result, "teacher")
            teacher_stage.update({"status": "completed", "result": teacher_result, "error": None})
            write_json(state_path, state)
        except Exception as exc:
            teacher_stage.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            write_json(state_path, state)
            raise

    student_results = []
    for seed in config["student_seeds"]:
        seed_key = str(seed)
        stage = state["students"][seed_key]
        if stage.get("status") == "completed":
            result = _validate_completed_stage(stage, int(seed), "student")
            student_results.append(result)
            continue

        resume_mode = _stage_resume_mode(stage) if resume else None
        stage["status"] = "running"
        stage["error"] = None
        stage["run_dir"] = str(output_root / f"student_seed_{seed}")
        write_json(state_path, state)
        try:
            raw_result = train_student(
                student_config,
                device=device,
                resume=resume_mode,
                seed=int(seed),
                teacher_checkpoint=teacher_result["best_checkpoint"],
                output_root=output_root,
            )
            result = _normalize_result(raw_result, int(seed))
            _validate_result_files(result, "student")
            stage.update({"status": "completed", "result": result, "error": None})
            student_results.append(result)
            write_json(state_path, state)
        except Exception as exc:
            stage.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            write_json(state_path, state)
            raise

    summary = _build_summary(teacher_result, student_results)
    write_json(output_root / "summary.json", summary)
    _write_readme(output_root / "README.md", summary)
    state.update({"status": "completed", "summary": str(output_root / "summary.json")})
    write_json(state_path, state)
    return summary


def _validate_result_files(result: Mapping[str, Any], model_type: str) -> None:
    """Require readable completion artifacts before recording a stage as complete."""
    if not Path(result["best_checkpoint"]).is_file():
        raise FileNotFoundError(f"Best checkpoint is missing: {result['best_checkpoint']}")
    _validate_checkpoint_readable(
        Path(result["best_checkpoint"]), model_type, int(result["seed"])
    )
    _validate_test_result_file(
        Path(result["run_dir"]), model_type, int(result["seed"]), result["test_metrics"]
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the public command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run one Lite-MyoNet teacher and five sequential student seeds."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Pipeline YAML path (default: configs/train_all.yaml).")
    parser.add_argument("--device", default="auto", help="Training device, such as auto, cpu, or cuda.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue unfinished stages from existing checkpoints.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the ordered execution plan without starting training.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the training pipeline CLI."""
    args = build_parser().parse_args(argv)
    result = run_pipeline(args.config, device=args.device, resume=args.resume, dry_run=args.dry_run)
    print(json.dumps(_json_compatible(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
