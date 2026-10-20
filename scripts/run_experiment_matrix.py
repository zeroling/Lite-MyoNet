"""串行运行 Lite-MyoNet 实验矩阵，并以状态文件支持安全续跑。"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = PROJECT_ROOT / "configs" / "experiment_matrix.yaml"
STUDENT_SEEDS = (11, 21, 31, 41, 51)
STUDENT_CHECKPOINT_PLACEHOLDER = "__INJECT_PRIMARY_TEACHER_CHECKPOINT__"
ZERO_SHA256 = "0" * 64
_CONSOLE_LOCK = threading.RLock()
tqdm.set_lock(_CONSOLE_LOCK)
_PROGRESS_PREFIX = "__LITE_MYONET_PROGRESS_V1__ "
_ACTIVE_PROGRESS = {}


def _resume_readable_only():
    """Check the inherited resume policy without importing the training runtime."""
    return True


class MatrixProgress:
    """Render one child's events in its fixed slot under a shared terminal lock."""

    def __init__(self, position, prefix=None):
        self.position = max(0, int(position))
        self.prefix = prefix
        self.bar = None

    def consume(self, line):
        """Consume structured batch events without adding them to epoch logs."""
        if not line.startswith(_PROGRESS_PREFIX):
            return False
        payload = json.loads(line[len(_PROGRESS_PREFIX):])
        if payload.get("event") not in {"start", "update", "close"}:
            raise ValueError("Unknown batch progress event.")
        count = int(payload["n"])
        total = payload.get("total")
        if count < 0 or (total is not None and count > int(total)):
            raise ValueError("Invalid batch progress count.")
        with _CONSOLE_LOCK:
            if payload["event"] == "start":
                self.close()
            if self.bar is None:
                description = str(payload.get("description", self.prefix or "训练"))
                self.bar = tqdm(
                    total=total, desc=description, position=self.position,
                    leave=False, dynamic_ncols=True, colour="red", mininterval=0.2,
                    file=sys.stderr, disable=False,
                )
                _ACTIVE_PROGRESS[id(self)] = self
            if count < self.bar.n:
                raise ValueError("Batch progress cannot move backwards within an epoch.")
            postfix = payload.get("postfix", {})
            self.bar.set_postfix_str(
                ", ".join(f"{key}={value}" for key, value in postfix.items()), refresh=False,
            )
            self.bar.update(count - self.bar.n)
            if payload["event"] == "close":
                self.close()
        return True

    def close(self):
        """Release the fixed slot, including on a failed or interrupted child."""
        with _CONSOLE_LOCK:
            if self.bar is not None:
                self.bar.close()
                self.bar = None
            _ACTIVE_PROGRESS.pop(id(self), None)


def _console_line(line):
    """Print summaries above active bars instead of overwriting their cursors."""
    with _CONSOLE_LOCK:
        if _ACTIVE_PROGRESS:
            tqdm.write(line.rstrip("\r\n"), file=sys.stderr)
        else:
            print(line, end="", flush=True)


class ProcessRegistry:
    """跟踪并停止本矩阵启动的训练子进程。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._processes = set()
        self._stopping = False

    def register(self, process: subprocess.Popen) -> None:
        with self._lock:
            should_stop = self._stopping
            if not should_stop:
                self._processes.add(process)
        if should_stop and process.poll() is None:
            process.terminate()

    def discard(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._processes.discard(process)

    def stop_all(self, timeout: float = 15.0) -> None:
        with self._lock:
            self._stopping = True
            processes = list(self._processes)
        for process in processes:
            if process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    raise TypeError("无法序列化状态字段 {!r}。".format(type(value).__name__))


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=_json_default)
        stream.write("\n")
    os.replace(temporary, path)


def load_yaml(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("YAML 顶层必须是映射：{}".format(path))
    return value


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """递归合并配置，override 只覆盖明确写出的字段。"""
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_from_project(value: str, base_dir: Path = PROJECT_ROOT) -> Path:
    candidate = Path(os.path.expandvars(str(value))).expanduser()
    return (candidate if candidate.is_absolute() else base_dir / candidate).resolve()


def load_plan(path: Path) -> Tuple[Dict[str, Any], Path]:
    path = Path(path).expanduser().resolve()
    plan = load_yaml(path)
    required = {
        "schema_version", "state_path", "resolved_config_dir", "output_root",
        "device", "archive_env", "archive_path", "archive_sha256", "teacher_seed",
        "student_seeds", "cpu_threads_per_process", "base_configs", "items",
    }
    missing = sorted(required - set(plan))
    if missing:
        raise ValueError("矩阵清单缺少字段：{}。".format(missing))
    if int(plan["schema_version"]) != 1:
        raise ValueError("不支持的实验矩阵 schema_version。")
    if not isinstance(plan["base_configs"], Mapping):
        raise ValueError("base_configs 必须是映射。")
    for key in ("teacher", "student"):
        if not plan["base_configs"].get(key):
            raise ValueError("base_configs.{} 必须指定。".format(key))
    if not isinstance(plan["items"], list) or not plan["items"]:
        raise ValueError("items 必须是非空列表。")
    ids = [str(item.get("id", "")) for item in plan["items"]]
    if any(not item_id for item_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("矩阵 item id 必须非空且唯一。")
    if [int(seed) for seed in plan["student_seeds"]] != list(STUDENT_SEEDS):
        raise ValueError("正式学生主实验和消融必须使用预定五种子 {}。".format(STUDENT_SEEDS))
    student_jobs = int(plan.get("student_jobs", 5))
    if student_jobs < 1 or student_jobs > len(plan["student_seeds"]):
        raise ValueError("student_jobs 必须位于 1 到 {} 之间。".format(len(plan["student_seeds"])))
    plan["student_jobs"] = student_jobs
    cpu_threads = plan.get("cpu_threads_per_process")
    if isinstance(cpu_threads, bool) or int(cpu_threads) <= 0:
        raise ValueError("cpu_threads_per_process 必须为正整数。")
    plan["cpu_threads_per_process"] = int(cpu_threads)
    if not _resume_readable_only():
        if len(str(plan["archive_sha256"])) != 64:
            raise ValueError("archive_sha256 必须是 64 位 SHA-256。")
        try:
            int(str(plan["archive_sha256"]), 16)
        except ValueError as error:
            raise ValueError("archive_sha256 不是十六进制 SHA-256。") from error

    for item in plan["items"]:
        if item.get("kind") not in {"teacher", "student", "reference"}:
            raise ValueError("{} 的 kind 无效。".format(item.get("id")))
        if item["kind"] == "teacher" and "seed" not in item:
            raise ValueError("教师项目 {} 缺少单种子。".format(item.get("id")))
        if item["kind"] in {"student", "reference"} and not item.get("override") and item.get("id") != "student_main":
            raise ValueError("学生矩阵项目 {} 缺少配置覆盖文件。".format(item.get("id")))
        if item["kind"] == "reference" and not item.get("reference_to"):
            raise ValueError("共享对照项目 {} 缺少 reference_to。".format(item.get("id")))
        if item["kind"] in {"teacher", "student", "reference"} and not item.get("role") and item["kind"] != "reference" and item["id"] != "student_main":
            if not item.get("evidence_role"):
                raise ValueError("项目 {} 未声明结果用途。".format(item.get("id")))

    primary_teacher_indices = [
        index
        for index, item in enumerate(plan["items"])
        if item["kind"] == "teacher" and item.get("role") == "primary_teacher"
    ]
    if len(primary_teacher_indices) != 1:
        raise ValueError("矩阵必须且只能包含一个 primary_teacher 主教师项目。")
    if primary_teacher_indices[0] != 0:
        raise ValueError("完整主教师必须是矩阵的第一个项目。")

    student_stage_indices = [
        index
        for index, item in enumerate(plan["items"])
        if item["kind"] in {"student", "reference"}
    ]
    teacher_ablation_indices = [
        index
        for index, item in enumerate(plan["items"])
        if item["kind"] == "teacher" and item.get("role") != "primary_teacher"
    ]
    if not student_stage_indices:
        raise ValueError("矩阵必须包含学生实验。")
    if teacher_ablation_indices and max(student_stage_indices) > min(teacher_ablation_indices):
        raise ValueError("教师结构消融必须位于所有学生实验之后。")

    _validate_plan_references(plan, path)

    return plan, path


def plan_fingerprint(plan: Mapping[str, Any], plan_path: Path) -> Tuple[str, Dict[str, str]]:
    config_root = plan_path.parent
    paths = {"matrix_plan": plan_path}
    for value in plan["base_configs"].values():
        paths["base:" + str(value)] = (config_root / value).resolve()
    for item in plan["items"]:
        if item.get("override"):
            paths["override:" + item["id"]] = (config_root / item["override"]).resolve()
    source_hashes = {}
    digest = hashlib.sha256()
    for label, path in sorted(paths.items()):
        if not path.is_file():
            raise FileNotFoundError("矩阵配置文件不存在：{}".format(path))
        file_hash = sha256_file(path)
        source_hashes[label] = file_hash
        digest.update(label.encode("utf-8"))
        digest.update(bytes.fromhex(file_hash))
    return digest.hexdigest(), source_hashes


def _source_config_path(plan_path: Path, plan: Mapping[str, Any], item: Mapping[str, Any]) -> Path:
    config_root = plan_path.parent
    base_key = "teacher" if item["kind"] == "teacher" else "student"
    return (config_root / plan["base_configs"][base_key]).resolve()


def _build_config(
    plan: Mapping[str, Any], plan_path: Path, item: Mapping[str, Any],
    output_dir: Path, teacher_checkpoint: Optional[Mapping[str, str]],
    allow_unresolved_archive: bool,
    teacher_cache: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    base_path = _source_config_path(plan_path, plan, item)
    config = load_yaml(base_path)
    if item['kind'] in {'student', 'reference'} and plan.get('selected_fourier_degree') is not None:
        config['model']['fourier_degree'] = int(plan['selected_fourier_degree'])
    if item.get("override"):
        override_path = (plan_path.parent / item["override"]).resolve()
        config = deep_merge(config, load_yaml(override_path))
    if item["kind"] == "teacher" and plan.get("selected_teacher_scheme"):
        config["model"]["input_scheme"] = plan["selected_teacher_scheme"]
    if item.get("save_epoch_weights"):
        config["save_epoch_weights"] = True

    config["output_root"] = str(output_dir.resolve())
    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError("{} 缺少 data 映射。".format(item["id"]))
    expected_archive_hash = str(plan["archive_sha256"]).lower()
    if not _resume_readable_only() and str(data.get("archive_sha256", "")).lower() != expected_archive_hash:
        raise ValueError("{} 使用的数据包哈希与矩阵要求不一致。".format(item["id"]))
    label_map_value = data.get("label_map")
    if not label_map_value:
        raise ValueError("{} 缺少 data.label_map。".format(item["id"]))
    label_map_path = Path(str(label_map_value)).expanduser()
    if not label_map_path.is_absolute():
        label_map_path = (base_path.parent / label_map_path).resolve()
    data["label_map"] = str(label_map_path)

    archive_env = str(plan["archive_env"])
    archive_value = str(data.get("archive", ""))
    archive_path_text = os.path.expandvars(archive_value)
    if "${" in archive_path_text or ("$" + archive_env) in archive_path_text:
        archive_value_from_env = os.environ.get(archive_env)
        if archive_value_from_env:
            archive_path_text = str(Path(archive_value_from_env).expanduser().resolve())
        elif not allow_unresolved_archive:
            raise ValueError("请先设置环境变量 {} 指向唯一的 NPZ 数据包。".format(archive_env))
    archive_path = Path(archive_path_text).expanduser()
    if archive_path.is_absolute():
        data["archive"] = str(archive_path.resolve())
    elif "${" not in archive_path_text and not ("$" + archive_env) in archive_path_text:
        data["archive"] = str((base_path.parent / archive_path).resolve())

    if item["kind"] == "teacher":
        config["seeds"] = [int(item.get("seed", plan["teacher_seed"]))]
    elif item["kind"] in {"student", "reference"}:
        config["seeds"] = [int(seed) for seed in plan["student_seeds"]]
        teacher = config.get("teacher")
        if not isinstance(teacher, dict):
            raise ValueError("学生配置缺少 teacher 映射。")
        if teacher_checkpoint is None:
            teacher["checkpoint"] = STUDENT_CHECKPOINT_PLACEHOLDER
            teacher["sha256"] = ZERO_SHA256
        else:
            teacher["checkpoint"] = str(Path(teacher_checkpoint["path"]).resolve())
            teacher["sha256"] = str(teacher_checkpoint["sha256"]).lower()
        if item["kind"] == "student" and teacher_cache is not None:
            config["teacher_cache"] = {
                "directory": str(Path(teacher_cache["path"]).resolve()),
                "manifest_sha256": str(teacher_cache["manifest_sha256"]).lower(),
            }
    return config


def _validate_materialized_config(config: Mapping[str, Any], kind: str) -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    if kind == "teacher":
        from lite_myonet.training.teacher_trainer import validate_experiment_config

        validate_experiment_config(config)
    else:
        from lite_myonet.training.student_trainer import validate_student_config

        validate_student_config(config)


def _archive_check(plan: Mapping[str, Any], allow_unresolved: bool) -> Dict[str, Any]:
    env_name = str(plan["archive_env"])
    value = os.environ.get(env_name)
    source = "environment"
    if not value:
        value = plan.get("archive_path")
        source = "plan"
    if not value:
        if allow_unresolved:
            return {"resolved": False, "environment_variable": env_name}
        raise ValueError("矩阵既没有 {}，也没有 archive_path。".format(env_name))
    path = _resolve_from_project(str(value))
    if not path.is_file():
        raise FileNotFoundError("数据包不存在：{}".format(path))
    actual = sha256_file(path)
    expected = str(plan["archive_sha256"]).lower()
    if not _resume_readable_only() and actual.lower() != expected:
        raise ValueError("数据包哈希不匹配：清单要求 {}，实际 {}。".format(expected, actual))
    os.environ[env_name] = str(path)
    return {"resolved": True, "path": str(path), "sha256": actual, "source": source}


def _restore_archive_environment_from_state(
    plan: Mapping[str, Any], state: Mapping[str, Any]
) -> Optional[str]:
    """续跑时从已校验状态恢复数据路径，避免要求用户重复设置环境变量。"""
    env_name = str(plan["archive_env"])
    if os.environ.get(env_name):
        return None
    archive = state.get("archive", {})
    saved_path = archive.get("path") if isinstance(archive, Mapping) else None
    if not saved_path:
        return None
    path = Path(saved_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("状态文件记录的数据包不存在：{}".format(path))
    os.environ[env_name] = str(path)
    return str(path)


def _select_python(plan: Mapping[str, Any]) -> str:
    configured = plan.get("python")
    if configured:
        path = Path(os.path.expandvars(str(configured))).expanduser()
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        if not path.is_file():
            raise FileNotFoundError("配置的 Python 解释器不存在：{}".format(path))
        return str(path)
    preferred = Path(sys.executable)
    if preferred.is_file():
        return str(preferred)
    return sys.executable


def _resolve_python_runtime(
    python: str, inherited_env: Optional[Mapping[str, str]] = None,
) -> Tuple[str, Dict[str, str]]:
    """绕过 Windows virtualenv 常驻启动器，同时继续使用该环境中的包。"""
    launcher = Path(python).expanduser().resolve()
    if os.name != "nt" or launcher.parent.name.lower() != "scripts":
        return str(launcher), {}
    venv_root = launcher.parent.parent
    config_path = venv_root / "pyvenv.cfg"
    site_packages = venv_root / "Lib" / "site-packages"
    if not config_path.is_file() or not site_packages.is_dir():
        return str(launcher), {}

    values = {}
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = raw_line.partition("=")
        if separator:
            values[key.strip().lower()] = value.strip()
    base_value = values.get("base-executable") or values.get("executable")
    if not base_value:
        return str(launcher), {}
    base_executable = Path(base_value).expanduser()
    if not base_executable.is_absolute():
        base_executable = (venv_root / base_executable).resolve()
    else:
        base_executable = base_executable.resolve()
    if not base_executable.is_file():
        return str(launcher), {}

    inherited = dict(inherited_env or {})
    python_paths = [str(site_packages.resolve())]
    if inherited.get("PYTHONPATH"):
        python_paths.append(str(inherited["PYTHONPATH"]))
    path_entries = [str(launcher.parent.resolve())]
    if inherited.get("PATH"):
        path_entries.append(str(inherited["PATH"]))
    return str(base_executable), {
        "VIRTUAL_ENV": str(venv_root.resolve()),
        "PYTHONPATH": os.pathsep.join(python_paths),
        "PATH": os.pathsep.join(path_entries),
    }


def _training_environment(
    plan: Mapping[str, Any], inherited_env: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """限制每个训练进程的 CPU 数学库线程，避免五种子过度订阅。"""
    cpu_threads = plan.get("cpu_threads_per_process")
    if isinstance(cpu_threads, bool) or int(cpu_threads) <= 0:
        raise ValueError("cpu_threads_per_process 必须为正整数。")
    environment = dict(inherited_env or {})
    value = str(int(cpu_threads))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TORCH_NUM_THREADS",
        "TORCH_NUM_INTEROP_THREADS",
    ):
        environment[name] = value
    return environment


def _config_paths(plan: Mapping[str, Any], state_path: Path, item_id: str) -> Tuple[Path, Path]:
    config_root = _resolve_from_project(plan["resolved_config_dir"])
    output_root = _resolve_from_project(plan["output_root"])
    return config_root / (item_id + ".yaml"), output_root / item_id


def _teacher_run_dirs(output_root: Path, seed: int) -> List[Path]:
    if not output_root.is_dir():
        return []
    return sorted(
        (path for path in output_root.glob("teacher_seed_{}_*".format(seed)) if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )


def _student_seed_dirs(output_root: Path, seed: int) -> List[Path]:
    if not output_root.is_dir():
        return []
    candidates = []
    for experiment_dir in output_root.glob("student_*"):
        run_dir = experiment_dir / ("seed_{}".format(seed))
        if run_dir.is_dir():
            candidates.append(run_dir)
    return sorted(candidates, key=lambda path: path.stat().st_mtime)


def _discover_teacher_complete(output_root: Path, seed: int) -> Optional[Tuple[Path, Path]]:
    complete = []
    for run_dir in _teacher_run_dirs(output_root, seed):
        checkpoint = run_dir / "best_teacher.pth"
        results = run_dir / "test_results.json"
        if checkpoint.is_file() and results.is_file():
            result = json.loads(results.read_text(encoding="utf-8"))
            if int(result.get("seed", -1)) == int(seed):
                complete.append((run_dir, checkpoint))
    if len(complete) > 1:
        raise RuntimeError("{} 出现多个完整教师 run，无法唯一确定 checkpoint。".format(output_root))
    return complete[0] if complete else None


def _discover_student_complete(output_root: Path, seed: int) -> Optional[Path]:
    complete = []
    for run_dir in _student_seed_dirs(output_root, seed):
        result_path = run_dir / "test_results.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if int(result.get("seed", -1)) == int(seed) and int(result.get("test_evaluation_count", 0)) == 1:
                complete.append(run_dir)
    if len(complete) > 1:
        raise RuntimeError("{} 的 seed {} 出现多个完整 run。".format(output_root, seed))
    return complete[0] if complete else None


def _find_interrupted_teacher(output_root: Path, seed: int) -> Optional[Path]:
    candidates = []
    for run_dir in _teacher_run_dirs(output_root, seed):
        if (run_dir / "test_results.json").is_file():
            continue
        if any((run_dir / name).is_file() for name in (
            "last_teacher.pth", "last_teacher.pth.bak",
        )):
            candidates.append(run_dir)
    if len(candidates) > 1:
        raise RuntimeError("检测到多个不完整教师 run，不能自动决定从哪一个续跑。")
    return candidates[0] if candidates else None


def _find_interrupted_student(output_root: Path, seed: int) -> Optional[Path]:
    candidates = []
    unresumable_test = []
    for run_dir in _student_seed_dirs(output_root, seed):
        if (run_dir / "test_results.json").is_file():
            continue
        marker_path = run_dir / "test_evaluation_started.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8")) if marker_path.is_file() else {}
        has_checkpoint = any((run_dir / name).is_file() for name in (
            "last_student.pth", "last_student.pth.bak",
            "best_student.pth", "best_student.pth.bak",
        ))
        test_already_completed = marker.get("status") == "complete" or int(marker.get("test_evaluation_count", 0)) == 1
        if test_already_completed:
            unresumable_test.append(run_dir)
        elif has_checkpoint:
            # student_trainer 记录 test_attempt_count，可从 last_student.pth 恢复未完成评估。
            candidates.append(run_dir)
        elif marker_path.is_file():
            unresumable_test.append(run_dir)
    if unresumable_test:
        raise RuntimeError(
            "seed {} 有 test 标记显示评估已结束但缺少结果，或没有 last_student.pth，"
            "不能安全恢复；目录 {}。".format(seed, unresumable_test[-1])
        )
    if len(candidates) > 1:
        raise RuntimeError("检测到多个不完整学生 run，不能自动决定从哪一个续跑。")
    return candidates[0] if candidates else None


def _command_for(item: Mapping[str, Any], config_path: Path, python: str, device: str,
                 seed: Optional[int] = None, resume: Optional[Path] = None) -> List[str]:
    script = "train_teacher.py" if item["kind"] == "teacher" else "train_student.py"
    command = [python, str(PROJECT_ROOT / "scripts" / script), "--config", str(config_path), "--device", device]
    if resume is not None:
        command.extend(["--resume", str(resume)])
    elif item["kind"] == "student":
        if seed is None:
            raise ValueError("学生命令缺少 seed。")
        command.extend(["--seeds", str(seed)])
    else:
        command.extend(["--seeds", str(int(item["seed"]))])
    return command


def build_dry_run(plan: Mapping[str, Any], plan_path: Path, device: Optional[str] = None,
                  student_jobs: Optional[int] = None, teacher_only: bool = False) -> Dict[str, Any]:
    archive = _archive_check(plan, allow_unresolved=True)
    python_launcher = _select_python(plan)
    python, runtime_updates = _resolve_python_runtime(python_launcher, os.environ)
    runtime_environment = _training_environment(
        plan, {**dict(os.environ), **runtime_updates}
    )
    chosen_device = device or str(plan.get("device", "cuda"))
    state_path = _resolve_from_project(plan["state_path"])
    commands = []
    training_command_count = 0
    seed_process_count = 0
    materialized = {}
    teacher_checkpoint = {"path": str(PROJECT_ROOT / "runs" / "DRY_RUN" / "best_teacher.pth"), "sha256": ZERO_SHA256}
    for item in plan["items"]:
        config_path, output_dir = _config_paths(plan, state_path, item["id"])
        if item["kind"] == "reference":
            source_id = str(item["reference_to"])
            if source_id not in {str(entry["id"]) for entry in plan["items"]}:
                raise ValueError("共享对照 {} 引用不存在项目 {}。".format(item["id"], source_id))
            config = _build_config(plan, plan_path, item, output_dir, teacher_checkpoint, True)
            _validate_materialized_config(config, "student")
            command_list = []
        else:
            checkpoint = None if item["kind"] == "teacher" else teacher_checkpoint
            config = _build_config(plan, plan_path, item, output_dir, checkpoint, True)
            _validate_materialized_config(config, str(item["kind"]))
            if item["kind"] == "student":
                command_list = [
                    _command_for(item, config_path, python, chosen_device, seed=int(seed))
                    for seed in plan["student_seeds"]
                ]
            else:
                command_list = [_command_for(item, config_path, python, chosen_device)]
            training_command_count += 1
            seed_process_count += len(command_list)
        materialized[item["id"]] = config
        commands.append({
            "id": item["id"], "kind": item["kind"], "commands": command_list,
            "config_path": str(config_path), "output_dir": str(output_dir),
            "reference_to": item.get("reference_to"),
            "evidence_role": item.get("evidence_role", item.get("role")),
        })

    hashes = {str(config.get("data", {}).get("archive_sha256", "")).lower() for config in materialized.values()}
    if hashes != {str(plan["archive_sha256"]).lower()}:
        raise ValueError("不同实验配置的数据包 SHA-256 不统一。")
    if materialized["student_main"]["model"]["temporal_basis"] != "fourier" or int(materialized["student_main"]["model"]["fourier_degree"]) not in (1,2,3):
        raise ValueError("student_main 必须使用候选 Fourier D=1/2/3。")
    configured_jobs = int(plan.get("student_jobs", 5))
    chosen_jobs = configured_jobs if student_jobs is None else int(student_jobs)
    if not 1 <= chosen_jobs <= len(plan["student_seeds"]):
        raise ValueError("--student-jobs 必须位于 1 到 {} 之间。".format(len(plan["student_seeds"])))
    if teacher_only:
        commands = [entry for entry in commands if entry["id"] == plan["items"][0]["id"]]
        training_command_count = 1
        seed_process_count = 1
    return {
        "dry_run": True,
        "teacher_only": teacher_only,
        "plan_path": str(plan_path),
        "archive": archive,
        "archive_sha256": str(plan["archive_sha256"]).lower(),
        "device": chosen_device,
        "python_launcher": python_launcher,
        "python": python,
        "cpu_threads_per_process": int(plan["cpu_threads_per_process"]),
        "thread_environment": {
            name: runtime_environment[name]
            for name in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "TORCH_NUM_THREADS",
                "TORCH_NUM_INTEROP_THREADS",
            )
        },
        "items": commands,
        "training_commands": training_command_count,
        "seed_processes": seed_process_count,
        "shared_reference_items": sum(entry["kind"] == "reference" for entry in commands),
        "student_seeds": [int(seed) for seed in plan["student_seeds"]],
        "student_jobs": 0 if teacher_only else chosen_jobs,
        "student_parallelism_scope": "仅同一学生配置内的不同种子；不同配置不交叉，教师始终串行",
        "teacher_seed": int(plan["teacher_seed"]),
        "teacher_attention_recomputation": bool(plan.get("teacher_attention_recomputation", True)),
    }


def _stream_process(command: List[str], log_path: Path, cwd: Path, env: Mapping[str, str],
                    registry: Optional[ProcessRegistry] = None,
                    prefix: Optional[str] = None) -> int:
    """实时转发子进程输出，并且只把每轮汇总持久化到实验日志。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = None
    progress = MatrixProgress(env.get("LITE_MYONET_PROGRESS_POSITION", "0"), prefix)
    try:
        with log_path.open("a", encoding="utf-8", errors="replace") as log_stream:
            child_env = dict(os.environ)
            child_env.update(env)
            # Windows 后台进程可能默认使用 GBK，而这里固定按 UTF-8 读取子进程输出。
            # 显式统一两端编码，避免进度条中的非 ASCII 字符变成 U+FFFD 后令主控崩溃。
            child_env["PYTHONUTF8"] = "1"
            child_env["PYTHONIOENCODING"] = "utf-8"
            # stdout 被主控管道读取时，Python 默认会使用块缓冲。强制即时输出，
            # 保证每轮汇总一产生就能显示在 PyCharm 控制台中。
            child_env["PYTHONUNBUFFERED"] = "1"
            child_env["LITE_MYONET_PROGRESS_TRANSPORT"] = "json"
            # scripts/*.py 是源码树入口，并非要求用户先执行 pip install 的已安装命令。
            # 矩阵必须把 src 显式传给所有教师、学生和缓存子进程，保证普通
            # `python scripts/run_experiment_matrix.py ...` 启动方式可直接工作。
            source_root = str((PROJECT_ROOT / "src").resolve())
            inherited_pythonpath = child_env.get("PYTHONPATH")
            child_env["PYTHONPATH"] = (
                source_root + os.pathsep + inherited_pythonpath
                if inherited_pythonpath
                else source_root
            )
            process = subprocess.Popen(
                command, cwd=str(cwd), env=child_env, stdout=subprocess.PIPE,
                # 所有子进程只上报进度，由主控统一控制光标；错误也实时转发。
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
            if registry is not None:
                registry.register(process)
            assert process.stdout is not None
            for line in process.stdout:
                if progress.consume(line):
                    continue
                _console_line("[{}] {}".format(prefix, line) if prefix else line)
                if line.startswith("轮次结果 "):
                    log_stream.write(line)
                    log_stream.flush()
            return int(process.wait())
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        progress.close()
        if process is not None and process.stdout is not None:
            process.stdout.close()
        if registry is not None and process is not None:
            registry.discard(process)


def _initial_state(
    plan: Mapping[str, Any], plan_path: Path, fingerprint: str,
    source_hashes: Mapping[str, str], archive_report: Mapping[str, Any], state_path: Path,
) -> Dict[str, Any]:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    items = []
    for item in plan["items"]:
        state_item = {
            "id": str(item["id"]), "kind": str(item["kind"]),
            "status": "pending", "source_item": copy.deepcopy(item),
            "output_dir": str(_config_paths(plan, _resolve_from_project(plan["state_path"]), item["id"])[1]),
            "started_at": None, "finished_at": None,
            "returncode": None, "error": None,
        }
        if item["kind"] == "student":
            item_output_dir = Path(state_item["output_dir"])
            state_item["seeds"] = {
                str(seed): {
                    "status": "pending",
                    "returncode": None,
                    "output_dir": str(item_output_dir / "seeds" / ("seed_{}".format(seed))),
                }
                for seed in plan["student_seeds"]
            }
        items.append(state_item)
    return {
        "schema_version": 1,
        "status": "pending",
        "created_at": now,
        "updated_at": now,
        "plan_path": str(plan_path),
        "state_path": str(state_path.resolve()),
        "matrix_root": str(state_path.parent.resolve()),
        "plan_fingerprint": fingerprint,
        "source_sha256": dict(source_hashes),
        "archive": dict(archive_report),
        "archive_sha256": str(plan["archive_sha256"]).lower(),
        "student_seeds": [int(seed) for seed in plan["student_seeds"]],
        "student_jobs": int(plan.get("student_jobs", 5)),
        "teacher_seed": int(plan["teacher_seed"]),
        "device": str(plan.get("device", "cuda")),
        "teacher_checkpoint": None,
        "teacher_cache": None,
        "items": items,
    }


def _reference_configuration(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Compare computational recipes while excluding output locations and hash provenance."""
    ignored = {"output_root", "teacher_cache", "sha256", "archive_sha256", "manifest_sha256", "save_epoch_weights"}
    def clean(value):
        if isinstance(value, Mapping):
            return {key: clean(item) for key, item in value.items() if key not in ignored}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value
    return clean(config)


def _validate_plan_references(plan: Mapping[str, Any], plan_path: Path) -> None:
    """Require each shared control to match an earlier real student configuration."""
    earlier = {}
    for item in plan["items"]:
        if item["kind"] == "reference":
            target = earlier.get(str(item.get("reference_to")))
            if target is None or target["kind"] not in {'student', 'reference'}:
                raise ValueError("共享对照 {} 必须引用其前方的学生或共享项目。".format(item["id"]))
            same_output = PROJECT_ROOT / "reference_validation"
            own = _build_config(plan, plan_path, item, same_output, None, True)
            baseline = _build_config(plan, plan_path, target, same_output, None, True)
            if _reference_configuration(own) != _reference_configuration(baseline):
                raise ValueError("共享对照 {} 的计算配置与引用目标不一致。".format(item["id"]))
        earlier[str(item["id"])] = item


def _assert_unstarted_control(entry: Mapping[str, Any]) -> None:
    """Refuse to turn an already started or orphaned training run into a shared result."""
    records = [entry] + list(entry.get("seeds", {}).values())
    markers = ("started_at", "finished_at", "last_command", "run_dir", "resume_target", "checkpoint")
    for record in records:
        if (record.get("status", "pending") != "pending"
                or record.get("returncode") is not None
                or any(record.get(key) for key in markers)):
            raise ValueError("{} 训练已开始，不能改为共享对照。".format(entry["id"]))
    roots = {Path(str(record["output_dir"])).resolve() for record in records if record.get("output_dir")}
    item_root = Path(str(entry["output_dir"])).resolve()
    allowed_directories = set(roots)
    for root in roots:
        for parent in root.parents:
            if parent == item_root or item_root in parent.parents:
                allowed_directories.add(parent)
    for root in roots:
        if root.is_dir():
            for path in root.rglob("*"):
                if path.is_dir():
                    if path.resolve() not in allowed_directories:
                        raise ValueError("{} 目录中已有权重或训练产物（未知运行目录）。".format(entry["id"]))
                    continue
                if path.name == "README.md":
                    continue
                if path.name == "status.json":
                    try:
                        status = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as error:
                        raise ValueError("{} 状态文档无法读取，不能确认训练未开始。".format(entry["id"])) from error
                    if (isinstance(status, dict) and status.get("status") == "pending"
                            and status.get("returncode") is None
                            and not any(status.get(key) for key in markers)):
                        continue
                if path.is_file():
                    raise ValueError("{} 目录中已有权重或训练产物，不能改为共享对照。".format(entry["id"]))


def _reconcile_resume_plan(plan: Mapping[str, Any], plan_path: Path, state_path: Path,
                           state: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge plan additions in memory without changing past or active training records."""
    _validate_plan_references(plan, plan_path)
    merged = copy.deepcopy(state)
    existing = {str(item["id"]): item for item in merged["items"]}
    planned = {str(item["id"]) for item in plan["items"]}
    if len(existing) != len(merged["items"]):
        raise ValueError("旧矩阵状态含重复项目 ID。")
    removed = sorted(set(existing) - planned)
    if removed:
        raise ValueError("新清单未保留旧项目 {}；不会删除历史记录。".format(removed))
    templates = _initial_state(plan, plan_path, str(state.get("plan_fingerprint", "")),
                               state.get("source_sha256", {}), state.get("archive", {}), state_path)
    initial = {item["id"]: item for item in templates["items"]}
    ordered, added, converted = [], [], []
    for spec in plan["items"]:
        item_id = str(spec["id"])
        if item_id not in existing:
            ordered.append(initial[item_id])
            added.append(item_id)
            continue
        entry = existing[item_id]
        if entry["kind"] != spec["kind"]:
            if entry.get("status") in {"completed", "reused", "running"}:
                # These records are evidence of actual prior work, not retroactive aliases.
                ordered.append(entry)
                continue
            if entry["kind"] != "student" or spec["kind"] != "reference":
                raise ValueError("{} 不支持改变已保存的项目类型。".format(item_id))
            _assert_unstarted_control(entry)
            entry["kind"] = "reference"
            entry["source_item"] = copy.deepcopy(spec)
            entry["reference_to"] = str(spec["reference_to"])
            entry.pop("seeds", None)
            converted.append(item_id)
        ordered.append(entry)
    merged["items"] = ordered
    if added or converted:
        merged.setdefault("plan_revisions", []).append({
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "plan_path": str(plan_path), "added_items": added,
            "converted_to_reference": converted,
            "reason": "增加参数扫描并复用未开始的等效对照；保留已有训练记录。",
        })
    return merged


def _save_state(path: Path, state: Dict[str, Any]) -> None:
    state["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    state["state_path"] = str(Path(path).resolve())
    write_json_atomic(path, state)
    _ensure_matrix_readme(Path(state["matrix_root"]) / "README.md", state)
    matrix_resume = _resume_command_text(path, state.get("python"))
    for item in state.get("items", []):
        output_dir = Path(item["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        _ensure_item_readme(output_dir / "README.md", item, state)
        status = {
            "item_id": item["id"],
            "kind": item["kind"],
            "status": item["status"],
            "returncode": item.get("returncode"),
            "output_dir": str(output_dir.resolve()),
            "resume_command": matrix_resume,
            "last_command": item.get("last_command"),
            "log_path": item.get("log_path"),
            "resolved_config_path": item.get("resolved_config_path"),
            "resolved_config_sha256": item.get("resolved_config_sha256"),
            "reference_to": item.get("reference_to"),
            "checkpoint": item.get("checkpoint"),
            "teacher_cache": state.get("teacher_cache"),
            "seeds": item.get("seeds"),
            "returncodes": {seed: entry.get("returncode") for seed, entry in item.get("seeds", {}).items()},
            "error": item.get("error"),
            "updated_at": state["updated_at"],
        }
        write_json_atomic(output_dir / "status.json", status)
        for seed, seed_state in item.get("seeds", {}).items():
            seed_dir = Path(seed_state["output_dir"])
            seed_dir.mkdir(parents=True, exist_ok=True)
            _ensure_seed_readme(seed_dir / "README.md", item, seed, seed_state, state)
            write_json_atomic(seed_dir / "status.json", {
                "item_id": item["id"], "seed": int(seed),
                "status": seed_state.get("status", "pending"),
                "returncode": seed_state.get("returncode"),
                "output_dir": str(seed_dir.resolve()),
                "run_dir": seed_state.get("run_dir"),
                "resume_target": seed_state.get("resume_target"),
                "resume_command": matrix_resume,
                "last_command": seed_state.get("last_command"),
                "log_path": seed_state.get("log_path"),
                "error": seed_state.get("error"),
                "teacher_cache": state.get("teacher_cache"),
                "updated_at": state["updated_at"],
            })
        for run_dir_text in _known_run_dirs(item):
            run_dir = Path(run_dir_text)
            if run_dir.is_dir():
                _write_run_docs(run_dir, item, state, matrix_resume)


def _resume_command_text(state_path: Path, python: Optional[str]) -> str:
    python_value = python or sys.executable
    command = [python_value, str(PROJECT_ROOT / "scripts" / "run_experiment_matrix.py")]
    if state_path.is_file():
        saved_plan = json.loads(state_path.read_text(encoding="utf-8")).get("plan_path")
        if saved_plan:
            command.extend(["--plan", str(saved_plan)])
    command.extend(["--resume", str(Path(state_path).resolve())])
    return subprocess.list2cmdline(command)


def _known_run_dirs(item: Mapping[str, Any]) -> List[str]:
    values = []
    for key in ("run_dir",):
        if item.get(key):
            values.append(str(item[key]))
    for seed_state in item.get("seeds", {}).values():
        if seed_state.get("run_dir"):
            values.append(str(seed_state["run_dir"]))
        if seed_state.get("resume_target"):
            values.append(str(seed_state["resume_target"]))
    if item.get("resume_target"):
        values.append(str(item["resume_target"]))
    output_dir = Path(item.get("output_dir", ""))
    if item.get("kind") == "teacher" and output_dir.is_dir():
        values.extend(str(path) for path in output_dir.glob("teacher_seed_*") if path.is_dir())
    if item.get("kind") == "student" and output_dir.is_dir():
        for experiment_dir in output_dir.glob("student_*"):
            values.extend(str(path) for path in experiment_dir.glob("seed_*") if path.is_dir())
    return sorted(set(values))


def _ensure_item_readme(path: Path, item: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    spec = item.get("source_item", {})
    override = spec.get("override", "无；使用主配置")
    lines = [
        "# 实验项目：{}".format(item["id"]),
        "",
        "此目录保存本项目的运行文件、日志、状态，以及从矩阵总目录续跑所需的信息。",
        "",
        "## 配置和用途",
        "",
        "- 类型：{}".format(item.get("kind")),
        "- 覆盖配置：{}".format(override),
        "- 结果用途：{}".format(spec.get("evidence_role", spec.get("role", "正式实验"))),
        "- 输出目录：`{}`".format(item.get("output_dir")),
        "- 数据包 SHA-256：`{}`".format(state.get("archive_sha256")),
    ]
    if item.get("kind") == "teacher":
        lines.extend([
            "- 教师种子：`{}`".format(spec.get("seed", 2026)),
            "- 主教师：该 checkpoint 供所有学生项目统一蒸馏使用；教师消融只用于单种子敏感性检查。",
        ])
    elif item.get("kind") == "student":
        lines.extend([
            "- 学生种子：`{}`".format(", ".join(map(str, state.get("student_seeds", STUDENT_SEEDS)))),
            "- 学生并行数上限：`{}`；只并行本项目内部的种子，不同配置仍按矩阵顺序运行。".format(state.get("student_jobs", 5)),
            "- 教师来源：矩阵状态中的 `teacher_checkpoint.path` 和 `teacher_checkpoint.sha256`。",
            "- 每个种子独立保存日志、README、JSON 状态和训练器 run 目录；同一配置最多并行运行指定种子数。",
        ])
    elif item.get("kind") == "reference":
        lines.extend([
            "- 共享对照：`{}`。".format(spec.get("reference_to")),
            "- 本配置与共享对照相同，只登记结果引用，不会重复训练或占用 GPU。",
        ])
    lines.extend([
        "",
        "## 状态和续跑",
        "",
        "当前状态见同目录 `status.json`；矩阵总状态见 `{}`。".format(state.get("state_path", "矩阵状态文件")),
        "遇到中断或失败后，在项目根目录运行：",
        "",
        "```powershell",
        _resume_command_text(Path(state.get("state_path", "runs/experiment_matrix/state.json")), state.get("python")),
        "```",
        "",
        "矩阵按配置清单顺序串行执行，任一项目失败会停止后续项目。",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_seed_readme(path: Path, item: Mapping[str, Any], seed: str,
                        seed_state: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    text = [
        "# {}：seed {}".format(item["id"], seed),
        "",
        "这是该配置单独种子的状态目录。训练日志、训练器 run 目录和矩阵状态分别记录在本目录及 `status.json` 中。",
        "",
        "- 矩阵配置：`{}`。".format(item.get("resolved_config_path", "训练启动时生成")),
        "- 数据包 SHA-256：`{}`。".format(state.get("archive_sha256")),
        "- 教师权重：`{}`。".format((state.get("teacher_checkpoint") or {}).get("path", "等待主教师完成")),
        "- 并行规则：同一学生配置最多同时运行 {} 个种子；不同配置不交叉。".format(state.get("student_jobs", 5)),
        "- 子进程命令：`{}`。".format(subprocess.list2cmdline(seed_state["last_command"]) if seed_state.get("last_command") else "尚未启动"),
        "- 子进程日志：`{}`。".format(seed_state.get("log_path", "启动后生成")),
        "",
        "恢复整个矩阵：",
        "",
        "```powershell",
        _resume_command_text(Path(state.get("state_path", "runs/experiment_matrix/state.json")), state.get("python")),
        "```",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def _write_run_docs(run_dir: Path, item: Mapping[str, Any], state: Mapping[str, Any], resume_command: str) -> None:
    status_path = run_dir / "matrix_status.json"
    if item.get("kind") == "teacher":
        run_status = "completed" if (run_dir / "test_results.json").is_file() else item.get("status")
        run_name = "教师单种子训练 run"
    else:
        seeds = item.get("seeds", {})
        seed_key = run_dir.name.removeprefix("seed_")
        run_status = seeds.get(seed_key, {}).get("status", item.get("status"))
        if (run_dir / "test_results.json").is_file():
            run_status = "completed"
        run_name = "学生单种子训练 run"
    readme = run_dir / "README.md"
    if not readme.is_file():
        readme.write_text(
            "# {}\n\n矩阵项目：`{}`。\n\n"
            "完整训练配置、数据哈希、checkpoint 路径及逐种子状态见上级矩阵项目目录和 `provenance.json`。\n\n"
            "矩阵恢复命令：\n\n```powershell\n{}\n```\n".format(run_name, item["id"], resume_command),
            encoding="utf-8",
        )
    write_json_atomic(status_path, {
        "item_id": item["id"], "kind": item["kind"], "status": run_status,
        "run_dir": str(run_dir.resolve()), "matrix_output_dir": item.get("output_dir"),
        "matrix_resume_command": resume_command,
        "last_command": item.get("last_command"),
        "returncode": item.get("returncode"),
        "data_archive_sha256": state.get("archive_sha256"),
        "teacher_checkpoint": state.get("teacher_checkpoint"),
        "teacher_cache": state.get("teacher_cache"),
    })


def _ensure_matrix_readme(path: Path, state: Mapping[str, Any]) -> None:
    counts = {kind: sum(item["kind"] == kind for item in state.get("items", []))
              for kind in ("teacher", "student", "reference")}
    actual_training_count = counts["teacher"] + counts["student"] * len(state.get("student_seeds", STUDENT_SEEDS))
    item_lines = []
    for index, item in enumerate(state.get("items", []), start=1):
        spec = item.get("source_item", {})
        if item.get("kind") == "reference":
            description = "共享对照，复用 `{}` 的五种子结果".format(spec.get("reference_to"))
        elif item.get("kind") == "teacher":
            description = "教师单种子，用途：{}".format(spec.get("evidence_role", spec.get("role", "教师实验")))
        else:
            description = "学生五种子，配置覆盖：{}".format(spec.get("override", "主配置"))
        item_lines.append("{}. `{}`：{}；状态 `{}`；目录 `{}`".format(
            index, item["id"], description, item.get("status"), item.get("output_dir")))
    resume_command = _resume_command_text(Path(state.get("state_path", path.parent / "state.json")), state.get("python"))
    text = [
        "# Lite-MyoNet 返修实验矩阵",
        "",
        "本目录保存完整矩阵状态、生成后的训练配置、逐项输出、日志与恢复说明。教师任务严格串行；学生任务按配置顺序逐项执行，每项内部最多并行指定数量的种子。任一种子失败会停止该项新增种子并回收同项正在运行的子进程，后续项目暂停。",
        "",
        "## 实验顺序",
        "",
        "顺序：300 轮主教师单种子与教师目标缓存 → 300 轮学生主模型、消融与参数扫描 → 通道宽度和注意力头数按 3/4、1/2 缩减的两个单种子教师。缩减教师不替换学生蒸馏所用的主教师。",
        "",
        "学生主配置为傅里叶 D=2。参数扫描包括 D=1/2/3、固定三次 B 样条的网格 B=1/2/3，以及静态/动态蒸馏系数 0.5/1/2 的 3×3 网格；中心沿用主学生。相同的 D2、完整蒸馏、完整图和余弦头对照不再额外启动训练，已经完成的重复运行仍保留原始记录。",
        "当前状态有 {} 个教师、{} 个真实学生项目、{} 个共享对照，包含历史重复在内共 {} 次训练记录/待执行任务。新清单的去重条件数见 docs/EXPERIMENT_MATRIX.zh-CN.md；同配置同种子的历史重复不得合并成新增独立种子。".format(counts["teacher"], counts["student"], counts["reference"], actual_training_count),
        "",
        "## 教师 checkpoint 注入",
        "",
        "主教师完成后，调度器向学生配置注入 best_teacher.pth 路径；SHA-256 字段用于溯源。--resume 不比较数据、配置、权重或缓存哈希；检查点仍须能够实际读取并加载训练状态。模板中的占位路径不能直接训练。",
        "",
        "## 教师目标缓存",
        "",
        "合成主教师只生成一份训练/验证目标缓存，生成批量与主学生训练批量相同。缓存保存未加权目标，扫描静态/动态损失系数不需重建。恢复检查可读取性和数组结构，不以缓存身份哈希作门槛。缓存不含测试集，测试时在线调用教师。UTD 使用自己的教师，不复用本缓存。",
        "",
        "- 缓存状态：`{}`；目录：`{}`；生成 batch size：`{}`；manifest SHA-256：`{}`。".format(
            (state.get("teacher_cache") or {}).get("status", "等待主教师完成"),
            (state.get("teacher_cache") or {}).get("path", "尚未生成"),
            (state.get("teacher_cache") or {}).get("generation_batch_size", "尚未生成"),
            (state.get("teacher_cache") or {}).get("manifest_sha256", "尚未生成"),
        ),
        "",
        "## 数据与种子",
        "",
        "- 数据包 SHA-256：`{}`。".format(state.get("archive_sha256")),
        "- 教师目标缓存状态：`{}`；路径：`{}`；manifest SHA-256：`{}`。".format(
            (state.get("teacher_cache") or {}).get("status", "等待主教师完成"),
            (state.get("teacher_cache") or {}).get("path", "尚未生成"),
            (state.get("teacher_cache") or {}).get("manifest_sha256", "尚未生成"),
        ),
        "- 缓存只含 train/val；test 始终在线运行教师。学生种子共享同一只读 memmap。",
        "- 主教师种子：`{}`。".format(state.get("teacher_seed", 2026)),
        "- 学生主模型和消融种子：`{}`。".format(", ".join(map(str, state.get("student_seeds", STUDENT_SEEDS)))),
        "- 每个学生配置的并行种子数上限：`{}`。".format(state.get("student_jobs", 5)),
        "- 每个训练子进程的 CPU 计算线程上限：`{}`；五个学生种子对应五个实际训练进程。".format(state.get("cpu_threads_per_process", 1)),
        "- 默认五种子并行；父调度只处理状态与进度，不加载 GPU 模型。并行吞吐与显存取决于实际配置，历史短基准不作为本队列的实测结果。",
        "- 教师结构消融是单种子结果，只作结构敏感性检查，不并入教师主结果统计。",
        "",
        "## 命令",
        "",
        "正式矩阵清单已记录并校验 NPZ 路径；检查命令和配置：",
        "",
        "```powershell",
        "python scripts/run_experiment_matrix.py --dry-run",
        "```",
        "",
        "正式启动整套串行矩阵：",
        "",
        "```powershell",
        "python scripts/run_experiment_matrix.py",
        "```",
        "",
        "如果训练中断或子进程失败，直接执行：",
        "",
        "```powershell",
        resume_command,
        "```",
        "",
        "续跑合并新项目，保留完成或运行中的原始记录，跳过已完成任务，从可读取检查点继续未完成种子。无可恢复权重时明确报错，不假装恢复。新增参数条件按新配置从头训练；恢复不检验哈希。",
        "",
        "## 项目目录和状态",
        "",
        "每个项目保留说明和状态 JSON；真实训练目录保留原始权重、逐轮日志和测试结果，共享对照通过 reference.json 指向主学生，不伪造独立训练。教师与学生指标以 test_results.json、multiseed_summary.json 和 provenance.json 为准。",
        "",
        "| 序号 | 项目 | 说明 | 状态 | 输出目录 |",
        "| ---: | --- | --- | --- | --- |",
        *["| {} | `{}` | {} | `{}` | `{}` |".format(
            index, item["id"],
            "共享对照，复用 `{}`".format(item["source_item"].get("reference_to"))
            if item["kind"] == "reference" else str(item["kind"]),
            item.get("status"), item.get("output_dir"))
          for index, item in enumerate(state.get("items", []), start=1)],
        "",
        "全局状态文件：`{}`。".format(state.get("state_path", "state.json")),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def _item_state(state: Mapping[str, Any], item_id: str) -> Dict[str, Any]:
    for item in state["items"]:
        if item["id"] == item_id:
            return item
    raise KeyError("状态文件缺少矩阵项目 {}。".format(item_id))


def _prepare_config_file(
    plan: Mapping[str, Any], plan_path: Path, state_path: Path,
    item: Mapping[str, Any], item_state: Dict[str, Any],
    teacher_checkpoint: Optional[Mapping[str, str]],
    teacher_cache: Optional[Mapping[str, str]] = None,
) -> Path:
    config_path, output_dir = _config_paths(plan, state_path, item["id"])
    config = _build_config(
        plan, plan_path, item, output_dir, teacher_checkpoint, False, teacher_cache
    )
    _validate_materialized_config(config, str(item["kind"]))
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    item_state["resolved_config_path"] = str(config_path)
    item_state["resolved_config_sha256"] = sha256_file(config_path)
    item_state["source_config_sha256"] = sha256_file(_source_config_path(plan_path, plan, item))
    if item.get("override"):
        item_state["override_config_sha256"] = sha256_file((plan_path.parent / item["override"]).resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    return config_path


def _run_teacher(
    plan: Mapping[str, Any], plan_path: Path, state_path: Path,
    item: Mapping[str, Any], item_state: Dict[str, Any], state: Dict[str, Any],
    python: str, device: str, resume_mode: bool,
) -> None:
    output_dir = Path(item_state["output_dir"])
    seed = int(item.get("seed", plan["teacher_seed"]))
    if item.get("role") == "primary_teacher" and plan.get("external_teacher"):
        checkpoint_path = Path(plan["external_teacher"]["path"]).resolve()
        import torch
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint = {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)}
        item_state.update(status="completed", run_dir=str(checkpoint_path.parent),
                          checkpoint=checkpoint, external=True,
                          finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
        state["teacher_checkpoint"] = checkpoint
        _save_state(state_path, state)
        return
    completed = _discover_teacher_complete(output_dir, seed)
    if completed:
        run_dir, checkpoint_path = completed
        checkpoint = {"path": str(checkpoint_path.resolve()), "sha256": sha256_file(checkpoint_path)}
        item_state.update({"status": "completed", "run_dir": str(run_dir), "checkpoint": checkpoint,
                           "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(), "inferred_complete": True})
        if item.get("role") == "primary_teacher":
            state["teacher_checkpoint"] = checkpoint
        _save_state(state_path, state)
        return

    config_path = _prepare_config_file(plan, plan_path, state_path, item, item_state, None)
    interrupted_run = _find_interrupted_teacher(output_dir, seed) if resume_mode else None
    command = _command_for(item, config_path, python, device, resume=interrupted_run)
    log_path = output_dir / "matrix.log"
    item_state.update({"status": "running", "started_at": item_state.get("started_at") or dt.datetime.now(dt.timezone.utc).isoformat(),
                       "resume_target": str(interrupted_run) if interrupted_run else None,
                       "last_command": command, "log_path": str(log_path)})
    state["status"] = "running"
    teacher_environment = dict(os.environ)
    recomputation = bool(plan.get("teacher_attention_recomputation", True))
    teacher_environment["LITE_MYONET_CHECKPOINT_ATTENTION"] = "1" if recomputation else "0"
    item_state["attention_recomputation"] = recomputation
    _save_state(state_path, state)
    code = _stream_process(command, log_path, PROJECT_ROOT, teacher_environment)
    item_state["returncode"] = code
    if code != 0:
        raise subprocess.CalledProcessError(code, command)
    completed = _discover_teacher_complete(output_dir, seed)
    if not completed:
        raise RuntimeError("教师命令退出码为 0，但没有找到 best_teacher.pth 和 test_results.json。")
    run_dir, checkpoint_path = completed
    checkpoint = {"path": str(checkpoint_path.resolve()), "sha256": sha256_file(checkpoint_path)}
    item_state.update({"status": "completed", "run_dir": str(run_dir), "checkpoint": checkpoint,
                       "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(), "returncode": code})
    if item.get("role") == "primary_teacher":
        state["teacher_checkpoint"] = checkpoint
    _save_state(state_path, state)


def _teacher_cache_identity(plan, plan_path, state, teacher_checkpoint, device):
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import torch
    from lite_myonet.training.student_trainer import effective_precision
    from lite_myonet.training.teacher_target_cache import make_cache_identity
    from lite_myonet.training.teacher_trainer import (
        code_tree_sha256,
        load_config,
        load_label_map,
    )

    primary = next(
        item for item in plan["items"]
        if item.get("kind") == "teacher" and item.get("role") == "primary_teacher"
    )
    primary_state = _item_state(state, str(primary["id"]))
    run_dir = Path(primary_state["run_dir"])
    teacher_config, teacher_config_path = load_config(run_dir / "resolved_config.yaml")
    label_map, label_map_path = load_label_map(teacher_config, teacher_config_path)
    student_main = next(item for item in plan["items"] if item.get("id") == "student_main")
    student_config = load_yaml(_source_config_path(plan_path, plan, student_main))
    requested_precision = str(student_config["precision"])
    cache_device = device
    if device == "auto":
        cache_device = "cuda" if torch.cuda.is_available() else "cpu"
    effective = effective_precision(torch.device(cache_device), requested_precision)
    loss_config = {
        "kd_temperature": float(student_config["distillation"]["kd_temperature"]),
        "topology_temperature": float(student_config["distillation"]["topology_temperature"]),
    }
    generation_batch_size = int(student_config["batch_size"])
    if generation_batch_size <= 0:
        raise ValueError("student_main.batch_size 必须为正整数。")
    code_sha = code_tree_sha256(PROJECT_ROOT)
    identity = make_cache_identity(
        teacher_checkpoint_sha256=teacher_checkpoint["sha256"],
        archive_sha256=state["archive_sha256"],
        label_map_sha256=sha256_file(label_map_path),
        requested_precision=requested_precision,
        effective_precision=effective,
        teacher_model_config=teacher_config["model"],
        loss_config=loss_config,
        code_sha256=code_sha,
        generation_batch_size=generation_batch_size,
    )
    archive = state.get("archive", {})
    if not archive.get("resolved") or not archive.get("path"):
        raise RuntimeError("构建教师目标缓存需要已校验的数据包路径。")
    return {
        "run_dir": run_dir,
        "teacher_config": teacher_config,
        "teacher_config_path": teacher_config_path,
        "teacher_model_config": teacher_config["model"],
        "label_map": label_map,
        "label_map_path": label_map_path,
        "requested_precision": requested_precision,
        "effective_precision": effective,
        "loss_config": loss_config,
        "generation_batch_size": generation_batch_size,
        "code_sha256": code_sha,
        "identity": identity,
        "archive_path": Path(archive["path"]),
        "device": cache_device,
    }


def _ensure_teacher_target_cache(plan, plan_path, state_path, state, teacher_checkpoint, device):
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import torch
    from lite_myonet.training.student_trainer import _create_teacher_model, _torch_load_checkpoint
    from lite_myonet.training.teacher_target_cache import (
        TeacherTargetCache,
        build_teacher_target_cache,
    )

    if not isinstance(teacher_checkpoint, Mapping):
        raise RuntimeError("主教师 checkpoint 未登记，不能构建教师目标缓存。")
    checkpoint_path = Path(teacher_checkpoint["path"]).resolve()
    if not checkpoint_path.is_file() or (
        not _resume_readable_only() and sha256_file(checkpoint_path) != teacher_checkpoint["sha256"]
    ):
        raise ValueError("缓存构建前教师 checkpoint 路径或 SHA-256 校验失败。")
    details = _teacher_cache_identity(plan, plan_path, state, teacher_checkpoint, device)
    cache_dir = Path(state["matrix_root"]).resolve() / "teacher_target_cache"
    expected = details["identity"]
    previous = state.get("teacher_cache")
    if previous is not None:
        if Path(previous.get("path", "")).resolve() != cache_dir:
            raise ValueError("矩阵恢复时教师目标缓存路径与既有状态不一致。")
        previous_identity = previous.get("identity")
        if not _resume_readable_only() and (
            not isinstance(previous_identity, Mapping) or set(previous_identity) != set(expected)
        ):
            raise ValueError("矩阵恢复状态缺少完整的教师目标缓存身份。")
        if not _resume_readable_only() and any(
            previous_identity[key] != value
            for key, value in expected.items() if key != "code_sha256"
        ):
            raise ValueError("矩阵恢复时教师目标缓存身份与当前教师/数据/精度/损失不一致。")
        if not _resume_readable_only() and previous.get("generation_batch_size") != details["generation_batch_size"]:
            raise ValueError("矩阵恢复时教师目标缓存生成 batch size 与当前学生配置不一致。")
        if previous.get("status") == "complete" and not cache_dir.is_dir():
            raise FileNotFoundError("矩阵状态指向的已完成教师目标缓存目录缺失。")
        if not _resume_readable_only() and previous.get("status") == "complete" and not previous.get("manifest_sha256"):
            raise ValueError("矩阵已完成缓存状态缺少固定的 manifest SHA-256。")

    if cache_dir.exists():
        reader = TeacherTargetCache(
            cache_dir,
            expected_hashes=expected,
            verify_hashes=True,
            allow_code_provenance_mismatch=previous is not None,
        )
        manifest_sha = reader.status["manifest_sha256"]
        if not _resume_readable_only() and previous and previous.get("manifest_sha256") not in (None, manifest_sha):
            reader.close()
            raise ValueError("矩阵状态记录的教师目标缓存 manifest 与磁盘不一致。")
        manifest_identity = reader.manifest["identity"]
        if not _resume_readable_only() and previous is not None and previous["identity"] != manifest_identity:
            reader.close()
            raise ValueError("矩阵状态固定身份与教师目标缓存 manifest 身份不一致。")
        state["teacher_cache"] = {
            "status": "complete",
            "path": str(cache_dir),
            "generation_batch_size": details["generation_batch_size"],
            "manifest_path": str((cache_dir / "manifest.json").resolve()),
            "manifest_sha256": manifest_sha,
            "identity": manifest_identity,
            "total_rows": sum(int(value["count"]) for value in reader.manifest["splits"].values()),
            "test_cached": False,
        }
        reader.close()
        _save_state(state_path, state)
        return state["teacher_cache"]

    state["teacher_cache"] = {
        "status": "building", "path": str(cache_dir), "identity": expected,
        "generation_batch_size": details["generation_batch_size"],
        "test_cached": False,
    }
    _save_state(state_path, state)
    checkpoint = _torch_load_checkpoint(checkpoint_path, map_location="cpu")
    teacher = _create_teacher_model(
        details["teacher_model_config"], len(details["label_map"])
    ).to(details["device"]).float()
    teacher.load_state_dict(checkpoint["model_state_dict"], strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    try:
        manifest = build_teacher_target_cache(
            cache_dir,
            details["archive_path"],
            checkpoint_path,
            details["label_map_path"],
            teacher,
            details["device"],
            teacher_config=details["teacher_model_config"],
            loss_config=details["loss_config"],
            requested_precision=details["requested_precision"],
            effective_precision=details["effective_precision"],
            code_sha256=details["code_sha256"],
            generation_batch_size=details["generation_batch_size"],
            expected_hashes=expected,
        )
        reader = TeacherTargetCache(cache_dir, expected_hashes=expected, verify_hashes=True)
        state["teacher_cache"] = {
            "status": "complete",
            "path": str(cache_dir),
            "generation_batch_size": details["generation_batch_size"],
            "manifest_path": str((cache_dir / "manifest.json").resolve()),
            "manifest_sha256": reader.status["manifest_sha256"],
            "identity": expected,
            "total_rows": sum(int(value["count"]) for value in manifest["splits"].values()),
            "test_cached": False,
        }
        reader.close()
    finally:
        del teacher
        if torch.device(details["device"]).type == "cuda":
            torch.cuda.empty_cache()
    _save_state(state_path, state)
    return state["teacher_cache"]


def _run_student(
    plan: Mapping[str, Any], plan_path: Path, state_path: Path,
    item: Mapping[str, Any], item_state: Dict[str, Any], state: Dict[str, Any],
    python: str, device: str, resume_mode: bool, student_jobs: int = 5,
) -> None:
    teacher_checkpoint = state.get("teacher_checkpoint")
    if not isinstance(teacher_checkpoint, Mapping):
        raise RuntimeError("主教师 checkpoint 尚未登记，不能运行学生模型。")
    teacher_path = Path(teacher_checkpoint["path"])
    if not teacher_path.is_file() or (
        not _resume_readable_only() and sha256_file(teacher_path) != teacher_checkpoint["sha256"]
    ):
        raise RuntimeError("主教师 checkpoint 路径或 SHA-256 校验失败。")
    teacher_cache = state.get("teacher_cache")
    if not isinstance(teacher_cache, Mapping) or teacher_cache.get("status") != "complete":
        raise RuntimeError("主教师目标缓存尚未完成，不能启动学生训练。")
    output_dir = Path(item_state["output_dir"])
    config_path = _prepare_config_file(
        plan, plan_path, state_path, item, item_state, teacher_checkpoint, teacher_cache
    )
    if not 1 <= int(student_jobs) <= len(plan["student_seeds"]):
        raise ValueError("student_jobs 必须位于 1 到 {} 之间。".format(len(plan["student_seeds"])))
    pending = []
    for progress_position, seed in enumerate(plan["student_seeds"]):
        seed = int(seed)
        seed_state = item_state["seeds"][str(seed)]
        completed_run = _discover_student_complete(output_dir, seed)
        if completed_run:
            seed_state.update({"status": "completed", "run_dir": str(completed_run), "inferred_complete": True})
            _save_state(state_path, state)
            continue
        interrupted_run = _find_interrupted_student(output_dir, seed) if resume_mode else None
        command = _command_for(
            item, config_path, python, device, seed=seed, resume=interrupted_run,
        )
        seed_dir = Path(seed_state["output_dir"])
        log_path = seed_dir / "training.log"
        seed_state.update({
            "resume_target": str(interrupted_run) if interrupted_run else None,
            "last_command": command, "log_path": str(log_path),
        })
        process_env = dict(os.environ)
        process_env["LITE_MYONET_PROGRESS_POSITION"] = str(progress_position)
        pending.append((seed, seed_state, command, log_path, process_env))

    if pending:
        registry = ProcessRegistry()
        active = {}
        next_index = 0
        failure = None
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=min(int(student_jobs), len(pending)))

        def submit_next():
            nonlocal next_index
            seed, seed_state, command, log_path, process_env = pending[next_index]
            next_index += 1
            seed_state.update({
                "status": "running",
                "started_at": seed_state.get("started_at") or dt.datetime.now(dt.timezone.utc).isoformat(),
            })
            item_state["status"] = "running"
            item_state["started_at"] = item_state.get("started_at") or dt.datetime.now(dt.timezone.utc).isoformat()
            state["status"] = "running"
            _save_state(state_path, state)
            future = executor.submit(
                _stream_process, command, log_path, PROJECT_ROOT, process_env,
                registry, "{}/seed_{}".format(item["id"], seed),
            )
            active[future] = (seed, seed_state, command)

        try:
            for _ in range(min(int(student_jobs), len(pending))):
                submit_next()
            while active:
                done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    seed, seed_state, command = active.pop(future)
                    try:
                        code = int(future.result())
                    except BaseException as error:
                        code = None
                        if failure is None:
                            failure = error
                        seed_state.update({"status": "failed", "returncode": None,
                                           "error": "子进程启动或读取失败：{}".format(error)})
                    else:
                        seed_state["returncode"] = code
                        completed_run = _discover_student_complete(output_dir, seed)
                        if code == 0 and completed_run:
                            seed_state.update({"status": "completed", "run_dir": str(completed_run),
                                               "finished_at": dt.datetime.now(dt.timezone.utc).isoformat()})
                        elif code != 0:
                            seed_state.update({"status": "failed", "error": "训练子进程退出码 {}".format(code)})
                            if failure is None:
                                failure = subprocess.CalledProcessError(code, command)
                        else:
                            seed_state.update({"status": "failed", "error": "训练命令成功退出，但没有唯一测试结果。"})
                            if failure is None:
                                failure = RuntimeError("学生 seed {} 未生成唯一测试结果。".format(seed))
                    _save_state(state_path, state)
                if failure is not None:
                    registry.stop_all()
                    # 等待并登记所有正在运行的种子；不再提交 pending 中的种子。
                    for future, (seed, seed_state, command) in list(active.items()):
                        try:
                            code = int(future.result())
                        except BaseException as error:
                            code = None
                            detail = "同项其他种子失败后已停止：{}".format(error)
                        else:
                            detail = "同项其他种子失败后停止新增任务。"
                        completed_run = _discover_student_complete(output_dir, seed) if code == 0 else None
                        if completed_run:
                            seed_state.update({"status": "completed", "returncode": code,
                                               "run_dir": str(completed_run),
                                               "finished_at": dt.datetime.now(dt.timezone.utc).isoformat()})
                        else:
                            seed_state.update({"status": "failed", "returncode": code, "error": detail,
                                               "finished_at": dt.datetime.now(dt.timezone.utc).isoformat()})
                        active.pop(future, None)
                        _save_state(state_path, state)
                    break
                while next_index < len(pending) and len(active) < int(student_jobs):
                    submit_next()
        except BaseException:
            registry.stop_all()
            for future, (_, seed_state, _) in list(active.items()):
                try:
                    code = int(future.result())
                except BaseException:
                    code = None
                seed_state.update({"status": "failed", "returncode": code,
                                   "error": "矩阵中断；可通过 --resume 恢复该种子。"})
            _save_state(state_path, state)
            raise
        finally:
            executor.shutdown(wait=True)
        if failure is not None:
            raise failure
    item_state.update({"status": "completed", "finished_at": dt.datetime.now(dt.timezone.utc).isoformat()})
    _save_state(state_path, state)


def _run_reference(
    plan: Mapping[str, Any], plan_path: Path, state_path: Path,
    item: Mapping[str, Any], item_state: Dict[str, Any], state: Dict[str, Any],
) -> None:
    target_id = str(item["reference_to"])
    target = _item_state(state, target_id)
    while target.get('kind') == 'reference':
        target = _item_state(state, str(target['reference_to']))
    if target.get("status") != "completed":
        raise RuntimeError("共享对照 {} 等待 {} 完成。".format(item["id"], target_id))
    config_path, output_dir = _config_paths(plan, state_path, item["id"])
    config = _build_config(plan, plan_path, item, output_dir, state.get("teacher_checkpoint"), False)
    _validate_materialized_config(config, "student")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = {
        "item_id": item["id"], "kind": "shared_reference",
        "reference_to": target_id,
        "source_runs": {
            key: value.get("run_dir")
            for key, value in target.get("seeds", {}).items()
        },
        "source_config_sha256": target.get("resolved_config_sha256"),
        "reference_config_sha256": sha256_file(config_path),
        "reason": "该配置与 student_main 完全相同，共享五种子结果以避免重复训练。",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    write_json_atomic(output_dir / "reference.json", reference)
    item_state.update({"status": "reused", "reference_to": target_id,
                       "resolved_config_path": str(config_path),
                       "resolved_config_sha256": reference["reference_config_sha256"],
                       "reference_record": str(output_dir / "reference.json"),
                       "finished_at": dt.datetime.now(dt.timezone.utc).isoformat()})
    _save_state(state_path, state)


def run_matrix(plan: Dict[str, Any], plan_path: Path, state_path: Path, device: Optional[str],
               resume: bool, student_jobs: Optional[int] = None,
               teacher_only: bool = False, only_items: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    selected_ids = set(only_items) if only_items is not None else None
    if selected_ids is not None:
        unknown = selected_ids - {item['id'] for item in plan['items']}
        if unknown or not selected_ids:
            raise ValueError('指定的矩阵任务为空或不存在：{}'.format(sorted(unknown)))
    state_path = Path(state_path).expanduser().resolve()
    os.environ["LITE_MYONET_RESUME_READABLE_ONLY"] = "1"
    fingerprint, source_hashes = plan_fingerprint(plan, plan_path)
    resume_state = None
    if resume:
        if not state_path.is_file():
            raise FileNotFoundError("--resume 找不到矩阵状态文件：{}".format(state_path))
        resume_state = json.loads(state_path.read_text(encoding="utf-8"))
        _restore_archive_environment_from_state(plan, resume_state)
    archive_report = _archive_check(plan, allow_unresolved=False)
    python_launcher = _select_python(plan)
    python, runtime_updates = _resolve_python_runtime(python_launcher, os.environ)
    runtime_environment = _training_environment(
        plan, {**dict(os.environ), **runtime_updates}
    )
    for name in (
        "VIRTUAL_ENV", "PYTHONPATH", "PATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        "TORCH_NUM_THREADS", "TORCH_NUM_INTEROP_THREADS",
    ):
        if name in runtime_environment:
            os.environ[name] = runtime_environment[name]
    chosen_device = device or str(plan.get("device", "cuda"))
    if chosen_device == "cuda" and shutil.which(python) is None and not Path(python).is_file():
        raise FileNotFoundError("Python 解释器不可用：{}".format(python))
    if resume:
        state = resume_state
        if state.get("schema_version") != 1:
            raise ValueError("矩阵状态结构无法读取。")
        state = _reconcile_resume_plan(plan, plan_path, state_path, state)
        # 设备类型改变会使随机数状态和续训语义不一致。
        prior_device = state.get("active_device", state.get("device"))
        if not _resume_readable_only() and prior_device != chosen_device:
            raise ValueError("续跑设备 {} 与原状态设备 {} 不同。".format(chosen_device, prior_device))
        selected_student_jobs = int(state.get("student_jobs", plan.get("student_jobs", 5)))
    else:
        if state_path.exists():
            raise FileExistsError("状态文件已存在；使用 --resume，或为新实验指定新 --state。")
        occupied = [
            item["id"] for item in plan["items"]
            if _config_paths(plan, state_path, item["id"])[1].exists()
        ]
        if occupied:
            raise FileExistsError(
                "矩阵状态文件不存在，但以下项目输出目录已存在：{}。"
                "为避免覆盖旧结果，请恢复原状态文件或使用新的 output_root。".format(occupied)
            )
        state = _initial_state(plan, plan_path, fingerprint, source_hashes, archive_report, state_path)
        state["device"] = chosen_device
        state["active_device"] = chosen_device
        selected_student_jobs = int(plan.get("student_jobs", 5))
        _save_state(state_path, state)

    if student_jobs is not None:
        selected_student_jobs = int(student_jobs)
    if not 1 <= selected_student_jobs <= len(plan["student_seeds"]):
        raise ValueError("--student-jobs 必须位于 1 到 {} 之间。".format(len(plan["student_seeds"])))
    state["student_jobs"] = selected_student_jobs
    state["active_device"] = chosen_device
    state["python_launcher"] = python_launcher
    state["python"] = python
    state["cpu_threads_per_process"] = int(plan["cpu_threads_per_process"])
    state["status"] = "running"
    state["resume_validation"] = {
        "mode": "readable_only" if _resume_readable_only() else "strict",
        "hash_checks_enabled": not _resume_readable_only(),
        "checkpoint_deserialization_required": True,
    }
    if state.get('fourier_selection'):
        from fourier_selection import apply_selection
        apply_selection(plan, state, state['fourier_selection'])
    _save_state(state_path, state)
    plan_items = {str(item["id"]): item for item in plan["items"]}
    try:
        for item in (plan["items"][:1] if teacher_only else plan["items"]):
            item_id = str(item["id"])
            if selected_ids is not None and item_id not in selected_ids:
                continue
            if item_id == 'student_main' and plan.get('degree_selection'):
                from fourier_selection import apply_selection, choose_degree
                selection = state.get('fourier_selection') or choose_degree(state, plan['student_seeds'])
                apply_selection(plan, state, selection)
                write_json_atomic(state_path.parent / 'fourier_selection.json', selection)
                _save_state(state_path, state)
                print(f"主配置已选择 Fourier D={selection['selected_degree']}；依据五种子最佳验证准确率。", flush=True)
            item_state = _item_state(state, item_id)
            if item_state["kind"] != item["kind"] and item_state.get("status") not in {"completed", "reused"}:
                # An interrupted former duplicate remains a real training task until complete.
                item = dict(item, kind=item_state["kind"])
            if item_state.get("status") in {"completed", "reused"}:
                if not teacher_only and item["kind"] == "teacher" and item.get("role") == "primary_teacher":
                    primary_checkpoint = state.get("teacher_checkpoint") or item_state.get("checkpoint")
                    _ensure_teacher_target_cache(
                        plan, plan_path, state_path, state, primary_checkpoint, chosen_device
                    )
                continue
            item_state["error"] = None
            item_state["status"] = "running"
            _save_state(state_path, state)
            if item["kind"] == "teacher":
                _run_teacher(plan, plan_path, state_path, item, item_state, state, python, chosen_device, resume)
                if not teacher_only and item.get("role") == "primary_teacher":
                    _ensure_teacher_target_cache(
                        plan, plan_path, state_path, state,
                        state.get("teacher_checkpoint"), chosen_device,
                    )
            elif item["kind"] == "student":
                _run_student(plan, plan_path, state_path, item, item_state, state, python,
                             chosen_device, resume, selected_student_jobs)
            else:
                _run_reference(plan, plan_path, state_path, item, item_state, state)
        if teacher_only:
            state["status"] = "teacher_ready"
            _save_state(state_path, state)
            return state
        # Every item must be complete before the matrix can report success.
        incomplete = [entry["id"] for entry in state["items"] if entry["status"] not in {"completed", "reused"}]
        if incomplete and selected_ids is not None:
            state['status'] = 'partial'
            state['completed_scope'] = sorted(selected_ids)
            _save_state(state_path, state)
            return state
        if incomplete:
            raise RuntimeError("矩阵尚有未完成项目：{}。".format(incomplete))
        state["status"] = "completed"
        state["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _save_state(state_path, state)
        return state
    except KeyboardInterrupt:
        current = next((entry for entry in state["items"] if entry.get("status") == "running"), None)
        if current:
            current["status"] = "failed"
            current["error"] = "用户中断；使用 --resume 从 last checkpoint 接续。"
            for seed_state in current.get("seeds", {}).values():
                if seed_state.get("status") == "running":
                    seed_state["status"] = "failed"
        state["status"] = "failed"
        _save_state(state_path, state)
        raise
    except Exception as error:
        current = next((entry for entry in state["items"] if entry.get("status") == "running"), None)
        if current:
            current["status"] = "failed"
            current["error"] = "{}: {}".format(type(error).__name__, error)
            current["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            for seed_state in current.get("seeds", {}).values():
                if seed_state.get("status") == "running":
                    seed_state["status"] = "failed"
                    seed_state["error"] = current["error"]
        state["status"] = "failed"
        state["error"] = "{}: {}".format(type(error).__name__, error)
        _save_state(state_path, state)
        raise


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按清单运行 Lite-MyoNet 教师、学生和消融训练。")
    parser.add_argument("--plan", default=str(DEFAULT_PLAN), help="实验矩阵 YAML 清单。")
    parser.add_argument("--state", help="覆盖清单中的状态 JSON 路径。")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), help="覆盖清单设备。")
    parser.add_argument("--student-jobs", type=int, help="每个学生配置内并行的种子数上限；教师仍严格串行。")
    parser.add_argument("--teacher-only", action="store_true", help="只运行主教师，完成测试后停止；不构建缓存或运行学生")
    parser.add_argument("--only-items", nargs="+", help="只执行指定配置，其他任务保留原状态，供双机分工使用。")
    parser.add_argument("--dry-run", action="store_true", help="校验所有配置并打印将执行的命令，不训练、不写状态。")
    parser.add_argument("--resume", nargs="?", const="", help="从状态 JSON 继续，默认不检验哈希；可选地传入状态文件路径。")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    # Start-Process 重定向日志时，Windows 仍可能给本进程配置 GBK 文本流。
    # 主控与所有训练子进程统一使用 UTF-8，确保中文路径和进度条可以稳定写入日志。
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    if args.resume is not None:
        os.environ["LITE_MYONET_RESUME_READABLE_ONLY"] = "1"
    plan, plan_path = load_plan(Path(args.plan))
    state_path = Path(args.state).expanduser().resolve() if args.state else _resolve_from_project(plan["state_path"])
    if args.dry_run:
        report = build_dry_run(plan, plan_path, args.device, args.student_jobs, args.teacher_only)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    resume = args.resume is not None
    if resume and args.resume:
        state_path = Path(args.resume).expanduser().resolve()
    state = run_matrix(plan, plan_path, state_path, args.device, resume, args.student_jobs,
                       args.teacher_only, args.only_items)
    print(json.dumps({"state": str(state_path), "status": state["status"], "items": len(state["items"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
