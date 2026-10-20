"""严格在中央处理器上导出 Lite-MyoNet 正式学生 checkpoint。"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import os
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch import nn

from lite_myonet.models.student import MyoFormerStudent


INPUT_NAMES = ("joint_coordinates", "bone_vectors")
OUTPUT_NAME = "logits"
INPUT_SHAPE = (3, 120, 23)
NUM_CLASSES = 33
DEFAULT_OPSET = 17
_REQUIRED_CHECKPOINT_KEYS = frozenset(
    {"student_state_dict", "student_config", "seed", "epoch", "selection_metric"}
)
_STUDENT_CONFIG_KEYS = frozenset(
    key
    for key in inspect.signature(MyoFormerStudent.__init__).parameters
    if key != "self"
)


class ONNXExportDependencyError(RuntimeError):
    """Raised when an optional ONNX export/runtime dependency is unavailable."""


class StudentLogitsExportWrapper(nn.Module):
    """按稳定的双输入接口仅输出学生模型分类 logits。"""

    def __init__(self, student: MyoFormerStudent):
        super().__init__()
        self.student = student

    def forward(self, joint_coordinates: torch.Tensor, bone_vectors: torch.Tensor):
        return self.student(joint_coordinates, bone_vectors)


def _require_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as exc:
        install_name = "onnxruntime" if module_name == "onnxruntime" else "onnx"
        raise ONNXExportDependencyError(
            f"缺少 ONNX 导出依赖 `{install_name}`；请安装 `pip install {install_name}` 后重试。"
        ) from exc


def _require_export_dependencies():
    onnx = _require_module("onnx")
    onnxruntime = _require_module("onnxruntime")
    return onnx, onnxruntime


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_state_dict(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _sha256_model(
    model_config: Mapping[str, Any], state_dict: Mapping[str, torch.Tensor]
) -> Tuple[str, str]:
    state_dict_sha256 = _sha256_state_dict(state_dict)
    canonical_config = json.dumps(
        _json_compatible(model_config), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"config\0")
    digest.update(canonical_config)
    digest.update(b"\0state_dict\0")
    digest.update(state_dict_sha256.encode("ascii"))
    return digest.hexdigest(), state_dict_sha256


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"checkpoint 配置包含不能写入 JSON 的值：{type(value).__name__}。")


def _validate_checkpoint_metadata(checkpoint: Any) -> Dict[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("学生 checkpoint 顶层必须是包含元数据的映射。")
    missing = sorted(_REQUIRED_CHECKPOINT_KEYS - set(checkpoint))
    if missing:
        raise ValueError(f"学生 checkpoint 缺少必需字段：{missing}。")

    config = checkpoint["student_config"]
    if not isinstance(config, Mapping):
        raise ValueError("student_config 必须是模型构造参数映射。")
    unknown_config = sorted(set(config) - _STUDENT_CONFIG_KEYS)
    if unknown_config:
        raise ValueError(f"student_config 含有模型不支持的字段：{unknown_config}。")

    for key in ("seed", "epoch"):
        value = checkpoint[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"checkpoint 字段 {key} 必须是非负整数。")
    metric = checkpoint["selection_metric"]
    if not isinstance(metric, str) or not metric.strip():
        raise ValueError("selection_metric 必须是非空字符串。")

    normalized_config = dict(config)
    expected_dimensions = {
        "num_classes": NUM_CLASSES,
        "in_channels": INPUT_SHAPE[0],
        "num_points": INPUT_SHAPE[2],
    }
    for key, expected in expected_dimensions.items():
        configured = normalized_config.get(key, expected)
        if configured != expected:
            raise ValueError(
                f"正式 ONNX 协议要求 {key}={expected}，checkpoint 配置为 {configured}。"
            )

    state_dict = checkpoint["student_state_dict"]
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("student_state_dict 必须是非空参数映射。")
    invalid_values = [
        key for key, value in state_dict.items() if not isinstance(key, str) or not isinstance(value, torch.Tensor)
    ]
    if invalid_values:
        raise ValueError("student_state_dict 的键必须是字符串，值必须是张量。")

    try:
        json_config = _json_compatible(normalized_config)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc
    return {
        "student_config": json_config,
        "seed": int(checkpoint["seed"]),
        "epoch": int(checkpoint["epoch"]),
        "selection_metric": metric.strip(),
    }


def load_formal_student_checkpoint(
    checkpoint_path: os.PathLike | str,
) -> Tuple[MyoFormerStudent, Dict[str, Any]]:
    """Load the agreed formal checkpoint schema with exact state-key matching."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到正式学生 checkpoint：{path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "当前 PyTorch 不支持安全的 weights_only checkpoint 加载；请升级 PyTorch。"
        ) from exc

    metadata = _validate_checkpoint_metadata(checkpoint)
    model = MyoFormerStudent(**metadata["student_config"])
    try:
        model.load_state_dict(checkpoint["student_state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError(f"student_state_dict 与 student_config 不匹配：{exc}") from exc
    model.to(device=torch.device("cpu"))
    model.eval()
    return model, metadata


def _validate_export_args(
    opset_version: int,
    batch_sizes: Iterable[int],
    atol: float,
    rtol: float,
) -> Tuple[int, Tuple[int, ...], float, float]:
    if isinstance(opset_version, bool) or not isinstance(opset_version, int) or opset_version < 11:
        raise ValueError("opset_version 必须是大于等于 11 的整数。")
    if isinstance(batch_sizes, (str, bytes)):
        raise ValueError("batch_sizes 必须是正整数序列。")
    normalized_batches = tuple(batch_sizes)
    if not normalized_batches:
        raise ValueError("batch_sizes 不能为空。")
    if any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0
        for size in normalized_batches
    ):
        raise ValueError("batch_sizes 中的每个批大小都必须是正整数。")
    if len(set(normalized_batches)) != len(normalized_batches):
        raise ValueError("batch_sizes 不应重复。")
    for name, tolerance in (("atol", atol), ("rtol", rtol)):
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            raise ValueError(f"{name} 必须是非负有限数。")
        if not math.isfinite(float(tolerance)) or tolerance < 0:
            raise ValueError(f"{name} 必须是非负有限数。")
    return int(opset_version), normalized_batches, float(atol), float(rtol)


def _validate_runtime_signature(session) -> Dict[str, Any]:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if [item.name for item in inputs] != list(INPUT_NAMES):
        raise RuntimeError(
            f"ONNX 输入名称或顺序不符合约定：{[item.name for item in inputs]}。"
        )
    if len(outputs) != 1 or outputs[0].name != OUTPUT_NAME:
        raise RuntimeError("ONNX 必须且只能输出名为 logits 的分类张量。")

    expected_inputs = [
        ["batch", INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]],
        ["batch", INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]],
    ]
    actual_inputs = [item.shape for item in inputs]
    if actual_inputs != expected_inputs:
        raise RuntimeError(
            f"ONNX 双输入维度不符合 [batch,3,120,23]：{actual_inputs}。"
        )
    expected_output_shape = ["batch", NUM_CLASSES]
    if outputs[0].shape != expected_output_shape:
        raise RuntimeError(
            f"ONNX 输出维度不符合 [batch,33]：{outputs[0].shape}。"
        )
    if any(item.type != "tensor(float)" for item in inputs + outputs):
        raise RuntimeError("ONNX 输入和输出都必须是 float32 张量。")

    return {
        "inputs": [
            {"name": item.name, "dtype": "float32", "shape": list(item.shape)}
            for item in inputs
        ],
        "outputs": [
            {"name": item.name, "dtype": "float32", "shape": list(item.shape)}
            for item in outputs
        ],
    }


def _numerical_comparison(
    model: MyoFormerStudent,
    session,
    batch_sizes: Tuple[int, ...],
    atol: float,
    rtol: float,
) -> Tuple[float, Dict[str, Any]]:
    generator = torch.Generator(device="cpu").manual_seed(260927)
    max_absolute_error = 0.0
    comparisons = []
    model.eval()
    with torch.inference_mode():
        for batch_size in batch_sizes:
            joint = torch.randn(
                (batch_size,) + INPUT_SHAPE, generator=generator, dtype=torch.float32
            )
            bone = torch.randn(
                (batch_size,) + INPUT_SHAPE, generator=generator, dtype=torch.float32
            )
            torch_output = model(joint, bone).cpu().numpy()
            runtime_output = session.run(
                [OUTPUT_NAME],
                {
                    INPUT_NAMES[0]: joint.numpy(),
                    INPUT_NAMES[1]: bone.numpy(),
                },
            )[0]
            if runtime_output.shape != torch_output.shape:
                raise RuntimeError(
                    f"ONNX Runtime 输出形状 {runtime_output.shape} 与 PyTorch {torch_output.shape} 不同。"
                )
            if not np.isfinite(runtime_output).all() or not np.isfinite(torch_output).all():
                raise RuntimeError("ONNX 或 PyTorch 输出包含 NaN/Inf。")
            absolute_error = np.abs(torch_output - runtime_output)
            batch_max_error = float(absolute_error.max(initial=0.0))
            max_absolute_error = max(max_absolute_error, batch_max_error)
            if not np.allclose(torch_output, runtime_output, atol=atol, rtol=rtol):
                raise RuntimeError(
                    "PyTorch 与 ONNX Runtime 数值不一致："
                    f"batch={batch_size}, 最大绝对误差={batch_max_error:.8g}, "
                    f"atol={atol}, rtol={rtol}。"
                )
            comparisons.append(
                {"batch_size": batch_size, "max_absolute_error": batch_max_error}
            )
    return max_absolute_error, {
        "status": "passed",
        "runtime": "onnxruntime-CPUExecutionProvider",
        "atol": atol,
        "rtol": rtol,
        "batch_sizes": list(batch_sizes),
        "batches": comparisons,
        "max_absolute_error": max_absolute_error,
    }


def export_student_checkpoint(
    checkpoint_path: os.PathLike | str,
    output_path: os.PathLike | str,
    manifest_path: Optional[os.PathLike | str] = None,
    opset_version: int = DEFAULT_OPSET,
    batch_sizes: Iterable[int] = (1, 4),
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> Dict[str, Any]:
    """严格恢复 checkpoint，完成双运行时核验后才发布模型和清单。"""
    opset_version, batch_sizes, atol, rtol = _validate_export_args(
        opset_version, batch_sizes, atol, rtol
    )
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    manifest = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else output.with_suffix(".manifest.json")
    )
    if output == checkpoint or manifest == checkpoint or output == manifest:
        raise ValueError("checkpoint、ONNX 输出和清单必须是三个不同文件。")
    model, checkpoint_metadata = load_formal_student_checkpoint(checkpoint)
    checkpoint_sha256 = _sha256_file(checkpoint)
    model_sha256, state_dict_sha256 = _sha256_model(
        checkpoint_metadata["student_config"], model.state_dict()
    )
    onnx, onnxruntime = _require_export_dependencies()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    wrapper = StudentLogitsExportWrapper(model).eval().cpu()
    example_joint = torch.zeros((1,) + INPUT_SHAPE, dtype=torch.float32)
    example_bone = torch.zeros((1,) + INPUT_SHAPE, dtype=torch.float32)

    nonce = uuid.uuid4().hex
    temporary_output = output.with_name(f".{output.stem}.{nonce}.tmp.onnx")
    temporary_manifest = manifest.with_name(f".{manifest.name}.{nonce}.tmp")
    try:
        torch.onnx.export(
            wrapper,
            (example_joint, example_bone),
            str(temporary_output),
            export_params=True,
            do_constant_folding=True,
            input_names=list(INPUT_NAMES),
            output_names=[OUTPUT_NAME],
            dynamic_axes={
                INPUT_NAMES[0]: {0: "batch"},
                INPUT_NAMES[1]: {0: "batch"},
                OUTPUT_NAME: {0: "batch"},
            },
            opset_version=opset_version,
            dynamo=False,
        )
        if not temporary_output.is_file() or temporary_output.stat().st_size == 0:
            raise RuntimeError("PyTorch ONNX 导出器没有生成有效模型文件。")

        onnx.checker.check_model(str(temporary_output))
        session = onnxruntime.InferenceSession(
            str(temporary_output), providers=["CPUExecutionProvider"]
        )
        signature = _validate_runtime_signature(session)
        max_absolute_error, verification = _numerical_comparison(
            model, session, batch_sizes, atol, rtol
        )
        onnx_sha256 = _sha256_file(temporary_output)
        model_parameters = sum(parameter.numel() for parameter in model.parameters())

        manifest_content = {
            "schema_version": 1,
            "exported_at_utc": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "model": {
                "class": "MyoFormerStudent",
                "parameters": model_parameters,
                "sha256": model_sha256,
                "state_dict_sha256": state_dict_sha256,
                "config": checkpoint_metadata["student_config"],
                "checkpoint": {
                    "file_name": checkpoint.name,
                    "sha256": checkpoint_sha256,
                    "seed": checkpoint_metadata["seed"],
                    "epoch": checkpoint_metadata["epoch"],
                    "selection_metric": checkpoint_metadata["selection_metric"],
                },
            },
            "artifact": {
                "file_name": output.name,
                "format": "ONNX",
                "sha256": onnx_sha256,
                "opset_version": opset_version,
            },
            "versions": {
                "pytorch": torch.__version__,
                "onnx": onnx.__version__,
                "onnxruntime": onnxruntime.__version__,
            },
            "signature": signature,
            "verification": verification,
        }
        manifest_content["verification"]["max_absolute_error"] = max_absolute_error

        with temporary_manifest.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest_content, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_output, output)
        os.replace(temporary_manifest, manifest)
        return manifest_content
    finally:
        for temp_path in (temporary_output, temporary_manifest):
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="严格恢复 Lite-MyoNet 学生 checkpoint，并导出经过数值核验的 CPU ONNX 模型。"
    )
    parser.add_argument("--checkpoint", required=True, help="正式学生 checkpoint 路径")
    parser.add_argument("--output", required=True, help="输出 .onnx 文件路径")
    parser.add_argument("--manifest", help="结构化 JSON 清单路径；默认与 ONNX 同目录")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET, help="ONNX opset 版本")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4],
        help="用于动态 batch 数值一致性检查的批大小，默认 1 4",
    )
    parser.add_argument("--atol", type=float, default=1e-5, help="绝对误差阈值")
    parser.add_argument("--rtol", type=float, default=1e-4, help="相对误差阈值")
    args = parser.parse_args(argv)
    try:
        result = export_student_checkpoint(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            manifest_path=args.manifest,
            opset_version=args.opset,
            batch_sizes=args.batch_sizes,
            atol=args.atol,
            rtol=args.rtol,
        )
    except (ONNXExportDependencyError, FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
