"""Export a self-contained Lite-MyoNet checkpoint to dynamic-batch ONNX."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


class _JointsOnlyExportWrapper(torch.nn.Module):
    """Expose only joints while fixing optional Python flags for tracing."""

    def __init__(self, model: torch.nn.Module) -> None:
        """Register the restored student as the only exported inference module."""
        super().__init__()
        self.model = model

    def forward(self, joints: torch.Tensor) -> torch.Tensor:
        """Run logits inference without exposing feature-capture options."""
        return self.model(joints, return_features=False)


def _require_onnx() -> None:
    """Require the optional ONNX exporter package with an actionable message."""
    if importlib.util.find_spec("onnx") is None:
        raise RuntimeError(
            "ONNX export requires optional dependencies. Install them with "
            "`pip install -e .[export]`."
        )


def _input_shape(model: torch.nn.Module) -> tuple[int, int, int]:
    """Derive the fixed channel, frame, and point dimensions from the model."""
    channels = getattr(model, "in_channels", None)
    points = getattr(model, "num_points", None)
    skeleton = getattr(model, "skeleton", None)
    frames = getattr(skeleton, "frame_count", None)
    if channels is None or points is None:
        raise ValueError("Restored model does not expose input channel and point dimensions.")
    if frames is None:
        raise ValueError(
            "Checkpoint skeleton must define frame_count; ONNX export will not guess a frame length."
        )
    if isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0:
        raise ValueError("Checkpoint skeleton.frame_count must be a positive integer.")
    shape = (int(channels), int(frames), int(points))
    if any(dimension <= 0 for dimension in shape):
        raise ValueError(f"Invalid model input dimensions: {shape}.")
    return shape


def _validate_batch_sizes(batch_sizes: Sequence[int]) -> tuple[int, ...]:
    """Validate distinct positive batch sizes used for numerical checking."""
    if isinstance(batch_sizes, (str, bytes)) or not isinstance(batch_sizes, Sequence):
        raise TypeError("batch_sizes must be a sequence of positive integers.")
    parsed = []
    for index, value in enumerate(batch_sizes):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"batch_sizes[{index}] must be a positive integer.")
        parsed.append(int(value))
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError("batch_sizes must be non-empty and contain no duplicates.")
    return tuple(parsed)


def _validate_export(
    model: torch.nn.Module,
    onnx_path: Path,
    input_dimensions: tuple[int, int, int],
    batch_sizes: tuple[int, ...],
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Check ONNX structure and compare CPU inference for several batch sizes."""
    if importlib.util.find_spec("onnxruntime") is None:
        raise RuntimeError(
            "ONNX Runtime validation requires the optional export extra: "
            "`pip install -e .[export]`."
        )
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    initializer_names = {initializer.name for initializer in onnx_model.graph.initializer}
    graph_inputs = [item.name for item in onnx_model.graph.input
                    if item.name not in initializer_names]
    graph_outputs = [item.name for item in onnx_model.graph.output]
    if graph_inputs != ["joints"]:
        raise RuntimeError(f"ONNX must expose only the joints input, found {graph_inputs}.")
    if graph_outputs != ["logits"]:
        raise RuntimeError(f"ONNX must expose logits output, found {graph_outputs}.")

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path), sess_options=session_options, providers=["CPUExecutionProvider"]
    )
    input_meta = session.get_inputs()
    output_meta = session.get_outputs()
    if len(input_meta) != 1 or input_meta[0].name != "joints":
        raise RuntimeError("ONNX Runtime did not report the single joints input.")
    if len(output_meta) != 1 or output_meta[0].name != "logits":
        raise RuntimeError("ONNX Runtime did not report the logits output.")
    if isinstance(input_meta[0].shape[0], int) or isinstance(output_meta[0].shape[0], int):
        raise RuntimeError("ONNX batch dimensions must remain dynamic.")

    max_absolute_error = 0.0
    checked_shapes = []
    with torch.inference_mode():
        for batch_size in batch_sizes:
            sample = torch.randn((batch_size, *input_dimensions), dtype=torch.float32)
            expected = model(sample).detach().cpu().numpy()
            actual = session.run(["logits"], {"joints": sample.numpy()})[0]
            if actual.shape != expected.shape or actual.shape[0] != batch_size:
                raise RuntimeError(
                    f"ONNX output shape {actual.shape} does not match PyTorch {expected.shape}."
                )
            if not np.allclose(actual, expected, atol=atol, rtol=rtol):
                difference = float(np.max(np.abs(actual - expected)))
                raise RuntimeError(
                    f"ONNX output differs from PyTorch at batch={batch_size}; "
                    f"max_abs_error={difference:.8g}."
                )
            max_absolute_error = max(
                max_absolute_error, float(np.max(np.abs(actual - expected)))
            )
            checked_shapes.append(list(actual.shape))
    return {
        "validated": True,
        "provider": "CPUExecutionProvider",
        "batch_sizes": list(batch_sizes),
        "output_shapes": checked_shapes,
        "max_absolute_error": max_absolute_error,
    }


def export_checkpoint_to_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    validate: bool = True,
    batch_sizes: Sequence[int] = (1, 2, 4),
    atol: float = 1e-4,
    rtol: float = 1e-3,
    overwrite: bool = False,
) -> dict[str, object]:
    """Export a restored model with joints-only input and dynamic batch axes.

    The model's forward method derives bone vectors internally, so the exported
    graph accepts no precomputed or secondary modality input.
    """
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if destination.suffix.lower() != ".onnx":
        raise ValueError("output_path must use the .onnx extension.")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"ONNX output already exists: {destination}")
    if not isinstance(validate, bool) or not isinstance(overwrite, bool):
        raise TypeError("validate and overwrite must be booleans.")
    if (
        isinstance(atol, bool) or isinstance(rtol, bool)
        or not isinstance(atol, (int, float)) or not isinstance(rtol, (int, float))
        or atol < 0 or rtol < 0
    ):
        raise ValueError("atol and rtol must be non-negative numbers.")
    batches = _validate_batch_sizes(batch_sizes) if validate else tuple()
    _require_onnx()

    from ..training.engine import restore_model

    model, payload = restore_model(checkpoint, device="cpu")
    if payload.get("model_type") != "student":
        raise ValueError("ONNX export accepts only the complete Fourier student checkpoint.")
    model = model.to(device="cpu", dtype=torch.float32).eval()
    export_model = _JointsOnlyExportWrapper(model).eval()
    input_dimensions = _input_shape(model)
    dummy = torch.zeros((1, *input_dimensions), dtype=torch.float32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}.", suffix=".onnx", dir=destination.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        export_options = {
            "export_params": True,
            "opset_version": 17,
            "do_constant_folding": True,
            "input_names": ["joints"],
            "output_names": ["logits"],
            "dynamic_axes": {
                "joints": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        }
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            export_options["dynamo"] = False
        torch.onnx.export(export_model, dummy, str(temporary), **export_options)
        validation = (
            _validate_export(model, temporary, input_dimensions, batches, float(atol), float(rtol))
            if validate else {"validated": False, "reason": "validation disabled by caller"}
        )
        if destination.exists() and not overwrite:
            raise FileExistsError(f"ONNX output already exists: {destination}")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "checkpoint": str(checkpoint),
        "output_path": str(destination),
        "input_name": "joints",
        "input_shape": [None, *input_dimensions],
        "output_name": "logits",
        "fourier_degree": getattr(model, "fourier_degree", None),
        **validation,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the ONNX export command-line parser."""
    parser = argparse.ArgumentParser(description="Export a Lite-MyoNet checkpoint to ONNX.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Best .pth checkpoint.")
    parser.add_argument("--output", required=True, type=Path, help="Destination .onnx file.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip CPU ONNX Runtime comparison (onnx is still required).")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4],
                        help="Batch sizes used by CPU numerical validation.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run ONNX export from the public command-line interface."""
    args = build_parser().parse_args(argv)
    result = export_checkpoint_to_onnx(
        args.checkpoint,
        args.output,
        validate=not args.no_validate,
        batch_sizes=args.batch_sizes,
        overwrite=args.overwrite,
    )
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
