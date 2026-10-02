"""Check public model imports and removal of legacy model names."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import lite_myonet.models as models
from lite_myonet.models import LiteMyoNet, MyoFormer
from lite_myonet.models import lite_myonet as lite_myonet_module
from lite_myonet.models import myoformer as myoformer_module


def test_public_models_are_exported_from_the_models_package():
    """Expose both primary model classes from the stable package API."""
    assert models.MyoFormer is MyoFormer
    assert models.LiteMyoNet is LiteMyoNet
    assert MyoFormer is myoformer_module.MyoFormer
    assert LiteMyoNet is lite_myonet_module.LiteMyoNet


def test_legacy_model_class_names_are_not_exported():
    """Keep removed teacher and student class names out of the public API."""
    assert not hasattr(models, "MyoFormerTeacher")
    assert not hasattr(models, "MyoFormerStudent")
    assert not hasattr(myoformer_module, "MyoFormerTeacher")
    assert not hasattr(lite_myonet_module, "MyoFormerStudent")


@pytest.mark.parametrize(
    "module_name",
    ("lite_myonet.models.teacher", "lite_myonet.models.student"),
)
def test_legacy_model_modules_are_removed(module_name):
    """Reject imports from the former public model module paths."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize(
    "statement",
    (
        "import lite_myonet.data.bones as bones; assert callable(bones.joints_to_bones)",
        "from lite_myonet.models.skeleton import load_skeleton; assert callable(load_skeleton)",
        "from lite_myonet.models import MyoFormer, LiteMyoNet; assert MyoFormer and LiteMyoNet",
    ),
)
def test_public_imports_succeed_in_fresh_processes(statement):
    """Verify each public import path independently in a clean interpreter."""
    environment = os.environ.copy()
    source_root = str(PROJECT_ROOT / "src")
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root
        if not existing_python_path
        else source_root + os.pathsep + existing_python_path
    )
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    subprocess.run(
        [sys.executable, "-c", statement],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
