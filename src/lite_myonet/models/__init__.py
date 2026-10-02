"""Public MyoFormer teacher and Lite-MyoNet student models."""

from typing import Any

__all__ = ["MyoFormer", "LiteMyoNet"]


def __getattr__(name: str) -> Any:
    """Load public model classes lazily to avoid data and model import cycles."""
    if name == "MyoFormer":
        from .myoformer import MyoFormer

        return MyoFormer
    if name == "LiteMyoNet":
        from .lite_myonet import LiteMyoNet

        return LiteMyoNet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
