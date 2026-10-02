"""Tests for classification metric serialization."""

from __future__ import annotations

from lite_myonet.evaluation import classification_metrics


def test_classification_metrics_include_per_class_support() -> None:
    """Return aggregate metrics and fixed-length per-class values."""
    metrics = classification_metrics([0, 0, 1, 2], [0, 1, 1, 2], num_classes=3)
    assert metrics["accuracy"] == 0.75
    assert metrics["per_class_support"] == [2, 1, 1]
    assert len(metrics["per_class_recall"]) == 3
