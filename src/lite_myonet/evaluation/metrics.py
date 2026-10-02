"""Classification metrics used by teacher and student evaluation."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def classification_metrics(
    targets: Sequence[int], predictions: Sequence[int], num_classes: int
) -> dict[str, object]:
    """Compute aggregate and per-class classification metrics."""
    target_array = np.asarray(targets, dtype=np.int64)
    prediction_array = np.asarray(predictions, dtype=np.int64)
    if target_array.shape != prediction_array.shape or target_array.ndim != 1:
        raise ValueError("Targets and predictions must be one-dimensional arrays of equal size.")
    labels = np.arange(num_classes, dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        target_array,
        prediction_array,
        labels=labels,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(target_array, prediction_array)),
        "balanced_accuracy": float(
            balanced_accuracy_score(target_array, prediction_array)
        ),
        "macro_f1": float(
            f1_score(
                target_array,
                prediction_array,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.astype(int).tolist(),
        "confusion_matrix": confusion_matrix(
            target_array, prediction_array, labels=labels
        ).astype(int).tolist(),
    }
