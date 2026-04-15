from __future__ import annotations

from sklearn.metrics import roc_auc_score


def binary_classification_metrics(y_true, y_score) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }
