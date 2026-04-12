"""Model evaluation metrics and backtest utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
)


def evaluate_model(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """Compute standard classification metrics."""
    return {
        "auc": roc_auc_score(y_true, y_proba),
        "ap": average_precision_score(y_true, y_proba),
        "brier": brier_score_loss(y_true, y_proba),
        "base_rate": y_true.mean(),
        "n_samples": len(y_true),
        "n_positive": int(y_true.sum()),
    }


def backtest_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Compute hit rate, coverage, lift for each confidence threshold."""
    if thresholds is None:
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    rows = []
    base_rate = y_true.mean()
    total_pos = y_true.sum()

    for t in thresholds:
        mask = y_proba >= t
        n_alerts = mask.sum()
        if n_alerts > 0:
            hits = y_true[mask].sum()
            hit_rate = y_true[mask].mean()
            coverage = hits / total_pos if total_pos > 0 else 0
            lift = hit_rate / base_rate if base_rate > 0 else 0
        else:
            hits = 0
            hit_rate = 0
            coverage = 0
            lift = 0

        rows.append({
            "threshold": t,
            "alerts": int(n_alerts),
            "hits": int(hits),
            "hit_rate": hit_rate,
            "coverage": coverage,
            "lift": lift,
        })

    return pd.DataFrame(rows)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
) -> float:
    """Find threshold that maximizes F1 on validation set."""
    best_thresh = 0.5
    best_score = 0

    for t in np.arange(0.1, 0.9, 0.01):
        pred = (y_proba >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        if f1 > best_score:
            best_score = f1
            best_thresh = t

    return best_thresh


def print_classification_report(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> str:
    """Print sklearn classification report at given threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    return classification_report(
        y_true, y_pred,
        target_names=["No Large Move", "Large Move"],
    )
