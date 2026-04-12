"""Tests for model evaluation module."""
import numpy as np
from qqq_trading.models.evaluation import (
    evaluate_model, backtest_thresholds, find_optimal_threshold,
)


def test_evaluate_model():
    np.random.seed(42)
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_proba = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4, 0.15, 0.9, 0.25])

    metrics = evaluate_model(y_true, y_proba)
    assert "auc" in metrics
    assert "ap" in metrics
    assert "brier" in metrics
    assert 0 <= metrics["auc"] <= 1
    assert metrics["n_samples"] == 10
    assert metrics["n_positive"] == 4


def test_backtest_thresholds():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_proba = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4, 0.15, 0.9, 0.25])

    bt = backtest_thresholds(y_true, y_proba, [0.3, 0.5, 0.7])
    assert len(bt) == 3
    assert "threshold" in bt.columns
    assert "alerts" in bt.columns
    assert "hit_rate" in bt.columns
    assert "lift" in bt.columns

    # Higher threshold should have fewer alerts
    assert bt.iloc[0]["alerts"] >= bt.iloc[1]["alerts"]


def test_find_optimal_threshold():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.6])

    thresh = find_optimal_threshold(y_true, y_proba)
    assert 0.1 <= thresh <= 0.9
