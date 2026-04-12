"""Tests for model training module."""
import numpy as np
import tempfile
from pathlib import Path
from qqq_trading.models.training import (
    create_model, train_model, compute_pos_weight,
    save_model, load_model,
)
from qqq_trading.config import ModelConfig


def test_create_xgboost():
    model = create_model("xgboost")
    assert hasattr(model, "fit")
    assert hasattr(model, "predict_proba")


def test_create_lightgbm():
    model = create_model("lightgbm")
    assert hasattr(model, "fit")


def test_create_random_forest():
    model = create_model("random_forest")
    assert hasattr(model, "fit")


def test_compute_pos_weight():
    y = np.array([0, 0, 0, 0, 1])
    w = compute_pos_weight(y)
    assert w == 4.0

    y_balanced = np.array([0, 1, 0, 1])
    w2 = compute_pos_weight(y_balanced)
    assert w2 == 1.0


def test_train_model():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)
    model = train_model(X, y, "xgboost")
    proba = model.predict_proba(X)[:, 1]
    assert len(proba) == 100
    assert proba.min() >= 0
    assert proba.max() <= 1


def test_save_load_model():
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = (X[:, 0] > 0).astype(int)
    model = train_model(X, y, "xgboost")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_model.joblib"
        features = ["feat_a", "feat_b", "feat_c"]
        save_model(model, features, path)

        loaded_model, loaded_features = load_model(path)
        assert loaded_features == features
        # Predictions should match
        orig_pred = model.predict_proba(X)[:, 1]
        loaded_pred = loaded_model.predict_proba(X)[:, 1]
        np.testing.assert_array_almost_equal(orig_pred, loaded_pred)
