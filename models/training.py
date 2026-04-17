"""Unified model training with configurable hyperparameters."""
from __future__ import annotations

from typing import Literal, Optional
import numpy as np
import joblib
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

from config import ModelConfig


def create_model(
    model_type: Literal["xgboost", "lightgbm", "random_forest"] = "xgboost",
    config: Optional[ModelConfig] = None,
    pos_weight: float = 1.0,
    random_state: int = 42,
):
    """Create a classifier with given config."""
    if config is None:
        config = ModelConfig()

    if model_type == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            scale_pos_weight=pos_weight,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            reg_alpha=config.reg_alpha,
            reg_lambda=config.reg_lambda,
            random_state=random_state,
            verbosity=0,
            n_jobs=-1,
        )
    elif model_type == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            is_unbalance=True,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            reg_alpha=config.reg_alpha,
            reg_lambda=config.reg_lambda,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
        )
    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=min(config.max_depth + 3, 10),
            min_samples_leaf=15,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def compute_pos_weight(y: np.ndarray) -> float:
    """Compute positive class weight for imbalanced data."""
    mean = y.mean()
    return max(1.0, (1 - mean) / mean) if mean > 0 else 1.0


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: Literal["xgboost", "lightgbm", "random_forest"] = "xgboost",
    config: Optional[ModelConfig] = None,
    random_state: int = 42,
):
    """Train a model and return the fitted estimator."""
    pos_weight = compute_pos_weight(y_train)
    model = create_model(model_type, config, pos_weight, random_state)

    # RandomForest can't handle NaN
    if model_type == "random_forest":
        X_fit = np.nan_to_num(X_train, nan=-999)
    else:
        X_fit = X_train

    model.fit(X_fit, y_train)
    return model


def save_model(model, feature_cols: list[str], path: Path) -> None:
    """Save model and feature list."""
    joblib.dump(model, path)
    feat_path = path.with_suffix(".csv")
    pd.Series(feature_cols).to_csv(feat_path, index=False, header=["feature"])


def load_model(path: Path):
    """Load model and feature list.

    Handles both old-format CSV (header='0') and new-format (header='feature').
    Also checks for the naming convention: model.joblib -> model_feature_columns.csv
    """
    model = joblib.load(path)

    # Try multiple CSV naming conventions
    candidates = [
        path.with_suffix(".csv"),                                        # new: same stem
        path.parent / f"{path.stem}_feature_columns.csv",                # pattern: X_feature_columns.csv
        path.parent / "interaction_feature_columns.csv",                 # legacy name
        path.parent / "feature_columns.csv",                             # legacy base
    ]

    feature_cols = None
    for feat_path in candidates:
        if feat_path.exists():
            df_feat = pd.read_csv(feat_path)
            col = df_feat.columns[0]  # use whatever the first column is named
            feature_cols = df_feat[col].tolist()
            break

    return model, feature_cols
