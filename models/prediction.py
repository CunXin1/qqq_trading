"""Inference pipeline: load model and predict."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from models.training import load_model
from features.base import engineer_base_features
from features.external import engineer_all_external
from features.interactions import build_interaction_features
from features.registry import get_full_features, get_base_features


@dataclass
class PredictionResult:
    date: str
    target: str
    probability: float
    signal: str
    threshold: float
    model_name: str


def _classify_signal(prob: float, threshold: float) -> str:
    if prob >= threshold + 0.2:
        return "HIGH"
    elif prob >= threshold:
        return "ELEVATED"
    elif prob >= threshold - 0.1:
        return "MODERATE"
    return "LOW"


def build_features_for_prediction(
    daily_metrics: pd.DataFrame,
    external_data: pd.DataFrame,
    include_interactions: bool = True,
) -> pd.DataFrame:
    """Run the full feature pipeline for inference."""
    df = engineer_base_features(daily_metrics)
    df = engineer_all_external(df, external_data)
    if include_interactions:
        df = build_interaction_features(df)
    return df


def predict(
    model_path: Path,
    daily_metrics: pd.DataFrame,
    external_data: pd.DataFrame,
    threshold: float = 0.5,
    target_name: str = "1DTE |C2C|>2%",
) -> PredictionResult:
    """Load model and make prediction for the latest date."""
    model, feature_cols = load_model(model_path)

    df = build_features_for_prediction(daily_metrics, external_data)

    # Use the latest row
    latest = df.iloc[[-1]]
    date_str = str(latest.index[0].date())

    # Filter to available features
    available = [f for f in feature_cols if f in latest.columns]
    X = latest[available].values

    proba = model.predict_proba(X)[:, 1][0]
    signal = _classify_signal(proba, threshold)

    return PredictionResult(
        date=date_str,
        target=target_name,
        probability=float(proba),
        signal=signal,
        threshold=threshold,
        model_name=model_path.stem,
    )
