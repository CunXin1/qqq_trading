"""Tests for path dependency features."""
import numpy as np
import pandas as pd


def test_path_features_smoke(sample_daily_metrics, sample_external_data):
    """Smoke test: path features can be computed without errors."""
    from qqq_trading.features.base import engineer_base_features
    from qqq_trading.features.external import engineer_all_external
    from qqq_trading.features.interactions import build_interaction_features
    from qqq_trading.features.path import build_path_features

    df = engineer_base_features(sample_daily_metrics)
    df = engineer_all_external(df, sample_external_data)
    df = build_interaction_features(df)

    # Path features need enough data for 63-day window
    # With only 50 rows, most will be NaN, but should not error
    df = build_path_features(df)

    assert "trend_r2_63d" in df.columns
    assert "fractal_eff_63d" in df.columns
    assert "choppiness_63d" in df.columns
    assert "hurst_63d" in df.columns
