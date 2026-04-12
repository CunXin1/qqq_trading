"""Tests for external feature engineering."""
import numpy as np
from qqq_trading.features.external import (
    engineer_vrp_features, engineer_vix_dynamics,
    engineer_vvix_features, engineer_rate_features,
    engineer_all_external,
)
from qqq_trading.features.registry import get_refined_external_features


def test_vrp_features(sample_daily_metrics, sample_external_data):
    # Need realized vol first
    from qqq_trading.features.base import engineer_base_features
    df = engineer_base_features(sample_daily_metrics)
    ext = sample_external_data.reindex(df.index, method="ffill")
    df = engineer_vrp_features(df, ext)

    assert "vrp_20d" in df.columns
    assert "vrp_10d" in df.columns
    assert "vrp_5d" in df.columns
    assert "vrp_20d_zscore" in df.columns


def test_vix_dynamics(sample_daily_metrics, sample_external_data):
    from qqq_trading.features.base import engineer_base_features
    df = engineer_base_features(sample_daily_metrics)
    ext = sample_external_data.reindex(df.index, method="ffill")
    df = engineer_vix_dynamics(df, ext)

    assert "vix_pct_change_1d" in df.columns
    assert "vix_spike" in df.columns
    # vix_spike should be 0 or 1
    assert set(df["vix_spike"].dropna().unique()).issubset({0, 1})


def test_rate_features(sample_daily_metrics, sample_external_data):
    from qqq_trading.features.base import engineer_base_features
    df = engineer_base_features(sample_daily_metrics)
    ext = sample_external_data.reindex(df.index, method="ffill")
    df = engineer_rate_features(df, ext)

    assert "yield_curve_slope" in df.columns
    assert "yield_curve_inverted" in df.columns
    assert set(df["yield_curve_inverted"].dropna().unique()).issubset({0, 1})


def test_all_external_features(sample_daily_metrics, sample_external_data):
    from qqq_trading.features.base import engineer_base_features
    df = engineer_base_features(sample_daily_metrics)
    df = engineer_all_external(df, sample_external_data)

    expected = get_refined_external_features()
    for feat in expected:
        assert feat in df.columns, f"Missing external feature: {feat}"
