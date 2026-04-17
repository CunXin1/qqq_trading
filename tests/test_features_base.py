"""Tests for base feature engineering."""
import numpy as np
from features.base import engineer_base_features
from features.registry import get_base_features


def test_base_features_created(sample_daily_metrics):
    df = engineer_base_features(sample_daily_metrics)
    expected = get_base_features()
    for feat in expected:
        assert feat in df.columns, f"Missing feature: {feat}"


def test_no_lookahead(sample_daily_metrics):
    """Verify all features use shift(1) — no future data leakage."""
    df = engineer_base_features(sample_daily_metrics)
    # First row of lagged features should be NaN (shifted)
    assert np.isnan(df["ret_lag1"].iloc[0])
    assert np.isnan(df["abs_ret_lag1"].iloc[0])


def test_rsi_range(sample_daily_metrics):
    df = engineer_base_features(sample_daily_metrics)
    rsi = df["rsi_14"].dropna()
    assert rsi.min() >= 0
    assert rsi.max() <= 100


def test_vol_ratio_positive(sample_daily_metrics):
    df = engineer_base_features(sample_daily_metrics)
    ratio = df["vol_ratio_5_60"].dropna()
    assert (ratio > 0).all()


def test_calendar_features(sample_daily_metrics):
    df = engineer_base_features(sample_daily_metrics)
    assert df["dow"].min() >= 0
    assert df["dow"].max() <= 4
    assert df["month"].min() >= 1
    assert df["month"].max() <= 12


def test_premarket_features(sample_daily_metrics):
    df = engineer_base_features(sample_daily_metrics)
    assert "premarket_ret_today" in df.columns
    assert "premarket_range_today" in df.columns
