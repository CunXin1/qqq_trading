"""Tests for prediction module."""
from models.prediction import (
    _classify_signal, build_features_for_prediction,
)
from features.registry import get_full_features


def test_classify_signal():
    assert _classify_signal(0.8, 0.5) == "HIGH"
    assert _classify_signal(0.6, 0.5) == "ELEVATED"
    assert _classify_signal(0.45, 0.5) == "MODERATE"
    assert _classify_signal(0.2, 0.5) == "LOW"


def test_build_features(sample_daily_metrics, sample_external_data):
    df = build_features_for_prediction(
        sample_daily_metrics, sample_external_data, include_interactions=True
    )
    # Should have all base + external + interaction features
    expected = get_full_features(include_interactions=True)
    present = [f for f in expected if f in df.columns]
    # Most features should be present (some may have NaN but columns exist)
    assert len(present) >= len(expected) * 0.9
