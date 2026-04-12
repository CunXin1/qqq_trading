"""Tests for interaction feature engineering."""
from qqq_trading.features.interactions import build_interaction_features
from qqq_trading.features.registry import get_interaction_features


def test_interaction_features_created(sample_daily_metrics, sample_external_data):
    from qqq_trading.features.base import engineer_base_features
    from qqq_trading.features.external import engineer_all_external

    df = engineer_base_features(sample_daily_metrics)
    df = engineer_all_external(df, sample_external_data)
    df = build_interaction_features(df)

    expected = get_interaction_features()
    for feat in expected:
        assert feat in df.columns, f"Missing interaction feature: {feat}"


def test_binary_flags_are_01(sample_daily_metrics, sample_external_data):
    from qqq_trading.features.base import engineer_base_features
    from qqq_trading.features.external import engineer_all_external

    df = engineer_base_features(sample_daily_metrics)
    df = engineer_all_external(df, sample_external_data)
    df = build_interaction_features(df)

    binary_flags = ["vrp_high", "vrp_extreme", "vrp_positive", "vrp_negative",
                    "high_vol_regime", "low_vol_regime", "fomc_imminent"]
    for flag in binary_flags:
        vals = set(df[flag].dropna().unique())
        assert vals.issubset({0, 1}), f"{flag} has values: {vals}"


def test_cross_features_are_product(sample_daily_metrics, sample_external_data):
    from qqq_trading.features.base import engineer_base_features
    from qqq_trading.features.external import engineer_all_external

    df = engineer_base_features(sample_daily_metrics)
    df = engineer_all_external(df, sample_external_data)
    df = build_interaction_features(df)

    # Cross should be product of parents
    cross = df["vrp_high_X_fomc_imminent"].dropna()
    parent1 = df["vrp_high"].loc[cross.index]
    parent2 = df["fomc_imminent"].loc[cross.index]
    expected = parent1 * parent2
    assert (cross == expected).all()
