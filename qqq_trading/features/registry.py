"""Canonical feature lists — single source of truth.

Every script that needs a feature list imports from here.
"""
from __future__ import annotations


def get_base_features(include_premarket: bool = False) -> list[str]:
    """53 base features from QQQ price/volume only."""
    features = []

    # Lagged returns (15)
    for lag in range(1, 6):
        features += [f"ret_lag{lag}", f"abs_ret_lag{lag}", f"range_lag{lag}"]

    # Rolling stats (12)
    for w in [5, 10, 20, 60]:
        features += [f"mean_abs_ret_{w}d", f"std_ret_{w}d", f"realized_vol_{w}d"]

    # Vol ratios (2)
    features += ["vol_ratio_5_60", "vol_ratio_10_60"]

    # Drawdown/runup lags (6)
    for lag in range(1, 4):
        features += [f"max_dd_lag{lag}", f"max_ru_lag{lag}"]

    # Volume (2)
    features += ["vol_ratio_20d", "vol_trend_5_20"]

    # Calendar (6)
    features += [
        "dow", "month", "is_month_start", "is_month_end",
        "is_opex_week", "days_since_2pct_move",
    ]

    # Technical (8)
    for w in [20, 50, 200]:
        features += [f"dist_from_ma{w}"]
    features += ["rsi_14"]
    for w in [20, 50]:
        features += [f"proximity_{w}d_high", f"proximity_{w}d_low"]

    # Gap (2)
    features += ["gap_ret_lag1", "abs_gap_lag1"]

    if include_premarket:
        features += ["premarket_ret_today", "premarket_range_today", "premarket_vol_ratio"]

    return features


def get_refined_external_features() -> list[str]:
    """26 refined external features — VRP, VIX dynamics, rates, events."""
    return [
        # VRP (5)
        "vrp_20d", "vrp_10d", "vrp_5d", "vrp_20d_zscore", "vrp_20d_change_5d",
        # VIX dynamics (4)
        "vix_pct_change_1d", "vix_pct_change_5d", "vix_range_1d", "vix_spike",
        # VVIX (2)
        "vvix_vix_ratio", "vvix_change_1d",
        # Interest rates (5)
        "yield_curve_slope", "yield_curve_inverted",
        "yield_10y_change_1d", "yield_10y_vol_20d", "rate_shock",
        # Event calendar (10)
        "is_fomc_day", "is_fomc_eve", "days_to_fomc", "fomc_week",
        "is_nfp_day", "is_nfp_eve", "days_to_nfp",
        "is_macro_event_day", "is_macro_event_eve",
        "is_earnings_season",
    ]


def get_interaction_features() -> list[str]:
    """43 interaction features — regime flags + cross signals."""
    return [
        # Regime flags (6)
        "vrp_high", "vrp_extreme", "vrp_positive", "vrp_negative",
        "high_vol_regime", "low_vol_regime",
        # Catalyst flags (7)
        "fomc_imminent", "fomc_this_week", "nfp_imminent", "nfp_this_week",
        "any_catalyst_imminent", "vix_spiked_3d", "big_move_recent_3d",
        # VRP x Event crosses (9)
        "vrp_high_X_fomc_imminent", "vrp_high_X_nfp_imminent",
        "vrp_high_X_earnings", "vrp_high_X_any_catalyst",
        "vrp_extreme_X_fomc_imminent", "vrp_extreme_X_any_catalyst",
        "vrp_pos_X_fomc", "vrp_pos_X_nfp", "vrp_pos_X_earnings",
        # High vol x Event (4)
        "highvol_X_fomc", "highvol_X_nfp", "highvol_X_earnings", "highvol_X_any_catalyst",
        # Gamma trap (4)
        "complacent_X_fomc", "complacent_X_nfp",
        "lowvol_X_fomc", "lowvol_X_any_catalyst",
        # Momentum x Event (5)
        "vix_spike3d_X_fomc", "vix_spike3d_X_any_catalyst",
        "big_move_3d_X_fomc", "big_move_3d_X_any_catalyst", "big_move_3d_X_earnings",
        # Rate x Event (1)
        "rate_shock_X_fomc",
        # Continuous interactions (4)
        "vrp_X_fomc_urgency", "vrp_X_nfp_urgency",
        "vrp_zscore_X_any_catalyst", "vrp_zscore_X_fomc",
        # Yield curve x Event (2)
        "curve_inverted_X_fomc", "curve_inverted_X_any_catalyst",
    ]


def get_path_features() -> list[str]:
    """30 path dependency / smoothness features."""
    features = []
    for w in ["63d", "126d"]:
        features += [
            f"trend_r2_{w}", f"fractal_eff_{w}", f"choppiness_{w}",
            f"hurst_{w}", f"trend_strength_{w}", f"up_day_ratio_{w}",
            f"max_dd_window_{w}",
            f"smooth_trend_{w}", f"very_smooth_{w}", f"choppy_{w}",
            f"smooth_{w}_X_catalyst", f"smooth_{w}_X_fomc",
            f"smooth_{w}_X_vrp_neg", f"smooth_{w}_X_vrp_neg_X_catalyst",
            f"smooth_{w}_X_lowvol",
        ]
    return features


def get_0dte_premarket_features() -> list[str]:
    """Additional premarket features for 0DTE models."""
    return ["premarket_ret_today", "premarket_range_today", "premarket_vol_ratio"]


def get_full_features(
    include_interactions: bool = True,
    include_path: bool = False,
    include_premarket: bool = False,
) -> list[str]:
    """Composite feature list."""
    feats = get_base_features()
    feats += get_refined_external_features()
    if include_interactions:
        feats += get_interaction_features()
    if include_path:
        feats += get_path_features()
    if include_premarket:
        feats += get_0dte_premarket_features()
    return feats
