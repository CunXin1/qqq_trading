"""Interaction features: VRP regime x catalyst cross signals."""
from __future__ import annotations

import pandas as pd
import numpy as np


def build_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build cross features: VRP state x catalyst proximity."""
    out = df.copy()

    # ── VRP regime buckets ──
    vrp = out["vrp_20d"]
    out["vrp_high"] = (vrp > vrp.quantile(0.75)).astype(int)
    out["vrp_extreme"] = (vrp > vrp.quantile(0.90)).astype(int)
    out["vrp_positive"] = (vrp > 0).astype(int)
    out["vrp_negative"] = (vrp < -0.05).astype(int)

    # ── Vol regime ──
    rv20 = out["realized_vol_20d"]
    out["high_vol_regime"] = (rv20 > rv20.quantile(0.75)).astype(int)
    out["low_vol_regime"] = (rv20 < rv20.quantile(0.25)).astype(int)

    # ── Catalyst proximity flags ──
    out["fomc_imminent"] = (out["days_to_fomc"] <= 1).astype(int)
    out["fomc_this_week"] = (out["days_to_fomc"] <= 5).astype(int)
    out["nfp_imminent"] = (out["days_to_nfp"] <= 1).astype(int)
    out["nfp_this_week"] = (out["days_to_nfp"] <= 5).astype(int)
    out["any_catalyst_imminent"] = (
        (out["fomc_imminent"] == 1)
        | (out["nfp_imminent"] == 1)
        | (out["is_earnings_season"] == 1)
    ).astype(int)

    # ── Recent vol spikes ──
    out["vix_spiked_3d"] = out["vix_spike"].rolling(3).max().fillna(0).astype(int)
    out["big_move_recent_3d"] = (
        out["abs_close_to_close_gt_2pct"].rolling(3).max().shift(1).fillna(0).astype(int)
    )

    # ══ CROSS FEATURES: regime x catalyst ══

    # VRP x Event
    out["vrp_high_X_fomc_imminent"] = out["vrp_high"] * out["fomc_imminent"]
    out["vrp_high_X_nfp_imminent"] = out["vrp_high"] * out["nfp_imminent"]
    out["vrp_high_X_earnings"] = out["vrp_high"] * out["is_earnings_season"]
    out["vrp_high_X_any_catalyst"] = out["vrp_high"] * out["any_catalyst_imminent"]
    out["vrp_extreme_X_fomc_imminent"] = out["vrp_extreme"] * out["fomc_imminent"]
    out["vrp_extreme_X_any_catalyst"] = out["vrp_extreme"] * out["any_catalyst_imminent"]

    # VRP positive x Event
    out["vrp_pos_X_fomc"] = out["vrp_positive"] * out["fomc_imminent"]
    out["vrp_pos_X_nfp"] = out["vrp_positive"] * out["nfp_imminent"]
    out["vrp_pos_X_earnings"] = out["vrp_positive"] * out["is_earnings_season"]

    # High vol x Event
    out["highvol_X_fomc"] = out["high_vol_regime"] * out["fomc_imminent"]
    out["highvol_X_nfp"] = out["high_vol_regime"] * out["nfp_imminent"]
    out["highvol_X_earnings"] = out["high_vol_regime"] * out["is_earnings_season"]
    out["highvol_X_any_catalyst"] = out["high_vol_regime"] * out["any_catalyst_imminent"]

    # Gamma trap (complacent x Event)
    out["complacent_X_fomc"] = out["vrp_negative"] * out["fomc_imminent"]
    out["complacent_X_nfp"] = out["vrp_negative"] * out["nfp_imminent"]
    out["lowvol_X_fomc"] = out["low_vol_regime"] * out["fomc_imminent"]
    out["lowvol_X_any_catalyst"] = out["low_vol_regime"] * out["any_catalyst_imminent"]

    # Momentum x Event
    out["vix_spike3d_X_fomc"] = out["vix_spiked_3d"] * out["fomc_imminent"]
    out["vix_spike3d_X_any_catalyst"] = out["vix_spiked_3d"] * out["any_catalyst_imminent"]
    out["big_move_3d_X_fomc"] = out["big_move_recent_3d"] * out["fomc_imminent"]
    out["big_move_3d_X_any_catalyst"] = out["big_move_recent_3d"] * out["any_catalyst_imminent"]
    out["big_move_3d_X_earnings"] = out["big_move_recent_3d"] * out["is_earnings_season"]

    # Rate x Event
    out["rate_shock_X_fomc"] = out["rate_shock"] * out["fomc_imminent"]

    # Continuous interactions
    out["vrp_X_fomc_urgency"] = out["vrp_20d"] * (1 / (out["days_to_fomc"] + 1))
    out["vrp_X_nfp_urgency"] = out["vrp_20d"] * (1 / (out["days_to_nfp"] + 1))
    out["vrp_zscore_X_any_catalyst"] = out["vrp_20d_zscore"] * out["any_catalyst_imminent"]
    out["vrp_zscore_X_fomc"] = out["vrp_20d_zscore"] * out["fomc_imminent"]

    # Yield curve x Event
    out["curve_inverted_X_fomc"] = out["yield_curve_inverted"] * out["fomc_imminent"]
    out["curve_inverted_X_any_catalyst"] = out["yield_curve_inverted"] * out["any_catalyst_imminent"]

    return out
