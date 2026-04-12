"""External feature engineering: VRP, VIX dynamics, rates, events."""
from __future__ import annotations

import pandas as pd
import numpy as np

from qqq_trading.data.event_calendar import (
    load_fomc_dates, compute_nfp_dates, _compute_eve_dates,
    compute_days_to_event, compute_earnings_season,
)


def engineer_vrp_features(df: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    """Volatility Risk Premium = Implied Vol (VIX) - Realized Vol."""
    vix = ext["vix_close"]

    rv20 = df.get("realized_vol_20d", df["close_to_close_ret"].rolling(20).std().shift(1) * np.sqrt(252))
    rv10 = df.get("realized_vol_10d", df["close_to_close_ret"].rolling(10).std().shift(1) * np.sqrt(252))
    rv5 = df.get("realized_vol_5d", df["close_to_close_ret"].rolling(5).std().shift(1) * np.sqrt(252))

    df["vrp_20d"] = (vix.shift(1) / 100) - rv20
    df["vrp_10d"] = (vix.shift(1) / 100) - rv10
    df["vrp_5d"] = (vix.shift(1) / 100) - rv5

    df["vrp_20d_zscore"] = (
        (df["vrp_20d"] - df["vrp_20d"].rolling(60).mean())
        / df["vrp_20d"].rolling(60).std()
    )
    df["vrp_20d_change_5d"] = df["vrp_20d"] - df["vrp_20d"].shift(5)

    return df


def engineer_vix_dynamics(df: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    """VIX changes and spikes (not raw levels)."""
    vix = ext["vix_close"]

    df["vix_pct_change_1d"] = vix.pct_change().shift(1)
    df["vix_pct_change_5d"] = vix.pct_change(5).shift(1)
    df["vix_range_1d"] = ((ext["vix_high"] - ext["vix_low"]) / vix).shift(1)
    df["vix_spike"] = (vix.pct_change().shift(1) > 0.10).astype(int)

    return df


def engineer_vvix_features(df: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    """VVIX features (available from 2007, NaN-tolerant)."""
    vvix = ext["vvix_close"]
    vix = ext["vix_close"]

    df["vvix_vix_ratio"] = (vvix / vix).shift(1)
    df["vvix_change_1d"] = vvix.pct_change().shift(1)

    return df


def engineer_rate_features(df: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    """Interest rate and yield curve features."""
    tnx = ext["tnx_10y_close"]
    irx = ext["irx_3m_close"]

    df["yield_curve_slope"] = (tnx - irx).shift(1)
    df["yield_curve_inverted"] = ((tnx - irx) < 0).astype(int).shift(1)
    df["yield_10y_change_1d"] = tnx.diff().shift(1)
    df["yield_10y_vol_20d"] = tnx.diff().rolling(20).std().shift(1)
    df["rate_shock"] = (
        tnx.diff().abs().shift(1) > tnx.diff().rolling(20).std().shift(1) * 2
    ).astype(int)

    return df


def engineer_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """FOMC, NFP, earnings season, macro event flags."""
    fomc_dates = load_fomc_dates()
    nfp_dates = compute_nfp_dates()

    # FOMC
    df["is_fomc_day"] = df.index.isin(fomc_dates).astype(int)
    fomc_eve = _compute_eve_dates(fomc_dates)
    df["is_fomc_eve"] = df.index.isin(fomc_eve).astype(int)
    df["days_to_fomc"] = compute_days_to_event(df.index, fomc_dates)
    df["fomc_week"] = (df["days_to_fomc"] <= 5).astype(int)

    # NFP
    df["is_nfp_day"] = df.index.isin(nfp_dates).astype(int)
    nfp_eve = _compute_eve_dates(nfp_dates)
    df["is_nfp_eve"] = df.index.isin(nfp_eve).astype(int)
    df["days_to_nfp"] = compute_days_to_event(df.index, nfp_dates, default=30)

    # Combined
    df["is_macro_event_day"] = ((df["is_fomc_day"] == 1) | (df["is_nfp_day"] == 1)).astype(int)
    df["is_macro_event_eve"] = ((df["is_fomc_eve"] == 1) | (df["is_nfp_eve"] == 1)).astype(int)

    # Earnings season
    df["is_earnings_season"] = compute_earnings_season(df.index)

    return df


def engineer_all_external(df: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    """Run full external feature engineering pipeline."""
    ext_aligned = ext.reindex(df.index).ffill()
    df = engineer_vrp_features(df, ext_aligned)
    df = engineer_vix_dynamics(df, ext_aligned)
    df = engineer_vvix_features(df, ext_aligned)
    df = engineer_rate_features(df, ext_aligned)
    df = engineer_event_features(df)
    return df
