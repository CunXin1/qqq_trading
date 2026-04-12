"""Base feature engineering from daily QQQ metrics (53 features)."""
from __future__ import annotations

import pandas as pd
import numpy as np


def engineer_base_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Build all base predictive features. All features use only past data (shift(1))."""
    df = daily.copy()

    # ── A. Lagged return features ──
    for lag in range(1, 6):
        df[f"ret_lag{lag}"] = df["close_to_close_ret"].shift(lag)
        df[f"abs_ret_lag{lag}"] = df["abs_close_to_close"].shift(lag)
        df[f"range_lag{lag}"] = df["intraday_range"].shift(lag)

    # Rolling mean absolute return & std
    for w in [5, 10, 20, 60]:
        df[f"mean_abs_ret_{w}d"] = df["abs_close_to_close"].rolling(w).mean().shift(1)
        df[f"std_ret_{w}d"] = df["close_to_close_ret"].rolling(w).std().shift(1)

    # ── B. Volatility features ──
    for w in [5, 10, 20, 60]:
        df[f"realized_vol_{w}d"] = (
            df["close_to_close_ret"].rolling(w).std().shift(1) * np.sqrt(252)
        )

    # Vol regime indicator
    df["vol_ratio_5_60"] = df["realized_vol_5d"] / df["realized_vol_60d"]
    df["vol_ratio_10_60"] = df["realized_vol_10d"] / df["realized_vol_60d"]

    # Max drawdown/runup lags
    for lag in range(1, 4):
        df[f"max_dd_lag{lag}"] = df["max_drawdown"].shift(lag)
        df[f"max_ru_lag{lag}"] = df["max_runup"].shift(lag)

    # ── C. Pre-market features (same-day, available before open) ──
    df["premarket_ret_today"] = df["premarket_ret"]
    df["premarket_range_today"] = df["premarket_range"]
    df["premarket_vol_ratio"] = (
        df["volume_premarket"]
        / df["volume_premarket"].rolling(20).mean().shift(1)
    )

    # ── D. Volume features ──
    df["vol_ratio_20d"] = (
        df["volume_regular"].shift(1)
        / df["volume_regular"].rolling(20).mean().shift(1)
    )
    df["vol_trend_5_20"] = (
        df["volume_regular"].rolling(5).mean().shift(1)
        / df["volume_regular"].rolling(20).mean().shift(1)
    )

    # ── E. Calendar features ──
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_month_start"] = 0
    df["is_month_end"] = 0
    month_periods = df.index.to_period("M")
    for mp in month_periods.unique():
        mask = month_periods == mp
        idx = df.index[mask]
        if len(idx) > 0:
            df.loc[idx[0], "is_month_start"] = 1
            df.loc[idx[-1], "is_month_end"] = 1

    # Options expiration week (3rd Friday ± 2 days)
    df["is_opex_week"] = 0
    for idx in df.index:
        if idx.weekday() == 4 and 15 <= idx.day <= 21:
            week_start = idx - pd.Timedelta(days=4)
            week_end = idx
            mask = (df.index >= week_start) & (df.index <= week_end)
            df.loc[mask, "is_opex_week"] = 1

    # Days since last >2% move
    large_move = df["abs_close_to_close_gt_2pct"].astype(int)
    days_since = []
    count = 0
    for v in large_move:
        if v == 1:
            count = 0
        else:
            count += 1
        days_since.append(count)
    df["days_since_2pct_move"] = days_since
    df["days_since_2pct_move"] = df["days_since_2pct_move"].shift(1)

    # ── F. Technical features ──
    close = df["reg_close"]
    for w in [20, 50, 200]:
        ma = close.rolling(w).mean()
        df[f"dist_from_ma{w}"] = ((close - ma) / ma).shift(1)

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = (100 - 100 / (1 + rs)).shift(1)

    # N-day high/low proximity
    for w in [20, 50]:
        hh = close.rolling(w).max()
        ll = close.rolling(w).min()
        df[f"proximity_{w}d_high"] = ((close - hh) / hh).shift(1)
        df[f"proximity_{w}d_low"] = ((close - ll) / ll).shift(1)

    # Gap features
    df["gap_ret_lag1"] = df["gap_return"].shift(1)
    df["abs_gap_lag1"] = df["gap_return"].abs().shift(1)

    return df
