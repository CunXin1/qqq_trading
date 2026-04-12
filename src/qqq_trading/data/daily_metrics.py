"""Build daily OHLCV metrics from 1-minute QQQ data."""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


def load_1min_data(parquet_path: Path) -> pd.DataFrame:
    """Load adjusted 1-min parquet data."""
    df = pd.read_parquet(parquet_path)
    df.index = pd.to_datetime(df.index)
    return df


def compute_regular_session_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute OHLCV metrics for regular trading hours (9:30-16:00)."""
    regular = df.between_time("09:30", "15:59")
    daily = regular.groupby(regular.index.date).agg(
        reg_open=("open", "first"),
        reg_high=("high", "max"),
        reg_low=("low", "min"),
        reg_close=("close", "last"),
        volume_regular=("volume", "sum"),
    )
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"
    return daily


def compute_premarket_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pre-market metrics (4:00-9:29)."""
    pre = df.between_time("04:00", "09:29")
    if pre.empty:
        return pd.DataFrame()

    daily_pre = pre.groupby(pre.index.date).agg(
        premarket_open=("open", "first"),
        premarket_high=("high", "max"),
        premarket_low=("low", "min"),
        premarket_close=("close", "last"),
        volume_premarket=("volume", "sum"),
    )
    daily_pre.index = pd.to_datetime(daily_pre.index)
    daily_pre.index.name = "date"

    daily_pre["premarket_ret"] = (
        daily_pre["premarket_close"] / daily_pre["premarket_open"] - 1
    )
    daily_pre["premarket_range"] = (
        (daily_pre["premarket_high"] - daily_pre["premarket_low"])
        / daily_pre["premarket_open"]
    )
    return daily_pre


def compute_full_day_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute full-day high/low across all sessions."""
    daily_full = df.groupby(df.index.date).agg(
        full_high=("high", "max"),
        full_low=("low", "min"),
    )
    daily_full.index = pd.to_datetime(daily_full.index)
    daily_full.index.name = "date"
    return daily_full


def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily VWAP for regular session."""
    regular = df.between_time("09:30", "15:59").copy()
    regular["dollar_vol"] = regular["close"] * regular["volume"]
    daily_vwap = regular.groupby(regular.index.date).agg(
        total_dollar_vol=("dollar_vol", "sum"),
        total_vol=("volume", "sum"),
    )
    daily_vwap["vwap"] = daily_vwap["total_dollar_vol"] / daily_vwap["total_vol"]
    daily_vwap.index = pd.to_datetime(daily_vwap.index)
    daily_vwap.index.name = "date"
    return daily_vwap[["vwap"]]


def compute_intraday_extremes(df: pd.DataFrame) -> pd.DataFrame:
    """Compute max drawdown and max runup within regular session."""
    regular = df.between_time("09:30", "15:59")
    results = []

    for date, group in regular.groupby(regular.index.date):
        prices = group["close"].values
        if len(prices) < 2:
            results.append((date, np.nan, np.nan))
            continue

        cummax = np.maximum.accumulate(prices)
        drawdowns = (prices - cummax) / cummax
        max_dd = drawdowns.min()

        cummin = np.minimum.accumulate(prices)
        runups = (prices - cummin) / cummin
        max_ru = runups.max()

        results.append((date, max_dd, max_ru))

    extremes = pd.DataFrame(results, columns=["date", "max_drawdown", "max_runup"])
    extremes["date"] = pd.to_datetime(extremes["date"])
    extremes = extremes.set_index("date")
    return extremes


def build_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Build complete daily metrics from 1-minute data."""
    daily = compute_regular_session_metrics(df)
    pre = compute_premarket_metrics(df)
    full = compute_full_day_metrics(df)
    vwap = compute_vwap(df)
    extremes = compute_intraday_extremes(df)

    daily = daily.join(pre, how="left").join(full).join(vwap).join(extremes)

    # Derived return metrics
    daily["close_to_close_ret"] = daily["reg_close"].pct_change()
    daily["open_to_close_ret"] = daily["reg_close"] / daily["reg_open"] - 1
    daily["intraday_range"] = (daily["reg_high"] - daily["reg_low"]) / daily["reg_open"]
    daily["gap_return"] = daily["reg_open"] / daily["reg_close"].shift(1) - 1

    # Absolute values
    daily["abs_close_to_close"] = daily["close_to_close_ret"].abs()
    daily["abs_open_to_close"] = daily["open_to_close_ret"].abs()

    # Large move flags
    for metric in ["abs_close_to_close", "abs_open_to_close", "intraday_range"]:
        for thresh in [0.01, 0.02, 0.03, 0.05]:
            col_name = f"{metric}_gt_{int(thresh * 100)}pct"
            daily[col_name] = daily[metric] > thresh

    return daily
