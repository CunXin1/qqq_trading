"""Event calendar: FOMC dates, NFP dates, earnings season flags."""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


def load_fomc_dates(csv_path: Path | None = None) -> pd.DatetimeIndex:
    """Load FOMC announcement dates from CSV."""
    if csv_path is None:
        from utils.paths import DATA_DIR
        csv_path = DATA_DIR / "fomc_dates.csv"
    df = pd.read_csv(csv_path)
    return pd.DatetimeIndex(pd.to_datetime(df["date"]))


def compute_nfp_dates(start_year: int = 2000, end_year: int = 2026) -> pd.DatetimeIndex:
    """NFP is released on the first Friday of each month."""
    dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            first_day = pd.Timestamp(year, month, 1)
            days_until_friday = (4 - first_day.dayofweek) % 7
            first_friday = first_day + pd.Timedelta(days=days_until_friday)
            dates.append(first_friday)
    return pd.DatetimeIndex(dates)


def _compute_eve_dates(event_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Compute business day before each event date (skip weekends)."""
    eves = []
    for d in event_dates:
        eve = d - pd.Timedelta(days=1)
        while eve.weekday() >= 5:
            eve -= pd.Timedelta(days=1)
        eves.append(eve)
    return pd.DatetimeIndex(eves)


def compute_days_to_event(
    index: pd.DatetimeIndex, event_dates: pd.DatetimeIndex, default: int = 60
) -> list[int]:
    """For each date in index, compute days until next event."""
    sorted_events = sorted(event_dates)
    result = []
    ei = 0
    for date in index:
        while ei < len(sorted_events) and sorted_events[ei] < date:
            ei += 1
        if ei < len(sorted_events):
            result.append((sorted_events[ei] - date).days)
        else:
            result.append(default)
    return result


def compute_days_since_event(
    index: pd.DatetimeIndex, event_dates: pd.DatetimeIndex, default: int = 60
) -> list[int]:
    """For each date in index, compute days since last event."""
    sorted_events = sorted(event_dates)
    result = []
    ei = 0
    for date in index:
        while ei < len(sorted_events) - 1 and sorted_events[ei + 1] <= date:
            ei += 1
        if sorted_events[ei] <= date:
            result.append((date - sorted_events[ei]).days)
        else:
            result.append(default)
    return result


def compute_earnings_season(index: pd.DatetimeIndex) -> pd.Series:
    """Binary flag for earnings season periods."""
    flag = pd.Series(0, index=index, dtype=int)
    # Late Jan, Apr, Jul, Oct
    for m, d_start, d_end in [(1, 20, 31), (4, 20, 30), (7, 20, 31), (10, 20, 31)]:
        mask = (index.month == m) & (index.day >= d_start) & (index.day <= d_end)
        flag.loc[mask] = 1
    # First 2 weeks of following month
    for m, d_end in [(2, 14), (5, 14), (8, 14), (11, 14)]:
        mask = (index.month == m) & (index.day <= d_end)
        flag.loc[mask] = 1
    return flag
