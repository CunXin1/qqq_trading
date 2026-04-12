"""Time-series train/val/test splitting utilities."""
from __future__ import annotations

import pandas as pd
from typing import Optional


def date_split(
    df: pd.DataFrame,
    train_end: str = "2019-12-31",
    val_start: str = "2020-01-01",
    val_end: str = "2022-12-31",
    test_start: str = "2023-01-01",
) -> dict[str, pd.DataFrame]:
    """Split DataFrame by date into train/val/test."""
    return {
        "train": df.loc[:train_end],
        "val": df.loc[val_start:val_end],
        "test": df.loc[test_start:],
    }


def train_test_split(
    df: pd.DataFrame,
    train_end: str = "2022-12-31",
    test_start: str = "2023-01-01",
) -> dict[str, pd.DataFrame]:
    """Simple train/test split by date."""
    return {
        "train": df.loc[:train_end],
        "test": df.loc[test_start:],
    }


def walk_forward_splits(
    df: pd.DataFrame,
    test_years: list[int],
    train_window_years: int = 5,
    purge_days: int = 5,
) -> list[dict]:
    """Generate walk-forward CV splits with purge gap.

    Returns list of dicts with 'train', 'test', 'year' keys.
    """
    splits = []
    for year in test_years:
        test_start = f"{year}-01-01"
        test_end = f"{year}-12-31"
        train_start_year = year - train_window_years
        train_end_date = pd.Timestamp(test_start) - pd.Timedelta(days=purge_days)

        train = df.loc[f"{train_start_year}-01-01":train_end_date.strftime("%Y-%m-%d")]
        test = df.loc[test_start:test_end]

        if len(train) > 0 and len(test) > 0:
            splits.append({
                "train": train,
                "test": test,
                "year": year,
            })
    return splits
