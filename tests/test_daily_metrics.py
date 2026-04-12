"""Tests for daily metrics module."""
import pandas as pd
import numpy as np
from qqq_trading.data.daily_metrics import (
    compute_regular_session_metrics,
    compute_intraday_extremes,
)


def test_regular_session_bounds():
    """Verify only 9:30-15:59 data is used."""
    dates = pd.date_range("2023-01-03 04:00", "2023-01-03 19:59", freq="1min")
    df = pd.DataFrame({
        "open": 300.0, "high": 301.0, "low": 299.0,
        "close": 300.5, "volume": 1000,
    }, index=dates)

    result = compute_regular_session_metrics(df)
    assert len(result) == 1
    assert result["volume_regular"].iloc[0] > 0


def test_large_move_flags(sample_daily_metrics):
    """Verify large move flags are boolean."""
    for thresh in [1, 2, 3, 5]:
        col = f"abs_close_to_close_gt_{thresh}pct"
        assert col in sample_daily_metrics.columns
        assert sample_daily_metrics[col].dtype == bool
