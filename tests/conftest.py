"""Shared test fixtures."""
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_daily_metrics():
    """50 trading days of realistic QQQ daily metrics."""
    np.random.seed(42)
    n = 50
    dates = pd.bdate_range("2023-01-03", periods=n)

    price = 300 + np.cumsum(np.random.randn(n) * 2)
    opens = price + np.random.randn(n) * 0.5
    highs = np.maximum(price, opens) + np.abs(np.random.randn(n)) * 2
    lows = np.minimum(price, opens) - np.abs(np.random.randn(n)) * 2
    closes = price

    df = pd.DataFrame({
        "reg_open": opens,
        "reg_high": highs,
        "reg_low": lows,
        "reg_close": closes,
        "volume_regular": np.random.randint(50_000_000, 150_000_000, n),
        "premarket_open": opens - 0.5,
        "premarket_high": opens + 1,
        "premarket_low": opens - 1,
        "premarket_close": opens + np.random.randn(n) * 0.3,
        "volume_premarket": np.random.randint(1_000_000, 10_000_000, n),
        "full_high": highs + 0.5,
        "full_low": lows - 0.5,
        "vwap": (opens + closes) / 2,
        "max_drawdown": -np.abs(np.random.randn(n)) * 0.02,
        "max_runup": np.abs(np.random.randn(n)) * 0.02,
    }, index=dates)

    df["close_to_close_ret"] = df["reg_close"].pct_change()
    df["open_to_close_ret"] = df["reg_close"] / df["reg_open"] - 1
    df["intraday_range"] = (df["reg_high"] - df["reg_low"]) / df["reg_open"]
    df["gap_return"] = df["reg_open"] / df["reg_close"].shift(1) - 1
    df["abs_close_to_close"] = df["close_to_close_ret"].abs()
    df["abs_open_to_close"] = df["open_to_close_ret"].abs()

    df["premarket_ret"] = df["premarket_close"] / df["premarket_open"] - 1
    df["premarket_range"] = (df["premarket_high"] - df["premarket_low"]) / df["premarket_open"]

    for metric in ["abs_close_to_close", "abs_open_to_close", "intraday_range"]:
        for thresh in [0.01, 0.02, 0.03, 0.05]:
            df[f"{metric}_gt_{int(thresh * 100)}pct"] = df[metric] > thresh

    return df


@pytest.fixture
def sample_external_data():
    """50 days of VIX/VVIX/rates data."""
    np.random.seed(42)
    n = 50
    dates = pd.bdate_range("2023-01-03", periods=n)

    vix = 20 + np.cumsum(np.random.randn(n) * 0.5)
    vix = np.clip(vix, 10, 40)

    df = pd.DataFrame({
        "vix_close": vix,
        "vix_high": vix + np.abs(np.random.randn(n)),
        "vix_low": vix - np.abs(np.random.randn(n)),
        "vvix_close": 90 + np.random.randn(n) * 5,
        "vvix_high": 95 + np.random.randn(n) * 5,
        "vvix_low": 85 + np.random.randn(n) * 5,
        "tnx_10y_close": 4.0 + np.random.randn(n) * 0.1,
        "tnx_10y_high": 4.1 + np.random.randn(n) * 0.1,
        "tnx_10y_low": 3.9 + np.random.randn(n) * 0.1,
        "irx_3m_close": 5.0 + np.random.randn(n) * 0.05,
        "irx_3m_high": 5.05 + np.random.randn(n) * 0.05,
        "irx_3m_low": 4.95 + np.random.randn(n) * 0.05,
        "fvx_5y_close": 4.2 + np.random.randn(n) * 0.08,
        "fvx_5y_high": 4.25 + np.random.randn(n) * 0.08,
        "fvx_5y_low": 4.15 + np.random.randn(n) * 0.08,
    }, index=dates)

    return df


@pytest.fixture
def config():
    """Default test config."""
    from qqq_trading.config import Config
    return Config()
