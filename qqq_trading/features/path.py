"""Path dependency / smoothness features."""
from __future__ import annotations

import pandas as pd
import numpy as np
from scipy import stats


def _rolling_r2(series: pd.Series, w: int) -> pd.Series:
    """R-squared of linear regression over rolling window."""
    r2_vals = []
    x = np.arange(w)
    for i in range(len(series)):
        if i < w - 1:
            r2_vals.append(np.nan)
            continue
        y = series.iloc[i - w + 1 : i + 1].values
        if np.isnan(y).any() or np.std(y) == 0:
            r2_vals.append(np.nan)
            continue
        _, _, r_value, _, _ = stats.linregress(x, y)
        r2_vals.append(r_value ** 2)
    return pd.Series(r2_vals, index=series.index)


def _rolling_hurst(returns: pd.Series, w: int) -> pd.Series:
    """Hurst exponent approximation via R/S method."""
    hurst_vals = []
    for i in range(len(returns)):
        if i < w - 1:
            hurst_vals.append(np.nan)
            continue
        r = returns.iloc[i - w + 1 : i + 1].values
        if np.isnan(r).any():
            hurst_vals.append(np.nan)
            continue
        mean_r = np.mean(r)
        dev = np.cumsum(r - mean_r)
        R = np.max(dev) - np.min(dev)
        S = np.std(r, ddof=1)
        if S == 0 or R == 0:
            hurst_vals.append(np.nan)
            continue
        hurst_vals.append(np.log(R / S) / np.log(w))
    return pd.Series(hurst_vals, index=returns.index)


def _rolling_max_dd(series: pd.Series, w: int) -> pd.Series:
    """Max drawdown over rolling window."""
    dd_vals = []
    for i in range(len(series)):
        if i < w - 1:
            dd_vals.append(np.nan)
            continue
        prices = series.iloc[i - w + 1 : i + 1].values
        peak = np.maximum.accumulate(prices)
        dd_vals.append(((prices - peak) / peak).min())
    return pd.Series(dd_vals, index=series.index)


def build_path_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build smoothness / path dependency features."""
    out = df.copy()
    close = out["reg_close"]
    ret = out["close_to_close_ret"]
    high = out["reg_high"]
    low = out["reg_low"]
    prev_close = close.shift(1)

    # True Range for path length
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    for window in [63, 126]:
        suffix = f"_{window}d"

        # A. R-squared of linear regression
        out[f"trend_r2{suffix}"] = _rolling_r2(close, window).shift(1)

        # B. Fractal Efficiency
        net_move = close.rolling(window).apply(
            lambda x: abs(x.iloc[-1] - x.iloc[0]), raw=False
        )
        path_length = true_range.rolling(window).sum()
        out[f"fractal_eff{suffix}"] = (net_move / path_length).shift(1)

        # C. Choppiness Index
        hh = high.rolling(window).max()
        ll = low.rolling(window).min()
        ci = 100 * np.log10(path_length / (hh - ll)) / np.log10(window)
        out[f"choppiness{suffix}"] = ci.shift(1)

        # D. Hurst Exponent
        out[f"hurst{suffix}"] = _rolling_hurst(ret, window).shift(1)

        # E. Trend strength
        period_ret = close.pct_change(window)
        period_vol = ret.rolling(window).std() * np.sqrt(window)
        out[f"trend_strength{suffix}"] = (period_ret.abs() / period_vol).shift(1)

        # F. Up-day ratio
        up_days = (ret > 0).astype(int).rolling(window).sum()
        out[f"up_day_ratio{suffix}"] = (up_days / window).shift(1)

        # G. Max drawdown over window
        out[f"max_dd_window{suffix}"] = _rolling_max_dd(close, window).shift(1)

    # H. Cross features: smoothness x catalyst
    for w_suffix in ["_63d", "_126d"]:
        r2 = out[f"trend_r2{w_suffix}"]

        out[f"smooth_trend{w_suffix}"] = (r2 > 0.85).astype(int)
        out[f"very_smooth{w_suffix}"] = (r2 > 0.95).astype(int)
        out[f"choppy{w_suffix}"] = (r2 < 0.3).astype(int)

        if "any_catalyst_imminent" in out.columns:
            out[f"smooth{w_suffix}_X_catalyst"] = out[f"smooth_trend{w_suffix}"] * out["any_catalyst_imminent"]
            out[f"smooth{w_suffix}_X_fomc"] = out[f"smooth_trend{w_suffix}"] * out.get("fomc_imminent", 0)

        if "vrp_negative" in out.columns:
            out[f"smooth{w_suffix}_X_vrp_neg"] = out[f"smooth_trend{w_suffix}"] * out["vrp_negative"]
            out[f"smooth{w_suffix}_X_vrp_neg_X_catalyst"] = (
                out[f"smooth_trend{w_suffix}"] * out["vrp_negative"] * out.get("any_catalyst_imminent", 0)
            )

        if "low_vol_regime" in out.columns:
            out[f"smooth{w_suffix}_X_lowvol"] = out[f"smooth_trend{w_suffix}"] * out["low_vol_regime"]

    return out
