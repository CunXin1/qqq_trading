"""Path dependency / smoothness features (30 features, optional Layer 4).
路径依赖 / 平滑度特征（30 个特征，可选的第 4 层）。

This module captures HOW the market arrived at current prices, not just WHERE
it is. Two paths to the same price can have very different vol implications:
本模块捕捉市场是如何到达当前价格的，而不仅仅是当前在哪里。
到达同一价格的两条路径可能有截然不同的波动率含义：

  - Smooth uptrend (high R²) → vol likely to stay low, UNLESS a catalyst hits.
    平滑上升趋势（高 R²）→ 波动率可能保持低位，除非遇到催化事件。
  - Choppy sideways (low R²) → vol already elevated, further spikes possible.
    震荡横盘（低 R²）→ 波动率已升高，可能进一步飙升。

NOT used in production models by default (include_path=False) because:
默认不在生产模型中使用（include_path=False），因为：
  1. Computation is slow (rolling loops over 63/126 day windows).
     计算较慢（在 63/126 天窗口上滚动循环）。
  2. Marginal AUC gain is small (~0.005) vs complexity cost.
     相对于复杂度成本，AUC 边际提升很小（约 0.005）。
  3. Available for research experimentation via --include-path.
     可通过 --include-path 用于研究实验。

Feature groups per window (7 × 2 windows = 14, + 16 cross = 30 total):
每个窗口的特征组（7 × 2 窗口 = 14，+ 16 交叉 = 共 30 个）：
  A. trend_r2:        R² of linear regression (trend smoothness).
                      线性回归 R²（趋势平滑度）。
  B. fractal_eff:     Net move / path length (directional efficiency).
                      净移动 / 路径长度（方向效率）。
  C. choppiness:      Choppiness Index (100 = max chop, 0 = perfect trend).
                      震荡指数（100 = 最大震荡，0 = 完美趋势）。
  D. hurst:           Hurst exponent (>0.5 = trending, <0.5 = mean-reverting).
                      Hurst 指数（>0.5 = 趋势，<0.5 = 均值回归）。
  E. trend_strength:  |period return| / period vol (signal-to-noise).
                      |期间收益率| / 期间波动率（信噪比）。
  F. up_day_ratio:    Fraction of positive-return days (sentiment proxy).
                      正收益天数占比（情绪代理）。
  G. max_dd_window:   Max drawdown over the rolling window.
                      滚动窗口内的最大回撤。
  H. Cross features:  Smoothness regime × catalyst/VRP/vol flags.
                      平滑度状态 × 催化事件/VRP/波动率标记。
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from scipy import stats


def _rolling_r2(series: pd.Series, w: int) -> pd.Series:
    """R-squared of linear regression over rolling window.
    滚动窗口上线性回归的 R² 值。

    R² measures how well a straight line fits the price series:
    R² 衡量直线对价格序列的拟合程度：
      R² ≈ 1.0: Strong linear trend (smooth, predictable path).
                 强线性趋势（平滑、可预测的路径）。
      R² ≈ 0.0: No trend (random walk, choppy market).
                 无趋势（随机游走，震荡市场）。
    """
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
    """Hurst exponent approximation via R/S (Rescaled Range) method.
    通过 R/S（重标极差）方法近似计算 Hurst 指数。

    Interpretation / 解读:
      H > 0.5: Persistent (trending) — past returns predict future direction.
               持续性（趋势）——过去收益率预测未来方向。
      H = 0.5: Random walk — no memory.
               随机游走——无记忆性。
      H < 0.5: Anti-persistent (mean-reverting) — moves tend to reverse.
               反持续性（均值回归）——波动倾向于反转。
    """
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
        R = np.max(dev) - np.min(dev)     # Rescaled range / 重标极差
        S = np.std(r, ddof=1)             # Standard deviation / 标准差
        if S == 0 or R == 0:
            hurst_vals.append(np.nan)
            continue
        hurst_vals.append(np.log(R / S) / np.log(w))
    return pd.Series(hurst_vals, index=returns.index)


def _rolling_max_dd(series: pd.Series, w: int) -> pd.Series:
    """Max drawdown over rolling window.
    滚动窗口内的最大回撤。

    Measures the worst peak-to-trough decline in the lookback window.
    衡量回望窗口内最严重的峰值到谷底的跌幅。
    """
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
    """Build smoothness / path dependency features at 63d and 126d windows.
    在 63 天和 126 天窗口上构建平滑度 / 路径依赖特征。

    Two window sizes capture different time horizons:
    两个窗口尺寸捕捉不同时间跨度：
      63d  ≈ 1 quarter — captures quarterly trend regime.
      63 天 ≈ 1 季度——捕捉季度趋势状态。
      126d ≈ 6 months — captures longer-term market character.
      126 天 ≈ 6 个月——捕捉更长期的市场特征。

    Args:
        df: DataFrame with base + external + interaction features.
            包含基础 + 外部 + 交互特征的 DataFrame。

    Returns:
        DataFrame with 30 additional path features.
        添加了 30 个路径特征的 DataFrame。
    """
    out = df.copy()
    close = out["reg_close"]
    ret = out["close_to_close_ret"]
    high = out["reg_high"]
    low = out["reg_low"]
    prev_close = close.shift(1)

    # True Range: max of (H-L, |H-prevC|, |L-prevC|) — path length building block.
    # 真实波幅：(H-L, |H-前收|, |L-前收|) 的最大值——路径长度的基本单元。
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    for window in [63, 126]:
        suffix = f"_{window}d"

        # A. R-squared of linear regression / 线性回归 R²
        out[f"trend_r2{suffix}"] = _rolling_r2(close, window).shift(1)

        # B. Fractal Efficiency = net move / path length
        # 分形效率 = 净移动 / 路径长度
        # High = directional (efficient path), Low = choppy (wasted movement).
        # 高 = 方向性强（高效路径），低 = 震荡（无效运动）。
        net_move = close.rolling(window).apply(
            lambda x: abs(x.iloc[-1] - x.iloc[0]), raw=False
        )
        path_length = true_range.rolling(window).sum()
        out[f"fractal_eff{suffix}"] = (net_move / path_length).shift(1)

        # C. Choppiness Index = 100 * log10(path_length / range) / log10(window)
        # 震荡指数
        # High (>60) = choppy/range-bound, Low (<40) = trending.
        # 高 (>60) = 震荡/区间震荡，低 (<40) = 趋势运行。
        hh = high.rolling(window).max()
        ll = low.rolling(window).min()
        ci = 100 * np.log10(path_length / (hh - ll)) / np.log10(window)
        out[f"choppiness{suffix}"] = ci.shift(1)

        # D. Hurst Exponent / Hurst 指数
        out[f"hurst{suffix}"] = _rolling_hurst(ret, window).shift(1)

        # E. Trend strength = |period return| / period vol (signal-to-noise ratio)
        # 趋势强度 = |期间收益率| / 期间波动率（信噪比）
        period_ret = close.pct_change(window)
        period_vol = ret.rolling(window).std() * np.sqrt(window)
        out[f"trend_strength{suffix}"] = (period_ret.abs() / period_vol).shift(1)

        # F. Up-day ratio: fraction of positive return days
        # 上涨天数比例：正收益天数占比
        up_days = (ret > 0).astype(int).rolling(window).sum()
        out[f"up_day_ratio{suffix}"] = (up_days / window).shift(1)

        # G. Max drawdown over window / 窗口内最大回撤
        out[f"max_dd_window{suffix}"] = _rolling_max_dd(close, window).shift(1)

    # H. Cross features: smoothness regime × catalyst/VRP/vol
    # 交叉特征：平滑度状态 × 催化事件/VRP/波动率
    # Smooth trend + catalyst = potential for trend break (outsized vol).
    # 平滑趋势 + 催化事件 = 趋势打破的可能性（超常波动率）。
    for w_suffix in ["_63d", "_126d"]:
        r2 = out[f"trend_r2{w_suffix}"]

        out[f"smooth_trend{w_suffix}"] = (r2 > 0.85).astype(int)     # Smooth trend / 平滑趋势
        out[f"very_smooth{w_suffix}"] = (r2 > 0.95).astype(int)      # Very smooth / 极度平滑
        out[f"choppy{w_suffix}"] = (r2 < 0.3).astype(int)            # Choppy / 震荡

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
