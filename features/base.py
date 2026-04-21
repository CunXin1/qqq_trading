"""Base feature engineering from daily QQQ metrics (53 features).
基于 QQQ 日频指标的基础特征工程（53 个特征）。

This is Layer 1 of the feature pipeline — it uses only QQQ's own
price, volume, and derived daily metrics. No external data required.
这是特征管线的第 1 层——仅使用 QQQ 自身的价格、成交量和衍生日频指标。
无需外部数据。

All features are strictly CAUSAL: they use only data available before
the prediction point (shift(1) for yesterday's data, or same-day
pre-market data available before the open).
所有特征严格满足因果性：仅使用预测时点之前的数据
（shift(1) 取昨天数据，或取开盘前可获得的当天盘前数据）。

Feature groups (53 total) / 特征组（共 53 个）:
  A. Lagged returns (15):    ret/abs_ret/range for lag 1-5.
     滞后收益率 (15)：       收益率/绝对收益率/振幅 滞后 1-5 天。
  B. Volatility (10):        Realized vol at 5/10/20/60d, vol ratios, DD/RU lags.
     波动率 (10)：           5/10/20/60 天已实现波动率、波动率比率、回撤/反弹滞后。
  C. Pre-market (3):         Same-day premarket ret/range/volume ratio.
     盘前 (3)：              当天盘前收益率/振幅/成交量比率。
  D. Volume (2):             Yesterday's volume vs 20d avg, 5d/20d volume trend.
     成交量 (2)：            昨日成交量 vs 20 天均值、5 天/20 天成交量趋势。
  E. Calendar (6):           Day of week, month, month start/end, OPEX week, days since big move.
     日历 (6)：              星期几、月份、月初/月末、期权到期周、距上次大波动天数。
  F. Technical (8):          Distance from MA20/50/200, RSI(14), high/low proximity.
     技术指标 (8)：          距 MA20/50/200 的距离、RSI(14)、高低点接近度。
  G. Gap (2):                Yesterday's gap return and its absolute value.
     跳空 (2)：              昨日跳空收益率及其绝对值。
  H. Rolling stats (8):      Mean abs return and std at 5/10/20/60d windows.
     滚动统计 (8)：          5/10/20/60 天窗口的平均绝对收益率和标准差。
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def engineer_base_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Build all base predictive features from daily QQQ metrics.
    从 QQQ 日频指标构建所有基础预测特征。

    All features use only past data (shift(1)) to prevent lookahead bias.
    Pre-market features are an exception: they use same-day data available
    before the regular session opens at 9:30.
    所有特征仅使用过去数据（shift(1)）以防止未来信息泄漏。
    盘前特征是例外：它们使用常规交易时段 9:30 开盘前可获得的当天数据。

    Args:
        daily: DataFrame from build_daily_metrics() with OHLCV + derived columns.
               来自 build_daily_metrics() 的 DataFrame，包含 OHLCV + 衍生列。

    Returns:
        DataFrame with all 53 base features added as new columns.
        添加了全部 53 个基础特征新列的 DataFrame。
    """
    df = daily.copy()

    # ── A. Lagged return features (15 features) ──
    # 滞后收益率特征（15 个）
    # Captures recent return/volatility patterns and mean-reversion/momentum signals.
    # 捕捉近期收益率/波动率模式以及均值回归/动量信号。
    for lag in range(1, 6):
        df[f"ret_lag{lag}"] = df["close_to_close_ret"].shift(lag)       # Signed return / 带符号收益率
        df[f"abs_ret_lag{lag}"] = df["abs_close_to_close"].shift(lag)   # Absolute return / 绝对收益率
        df[f"range_lag{lag}"] = df["intraday_range"].shift(lag)         # Intraday range / 日内振幅

    # Rolling mean absolute return & std (8 features)
    # 滚动平均绝对收益率 & 标准差（8 个）
    for w in [5, 10, 20, 60]:
        df[f"mean_abs_ret_{w}d"] = df["abs_close_to_close"].rolling(w).mean().shift(1)
        df[f"std_ret_{w}d"] = df["close_to_close_ret"].rolling(w).std().shift(1)

    # ── B. Volatility features (10 features) ──
    # 波动率特征（10 个）
    # Annualized realized volatility at multiple lookback windows.
    # 多回望窗口的年化已实现波动率。
    for w in [5, 10, 20, 60]:
        df[f"realized_vol_{w}d"] = (
            df["close_to_close_ret"].rolling(w).std().shift(1) * np.sqrt(252)
        )

    # Vol regime indicators: short-term vol / long-term vol.
    # 波动率状态指标：短期波动率 / 长期波动率。
    # Ratio > 1 = vol expanding (risk-on signal); < 1 = vol contracting (complacent).
    # 比率 > 1 = 波动率扩张（风险偏好信号）；< 1 = 波动率收缩（自满状态）。
    df["vol_ratio_5_60"] = df["realized_vol_5d"] / df["realized_vol_60d"]
    df["vol_ratio_10_60"] = df["realized_vol_10d"] / df["realized_vol_60d"]

    # Regime switch detectors: short vol vs medium vol.
    # 状态切换检测器：短期波动率 vs 中期波动率。
    # RV5/RV20 > 1.5 = vol suddenly spiking while 20d average still calm → regime switch.
    # RV5/RV20 > 1.5 = 波动率突然飙升但 20 天均值仍低 → 状态切换。
    df["vol_ratio_5_20"] = df["realized_vol_5d"] / df["realized_vol_20d"]
    df["vol_ratio_5_20_change"] = df["vol_ratio_5_20"] - df["vol_ratio_5_20"].shift(1)
    df["vol_regime_switch"] = (df["vol_ratio_5_20"] > 1.5).astype(int)

    # Days since vol compression: count consecutive days with range < 1%.
    # 距波动率压缩期天数：连续 range < 1% 的天数。
    # After prolonged compression, vol tends to snap back violently.
    # 长期压缩后，波动率往往剧烈反弹。
    range_below_1pct = (df["intraday_range"].shift(1) < 0.01).astype(int)
    # Compute streak: consecutive days below 1%
    streak = range_below_1pct.copy()
    for i in range(1, len(streak)):
        if streak.iloc[i] == 1:
            streak.iloc[i] = streak.iloc[i - 1] + 1
    df["calm_streak_days"] = streak

    # Vol snap: yesterday's range > 1.5% after 5+ calm days → vol is waking up.
    # 波动反弹：连续 5 天以上低波动后，昨天振幅 > 1.5% → 波动率正在回归。
    df["vol_snap"] = (
        (df["calm_streak_days"].shift(1) >= 5) &
        (df["intraday_range"].shift(1) > 0.015)
    ).astype(int)

    # Max drawdown/runup lags: capture recent intraday extremes.
    # 最大回撤/反弹滞后：捕捉近期日内极端波动。
    for lag in range(1, 4):
        df[f"max_dd_lag{lag}"] = df["max_drawdown"].shift(lag)
        df[f"max_ru_lag{lag}"] = df["max_runup"].shift(lag)

    # ── C. Pre-market features (3 features, same-day, available before 9:30) ──
    # 盘前特征（3 个，当天数据，9:30 前可用）
    # These are NOT shifted because they represent today's pre-market activity,
    # which is known before the regular session opens.
    # 这些没有 shift，因为它们代表今天的盘前活动，在常规交易开盘前已知。
    df["premarket_ret_today"] = df["premarket_ret"]           # Pre-market return / 盘前收益率
    df["premarket_range_today"] = df["premarket_range"]       # Pre-market range / 盘前振幅
    df["premarket_vol_ratio"] = (                             # Pre-market volume vs 20d avg / 盘前成交量 vs 20日均值
        df["volume_premarket"]
        / df["volume_premarket"].rolling(20).mean().shift(1)
    )

    # ── D. Volume features (2 features) ──
    # 成交量特征（2 个）
    df["vol_ratio_20d"] = (        # Yesterday's volume relative to 20d avg / 昨日成交量相对于 20 日均值
        df["volume_regular"].shift(1)
        / df["volume_regular"].rolling(20).mean().shift(1)
    )
    df["vol_trend_5_20"] = (       # 5d avg volume / 20d avg volume (volume trend) / 5日均量/20日均量（量能趋势）
        df["volume_regular"].rolling(5).mean().shift(1)
        / df["volume_regular"].rolling(20).mean().shift(1)
    )

    # ── E. Calendar features (6 features) ──
    # 日历特征（6 个）
    df["dow"] = df.index.dayofweek     # 0=Monday ... 4=Friday / 0=周一 ... 4=周五
    df["month"] = df.index.month       # 1-12

    # Month start/end flags (rebalancing effects)
    # 月初/月末标记（再平衡效应）
    df["is_month_start"] = 0
    df["is_month_end"] = 0
    month_periods = df.index.to_period("M")
    for mp in month_periods.unique():
        mask = month_periods == mp
        idx = df.index[mask]
        if len(idx) > 0:
            df.loc[idx[0], "is_month_start"] = 1
            df.loc[idx[-1], "is_month_end"] = 1

    # Options expiration week: 3rd Friday ± 2 business days.
    # 期权到期周：每月第三个周五 ± 2 个工作日。
    # Gamma exposure unwinds during OPEX week often amplify intraday range.
    # 期权到期周的 Gamma 敞口平仓通常会放大日内振幅。
    df["is_opex_week"] = 0
    for idx in df.index:
        if idx.weekday() == 4 and 15 <= idx.day <= 21:
            week_start = idx - pd.Timedelta(days=4)
            week_end = idx
            mask = (df.index >= week_start) & (df.index <= week_end)
            df.loc[mask, "is_opex_week"] = 1

    # Days since last >2% move: longer gap = more "coiled spring" potential.
    # 距上次 >2% 波动的天数：间隔越长 = "弹簧压缩"效应越强。
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

    # ── F. Technical features (8 features) ──
    # 技术指标特征（8 个）
    close = df["reg_close"]

    # Distance from moving averages (mean-reversion signal).
    # 距移动平均线的距离（均值回归信号）。
    for w in [20, 50, 200]:
        ma = close.rolling(w).mean()
        df[f"dist_from_ma{w}"] = ((close - ma) / ma).shift(1)

    # RSI(14): overbought (>70) or oversold (<30) conditions.
    # RSI(14)：超买 (>70) 或超卖 (<30) 状态。
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = (100 - 100 / (1 + rs)).shift(1)

    # N-day high/low proximity: how close is price to recent extremes.
    # N 天高低点接近度：价格距近期极值有多近。
    # Near 0 = at the high; large negative = far from high.
    # 接近 0 = 在高点；大负值 = 远离高点。
    for w in [20, 50]:
        hh = close.rolling(w).max()
        ll = close.rolling(w).min()
        df[f"proximity_{w}d_high"] = ((close - hh) / hh).shift(1)
        df[f"proximity_{w}d_low"] = ((close - ll) / ll).shift(1)

    # ── G. Gap features (2 features) ──
    # 跳空特征（2 个）
    # Overnight gaps often predict intraday continuation or reversal.
    # 隔夜跳空通常预示日内趋势延续或反转。
    df["gap_ret_lag1"] = df["gap_return"].shift(1)         # Yesterday's gap (signed) / 昨日跳空（带符号）
    df["abs_gap_lag1"] = df["gap_return"].abs().shift(1)   # Yesterday's gap (absolute) / 昨日跳空（绝对值）

    return df
