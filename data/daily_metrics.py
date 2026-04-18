"""Build daily OHLCV metrics from 1-minute QQQ data.
从 1 分钟 QQQ 数据构建日频 OHLCV 指标。

This module is the first stage of the data pipeline. It reads raw 1-min bars
and produces a single daily DataFrame with the following groups of columns:
本模块是数据管线的第一阶段。读入原始 1 分钟 K 线，产出包含以下列组的日频 DataFrame：

1. Regular session OHLCV  (reg_open, reg_high, reg_low, reg_close, volume_regular)
   常规交易时段 OHLCV     （开盘价、最高价、最低价、收盘价、成交量）
2. Pre-market metrics     (premarket_open/high/low/close, premarket_ret, premarket_range)
   盘前指标               （盘前 OHLC、盘前收益率、盘前振幅）
3. Full-day high/low      (full_high, full_low — spanning all sessions)
   全时段最高/最低价       （含盘前盘后的极值）
4. VWAP                   (volume-weighted average price, regular session only)
   成交量加权均价          （仅限常规时段）
5. Intraday extremes      (max_drawdown, max_runup — peak-to-trough & trough-to-peak)
   日内极端回撤/反弹       （从累计最高点回撤 & 从累计最低点反弹）
6. Derived return metrics  (close_to_close_ret, open_to_close_ret, intraday_range,
                            gap_return, abs_close_to_close, abs_open_to_close)
   衍生收益指标            （收盘对收盘收益率、开盘对收盘收益率、日内振幅、
                            跳空收益率、绝对值版本）
7. Large-move boolean flags (e.g. abs_close_to_close_gt_2pct)
   大波动布尔标记           （如 |C2C| > 2% 等，用于构造训练标签）
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


def load_1min_data(parquet_path: Path) -> pd.DataFrame:
    """Load adjusted 1-min parquet data.
    读取经过拆分/分红调整的 1 分钟 K 线数据。

    Args:
        parquet_path: Path to the parquet file (e.g. datasets/QQQ_1min_adjusted.parquet).
                      parquet 文件路径（如 datasets/QQQ_1min_adjusted.parquet）。

    Returns:
        DataFrame indexed by datetime with columns: open, high, low, close, volume.
        以 datetime 为索引的 DataFrame，列包含 open/high/low/close/volume。
    """
    df = pd.read_parquet(parquet_path)
    df.index = pd.to_datetime(df.index)
    return df


def compute_regular_session_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute OHLCV metrics for regular trading hours (9:30-16:00).
    计算常规交易时段 (9:30-16:00) 的日频 OHLCV 指标。

    Filters 1-min bars to 9:30–15:59 (inclusive), then groups by calendar date
    to produce: reg_open (first bar open), reg_high (session high), reg_low
    (session low), reg_close (last bar close), volume_regular (total volume).
    筛选 9:30–15:59 的 1 分钟 K 线，按日期分组聚合：
    reg_open（首根开盘价）、reg_high（时段最高价）、reg_low（时段最低价）、
    reg_close（末根收盘价）、volume_regular（时段总成交量）。
    """
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
    """Compute pre-market metrics (4:00-9:29).
    计算盘前时段 (4:00-9:29) 指标。

    Produces OHLCV for the pre-market window, plus two derived columns:
    产出盘前 OHLCV，以及两个衍生列：

    - premarket_ret:   (premarket_close / premarket_open) - 1
                       盘前收益率 = 盘前收盘价 / 盘前开盘价 - 1
    - premarket_range: (premarket_high - premarket_low) / premarket_open
                       盘前振幅 = (盘前最高 - 盘前最低) / 盘前开盘价

    These capture overnight sentiment and volatility before the main session.
    用于捕捉隔夜情绪与开盘前波动率。
    """
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
    """Compute full-day high/low across all sessions.
    计算全时段（含盘前盘后）最高价和最低价。

    Unlike regular session metrics, this includes pre-market and after-hours bars,
    giving the true daily extremes.
    与常规时段指标不同，此处包含盘前和盘后数据，给出真实的全天极值。
    """
    daily_full = df.groupby(df.index.date).agg(
        full_high=("high", "max"),
        full_low=("low", "min"),
    )
    daily_full.index = pd.to_datetime(daily_full.index)
    daily_full.index.name = "date"
    return daily_full


def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily VWAP for regular session.
    计算常规时段的日内成交量加权均价（VWAP）。

    VWAP = sum(price × volume) / sum(volume), computed over 9:30–15:59.
    VWAP = Σ(价格 × 成交量) / Σ(成交量)，仅限 9:30–15:59。

    VWAP is a key institutional benchmark; price relative to VWAP is used
    downstream as a feature (e.g. close/vwap ratio).
    VWAP 是机构常用基准价，下游特征工程会用到 收盘价/VWAP 比值。
    """
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
    """Compute max drawdown and max runup within regular session.
    计算常规时段内的最大回撤和最大反弹。

    For each trading day:
    对每个交易日：

    - max_drawdown: largest peak-to-trough decline (negative value).
                    最大峰值到谷底跌幅（负值）。
      Calculated as min((price - cumulative_max) / cumulative_max).
      计算方式：min((当前价 - 累计最高价) / 累计最高价)。

    - max_runup:    largest trough-to-peak rally (positive value).
                    最大谷底到峰值涨幅（正值）。
      Calculated as max((price - cumulative_min) / cumulative_min).
      计算方式：max((当前价 - 累计最低价) / 累计最低价)。

    These measure intraday path dependency — a day can have high range but
    low drawdown (steady trend), or vice versa (volatile reversal).
    这两个指标衡量日内路径依赖性：一天可能振幅大但回撤小（单边趋势），
    也可能反之（剧烈反转）。
    """
    regular = df.between_time("09:30", "15:59")
    results = []

    for date, group in regular.groupby(regular.index.date):
        prices = group["close"].values
        if len(prices) < 2:
            results.append((date, np.nan, np.nan))
            continue

        # Max drawdown: track running peak, compute drop from peak
        # 最大回撤：追踪运行中的峰值，计算从峰值的跌幅
        cummax = np.maximum.accumulate(prices)
        drawdowns = (prices - cummax) / cummax
        max_dd = drawdowns.min()

        # Max runup: track running trough, compute rise from trough
        # 最大反弹：追踪运行中的谷值，计算从谷值的涨幅
        cummin = np.minimum.accumulate(prices)
        runups = (prices - cummin) / cummin
        max_ru = runups.max()

        results.append((date, max_dd, max_ru))

    extremes = pd.DataFrame(results, columns=["date", "max_drawdown", "max_runup"])
    extremes["date"] = pd.to_datetime(extremes["date"])
    extremes = extremes.set_index("date")
    return extremes


def build_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Build complete daily metrics from 1-minute data.
    从 1 分钟数据构建完整的日频指标表。

    This is the main entry point of the module. It orchestrates all sub-computations
    and joins them into a single DataFrame, then adds derived return metrics and
    large-move boolean flags.
    这是本模块的主入口函数。它编排所有子计算并合并为一张 DataFrame，
    然后添加衍生收益指标和大波动布尔标记。

    Derived columns produced / 生成的衍生列：
    - close_to_close_ret:  daily close-to-close return (收盘对收盘日收益率)
    - open_to_close_ret:   intraday open-to-close return (日内开盘到收盘收益率)
    - intraday_range:      (high - low) / open — key target for range_0dte model
                           (最高 - 最低) / 开盘价 — range_0dte 模型的核心预测目标
    - gap_return:          overnight gap = today's open / yesterday's close - 1
                           隔夜跳空 = 今日开盘价 / 昨日收盘价 - 1
    - abs_close_to_close:  |C2C| — key target for c2c_1dte model
                           |收盘对收盘| — c2c_1dte 模型的核心预测目标
    - abs_open_to_close:   |O2C| — key target for otc_0dte model
                           |开盘对收盘| — otc_0dte 模型的核心预测目标

    Large-move flags (e.g. abs_close_to_close_gt_2pct) are boolean columns
    indicating whether the metric exceeds 1%/2%/3%/5% thresholds.
    大波动标记（如 abs_close_to_close_gt_2pct）是布尔列，
    标记该指标是否超过 1%/2%/3%/5% 阈值。
    """
    daily = compute_regular_session_metrics(df)
    pre = compute_premarket_metrics(df)
    full = compute_full_day_metrics(df)
    vwap = compute_vwap(df)
    extremes = compute_intraday_extremes(df)

    daily = daily.join(pre, how="left").join(full).join(vwap).join(extremes)

    # Derived return metrics / 衍生收益指标
    daily["close_to_close_ret"] = daily["reg_close"].pct_change()
    daily["open_to_close_ret"] = daily["reg_close"] / daily["reg_open"] - 1
    daily["intraday_range"] = (daily["reg_high"] - daily["reg_low"]) / daily["reg_open"]
    daily["gap_return"] = daily["reg_open"] / daily["reg_close"].shift(1) - 1

    # Absolute values (direction-agnostic, used as model targets)
    # 绝对值版本（不关心方向，用作模型预测目标）
    daily["abs_close_to_close"] = daily["close_to_close_ret"].abs()
    daily["abs_open_to_close"] = daily["open_to_close_ret"].abs()

    # Large move flags at multiple thresholds for quick filtering and label generation
    # 多阈值大波动标记，用于快速筛选和标签生成
    for metric in ["abs_close_to_close", "abs_open_to_close", "intraday_range"]:
        for thresh in [0.01, 0.02, 0.03, 0.05]:
            col_name = f"{metric}_gt_{int(thresh * 100)}pct"
            daily[col_name] = daily[metric] > thresh

    return daily
