"""External feature engineering: VRP, VIX dynamics, rates, events (26 features).
外部特征工程：VRP、VIX 动态、利率、事件（26 个特征）。

This is Layer 2 of the feature pipeline. It incorporates cross-asset
and macro-event data that is not derivable from QQQ price alone.
这是特征管线的第 2 层。引入无法仅从 QQQ 价格推导的跨资产和宏观事件数据。

Feature groups (26 total) / 特征组（共 26 个）:
  1. VRP — Volatility Risk Premium (5):    VRP = VIX/100 - Realized Vol.
     VRP — 波动率风险溢价 (5)：             VRP = VIX/100 - 已实现波动率。
     Positive VRP = market fears more vol than realized → vol expansion likely.
     正 VRP = 市场恐惧波动率高于已实现 → 波动率扩张可能性大。
     Negative VRP = complacency → surprise vol spikes possible.
     负 VRP = 自满 → 可能出现意外波动率飙升。

  2. VIX dynamics (4):                     VIX changes, range, spikes.
     VIX 动态 (4)：                         VIX 变化、振幅、飙升。
     Captures fear momentum — not the VIX level itself (non-stationary).
     捕捉恐慌动量——而非 VIX 水平本身（非平稳的）。

  3. VVIX features (2):                    VVIX/VIX ratio and VVIX change.
     VVIX 特征 (2)：                        VVIX/VIX 比率和 VVIX 变化。
     "Vol of vol" — elevated VVIX signals uncertainty about vol direction.
     "波动率的波动率"——VVIX 升高意味着对波动率方向的不确定性。

  4. Interest rates (5):                   Yield curve, rate shocks.
     利率 (5)：                              收益率曲线、利率冲击。
     Inverted yield curve and rate shocks historically amplify equity vol.
     收益率曲线倒挂和利率冲击在历史上会放大股票波动率。

  5. Event calendar (10):                  FOMC, NFP, earnings season flags.
     事件日历 (10)：                         FOMC、NFP、财报季标记。
     Binary flags + days-to-event countdowns for proximity features.
     布尔标记 + 距事件倒计时用于接近度特征。
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from data.event_calendar import (
    load_fomc_dates, compute_nfp_dates, compute_cpi_dates, compute_pce_dates,
    _compute_eve_dates, compute_days_to_event, compute_earnings_season,
    load_megacap_earnings, build_megacap_earnings_flags,
)


def engineer_vrp_features(df: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    """Volatility Risk Premium = Implied Vol (VIX) - Realized Vol.
    波动率风险溢价 = 隐含波动率 (VIX) - 已实现波动率。

    VRP is the single most predictive external feature for next-day vol.
    VRP 是预测次日波动率最强的单一外部特征。

    Produces 5 features / 生成 5 个特征:
      vrp_20d:            VIX/100 - RV(20d). Core VRP signal.
                          核心 VRP 信号。
      vrp_10d, vrp_5d:    Short-window VRP for faster regime detection.
                          短窗口 VRP 用于更快的状态检测。
      vrp_20d_zscore:     VRP normalized to 60d rolling distribution.
                          VRP 相对于 60 天滚动分布的标准化值。
      vrp_20d_change_5d:  5-day change in VRP (VRP momentum).
                          VRP 的 5 天变化量（VRP 动量）。
    """
    vix = ext["vix_close"]

    rv20 = df.get("realized_vol_20d", df["close_to_close_ret"].rolling(20).std().shift(1) * np.sqrt(252))
    rv10 = df.get("realized_vol_10d", df["close_to_close_ret"].rolling(10).std().shift(1) * np.sqrt(252))
    rv5 = df.get("realized_vol_5d", df["close_to_close_ret"].rolling(5).std().shift(1) * np.sqrt(252))

    df["vrp_20d"] = (vix.shift(1) / 100) - rv20
    df["vrp_10d"] = (vix.shift(1) / 100) - rv10
    df["vrp_5d"] = (vix.shift(1) / 100) - rv5

    df["vrp_20d_zscore"] = (
        (df["vrp_20d"] - df["vrp_20d"].rolling(60).mean())
        / df["vrp_20d"].rolling(60).std()
    )
    df["vrp_20d_change_5d"] = df["vrp_20d"] - df["vrp_20d"].shift(5)

    return df


def engineer_vix_dynamics(df: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    """VIX changes and spikes — momentum of fear, not level.
    VIX 变化和飙升——恐慌的动量，而非水平。

    We avoid using raw VIX levels because they are non-stationary.
    Instead, we use pct_change and spike detection.
    我们避免使用原始 VIX 水平，因为它们是非平稳的。
    改为使用百分比变化和飙升检测。

    Produces 4 features / 生成 4 个特征:
      vix_pct_change_1d:  1-day VIX change (fear acceleration).
                          1 天 VIX 变化（恐慌加速度）。
      vix_pct_change_5d:  5-day VIX change (fear trend).
                          5 天 VIX 变化（恐慌趋势）。
      vix_range_1d:       Intraday VIX range / VIX close (vol of vol proxy).
                          VIX 日内振幅 / VIX 收盘价（波动率的波动率代理）。
      vix_spike:          Binary flag for VIX >10% single-day jump.
                          VIX 单日涨幅 >10% 的布尔标记。
    """
    vix = ext["vix_close"]

    df["vix_pct_change_1d"] = vix.pct_change().shift(1)
    df["vix_pct_change_5d"] = vix.pct_change(5).shift(1)
    df["vix_range_1d"] = ((ext["vix_high"] - ext["vix_low"]) / vix).shift(1)
    df["vix_spike"] = (vix.pct_change().shift(1) > 0.10).astype(int)

    return df


def engineer_vvix_features(df: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    """VVIX features — volatility of VIX (available from 2007, NaN-tolerant).
    VVIX 特征——VIX 的波动率（2007 年起可用，容忍 NaN）。

    Produces 2 features / 生成 2 个特征:
      vvix_vix_ratio:  VVIX / VIX. High ratio = market expects VIX to move sharply.
                       VVIX / VIX。高比率 = 市场预期 VIX 将剧烈波动。
      vvix_change_1d:  1-day VVIX change. VVIX spikes often precede QQQ vol events.
                       1 天 VVIX 变化。VVIX 飙升通常先于 QQQ 波动率事件。
    """
    vvix = ext["vvix_close"]
    vix = ext["vix_close"]

    df["vvix_vix_ratio"] = (vvix / vix).shift(1)
    df["vvix_change_1d"] = vvix.pct_change().shift(1)

    return df


def engineer_skew_features(df: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    """CBOE SKEW index features — tail risk and crash protection demand.
    CBOE SKEW 指数特征——尾部风险和崩盘保护需求。

    High SKEW = market buying OTM puts = expecting tail events.
    高 SKEW = 市场在购买虚值看跌期权 = 预期尾部事件。

    Produces 4 features / 生成 4 个特征:
      skew_level:       SKEW index value (shifted 1d to avoid lookahead).
                        SKEW 指数值（滞后 1 天防止未来信息泄露）。
      skew_change_1d:   1-day SKEW change (tail risk acceleration).
                        1 天 SKEW 变化（尾部风险加速度）。
      skew_change_5d:   5-day SKEW change (tail risk trend).
                        5 天 SKEW 变化（尾部风险趋势）。
      skew_extreme:     Binary flag for SKEW > 150 (elevated tail risk).
                        SKEW > 150 的标记（高尾部风险）。
    """
    if "skew_close" not in ext.columns:
        return df

    skew = ext["skew_close"]
    df["skew_level"] = skew.shift(1)
    df["skew_change_1d"] = skew.pct_change().shift(1)
    df["skew_change_5d"] = skew.pct_change(5).shift(1)
    df["skew_extreme"] = (skew.shift(1) > 150).astype(int)

    return df


def engineer_rate_features(df: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    """Interest rate and yield curve features.
    利率和收益率曲线特征。

    Produces 5 features / 生成 5 个特征:
      yield_curve_slope:     10Y - 3M spread. Positive = normal, negative = inverted.
                             10 年 - 3 个月利差。正 = 正常，负 = 倒挂。
      yield_curve_inverted:  Binary flag for inverted curve (recession indicator).
                             收益率曲线倒挂标记（衰退指标）。
      yield_10y_change_1d:   Daily change in 10Y yield (rate shock proxy).
                             10 年期收益率日变化（利率冲击代理）。
      yield_10y_vol_20d:     20d rolling std of 10Y yield changes.
                             10 年期收益率变化的 20 天滚动标准差。
      rate_shock:            Binary flag for 10Y move > 2σ (tail event).
                             10 年期波动 > 2σ 的标记（尾部事件）。
    """
    tnx = ext["tnx_10y_close"]
    irx = ext["irx_3m_close"]

    df["yield_curve_slope"] = (tnx - irx).shift(1)
    df["yield_curve_inverted"] = ((tnx - irx) < 0).astype(int).shift(1)
    df["yield_10y_change_1d"] = tnx.diff().shift(1)
    df["yield_10y_vol_20d"] = tnx.diff().rolling(20).std().shift(1)
    df["rate_shock"] = (
        tnx.diff().abs().shift(1) > tnx.diff().rolling(20).std().shift(1) * 2
    ).astype(int)

    return df


def engineer_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """FOMC, NFP, earnings season, and combined macro event flags.
    FOMC、NFP、财报季及组合宏观事件标记。

    Produces 10 features / 生成 10 个特征:
      is_fomc_day/eve:       FOMC announcement day / eve (business day before).
                             FOMC 公告日 / 前夜（前一个工作日）。
      days_to_fomc:          Calendar days until next FOMC (countdown).
                             距下次 FOMC 的自然日天数（倒计时）。
      fomc_week:             Within 5 days of FOMC (positioning window).
                             距 FOMC 5 天以内（布局窗口）。
      is_nfp_day/eve:        Non-Farm Payrolls day / eve.
                             非农公布日 / 前夜。
      days_to_nfp:           Calendar days until next NFP.
                             距下次非农的自然日天数。
      is_macro_event_day/eve: Combined FOMC | NFP flag.
                              FOMC | NFP 组合标记。
      is_earnings_season:    ~4 week window around mega-cap earnings.
                             大型股财报集中发布的约 4 周窗口。
    """
    fomc_dates = load_fomc_dates()
    nfp_dates = compute_nfp_dates()

    # FOMC features / FOMC 特征
    df["is_fomc_day"] = df.index.isin(fomc_dates).astype(int)
    fomc_eve = _compute_eve_dates(fomc_dates)
    df["is_fomc_eve"] = df.index.isin(fomc_eve).astype(int)
    df["days_to_fomc"] = compute_days_to_event(df.index, fomc_dates)
    df["fomc_week"] = (df["days_to_fomc"] <= 5).astype(int)

    # NFP features / NFP 特征
    df["is_nfp_day"] = df.index.isin(nfp_dates).astype(int)
    nfp_eve = _compute_eve_dates(nfp_dates)
    df["is_nfp_eve"] = df.index.isin(nfp_eve).astype(int)
    df["days_to_nfp"] = compute_days_to_event(df.index, nfp_dates, default=30)

    # CPI features / CPI 特征
    cpi_dates = compute_cpi_dates()
    df["is_cpi_day"] = df.index.isin(cpi_dates).astype(int)
    cpi_eve = _compute_eve_dates(cpi_dates)
    df["is_cpi_eve"] = df.index.isin(cpi_eve).astype(int)
    df["days_to_cpi"] = compute_days_to_event(df.index, cpi_dates, default=30)

    # PCE features / PCE 特征
    pce_dates = compute_pce_dates()
    df["is_pce_day"] = df.index.isin(pce_dates).astype(int)
    pce_eve = _compute_eve_dates(pce_dates)
    df["is_pce_eve"] = df.index.isin(pce_eve).astype(int)

    # Combined macro event flags / 组合宏观事件标记
    df["is_macro_event_day"] = (
        (df["is_fomc_day"] == 1) | (df["is_nfp_day"] == 1) |
        (df["is_cpi_day"] == 1) | (df["is_pce_day"] == 1)
    ).astype(int)
    df["is_macro_event_eve"] = (
        (df["is_fomc_eve"] == 1) | (df["is_nfp_eve"] == 1) |
        (df["is_cpi_eve"] == 1) | (df["is_pce_eve"] == 1)
    ).astype(int)

    # Earnings season / 财报季
    df["is_earnings_season"] = compute_earnings_season(df.index)

    # Mega-cap earnings (NVDA/AAPL/MSFT) / 大型股财报
    try:
        earnings_df = load_megacap_earnings()
        if not earnings_df.empty:
            mc_flags = build_megacap_earnings_flags(df.index, earnings_df)
            for col in mc_flags.columns:
                df[col] = mc_flags[col]
    except Exception:
        pass  # graceful fallback if yfinance unavailable

    return df


def engineer_all_external(df: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    """Run full external feature engineering pipeline (26 features).
    运行完整的外部特征工程管线（26 个特征）。

    First aligns external data (VIX/VVIX/rates) to the daily_metrics index
    using forward-fill (weekends/holidays filled with last available value),
    then runs all 5 sub-pipelines.
    首先将外部数据（VIX/VVIX/利率）通过前向填充对齐到 daily_metrics 索引
    （周末/节假日用最近可用值填充），然后运行全部 5 个子管线。

    Args:
        df:  DataFrame with base features (from engineer_base_features).
             包含基础特征的 DataFrame（来自 engineer_base_features）。
        ext: External data DataFrame (from download_external_data).
             外部数据 DataFrame（来自 download_external_data）。

    Returns:
        DataFrame with 26 additional external features.
        添加了 26 个外部特征的 DataFrame。
    """
    ext_aligned = ext.reindex(df.index).ffill()
    df = engineer_vrp_features(df, ext_aligned)
    df = engineer_vix_dynamics(df, ext_aligned)
    df = engineer_vvix_features(df, ext_aligned)
    df = engineer_skew_features(df, ext_aligned)
    df = engineer_rate_features(df, ext_aligned)
    df = engineer_event_features(df)
    return df
