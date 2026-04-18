"""Event calendar: FOMC dates, NFP dates, earnings season flags.
事件日历：FOMC 日期、非农就业（NFP）日期、财报季标记。

Macro events systematically drive QQQ volatility. This module provides:
宏观事件系统性地驱动 QQQ 波动率。本模块提供：

1. FOMC dates       — loaded from a curated CSV (datasets/fomc_dates.csv).
   FOMC 日期        — 从手工维护的 CSV 文件加载。
2. NFP dates        — computed algorithmically (first Friday of each month).
   非农日期          — 通过算法计算（每月第一个周五）。
3. Eve dates        — the business day before an event (for "day-before" effects).
   前夜日期          — 事件前一个工作日（用于捕捉"事件前一天"效应）。
4. Days-to / since  — countdown/countup features for proximity to events.
   距事件天数         — 距下次事件天数 / 距上次事件天数（用作特征）。
5. Earnings season  — binary flag for the ~4-week window around mega-cap earnings.
   财报季标记         — 大型股集中发布财报的约 4 周窗口期布尔标记。

These are consumed by features/external.py to build event-aware features
(e.g. is_fomc, is_nfp_eve, days_to_fomc, is_earnings_season).
这些被 features/external.py 使用，构建事件感知特征
（如 is_fomc、is_nfp_eve、days_to_fomc、is_earnings_season）。
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


def load_fomc_dates(csv_path: Path | None = None) -> pd.DatetimeIndex:
    """Load FOMC announcement dates from CSV.
    从 CSV 文件加载 FOMC 议息会议公告日期。

    The CSV (datasets/fomc_dates.csv) contains one column "date" with
    historical and scheduled FOMC announcement dates.
    CSV 文件（datasets/fomc_dates.csv）包含一列 "date"，
    记录历史和已排期的 FOMC 公告日期。

    Args:
        csv_path: Override path. Defaults to datasets/fomc_dates.csv.
                  自定义路径。默认为 datasets/fomc_dates.csv。

    Returns:
        DatetimeIndex of FOMC announcement dates.
        FOMC 公告日期的 DatetimeIndex。
    """
    if csv_path is None:
        from utils.paths import DATA_DIR
        csv_path = DATA_DIR / "fomc_dates.csv"
    df = pd.read_csv(csv_path)
    return pd.DatetimeIndex(pd.to_datetime(df["date"]))


def compute_nfp_dates(start_year: int = 2000, end_year: int = 2026) -> pd.DatetimeIndex:
    """Compute Non-Farm Payroll release dates (first Friday of each month).
    计算非农就业数据发布日期（每月第一个周五）。

    NFP is one of the highest-impact macro events for equity volatility.
    The Bureau of Labor Statistics releases it at 8:30 AM ET on the first
    Friday of each month.
    非农是对股票波动率影响最大的宏观事件之一。
    美国劳工统计局于每月第一个周五东部时间 8:30 发布。

    Args:
        start_year: First year to generate dates for (default: 2000).
                    起始年份（默认：2000）。
        end_year:   Last year to generate dates for (default: 2026).
                    结束年份（默认：2026）。

    Returns:
        DatetimeIndex of computed NFP release dates.
        计算得出的非农发布日期 DatetimeIndex。
    """
    dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            first_day = pd.Timestamp(year, month, 1)
            # dayofweek: Monday=0 ... Friday=4
            # 周几：周一=0 ... 周五=4
            days_until_friday = (4 - first_day.dayofweek) % 7
            first_friday = first_day + pd.Timedelta(days=days_until_friday)
            dates.append(first_friday)
    return pd.DatetimeIndex(dates)


def _compute_eve_dates(event_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Compute the business day before each event date (skip weekends).
    计算每个事件日期的前一个工作日（跳过周末）。

    "Eve" effects are important: markets often pre-position before FOMC/NFP,
    leading to compressed range on the eve and explosive range on the event day.
    "前夜"效应很重要：市场通常在 FOMC/NFP 前提前布局，
    导致前一天振幅压缩，事件当天振幅放大。

    Args:
        event_dates: DatetimeIndex of event dates.
                     事件日期的 DatetimeIndex。

    Returns:
        DatetimeIndex of the business day before each event.
        每个事件前一个工作日的 DatetimeIndex。
    """
    eves = []
    for d in event_dates:
        eve = d - pd.Timedelta(days=1)
        # Skip Saturday (5) and Sunday (6) / 跳过周六(5)和周日(6)
        while eve.weekday() >= 5:
            eve -= pd.Timedelta(days=1)
        eves.append(eve)
    return pd.DatetimeIndex(eves)


def compute_days_to_event(
    index: pd.DatetimeIndex, event_dates: pd.DatetimeIndex, default: int = 60
) -> list[int]:
    """For each date in index, compute calendar days until the next event.
    对索引中的每个日期，计算距下一个事件的自然日天数。

    Used as a numeric feature: days_to_fomc, days_to_nfp. Smaller values
    indicate the market is approaching a catalyst, which often increases
    implied volatility and suppresses realized moves (positioning phase).
    用作数值特征：days_to_fomc、days_to_nfp。值越小表示市场越接近催化事件，
    通常隐含波动率上升而实际波动被压制（布局阶段）。

    Args:
        index:       Date index to compute for (daily_metrics.index).
                     要计算的日期索引（daily_metrics.index）。
        event_dates: DatetimeIndex of event dates.
                     事件日期 DatetimeIndex。
        default:     Value to return when no future event exists (default: 60).
                     无后续事件时的默认值（默认：60）。

    Returns:
        List of integer day counts, aligned with index.
        与索引对齐的整数天数列表。
    """
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
    """For each date in index, compute calendar days since the last event.
    对索引中的每个日期，计算距上一个事件的自然日天数。

    Used as a numeric feature: days_since_fomc, days_since_nfp. Larger values
    indicate the market has had time to digest the last data release.
    用作数值特征：days_since_fomc、days_since_nfp。值越大表示市场已有
    足够时间消化上一次数据发布。

    Args:
        index:       Date index to compute for.
                     要计算的日期索引。
        event_dates: DatetimeIndex of event dates.
                     事件日期 DatetimeIndex。
        default:     Value to return when no past event exists (default: 60).
                     无过往事件时的默认值（默认：60）。

    Returns:
        List of integer day counts, aligned with index.
        与索引对齐的整数天数列表。
    """
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
    """Binary flag for earnings season periods.
    财报季时间窗口的布尔标记。

    Earnings season is defined as the ~4-week window when mega-cap tech
    (AAPL, MSFT, GOOG, AMZN, META, NVDA) typically reports quarterly results.
    财报季定义为大型科技股（AAPL、MSFT、GOOG、AMZN、META、NVDA）
    集中发布季报的约 4 周窗口期。

    Windows: Jan 20–31, Feb 1–14, Apr 20–30, May 1–14,
             Jul 20–31, Aug 1–14, Oct 20–31, Nov 1–14.
    窗口期：1/20–1/31、2/1–2/14、4/20–4/30、5/1–5/14、
            7/20–7/31、8/1–8/14、10/20–10/31、11/1–11/14。

    During these periods, QQQ volatility is significantly elevated due to
    single-stock gap risk propagating through the cap-weighted index.
    在这些时段，由于个股跳空风险通过市值加权传导至指数，QQQ 波动率显著升高。

    Args:
        index: Date index (daily_metrics.index).
               日期索引（daily_metrics.index）。

    Returns:
        Series of 0/1 flags, 1 = within earnings season window.
        0/1 标记的 Series，1 = 处于财报季窗口内。
    """
    flag = pd.Series(0, index=index, dtype=int)
    # Late Jan, Apr, Jul, Oct — when companies start reporting
    # 1月下旬、4月下旬、7月下旬、10月下旬 — 公司开始发布财报
    for m, d_start, d_end in [(1, 20, 31), (4, 20, 30), (7, 20, 31), (10, 20, 31)]:
        mask = (index.month == m) & (index.day >= d_start) & (index.day <= d_end)
        flag.loc[mask] = 1
    # First 2 weeks of the following month — peak reporting period
    # 下月前两周 — 财报发布高峰期
    for m, d_end in [(2, 14), (5, 14), (8, 14), (11, 14)]:
        mask = (index.month == m) & (index.day <= d_end)
        flag.loc[mask] = 1
    return flag
