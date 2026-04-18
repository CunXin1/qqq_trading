"""Event calendar: FOMC, NFP, CPI, PCE, mega-cap earnings dates.
事件日历：FOMC、非农（NFP）、CPI、PCE、大型股财报日期。

Macro events systematically drive QQQ volatility. This module provides:
宏观事件系统性地驱动 QQQ 波动率。本模块提供：

1. FOMC dates       — loaded from a curated CSV (datasets/fomc_dates.csv).
   FOMC 日期        — 从手工维护的 CSV 文件加载。
2. NFP dates        — computed algorithmically (first Friday of each month).
   非农日期          — 通过算法计算（每月第一个周五）。
3. CPI release dates — computed algorithmically (~10th-13th of each month).
   CPI 公布日        — 通过算法计算（每月约10-13日）。
4. PCE release dates — computed algorithmically (~last Friday of each month).
   PCE 公布日        — 通过算法计算（每月约最后一个周五）。
5. Eve dates        — the business day before an event (for "day-before" effects).
   前夜日期          — 事件前一个工作日（用于捕捉"事件前一天"效应）。
6. Days-to / since  — countdown/countup features for proximity to events.
   距事件天数         — 距下次事件天数 / 距上次事件天数（用作特征）。
7. Earnings season  — binary flag for the ~4-week window around mega-cap earnings.
   财报季标记         — 大型股集中发布财报的约 4 周窗口期布尔标记。
8. Mega-cap earnings — exact dates from yfinance for NVDA/AAPL/MSFT.
   大型股财报日        — 从 yfinance 获取 NVDA/AAPL/MSFT 的精确财报日期。

These are consumed by features/external.py to build event-aware features.
这些被 features/external.py 使用，构建事件感知特征。
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


def compute_cpi_dates(start_year: int = 2000, end_year: int = 2027) -> pd.DatetimeIndex:
    """Approximate CPI release dates (~2nd week of each month, usually Tuesday-Thursday).
    近似 CPI 公布日期（每月约第二周，通常为周二至周四）。

    BLS releases CPI around the 10th-13th of each month at 8:30 AM ET.
    This approximation uses the 2nd Tuesday-Wednesday of each month.
    Since the exact pattern varies, this covers ~90% of actual dates within 1 day.
    BLS 在每月 10-13 日左右东部时间 8:30 公布 CPI。
    此近似使用每月第二个周二/周三，覆盖约 90% 的实际日期（误差 1 天以内）。
    """
    dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            first_day = pd.Timestamp(year, month, 1)
            # Target: ~10th-13th. Use 2nd Tuesday (dayofweek=1)
            days_to_tuesday = (1 - first_day.dayofweek) % 7
            first_tuesday = first_day + pd.Timedelta(days=days_to_tuesday)
            second_tuesday = first_tuesday + pd.Timedelta(days=7)
            # CPI is usually the day after the 2nd Tuesday (Wednesday)
            cpi_day = second_tuesday + pd.Timedelta(days=1)
            dates.append(cpi_day)
    return pd.DatetimeIndex(dates)


def compute_pce_dates(start_year: int = 2000, end_year: int = 2027) -> pd.DatetimeIndex:
    """Approximate PCE release dates (~last Friday of each month).
    近似 PCE 公布日期（每月约最后一个周五）。

    BEA releases Personal Income & Outlays (which includes PCE) at the
    end of each month, typically the last Thursday or Friday.
    BEA 在每月末发布个人收入与支出报告（含 PCE），通常在最后一个周四或周五。
    """
    dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Last day of month
            if month == 12:
                last_day = pd.Timestamp(year, 12, 31)
            else:
                last_day = pd.Timestamp(year, month + 1, 1) - pd.Timedelta(days=1)
            # Find last Friday
            days_back = (last_day.dayofweek - 4) % 7
            last_friday = last_day - pd.Timedelta(days=days_back)
            dates.append(last_friday)
    return pd.DatetimeIndex(dates)


def load_megacap_earnings(tickers: list[str] | None = None,
                          cache_path: Path | None = None) -> pd.DataFrame:
    """Load historical earnings dates for QQQ mega-cap weights from yfinance.
    从 yfinance 加载 QQQ 大权重股的历史财报日期。

    Returns DataFrame with columns: date, ticker.
    返回包含 date 和 ticker 列的 DataFrame。

    Caches results to CSV to avoid repeated API calls.
    结果缓存至 CSV 以避免重复 API 调用。
    """
    if tickers is None:
        tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOG",
                   "META", "TSLA", "BRK-B", "AVGO", "JPM"]

    if cache_path is None:
        from utils.paths import DATA_DIR
        cache_path = DATA_DIR / "megacap_earnings.csv"

    if cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=["date"])
        return df

    import yfinance as yf

    rows = []
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            cal = t.get_earnings_dates(limit=100)
            if cal is not None:
                for dt in cal.index:
                    hour = dt.hour if hasattr(dt, 'hour') else 16
                    # BMO (before market open): hour < 12; AMC (after market close): hour >= 12
                    timing = "BMO" if hour < 12 else "AMC"
                    rows.append({
                        "date": dt.tz_localize(None).normalize(),
                        "ticker": ticker,
                        "timing": timing,
                    })
        except Exception:
            pass

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["date", "ticker"]).sort_values("date").reset_index(drop=True)
        df.to_csv(cache_path, index=False)
    return df


def build_megacap_earnings_flags(index: pd.DatetimeIndex,
                                 earnings_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-ticker earnings impact day flags, respecting AMC/BMO timing.
    构建每个标的的财报波动日标记，区分盘后(AMC)和盘前(BMO)发布。

    AMC (after market close, e.g. AAPL/NVDA): impact = NEXT trading day.
    BMO (before market open, e.g. JPM/BRK-B): impact = SAME day.
    AMC（盘后发布，如 AAPL/NVDA）：波动日 = 次日。
    BMO（盘前发布，如 JPM/BRK-B）：波动日 = 当日。

    Returns DataFrame with columns:
      {ticker}_earnings       — earnings release day (for reference)
      {ticker}_impact         — actual volatility impact day (AMC→next, BMO→same)
      any_megacap_earnings    — any company releasing
      any_megacap_impact      — any company's impact day
    """
    flags = pd.DataFrame(index=index)

    if earnings_df.empty:
        return flags

    has_timing = "timing" in earnings_df.columns

    all_impact_dates = set()

    for ticker in earnings_df["ticker"].unique():
        tk_lower = ticker.lower().replace("-", "")  # BRK-B → brkb
        tk_data = earnings_df[earnings_df["ticker"] == ticker]
        tk_dates = set(pd.to_datetime(tk_data["date"]).dt.normalize())

        col_day = f"{tk_lower}_earnings"
        col_impact = f"{tk_lower}_impact"
        flags[col_day] = index.isin(tk_dates).astype(int)

        # Compute impact dates based on timing
        impact_dates = set()
        for _, row in tk_data.iterrows():
            d = pd.Timestamp(row["date"]).normalize()
            timing = row.get("timing", "AMC") if has_timing else "AMC"
            if timing == "BMO":
                # Before market open → impact is same day
                impact_dates.add(d)
            else:
                # After market close → impact is next business day
                nxt = d + pd.Timedelta(days=1)
                while nxt.weekday() >= 5:
                    nxt += pd.Timedelta(days=1)
                impact_dates.add(nxt)
            all_impact_dates.add(d)  # release day for any_megacap_earnings

        flags[col_impact] = index.isin(impact_dates).astype(int)
        all_impact_dates.update(impact_dates)

    # Any mega-cap flags
    day_cols = [c for c in flags.columns if c.endswith("_earnings")]
    impact_cols = [c for c in flags.columns if c.endswith("_impact")]
    if day_cols:
        flags["any_megacap_earnings"] = flags[day_cols].max(axis=1)
    if impact_cols:
        flags["any_megacap_impact"] = flags[impact_cols].max(axis=1)

    return flags
