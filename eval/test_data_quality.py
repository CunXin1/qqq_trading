"""
Test live data quality: compare IBKR 1-min aggregation vs historical daily_metrics.
测试实时数据质量：对比IBKR 1分钟聚合数据与历史daily_metrics。

Checks that live data (from IBKR or yfinance) produces values consistent
with the original parquet files used for training. Data inconsistency can
silently degrade model predictions — this script catches it early.
检查实时数据（来自IBKR或yfinance）产生的值是否与训练用的原始parquet文件一致。
数据不一致会悄无声息地降低模型预测质量——本脚本提前发现问题。

Four comparison sections / 四个对比部分:
    1. QQQ DAILY: OHLC prices and derived returns (historical vs live)
       QQQ日线：OHLC价格和衍生收益率（历史 vs 实时）
    2. PREMARKET: Premarket range and return comparison
       盘前：盘前范围和收益率对比
    3. VIX: Cross-check live VIX vs yfinance download
       VIX：实时VIX与yfinance下载的交叉验证
    4. 1-MIN BARS: Bar count, gaps, OHLC consistency, data integrity
       1分钟K线：K线数量、缺口、OHLC一致性、数据完整性

Status thresholds / 状态阈值:
    OK   - Difference within acceptable range / 差异在可接受范围内
    WARN - Small discrepancy, investigate / 小差异，需调查
    FAIL - Significant mismatch, data problem / 显著不匹配，数据问题

    For OHLC prices: OK < 0.1% diff, WARN < 1.0%, FAIL >= 1.0%
    对于OHLC价格：OK < 0.1%差异，WARN < 1.0%，FAIL >= 1.0%

    For returns: OK < 0.1% abs diff, WARN < 0.5%, FAIL >= 0.5%
    对于收益率：OK < 0.1%绝对差异，WARN < 0.5%，FAIL >= 0.5%

Usage / 用法:
    python eval/test_data_quality.py              # compare overlapping dates / 对比重叠日期

Prerequisites / 前置条件:
    Live data must be fetched first: / 必须先获取实时数据：
        python -m live.fetch_data --merge --csv
    This creates CSV files in output/live/ that this script reads.
    这会在output/live/创建本脚本读取的CSV文件。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import pandas as pd
import numpy as np
from utils.paths import OUTPUT_DIR


def load_historical():
    """Load the original daily_metrics.parquet (ground truth).
    加载原始daily_metrics.parquet（真值数据）。"""
    df = pd.read_parquet(OUTPUT_DIR / "daily_metrics.parquet")
    df.index = pd.to_datetime(df.index)
    return df


def load_live():
    """Load live-fetched CSV files from output/live/ directory.
    从output/live/目录加载实时获取的CSV文件。

    Expected files / 期望的文件:
        live_qqq.csv      - QQQ daily OHLCV / QQQ日线OHLCV
        live_premarket.csv - Premarket session data / 盘前交易数据
        live_vix.csv       - VIX data / VIX数据
        live_yields.csv    - Treasury yield data / 国债收益率数据
        live_qqq_1min.csv  - QQQ 1-minute bars / QQQ 1分钟K线
    """
    live_dir = OUTPUT_DIR / "live"
    result = {}

    for name, file in [("qqq", "live_qqq.csv"), ("premarket", "live_premarket.csv"),
                        ("vix", "live_vix.csv"), ("yields", "live_yields.csv"),
                        ("qqq_1min", "live_qqq_1min.csv")]:
        path = live_dir / file
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            result[name] = df
            print(f"  Loaded {name}: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
        else:
            print(f"  {file} not found / 未找到")
            result[name] = pd.DataFrame()

    return result


def compare_qqq(hist, live_qqq):
    """Compare QQQ daily OHLC fields between historical and live data.
    对比历史和实时数据中QQQ日线OHLC字段。

    Checks both raw prices (open/high/low/close) and derived metrics
    (returns, range, gap). Price differences > 0.1% suggest a data
    source mismatch or adjustment issue.
    检查原始价格和衍生指标。价格差异>0.1%表明数据源不匹配或调整问题。
    """
    # Find overlapping dates between historical and live data
    # 找到历史和实时数据的重叠日期
    overlap = hist.index.intersection(live_qqq.index)
    if len(overlap) == 0:
        print("\n  NO OVERLAPPING DATES — cannot compare / 无重叠日期——无法对比")
        print(f"  Historical ends: {hist.index[-1].date()}")
        print(f"  Live starts:     {live_qqq.index[0].date()}")
        return

    print(f"\n  Overlapping dates: {len(overlap)}")
    print(f"  Range: {overlap[0].date()} to {overlap[-1].date()}")

    # Field mapping: historical column name -> live column name
    # 字段映射：历史列名 -> 实时列名
    field_map = {
        "reg_open": "open",
        "reg_high": "high",
        "reg_low": "low",
        "reg_close": "close",
    }

    print(f"\n  {'Field':<20} {'Hist':>10} {'Live':>10} {'Diff':>10} {'Diff%':>8} {'Status':>8}")
    print(f"  {'-' * 68}")

    issues = []
    for hist_col, live_col in field_map.items():
        if hist_col not in hist.columns or live_col not in live_qqq.columns:
            continue

        for dt in overlap:
            h_val = hist.loc[dt, hist_col]
            l_val = live_qqq.loc[dt, live_col]

            if pd.isna(h_val) or pd.isna(l_val):
                continue

            diff = abs(h_val - l_val)
            diff_pct = diff / h_val * 100

            # Status thresholds for OHLC prices / OHLC价格的状态阈值
            # OK: < 0.1% (rounding diff) / 四舍五入差异
            # WARN: 0.1% - 1.0% (possible adjustment diff) / 可能的调整差异
            # FAIL: > 1.0% (data source problem) / 数据源问题
            status = "OK" if diff_pct < 0.1 else ("WARN" if diff_pct < 1.0 else "FAIL")
            if status != "OK":
                issues.append((dt.date(), hist_col, h_val, l_val, diff_pct))

            print(f"  {str(dt.date()) + ' ' + hist_col:<20} {h_val:>10.2f} {l_val:>10.2f} {diff:>10.4f} {diff_pct:>7.4f}% {status:>8}")

    # Compare derived fields (returns, range, gap) / 对比衍生字段（收益率、范围、缺口）
    # These are more sensitive — even small OHLC diffs can compound in returns
    # 这些更敏感——即使小的OHLC差异也会在收益率中放大
    derived_map = {
        "close_to_close_ret": "close_to_close_ret",
        "open_to_close_ret": "open_to_close_ret",
        "intraday_range": "intraday_range",
        "gap_return": "gap_return",
    }

    print(f"\n  {'Derived Field':<25} {'Hist':>10} {'Live':>10} {'AbsDiff':>10} {'Status':>8}")
    print(f"  {'-' * 65}")

    for col in derived_map:
        if col not in hist.columns or col not in live_qqq.columns:
            continue
        for dt in overlap:
            h_val = hist.loc[dt, col]
            l_val = live_qqq.loc[dt, col]
            if pd.isna(h_val) or pd.isna(l_val):
                continue

            diff = abs(h_val - l_val)
            # For returns: diff > 0.001 (0.1%) is concerning
            # 对于收益率：差异 > 0.001 (0.1%) 值得关注
            status = "OK" if diff < 0.001 else ("WARN" if diff < 0.005 else "FAIL")
            if status != "OK":
                issues.append((dt.date(), col, h_val, l_val, diff * 100))

            print(f"  {str(dt.date()) + ' ' + col:<25} {h_val:>10.4f} {l_val:>10.4f} {diff:>10.6f} {status:>8}")

    return issues


def compare_premarket(hist, live_pre):
    """Compare premarket fields (range, return) between historical and live.
    对比历史和实时数据中的盘前字段（范围、收益率）。"""
    overlap = hist.index.intersection(live_pre.index)
    if len(overlap) == 0:
        print("\n  NO OVERLAPPING DATES for premarket / 盘前数据无重叠日期")
        return

    print(f"\n  Premarket overlapping dates: {len(overlap)}")

    fields = ["premarket_range", "premarket_ret"]

    print(f"\n  {'Field':<25} {'Hist':>10} {'Live':>10} {'AbsDiff':>10} {'Status':>8}")
    print(f"  {'-' * 65}")

    for col in fields:
        if col not in hist.columns or col not in live_pre.columns:
            continue
        for dt in overlap:
            h_val = hist.loc[dt, col]
            l_val = live_pre.loc[dt, col]
            if pd.isna(h_val) or pd.isna(l_val):
                continue

            diff = abs(h_val - l_val)
            # Premarket has wider spreads, so tolerance is higher
            # 盘前价差更大，所以容差更高
            status = "OK" if diff < 0.002 else ("WARN" if diff < 0.01 else "FAIL")

            print(f"  {str(dt.date()) + ' ' + col:<25} {h_val:>10.4f} {l_val:>10.4f} {diff:>10.6f} {status:>8}")


def compare_vix(live_vix):
    """Cross-check VIX live data against yfinance as independent source.
    用yfinance作为独立数据源交叉验证VIX实时数据。

    VIX is critical for VRP calculation — if live VIX is wrong,
    the entire prediction pipeline produces garbage.
    VIX对VRP计算至关重要——如果实时VIX错误，整个预测管道产出垃圾。
    """
    try:
        import yfinance as yf
        from datetime import timedelta

        if live_vix.empty:
            print("\n  No live VIX data / 无实时VIX数据")
            return

        start = (live_vix.index[0] - timedelta(days=3)).strftime("%Y-%m-%d")
        end = (live_vix.index[-1] + timedelta(days=1)).strftime("%Y-%m-%d")

        yf_vix = yf.download("^VIX", start=start, end=end, progress=False)
        if isinstance(yf_vix.columns, pd.MultiIndex):
            yf_vix.columns = [c[0].lower() for c in yf_vix.columns]
        else:
            yf_vix.columns = [c.lower() for c in yf_vix.columns]

        overlap = live_vix.index.intersection(yf_vix.index)
        if len(overlap) == 0:
            print("\n  No VIX overlap dates for cross-check / VIX无重叠日期可交叉验证")
            return

        print(f"\n  VIX cross-check ({len(overlap)} dates):")
        print(f"  {'Date':<12} {'Live':>8} {'YF':>8} {'Diff':>8} {'Status':>8}")
        print(f"  {'-' * 48}")

        for dt in overlap:
            live_val = live_vix.loc[dt, "vix_close"]
            yf_val = yf_vix.loc[dt, "close"]
            diff = abs(live_val - yf_val)
            # VIX: 0.1 point diff is normal rounding; > 0.5 is suspicious
            # VIX：0.1点差异是正常舍入；> 0.5可疑
            status = "OK" if diff < 0.1 else ("WARN" if diff < 0.5 else "FAIL")
            print(f"  {str(dt.date()):<12} {live_val:>8.2f} {yf_val:>8.2f} {diff:>8.4f} {status:>8}")

    except Exception as e:
        print(f"\n  VIX cross-check failed / VIX交叉验证失败: {e}")


def check_1min_bars(qqq_1min):
    """Validate 1-min bar data quality: completeness, gaps, OHLC consistency.
    验证1分钟K线数据质量：完整性、缺口、OHLC一致性。

    Checks / 检查项:
        - Bar count per day (expect ~390 for regular session 9:30-16:00)
          每日K线数量（常规交易时段9:30-16:00预期约390根）
        - Time gaps > 2 minutes during regular hours (data feed interruptions)
          常规交易时段中>2分钟的时间缺口（数据馈送中断）
        - Negative volume (data corruption)
          负成交量（数据损坏）
        - Zero volume (illiquid bars or feed issues)
          零成交量（流动性差或馈送问题）
        - Null prices (missing data)
          价格空值（数据缺失）
        - OHLC violations: high must >= open,close; low must <= open,close
          OHLC违规：最高价必须 >= 开盘/收盘；最低价必须 <= 开盘/收盘
    """
    if qqq_1min.empty:
        print("\n  No 1-min bar data / 无1分钟K线数据")
        return

    print(f"\n  1-min bars: {len(qqq_1min)} total")

    # Per-day statistics / 每日统计
    qqq_1min.index = pd.to_datetime(qqq_1min.index, utc=True).tz_convert("US/Eastern")
    daily_counts = qqq_1min.groupby(qqq_1min.index.date).size()

    print(f"\n  {'Date':<12} {'Bars':>6} {'First':>10} {'Last':>10} {'Gaps':>6} {'Status':>8}")
    print(f"  {'-' * 56}")

    for day, count in daily_counts.items():
        day_data = qqq_1min.loc[qqq_1min.index.date == day]
        first = day_data.index[0].strftime("%H:%M")
        last = day_data.index[-1].strftime("%H:%M")

        # Check for gaps > 2 minutes during regular hours (9:30-16:00)
        # 检查常规交易时段中 > 2分钟的缺口
        regular = day_data.between_time("09:30", "15:59")
        if len(regular) > 1:
            time_diffs = regular.index.to_series().diff().dt.total_seconds().dropna()
            gaps = (time_diffs > 120).sum()
        else:
            gaps = 0

        expected_regular = 390  # 6.5 hours * 60 min / 6.5小时 * 60分钟
        status = "OK" if len(regular) >= expected_regular - 5 else "WARN"

        print(f"  {day}  {count:>6} {first:>10} {last:>10} {gaps:>6} {status:>8}")

    # Data integrity checks / 数据完整性检查
    neg_vol = (qqq_1min["volume"] < 0).sum()      # should be 0 / 应为0
    zero_vol = (qqq_1min["volume"] == 0).sum()     # some zeros normal in extended hours / 延长时段中少量零正常
    null_price = qqq_1min[["open", "high", "low", "close"]].isnull().sum().sum()  # should be 0 / 应为0

    print(f"\n  Negative volume bars / 负成交量K线: {neg_vol}")
    print(f"  Zero volume bars / 零成交量K线:    {zero_vol}")
    print(f"  Null price bars / 价格空值K线:     {null_price}")

    # OHLC consistency: high >= max(open, close), low <= min(open, close)
    # OHLC一致性：最高价 >= max(开盘,收盘)，最低价 <= min(开盘,收盘)
    ohlc_violations = (
        (qqq_1min["high"] < qqq_1min["open"]) |
        (qqq_1min["high"] < qqq_1min["close"]) |
        (qqq_1min["low"] > qqq_1min["open"]) |
        (qqq_1min["low"] > qqq_1min["close"])
    ).sum()
    print(f"  OHLC violations / OHLC违规:     {ohlc_violations}")


def main():
    parser = argparse.ArgumentParser(description="Test live data quality")
    parser.parse_args()

    print("Loading historical daily_metrics (ground truth)...")
    print("加载历史daily_metrics（真值数据）...")
    hist = load_historical()
    print(f"  {len(hist)} days, {hist.index[0].date()} to {hist.index[-1].date()}")

    print("\nLoading live data... / 加载实时数据...")
    live = load_live()

    # ── Section 1: QQQ daily comparison / 第一部分：QQQ日线对比 ──
    print("\n" + "=" * 70)
    print("QQQ DAILY: Historical vs Live / QQQ日线：历史 vs 实时")
    print("=" * 70)
    if not live["qqq"].empty:
        issues = compare_qqq(hist, live["qqq"])
    else:
        print("  No live QQQ data / 无实时QQQ数据")

    # ── Section 2: Premarket comparison / 第二部分：盘前对比 ──
    print("\n" + "=" * 70)
    print("PREMARKET: Historical vs Live / 盘前：历史 vs 实时")
    print("=" * 70)
    if not live["premarket"].empty:
        compare_premarket(hist, live["premarket"])
    else:
        print("  No live premarket data / 无实时盘前数据")

    # ── Section 3: VIX cross-check / 第三部分：VIX交叉验证 ──
    print("\n" + "=" * 70)
    print("VIX: Live vs yfinance cross-check / VIX：实时 vs yfinance 交叉验证")
    print("=" * 70)
    compare_vix(live["vix"])

    # ── Section 4: 1-min bar quality / 第四部分：1分钟K线质量 ──
    print("\n" + "=" * 70)
    print("1-MIN BARS: Quality Check / 1分钟K线：质量检查")
    print("=" * 70)
    check_1min_bars(live["qqq_1min"])

    print("\n" + "=" * 70)
    print("DONE / 完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
