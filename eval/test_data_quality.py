"""
Test live data quality: compare IBKR 1-min aggregation vs historical daily_metrics.
Checks that fetch_data produces data consistent with the original parquet.

Usage:
    python eval/test_data_quality.py              # compare overlapping dates
    python eval/test_data_quality.py --days 30    # more overlap days
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import pandas as pd
import numpy as np
from qqq_trading.utils.paths import OUTPUT_DIR


def load_historical():
    """Load the original daily_metrics.parquet (ground truth)."""
    df = pd.read_parquet(OUTPUT_DIR / "daily_metrics.parquet")
    df.index = pd.to_datetime(df.index)
    return df


def load_live():
    """Load live-fetched CSV files."""
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
            print(f"  {file} not found")
            result[name] = pd.DataFrame()

    return result


def compare_qqq(hist, live_qqq):
    """Compare QQQ daily fields between historical and live."""
    # Find overlapping dates
    overlap = hist.index.intersection(live_qqq.index)
    if len(overlap) == 0:
        print("\n  NO OVERLAPPING DATES — cannot compare")
        print(f"  Historical ends: {hist.index[-1].date()}")
        print(f"  Live starts:     {live_qqq.index[0].date()}")
        return

    print(f"\n  Overlapping dates: {len(overlap)}")
    print(f"  Range: {overlap[0].date()} to {overlap[-1].date()}")

    # Field mapping: historical name -> live name
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

            status = "OK" if diff_pct < 0.1 else ("WARN" if diff_pct < 1.0 else "FAIL")
            if status != "OK":
                issues.append((dt.date(), hist_col, h_val, l_val, diff_pct))

            print(f"  {str(dt.date()) + ' ' + hist_col:<20} {h_val:>10.2f} {l_val:>10.2f} {diff:>10.4f} {diff_pct:>7.4f}% {status:>8}")

    # Compare derived fields
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
            # For returns, diff > 0.001 (0.1%) is concerning
            status = "OK" if diff < 0.001 else ("WARN" if diff < 0.005 else "FAIL")
            if status != "OK":
                issues.append((dt.date(), col, h_val, l_val, diff * 100))

            print(f"  {str(dt.date()) + ' ' + col:<25} {h_val:>10.4f} {l_val:>10.4f} {diff:>10.6f} {status:>8}")

    return issues


def compare_premarket(hist, live_pre):
    """Compare premarket fields."""
    overlap = hist.index.intersection(live_pre.index)
    if len(overlap) == 0:
        print("\n  NO OVERLAPPING DATES for premarket")
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
            status = "OK" if diff < 0.002 else ("WARN" if diff < 0.01 else "FAIL")

            print(f"  {str(dt.date()) + ' ' + col:<25} {h_val:>10.4f} {l_val:>10.4f} {diff:>10.6f} {status:>8}")


def compare_vix(live_vix):
    """Cross-check VIX live data with yfinance."""
    try:
        import yfinance as yf
        from datetime import timedelta

        if live_vix.empty:
            print("\n  No live VIX data")
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
            print("\n  No VIX overlap dates for cross-check")
            return

        print(f"\n  VIX cross-check ({len(overlap)} dates):")
        print(f"  {'Date':<12} {'Live':>8} {'YF':>8} {'Diff':>8} {'Status':>8}")
        print(f"  {'-' * 48}")

        for dt in overlap:
            live_val = live_vix.loc[dt, "vix_close"]
            yf_val = yf_vix.loc[dt, "close"]
            diff = abs(live_val - yf_val)
            status = "OK" if diff < 0.1 else ("WARN" if diff < 0.5 else "FAIL")
            print(f"  {str(dt.date()):<12} {live_val:>8.2f} {yf_val:>8.2f} {diff:>8.4f} {status:>8}")

    except Exception as e:
        print(f"\n  VIX cross-check failed: {e}")


def check_1min_bars(qqq_1min):
    """Validate 1-min bar data quality."""
    if qqq_1min.empty:
        print("\n  No 1-min bar data")
        return

    print(f"\n  1-min bars: {len(qqq_1min)} total")

    # Per-day stats
    qqq_1min.index = pd.to_datetime(qqq_1min.index, utc=True).tz_convert("US/Eastern")
    daily_counts = qqq_1min.groupby(qqq_1min.index.date).size()

    print(f"\n  {'Date':<12} {'Bars':>6} {'First':>10} {'Last':>10} {'Gaps':>6} {'Status':>8}")
    print(f"  {'-' * 56}")

    for day, count in daily_counts.items():
        day_data = qqq_1min.loc[qqq_1min.index.date == day]
        first = day_data.index[0].strftime("%H:%M")
        last = day_data.index[-1].strftime("%H:%M")

        # Check for gaps > 2 minutes during regular hours (9:30-16:00)
        regular = day_data.between_time("09:30", "15:59")
        if len(regular) > 1:
            time_diffs = regular.index.to_series().diff().dt.total_seconds().dropna()
            gaps = (time_diffs > 120).sum()  # gaps > 2 min
        else:
            gaps = 0

        expected_regular = 390  # 6.5 hours * 60 min
        status = "OK" if len(regular) >= expected_regular - 5 else "WARN"

        print(f"  {day}  {count:>6} {first:>10} {last:>10} {gaps:>6} {status:>8}")

    # Check for negative volume
    neg_vol = (qqq_1min["volume"] < 0).sum()
    zero_vol = (qqq_1min["volume"] == 0).sum()
    null_price = qqq_1min[["open", "high", "low", "close"]].isnull().sum().sum()

    print(f"\n  Negative volume bars: {neg_vol}")
    print(f"  Zero volume bars:    {zero_vol}")
    print(f"  Null price bars:     {null_price}")

    # OHLC consistency: high >= open, close, low; low <= open, close
    ohlc_violations = (
        (qqq_1min["high"] < qqq_1min["open"]) |
        (qqq_1min["high"] < qqq_1min["close"]) |
        (qqq_1min["low"] > qqq_1min["open"]) |
        (qqq_1min["low"] > qqq_1min["close"])
    ).sum()
    print(f"  OHLC violations:     {ohlc_violations}")


def main():
    parser = argparse.ArgumentParser(description="Test live data quality")
    parser.parse_args()

    print("Loading historical daily_metrics (ground truth)...")
    hist = load_historical()
    print(f"  {len(hist)} days, {hist.index[0].date()} to {hist.index[-1].date()}")

    print("\nLoading live data...")
    live = load_live()

    # ── QQQ daily comparison ──
    print("\n" + "=" * 70)
    print("QQQ DAILY: Historical vs Live")
    print("=" * 70)
    if not live["qqq"].empty:
        issues = compare_qqq(hist, live["qqq"])
    else:
        print("  No live QQQ data")

    # ── Premarket comparison ──
    print("\n" + "=" * 70)
    print("PREMARKET: Historical vs Live")
    print("=" * 70)
    if not live["premarket"].empty:
        compare_premarket(hist, live["premarket"])
    else:
        print("  No live premarket data")

    # ── VIX cross-check ──
    print("\n" + "=" * 70)
    print("VIX: Live vs yfinance cross-check")
    print("=" * 70)
    compare_vix(live["vix"])

    # ── 1-min bar quality ──
    print("\n" + "=" * 70)
    print("1-MIN BARS: Quality Check")
    print("=" * 70)
    check_1min_bars(live["qqq_1min"])

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
