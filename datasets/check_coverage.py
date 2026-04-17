"""
Check QQQ 1-min parquet coverage against NYSE calendar.

Compares actual data against the official NYSE trading calendar so that:
  - Real holidays (Good Friday, MLK, Juneteenth, etc.) are NOT flagged as gaps
  - Half-days (9:30-13:00) are reported with their reduced expected bar count
  - Only genuinely missing trading sessions are flagged

Usage:
    python3 datasets/check_coverage.py                  # 2010 - present
    python3 datasets/check_coverage.py --from 2015-01-01
    python3 datasets/check_coverage.py --file unadjusted
"""

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

DATA_DIR = Path(__file__).resolve().parent

PREMARKET_START = (4, 0)
RTH_START = (9, 30)
RTH_END = (16, 0)
AFTERHOURS_END = (20, 0)


def classify_session(ts: pd.Timestamp) -> str:
    hm = (ts.hour, ts.minute)
    if PREMARKET_START <= hm < RTH_START:
        return "pre"
    if RTH_START <= hm < RTH_END:
        return "rth"
    if RTH_END <= hm < AFTERHOURS_END:
        return "post"
    return "other"


def analyze(df: pd.DataFrame) -> pd.DataFrame:
    sessions = df.index.to_series().map(classify_session)
    per = pd.DataFrame({"date": df.index.date, "session": sessions.values})
    counts = per.groupby(["date", "session"]).size().unstack(fill_value=0)
    for c in ("pre", "rth", "post", "other"):
        if c not in counts.columns:
            counts[c] = 0
    counts["total"] = counts[["pre", "rth", "post", "other"]].sum(axis=1)
    return counts.sort_index()


def nyse_schedule(start: date, end: date) -> pd.DataFrame:
    """Return NYSE schedule with expected RTH minutes for each trading day (handles half-days)."""
    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date=start.isoformat(), end_date=end.isoformat())
    # market_open / market_close are in UTC; convert to ET for minute math
    et_open = sched["market_open"].dt.tz_convert("US/Eastern")
    et_close = sched["market_close"].dt.tz_convert("US/Eastern")
    # Expected RTH bars: close - open in minutes (open bar is the first minute)
    expected = ((et_close - et_open).dt.total_seconds() / 60).astype(int)
    out = pd.DataFrame({
        "expected_rth": expected.values,
        "early_close": (expected.values < 390),
    }, index=sched.index.date)
    return out


def report(name: str, path: Path, start: date, end: date):
    if not path.exists():
        print(f"\n[{name}] {path} NOT FOUND")
        return

    df = pd.read_parquet(path)
    df = df[(df.index.date >= start) & (df.index.date <= end)]
    if df.empty:
        print(f"\n[{name}] no bars in range {start}..{end}")
        return

    counts = analyze(df)
    sched = nyse_schedule(start, end)

    # Merge: every NYSE trading day + actual counts (fill 0 if missing)
    merged = sched.join(counts, how="left").fillna(0)
    for c in ("pre", "rth", "post", "total"):
        merged[c] = merged[c].astype(int)

    nyse_days = len(sched)
    data_days = int((merged["total"] > 0).sum())
    missing_sessions = merged[merged["total"] == 0]

    # Pre/post presence — only meaningful for full trading days (not half-days)
    full_days = merged[~merged["early_close"]]
    days_with_pre = int((full_days["pre"] > 0).sum())
    days_with_post = int((full_days["post"] > 0).sum())

    # RTH gap: actual RTH bars < expected - 5 (tolerate a couple missing)
    rth_deficit = merged[(merged["rth"] > 0) & (merged["rth"] < merged["expected_rth"] - 5)]

    # Fully-missing full days (no data at all on a day NYSE said was open)
    truly_missing = merged[merged["total"] == 0]

    print(f"\n=== {name}  ({start} → {end}) ===")
    print(f"  file:                       {path.name}")
    print(f"  NYSE trading days in range: {nyse_days}")
    print(f"  days with any data:         {data_days}")
    print(f"  days with NO data (gap):    {len(truly_missing)}")
    print(f"  NYSE half-days in range:    {int(merged['early_close'].sum())}")
    print(f"  full days w/ pre-market:    {days_with_pre}/{len(full_days)} "
          f"({days_with_pre/len(full_days)*100:.2f}%)")
    print(f"  full days w/ after-hours:   {days_with_post}/{len(full_days)} "
          f"({days_with_post/len(full_days)*100:.2f}%)")
    print(f"  days with RTH deficit (>5): {len(rth_deficit)}")

    if len(truly_missing) > 0:
        print(f"\n  --- trading days with NO data ({len(truly_missing)}) ---")
        for dt, row in truly_missing.iterrows():
            print(f"    {dt}  ({pd.Timestamp(dt).day_name()})  "
                  f"expected_rth={row['expected_rth']}")

    # Full-session days missing pre-market
    missing_pre = full_days[full_days["pre"] == 0]
    if len(missing_pre) > 0:
        print(f"\n  --- full days MISSING pre-market ({len(missing_pre)}; first 30) ---")
        for dt, row in missing_pre.head(30).iterrows():
            print(f"    {dt}  ({pd.Timestamp(dt).day_name()})  total_bars={row['total']}")
        if len(missing_pre) > 30:
            print(f"    ... and {len(missing_pre)-30} more")

    missing_post = full_days[full_days["post"] == 0]
    if len(missing_post) > 0:
        print(f"\n  --- full days MISSING after-hours ({len(missing_post)}; first 30) ---")
        for dt, row in missing_post.head(30).iterrows():
            print(f"    {dt}  ({pd.Timestamp(dt).day_name()})  total_bars={row['total']}")
        if len(missing_post) > 30:
            print(f"    ... and {len(missing_post)-30} more")

    if len(rth_deficit) > 0:
        print(f"\n  --- days with RTH deficit ({len(rth_deficit)}; first 20) ---")
        for dt, row in rth_deficit.head(20).iterrows():
            print(f"    {dt}  rth={row['rth']}/{row['expected_rth']}  "
                  f"short={row['expected_rth']-row['rth']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="start", default="2010-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--to", dest="end", default=None, help="End date (default: today)")
    p.add_argument("--file", choices=["adjusted", "unadjusted", "both"], default="both")
    args = p.parse_args()

    start = pd.Timestamp(args.start).date()
    end = pd.Timestamp(args.end).date() if args.end else date.today()

    targets = []
    if args.file in ("adjusted", "both"):
        targets.append(("adjusted", DATA_DIR / "QQQ_1min_adjusted.parquet"))
    if args.file in ("unadjusted", "both"):
        targets.append(("unadjusted", DATA_DIR / "QQQ_1min_unadjusted.parquet"))

    for name, path in targets:
        report(name, path, start, end)


if __name__ == "__main__":
    main()
