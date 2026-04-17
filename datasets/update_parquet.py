"""
Update QQQ 1-min parquet files from IBKR.

Reads existing parquets, finds the gap to the latest closed trading day,
fetches missing 1-min bars from IBKR, and appends them.

Usage:
    python datasets/update_parquet.py                # default: IBKR live port 7496
    python datasets/update_parquet.py --port 7497    # paper trading
    python datasets/update_parquet.py --dry-run      # show gap without fetching
"""

import argparse
import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent


def last_closed_trading_day():
    """Return the most recent trading day whose after-hours session has ended.

    After-hours ends at 20:00 ET.  If now is before 20:00 ET on a weekday,
    the last fully closed day is the previous trading day.
    """
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo

    now_et = datetime.now(ZoneInfo("US/Eastern"))
    d = now_et.date()

    # If it's before 20:00 ET, today's session isn't closed yet
    if now_et.hour < 20:
        d -= timedelta(days=1)

    # Walk back over weekends
    while d.weekday() >= 5:
        d -= timedelta(days=1)

    return d


async def fetch_1min_range(ib, contract, start_date, end_date):
    """Fetch 1-min bars for a date range, one trading day per request."""
    all_bars = []
    d = start_date

    while d <= end_date:
        # Skip weekends
        if d.weekday() >= 5:
            d += timedelta(days=1)
            continue

        end_str = d.strftime("%Y%m%d") + " 23:59:59 US/Eastern"
        try:
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime=end_str,
                durationStr="1 D",
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=False,
                formatDate=1,
            )
            if bars:
                all_bars.extend(bars)
                print(f"    {d}  +{len(bars)} bars")
            else:
                print(f"    {d}  (no data - holiday?)")
        except Exception as e:
            print(f"    {d}  ERROR: {e}")

        # IBKR pacing: ~60 requests / 10 min for 1-min bars
        await asyncio.sleep(1)
        d += timedelta(days=1)

    if not all_bars:
        return pd.DataFrame()

    data = [{"datetime": pd.Timestamp(b.date), "open": b.open, "high": b.high,
             "low": b.low, "close": b.close, "volume": int(b.volume)} for b in all_bars]
    df = pd.DataFrame(data).set_index("datetime").sort_index()
    # Strip timezone to match existing tz-naive parquets
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="last")]
    return df


async def run(args):
    adj_path = DATA_DIR / "QQQ_1min_adjusted.parquet"
    unadj_path = DATA_DIR / "QQQ_1min_unadjusted.parquet"

    # ── Read existing data ──
    files = {}
    for name, path in [("adjusted", adj_path), ("unadjusted", unadj_path)]:
        if not path.exists():
            print(f"  {name}: file not found, skipping")
            continue
        df = pd.read_parquet(path)
        last_date = df.index[-1].date()
        files[name] = {"path": path, "df": df, "last_date": last_date}
        print(f"  {name}: {len(df):,} bars, last = {last_date}")

    if not files:
        print("No parquet files found in datasets/")
        sys.exit(1)

    # ── Determine gap ──
    target = last_closed_trading_day()
    earliest_last = min(f["last_date"] for f in files.values())
    start = earliest_last + timedelta(days=1)

    print(f"\n  Last closed trading day: {target}")
    print(f"  Data ends at:           {earliest_last}")

    if start > target:
        print("\n  Already up to date!")
        return

    # Count approximate trading days
    trading_days = sum(1 for i in range((target - earliest_last).days + 1)
                       if (earliest_last + timedelta(days=i)).weekday() < 5) - 1
    print(f"  Gap: {earliest_last} -> {target} (~{trading_days} trading days)")

    if args.dry_run:
        print("\n  [dry-run] No data fetched.")
        return

    # ── Connect to IBKR ──
    from ib_async import IB, Stock

    ib = IB()
    print(f"\n  Connecting to IBKR (port {args.port})...")
    await ib.connectAsync("127.0.0.1", args.port, clientId=args.client_id)
    print(f"  Connected (server v{ib.client.serverVersion()})")

    contract = Stock("QQQ", "SMART", "USD")
    qualified = await ib.qualifyContractsAsync(contract)
    if not qualified:
        print("  ERROR: Cannot qualify QQQ contract")
        ib.disconnect()
        sys.exit(1)
    contract = qualified[0]

    # ── Fetch missing bars ──
    print(f"\n  Fetching 1-min bars from {start} to {target}...")
    new_bars = await fetch_1min_range(ib, contract, start, target)
    ib.disconnect()

    if new_bars.empty:
        print("\n  No new bars returned from IBKR.")
        return

    print(f"\n  Fetched {len(new_bars):,} new bars "
          f"({new_bars.index[0].date()} to {new_bars.index[-1].date()})")

    # ── Append to both parquets ──
    for name, info in files.items():
        df = info["df"]
        # Only keep bars strictly after existing data
        append = new_bars[new_bars.index > df.index[-1]]
        if append.empty:
            print(f"\n  {name}: no new bars to append (already covered)")
            continue

        merged = pd.concat([df, append])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged.to_parquet(info["path"])
        print(f"  {name}: +{len(append):,} bars -> {len(merged):,} total "
              f"(now to {merged.index[-1].date()})")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Update datasets/ QQQ 1-min parquets from IBKR")
    parser.add_argument("--port", type=int, default=7496, help="IBKR port (7496=live, 7497=paper)")
    parser.add_argument("--client-id", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", help="Show gap without fetching")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
