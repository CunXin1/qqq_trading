"""
Update QQQ 1-min parquet files from IBKR.
从 IBKR（盈透证券）更新 QQQ 1 分钟 parquet 数据文件。

Reads existing parquets, finds the gap to the latest closed trading day,
fetches missing 1-min bars from IBKR, and appends them.
读取现有 parquet 文件，计算与最近已收盘交易日之间的数据缺口，
从 IBKR 获取缺失的 1 分钟 K 线，并追加到文件中。

This is the primary data ingestion script. It maintains two parquet files:
这是主要的数据摄入脚本。它维护两个 parquet 文件：
  - QQQ_1min_adjusted.parquet:   Split/dividend adjusted prices (for training).
                                  经拆分/分红调整的价格（用于训练）。
  - QQQ_1min_unadjusted.parquet: Raw unadjusted prices (for live comparison).
                                  原始未调整价格（用于实时对比）。

Workflow / 工作流程:
  1. Read existing parquets, find the last bar date.
     读取现有 parquet，找到最后一根 K 线日期。
  2. Determine the latest fully closed trading day (after-hours ended at 20:00 ET).
     确定最近一个完全收盘的交易日（盘后于美东 20:00 结束）。
  3. Fetch missing 1-min bars from IBKR one day at a time (respecting rate limits).
     从 IBKR 逐天获取缺失的 1 分钟 K 线（遵守速率限制）。
  4. Append new bars and save back to parquet.
     追加新 K 线并保存回 parquet 文件。

IBKR rate limiting / IBKR 速率限制:
  IBKR allows ~60 historical data requests per 10 minutes for 1-min bars.
  The script sleeps 1 second between requests to stay within this limit.
  IBKR 允许每 10 分钟约 60 次 1 分钟 K 线的历史数据请求。
  脚本在每次请求之间等待 1 秒以保持在限制之内。

Usage / 用法:
    python datasets/update_parquet.py                # default: IBKR live port 7496 / 默认实盘端口
    python datasets/update_parquet.py --port 7497    # paper trading / 模拟交易
    python datasets/update_parquet.py --dry-run      # show gap without fetching / 仅显示缺口不获取

Prerequisites / 前置条件:
    - IBKR Trader Workstation or Gateway running. / IBKR TWS 或 Gateway 正在运行。
    - ib_async package installed (pip install ib_async). / 已安装 ib_async 包。
    - Market data subscription for QQQ on IBKR. / IBKR 上 QQQ 的市场数据订阅。
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
    返回盘后交易已结束的最近一个交易日。

    After-hours ends at 20:00 ET. If now is before 20:00 ET on a weekday,
    the last fully closed day is the previous trading day.
    盘后交易于美东时间 20:00 结束。如果当前时间在工作日 20:00 之前，
    则最近一个完全收盘日为前一个交易日。

    Returns:
        date: The most recent fully closed trading day.
              最近一个完全收盘的交易日。
    """
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo

    now_et = datetime.now(ZoneInfo("US/Eastern"))
    d = now_et.date()

    # If it's before 20:00 ET, today's session isn't closed yet
    # 如果在美东 20:00 之前，今天的交易日尚未完全结束
    if now_et.hour < 20:
        d -= timedelta(days=1)

    # Walk back over weekends / 跳过周末
    while d.weekday() >= 5:
        d -= timedelta(days=1)

    return d


async def fetch_1min_range(ib, contract, start_date, end_date):
    """Fetch 1-min bars for a date range, one trading day per request.
    逐天从 IBKR 获取指定日期范围的 1 分钟 K 线。

    Each request fetches one full day (including pre-market and after-hours)
    using useRTH=False. Sleeps 1 second between requests to respect IBKR
    pacing limits (~60 requests / 10 min).
    每次请求获取一整天（含盘前和盘后），使用 useRTH=False。
    请求之间等待 1 秒以遵守 IBKR 速率限制（约 60 次/10 分钟）。

    Args:
        ib:         Connected IB client instance.
                    已连接的 IB 客户端实例。
        contract:   Qualified QQQ stock contract.
                    已验证的 QQQ 股票合约。
        start_date: First date to fetch (inclusive).
                    要获取的起始日期（含）。
        end_date:   Last date to fetch (inclusive).
                    要获取的结束日期（含）。

    Returns:
        DataFrame with columns: open, high, low, close, volume, indexed by datetime.
        以 datetime 为索引的 DataFrame，列包含 open/high/low/close/volume。
    """
    all_bars = []
    d = start_date

    while d <= end_date:
        # Skip weekends / 跳过周末
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
                useRTH=False,         # Include pre-market + after-hours / 包含盘前盘后
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
        # IBKR 速率限制：1 分钟 K 线约 60 次/10 分钟
        await asyncio.sleep(1)
        d += timedelta(days=1)

    if not all_bars:
        return pd.DataFrame()

    data = [{"datetime": pd.Timestamp(b.date), "open": b.open, "high": b.high,
             "low": b.low, "close": b.close, "volume": int(b.volume)} for b in all_bars]
    df = pd.DataFrame(data).set_index("datetime").sort_index()
    # Strip timezone to match existing tz-naive parquets
    # 去除时区信息以匹配现有的无时区 parquet 文件
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="last")]
    return df


async def run(args):
    """Main async workflow: detect gap, fetch from IBKR, append to parquets.
    主异步流程：检测数据缺口，从 IBKR 获取数据，追加到 parquet 文件。
    """
    adj_path = DATA_DIR / "QQQ_1min_adjusted.parquet"
    unadj_path = DATA_DIR / "QQQ_1min_unadjusted.parquet"

    # ── Read existing data / 读取现有数据 ──
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

    # ── Determine gap / 计算数据缺口 ──
    target = last_closed_trading_day()
    earliest_last = min(f["last_date"] for f in files.values())
    start = earliest_last + timedelta(days=1)

    print(f"\n  Last closed trading day: {target}")
    print(f"  Data ends at:           {earliest_last}")

    if start > target:
        print("\n  Already up to date!")
        return

    # Count approximate trading days / 估算交易日数量
    trading_days = sum(1 for i in range((target - earliest_last).days + 1)
                       if (earliest_last + timedelta(days=i)).weekday() < 5) - 1
    print(f"  Gap: {earliest_last} -> {target} (~{trading_days} trading days)")

    if args.dry_run:
        print("\n  [dry-run] No data fetched.")
        return

    # ── Connect to IBKR / 连接 IBKR ──
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

    # ── Fetch missing bars / 获取缺失 K 线 ──
    print(f"\n  Fetching 1-min bars from {start} to {target}...")
    new_bars = await fetch_1min_range(ib, contract, start, target)
    ib.disconnect()

    if new_bars.empty:
        print("\n  No new bars returned from IBKR.")
        return

    print(f"\n  Fetched {len(new_bars):,} new bars "
          f"({new_bars.index[0].date()} to {new_bars.index[-1].date()})")

    # ── Append to both parquets / 追加到两个 parquet 文件 ──
    for name, info in files.items():
        df = info["df"]
        # Only keep bars strictly after existing data / 仅保留现有数据之后的 K 线
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
