"""
Collect QQQ options 1-min OHLCV bars from IBKR.
从 IBKR 采集 QQQ 期权 1 分钟 OHLCV K 线。

Fetches 1-min bars for QQQ options near ATM (within ±pct_range of spot)
for specified expiration dates. Designed for 0DTE/1DTE straddle research.
获取指定到期日、ATM 附近（现价 ±pct_range 范围内）的 QQQ 期权 1 分钟 K 线。
专为 0DTE/1DTE 跨式策略研究设计。

Key constraints / 关键约束:
  - IBKR does NOT provide historical data for expired options.
    IBKR 不提供已过期期权的历史数据。
  - Must collect data BEFORE (or on the day of) expiration.
    必须在到期前（或到期当天）采集数据。
  - Rate limit: ~60 requests / 10 min, script sleeps 1s between requests.
    速率限制：约 60 次/10 分钟，脚本每次请求间隔 1 秒。

Storage / 存储:
  One parquet file per expiry date under datasets/options_1min/:
  每个到期日一个 parquet 文件，存于 datasets/options_1min/：

    datasets/options_1min/
      exp_20260417.parquet   ← 4/17 到期的所有合约的全部交易日数据
      exp_20260420.parquet
      exp_20260421.parquet

  Columns: datetime, strike, right, open, high, low, close, volume
  Why per-expiry: each 0DTE expiry is a self-contained analysis unit;
  querying "all bars for contracts expiring 4/17" = read one file.
  为什么按到期日：每个 0DTE 到期日是一个独立的分析单元；
  查询"4/17 到期的所有 K 线" = 读一个文件。

Usage / 用法:
    # Collect options expiring next Monday & Tuesday (ATM ±5%)
    python datasets/collect_options.py --expiry 20260420 20260421

    # Wider strike range (±10%), paper trading port
    python datasets/collect_options.py --expiry 20260420 --pct-range 0.10 --port 7497

    # Dry run: show what would be collected without fetching
    python datasets/collect_options.py --expiry 20260420 --dry-run

    # Collect today's 0DTE options (run after market close)
    python datasets/collect_options.py --today

Prerequisites / 前置条件:
    - IBKR TWS or Gateway running (weekends OK for historical data).
      IBKR TWS 或 Gateway 运行中（周末也可查历史数据）。
    - ib_async installed.
    - QQQ market data subscription on IBKR.
      IBKR 上有 QQQ 市场数据订阅。
"""

import argparse
import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent
OPTIONS_DIR = DATA_DIR / "options_1min"


async def get_qqq_price(ib):
    """Get QQQ last close price for ATM strike calculation.
    获取 QQQ 最近收盘价，用于计算 ATM 行权价。"""
    from ib_async import Stock

    contract = Stock("QQQ", "SMART", "USD")
    await ib.qualifyContractsAsync(contract)

    bars = await ib.reqHistoricalDataAsync(
        contract, endDateTime="", durationStr="2 D",
        barSizeSetting="1 day", whatToShow="TRADES",
        useRTH=True, formatDate=1,
    )
    if not bars:
        raise RuntimeError("Cannot get QQQ price")

    price = bars[-1].close
    print(f"  QQQ last close: ${price:.2f}")
    return price


async def get_option_chain(ib):
    """Get available expirations and strikes for QQQ options.
    获取 QQQ 期权可用的到期日和行权价列表。"""
    from ib_async import Stock

    qqq = Stock("QQQ", "SMART", "USD")
    await ib.qualifyContractsAsync(qqq)

    chains = await ib.reqSecDefOptParamsAsync(
        qqq.symbol, "", qqq.secType, qqq.conId
    )

    # Find the SMART exchange chain with QQQ trading class
    for chain in chains:
        if chain.exchange == "SMART" and chain.tradingClass == "QQQ":
            return chain

    # Fallback: any SMART chain
    for chain in chains:
        if chain.exchange == "SMART":
            return chain

    raise RuntimeError("Cannot find QQQ option chain")


def filter_strikes(all_strikes, spot_price, pct_range):
    """Filter strikes to those within ±pct_range of spot price.
    筛选现价 ±pct_range 范围内的行权价。

    Only keeps integer strikes (QQQ doesn't have $0.50 increments
    for most expirations, and half-dollar strikes waste API requests).
    只保留整数行权价（QQQ 大部分到期日没有 $0.50 间隔，
    半美元行权价会浪费 API 请求）。
    """
    low = spot_price * (1 - pct_range)
    high = spot_price * (1 + pct_range)
    filtered = sorted(s for s in all_strikes if low <= s <= high and s == int(s))
    return filtered


async def fetch_option_bars(ib, symbol, expiry, strike, right, sleep_sec=1.0):
    """Fetch 1-min bars for a single option contract.
    获取单个期权合约的 1 分钟 K 线。"""
    from ib_async import Option

    contract = Option(symbol, expiry, strike, right, "SMART")
    try:
        qualified = await ib.qualifyContractsAsync(contract)
        if not qualified:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    try:
        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr="5 D",  # covers recent trading days for this contract
            barSizeSetting="1 min",
            whatToShow="TRADES",
            useRTH=False,
            formatDate=1,
        )
    except Exception as e:
        print(f"      ERROR {expiry} {strike}{right}: {e}")
        await asyncio.sleep(sleep_sec)
        return pd.DataFrame()

    await asyncio.sleep(sleep_sec)

    if not bars:
        return pd.DataFrame()

    data = [{
        "datetime": pd.Timestamp(b.date),
        "open": b.open,
        "high": b.high,
        "low": b.low,
        "close": b.close,
        "volume": int(b.volume),
    } for b in bars]

    df = pd.DataFrame(data)
    df["strike"] = strike
    df["right"] = right
    return df


def save_expiry(exp, df):
    """Save one expiry's data to its own parquet file, merging if exists.
    将一个到期日的数据保存到独立 parquet 文件，已有则合并。"""
    OPTIONS_DIR.mkdir(exist_ok=True)
    out_path = OPTIONS_DIR / f"exp_{exp}.parquet"

    df = df.sort_values(["strike", "right", "datetime"])

    if out_path.exists():
        existing = pd.read_parquet(out_path)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["datetime", "strike", "right"], keep="last"
        )
        combined = combined.sort_values(["strike", "right", "datetime"])
        combined.to_parquet(out_path, index=False)
        new_bars = len(combined) - len(existing)
        print(f"    Saved: {out_path.name} (+{new_bars:,} new, total {len(combined):,})")
    else:
        df.to_parquet(out_path, index=False)
        print(f"    Saved: {out_path.name} ({len(df):,} bars)")


async def collect(args):
    """Main collection workflow.
    主采集流程。"""
    from ib_async import IB

    # ── Determine expiry dates ──
    if args.today:
        today_str = date.today().strftime("%Y%m%d")
        expiries = [today_str]
    else:
        expiries = args.expiry

    print(f"\n{'=' * 70}")
    print(f"QQQ OPTIONS 1-MIN DATA COLLECTOR")
    print(f"{'=' * 70}")
    print(f"  Expiries:    {', '.join(expiries)}")
    print(f"  Strike range: ATM ±{args.pct_range:.0%}")
    print(f"  Port:        {args.port}")
    print(f"  Dry run:     {args.dry_run}")
    print(f"  Output:      {OPTIONS_DIR}/exp_<YYYYMMDD>.parquet")

    # ── Connect ──
    ib = IB()
    print(f"\n  Connecting to IBKR (port {args.port})...")
    try:
        await ib.connectAsync("127.0.0.1", args.port, clientId=args.client_id)
    except Exception as e:
        print(f"  Connection failed: {e}")
        print(f"  Make sure IBKR TWS or Gateway is running.")
        sys.exit(1)
    print(f"  Connected (server v{ib.client.serverVersion()})")

    # ── Get spot price and option chain ──
    spot = await get_qqq_price(ib)
    chain = await get_option_chain(ib)

    available_expiries = sorted(chain.expirations)
    all_strikes = chain.strikes

    print(f"\n  Available expiries (next 10): {available_expiries[:10]}")
    print(f"  Total strikes available: {len(all_strikes)}")

    # ── Validate requested expiries ──
    for exp in expiries:
        if exp not in chain.expirations:
            print(f"\n  WARNING: Expiry {exp} not in available chain.")
            print(f"  Available near dates: {[e for e in available_expiries if abs(int(e) - int(exp)) < 10]}")

    # ── Filter strikes ──
    strikes = filter_strikes(all_strikes, spot, args.pct_range)
    print(f"\n  ATM strike: ~${round(spot)}")
    print(f"  Strike range: ${strikes[0]} - ${strikes[-1]} ({len(strikes)} strikes, integers only)")
    print(f"  Rights: C + P")

    total_contracts = len(expiries) * len(strikes) * 2
    est_minutes = total_contracts / 60
    print(f"\n  Total contracts to fetch: {total_contracts}")
    print(f"  Estimated time: ~{est_minutes:.1f} minutes")

    if args.dry_run:
        print(f"\n  [DRY RUN] Would fetch {total_contracts} contracts. Exiting.")
        ib.disconnect()
        return

    # ── Fetch bars per expiry ──
    for exp in expiries:
        print(f"\n  ── Expiry: {exp} ──")
        exp_results = []
        exp_fetched = 0
        exp_empty = 0
        exp_total = len(strikes) * 2

        for strike in strikes:
            for right in ["C", "P"]:
                exp_fetched += 1
                df = await fetch_option_bars(ib, "QQQ", exp, strike, right)
                if df.empty:
                    exp_empty += 1
                    sys.stdout.write(f"\r    [{exp_fetched}/{exp_total}] "
                                     f"{exp} {strike}{right} — no data    ")
                else:
                    exp_results.append(df)
                    sys.stdout.write(f"\r    [{exp_fetched}/{exp_total}] "
                                     f"{exp} {strike}{right} — {len(df)} bars    ")
                sys.stdout.flush()

        print()  # newline after progress

        if not exp_results:
            print(f"    No data for {exp}")
            continue

        exp_df = pd.concat(exp_results, ignore_index=True)

        # Per-expiry summary
        n_strikes = exp_df["strike"].nunique()
        total_vol = exp_df["volume"].sum()
        print(f"    {len(exp_df):,} bars, {n_strikes} strikes, "
              f"volume {total_vol:,.0f}, "
              f"w/data {exp_fetched - exp_empty}/{exp_fetched}")
        print(f"    Range: {exp_df['datetime'].min()} to {exp_df['datetime'].max()}")

        # Top strikes by volume
        vol_top = (exp_df.groupby(["strike", "right"])["volume"]
                   .sum().sort_values(ascending=False).head(5))
        print(f"    Top 5 by volume: ", end="")
        print("  ".join(f"{s}{r}:{v:,.0f}" for (s, r), v in vol_top.items()))

        # Save this expiry
        save_expiry(exp, exp_df)

    ib.disconnect()

    # ── Final summary ──
    print(f"\n{'=' * 70}")
    print(f"DONE")
    print(f"{'=' * 70}")

    if OPTIONS_DIR.exists():
        files = sorted(OPTIONS_DIR.glob("exp_*.parquet"))
        total_size = sum(f.stat().st_size for f in files)
        print(f"  Files in {OPTIONS_DIR}/:")
        for f in files:
            sz = f.stat().st_size
            df_tmp = pd.read_parquet(f)
            print(f"    {f.name:<25} {len(df_tmp):>8,} bars  {sz/1024:.0f} KB")
        print(f"  Total: {len(files)} files, {total_size/1024:.0f} KB")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect QQQ options 1-min bars from IBKR"
    )
    parser.add_argument(
        "--expiry", nargs="+", default=[],
        help="Expiration dates in YYYYMMDD format (e.g. 20260420 20260421)"
    )
    parser.add_argument(
        "--today", action="store_true",
        help="Collect today's 0DTE options"
    )
    parser.add_argument(
        "--pct-range", type=float, default=0.05,
        help="Strike range as fraction of spot (default: 0.05 = ±5%%)"
    )
    parser.add_argument(
        "--port", type=int, default=7497,
        help="IBKR port (7496=live, 7497=paper, default: 7497)"
    )
    parser.add_argument(
        "--client-id", type=int, default=30,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show plan without fetching data"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.expiry and not args.today:
        print("Error: specify --expiry YYYYMMDD or --today")
        sys.exit(1)
    asyncio.run(collect(args))


if __name__ == "__main__":
    main()
