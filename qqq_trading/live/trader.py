"""
Auto-trade QQQ 0DTE straddle based on model signal.

Strategy:
  - At 9:40 AM ET, buy ATM straddle (2 calls + 2 puts = 4 contracts)
  - Constraint: call + put premium <= $3.00/share ($300 for 1 pair)
  - When total position doubles in value, sell half (1 call + 1 put)
  - Keep remaining 1 call + 1 put for manual exit

Usage:
    python -m qqq_trading.live.trader --paper              # paper trading (SAFE)
    python -m qqq_trading.live.trader --paper --dry-run     # simulate only, no orders
    python -m qqq_trading.live.trader --live                # REAL MONEY (careful!)

WARNING: This trades real money when --live is used. Always test with --paper first.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, date, timedelta, time
from pathlib import Path

import pandas as pd
import numpy as np


# ═════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════

MAX_PAIR_COST = 3.00        # max $3.00/share for call+put (= $300 per pair)
NUM_PAIRS = 2               # buy 2 pairs = 4 contracts total
PROFIT_TARGET = 2.0         # sell half when value doubles
POLL_INTERVAL_SEC = 15      # check price every 15 seconds
ENTRY_TIME = time(9, 40)    # enter at 9:40 AM ET
CUTOFF_TIME = time(15, 30)  # stop monitoring at 3:30 PM ET


# ═════════════════════════════════════════════
# Find ATM 0DTE options
# ═════════════════════════════════════════════

async def find_0dte_straddle(ib, underlying_price):
    """Find ATM call + put expiring today."""
    from ib_async import Stock, Option

    today_str = date.today().strftime("%Y%m%d")

    # ATM strike: round to nearest integer
    atm_strike = round(underlying_price)

    # Try a few strikes near ATM to find valid contracts
    call_contract = Option("QQQ", today_str, atm_strike, "C", "SMART")
    put_contract = Option("QQQ", today_str, atm_strike, "P", "SMART")

    try:
        qualified = await ib.qualifyContractsAsync(call_contract, put_contract)
        if len(qualified) < 2:
            # Try 0.5 increments if integer strike doesn't exist
            atm_strike = round(underlying_price * 2) / 2
            call_contract = Option("QQQ", today_str, atm_strike, "C", "SMART")
            put_contract = Option("QQQ", today_str, atm_strike, "P", "SMART")
            qualified = await ib.qualifyContractsAsync(call_contract, put_contract)
    except Exception as e:
        print(f"  Error qualifying options: {e}")
        return None, None, None

    if len(qualified) < 2:
        print(f"  Could not find 0DTE options at strike {atm_strike}")
        return None, None, None

    call_contract, put_contract = qualified[0], qualified[1]
    print(f"  Found: {call_contract.localSymbol} + {put_contract.localSymbol}")
    print(f"  Strike: {atm_strike}, Expiry: {today_str}")

    return call_contract, put_contract, atm_strike


async def get_option_prices(ib, call_contract, put_contract):
    """Get current mid prices for call and put."""
    tickers = await asyncio.gather(
        get_ticker(ib, call_contract),
        get_ticker(ib, put_contract),
    )

    call_ticker, put_ticker = tickers

    call_mid = _mid_price(call_ticker)
    put_mid = _mid_price(put_ticker)

    return call_mid, put_mid


async def get_ticker(ib, contract):
    """Request market data and wait for a quote."""
    ticker = ib.reqMktData(contract, "", False, False)
    # Wait for data to arrive
    for _ in range(50):  # max 5 seconds
        await asyncio.sleep(0.1)
        if ticker.bid is not None and ticker.bid > 0:
            break
    return ticker


def _mid_price(ticker):
    """Calculate mid price from bid/ask."""
    bid = ticker.bid if ticker.bid and ticker.bid > 0 else None
    ask = ticker.ask if ticker.ask and ticker.ask > 0 else None
    if bid and ask:
        return (bid + ask) / 2
    elif ticker.last and ticker.last > 0:
        return ticker.last
    return None


# ═════════════════════════════════════════════
# Order execution
# ═════════════════════════════════════════════

async def place_buy_order(ib, contract, qty, dry_run=False):
    """Place a limit buy order at mid price."""
    from ib_async import LimitOrder

    ticker = await get_ticker(ib, contract)
    mid = _mid_price(ticker)

    if mid is None:
        print(f"    Cannot get price for {contract.localSymbol}")
        return None, None

    # Limit order slightly above mid for better fill
    limit_price = round(mid + 0.02, 2)

    print(f"    {contract.localSymbol}: bid={ticker.bid} ask={ticker.ask} mid={mid:.2f} -> limit={limit_price}")

    if dry_run:
        print(f"    [DRY RUN] Would buy {qty}x {contract.localSymbol} @ {limit_price}")
        return limit_price, None

    order = LimitOrder("BUY", qty, limit_price)
    order.tif = "DAY"
    trade = ib.placeOrder(contract, order)

    # Wait for fill (max 30 seconds)
    for _ in range(60):
        await asyncio.sleep(0.5)
        if trade.orderStatus.status == "Filled":
            fill_price = trade.orderStatus.avgFillPrice
            print(f"    FILLED: {qty}x {contract.localSymbol} @ {fill_price}")
            return fill_price, trade
        elif trade.orderStatus.status in ("Cancelled", "Inactive"):
            print(f"    Order {trade.orderStatus.status}")
            return None, None

    # Not filled in 30s, try to cancel
    print(f"    Order not filled in 30s, cancelling...")
    ib.cancelOrder(order)
    return None, None


async def place_sell_order(ib, contract, qty, dry_run=False):
    """Place a limit sell order at mid price."""
    from ib_async import LimitOrder

    ticker = await get_ticker(ib, contract)
    mid = _mid_price(ticker)

    if mid is None:
        print(f"    Cannot get price for {contract.localSymbol}")
        return None

    # Limit slightly below mid for faster fill
    limit_price = round(mid - 0.02, 2)
    limit_price = max(0.01, limit_price)

    if dry_run:
        print(f"    [DRY RUN] Would sell {qty}x {contract.localSymbol} @ {limit_price}")
        return limit_price

    order = LimitOrder("SELL", qty, limit_price)
    order.tif = "DAY"
    trade = ib.placeOrder(contract, order)

    for _ in range(60):
        await asyncio.sleep(0.5)
        if trade.orderStatus.status == "Filled":
            fill_price = trade.orderStatus.avgFillPrice
            print(f"    SOLD: {qty}x {contract.localSymbol} @ {fill_price}")
            return fill_price
        elif trade.orderStatus.status in ("Cancelled", "Inactive"):
            print(f"    Sell order {trade.orderStatus.status}")
            return None

    print(f"    Sell not filled in 30s, cancelling...")
    ib.cancelOrder(order)
    return None


# ═════════════════════════════════════════════
# Main trading logic
# ═════════════════════════════════════════════

async def run_trade(args):
    from ib_async import IB, Stock

    port = 7497 if args.paper else 7496
    ib = IB()

    print(f"\n{'=' * 60}")
    print(f"QQQ 0DTE STRADDLE TRADER")
    print(f"Mode: {'PAPER' if args.paper else '*** LIVE ***'}")
    print(f"Dry run: {args.dry_run}")
    print(f"Max pair cost: ${MAX_PAIR_COST}/share (${MAX_PAIR_COST * 100:.0f}/pair)")
    print(f"Pairs: {NUM_PAIRS} (= {NUM_PAIRS * 2} contracts)")
    print(f"Profit target: {PROFIT_TARGET}x (sell half)")
    print(f"{'=' * 60}")

    # ── Connect ──
    print(f"\nConnecting to IBKR (port {port})...")
    try:
        await ib.connectAsync("127.0.0.1", port, clientId=20)
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    print(f"Connected. Server v{ib.client.serverVersion()}")

    # ── Get QQQ price ──
    qqq = Stock("QQQ", "SMART", "USD")
    await ib.qualifyContractsAsync(qqq)
    qqq_ticker = await get_ticker(ib, qqq)
    underlying_price = _mid_price(qqq_ticker)

    if underlying_price is None:
        print("Cannot get QQQ price")
        ib.disconnect()
        return

    print(f"\nQQQ: ${underlying_price:.2f}")

    # ── Find 0DTE straddle ──
    print(f"\nFinding 0DTE ATM straddle...")
    call_c, put_c, strike = await find_0dte_straddle(ib, underlying_price)
    if call_c is None:
        print("Cannot find options. Is today a trading day?")
        ib.disconnect()
        return

    # ── Check premiums ──
    print(f"\nChecking premiums...")
    call_mid, put_mid = await get_option_prices(ib, call_c, put_c)

    if call_mid is None or put_mid is None:
        print("Cannot get option prices")
        ib.disconnect()
        return

    pair_cost = call_mid + put_mid
    total_cost = pair_cost * 100 * NUM_PAIRS

    print(f"  Call: ${call_mid:.2f}/share")
    print(f"  Put:  ${put_mid:.2f}/share")
    print(f"  Pair: ${pair_cost:.2f}/share (${pair_cost * 100:.0f} per pair)")
    print(f"  Total: ${total_cost:.0f} for {NUM_PAIRS} pairs")

    if pair_cost > MAX_PAIR_COST:
        print(f"\n  ABORT: Pair cost ${pair_cost:.2f} > limit ${MAX_PAIR_COST:.2f}")
        print(f"  Premiums too expensive. Consider lowering NUM_PAIRS or raising MAX_PAIR_COST.")
        ib.disconnect()
        return

    # ── Place buy orders ──
    print(f"\n{'=' * 60}")
    print(f"ENTERING STRADDLE")
    print(f"{'=' * 60}")

    print(f"\nBuying {NUM_PAIRS} calls...")
    call_fill, call_trade = await place_buy_order(ib, call_c, NUM_PAIRS, args.dry_run)

    print(f"\nBuying {NUM_PAIRS} puts...")
    put_fill, put_trade = await place_buy_order(ib, put_c, NUM_PAIRS, args.dry_run)

    if call_fill is None or put_fill is None:
        print("\nOrder(s) failed. Check TWS for partial fills.")
        ib.disconnect()
        return

    entry_pair_cost = call_fill + put_fill
    entry_total = entry_pair_cost * 100 * NUM_PAIRS

    print(f"\n  Entry: Call={call_fill:.2f} + Put={put_fill:.2f} = {entry_pair_cost:.2f}/share")
    print(f"  Total invested: ${entry_total:.0f}")
    print(f"  Target (2x): ${entry_total * 2:.0f} -> sell {NUM_PAIRS // 2} pair(s)")

    if args.dry_run:
        print(f"\n[DRY RUN] Skipping position monitoring")
        ib.disconnect()
        return

    # ── Monitor for profit target ──
    print(f"\n{'=' * 60}")
    print(f"MONITORING FOR {PROFIT_TARGET}x TARGET")
    print(f"Checking every {POLL_INTERVAL_SEC}s until {CUTOFF_TIME.strftime('%H:%M')} ET")
    print(f"{'=' * 60}")

    sell_pairs = NUM_PAIRS // 2  # sell half
    target_value = entry_pair_cost * PROFIT_TARGET

    while True:
        now = datetime.now()
        # Check if past cutoff
        if now.time() >= CUTOFF_TIME:
            print(f"\n  {now.strftime('%H:%M:%S')} - Past cutoff time. Stopping monitor.")
            print(f"  Remaining {NUM_PAIRS} call(s) + {NUM_PAIRS} put(s) are yours to manage.")
            break

        # Get current prices
        call_now, put_now = await get_option_prices(ib, call_c, put_c)
        if call_now is None or put_now is None:
            await asyncio.sleep(POLL_INTERVAL_SEC)
            continue

        current_pair = call_now + put_now
        current_total = current_pair * 100 * NUM_PAIRS
        pnl_pct = (current_pair / entry_pair_cost - 1) * 100

        sys.stdout.write(
            f"\r  {now.strftime('%H:%M:%S')} | "
            f"C={call_now:.2f} P={put_now:.2f} | "
            f"Pair={current_pair:.2f} | "
            f"Total=${current_total:.0f} | "
            f"P&L={pnl_pct:+.0f}%    "
        )
        sys.stdout.flush()

        # Check target
        if current_pair >= target_value:
            print(f"\n\n  >>> TARGET HIT! {current_pair:.2f} >= {target_value:.2f} ({PROFIT_TARGET}x) <<<")
            print(f"\n  Selling {sell_pairs} pair(s)...")

            print(f"  Selling {sell_pairs} call(s)...")
            await place_sell_order(ib, call_c, sell_pairs)

            print(f"  Selling {sell_pairs} put(s)...")
            await place_sell_order(ib, put_c, sell_pairs)

            remaining = NUM_PAIRS - sell_pairs
            print(f"\n  Done. Remaining: {remaining} call(s) + {remaining} put(s)")
            print(f"  These are yours to manage manually.")
            break

        await asyncio.sleep(POLL_INTERVAL_SEC)

    # ── Cleanup ──
    ib.disconnect()
    print(f"\nDisconnected. Session complete.")


# ═════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="QQQ 0DTE Straddle Auto-Trader")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--paper", action="store_true", help="Paper trading (port 7497)")
    mode.add_argument("--live", action="store_true", help="Live trading (port 7496) - REAL MONEY")

    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate everything but don't place orders")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.live and not args.dry_run:
        print("\n" + "!" * 60)
        print("  WARNING: LIVE TRADING MODE - REAL MONEY")
        print("  This will place real orders on your IBKR account.")
        print("!" * 60)
        confirm = input("\n  Type 'YES' to confirm: ")
        if confirm != "YES":
            print("  Aborted.")
            return

    asyncio.run(run_trade(args))


if __name__ == "__main__":
    main()
