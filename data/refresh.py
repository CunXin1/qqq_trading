"""Auto-refresh data: check staleness, fetch via IBKR, validate, merge.
自动刷新数据：检查过期、通过 IBKR 获取、验证、合并。

Called on server startup or via CLI. Uses IBKR for full-fidelity 1-min data
(max_drawdown, max_runup, VWAP, premarket), with yfinance fallback for
external data (VIX, VVIX, SKEW, yields).
在服务器启动或通过 CLI 调用。使用 IBKR 获取完整精度的 1 分钟数据
（最大回撤、最大反弹、VWAP、盘前），yfinance 用于外部数据。

Usage / 用法:
    from data.refresh import refresh_if_stale
    await refresh_if_stale()                   # check + refresh
    await refresh_if_stale(force=True)         # force refresh
"""
from __future__ import annotations

import asyncio
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np


def _last_trading_day() -> date:
    """Return the most recent trading day (skip weekends).
    返回最近的交易日（跳过周末）。"""
    today = date.today()
    # If before 16:30 ET, last trading day is yesterday (or Friday)
    # Simplified: just check weekday
    d = today
    if d.weekday() == 0:  # Monday → use Friday
        d -= timedelta(days=3)
    elif d.weekday() == 6:  # Sunday → use Friday
        d -= timedelta(days=2)
    elif d.weekday() == 5:  # Saturday → use Friday
        d -= timedelta(days=1)
    else:
        # Weekday: if market hasn't closed yet, use yesterday
        from datetime import datetime
        try:
            import zoneinfo
            et = datetime.now(zoneinfo.ZoneInfo("US/Eastern"))
        except Exception:
            et = datetime.utcnow().replace(hour=datetime.utcnow().hour - 4)
        if et.hour < 17:  # before 5 PM ET, market data not final
            d -= timedelta(days=1)
            if d.weekday() >= 5:
                d -= timedelta(days=(d.weekday() - 4))
    return d


def check_staleness() -> dict:
    """Check how stale daily_metrics and external_data are.
    检查 daily_metrics 和 external_data 的过期状态。

    Returns dict with: stale (bool), last_date, target_date, gap_days.
    返回字典：stale（是否过期）、last_date、target_date、gap_days。
    """
    from utils.paths import OUTPUT_DIR

    target = _last_trading_day()
    result = {"target_date": target, "stale": False, "gap_days": 0}

    dm_path = OUTPUT_DIR / "daily_metrics.parquet"
    if dm_path.exists():
        dm = pd.read_parquet(dm_path)
        dm.index = pd.to_datetime(dm.index)
        last = dm.index[-1].date()
        result["last_date"] = last
        result["gap_days"] = (target - last).days
        result["stale"] = last < target
    else:
        result["last_date"] = None
        result["stale"] = True
        result["gap_days"] = 999

    return result


async def _fetch_and_merge(days: int = 5, ibkr_port: int = 7497):
    """Fetch latest data via IBKR and merge into parquets.
    通过 IBKR 获取最新数据并合并到 parquet 文件。"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from live.fetch_data import (
        IBKRSource, fetch_from_source, fetch_yields, get_events,
        validate, merge_with_historical,
    )

    source = IBKRSource(port=ibkr_port)
    try:
        await source.connect()
        result = await fetch_from_source(source, days)
        yields = fetch_yields(days)
        events = get_events(days)

        ok = validate(result["qqq"], result["premarket"], result["vix"], yields, events, source.name)

        if ok and not result["qqq"].empty:
            merge_with_historical(result["qqq"], result["premarket"], result["vix"], yields, source.name)
            print("[refresh] Merge complete")
        elif not ok:
            print("[refresh] Validation failed, skipping merge")
        else:
            print("[refresh] No new QQQ data fetched")

        return ok
    except Exception as e:
        print(f"[refresh] IBKR fetch failed: {e}")
        return False
    finally:
        await source.disconnect()


async def _refresh_external():
    """Refresh external data (VIX/VVIX/SKEW/yields) via yfinance.
    通过 yfinance 刷新外部数据（VIX/VVIX/SKEW/国债收益率）。"""
    from data.external_data import download_external_data
    try:
        download_external_data(force=True)
        print("[refresh] External data refreshed")
    except Exception as e:
        print(f"[refresh] External data refresh failed: {e}")


async def refresh_if_stale(force: bool = False, ibkr_port: int = 7497):
    """Check staleness and refresh if needed.
    检查过期状态，如果需要则刷新。

    Args:
        force: Force refresh even if data is fresh.
               即使数据新鲜也强制刷新。
        ibkr_port: IBKR TWS/Gateway port (7497=paper, 7496=live).
                   IBKR 端口（7497=模拟，7496=实盘）。

    Returns:
        dict with: refreshed (bool), status details.
        返回字典：refreshed（是否刷新了）、状态详情。
    """
    status = check_staleness()
    print(f"[refresh] Last data: {status['last_date']}, Target: {status['target_date']}, "
          f"Gap: {status['gap_days']} days, Stale: {status['stale']}")

    if not status["stale"] and not force:
        print("[refresh] Data is fresh, skipping")
        return {"refreshed": False, **status}

    days_to_fetch = min(max(status["gap_days"] + 2, 3), 30)
    print(f"[refresh] Fetching {days_to_fetch} days from IBKR (port {ibkr_port})...")

    ok = await _fetch_and_merge(days=days_to_fetch, ibkr_port=ibkr_port)

    # Also refresh external data
    await _refresh_external()

    return {"refreshed": ok, **status}


def refresh_sync(force: bool = False, ibkr_port: int = 7497) -> dict:
    """Synchronous wrapper for refresh_if_stale.
    refresh_if_stale 的同步包装。"""
    return asyncio.run(refresh_if_stale(force=force, ibkr_port=ibkr_port))
