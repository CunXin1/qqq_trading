"""
Fetch market data for QQQ prediction: IBKR first, yfinance fallback.

Usage:
    python -m live.fetch_data                      # auto: IBKR -> yfinance
    python -m live.fetch_data --days 10
    python -m live.fetch_data --source ibkr --port 7496
    python -m live.fetch_data --source yfinance
    python -m live.fetch_data --validate --csv
"""

import argparse
import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np


# ═════════════════════════════════════════════
# IBKR Data Source
# ═════════════════════════════════════════════

class IBKRSource:
    """Fetch data from Interactive Brokers TWS/Gateway."""

    def __init__(self, host="127.0.0.1", port=7497, client_id=10):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None
        self.name = "IBKR"
        self._qqq_1min_cache = None  # avoid fetching 1-min bars twice

    async def connect(self):
        from ib_async import IB
        self.ib = IB()
        await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
        print(f"  [IBKR] Connected (port {self.port}, server v{self.ib.client.serverVersion()})")

    async def disconnect(self):
        if self.ib:
            self.ib.disconnect()

    async def _qualify(self, contract):
        result = await self.ib.qualifyContractsAsync(contract)
        return result[0] if result else None

    async def _hist_daily(self, contract, days, what="TRADES"):
        """Fetch daily bars (used for VIX/VVIX)."""
        bars = await self.ib.reqHistoricalDataAsync(
            contract, endDateTime="", durationStr=f"{days + 10} D",
            barSizeSetting="1 day", whatToShow=what, useRTH=True, formatDate=1,
        )
        if not bars:
            return pd.DataFrame()
        data = [{"date": pd.Timestamp(b.date), "open": b.open, "high": b.high,
                 "low": b.low, "close": b.close, "volume": b.volume} for b in bars]
        return pd.DataFrame(data).set_index("date").tail(days)

    async def _fetch_1min_bars(self, contract, days):
        """Fetch 1-minute bars from IBKR. One request per day."""
        all_bars = []

        for d in range(days + 4):
            end_dt = date.today() - timedelta(days=d)
            # Skip weekends
            if end_dt.weekday() >= 5:
                continue
            # IBKR format: "yyyymmdd HH:mm:ss US/Eastern"
            end_str = end_dt.strftime("%Y%m%d") + " 23:59:59 US/Eastern"
            try:
                bars = await self.ib.reqHistoricalDataAsync(
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
            except Exception as e:
                print(f"    [IBKR] 1min bars for {end_dt}: {e}")

        if not all_bars:
            return pd.DataFrame()

        data = [{"datetime": pd.Timestamp(b.date), "open": b.open, "high": b.high,
                 "low": b.low, "close": b.close, "volume": b.volume} for b in all_bars]
        df = pd.DataFrame(data).set_index("datetime").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df

    def _aggregate_daily(self, df_1min):
        """Aggregate 1-minute bars into daily metrics (same logic as daily_metrics.py)."""
        regular = df_1min.between_time("09:30", "15:59")

        daily = regular.groupby(regular.index.date).agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        daily.index = pd.to_datetime(daily.index)
        daily.index.name = "date"

        # Max drawdown / runup from 1-min close prices
        max_dd_list = []
        max_ru_list = []
        for day, group in regular.groupby(regular.index.date):
            prices = group["close"].values
            if len(prices) < 2:
                max_dd_list.append(np.nan)
                max_ru_list.append(np.nan)
                continue
            cummax = np.maximum.accumulate(prices)
            max_dd_list.append(((prices - cummax) / cummax).min())
            cummin = np.minimum.accumulate(prices)
            max_ru_list.append(((prices - cummin) / cummin).max())

        daily["max_drawdown"] = max_dd_list
        daily["max_runup"] = max_ru_list

        # VWAP approximation (close * volume weighted)
        regular_copy = regular.copy()
        regular_copy["dollar_vol"] = regular_copy["close"] * regular_copy["volume"]
        vwap_df = regular_copy.groupby(regular_copy.index.date).agg(
            total_dollar=("dollar_vol", "sum"), total_vol=("volume", "sum"),
        )
        vwap_df["vwap"] = vwap_df["total_dollar"] / vwap_df["total_vol"]
        vwap_df.index = pd.to_datetime(vwap_df.index)
        daily["vwap"] = vwap_df["vwap"]

        return daily

    def _aggregate_premarket(self, df_1min):
        """Aggregate 1-minute bars into premarket daily metrics."""
        pre = df_1min.between_time("04:00", "09:29")
        if pre.empty:
            return pd.DataFrame()

        results = []
        for day, group in pre.groupby(pre.index.date):
            if len(group) < 2:
                continue
            results.append({
                "date": pd.Timestamp(day),
                "premarket_open": group["open"].iloc[0],
                "premarket_high": group["high"].max(),
                "premarket_low": group["low"].min(),
                "premarket_close": group["close"].iloc[-1],
                "premarket_volume": group["volume"].sum(),
                "premarket_range": (group["high"].max() - group["low"].min()) / group["open"].iloc[0],
                "premarket_ret": group["close"].iloc[-1] / group["open"].iloc[0] - 1,
                "premarket_bars": len(group),
            })

        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results).set_index("date")

    async def _get_qqq_1min(self, days):
        """Get QQQ 1-min bars, with caching to avoid duplicate requests."""
        if self._qqq_1min_cache is not None:
            return self._qqq_1min_cache

        from ib_async import Stock
        contract = await self._qualify(Stock("QQQ", "SMART", "USD"))
        if not contract:
            raise RuntimeError("Cannot qualify QQQ contract")

        print(f"    Fetching 1-min bars ({days} days)...")
        df_1min = await self._fetch_1min_bars(contract, days)
        if not df_1min.empty:
            n_bars = len(df_1min)
            n_days_raw = df_1min.index.normalize().nunique()
            print(f"    Got {n_bars:,} bars across {n_days_raw} days")

        self._qqq_1min_cache = df_1min
        return df_1min

    async def fetch_qqq(self, days):
        """Fetch QQQ 1-min bars and aggregate to daily."""
        df_1min = await self._get_qqq_1min(days)
        if df_1min.empty:
            return pd.DataFrame()
        daily = self._aggregate_daily(df_1min)
        return daily.tail(days)

    async def fetch_qqq_premarket(self, days):
        """Extract premarket metrics from cached 1-min bars."""
        df_1min = await self._get_qqq_1min(days)
        if df_1min.empty:
            return pd.DataFrame()
        premarket = self._aggregate_premarket(df_1min)
        return premarket.tail(days)

    async def fetch_vix(self, days):
        from ib_async import Index
        records = {}
        for symbol, prefix in [("VIX", "vix"), ("VVIX", "vvix")]:
            try:
                contract = await self._qualify(Index(symbol, "CBOE", "USD"))
                if not contract:
                    continue
                df = await self._hist_daily(contract, days, what="TRADES")
                if not df.empty:
                    records[f"{prefix}_close"] = df["close"]
                    records[f"{prefix}_high"] = df["high"]
                    records[f"{prefix}_low"] = df["low"]
            except Exception as e:
                print(f"    [IBKR] {symbol} failed: {e}")
        return pd.DataFrame(records) if records else pd.DataFrame()


class YFinanceSource:
    """Fetch data from Yahoo Finance (free, no connection needed)."""

    def __init__(self):
        self.name = "yfinance"

    async def connect(self):
        import yfinance  # just check import
        print(f"  [yfinance] Ready")

    async def disconnect(self):
        pass

    def _download(self, ticker, days):
        import yfinance as yf
        start = (date.today() - timedelta(days=days + 10)).isoformat()
        end = (date.today() + timedelta(days=1)).isoformat()
        df = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        return df.tail(days)

    async def fetch_qqq(self, days):
        return self._download("QQQ", days)

    async def fetch_qqq_premarket(self, days):
        import yfinance as yf
        start = (date.today() - timedelta(days=days + 10)).isoformat()
        end = (date.today() + timedelta(days=1)).isoformat()
        df = yf.download("QQQ", start=start, end=end, interval="1h",
                         prepost=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)

        results = []
        for day, group in df.groupby(df.index.date):
            pre = group.between_time("04:00", "09:29")
            if pre.empty:
                continue
            results.append({
                "date": pd.Timestamp(day),
                "premarket_range": (pre["high"].max() - pre["low"].min()) / pre["open"].iloc[0],
                "premarket_ret": pre["close"].iloc[-1] / pre["open"].iloc[0] - 1,
            })
        return pd.DataFrame(results).set_index("date").tail(days) if results else pd.DataFrame()

    async def fetch_vix(self, days):
        records = {}
        for ticker, prefix in [("^VIX", "vix"), ("^VVIX", "vvix")]:
            df = self._download(ticker, days)
            if not df.empty:
                records[f"{prefix}_close"] = df["close"]
                records[f"{prefix}_high"] = df["high"]
                records[f"{prefix}_low"] = df["low"]
        return pd.DataFrame(records) if records else pd.DataFrame()


# ═════════════════════════════════════════════
# Shared: Yields + Events (source-independent)
# ═════════════════════════════════════════════

def fetch_yields(days):
    """Treasury yields — always via yfinance (simplest for yields)."""
    import yfinance as yf
    start = (date.today() - timedelta(days=days + 10)).isoformat()
    end = (date.today() + timedelta(days=1)).isoformat()

    records = {}
    for ticker, name in [("^TNX", "yield_10y"), ("^IRX", "yield_3m"), ("^FVX", "yield_5y")]:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        if not df.empty:
            records[name] = df["close"]

    out = pd.DataFrame(records)
    if out.empty:
        return out
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"

    if "yield_10y" in out.columns and "yield_3m" in out.columns:
        out["yield_curve_slope"] = out["yield_10y"] - out["yield_3m"]
        out["yield_curve_inverted"] = (out["yield_curve_slope"] < 0).astype(int)
    if "yield_10y" in out.columns:
        out["yield_10y_change_1d"] = out["yield_10y"].diff()

    return out.tail(days)


def get_events(days=30):
    """FOMC, NFP, earnings season flags."""
    today = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(today - timedelta(days=5), today + timedelta(days=days))

    fomc_dates = pd.DatetimeIndex(pd.to_datetime([
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
        "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
        "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
    ]))

    nfp_dates = []
    for year in range(today.year - 1, today.year + 2):
        for month in range(1, 13):
            d1 = pd.Timestamp(year, month, 1)
            nfp_dates.append(d1 + timedelta(days=(4 - d1.dayofweek) % 7))
    nfp_dates = pd.DatetimeIndex(nfp_dates)

    rows = []
    for d in dates:
        future_fomc = fomc_dates[fomc_dates >= d]
        days_to_fomc = (future_fomc[0] - d).days if len(future_fomc) > 0 else 60
        future_nfp = nfp_dates[nfp_dates >= d]
        days_to_nfp = (future_nfp[0] - d).days if len(future_nfp) > 0 else 30

        m, day_num = d.month, d.day
        is_earn = int(
            (m in (1, 4, 7, 10) and day_num >= 20) or
            (m in (2, 5, 8, 11) and day_num <= 14)
        )
        rows.append({
            "date": d, "is_fomc_day": int(d in fomc_dates), "days_to_fomc": days_to_fomc,
            "fomc_imminent": int(days_to_fomc <= 1), "is_nfp_day": int(d in nfp_dates),
            "days_to_nfp": days_to_nfp, "nfp_imminent": int(days_to_nfp <= 1),
            "is_earnings_season": is_earn,
        })
    return pd.DataFrame(rows).set_index("date")


# ═════════════════════════════════════════════
# Derived fields (applied to any source)
# ═════════════════════════════════════════════

def add_qqq_derived(df):
    """Add return/range fields to QQQ OHLCV."""
    df["close_to_close_ret"] = df["close"].pct_change()
    df["open_to_close_ret"] = df["close"] / df["open"] - 1
    df["intraday_range"] = (df["high"] - df["low"]) / df["open"]
    df["gap_return"] = df["open"] / df["close"].shift(1) - 1
    return df


def add_vix_derived(df):
    """Add VIX change/range/ratio fields."""
    if "vix_close" in df.columns:
        df["vix_change_1d"] = df["vix_close"].pct_change()
        df["vix_range"] = (df["vix_high"] - df["vix_low"]) / df["vix_close"]
    if "vvix_close" in df.columns and "vix_close" in df.columns:
        df["vvix_vix_ratio"] = df["vvix_close"] / df["vix_close"]
    return df


# ═════════════════════════════════════════════
# Validation
# ═════════════════════════════════════════════

def validate(qqq, premarket, vix, yields, events, source_name):
    """Check data completeness."""
    print(f"\n{'=' * 70}")
    print(f"DATA VALIDATION (source: {source_name})")
    print(f"{'=' * 70}")
    issues = []

    checks = [
        ("QQQ", qqq, ["close", "intraday_range"]),
        ("VIX", vix, ["vix_close"]),
        ("Yields", yields, ["yield_10y"]),
        ("Premarket", premarket, ["premarket_range"]),
    ]

    for name, df, cols in checks:
        print(f"\n{name}: ", end="")
        if df.empty:
            print("EMPTY")
            issues.append(f"{name} is empty")
            continue
        print(f"{len(df)} days, latest {df.index[-1].date()}")
        for col in cols:
            if col in df.columns:
                v = df[col].iloc[-1]
                fmt = f"{v:.2%}" if "range" in col or "ret" in col else f"{v:.2f}"
                print(f"  {col}: {fmt}")
            else:
                issues.append(f"{name} missing {col}")

    # Date alignment
    latest_dates = {n: df.index[-1].date() for n, df, _ in checks if not df.empty}
    if len(set(latest_dates.values())) > 1:
        issues.append(f"Date mismatch: {latest_dates}")
        print(f"\nDate mismatch: {latest_dates}")

    print(f"\n{'=' * 70}")
    print(f"{'ALL CHECKS PASSED' if not issues else f'{len(issues)} ISSUES: ' + '; '.join(issues)}")
    print(f"{'=' * 70}")
    return len(issues) == 0


# ═════════════════════════════════════════════
# Main orchestrator
# ═════════════════════════════════════════════

async def fetch_from_source(source, days):
    """Fetch all data from one source. Returns dict of DataFrames."""
    qqq = await source.fetch_qqq(days)
    if not qqq.empty:
        qqq = add_qqq_derived(qqq)

    premarket = await source.fetch_qqq_premarket(days)

    vix = await source.fetch_vix(days)
    if not vix.empty:
        vix.index.name = "date"
        vix = add_vix_derived(vix)

    return {"qqq": qqq, "premarket": premarket, "vix": vix}


async def run(args):
    out_dir = Path(__file__).resolve().parents[1] / args.output
    out_dir.mkdir(exist_ok=True)

    source_used = None
    result = None

    # ── Try IBKR first ──
    if args.source in ("auto", "ibkr"):
        print(f"\n[1] Trying IBKR (port {args.port})...")
        ibkr = IBKRSource(args.host, args.port, args.client_id)
        try:
            await ibkr.connect()
            result = await fetch_from_source(ibkr, args.days)
            await ibkr.disconnect()

            # Check if we got meaningful data
            if result["qqq"].empty:
                print("  [IBKR] QQQ data empty, falling back...")
                result = None
            else:
                source_used = "IBKR"
                # Save raw 1-min bars if available
                if ibkr._qqq_1min_cache is not None and not ibkr._qqq_1min_cache.empty:
                    result["qqq_1min"] = ibkr._qqq_1min_cache
                print(f"  [IBKR] Success: QQQ={len(result['qqq'])}d, "
                      f"VIX={len(result['vix'])}d, Premarket={len(result['premarket'])}d"
                      + (f", 1min={len(result.get('qqq_1min', []))} bars" if 'qqq_1min' in result else ""))
        except Exception as e:
            print(f"  [IBKR] Failed: {e}")
            try:
                await ibkr.disconnect()
            except Exception:
                pass

    # ── Fallback to yfinance ──
    if result is None and args.source in ("auto", "yfinance"):
        print(f"\n{'[2] ' if args.source == 'auto' else '[1] '}Trying yfinance...")
        yf_src = YFinanceSource()
        try:
            await yf_src.connect()
            result = await fetch_from_source(yf_src, args.days)
            source_used = "yfinance"
            print(f"  [yfinance] Success: QQQ={len(result['qqq'])}d, "
                  f"VIX={len(result['vix'])}d, Premarket={len(result['premarket'])}d")
        except Exception as e:
            print(f"  [yfinance] Failed: {e}")

    if result is None:
        print("\nERROR: All data sources failed.")
        sys.exit(1)

    # ── Yields + Events (always same source) ──
    print(f"\nFetching yields (yfinance) + events (calendar)...")
    yields = fetch_yields(args.days)
    events = get_events()
    print(f"  Yields: {len(yields)}d, Events: {len(events)}d")

    qqq = result["qqq"]
    premarket = result["premarket"]
    vix = result["vix"]

    # ── Preview ──
    print(f"\n{'=' * 70}")
    print(f"DATA SUMMARY (source: {source_used})")
    print(f"{'=' * 70}")

    if not qqq.empty:
        last = qqq.iloc[-1]
        print(f"\nQQQ {qqq.index[-1].date()}:")
        print(f"  O={last['open']:.2f}  H={last['high']:.2f}  L={last['low']:.2f}  C={last['close']:.2f}")
        print(f"  Range={last['intraday_range']:.2%}  C2C={last['close_to_close_ret']:.2%}  "
              f"Gap={last['gap_return']:.2%}  Vol={last['volume']:,.0f}")

    if not premarket.empty:
        lp = premarket.iloc[-1]
        print(f"\nPremarket {premarket.index[-1].date()}:")
        print(f"  Range={lp['premarket_range']:.2%}  Ret={lp['premarket_ret']:.2%}")

    if not vix.empty:
        lv = vix.iloc[-1]
        parts = []
        if "vix_close" in lv.index: parts.append(f"VIX={lv['vix_close']:.2f}")
        if "vvix_close" in lv.index and not pd.isna(lv["vvix_close"]): parts.append(f"VVIX={lv['vvix_close']:.2f}")
        if "vix_change_1d" in lv.index and not pd.isna(lv["vix_change_1d"]): parts.append(f"chg={lv['vix_change_1d']:+.1%}")
        print(f"\nVIX {vix.index[-1].date()}: {'  '.join(parts)}")

    if not yields.empty:
        ly = yields.iloc[-1]
        parts = []
        for c in ["yield_10y", "yield_3m"]:
            if c in ly.index: parts.append(f"{c.split('_')[1].upper()}={ly[c]:.2f}%")
        if "yield_curve_slope" in ly.index:
            s = ly["yield_curve_slope"]
            parts.append(f"Slope={s:+.2f}% ({'INV' if s < 0 else 'NRM'})")
        print(f"\nYields {yields.index[-1].date()}: {'  '.join(parts)}")

    today = pd.Timestamp.today().normalize()
    if today in events.index:
        e = events.loc[today]
        flags = []
        if e["is_fomc_day"]: flags.append("FOMC TODAY")
        elif e["fomc_imminent"]: flags.append(f"FOMC in {e['days_to_fomc']}d")
        else: flags.append(f"FOMC in {e['days_to_fomc']}d")
        if e["is_nfp_day"]: flags.append("NFP TODAY")
        elif e["nfp_imminent"]: flags.append(f"NFP in {e['days_to_nfp']}d")
        else: flags.append(f"NFP in {e['days_to_nfp']}d")
        if e["is_earnings_season"]: flags.append("EARNINGS SEASON")
        print(f"\nEvents: {' | '.join(flags)}")

    # ── Validate ──
    if args.validate:
        validate(qqq, premarket, vix, yields, events, source_used)

    # ── Save CSV ──
    if args.csv:
        qqq.to_csv(out_dir / "live_qqq.csv", float_format="%.4f")
        premarket.to_csv(out_dir / "live_premarket.csv", float_format="%.4f")
        vix.to_csv(out_dir / "live_vix.csv", float_format="%.4f")
        yields.to_csv(out_dir / "live_yields.csv", float_format="%.4f")
        events.to_csv(out_dir / "live_events.csv")
        if "qqq_1min" in result:
            result["qqq_1min"].to_csv(out_dir / "live_qqq_1min.csv", float_format="%.4f")
            print(f"\nSaved 6 CSV files to {out_dir}/ (including 1-min bars)")
        else:
            print(f"\nSaved 5 CSV files to {out_dir}/")

    # ── Merge with historical parquets ──
    if args.merge:
        merge_with_historical(qqq, premarket, vix, yields, source_used)

    print(f"\n[Source: {source_used}]")


# ═════════════════════════════════════════════
# Merge live data into historical parquets
# ═════════════════════════════════════════════

def _live_qqq_to_daily_metrics(qqq, premarket):
    """Convert live QQQ + premarket DataFrames to daily_metrics format (35 columns)."""
    df = pd.DataFrame(index=qqq.index)
    df.index.name = "date"

    # OHLCV — rename to match historical
    df["reg_open"] = qqq["open"]
    df["reg_high"] = qqq["high"]
    df["reg_low"] = qqq["low"]
    df["reg_close"] = qqq["close"]
    df["volume_regular"] = qqq["volume"]

    # Premarket
    if not premarket.empty:
        pre_aligned = premarket.reindex(df.index)
        df["premarket_open"] = pre_aligned.get("premarket_open")
        df["premarket_high"] = pre_aligned.get("premarket_high")
        df["premarket_low"] = pre_aligned.get("premarket_low")
        df["premarket_close"] = pre_aligned.get("premarket_close")
        df["volume_premarket"] = pre_aligned.get("premarket_volume")
        df["premarket_ret"] = pre_aligned.get("premarket_ret")
        df["premarket_range"] = pre_aligned.get("premarket_range")
    else:
        for col in ["premarket_open", "premarket_high", "premarket_low",
                     "premarket_close", "volume_premarket", "premarket_ret", "premarket_range"]:
            df[col] = np.nan

    # Full day high/low (use premarket + regular)
    if not premarket.empty:
        pre_aligned = premarket.reindex(df.index)
        df["full_high"] = pd.concat([qqq[["high"]], pre_aligned[["premarket_high"]].rename(
            columns={"premarket_high": "high"})], axis=1).max(axis=1)
        df["full_low"] = pd.concat([qqq[["low"]], pre_aligned[["premarket_low"]].rename(
            columns={"premarket_low": "low"})], axis=1).min(axis=1)
    else:
        df["full_high"] = qqq["high"]
        df["full_low"] = qqq["low"]

    # VWAP, max_drawdown, max_runup
    df["vwap"] = qqq.get("vwap", np.nan)
    df["max_drawdown"] = qqq.get("max_drawdown", np.nan)
    df["max_runup"] = qqq.get("max_runup", np.nan)

    # Derived returns
    df["close_to_close_ret"] = qqq.get("close_to_close_ret", df["reg_close"].pct_change())
    df["open_to_close_ret"] = qqq.get("open_to_close_ret", df["reg_close"] / df["reg_open"] - 1)
    df["intraday_range"] = qqq.get("intraday_range",
                                    (df["reg_high"] - df["reg_low"]) / df["reg_open"])
    df["gap_return"] = qqq.get("gap_return",
                                df["reg_open"] / df["reg_close"].shift(1) - 1)

    # Absolute values
    df["abs_close_to_close"] = df["close_to_close_ret"].abs()
    df["abs_open_to_close"] = df["open_to_close_ret"].abs()

    # Large move flags
    for metric in ["abs_close_to_close", "abs_open_to_close", "intraday_range"]:
        for thresh in [0.01, 0.02, 0.03, 0.05]:
            df[f"{metric}_gt_{int(thresh * 100)}pct"] = df[metric] > thresh

    return df


def merge_with_historical(qqq, premarket, vix, yields, source_name):
    """Merge live data into historical parquets, update in place."""
    from utils.paths import OUTPUT_DIR

    print(f"\n{'=' * 70}")
    print("MERGING LIVE DATA INTO HISTORICAL PARQUETS")
    print(f"{'=' * 70}")

    # ── 1. daily_metrics.parquet ──
    hist_path = OUTPUT_DIR / "daily_metrics.parquet"
    hist = pd.read_parquet(hist_path)
    hist.index = pd.to_datetime(hist.index)

    live_daily = _live_qqq_to_daily_metrics(qqq, premarket)

    # Fix: first row of live needs gap_return from historical last close
    if len(live_daily) > 0 and len(hist) > 0:
        first_live_date = live_daily.index[0]
        if first_live_date not in hist.index:
            # Compute gap_return using historical last close
            last_hist_close = hist["reg_close"].iloc[-1]
            first_live_open = live_daily["reg_open"].iloc[0]
            live_daily.loc[first_live_date, "gap_return"] = first_live_open / last_hist_close - 1
        # Also fix close_to_close_ret for first new day
        if first_live_date not in hist.index:
            last_hist_close = hist["reg_close"].iloc[-1]
            first_live_close = live_daily["reg_close"].iloc[0]
            live_daily.loc[first_live_date, "close_to_close_ret"] = first_live_close / last_hist_close - 1
            live_daily.loc[first_live_date, "abs_close_to_close"] = abs(live_daily.loc[first_live_date, "close_to_close_ret"])
            # Recompute large move flags for first day
            for metric in ["abs_close_to_close", "abs_open_to_close", "intraday_range"]:
                for thresh in [0.01, 0.02, 0.03, 0.05]:
                    live_daily.loc[first_live_date, f"{metric}_gt_{int(thresh * 100)}pct"] = (
                        live_daily.loc[first_live_date, metric] > thresh
                    )

    # Remove overlap days from live (keep historical for overlap)
    new_days = live_daily[~live_daily.index.isin(hist.index)]
    if len(new_days) == 0:
        print("  daily_metrics: no new days to add")
    else:
        merged = pd.concat([hist, new_days]).sort_index()
        merged.to_parquet(hist_path)
        print(f"  daily_metrics: +{len(new_days)} days "
              f"({new_days.index[0].date()} to {new_days.index[-1].date()}), "
              f"total {len(merged)} days")

    # ── 2. external_data.parquet ──
    ext_path = OUTPUT_DIR / "external_data.parquet"
    if ext_path.exists():
        ext_hist = pd.read_parquet(ext_path)
        ext_hist.index = pd.to_datetime(ext_hist.index)
    else:
        ext_hist = pd.DataFrame()

    # Build external data from live vix + yields
    ext_live = pd.DataFrame(index=vix.index if not vix.empty else yields.index)
    ext_live.index.name = "date"

    if not vix.empty:
        for col in ["vix_close", "vix_high", "vix_low", "vvix_close", "vvix_high", "vvix_low"]:
            if col in vix.columns:
                ext_live[col] = vix[col]

    if not yields.empty:
        yields_aligned = yields.reindex(ext_live.index) if not ext_live.empty else yields
        for src, dst in [("yield_10y", "tnx_10y_close"), ("yield_3m", "irx_3m_close"),
                         ("yield_5y", "fvx_5y_close")]:
            if src in yields.columns:
                ext_live[dst] = yields_aligned.get(src, yields.reindex(ext_live.index).get(src))

    if not ext_live.empty and not ext_hist.empty:
        new_ext = ext_live[~ext_live.index.isin(ext_hist.index)].copy()
        if len(new_ext) > 0:
            # Align columns
            for col in ext_hist.columns:
                if col not in new_ext.columns:
                    new_ext[col] = np.nan
            new_ext = new_ext[ext_hist.columns]
            merged_ext = pd.concat([ext_hist, new_ext]).sort_index()
            merged_ext.to_parquet(ext_path)
            print(f"  external_data: +{len(new_ext)} days, total {len(merged_ext)} days")
        else:
            print("  external_data: no new days to add")
    else:
        print("  external_data: skipped (no data)")

    print(f"\n  Source: {source_name}")
    print("  Parquets updated in output/")


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch data: IBKR first, yfinance fallback")
    parser.add_argument("--source", type=str, default="auto",
                        choices=["auto", "ibkr", "yfinance"], help="Data source (default: auto)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7496, help="IBKR port (7496=live, 7497=paper)")
    parser.add_argument("--client-id", type=int, default=10)
    parser.add_argument("--days", type=int, default=5, help="Days of history")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--csv", action="store_true", help="Save to CSV files")
    parser.add_argument("--merge", action="store_true",
                        help="Merge live data into historical parquets (daily_metrics + external_data)")
    parser.add_argument("--output", type=str, default="output/live",
                        help="Output directory relative to project root")
    return parser.parse_args()


def main():
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
