"""Download and cache external market data (VIX, VVIX, Treasury yields)."""
from __future__ import annotations

import pandas as pd
from pathlib import Path


def download_external_data(
    cache_path: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Download VIX, VVIX, Treasury yields from Yahoo Finance.

    Args:
        cache_path: Where to cache the parquet file. Default: output/external_data.parquet
        force: If True, re-download even if cache exists.
    """
    if cache_path is None:
        from qqq_trading.utils.paths import OUTPUT_DIR
        cache_path = OUTPUT_DIR / "external_data.parquet"

    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    import yfinance as yf
    from datetime import date

    tickers = {
        "^VIX": "vix",
        "^VVIX": "vvix",
        "^TNX": "tnx_10y",
        "^IRX": "irx_3m",
        "^FVX": "fvx_5y",
    }

    end_date = (date.today() + pd.Timedelta(days=1)).isoformat()

    all_data = {}
    for ticker, name in tickers.items():
        df = yf.download(ticker, start="2000-01-01", end=end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        all_data[f"{name}_close"] = df["close"]
        all_data[f"{name}_high"] = df["high"]
        all_data[f"{name}_low"] = df["low"]

    ext = pd.DataFrame(all_data)
    ext.index = pd.to_datetime(ext.index)
    ext.index.name = "date"

    ext.to_parquet(cache_path)
    return ext
