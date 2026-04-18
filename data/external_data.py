"""Download and cache external market data (VIX, VVIX, Treasury yields).
下载并缓存外部市场数据（VIX、VVIX、国债收益率）。

This module fetches cross-asset data that serves as inputs to the external
feature engineering pipeline (features/external.py). The data covers:
本模块获取跨资产数据，供外部特征工程管线（features/external.py）使用。数据涵盖：

1. ^VIX   — CBOE Volatility Index (30-day implied vol of SPX options).
             CBOE 波动率指数（SPX 期权的 30 天隐含波动率）。
             Key use: VRP (Volatility Risk Premium) = IV - RV.
             核心用途：波动率风险溢价 VRP = 隐含波动率 - 已实现波动率。

2. ^VVIX  — Volatility of VIX (implied vol of VIX options).
             VIX 的波动率（VIX 期权的隐含波动率）。
             Elevated VVIX signals uncertainty about future volatility regime.
             VVIX 升高意味着市场对未来波动率状态存在不确定性。

3. ^TNX   — 10-Year Treasury Note Yield.
             10 年期国债收益率。
             Used for yield curve slope and rate-shock features.
             用于收益率曲线斜率和利率冲击特征。

4. ^IRX   — 3-Month Treasury Bill Yield.
             3 个月期国库券收益率。
             Short end of yield curve; combined with TNX for term spread.
             收益率曲线短端；与 TNX 组合计算期限利差。

5. ^FVX   — 5-Year Treasury Note Yield.
             5 年期国债收益率。
             Mid-curve reference; captures belly-of-curve dynamics.
             曲线中段参考；捕捉曲线中腹部的动态变化。

For each ticker, we store close/high/low to enable downstream computation
of intraday ranges and rolling statistics on these instruments.
每个标的存储 close/high/low，以便下游计算这些工具的日内振幅和滚动统计量。

Data is cached as parquet at output/external_data.parquet. Pass force=True
or use --refresh-external in the pipeline CLI to re-download.
数据以 parquet 格式缓存于 output/external_data.parquet。传入 force=True
或在 pipeline CLI 中使用 --refresh-external 可强制重新下载。
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path


def download_external_data(
    cache_path: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Download VIX, VVIX, Treasury yields from Yahoo Finance.
    从 Yahoo Finance 下载 VIX、VVIX、国债收益率数据。

    Downloads daily OHLC data for 5 tickers from 2000-01-01 to today,
    extracts close/high/low for each, and caches the result as parquet.
    下载 5 个标的从 2000-01-01 至今的日频 OHLC 数据，
    提取每个标的的 close/high/low，并将结果缓存为 parquet。

    Output columns (15 total) / 输出列（共 15 列）:
        vix_close, vix_high, vix_low,
        vvix_close, vvix_high, vvix_low,
        tnx_10y_close, tnx_10y_high, tnx_10y_low,
        irx_3m_close, irx_3m_high, irx_3m_low,
        fvx_5y_close, fvx_5y_high, fvx_5y_low

    Args:
        cache_path: Where to cache the parquet file. Default: output/external_data.parquet.
                    parquet 缓存路径。默认：output/external_data.parquet。
        force:      If True, re-download even if cache exists.
                    若为 True，即使缓存存在也强制重新下载。

    Returns:
        DataFrame indexed by date with 15 columns (5 tickers × 3 fields).
        以日期为索引的 DataFrame，包含 15 列（5 个标的 × 3 个字段）。
    """
    if cache_path is None:
        from utils.paths import OUTPUT_DIR
        cache_path = OUTPUT_DIR / "external_data.parquet"

    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    import yfinance as yf
    from datetime import date

    # Ticker → internal name mapping / 标的代码 → 内部命名映射
    tickers = {
        "^VIX": "vix",        # 30-day SPX implied volatility / 30天SPX隐含波动率
        "^VVIX": "vvix",      # Volatility of VIX / VIX的波动率
        "^TNX": "tnx_10y",    # 10-year Treasury yield / 10年期国债收益率
        "^IRX": "irx_3m",     # 3-month T-bill yield / 3个月国库券收益率
        "^FVX": "fvx_5y",     # 5-year Treasury yield / 5年期国债收益率
    }

    end_date = (date.today() + pd.Timedelta(days=1)).isoformat()

    all_data = {}
    for ticker, name in tickers.items():
        df = yf.download(ticker, start="2000-01-01", end=end_date, progress=False)
        # yfinance may return MultiIndex columns (ticker, field); flatten them
        # yfinance 可能返回多级索引列（标的, 字段）；将其展平
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
