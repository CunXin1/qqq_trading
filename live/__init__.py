"""Live trading automation module.
实盘交易自动化模块。

This package provides the end-to-end live trading pipeline:
本包提供端到端的实盘交易管线：

1. fetch_data.py — Fetch live market data (IBKR primary, yfinance fallback),
                   merge into historical parquets for feature engineering.
                   获取实时市场数据（IBKR 为主，yfinance 备用），
                   合并到历史 parquet 文件用于特征工程。

2. notify.py    — Pre-market signal notification via Bark push to iPhone.
                   Runs at 9:29 AM ET: fetch data → predict → push.
                   通过 Bark 推送盘前信号通知到 iPhone。
                   美东 9:29 运行：获取数据 → 预测 → 推送。

3. trader.py    — Auto-trade 0DTE QQQ straddle via IBKR.
                   Buys ATM straddle at 9:40 AM, monitors for 2x target,
                   sells half on hit.
                   通过 IBKR 自动交易 0DTE QQQ 跨式期权。
                   9:40 买入 ATM 跨式，监控 2 倍目标，达标卖出一半。
"""
from live.fetch_data import IBKRSource, YFinanceSource, fetch_yields, get_events
