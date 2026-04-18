"""Data loading and preprocessing.
数据加载与预处理模块。

This package provides three core data pipelines for the QQQ trading system:
本包为 QQQ 交易系统提供三条核心数据管线：

1. daily_metrics  — Aggregate 1-min bars into daily OHLCV + derived return metrics.
                    将 1 分钟 K 线聚合为日频 OHLCV 及衍生收益指标。
2. external_data  — Download & cache VIX / VVIX / Treasury yield data from Yahoo Finance.
                    从 Yahoo Finance 下载并缓存 VIX / VVIX / 国债收益率数据。
3. event_calendar — Build event flags (FOMC, NFP, earnings season) used as features.
                    构建事件标记（FOMC、非农、财报季）供特征工程使用。
"""
from data.daily_metrics import load_1min_data, build_daily_metrics
from data.external_data import download_external_data
from data.event_calendar import (
    load_fomc_dates, compute_nfp_dates, compute_cpi_dates, compute_pce_dates,
    load_megacap_earnings, build_megacap_earnings_flags,
)
