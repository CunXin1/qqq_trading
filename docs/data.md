# Data Layer

## Raw Data

| File | Size | Description |
|------|------|-------------|
| `data/QQQ_1min_adjusted.parquet` | 75 MB | Split/dividend adjusted 1-min OHLCV |
| `data/QQQ_1min_unadjusted.parquet` | 70 MB | Raw unadjusted prices |

- Coverage: 2000-01-03 to 2026-02-20
- 4,124,413 candlesticks across 6,572 trading days
- Sessions: Regular (9:30-15:59), Pre-market (4:00-9:29), After-hours (16:01-19:59)

## Daily Metrics

`qqq_trading.data.daily_metrics` aggregates 1-minute data into 35 daily fields:

```python
from qqq_trading.data import load_1min_data, build_daily_metrics

raw = load_1min_data(Path("data/QQQ_1min_adjusted.parquet"))
daily = build_daily_metrics(raw)  # -> DataFrame (6572 x 35)
```

### Return Statistics

| Metric | Mean | Std | Kurtosis | Note |
|--------|------|-----|----------|------|
| Close-to-close | 0.04% | 1.69% | 7.87 | Fat tails >> 3.0 |
| Open-to-close | -0.001% | 1.44% | 12.14 | Even fatter tails |
| Intraday range | 1.87% | 1.50% | 32.83 | Extremely leptokurtic |
| Gap return | 0.04% | 0.79% | 14.63 | — |

### Large Move Frequencies

| Threshold | \|C2C\| | \|O2C\| | Range |
|-----------|---------|---------|-------|
| > 1% | 38.9% | 27.5% | 57.7% |
| > 2% | 16.1% | 11.5% | 30.9% |
| > 3% | 7.4% | 5.0% | 15.7% |
| > 5% | 2.3% | 1.5% | 5.8% |

## External Data

`qqq_trading.data.external_data` downloads and caches VIX/VVIX/Treasury yields from Yahoo Finance:

```python
from qqq_trading.data import download_external_data

ext = download_external_data()  # cached in output/external_data.parquet
```

| Ticker | Field Prefix | Available From |
|--------|-------------|----------------|
| ^VIX | `vix_` | 2000 |
| ^VVIX | `vvix_` | 2007 |
| ^TNX | `tnx_10y_` | 2000 |
| ^IRX | `irx_3m_` | 2000 |
| ^FVX | `fvx_5y_` | 2000 |

## Event Calendar

`qqq_trading.data.event_calendar` provides FOMC, NFP, and earnings season dates:

```python
from qqq_trading.data import load_fomc_dates, compute_nfp_dates

fomc = load_fomc_dates()          # 216 dates from data/fomc_dates.csv
nfp = compute_nfp_dates(2000, 2026)  # first Friday of each month
```

To add new FOMC dates, simply append rows to `data/fomc_dates.csv`.
