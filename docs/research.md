# Research Scripts

The `research/` directory contains analysis scripts that import from the `qqq_trading` core library. They are meant for exploration, validation, and chart generation — not for production.

## Prerequisites

```bash
pip install -e ".[dev]"
```

All scripts can be run standalone:
```bash
python research/09_full_backtest.py
```

## Script Overview

| Script | Purpose | Depends On |
|--------|---------|------------|
| `01_daily_metrics.py` | Build daily metrics from 1-min data | Raw parquet files |
| `02_pattern_analysis.py` | Discover 8 statistical patterns, generate charts | `daily_metrics.parquet` |
| `04_report.py` | Generate HTML report with all charts | `daily_metrics.parquet`, charts |
| `05_window_analysis.py` | Test optimal training window length | `engineered_features.parquet` |
| `09_full_backtest.py` | Full backtest across 1%/2%/3%/5% thresholds | `interaction_features.parquet` |
| `10_options_backtest.py` | 0DTE vs 1DTE straddle-specific backtest | `interaction_features.parquet` |
| `12_robustness_test.py` | Monotonicity, walk-forward CV, feature decay | `interaction_features.parquet` |
| `13_occam_razor.py` | Top-K feature ablation (minimum viable model) | `interaction_features.parquet` |

## Execution Order

If starting from scratch (no cached outputs):

```bash
# Step 1: Generate daily metrics from raw 1-min data
python research/01_daily_metrics.py

# Step 2: Build features + train model (creates interaction_features.parquet)
python -m qqq_trading.cli.pipeline --preset production

# Step 3: Run any analysis script
python research/02_pattern_analysis.py    # patterns + charts
python research/09_full_backtest.py       # full threshold sweep
python research/10_options_backtest.py    # 0DTE/1DTE comparison
python research/12_robustness_test.py     # walk-forward + monotonicity
python research/13_occam_razor.py         # feature ablation
python research/04_report.py              # HTML report
```

## What Each Script Does

### 01_daily_metrics.py
Aggregates 1-minute OHLCV data into daily metrics. Thin wrapper around `qqq_trading.data.daily_metrics.build_daily_metrics()`. Prints summary statistics and top-20 largest move days.

### 02_pattern_analysis.py
Generates 8 analysis charts exploring patterns that precede large moves:
1. Yearly large move frequency
2. Calendar effects (day-of-week, month, OPEX)
3. Volatility clustering (ACF, conditional probability)
4. Pre-market signal strength (quintile analysis)
5. Gap analysis
6. Volume as predictor
7. Volatility regime transitions
8. Consecutive pattern analysis

### 04_report.py
Generates a self-contained HTML report (`output/report.html`) embedding all charts as base64 images, with summary statistics and model comparison tables.

### 05_window_analysis.py
Tests 4 approaches to selecting training data:
1. Fixed test, variable start year
2. Walk-forward with rolling windows (3/5/7/10/15yr)
3. Performance by volatility regime
4. Exponential decay weighting

Finding: 10-20 year window with light decay is optimal.

### 09_full_backtest.py
Comprehensive backtest across all move thresholds (1%, 2%, 3%, 5%) using XGBoost and LightGBM. Shows AUC, hit rate, and alert count at each confidence threshold.

### 10_options_backtest.py
Options-specific backtest comparing:
- **0DTE**: Buy at open, expire at close. Uses pre-market features.
- **1DTE**: Buy at close, expire tomorrow. Uses full-day features.

Tests both |O2C| (directional) and Range (non-directional) targets.

### 12_robustness_test.py
Three robustness checks:
1. **Monotonicity**: Hit rate should increase with confidence threshold
2. **Walk-forward CV**: 5-year rolling train, 5-day purge, test each year 2010-2025
3. **Feature decay**: Compare feature importance rankings across eras

### 13_occam_razor.py
Top-K feature ablation: trains models with K=1,3,5,7,10,20,50,all features. Finds that top-5 features capture 91% of full model AUC. Also validates via walk-forward.

## Output

All scripts save results to:
- `output/charts/*.png` — Analysis charts
- `output/report.html` — HTML report
- Console stdout — Summary statistics and tables
