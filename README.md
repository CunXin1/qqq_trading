# QQQ Daily Large Move Analysis & Prediction

## Project Overview

Based on QQQ 1-minute OHLCV data (2000-2026, ~4.1M bars), this project:
1. Identifies days with large intraday moves (>1%, 2%, 3%, 5%)
2. Discovers statistical patterns that precede large moves
3. Builds ML models to predict large moves for 0DTE and 1DTE options straddles
4. Validates robustness via walk-forward CV and monotonicity tests

## Project Structure

```
qqq_trading/
├── pyproject.toml                    # Package config & dependencies
├── config/
│   └── default.yaml                  # Centralized parameters
├── data/
│   ├── fomc_dates.csv                # FOMC announcement dates (2000-2026)
│   ├── QQQ_1min_adjusted.parquet     # Split/dividend adjusted, 4.1M bars
│   └── QQQ_1min_unadjusted.parquet   # Raw prices
├── qqq_trading/                      # Core library
│   ├── config.py                     # Config dataclass + YAML loader
│   ├── data/
│   │   ├── daily_metrics.py          # 1-min → daily OHLCV aggregation
│   │   ├── external_data.py          # VIX/VVIX/rates via yfinance
│   │   └── event_calendar.py         # FOMC/NFP/earnings calendar
│   ├── features/
│   │   ├── base.py                   # 53 base features (price/volume)
│   │   ├── external.py               # VRP, VIX dynamics, rates, events
│   │   ├── interactions.py           # VRP × catalyst cross features
│   │   ├── path.py                   # Smoothness / path dependency
│   │   └── registry.py              # Canonical feature lists
│   ├── models/
│   │   ├── training.py               # Unified model creation & training
│   │   ├── evaluation.py             # Metrics, backtest, thresholds
│   │   └── prediction.py             # Load model & inference
│   ├── utils/
│   │   ├── paths.py                  # Project path constants
│   │   ├── plotting.py               # Matplotlib setup & helpers
│   │   └── splits.py                 # Time-series splitting
│   └── cli/
│       ├── predict.py                # Daily prediction CLI
│       └── pipeline.py               # Full retrain pipeline
├── research/                         # Research & analysis scripts
│   ├── 01_daily_metrics.py
│   ├── 02_pattern_analysis.py
│   ├── 04_report.py
│   ├── 05_window_analysis.py
│   ├── 09_full_backtest.py
│   ├── 10_options_backtest.py
│   ├── 12_robustness_test.py
│   └── 13_occam_razor.py
├── tests/                            # pytest test suite (53 tests)
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_daily_metrics.py
│   ├── test_features_*.py
│   ├── test_training.py
│   ├── test_evaluation.py
│   └── ...
└── output/                           # Generated outputs
    ├── model/                        # Saved models (joblib)
    ├── charts/                       # Analysis charts (PNG)
    └── *.parquet                     # Intermediate feature datasets
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run prediction (uses saved models)
python -m qqq_trading.cli.predict --mode both --threshold 0.5
python -m qqq_trading.cli.predict --format json

# Retrain model
python -m qqq_trading.cli.pipeline --preset production --train-end 2022-12-31

# Run tests
pytest tests/ -v

# Run research scripts
python research/09_full_backtest.py
python research/12_robustness_test.py
```

## Data
- `data/QQQ_1min_adjusted.parquet` — Split/dividend adjusted, 4,124,413 bars
- `data/QQQ_1min_unadjusted.parquet` — Raw prices
- Coverage: 2000-01-03 to 2026-02-20, 6,572 trading days
- Extended hours: pre-market (4:00-9:29) + post-market (16:01-19:59)

## Key Results (Test Period: 2023.1 - 2026.2)

### Best Models

| Scenario | Target | AUC | @0.7 Hit Rate | Use Case |
|----------|--------|-----|---------------|----------|
| **0DTE Range>2%** | Intraday range >2% | **0.826** | **66%** (59 alerts) | Buy straddle at open |
| **0DTE Range>3%** | Intraday range >3% | **0.864** | **57%** (14 alerts) | High-conviction straddle |
| 1DTE \|C2C\|>2% | Tomorrow return >2% | 0.743 | 62% (13 alerts) | Overnight straddle |
| 1DTE \|C2C\|>1% | Tomorrow return >1% | 0.644 | 69% (51 alerts) | Daily vol forecast |

### Key Pattern Discoveries
- **Volatility clustering**: P(>2% | yesterday >2%) = 32.4% vs 16.1% unconditional (2x)
- **Pre-market signal**: Top-20% pre-market range → 32.5% large move vs 3.8% bottom-20%
- **Gamma trap**: Complacent VRP + FOMC = 42.9% large move probability (2.7x)
- **High vol + catalyst**: High vol regime + FOMC = 47.6% (3.0x)

### Robustness
- Monotonicity: **ALL 6 targets PASS** (hit rate increases smoothly with confidence)
- Walk-forward: 1DTE models **CONSISTENT** (static vs WF gap < 5%)
- Walk-forward: 0DTE Range models show slight overfit warning (+5% gap)
- Feature stability: **UNSTABLE** across eras (Spearman rho ~0.4) — model needs periodic retraining

## Output Files

```
output/
  daily_metrics.parquet         # Daily OHLCV + returns + flags (6572 x 35)
  engineered_features.parquet   # Base ML features (~90 cols)
  interaction_features.parquet  # + cross features (production)
  path_features.parquet         # + smoothness features
  external_data.parquet         # VIX/VVIX/Treasury yields (cached)
  report.html                   # Full HTML report
  charts/                       # Analysis charts (PNG)
  model/
    interaction_model.joblib    # Best model (XGBoost, 121 features)
    next_day_2pct_model.joblib  # Baseline model (53 features)
    scaler.joblib               # StandardScaler
    *.csv                       # Feature column lists
```

## Documentation

| Document | Content |
|----------|---------|
| [docs/data.md](docs/data.md) | Data layer: raw data specs, daily metrics, external data, event calendar |
| [docs/features.md](docs/features.md) | Feature engineering: 4 feature groups, VRP rationale, registry usage |
| [docs/models.md](docs/models.md) | Models: training, evaluation, performance tables, robustness analysis |
| [docs/trading_guide.md](docs/trading_guide.md) | Trading: signal architecture, daily decision flow, risk management |
| [docs/research.md](docs/research.md) | Research scripts: what each does, execution order, outputs |

## Limitations
- No individual stock earnings dates (only earnings season proxy)
- No CPI/GDP/PMI event calendar
- Feature importance shifts significantly across market eras — periodic retraining required
- 0DTE models have slight overfitting risk vs walk-forward baseline
- >5% move prediction has insufficient test samples (3 events in 786 days)
