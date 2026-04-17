# QQQ Intraday Volatility Prediction & 0DTE Straddle Trading System

A quantitative trading system that predicts large intraday moves in QQQ (Nasdaq-100 ETF) and executes 0DTE/1DTE options straddle strategies. Built on 26 years of 1-minute OHLCV data (~4.1M bars), the system combines feature engineering, gradient-boosted models, and live IBKR integration for end-to-end signal generation and automated execution.

## Highlights

- **122 engineered features** across 4 groups: price/volume dynamics, external macro (VIX/VVIX/yields), event catalysts (FOMC/NFP/earnings), and cross-interaction terms (VRP × catalyst)
- **Walk-forward validated** with monotonicity checks — hit rate increases smoothly with model confidence across all 6 prediction targets
- **Live trading pipeline**: IBKR data fetch → feature computation → signal generation → 0DTE straddle execution → Bark push notifications
- **Web dashboard** (FastAPI) for real-time signal monitoring, trade history, and data status

## Key Results (Test Period: 2023.01 – 2026.02, 786 trading days)

| Scenario | Target | AUC | @0.7 Hit Rate | Use Case |
|----------|--------|-----|---------------|----------|
| **0DTE Range>2%** | Intraday range >2% | **0.826** | **66%** (59 alerts) | Buy straddle at open |
| **0DTE Range>3%** | Intraday range >3% | **0.864** | **57%** (14 alerts) | High-conviction straddle |
| 1DTE \|C2C\|>2% | Next-day abs return >2% | 0.743 | 62% (13 alerts) | Overnight straddle |
| 1DTE \|C2C\|>1% | Next-day abs return >1% | 0.644 | 69% (51 alerts) | Daily vol forecast |

### Pattern Discoveries
- **Volatility clustering**: P(>2% move | yesterday >2%) = 32.4% vs 16.1% unconditional (2× lift)
- **Pre-market signal**: Top-20% pre-market range → 32.5% large move rate vs 3.8% bottom-20% (8.6×)
- **Gamma trap**: Complacent VRP + FOMC week = 42.9% large move probability (2.7×)
- **High vol + catalyst**: Elevated vol regime + FOMC = 47.6% (3.0×)

### Robustness
- **Monotonicity**: All 6 targets pass — model confidence is well-calibrated
- **Walk-forward CV**: 1DTE models consistent (static vs WF gap < 5%); 0DTE range models show slight overfit warning (+5%)
- **Feature stability**: Importance rankings shift across market eras (Spearman ρ ≈ 0.4) — periodic retraining recommended

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                             │
│  datasets/          Raw 1-min parquets (adjusted + unadj)   │
│  data/              Aggregation, external data, events      │
│  datasets/update_parquet.py   Manual IBKR backfill script   │
├─────────────────────────────────────────────────────────────┤
│                    Feature Layer                            │
│  features/base.py         53 price/volume features          │
│  features/external.py     VRP, VIX dynamics, rates, events  │
│  features/interactions.py VRP × catalyst cross features     │
│  features/path.py         Intraday path smoothness          │
│  features/registry.py     Canonical feature lists (122 tot) │
├─────────────────────────────────────────────────────────────┤
│                     Model Layer                             │
│  models/training.py       XGBoost / LightGBM / RF training  │
│  models/evaluation.py     AUC, AP, threshold backtesting    │
│  models/prediction.py     Inference with saved models       │
├─────────────────────────────────────────────────────────────┤
│                   Live Trading Layer                        │
│  live/fetch_data.py       Auto-fetch (IBKR → yfinance)      │
│  live/trader.py           0DTE straddle execution via IBKR  │
│  live/notify.py           Bark push notifications           │
├─────────────────────────────────────────────────────────────┤
│                   Interface Layer                           │
│  cli/predict.py           Daily prediction CLI              │
│  cli/pipeline.py          Full retrain pipeline             │
│  server/                  FastAPI web dashboard             │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
qqq_trading/
├── config.py                         # Config dataclass + YAML loader
├── config/default.yaml               # Centralized hyperparameters & paths
├── datasets/                         # Raw data (git-ignored)
│   ├── QQQ_1min_adjusted.parquet     # Split/dividend adjusted, ~4.1M bars
│   ├── QQQ_1min_unadjusted.parquet   # Raw unadjusted prices
│   ├── fomc_dates.csv                # FOMC announcement dates (2000–2026)
│   └── update_parquet.py             # Manual IBKR backfill script
├── data/                             # Data processing
│   ├── daily_metrics.py              # 1-min → daily OHLCV + returns + flags
│   ├── external_data.py              # VIX / VVIX / Treasury yields (yfinance)
│   └── event_calendar.py             # FOMC / NFP / earnings season calendar
├── features/                         # Feature engineering (4 groups)
│   ├── base.py                       # 53 base features (returns, ranges, volume)
│   ├── external.py                   # VRP, VIX term structure, rates, events
│   ├── interactions.py               # VRP × catalyst cross features
│   ├── path.py                       # Intraday path smoothness metrics
│   └── registry.py                   # Canonical feature name lists
├── models/                           # ML models
│   ├── training.py                   # Train XGBoost / LightGBM / RandomForest
│   ├── evaluation.py                 # AUC, AP, precision-recall, threshold sweep
│   └── prediction.py                 # Load model → compute features → predict
├── live/                             # Live trading automation
│   ├── fetch_data.py                 # IBKR-first, yfinance-fallback data fetcher
│   ├── trader.py                     # 0DTE ATM straddle auto-trader (IBKR)
│   └── notify.py                     # Bark push notifications (iOS)
├── server/                           # Web dashboard (FastAPI + Jinja2)
│   ├── app.py                        # Routes: dashboard, signal, history, status
│   ├── services.py                   # Background fetch/predict services
│   └── templates/                    # HTML templates
├── cli/                              # Command-line interface
│   ├── predict.py                    # Daily prediction (1DTE / 0DTE / both)
│   └── pipeline.py                   # Full retrain: data → features → model
├── eval/                             # Evaluation & reporting
│   ├── model_eval.py                 # Cross-target AUC/AP comparison
│   ├── signal_report.py              # Signal alerts, monthly stats, missed moves
│   ├── daily_compare.py              # Daily prediction vs actual comparison
│   └── test_data_quality.py          # Data integrity checks
├── research/                         # Research & analysis scripts
│   ├── 01_daily_metrics.py           # Build & summarize daily metrics
│   ├── 02_pattern_analysis.py        # Volatility clustering & regime analysis
│   ├── 05_window_analysis.py         # Rolling window feature stability
│   ├── 09_full_backtest.py           # End-to-end strategy backtest
│   ├── 10_options_backtest.py        # Options P&L simulation
│   ├── 12_robustness_test.py         # Monotonicity & walk-forward validation
│   └── 13_occam_razor.py             # Feature reduction / model simplification
├── tests/                            # pytest suite (15 modules)
├── utils/                            # Shared utilities
│   ├── paths.py                      # Project path constants
│   ├── plotting.py                   # Matplotlib helpers
│   └── splits.py                     # Time-series train/val/test splitting
├── docs/                             # Documentation
│   ├── data.md                       # Data layer specs
│   ├── features.md                   # Feature engineering details
│   ├── models.md                     # Model training & performance
│   ├── trading_guide.md              # Signal → trade decision flow
│   └── research.md                   # Research script guide
└── output/                           # Generated outputs (git-ignored)
    ├── model/                        # Saved models (.joblib)
    ├── charts/                       # Analysis charts (.png)
    └── *.parquet                     # Intermediate feature datasets
```

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run daily prediction
python -m cli.predict --mode both --threshold 0.5
python -m cli.predict --format json

# Retrain model from scratch
python -m cli.pipeline --preset production --train-end 2022-12-31

# Update raw 1-min data from IBKR (requires TWS/Gateway running)
python datasets/update_parquet.py              # live port 7496
python datasets/update_parquet.py --port 7497  # paper trading
python datasets/update_parquet.py --dry-run    # check gap only

# Fetch live data for prediction (auto: IBKR → yfinance fallback)
python -m live.fetch_data --days 5 --merge --validate

# Start web dashboard
python -m server

# Run tests
pytest tests/ -v

# Run research scripts
python research/09_full_backtest.py
python research/12_robustness_test.py
```

## Data

| File | Description | Size |
|------|-------------|------|
| `QQQ_1min_adjusted.parquet` | Split/dividend adjusted 1-min OHLCV | ~4.1M bars |
| `QQQ_1min_unadjusted.parquet` | Raw unadjusted prices | ~4.1M bars |
| `fomc_dates.csv` | FOMC announcement dates | 2000–2026 |

- **Coverage**: 2000-01-03 to present (~6,600 trading days)
- **Sessions**: Regular hours (9:30–16:00) + extended hours (pre-market 4:00–9:29, post-market 16:01–19:59)
- **Update**: Run `python datasets/update_parquet.py` to backfill from IBKR

## Trading Strategy

The system targets **0DTE QQQ straddles** — buying ATM calls + puts at market open on days the model predicts high intraday volatility.

**Daily workflow:**
1. Pre-market: fetch latest data → compute features → generate signal
2. If signal confidence ≥ threshold → buy ATM straddle at 9:40 ET
3. Position sizing: 2 straddle pairs (4 contracts), max $3/share premium
4. Exit: sell half when position doubles, manual exit for remainder
5. Push notification via Bark with signal details

## Tech Stack

- **ML**: XGBoost, LightGBM, scikit-learn, SHAP
- **Data**: pandas, NumPy, yfinance
- **Broker**: Interactive Brokers (ib_async)
- **Web**: FastAPI, Jinja2, uvicorn
- **Testing**: pytest

## Limitations

- No individual stock earnings dates (only earnings season proxy)
- No CPI / GDP / PMI event calendar integration
- Feature importance shifts across market eras — periodic retraining required
- 0DTE models have slight overfitting risk vs walk-forward baseline
- \>5% move prediction has insufficient test samples (3 events in 786 days)

---

# QQQ 日内波动率预测与 0DTE 跨式期权交易系统

一套量化交易系统，用于预测 QQQ（纳斯达克100 ETF）的日内大幅波动，并执行 0DTE/1DTE 期权跨式策略。基于 26 年的 1 分钟 OHLCV 数据（约 410 万根 K 线），系统整合了特征工程、梯度提升模型和 IBKR 实时接口，实现从信号生成到自动执行的完整闭环。

## 核心亮点

- **122 个工程特征**，涵盖 4 大类：价格/成交量动态、宏观外部数据（VIX/VVIX/国债收益率）、事件催化剂（FOMC/NFP/财报季）、交叉项（VRP × 催化剂）
- **滚动前瞻验证**，所有 6 个预测目标均通过单调性检验——模型置信度越高，命中率越高
- **实盘交易管线**：IBKR 数据拉取 → 特征计算 → 信号生成 → 0DTE 跨式执行 → Bark 推送通知
- **Web 仪表盘**（FastAPI）实时展示信号、交易记录和数据状态

## 核心结果（测试期：2023.01 – 2026.02，786 个交易日）

| 场景 | 目标 | AUC | @0.7 命中率 | 用途 |
|------|------|-----|------------|------|
| **0DTE 振幅>2%** | 日内振幅 >2% | **0.826** | **66%**（59 次预警） | 开盘买跨式 |
| **0DTE 振幅>3%** | 日内振幅 >3% | **0.864** | **57%**（14 次预警） | 高确信跨式 |
| 1DTE \|C2C\|>2% | 次日绝对收益 >2% | 0.743 | 62%（13 次预警） | 隔夜跨式 |
| 1DTE \|C2C\|>1% | 次日绝对收益 >1% | 0.644 | 69%（51 次预警） | 日度波动率预测 |

### 规律发现
- **波动率聚集**：P(>2% | 昨日>2%) = 32.4%，无条件概率 16.1%（提升 2 倍）
- **盘前信号**：盘前振幅前 20% 的日子，大幅波动率 32.5%；后 20% 仅 3.8%（提升 8.6 倍）
- **Gamma 陷阱**：VRP 低迷 + FOMC 周 = 42.9% 大幅波动概率（提升 2.7 倍）
- **高波 + 催化剂**：高波动率环境 + FOMC = 47.6%（提升 3.0 倍）

### 稳健性
- **单调性**：6 个目标全部通过——模型置信度校准良好
- **滚动前瞻**：1DTE 模型稳定（静态 vs 滚动差距 < 5%）；0DTE 振幅模型存在轻微过拟合风险（+5%）
- **特征稳定性**：重要性排名在不同市场时期变化显著（Spearman ρ ≈ 0.4）——建议定期重训练

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                       数据层                                 │
│  datasets/          原始 1 分钟 K 线（调整后 + 未调整）          │
│  data/              日度聚合、外部数据、事件日历                  │
│  datasets/update_parquet.py   手动 IBKR 回填脚本               │
├─────────────────────────────────────────────────────────────┤
│                      特征层                                  │
│  features/base.py         53 个价格/成交量基础特征               │
│  features/external.py     VRP、VIX 期限结构、利率、事件          │
│  features/interactions.py VRP × 催化剂交叉特征                  │
│  features/path.py         日内路径平滑度                        │
│  features/registry.py     标准特征名列表（共 122 个）            │
├─────────────────────────────────────────────────────────────┤
│                      模型层                                  │
│  models/training.py       XGBoost / LightGBM / RF 训练        │
│  models/evaluation.py     AUC、AP、阈值回测                    │
│  models/prediction.py     加载模型 → 计算特征 → 推理             │
├─────────────────────────────────────────────────────────────┤
│                    实盘交易层                                  │
│  live/fetch_data.py       自动数据获取（IBKR → yfinance 回退）   │
│  live/trader.py           0DTE ATM 跨式自动交易（IBKR）         │
│  live/notify.py           Bark 推送通知（iOS）                  │
├─────────────────────────────────────────────────────────────┤
│                      接口层                                   │
│  cli/predict.py           每日预测命令行工具                     │
│  cli/pipeline.py          全量重训练管线                        │
│  server/                  FastAPI Web 仪表盘                   │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

```bash
# 安装依赖
pip install -e ".[dev]"

# 运行每日预测
python -m cli.predict --mode both --threshold 0.5

# 从零重训练模型
python -m cli.pipeline --preset production --train-end 2022-12-31

# 从 IBKR 更新原始 1 分钟数据（需启动 TWS/Gateway）
python datasets/update_parquet.py              # 实盘端口 7496
python datasets/update_parquet.py --port 7497  # 模拟盘
python datasets/update_parquet.py --dry-run    # 仅查看缺口

# 拉取实时数据用于预测（自动：IBKR → yfinance 回退）
python -m live.fetch_data --days 5 --merge --validate

# 启动 Web 仪表盘
python -m server

# 运行测试
pytest tests/ -v
```

## 交易策略

系统目标是 **0DTE QQQ 跨式期权**——在模型预测日内高波动的交易日，于开盘时买入 ATM 看涨 + 看跌期权。

**每日流程：**
1. 盘前：拉取最新数据 → 计算特征 → 生成信号
2. 信号置信度 ≥ 阈值 → 9:40 ET 买入 ATM 跨式
3. 仓位：2 组跨式（4 张合约），权利金上限 $3/股
4. 退出：翻倍减半，剩余人工管理
5. 通过 Bark 推送信号详情

## 技术栈

- **机器学习**：XGBoost、LightGBM、scikit-learn、SHAP
- **数据处理**：pandas、NumPy、yfinance
- **券商接口**：Interactive Brokers（ib_async）
- **Web**：FastAPI、Jinja2、uvicorn
- **测试**：pytest

## 局限性

- 无个股财报日期（仅财报季代理变量）
- 无 CPI / GDP / PMI 等宏观事件日历
- 特征重要性在不同市场时期变化显著——需定期重训练
- 0DTE 模型相对滚动前瞻基准有轻微过拟合风险
- \>5% 波动预测样本不足（786 天中仅 3 次事件）
