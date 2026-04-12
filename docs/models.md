# Models

## Architecture

- **Algorithm**: XGBoost (primary), LightGBM, RandomForest
- **Task**: Binary classification — P(large move tomorrow)
- **Features**: 122 (53 base + 26 external + 43 interaction)
- **Training**: 2000-2022 (~5,700 days)
- **Test**: 2023.01-2026.02 (~780 days)

## Hyperparameters

Two presets defined in `config/default.yaml`:

| Parameter | Base | Production |
|-----------|------|-----------|
| n_estimators | 300 | 500 |
| max_depth | 5 | 5 |
| learning_rate | 0.05 | 0.03 |
| subsample | 0.8 | 0.8 |
| colsample_bytree | 0.8 | 0.7 |
| reg_alpha | 0.0 | 0.1 |
| reg_lambda | 1.0 | 1.0 |

## Training

```python
from qqq_trading import train_model, load_config

config = load_config()
model = train_model(X_train, y_train, "xgboost", config.model.production)
```

Or via CLI:
```bash
python -m qqq_trading.cli.pipeline --preset production --train-end 2022-12-31
```

## Saved Models

| Model | File | Features | Use Case |
|-------|------|----------|----------|
| **Interaction** | `interaction_model.joblib` | 121 | Production: best AUC |
| **Base** | `next_day_2pct_model.joblib` | 53 | Baseline: no external data |

## Evaluation

```python
from qqq_trading import evaluate_model, backtest_thresholds

metrics = evaluate_model(y_test, y_proba)
# -> {'auc': 0.826, 'ap': 0.52, 'brier': 0.08, ...}

bt = backtest_thresholds(y_test, y_proba, [0.3, 0.5, 0.7])
# -> DataFrame with threshold, alerts, hits, hit_rate, coverage, lift
```

## Performance (Test 2023.1 - 2026.2)

### Best Models

| Scenario | Target | AUC | @0.7 Hit Rate | Annual Signals |
|----------|--------|-----|---------------|----------------|
| **0DTE Range>2%** | Intraday range >2% | **0.826** | **66%** | ~59 |
| **0DTE Range>3%** | Intraday range >3% | **0.864** | **57%** | ~14 |
| 1DTE \|C2C\|>2% | Tomorrow \|return\| >2% | 0.743 | 62% | ~13 |

### Expected Live Performance

Apply 5-8% AUC discount and 10-15% hit rate discount for live trading:

| Model | Reported AUC | Expected Live AUC |
|-------|-------------|-------------------|
| 0DTE Range>2% | 0.824 | 0.77-0.80 |
| 0DTE Range>3% | 0.848 | 0.80-0.83 |
| 1DTE \|C2C\|>2% | 0.740 | 0.69-0.72 |

### Avoid

- **0DTE \|O2C\|>2%**: Walk-forward gap 8.6% — overfitting confirmed
- **>5% targets**: Only 3 test events — insufficient data

## Model Evolution (Historical Context)

The model went through 7 phases of development:

1. **Base (53 features)**: Pure price/volume → AUC 0.73
2. **+Raw VIX/rates**: Hurt performance (collinearity) → AUC 0.718
3. **+VRP instead**: VRP = VIX - Realized Vol → fixed the signal
4. **+Interactions**: VRP x FOMC cross features → AUC 0.752
5. **+Path smoothness**: Marginal gain (+0.004), kept as optional

**Core insight**: Volatility clustering is the dominant alpha. Top-5 features capture 91% of total AUC.

## Robustness

### Monotonicity — ALL PASS

Confidence threshold 0.3 → 0.95: hit rate increases smoothly.

0DTE Range>2%: 38% → 92% with zero violations.

### Walk-Forward CV

5-year rolling window, 5-day purge, yearly test 2010-2025:

| Model | Static AUC | WF Mean | Gap | Verdict |
|-------|-----------|---------|-----|---------|
| 1DTE \|C2C\|>2% | 0.740 | 0.693 | +4.7% | CONSISTENT |
| 0DTE Range>2% | 0.824 | 0.773 | +5.0% | BORDERLINE |
| 0DTE Range>3% | 0.848 | 0.805 | +4.3% | CONSISTENT |

**Failure years**: 2013 (0.544), 2017 (0.409) — ultra-low volatility, base rate <3%.

### Feature Stability — UNSTABLE

Spearman rho between era feature importance rankings: 0.36-0.47.

Core category (volatility rolling stats) always dominates, but specific windows shift.

**Must retrain at least annually.**

### Occam's Razor

Top-K feature ablation shows 91% AUC retention with just 5 features:

| K | AUC | Retention |
|---|-----|-----------|
| 1 | 0.713 | 86.5% |
| 3 | 0.753 | 91.5% |
| 5 | 0.746 | 91.5% |
| 7 | 0.786 | 95.4% |
| Full (105) | 0.825 | 100% |

**Top-5 consensus features** (all volatility clustering):
1. `realized_vol_20d`
2. `mean_abs_ret_10d`
3. `std_ret_20d`
4. `realized_vol_10d`
5. `mean_abs_ret_20d`
