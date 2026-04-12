# Feature Engineering

All features use `shift(1)` to avoid look-ahead bias. Total: 122 base + 26 external + 43 interaction + 30 path = 221 possible features.

## Usage

```python
from qqq_trading.features import (
    engineer_base_features,
    engineer_all_external,
    build_interaction_features,
    build_path_features,
    get_full_features,
)

# Build progressively
df = engineer_base_features(daily_metrics)           # +53 features
df = engineer_all_external(df, external_data)        # +26 features
df = build_interaction_features(df)                  # +43 features
df = build_path_features(df)                         # +30 features (optional)

# Get matching feature list
cols = get_full_features(include_interactions=True, include_path=False)
```

## Feature Groups

### Base Features (53) — `features.base`

Price/volume only, no external data needed.

| Group | Count | Examples |
|-------|-------|---------|
| Lagged returns | 15 | `ret_lag1..5`, `abs_ret_lag1..5`, `range_lag1..5` |
| Rolling stats | 12 | `mean_abs_ret_{5,10,20,60}d`, `std_ret_*`, `realized_vol_*` |
| Vol ratios | 2 | `vol_ratio_5_60`, `vol_ratio_10_60` |
| Drawdown/runup | 6 | `max_dd_lag1..3`, `max_ru_lag1..3` |
| Volume | 2 | `vol_ratio_20d`, `vol_trend_5_20` |
| Calendar | 6 | `dow`, `month`, `is_opex_week`, `days_since_2pct_move` |
| Technical | 8 | `rsi_14`, `dist_from_ma{20,50,200}`, `proximity_*` |
| Gap | 2 | `gap_ret_lag1`, `abs_gap_lag1` |

### External Features (26) — `features.external`

Requires VIX/VVIX/Treasury data from yfinance.

| Group | Count | Key Insight |
|-------|-------|-------------|
| **VRP** | 5 | `vrp_20d = VIX/100 - realized_vol_20d` — core signal, replaces raw VIX |
| VIX dynamics | 4 | Changes and spikes, not levels |
| VVIX | 2 | `vvix_vix_ratio` — expects VIX to move? |
| Rates | 5 | Yield curve slope, inversion, rate shocks |
| Events | 10 | FOMC/NFP day/eve, days-to-event, earnings season |

**Why VRP > Raw VIX**: Raw VIX is highly correlated with realized vol (r~0.85). VRP = Implied - Realized captures the *fear premium*, which is the actual predictive signal. Adding raw VIX hurt the model (-0.008 AUC); switching to VRP improved it (+0.017 AUC).

### Interaction Features (43) — `features.interactions`

Cross signals: regime state x catalyst proximity.

| Category | Examples | Thesis |
|----------|---------|--------|
| **Gamma Trap** | `complacent_X_fomc`, `lowvol_X_fomc` | Market asleep + shock = big move |
| VRP x Event | `vrp_high_X_fomc_imminent`, `vrp_extreme_X_any_catalyst` | Over-hedged + event = explosion |
| Vol x Event | `highvol_X_fomc`, `highvol_X_any_catalyst` | Momentum + event = continuation |
| Continuous | `vrp_X_fomc_urgency = vrp * 1/(days_to_fomc+1)` | VRP effect grows as event approaches |

**Key conditional probabilities** (from training data):

| Condition | P(>2% tomorrow) | vs Base 16.1% |
|-----------|-----------------|---------------|
| VRP high + FOMC imminent | 32.4% | 2.0x |
| Complacent VRP + FOMC | 42.9% | 2.7x |
| High vol + FOMC | 47.6% | 3.0x |
| Low vol + no catalyst | 2.9% | 0.2x |

### Path Features (30) — `features.path` (optional)

Smoothness / trend dependency over 63-day and 126-day windows.

| Feature | Range | Interpretation |
|---------|-------|---------------|
| `trend_r2` | [0, 1] | Linear R-squared; 1=smooth trend, 0=random |
| `fractal_eff` | [0, 1] | Net displacement / path length |
| `choppiness` | [0, 100] | Dreiss index; high=choppy, low=trending |
| `hurst` | [0, 1] | >0.5 trending, <0.5 mean-reverting |
| `trend_strength` | [0, inf) | \|return\| / vol; smooth trend=high |

**Marginal value**: +0.004 AUC. Recommended as rule-based overlay, not ML features.

## Feature Registry

`features.registry` is the **single source of truth** for all feature lists. Never hardcode feature names elsewhere.

```python
from qqq_trading.features.registry import (
    get_base_features,              # 53 features
    get_refined_external_features,  # 26 features
    get_interaction_features,       # 43 features
    get_path_features,              # 30 features
    get_full_features,              # composite builder
)

# Production model uses:
cols = get_full_features(include_interactions=True)  # 122 features
```
