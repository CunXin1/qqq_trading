# Trading Guide

## Signal Architecture

4-layer signal stack, ordered by strength and stability:

| Layer | Signal | Source | Stability |
|-------|--------|--------|-----------|
| 1 | Realized volatility clustering | `realized_vol_20d`, `mean_abs_ret_*` | Highest — physical market property |
| 2 | Pre-market amplitude | `premarket_range_today`, `premarket_ret_today` | High — 0DTE exclusive |
| 3 | VRP x Catalyst | `vrp_high_X_fomc_imminent`, `complacent_X_fomc` | Medium — regime dependent |
| 4 | Path smoothness | `trend_r2`, `choppiness` | Low — rule overlay only |

## Daily Decision Flow

### Evening (after 16:00)

1. Run 1DTE model:
   ```bash
   python -m qqq_trading.cli.predict --mode 1dte --format text
   ```
2. Check catalysts: FOMC? NFP? Earnings season?
3. Check VRP status:
   - `vrp_20d > 0`: Market over-hedged (fearful) — vol may be priced in
   - `vrp_20d < -0.05`: Market complacent — **Gamma Trap territory**
4. If 1DTE confidence >= 0.6 → consider overnight straddle

### Pre-market (4:00 - 9:29)

1. Monitor pre-market amplitude and volume
2. Run 0DTE model:
   ```bash
   python -m qqq_trading.cli.predict --mode 0dte --format text
   ```
3. If 0DTE Range>2% confidence >= 0.6 → buy straddle at 9:30 open

## Operating Profiles

| Profile | Model | Threshold | Annual Signals | Hit Rate | Risk |
|---------|-------|-----------|----------------|----------|------|
| **Conservative** | 0DTE Range>2% | >= 0.7 | ~59 | 66% | Low |
| **Selective** | 0DTE Range>3% | >= 0.7 | ~14 | 57% | Low |
| **Aggressive** | 0DTE Range>2% | >= 0.5 | ~92 | 52% | Medium |
| **Overnight** | 1DTE \|C2C\|>2% | >= 0.7 | ~13 | 62% | Medium |
| **High Conviction** | 0DTE Range>2% | >= 0.8 | ~40 | 75% | Lowest |

## Key Conditional Probabilities

These are rule-based signals that work independently of the ML model:

### Gamma Trap (highest priority)
- **Condition**: VRP negative (VIX/100 < realized vol) + FOMC/NFP tomorrow
- **Probability**: 42.9% for >2% move
- **Action**: Buy straddle — volatility is underpriced
- **Why**: Market is complacent; FOMC shock will reprice options

### High Vol + Catalyst
- **Condition**: Realized vol > 75th percentile + FOMC tomorrow
- **Probability**: 47.6% for >2% move
- **Action**: Buy straddle, but premium already elevated
- **Why**: Momentum + event = likely continuation

### Smooth Trend + Complacency + Catalyst
- **Condition**: 126-day R-squared > 0.85 + VRP negative + catalyst imminent
- **Probability**: 51.2% for >2% move
- **Action**: Market most fragile; vol cheap
- **Why**: Long smooth trend breeds complacency → maximum vulnerability to shocks

### Safe Zone (sell vol / iron condor)
- **Condition**: Low volatility regime + no nearby catalysts
- **Probability**: Only 2.9% for >2% move
- **Action**: Sell vol for time decay

## 8 Key Statistical Patterns

1. **Volatility clustering**: P(>2% | yesterday >2%) = 32.4% vs 16.1% base (2.0x)
2. **Pre-market signal**: Top-20% premarket range → 32.5% large move vs 3.8% bottom-20%
3. **Overnight gap**: |Gap| > 1% → 49.6% P(|C2C| > 2%)
4. **Calendar effects**: Weak — Thursday highest (17.7%), not statistically significant
5. **Vol regimes**: High vol (>75th pct) persists ~85% of the time
6. **Volume**: Top-20% previous day volume → 20% vs 16% base (modest)
7. **Mean reversion**: Slight effect after 5-day down streaks
8. **Era clustering**: 2000-02, 2008-09, 2020, 2022 = high vol; 2013-17 = ultra low

## Risk Management

### What to expect in live trading
- AUC will be 5-8% lower than backtest
- Hit rates will degrade 10-15%
- Feature importance will drift — retrain annually
- Low-vol years (2013/2017 type) will have almost no signals
- High-vol years (2022 type) will have many signals but lower discrimination

### Don't
- Trade targets with AUC < 0.65 (e.g., >5% predictions)
- Ignore option premium costs — a 66% hit rate is useless if premium > expected move
- Go all-in on single signals
- Use models not retrained for > 1 year
- Trust 0DTE |O2C|>2% model (overfit, 8.6% WF gap)

### Limitations
- No individual stock earnings dates (only earnings season proxy)
- No CPI/GDP/PMI/Jobless Claims event calendar
- No geopolitical event tracking
- No GEX (gamma exposure) data
- Feature importance shifts significantly across market eras
