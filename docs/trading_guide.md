# Trading Guide / 交易指南

## Signal Architecture / 信号架构

4-layer signal stack, ordered by strength and stability:

> 4层信号体系，按强度和稳定性排序：

| Layer | Signal | Source | Stability | Description |
|-------|--------|--------|-----------|-------------|
| 1 | Realized volatility clustering | `realized_vol_20d`, `mean_abs_ret_*` | **Highest** — physical market property | Recent vol predicts future vol |
| 2 | Pre-market amplitude | `premarket_range_today`, `premarket_ret_today` | **High** — 0DTE exclusive | Large pre-market moves often continue intraday |
| 3 | VRP × Catalyst | `vrp_high_X_fomc_imminent`, `complacent_X_fomc` | **Medium** — regime dependent | Fear premium + event = volatility expansion |
| 4 | Path smoothness | `trend_r2`, `choppiness` | **Low** — rule overlay only | Smooth trends breed complacency → fragility |

> | 层级 | 信号 | 来源 | 稳定性 | 说明 |
> |------|------|------|--------|------|
> | 1 | 已实现波动率聚集 | `realized_vol_20d`, `mean_abs_ret_*` | **最高**——市场物理特性 | 近期波动率预测未来波动率 |
> | 2 | 盘前波幅 | `premarket_range_today`, `premarket_ret_today` | **高**——0DTE专属 | 盘前大波动通常延续到盘中 |
> | 3 | VRP × 催化剂 | `vrp_high_X_fomc_imminent`, `complacent_X_fomc` | **中**——取决于市场状态 | 恐惧溢价 + 事件 = 波动率扩张 |
> | 4 | 路径平滑度 | `trend_r2`, `choppiness` | **低**——仅作规则叠加 | 平滑趋势滋生自满 → 脆弱性 |

---

## Daily Decision Flow / 每日决策流程

### Evening (after 16:00 ET) / 盘后（东部时间16:00后）

1. **Run 1DTE model** to assess tomorrow's probability:
   ```bash
   python -m qqq_trading.cli.predict --mode 1dte --format text
   ```
2. **Check catalysts**: Is there an FOMC meeting, NFP release, or are we in earnings season?
3. **Check VRP status**:
   - `vrp_20d > 0`: Market is **over-hedged** (fearful) — vol may already be priced into options
   - `vrp_20d < -0.05`: Market is **complacent** — **Gamma Trap territory**, options are cheap
4. **Decision**: If 1DTE confidence ≥ 0.6 → consider buying an overnight straddle (buy at close, hold through tomorrow)

> 1. **运行1DTE模型**评估明日概率：（见上方命令）
> 2. **检查催化剂**：是否有FOMC会议、NFP发布、或处于财报季？
> 3. **检查VRP状态**：
>    - `vrp_20d > 0`：市场**过度对冲**（恐惧）——波动率可能已反映在期权价格中
>    - `vrp_20d < -0.05`：市场**自满**——**Gamma陷阱区域**，期权便宜
> 4. **决策**：若1DTE置信度 ≥ 0.6 → 考虑买入隔夜跨式（收盘买入，持有到明天）

### Pre-market (4:00 – 9:29 ET) / 盘前（东部时间4:00–9:29）

1. **Monitor pre-market amplitude and volume** — large pre-market range is one of the strongest 0DTE signals
2. **Run 0DTE model**:
   ```bash
   python -m qqq_trading.cli.predict --mode 0dte --format text
   ```
3. **Decision**: If 0DTE Range>2% confidence ≥ 0.6 → buy ATM straddle at 9:30 open
4. **Automated flow**: `live/notify.py` runs at 9:29 AM ET and sends a Bark push notification with the signal

> 1. **监控盘前波幅和成交量**——盘前大波幅是最强的0DTE信号之一
> 2. **运行0DTE模型**：（见上方命令）
> 3. **决策**：若0DTE Range>2% 置信度 ≥ 0.6 → 在9:30开盘买入ATM跨式
> 4. **自动化流程**：`live/notify.py` 在东部时间9:29运行，通过Bark推送通知信号

### Intraday (9:30 – 16:00 ET) / 盘中（东部时间9:30–16:00）

If you entered a 0DTE straddle:
1. **Monitor position**: `live/trader.py` checks every 15 seconds
2. **Profit target**: Sell half at 2x (straddle doubles in value)
3. **Keep remainder**: Hold other half for potential larger move or manual exit
4. **Cutoff**: If no 2x by 3:30 PM, evaluate manual exit to avoid theta decay into close

> 如果你建立了0DTE跨式仓位：
> 1. **监控仓位**：`live/trader.py` 每15秒检查一次
> 2. **止盈目标**：在2倍时卖出一半（跨式价值翻倍）
> 3. **保留剩余**：持有另一半等待更大波动或手动退出
> 4. **截止时间**：若到15:30仍未达2倍，评估手动退出以避免临近收盘的时间价值衰减

---

## Operating Profiles / 操作模式

Choose a profile based on your risk tolerance and trading frequency:

> 根据你的风险偏好和交易频率选择模式：

| Profile | Model | Threshold | Annual Signals | Hit Rate | Risk | Best For |
|---------|-------|-----------|----------------|----------|------|----------|
| **High Conviction** | 0DTE Range>2% | ≥ 0.8 | ~40 | 75% | Lowest | Small accounts, learning |
| **Conservative** | 0DTE Range>2% | ≥ 0.7 | ~59 | 66% | Low | Primary strategy |
| **Selective** | 0DTE Range>3% | ≥ 0.7 | ~14 | 57% | Low | Rare big-move hunting |
| **Aggressive** | 0DTE Range>2% | ≥ 0.5 | ~92 | 52% | Medium | Active traders |
| **Overnight** | 1DTE \|C2C\|>2% | ≥ 0.7 | ~13 | 62% | Medium | Swing traders |

> | 模式 | 模型 | 阈值 | 年信号数 | 命中率 | 风险 | 适合人群 |
> |------|------|------|---------|--------|------|---------|
> | **高确信** | 0DTE 范围>2% | ≥ 0.8 | ~40 | 75% | 最低 | 小账户、学习阶段 |
> | **保守** | 0DTE 范围>2% | ≥ 0.7 | ~59 | 66% | 低 | 主策略 |
> | **精选** | 0DTE 范围>3% | ≥ 0.7 | ~14 | 57% | 低 | 猎取罕见大波动 |
> | **激进** | 0DTE 范围>2% | ≥ 0.5 | ~92 | 52% | 中 | 活跃交易者 |
> | **隔夜** | 1DTE \|C2C\|>2% | ≥ 0.7 | ~13 | 62% | 中 | 波段交易者 |

---

## Key Conditional Probabilities / 关键条件概率

These are rule-based signals that work independently of the ML model. They can be used as manual overrides or confirmation signals.

> 这些是独立于ML模型的规则信号，可用作手动覆盖或确认信号。

### Gamma Trap (highest priority) / Gamma陷阱（最高优先级）

- **Condition**: VRP negative (VIX/100 < realized vol) + FOMC/NFP tomorrow
- **Probability**: 42.9% for >2% move (2.7x the base rate of 16.1%)
- **Action**: Buy straddle — volatility is **underpriced**
- **Why it works**: Market is complacent (implied vol < actual vol). An FOMC/NFP shock will force repricing of options, causing a volatility explosion. This is the highest-conviction setup.

> - **条件**：VRP为负（VIX/100 < 已实现波动率）+ 明天有FOMC/NFP
> - **概率**：>2%波动的概率42.9%（基准率16.1%的2.7倍）
> - **操作**：买入跨式——波动率被**低估**
> - **原理**：市场自满（隐含波动率 < 实际波动率）。FOMC/NFP冲击将迫使期权重新定价，引发波动率爆发。这是最高确信度的交易设置。

### High Vol + Catalyst / 高波动 + 催化剂

- **Condition**: Realized vol > 75th percentile + FOMC tomorrow
- **Probability**: 47.6% for >2% move (3.0x base rate)
- **Action**: Buy straddle, **but premium is already elevated** — need larger move to profit
- **Why it works**: Momentum + event = likely continuation. In high-vol regimes, catalysts amplify existing turbulence.

> - **条件**：已实现波动率 > 75分位 + 明天有FOMC
> - **概率**：>2%波动的概率47.6%（基准率的3.0倍）
> - **操作**：买入跨式，**但权利金已偏高**——需要更大波动才能盈利
> - **原理**：动量 + 事件 = 可能延续。在高波动状态中，催化剂放大现有动荡。

### Smooth Trend + Complacency + Catalyst / 平滑趋势 + 自满 + 催化剂

- **Condition**: 126-day R² > 0.85 + VRP negative + catalyst imminent
- **Probability**: 51.2% for >2% move
- **Action**: Market is most fragile; vol is cheap — strong buy straddle
- **Why it works**: Long, smooth trends breed extreme complacency → maximum vulnerability to shocks. When a catalyst hits, the repricing is violent because nobody is hedged.

> - **条件**：126日R² > 0.85 + VRP为负 + 催化剂临近
> - **概率**：>2%波动的概率51.2%
> - **操作**：市场最脆弱；波动率便宜——强买入跨式
> - **原理**：长期平滑趋势滋生极端自满 → 对冲击最脆弱。当催化剂到来时，因无人对冲，重新定价会非常剧烈。

### Safe Zone (sell vol / iron condor) / 安全区域（卖出波动率/铁秃鹰）

- **Condition**: Low volatility regime + no nearby catalysts
- **Probability**: Only 2.9% for >2% move (0.2x base rate)
- **Action**: Sell vol for time decay (iron condors, credit spreads)
- **Why it works**: In calm markets with no upcoming events, large moves are extremely rare. Theta decay works in your favor.

> - **条件**：低波动率状态 + 附近无催化剂
> - **概率**：>2%波动仅2.9%（基准率的0.2倍）
> - **操作**：卖出波动率赚时间价值衰减（铁秃鹰、信用价差）
> - **原理**：平静市场无即将到来的事件时，大波动极为罕见。Theta衰减对你有利。

---

## 8 Key Statistical Patterns / 8个关键统计模式

1. **Volatility clustering**: P(>2% | yesterday >2%) = 32.4% vs 16.1% base rate (2.0x). This is the single strongest pattern — volatility begets volatility.

   > **波动率聚集**：P(>2% | 昨日>2%) = 32.4% vs 基准16.1%（2.0倍）。这是最强的单一模式——波动产生波动。

2. **Pre-market signal**: Top-20% pre-market range → 32.5% large move probability vs 3.8% for bottom-20%. Almost 10x difference.

   > **盘前信号**：盘前波幅前20% → 32.5%大波动概率 vs 后20%的3.8%。接近10倍差异。

3. **Overnight gap**: |Gap| > 1% → 49.6% probability of |C2C| > 2%. Large gaps indicate overnight information flow.

   > **隔夜缺口**：|缺口| > 1% → 49.6%概率出现|C2C| > 2%。大缺口表明隔夜有重大信息流入。

4. **Calendar effects**: Weak — Thursday highest at 17.7%, but not statistically significant. Day-of-week is unreliable.

   > **日历效应**：弱——周四最高17.7%，但统计不显著。星期几不可靠。

5. **Vol regimes**: High vol regimes (>75th percentile) persist ~85% of the time. Once vol rises, it stays elevated.

   > **波动率状态**：高波动率状态（>75分位）有约85%的持续概率。一旦波动率升高，它会保持在高位。

6. **Volume**: Top-20% previous day volume → 20% large move probability vs 16% base rate. Modest effect (1.25x).

   > **成交量**：前一日成交量前20% → 20%大波动概率 vs 16%基准。效果温和（1.25倍）。

7. **Mean reversion**: Slight effect after 5-day down streaks — slightly higher probability of a large move.

   > **均值回复**：连续5天下跌后有轻微效应——大波动概率略有升高。

8. **Era clustering**: 2000–02, 2008–09, 2020, 2022 = high vol eras; 2013–17 = ultra-low vol. Model performance varies dramatically by era.

   > **时代聚集**：2000–02、2008–09、2020、2022 = 高波动时代；2013–17 = 超低波动。模型表现在不同时代差异巨大。

---

## Risk Management / 风险管理

### What to Expect in Live Trading / 实盘交易预期

- AUC will be **5–8% lower** than backtest due to data latency, execution slippage, and regime drift
- Hit rates will **degrade 10–15%** from reported numbers
- Feature importance **will drift** — retrain at least annually
- Low-vol years (2013/2017 type) will produce **almost no signals** — don't force trades
- High-vol years (2022 type) will produce **many signals** but with lower discrimination between days

> - AUC将比回测**低5-8%**，因为数据延迟、执行滑点和状态漂移
> - 命中率将比报告数字**下降10-15%**
> - 特征重要性**会漂移**——至少每年重训练一次
> - 低波动率年份（2013/2017类型）**几乎不会产生信号**——不要强行交易
> - 高波动率年份（2022类型）会产生**大量信号**但日间区分度更低

### Don'ts / 禁忌

| Don't | Why |
|-------|-----|
| Trade targets with AUC < 0.65 (e.g., >5%) | Insufficient statistical edge — you're gambling |
| Ignore option premium costs | A 66% hit rate is useless if premium > expected move |
| Go all-in on single signals | Even the best signal has 25–40% miss rate |
| Use models not retrained for > 1 year | Feature importance drifts too much |
| Trust 0DTE \|O2C\|>2% model | Confirmed overfit — 8.6% walk-forward gap |
| Trade in ultra-low vol (VIX < 12) | Base rate drops to ~3%, model has no edge |

> | 禁忌 | 原因 |
> |------|------|
> | 交易AUC < 0.65的目标（如>5%） | 统计优势不足——等于赌博 |
> | 忽视期权权利金成本 | 66%命中率在权利金>预期波动时毫无意义 |
> | 单一信号满仓 | 即使最好的信号也有25-40%的失误率 |
> | 使用超过1年未重训练的模型 | 特征重要性漂移过大 |
> | 信任0DTE \|O2C\|>2%模型 | 确认过拟合——Walk-forward差距8.6% |
> | 在超低波动率（VIX < 12）时交易 | 基准率降至约3%，模型无优势 |

### Position Sizing / 仓位管理

- Maximum **2 pairs** (4 contracts) per trade via `live/trader.py`
- Maximum pair cost: **$3.00/share** ($300/pair, $600 total risk per trade)
- On a $25,000 account, this is ~2.4% risk per trade
- Never risk more than **5% of account** on a single signal

> - 每笔交易最多**2组**（4份合约），通过`live/trader.py`控制
> - 每组最高成本：**$3.00/份**（每组$300，每笔交易总风险$600）
> - 在$25,000账户中，每笔交易约2.4%风险
> - 单一信号永远不要超过**账户的5%**风险

### Limitations / 局限性

| Limitation | Impact | Potential Fix |
|-----------|--------|---------------|
| No individual stock earnings dates | Miss AAPL/NVDA earnings day vol | Add earnings calendar API |
| No CPI/GDP/PMI/Jobless Claims | Miss macro data release vol | Add economic calendar |
| No geopolitical event tracking | Miss war/sanctions/tariff shocks | Manual override only |
| No GEX (gamma exposure) data | Miss dealer hedging flows | Add GEX feed (expensive) |
| Feature importance shifts across eras | Model degrades without retraining | Annual retrain + monitoring |
| 0DTE model slight overfit warning | +5% WF gap | Consider walk-forward ensemble |

> | 局限性 | 影响 | 潜在解决方案 |
> |--------|------|-------------|
> | 无个股财报日期 | 错过AAPL/NVDA财报日波动 | 添加财报日历API |
> | 无CPI/GDP/PMI/初请 | 错过宏观数据发布波动 | 添加经济日历 |
> | 无地缘政治事件追踪 | 错过战争/制裁/关税冲击 | 仅手动覆盖 |
> | 无GEX（gamma敞口）数据 | 错过做市商对冲流 | 添加GEX数据源（昂贵） |
> | 特征重要性跨时代漂移 | 不重训练模型退化 | 年度重训练+监控 |
> | 0DTE模型轻微过拟合警告 | +5% WF差距 | 考虑Walk-forward集成 |
