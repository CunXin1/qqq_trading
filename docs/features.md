# Feature Engineering / 特征工程

All features use `shift(1)` to avoid look-ahead bias — every feature is computed from data available **before** the prediction date. Total feature space: 53 base + 26 external + 43 interaction + 30 path = **152** possible features.

> 所有特征均使用 `shift(1)` 避免前视偏差——每个特征仅使用预测日期**之前**的数据计算。总特征空间：53基础 + 26外部 + 43交互 + 30路径 = **152**个可用特征。

---

## Usage / 使用方法

```python
from qqq_trading.features import (
    engineer_base_features,
    engineer_all_external,
    build_interaction_features,
    build_path_features,
    get_full_features,
)

# Build progressively / 逐层构建
df = engineer_base_features(daily_metrics)           # +53 features
df = engineer_all_external(df, external_data)        # +26 features
df = build_interaction_features(df)                  # +43 features
df = build_path_features(df)                         # +30 features (optional)

# Get matching feature list / 获取对应特征列表
cols = get_full_features(include_interactions=True, include_path=False)
```

The build order matters: external features depend on base features (e.g., `realized_vol_20d` for VRP), and interaction features depend on both base and external features.

> 构建顺序很重要：外部特征依赖基础特征（如VRP需要`realized_vol_20d`），交互特征同时依赖基础和外部特征。

---

## Feature Groups / 特征分组

### Base Features (53) — `features.base` / 基础特征

Price/volume features derived from QQQ daily metrics only. No external data needed.

> 仅从QQQ日线指标衍生的价格/成交量特征，不需要外部数据。

| Group | Count | Examples | Description |
|-------|-------|---------|-------------|
| Lagged returns | 15 | `ret_lag1..5`, `abs_ret_lag1..5`, `range_lag1..5` | Past 1-5 day returns (signed, absolute, and range) |
| Rolling stats | 12 | `mean_abs_ret_{5,10,20,60}d`, `std_ret_*`, `realized_vol_*` | Rolling window volatility at 4 time scales |
| Vol ratios | 2 | `vol_ratio_5_60`, `vol_ratio_10_60` | Short-term vol / long-term vol — detects vol regime changes |
| Drawdown/runup | 6 | `max_dd_lag1..3`, `max_ru_lag1..3` | Previous 1-3 day intraday max drawdown/runup |
| Volume | 2 | `vol_ratio_20d`, `vol_trend_5_20` | Relative volume vs 20d avg; 5d/20d volume trend |
| Calendar | 6 | `dow`, `month`, `is_opex_week`, `days_since_2pct_move` | Day of week, month, OPEX week flag, recency of last big move |
| Technical | 8 | `rsi_14`, `dist_from_ma{20,50,200}`, `proximity_*` | RSI, distance from moving averages, proximity to 20/50d high/low |
| Gap | 2 | `gap_ret_lag1`, `abs_gap_lag1` | Previous day's gap (overnight return) |

> | 组别 | 数量 | 示例 | 说明 |
> |------|------|------|------|
> | 滞后收益率 | 15 | `ret_lag1..5`, `abs_ret_lag1..5`, `range_lag1..5` | 过去1-5天的收益率（有方向、绝对值、日内范围） |
> | 滚动统计 | 12 | `mean_abs_ret_{5,10,20,60}d`, `std_ret_*`, `realized_vol_*` | 4个时间尺度的滚动波动率 |
> | 波动率比值 | 2 | `vol_ratio_5_60`, `vol_ratio_10_60` | 短期/长期波动率——检测波动率状态切换 |
> | 回撤/涨幅 | 6 | `max_dd_lag1..3`, `max_ru_lag1..3` | 前1-3天的日内最大回撤/最大涨幅 |
> | 成交量 | 2 | `vol_ratio_20d`, `vol_trend_5_20` | 相对成交量（vs 20日均值）；5日/20日量趋势 |
> | 日历 | 6 | `dow`, `month`, `is_opex_week`, `days_since_2pct_move` | 星期几、月份、OPEX周标记、距上次>2%波动天数 |
> | 技术指标 | 8 | `rsi_14`, `dist_from_ma{20,50,200}`, `proximity_*` | RSI、均线偏离度、接近20/50日高低点的程度 |
> | 缺口 | 2 | `gap_ret_lag1`, `abs_gap_lag1` | 前一日缺口（隔夜收益率） |

**Key design choice**: Rolling windows at 4 scales (5/10/20/60 day) capture different volatility regimes — 5d for recent shock, 60d for structural regime.

> **核心设计思路**：四个滚动窗口（5/10/20/60天）捕获不同的波动率状态——5天检测近期冲击，60天检测结构性状态。

---

### External Features (26) — `features.external` / 外部特征

Requires VIX/VVIX/Treasury yield data from yfinance, plus event calendar.

> 需要从 yfinance 获取的 VIX/VVIX/国债收益率数据，加上事件日历。

| Group | Count | Key Features | Insight |
|-------|-------|-------------|---------|
| **VRP** | 5 | `vrp_20d`, `vrp_10d`, `vrp_5d`, `vrp_zscore`, `vrp_change_5d` | Volatility Risk Premium = VIX/100 - Realized Vol. Core predictive signal. |
| VIX dynamics | 4 | `vix_pct_change_1d/5d`, `vix_range`, `vix_spike` | VIX momentum and regime (spike = 10%+ daily jump) |
| VVIX | 2 | `vvix_vix_ratio`, `vvix_change_1d` | Vol of vol — is VIX itself expected to move? |
| Rates | 5 | `yield_curve_slope`, `yield_curve_inverted`, `tnx_10y_change`, `tnx_10y_vol_20d`, `rate_shock` | Yield curve shape, rate volatility, 2σ rate shock |
| Events | 10 | `is_fomc_day/eve`, `days_to_fomc`, `fomc_week`, `is_nfp_day/eve`, `days_to_nfp`, `is_macro_event_day/eve`, `is_earnings_season` | Macro event proximity flags |

> | 组别 | 数量 | 核心特征 | 含义 |
> |------|------|---------|------|
> | **VRP（波动率风险溢价）** | 5 | `vrp_20d`, `vrp_10d`, `vrp_5d`, `vrp_zscore`, `vrp_change_5d` | VRP = VIX/100 - 已实现波动率。核心预测信号。 |
> | VIX动态 | 4 | `vix_pct_change_1d/5d`, `vix_range`, `vix_spike` | VIX的动量和状态（spike = 单日涨10%+） |
> | VVIX | 2 | `vvix_vix_ratio`, `vvix_change_1d` | 波动率的波动率——VIX本身是否预期大幅波动？ |
> | 利率 | 5 | `yield_curve_slope`, `yield_curve_inverted`, `tnx_10y_change`, `tnx_10y_vol_20d`, `rate_shock` | 收益率曲线形态、利率波动率、2σ利率冲击 |
> | 事件 | 10 | `is_fomc_day/eve`, `days_to_fomc` 等 | 宏观事件临近度标记 |

#### Why VRP > Raw VIX / 为什么用 VRP 而不是直接用 VIX

Raw VIX is highly correlated with realized vol (r ≈ 0.85). Both go up when markets are volatile. VRP = Implied - Realized captures the **fear premium** — the gap between what the market *expects* and what *actually happened*. This is the actual predictive signal.

> 原始 VIX 与已实现波动率高度相关（r ≈ 0.85），两者在市场波动时同涨同跌。VRP = 隐含波动率 - 已实现波动率，捕获的是**恐惧溢价**——市场*预期*与*实际发生*之间的差距。这才是真正的预测信号。

Experimental results:
- Adding raw VIX to base model: AUC **dropped** 0.008 (collinearity noise)
- Switching to VRP: AUC **improved** 0.017 (genuine new signal)

> 实验结果：
> - 加入原始VIX：AUC **下降** 0.008（共线性噪声）
> - 改用VRP：AUC **提升** 0.017（真正的新信号）

**Interpretation of VRP values**:
- `vrp > 0` (VIX > realized vol): Market is **over-hedged / fearful** — implied vol exceeds actual vol, options are expensive
- `vrp < 0` (VIX < realized vol): Market is **complacent** — implied vol is below actual vol, options are cheap → **Gamma Trap territory**
- `vrp_zscore > 2`: VRP is extremely elevated relative to recent history → unusual fear

> **VRP值的解读**：
> - `vrp > 0`（VIX > 已实现波动率）：市场**过度对冲/恐惧**——隐含波动率超过实际波动率，期权偏贵
> - `vrp < 0`（VIX < 已实现波动率）：市场**自满**——隐含波动率低于实际波动率，期权偏便宜 → **Gamma陷阱区域**
> - `vrp_zscore > 2`：VRP相对近期历史极端偏高 → 异常恐惧

---

### Interaction Features (43) — `features.interactions` / 交互特征

Cross signals modeling regime state × catalyst proximity. The hypothesis: volatility expansion is most acute when specific market conditions coincide with upcoming catalysts.

> 交叉信号，建模市场状态 × 催化剂临近度。核心假设：当特定市场条件与即将到来的催化剂事件同时出现时，波动率扩张最为剧烈。

| Category | Count | Examples | Thesis |
|----------|-------|---------|--------|
| **Regime flags** | 6 | `vrp_high/extreme/positive/negative`, `high_vol/low_vol_regime` | Quantile-based regime classification |
| **Catalyst flags** | 7 | `fomc_imminent`, `nfp_imminent`, `any_catalyst_imminent`, `vix_spiked_3d`, `big_move_recent_3d` | Event proximity and recent momentum |
| **Gamma Trap** | 4 | `complacent_X_fomc`, `complacent_X_nfp`, `lowvol_X_fomc`, `lowvol_X_any_catalyst` | Market asleep + shock = big move |
| **VRP × Event** | 9 | `vrp_high_X_fomc_imminent`, `vrp_extreme_X_any_catalyst`, `vrp_pos_X_fomc` | Over-hedged + event = volatility explosion |
| **High Vol × Event** | 4 | `highvol_X_fomc/nfp/earnings/any_catalyst` | Momentum + event = likely continuation |
| **Momentum × Event** | 5 | `vix_spike3d_X_fomc`, `big_move_3d_X_fomc/nfp/earnings` | Recent turbulence + upcoming catalyst |
| **Rate × Event** | 1 | `rate_shock_X_fomc` | Bond market stress + Fed decision |
| **Continuous** | 4 | `vrp_X_fomc_urgency`, `vrp_zscore_X_catalyst` | VRP effect grows as event approaches: `vrp * 1/(days_to_fomc+1)` |
| **Yield curve × Event** | 2 | `curve_inverted_X_fomc`, `curve_inverted_X_any_catalyst` | Inverted curve signals recession risk |

> | 类别 | 数量 | 示例 | 假设 |
> |------|------|------|------|
> | **状态标记** | 6 | `vrp_high/extreme/positive/negative`, `high_vol/low_vol_regime` | 基于分位数的市场状态分类 |
> | **催化剂标记** | 7 | `fomc_imminent`, `nfp_imminent`, `any_catalyst_imminent` 等 | 事件临近度和近期动量 |
> | **Gamma陷阱** | 4 | `complacent_X_fomc`, `lowvol_X_fomc` | 市场沉睡 + 冲击 = 大波动 |
> | **VRP × 事件** | 9 | `vrp_high_X_fomc_imminent` 等 | 过度对冲 + 事件 = 波动率爆发 |
> | **高波动 × 事件** | 4 | `highvol_X_fomc/nfp/earnings/any_catalyst` | 动量 + 事件 = 延续 |
> | **动量 × 事件** | 5 | `vix_spike3d_X_fomc` 等 | 近期动荡 + 即将到来的催化剂 |
> | **利率 × 事件** | 1 | `rate_shock_X_fomc` | 债市压力 + 美联储决议 |
> | **连续型** | 4 | `vrp_X_fomc_urgency` | VRP效应随事件临近而增强：`vrp * 1/(距FOMC天数+1)` |
> | **收益率曲线 × 事件** | 2 | `curve_inverted_X_fomc` | 倒挂的收益率曲线信号衰退风险 |

#### Key Conditional Probabilities / 关键条件概率

These conditional probabilities were measured from the training data (2000–2022):

> 以下条件概率从训练数据（2000–2022）中测算：

| Condition | P(>2% tomorrow) | vs Base 16.1% | Multiplier |
|-----------|-----------------|---------------|-----------|
| VRP high + FOMC imminent | 32.4% | +16.3pp | 2.0x |
| Complacent VRP + FOMC | 42.9% | +26.8pp | **2.7x** |
| High vol + FOMC | 47.6% | +31.5pp | **3.0x** |
| Low vol + no catalyst | 2.9% | -13.2pp | 0.2x |

> | 条件 | P(明日>2%) | vs 基准16.1% | 倍率 |
> |------|-----------|-------------|------|
> | VRP高 + FOMC临近 | 32.4% | +16.3个百分点 | 2.0倍 |
> | VRP自满 + FOMC | 42.9% | +26.8个百分点 | **2.7倍** |
> | 高波动 + FOMC | 47.6% | +31.5个百分点 | **3.0倍** |
> | 低波动 + 无催化剂 | 2.9% | -13.2个百分点 | 0.2倍 |

---

### Path Features (30) — `features.path` (optional) / 路径特征（可选）

Smoothness and trend dependency features over 63-day and 126-day windows. These capture the *quality* of the price trajectory, not just its direction.

> 基于63天和126天窗口的平滑度和趋势依赖特征。捕获价格轨迹的*质量*，而不仅仅是方向。

| Feature | Range | Interpretation |
|---------|-------|---------------|
| `trend_r2` | [0, 1] | Linear regression R² of close prices. 1 = perfectly smooth trend, 0 = random walk |
| `fractal_eff` | [0, 1] | Net displacement / total path length. 1 = straight line, 0 = pure noise |
| `choppiness` | [0, 100] | Dreiss Choppiness Index. High = choppy/range-bound, low = trending |
| `hurst` | [0, 1] | Hurst exponent. >0.5 = trending/persistent, <0.5 = mean-reverting, 0.5 = random |
| `trend_strength` | [0, ∞) | \|period return\| / period volatility. High = smooth directional move |
| `up_day_ratio` | [0, 1] | Fraction of up days in window. 0.5 = balanced, >0.6 = strong bullish bias |
| `max_dd_window` | (-∞, 0] | Maximum drawdown over the rolling window |

> | 特征 | 范围 | 解读 |
> |------|------|------|
> | `trend_r2` | [0, 1] | 收盘价线性回归R²。1=完美平滑趋势，0=随机游走 |
> | `fractal_eff` | [0, 1] | 净位移/总路径长度。1=直线，0=纯噪声 |
> | `choppiness` | [0, 100] | Dreiss波动率指数。高=震荡/盘整，低=趋势运行 |
> | `hurst` | [0, 1] | Hurst指数。>0.5=趋势持续，<0.5=均值回复，0.5=随机 |
> | `trend_strength` | [0, ∞) | |区间收益率|/区间波动率。高=平滑的方向性运动 |
> | `up_day_ratio` | [0, 1] | 窗口内上涨日占比。0.5=均衡，>0.6=强多头偏向 |
> | `max_dd_window` | (-∞, 0] | 滚动窗口内的最大回撤 |

**Binary flags**: `smooth_trend` (R² > 0.85), `very_smooth` (R² > 0.95), `choppy` (R² < 0.3).

> **二元标记**：`smooth_trend`（R² > 0.85）、`very_smooth`（R² > 0.95）、`choppy`（R² < 0.3）。

**Cross features**: `smooth_X_catalyst`, `smooth_X_fomc`, `smooth_X_vrp_neg` — only built if prior feature groups exist.

> **交叉特征**：`smooth_X_catalyst`、`smooth_X_fomc`、`smooth_X_vrp_neg`——仅在前置特征组存在时构建。

**Marginal value**: Path features add only +0.004 AUC to the full model. Recommended as a rule-based overlay (e.g., "smooth trend + complacency + catalyst = maximum fragility"), not as ML features.

> **边际价值**：路径特征仅为完整模型增加 +0.004 AUC。建议作为规则层叠加使用（如"平滑趋势 + 自满 + 催化剂 = 最大脆弱性"），而非ML特征。

---

## Feature Registry / 特征注册表

`features.registry` is the **single source of truth** for all feature lists. Never hardcode feature names elsewhere — always import from the registry.

> `features.registry` 是所有特征列表的**唯一真相来源**。永远不要在其他地方硬编码特征名——始终从注册表导入。

```python
from qqq_trading.features.registry import (
    get_base_features,              # 53 features / 53个特征
    get_refined_external_features,  # 26 features / 26个特征
    get_interaction_features,       # 43 features / 43个特征
    get_path_features,              # 30 features / 30个特征
    get_full_features,              # composite builder / 组合构建器
    get_0dte_premarket_features,    # 3 premarket features / 3个盘前特征
)

# Production model uses: / 生产模型使用：
cols = get_full_features(include_interactions=True)  # 122 features
```

### Why a Registry? / 为什么需要注册表？

1. **Consistency**: Training and prediction must use the exact same feature list. A mismatch → silent prediction errors.
2. **Evolution**: When adding/removing features, update one place, not dozens.
3. **Documentation**: The registry is self-documenting — `get_full_features()` shows exactly what the model sees.

> 1. **一致性**：训练和预测必须使用完全相同的特征列表。不匹配 → 静默的预测错误。
> 2. **可演进**：增删特征时只需更新一处，而非数十处。
> 3. **文档性**：注册表自带文档——`get_full_features()` 精确展示模型所见的特征。
