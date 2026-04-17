# Research Scripts / 研究脚本

The `research/` directory contains analysis scripts that import from the `qqq_trading` core library. They are meant for exploration, validation, and chart generation — **not for production use**.

> `research/` 目录包含从 `qqq_trading` 核心库导入的分析脚本。用于探索、验证和图表生成——**不用于生产环境**。

---

## Prerequisites / 前置条件

```bash
pip install -e ".[dev]"
```

All scripts can be run standalone:
> 所有脚本可独立运行：

```bash
python research/09_full_backtest.py
```

---

## Script Overview / 脚本概览

| Script | Purpose | Depends On | Output |
|--------|---------|------------|--------|
| `01_daily_metrics.py` | Build daily metrics from 1-min data | Raw parquet files | `output/daily_metrics.parquet` |
| `02_pattern_analysis.py` | Discover 8 statistical patterns, generate charts | `daily_metrics.parquet` | 8 PNG charts |
| `04_report.py` | Generate HTML report with all charts | `daily_metrics.parquet`, charts | `output/report.html` |
| `05_window_analysis.py` | Test optimal training window length | `engineered_features.parquet` | Console tables + charts |
| `09_full_backtest.py` | Full backtest across 1%/2%/3%/5% thresholds | `interaction_features.parquet` | Console tables |
| `10_options_backtest.py` | 0DTE vs 1DTE straddle-specific backtest | `interaction_features.parquet` | Console tables |
| `12_robustness_test.py` | Monotonicity, walk-forward CV, feature decay | `interaction_features.parquet` | Console tables + charts |
| `13_occam_razor.py` | Top-K feature ablation (minimum viable model) | `interaction_features.parquet` | Console tables |

> | 脚本 | 用途 | 依赖 | 输出 |
> |------|------|------|------|
> | `01_daily_metrics.py` | 从1分钟数据构建日线指标 | 原始parquet文件 | `output/daily_metrics.parquet` |
> | `02_pattern_analysis.py` | 发现8个统计模式并生成图表 | `daily_metrics.parquet` | 8张PNG图表 |
> | `04_report.py` | 生成包含所有图表的HTML报告 | `daily_metrics.parquet`，图表 | `output/report.html` |
> | `05_window_analysis.py` | 测试最优训练窗口长度 | `engineered_features.parquet` | 控制台表格+图表 |
> | `09_full_backtest.py` | 跨1%/2%/3%/5%阈值的完整回测 | `interaction_features.parquet` | 控制台表格 |
> | `10_options_backtest.py` | 0DTE vs 1DTE跨式专项回测 | `interaction_features.parquet` | 控制台表格 |
> | `12_robustness_test.py` | 单调性、Walk-forward CV、特征衰减 | `interaction_features.parquet` | 控制台表格+图表 |
> | `13_occam_razor.py` | Top-K特征消融（最小可行模型） | `interaction_features.parquet` | 控制台表格 |

---

## Execution Order / 执行顺序

If starting from scratch (no cached outputs):

> 如果从零开始（无缓存输出）：

```bash
# Step 1: Generate daily metrics from raw 1-min data
# 步骤1：从原始1分钟数据生成日线指标
python research/01_daily_metrics.py

# Step 2: Build features + train model (creates interaction_features.parquet)
# 步骤2：构建特征 + 训练模型（生成 interaction_features.parquet）
python -m cli.pipeline --preset production

# Step 3: Run any analysis script (order doesn't matter after step 2)
# 步骤3：运行任意分析脚本（步骤2之后顺序不限）
python research/02_pattern_analysis.py    # patterns + charts / 模式分析+图表
python research/09_full_backtest.py       # full threshold sweep / 全阈值扫描
python research/10_options_backtest.py    # 0DTE/1DTE comparison / 0DTE/1DTE对比
python research/12_robustness_test.py     # walk-forward + monotonicity / WF+单调性
python research/13_occam_razor.py         # feature ablation / 特征消融
python research/04_report.py              # HTML report (run last) / HTML报告（最后运行）
```

**Important**: Step 1 and Step 2 must run before any analysis. Step 2 produces the feature-enriched parquet files that all analysis scripts consume.

> **重要**：步骤1和步骤2必须在任何分析之前运行。步骤2生成所有分析脚本所需的特征增强parquet文件。

---

## What Each Script Does / 各脚本详解

### 01_daily_metrics.py / 日线指标构建

Aggregates 1-minute OHLCV data into daily metrics. Thin wrapper around `data.daily_metrics.build_daily_metrics()`.

> 将1分钟OHLCV数据聚合为日线指标。是`data.daily_metrics.build_daily_metrics()`的薄封装。

**What it produces**:
- `output/daily_metrics.parquet` — 6,572 rows × 35 columns
- Console output: summary statistics, top-20 largest move days, data coverage report

> **产出**：
> - `output/daily_metrics.parquet` — 6,572行 × 35列
> - 控制台输出：汇总统计、最大波动前20天、数据覆盖报告

**When to re-run**: Only when raw 1-min data is updated (new data appended).

> **何时重跑**：仅当原始1分钟数据更新时（追加了新数据）。

---

### 02_pattern_analysis.py / 模式分析

Generates 8 analysis charts exploring patterns that precede large moves:

> 生成8张分析图表，探索大幅波动前的模式：

| Chart | What It Shows | Key Finding |
|-------|--------------|-------------|
| 1. Yearly frequency | Large move count per year | 2000–02, 2008–09, 2020, 2022 cluster heavily |
| 2. Calendar effects | Day-of-week, month, OPEX week | Weak effects — not statistically significant |
| 3. Volatility clustering | ACF, conditional probability by lag | P(>2% \| yesterday >2%) = 32.4% vs 16.1% — the strongest pattern |
| 4. Pre-market signal | Quintile analysis of pre-market range | Top-20% → 32.5% vs bottom-20% 3.8% — nearly 10x |
| 5. Gap analysis | Overnight gap size vs next-day move | \|Gap\| > 1% → 49.6% P(\|C2C\| > 2%) |
| 6. Volume predictor | Previous day volume quintiles vs move probability | Modest 1.25x lift at top quintile |
| 7. Vol regime transitions | Transition matrix between vol regimes | High vol persists ~85% of the time |
| 8. Consecutive patterns | Down streak analysis, serial correlation | Slight mean reversion after 5-day streaks |

> | 图表 | 展示内容 | 关键发现 |
> |------|---------|---------|
> | 1. 年度频率 | 每年大波动天数 | 2000–02、2008–09、2020、2022集中出现 |
> | 2. 日历效应 | 星期几、月份、OPEX周 | 效应微弱——统计不显著 |
> | 3. 波动率聚集 | ACF、按滞后的条件概率 | P(>2% \| 昨日>2%) = 32.4% vs 16.1%——最强模式 |
> | 4. 盘前信号 | 盘前波幅五分位分析 | 前20% → 32.5% vs 后20% 3.8%——近10倍 |
> | 5. 缺口分析 | 隔夜缺口大小vs次日波动 | \|缺口\| > 1% → 49.6% P(\|C2C\| > 2%) |
> | 6. 成交量预测 | 前日成交量五分位vs波动概率 | 最高五分位温和提升1.25倍 |
> | 7. 波动率状态转换 | 波动率状态间的转换矩阵 | 高波动率持续概率约85% |
> | 8. 连续模式 | 连跌分析、序列相关性 | 连续5天下跌后有轻微均值回复 |

**Output**: 8 PNG charts saved to `output/charts/`.

> **输出**：8张PNG图表保存至`output/charts/`。

---

### 04_report.py / HTML报告生成

Generates a self-contained HTML report (`output/report.html`) embedding all charts as base64 images, with summary statistics and model comparison tables.

> 生成自包含的HTML报告（`output/report.html`），将所有图表以base64图片嵌入，附带汇总统计和模型对比表。

**Features of the report**:
- All charts embedded (no external image dependencies)
- Summary statistics tables
- Model performance comparison
- Can be shared via email or Slack without needing the source data

> **报告特点**：
> - 所有图表内嵌（无外部图片依赖）
> - 汇总统计表格
> - 模型性能对比
> - 可通过邮件或Slack分享，无需源数据

**When to re-run**: After updating charts or model results. Run this **last** after other analysis scripts.

> **何时重跑**：更新图表或模型结果后。在其他分析脚本之后**最后**运行。

---

### 05_window_analysis.py / 训练窗口分析

Tests 4 approaches to selecting the optimal amount of training data:

> 测试4种方法来选择最优训练数据量：

| Approach | Description | Finding |
|----------|-------------|---------|
| Fixed window, variable start | Start from 2000/2005/2010/2015, fixed test | More data is better (diminishing returns after 15yr) |
| Walk-forward rolling | 3/5/7/10/15-year rolling windows | 10-year window is the sweet spot |
| Regime-specific | Separate by vol regime, train/test within each | High-vol regime model is strongest |
| Exponential decay | Weight recent data more heavily | Light decay (λ=0.995) marginally improves |

> | 方法 | 描述 | 发现 |
> |------|------|------|
> | 固定窗口、变起点 | 从2000/2005/2010/2015开始，固定测试期 | 数据越多越好（15年后收益递减） |
> | Walk-forward滚动 | 3/5/7/10/15年滚动窗口 | 10年窗口是最佳平衡点 |
> | 按状态分组 | 按波动率状态分别训练/测试 | 高波动率状态模型最强 |
> | 指数衰减 | 更高权重给近期数据 | 轻微衰减（λ=0.995）略有改善 |

**Conclusion**: 10–20 year window with light exponential decay is optimal. Using all available data from 2000 is near-optimal.

> **结论**：10–20年窗口加轻微指数衰减是最优的。使用2000年以来全部数据接近最优。

---

### 09_full_backtest.py / 完整回测

Comprehensive backtest across all move thresholds (1%, 2%, 3%, 5%) using both XGBoost and LightGBM.

> 使用XGBoost和LightGBM对所有波动阈值（1%、2%、3%、5%）进行全面回测。

**What it produces**:
- AUC for each target × model combination
- Hit rate and alert count at each confidence threshold (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
- Lift analysis (how much better than random baseline)
- Console summary table comparing all configurations

> **产出**：
> - 每个目标×模型组合的AUC
> - 各置信度阈值（0.3, 0.4, 0.5, 0.6, 0.7, 0.8）下的命中率和信号数
> - 提升倍数分析（比随机基线好多少）
> - 控制台汇总表对比所有配置

**Key output**: The performance matrix that identifies which target × model × threshold combination is production-worthy.

> **关键输出**：性能矩阵，识别哪个目标×模型×阈值组合适合生产使用。

---

### 10_options_backtest.py / 期权专项回测

Options-specific backtest comparing 0DTE vs 1DTE straddle strategies:

> 期权专项回测，对比0DTE和1DTE跨式策略：

| Comparison | 0DTE | 1DTE |
|-----------|------|------|
| Entry | Buy at open (9:30) | Buy at previous close (16:00) |
| Expiry | Same day | Next day |
| Pre-market features | Yes (available before open) | No (buy the night before) |
| Targets tested | \|O2C\| > X%, Range > X% | \|C2C\| > X% |
| Best result | Range>2% AUC 0.826 | \|C2C\|>2% AUC 0.743 |

> | 对比项 | 0DTE | 1DTE |
> |--------|------|------|
> | 入场 | 开盘买入（9:30） | 前一日收盘买入（16:00） |
> | 到期 | 当天 | 次日 |
> | 盘前特征 | 有（开盘前可用） | 无（前一晚买入） |
> | 测试目标 | \|O2C\| > X%, Range > X% | \|C2C\| > X% |
> | 最佳结果 | Range>2% AUC 0.826 | \|C2C\|>2% AUC 0.743 |

**Key finding**: 0DTE Range targets outperform because they capture both-direction moves (a day that goes +1.5% then -1.5% has a 3% range but ~0% C2C return).

> **关键发现**：0DTE Range目标表现更好，因为它捕获双向波动（一天先涨1.5%再跌1.5%，范围3%但C2C收益约为0%）。

---

### 12_robustness_test.py / 鲁棒性测试

Three robustness checks that determine whether the model is production-ready:

> 三项鲁棒性检查，判断模型是否可投产：

#### 1. Monotonicity Test / 单调性测试

Hit rate should increase monotonically as confidence threshold rises from 0.3 to 0.95. Any violation (higher confidence → lower hit rate) indicates a calibration problem.

> 命中率应随置信度阈值从0.3升至0.95单调递增。任何违反（更高置信度→更低命中率）表明校准问题。

**Result**: ALL 6 targets pass with zero violations. This is the strongest robustness signal — the model's probability outputs are well-ordered.

> **结果**：全部6个目标零违反通过。这是最强鲁棒性信号——模型概率输出排序良好。

#### 2. Walk-Forward CV / 滚动前向交叉验证

5-year rolling training window, 5-day purge gap, yearly test blocks from 2010–2025. Measures the gap between static (full-data) AUC and walk-forward (realistic) AUC.

> 5年滚动训练窗口，5天清洗间隔，2010–2025年度测试块。测量静态（全数据）AUC和Walk-forward（现实）AUC之间的差距。

- **Gap < 5%**: CONSISTENT — model generalizes well
- **Gap 5–10%**: BORDERLINE — slight overfitting, monitor closely
- **Gap > 10%**: OVERFIT — do not use in production

> - **差距 < 5%**：一致——模型泛化良好
> - **差距 5-10%**：边界——轻微过拟合，密切监控
> - **差距 > 10%**：过拟合——不要用于生产

#### 3. Feature Decay / 特征衰减

Compares feature importance rankings across market eras (e.g., 2000–2009 vs 2010–2019 vs 2020+). Uses Spearman rank correlation to measure consistency.

> 比较不同市场时代（如2000–2009 vs 2010–2019 vs 2020+）的特征重要性排名。使用Spearman秩相关测量一致性。

**Result**: Rho = 0.36–0.47 (unstable). The volatility category always dominates, but specific features shift. **Annual retraining is mandatory**.

> **结果**：Rho = 0.36–0.47（不稳定）。波动率类别始终占主导，但具体特征会变。**必须每年重训练**。

---

### 13_occam_razor.py / 奥卡姆剃刀（特征消融）

Top-K feature ablation study: trains models with K = 1, 3, 5, 7, 10, 20, 50, all features. Measures how much AUC is retained at each level.

> Top-K特征消融研究：用K = 1, 3, 5, 7, 10, 20, 50和全部特征训练模型。测量每个级别保留多少AUC。

**Key findings**:
- **K=1** (just `realized_vol_20d`): retains 86.5% of full-model AUC
- **K=5**: retains 91.5% — diminishing returns after this point
- **K=7**: retains 95.4% — good balance of simplicity and performance
- All top features are volatility clustering metrics (no event or interaction features in top-5)

> **关键发现**：
> - **K=1**（仅`realized_vol_20d`）：保留全模型AUC的86.5%
> - **K=5**：保留91.5%——此后收益递减
> - **K=7**：保留95.4%——简洁性和性能的良好平衡
> - 所有顶级特征均为波动率聚集指标（前5名中无事件或交互特征）

**Implication**: The model's alpha is overwhelmingly driven by one phenomenon — **volatility clusters**. The 100+ additional features provide marginal refinement (~8.5% of total AUC).

> **含义**：模型的alpha压倒性地来自一个现象——**波动率聚集**。100多个额外特征仅提供边际改进（总AUC的约8.5%）。

---

## Output / 输出

All scripts save results to:
> 所有脚本将结果保存至：

| Location | Contents |
|----------|----------|
| `output/charts/*.png` | Analysis charts (8 pattern charts + robustness charts) |
| `output/report.html` | Self-contained HTML report with embedded charts |
| Console stdout | Summary statistics, comparison tables, model metrics |

> | 位置 | 内容 |
> |------|------|
> | `output/charts/*.png` | 分析图表（8张模式图 + 鲁棒性图表） |
> | `output/report.html` | 自包含HTML报告（内嵌图表） |
> | 控制台标准输出 | 汇总统计、对比表、模型指标 |
