# Models / 模型

## Architecture / 架构

- **Algorithm**: XGBoost (primary), LightGBM, RandomForest
- **Task**: Binary classification — P(large move tomorrow)
- **Features**: 122 (53 base + 26 external + 43 interaction)
- **Training period**: 2000–2022 (~5,700 days)
- **Test period**: 2023.01–2026.02 (~780 days, fully held out)

> - **算法**：XGBoost（主力）、LightGBM、RandomForest
> - **任务**：二分类——预测明日发生大幅波动的概率
> - **特征数**：122个（53基础 + 26外部 + 43交互）
> - **训练期**：2000–2022（约5,700个交易日）
> - **测试期**：2023.01–2026.02（约780天，完全未参与训练）

---

## Hyperparameters / 超参数

Two presets defined in `config/default.yaml`. See that file for detailed Chinese/English comments on each parameter.

> 两组预设定义在 `config/default.yaml` 中，每个参数的详细中英文注释请参见该文件。

| Parameter | Base (research) | Production (live) | Why Different |
|-----------|----------------|-------------------|--------------|
| n_estimators | 300 | 500 | More trees + lower LR = smoother ensemble |
| max_depth | 5 | 5 | Same — no depth increase to avoid overfitting |
| learning_rate | 0.05 | 0.03 | Smaller steps for finer fitting |
| subsample | 0.8 | 0.8 | Same — 80% row sampling |
| colsample_bytree | 0.8 | 0.7 | More aggressive in prod — higher diversity |
| reg_alpha (L1) | 0.0 | 0.1 | Light L1 in prod auto-prunes noisy features |
| reg_lambda (L2) | 1.0 | 1.0 | Same — standard leaf weight smoothing |

> | 参数 | Base（研究用） | Production（生产用） | 差异原因 |
> |------|--------------|-------------------|---------|
> | n_estimators | 300 | 500 | 更多树+更低学习率=更平滑的集成 |
> | max_depth | 5 | 5 | 相同——不增加深度以避免过拟合 |
> | learning_rate | 0.05 | 0.03 | 更小步长，更精细拟合 |
> | subsample | 0.8 | 0.8 | 相同——80%行采样 |
> | colsample_bytree | 0.8 | 0.7 | 生产中更激进——提高多样性 |
> | reg_alpha (L1) | 0.0 | 0.1 | 生产中轻微L1自动剪枝噪声特征 |
> | reg_lambda (L2) | 1.0 | 1.0 | 相同——标准叶子权重平滑 |

---

## Training / 训练

### Via Python API / 通过Python接口

```python
from qqq_trading import train_model, load_config

config = load_config()
model = train_model(X_train, y_train, "xgboost", config.model.production)
```

### Via CLI / 通过命令行

```bash
# Production training with default cutoff
python -m qqq_trading.cli.pipeline --preset production --train-end 2022-12-31

# Research mode with LightGBM
python -m qqq_trading.cli.pipeline --preset base --model-type lightgbm

# Force refresh external data before training
python -m qqq_trading.cli.pipeline --preset production --refresh-external
```

### Pipeline Steps / 训练管道步骤

1. **Load raw data**: Read 1-min parquet → aggregate to daily metrics
2. **Fetch external data**: VIX/VVIX/yields (cached, skipped if fresh)
3. **Feature engineering**: Base → external → interaction features
4. **Create target**: `target_next_day_2pct = (next day |C2C| > 2%)`
5. **Split data**: Train (up to `--train-end`), test (after `--train-end`)
6. **Compute class weight**: `pos_weight = n_negative / n_positive` for imbalanced target
7. **Train model**: XGBoost/LightGBM/RF with selected preset
8. **Evaluate**: AUC, AP, Brier, backtest at multiple thresholds
9. **Save**: Model (joblib) + feature list (CSV) to `output/model/`

> 1. **加载原始数据**：读取1分钟parquet → 聚合为日线指标
> 2. **获取外部数据**：VIX/VVIX/收益率（有缓存则跳过）
> 3. **特征工程**：基础 → 外部 → 交互特征
> 4. **创建目标**：`target_next_day_2pct = (次日|收盘收益|> 2%)`
> 5. **划分数据**：训练集（截止`--train-end`），测试集（之后）
> 6. **计算类权重**：`pos_weight = 负样本数/正样本数`，处理类别不平衡
> 7. **训练模型**：使用选定预设的XGBoost/LightGBM/RF
> 8. **评估**：AUC、AP、Brier评分、多阈值回测
> 9. **保存**：模型(joblib) + 特征列表(CSV)到`output/model/`

---

## Saved Models / 保存的模型

| Model | File | Features | Use Case |
|-------|------|----------|----------|
| **Interaction** | `interaction_model.joblib` | 121 | Production: best AUC, includes VRP × catalyst crosses |
| **Base** | `next_day_2pct_model.joblib` | 53 | Baseline: no external data dependency, simpler but weaker |

> | 模型 | 文件 | 特征数 | 用途 |
> |------|------|--------|------|
> | **交互模型** | `interaction_model.joblib` | 121 | 生产用：最佳AUC，包含VRP×催化剂交叉特征 |
> | **基础模型** | `next_day_2pct_model.joblib` | 53 | 基线：不依赖外部数据，更简单但更弱 |

Each model is saved alongside a CSV file listing its exact feature columns (e.g., `interaction_model_features.csv`). This ensures prediction uses the same features as training.

> 每个模型旁边都保存了一个CSV文件，列出其精确的特征列（如`interaction_model_features.csv`）。这确保预测时使用与训练时相同的特征。

---

## Evaluation / 评估

```python
from qqq_trading import evaluate_model, backtest_thresholds

metrics = evaluate_model(y_test, y_proba)
# -> {'auc': 0.826, 'ap': 0.52, 'brier': 0.08, 'base_rate': 0.161, ...}

bt = backtest_thresholds(y_test, y_proba, [0.3, 0.5, 0.7])
# -> DataFrame: threshold | alerts | hits | hit_rate | coverage | lift
```

### Metrics Explained / 指标详解

| Metric | What It Measures | Good Value |
|--------|-----------------|-----------|
| **AUC** (Area Under ROC) | Overall ranking quality — can the model separate large-move days from normal days? | > 0.75 |
| **AP** (Average Precision) | Precision-recall summary — useful for imbalanced classes | > 0.40 |
| **Brier Score** | Calibration — how close are predicted probabilities to actual frequencies? | < 0.10 |
| **Hit Rate @threshold** | Precision at a specific decision threshold | > 60% at 0.7 |
| **Coverage @threshold** | Recall — what fraction of actual large moves did we catch? | Trade-off with hit rate |
| **Lift** | Hit rate / base rate — how much better than random? | > 2.0x |

> | 指标 | 衡量什么 | 好的数值 |
> |------|---------|---------|
> | **AUC**（ROC曲线下面积） | 整体排序质量——模型能否区分大波动日和正常日？ | > 0.75 |
> | **AP**（平均精确度） | 精确率-召回率综合——适用于不平衡分类 | > 0.40 |
> | **Brier评分** | 校准度——预测概率与实际频率有多接近？ | < 0.10 |
> | **命中率 @阈值** | 特定决策阈值下的精确率 | 0.7阈值时>60% |
> | **覆盖率 @阈值** | 召回率——抓住了多少实际大波动？ | 与命中率权衡 |
> | **提升倍数** | 命中率/基准率——比随机猜好多少？ | > 2.0倍 |

---

## Performance (Test 2023.1 – 2026.2) / 测试期表现

### Best Models / 最佳模型

| Scenario | Target | AUC | @0.7 Hit Rate | Annual Signals |
|----------|--------|-----|---------------|----------------|
| **0DTE Range>2%** | Intraday range >2% | **0.826** | **66%** | ~59 |
| **0DTE Range>3%** | Intraday range >3% | **0.864** | **57%** | ~14 |
| 1DTE \|C2C\|>2% | Tomorrow \|return\| >2% | 0.743 | 62% | ~13 |

> | 场景 | 目标 | AUC | 0.7阈值命中率 | 年信号数 |
> |------|------|-----|-------------|---------|
> | **0DTE 日内范围>2%** | 日内范围超过2% | **0.826** | **66%** | ~59 |
> | **0DTE 日内范围>3%** | 日内范围超过3% | **0.864** | **57%** | ~14 |
> | 1DTE \|C2C\|>2% | 明日绝对收益>2% | 0.743 | 62% | ~13 |

### Expected Live Performance / 预期实盘表现

Apply 5–8% AUC discount and 10–15% hit rate discount for live trading due to execution slippage, data latency, and regime drift.

> 实盘交易因执行滑点、数据延迟和市场状态漂移，需对 AUC 折扣 5-8%，对命中率折扣 10-15%。

| Model | Reported AUC | Expected Live AUC | Reported Hit Rate @0.7 | Expected Live |
|-------|-------------|-------------------|----------------------|--------------|
| 0DTE Range>2% | 0.824 | 0.77–0.80 | 66% | 56–59% |
| 0DTE Range>3% | 0.848 | 0.80–0.83 | 57% | 48–51% |
| 1DTE \|C2C\|>2% | 0.740 | 0.69–0.72 | 62% | 53–56% |

> | 模型 | 回测AUC | 预期实盘AUC | 回测命中率@0.7 | 预期实盘 |
> |------|--------|-----------|-------------|---------|
> | 0DTE 范围>2% | 0.824 | 0.77–0.80 | 66% | 56–59% |
> | 0DTE 范围>3% | 0.848 | 0.80–0.83 | 57% | 48–51% |
> | 1DTE \|C2C\|>2% | 0.740 | 0.69–0.72 | 62% | 53–56% |

### Models to Avoid / 应避免的模型

- **0DTE |O2C|>2%**: Walk-forward gap 8.6% — overfitting confirmed. The model memorizes training patterns that don't generalize.
- **>5% targets**: Only 3 events in the 780-day test period — statistically meaningless. Cannot validate any model with N=3.

> - **0DTE |O2C|>2%**：Walk-forward差距8.6%——确认过拟合。模型记住了不能泛化的训练模式。
> - **>5%目标**：780天测试期内仅3个事件——统计上无意义。N=3无法验证任何模型。

---

## Model Evolution / 模型演化历程

The model went through 5 phases of development. Each phase was driven by a specific hypothesis and validated with held-out data:

> 模型经历了5个开发阶段。每个阶段由特定假设驱动，并用留出数据验证：

| Phase | Change | AUC | Delta | Lesson |
|-------|--------|-----|-------|--------|
| 1. Base | 53 price/volume features | 0.730 | — | Volatility clustering is the dominant signal |
| 2. +Raw VIX/rates | Added VIX, yields directly | 0.718 | -0.012 | Collinearity hurts — raw VIX ≈ realized vol |
| 3. +VRP | Replaced raw VIX with VRP | 0.747 | +0.029 | Fear premium is the real signal |
| 4. +Interactions | VRP × FOMC/NFP crosses | 0.752 | +0.005 | Regime × catalyst captures tail events |
| 5. +Path | Smoothness features | 0.756 | +0.004 | Marginal — better as rule overlay |

> | 阶段 | 变更 | AUC | 变化 | 教训 |
> |------|------|-----|------|------|
> | 1. 基础 | 53个价格/成交量特征 | 0.730 | — | 波动率聚集是主导信号 |
> | 2. +原始VIX | 直接加入VIX和收益率 | 0.718 | -0.012 | 共线性有害——原始VIX ≈ 已实现波动率 |
> | 3. +VRP | 用VRP替代原始VIX | 0.747 | +0.029 | 恐惧溢价才是真正信号 |
> | 4. +交互 | VRP × FOMC/NFP交叉特征 | 0.752 | +0.005 | 状态×催化剂捕获尾部事件 |
> | 5. +路径 | 平滑度特征 | 0.756 | +0.004 | 边际改善——更适合作为规则叠加 |

**Core insight**: Volatility clustering is the dominant alpha. The top-5 features capture 91% of total AUC, and they are all volatility rolling statistics.

> **核心洞察**：波动率聚集是主要的alpha来源。前5个特征捕获了91%的总AUC，且全部为波动率滚动统计量。

---

## Robustness / 鲁棒性

### Monotonicity — ALL PASS / 单调性——全部通过

As the model's confidence threshold increases from 0.3 to 0.95, the hit rate should increase monotonically. This confirms the model's probability outputs are well-calibrated.

> 当模型置信度阈值从0.3上升到0.95时，命中率应单调递增。这确认了模型概率输出的良好校准。

0DTE Range>2%: 38% → 92% hit rate with **zero violations** across all thresholds.

> 0DTE范围>2%：38% → 92%命中率，所有阈值间**零违反**。

All 6 targets (1%/2%/3% × 0DTE/1DTE) pass monotonicity. This is the strongest robustness signal.

> 全部6个目标（1%/2%/3% × 0DTE/1DTE）通过单调性检验。这是最强的鲁棒性信号。

### Walk-Forward CV / 滚动前向交叉验证

5-year rolling training window, 5-day purge gap (prevents information leakage across train/test boundary), yearly test blocks from 2010–2025:

> 5年滚动训练窗口，5天清洗间隔（防止训练/测试边界的信息泄漏），2010–2025年度测试块：

| Model | Static AUC | WF Mean AUC | Gap | Verdict |
|-------|-----------|-------------|-----|---------|
| 1DTE \|C2C\|>2% | 0.740 | 0.693 | +4.7% | **CONSISTENT** — gap < 5% |
| 0DTE Range>2% | 0.824 | 0.773 | +5.0% | **BORDERLINE** — right at 5% threshold |
| 0DTE Range>3% | 0.848 | 0.805 | +4.3% | **CONSISTENT** — gap < 5% |

> | 模型 | 静态AUC | WF均值AUC | 差距 | 结论 |
> |------|--------|----------|------|------|
> | 1DTE \|C2C\|>2% | 0.740 | 0.693 | +4.7% | **一致**——差距<5% |
> | 0DTE 范围>2% | 0.824 | 0.773 | +5.0% | **边界**——恰好在5%阈值 |
> | 0DTE 范围>3% | 0.848 | 0.805 | +4.3% | **一致**——差距<5% |

**Failure years**: 2013 (AUC 0.544), 2017 (AUC 0.409) — ultra-low volatility years where the base rate dropped below 3%. The model has almost no signal in dead-calm markets.

> **失败年份**：2013（AUC 0.544）、2017（AUC 0.409）——极低波动率年份，基准率降至3%以下。模型在极度平静的市场中几乎没有信号。

### Feature Stability — UNSTABLE / 特征稳定性——不稳定

Spearman rank correlation between feature importance rankings across market eras: **0.36–0.47**.

> 不同市场时代间特征重要性排名的Spearman秩相关：**0.36–0.47**。

The core category (volatility rolling stats) always dominates, but the specific windows and secondary features shift significantly. This means:

> 核心类别（波动率滚动统计）始终占主导，但具体窗口和次要特征显著变化。这意味着：

- **Must retrain at least annually** to adapt to regime shifts
- Feature selection from one era may not transfer to the next
- The interaction features (VRP × events) are the most unstable

> - **必须至少每年重训练一次**以适应市场状态变化
> - 某一时代的特征选择可能无法迁移到下一时代
> - 交互特征（VRP×事件）最不稳定

### Occam's Razor / 奥卡姆剃刀

Top-K feature ablation study: how many features do you actually need?

> Top-K特征消融研究：实际需要多少个特征？

| K | AUC | Retention | What You Get |
|---|-----|-----------|-------------|
| 1 | 0.713 | 86.5% | `realized_vol_20d` alone |
| 3 | 0.753 | 91.5% | + `mean_abs_ret_10d`, `std_ret_20d` |
| 5 | 0.746 | 91.5% | + `realized_vol_10d`, `mean_abs_ret_20d` |
| 7 | 0.786 | 95.4% | + secondary vol features |
| Full (105) | 0.825 | 100% | All features including interactions |

> | K | AUC | 保留率 | 包含内容 |
> |---|-----|--------|---------|
> | 1 | 0.713 | 86.5% | 仅`realized_vol_20d` |
> | 3 | 0.753 | 91.5% | + `mean_abs_ret_10d`, `std_ret_20d` |
> | 5 | 0.746 | 91.5% | + `realized_vol_10d`, `mean_abs_ret_20d` |
> | 7 | 0.786 | 95.4% | + 次要波动率特征 |
> | 全部(105) | 0.825 | 100% | 所有特征（含交互） |

**Top-5 consensus features** (all volatility clustering metrics):
1. `realized_vol_20d` — 20-day annualized realized volatility
2. `mean_abs_ret_10d` — 10-day average absolute return
3. `std_ret_20d` — 20-day return standard deviation
4. `realized_vol_10d` — 10-day annualized realized volatility
5. `mean_abs_ret_20d` — 20-day average absolute return

> **Top-5共识特征**（全部是波动率聚集指标）：
> 1. `realized_vol_20d` — 20日年化已实现波动率
> 2. `mean_abs_ret_10d` — 10日平均绝对收益率
> 3. `std_ret_20d` — 20日收益率标准差
> 4. `realized_vol_10d` — 10日年化已实现波动率
> 5. `mean_abs_ret_20d` — 20日平均绝对收益率

**Takeaway**: The model's alpha is overwhelmingly driven by one phenomenon — **volatility clusters**. Yesterday's vol predicts today's vol. Everything else is marginal refinement.

> **结论**：模型的alpha压倒性地来自一个现象——**波动率聚集**。昨天的波动率预测今天的波动率。其他一切都是边际改进。
