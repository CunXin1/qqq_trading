# Algorithms / 算法详解

This project supports three ML algorithms for binary classification. This document explains how each works, their strengths/weaknesses, and why XGBoost is the primary choice.

> 本项目支持三种ML算法进行二分类。本文档详细解释每种算法的工作原理、优缺点，以及为什么XGBoost是首选。

---

## Table of Contents / 目录

1. [Decision Trees (Foundation)](#decision-trees--决策树基础)
2. [Random Forest](#random-forest--随机森林)
3. [XGBoost](#xgboost--极端梯度提升)
4. [LightGBM](#lightgbm--轻量梯度提升机)
5. [Comparison](#comparison--三者对比)
6. [Why XGBoost for This Project](#why-xgboost-for-this-project--为什么本项目选择-xgboost)

---

## Decision Trees / 决策树（基础）

All three algorithms are built on **decision trees**, so understanding trees first is essential.

> 三种算法都建立在**决策树**之上，因此首先理解决策树至关重要。

### How It Works / 工作原理

A decision tree splits data into groups by asking yes/no questions about features, one at a time:

> 决策树通过逐个询问特征的是/否问题，将数据分成不同组：

```
                  realized_vol_20d > 0.25?
                  /                      \
               Yes                        No
              /                            \
     vix_spike = True?              days_to_fomc <= 2?
      /          \                    /            \
   Yes            No               Yes              No
  P=0.72        P=0.31           P=0.28           P=0.08
  (HIGH)      (MODERATE)       (MODERATE)         (LOW)
```

Each leaf node outputs a probability. The tree learns which splits best separate "large move days" from "normal days" by maximizing **information gain** (how much purer each child group becomes).

> 每个叶子节点输出一个概率。树通过最大化**信息增益**（每个子组变得多纯净）来学习哪些分割最能区分"大波动日"和"正常日"。

### Limitations / 局限性

- A single tree is **unstable** — small changes in data produce very different splits
- Prone to **overfitting** — a deep tree memorizes training data perfectly but fails on new data
- Limited **expressiveness** — one tree can only capture simple patterns

> - 单棵树**不稳定**——数据的微小变化会产生截然不同的分割
> - 容易**过拟合**——深树完美记忆训练数据但在新数据上失败
> - **表达力有限**——一棵树只能捕获简单模式

**Solution**: Combine many trees together. The three algorithms below differ in *how* they combine trees.

> **解决方案**：将多棵树组合在一起。以下三种算法的区别在于*如何*组合树。

---

## Random Forest / 随机森林

### Core Idea / 核心思想

Train many independent trees in parallel, each on a random subset of data and features. Final prediction = average of all trees.

> 并行训练多棵独立的树，每棵使用数据和特征的随机子集。最终预测 = 所有树的平均值。

```
Training Data
    |
    |── Random Sample 1 + Random Features → Tree 1 → P₁ = 0.65
    |── Random Sample 2 + Random Features → Tree 2 → P₂ = 0.72
    |── Random Sample 3 + Random Features → Tree 3 → P₃ = 0.58
    |── ...
    |── Random Sample N + Random Features → Tree N → Pₙ = 0.61
    |
    └── Final Prediction = mean(P₁, P₂, ..., Pₙ) = 0.64
```

### How It Works Step by Step / 逐步工作流程

1. **Bootstrap sampling (Bagging)**: For each tree, randomly draw N samples **with replacement** from the training set. ~63% of data is selected, ~37% is left out (called "out-of-bag" / OOB samples).

   > **自助采样（Bagging）**：对每棵树，从训练集中**有放回地**随机抽取N个样本。约63%被选中，约37%被遗漏（称为"袋外"样本）。

2. **Feature randomization**: At each split, only consider a random subset of features (typically √p or p/3 where p = total features). This forces trees to be diverse.

   > **特征随机化**：在每次分裂时，仅考虑特征的随机子集（通常√p或p/3，p=总特征数）。这迫使树具有多样性。

3. **Independent training**: Each tree is grown to full depth (or a max depth limit). Trees do NOT communicate with each other.

   > **独立训练**：每棵树生长到完全深度（或最大深度限制）。树之间**不**相互通信。

4. **Aggregation**: For classification, take the average probability across all trees. This averaging reduces variance (noise) dramatically.

   > **聚合**：对于分类任务，取所有树的概率平均值。这种平均化大幅降低方差（噪声）。

### Why Averaging Works / 为什么平均有效

Imagine 100 trees, each with 70% accuracy and independent errors. By averaging, errors cancel out. The ensemble accuracy is much higher than any single tree.

> 想象100棵树，每棵70%准确率且误差独立。通过平均，误差相互抵消。集成准确率远高于任何单棵树。

**Analogy**: Asking 100 people to guess the weight of an ox. Individual guesses vary wildly, but the average is remarkably close to the true weight (Galton's "Wisdom of Crowds").

> **类比**：让100人猜一头牛的重量。个人猜测差异很大，但平均值非常接近真实重量（高尔顿的"群体智慧"）。

### Key Parameters in This Project / 本项目中的关键参数

| Parameter | Role | Typical Value |
|-----------|------|---------------|
| `n_estimators` | Number of trees | 300–500 |
| `max_depth` | Maximum depth per tree | 5 (shallow to prevent overfitting) |
| `max_features` | Features considered per split | sqrt(122) ≈ 11 |

> | 参数 | 作用 | 典型值 |
> |------|------|--------|
> | `n_estimators` | 树的数量 | 300–500 |
> | `max_depth` | 每棵树的最大深度 | 5（浅层防止过拟合） |
> | `max_features` | 每次分裂考虑的特征数 | sqrt(122) ≈ 11 |

### Strengths / 优点

- **Hard to overfit**: Averaging + randomization makes it very robust
- **No tuning needed**: Works well out of the box with default parameters
- **Handles missing data**: Can work with NaN values (sklearn RF)
- **Feature importance**: Built-in importance ranking via impurity or permutation
- **Parallelizable**: Trees are independent → easy to distribute across CPU cores

> - **难以过拟合**：平均化+随机化使其非常鲁棒
> - **无需调参**：默认参数即可良好工作
> - **处理缺失数据**：可处理NaN值（sklearn RF）
> - **特征重要性**：内置基于不纯度或置换的重要性排名
> - **可并行化**：树之间独立→易于分布到多个CPU核心

### Weaknesses / 缺点

- **Cannot correct mistakes**: Since trees are independent, later trees cannot fix earlier trees' errors
- **Lower ceiling**: Typically 2–5% lower AUC than boosting methods on tabular data
- **Larger models**: 500 full-depth trees use more memory than 500 shallow boosted trees
- **Weak on rare events**: With bootstrapping, rare events (>2% moves at 16% base rate) may be undersampled in some trees

> - **无法纠错**：由于树相互独立，后面的树无法修正前面树的错误
> - **上限较低**：在表格数据上，AUC通常比提升方法低2-5%
> - **模型更大**：500棵完全深度的树比500棵浅层提升树占用更多内存
> - **对稀有事件弱**：通过自助采样，稀有事件（>2%波动在16%基准率下）可能在某些树中被欠采样

---

## XGBoost / 极端梯度提升

**XGBoost** = e**X**treme **G**radient **Boost**ing. The most widely used algorithm for tabular data competitions and production ML systems.

> **XGBoost** = 极端梯度提升。表格数据竞赛和生产ML系统中最广泛使用的算法。

### Core Idea / 核心思想

Train trees **sequentially**, where each new tree specifically learns to correct the mistakes of all previous trees. This is called **boosting**.

> **顺序**训练树，每棵新树专门学习纠正所有前面树的错误。这称为**提升（Boosting）**。

```
Tree 1: Predict → Errors (residuals)
    ↓
Tree 2: Learn from Tree 1's errors → Smaller errors
    ↓
Tree 3: Learn from remaining errors → Even smaller errors
    ↓
...
    ↓
Tree 500: Final refinement → Very small remaining errors

Final Prediction = Tree₁ × η + Tree₂ × η + Tree₃ × η + ... + Tree₅₀₀ × η
                   (η = learning_rate = 0.03)
```

### How It Works Step by Step / 逐步工作流程

1. **Initialize**: Start with a constant prediction (e.g., the base rate: P = 0.161 for >2% moves).

   > **初始化**：从常数预测开始（如基准率：>2%波动的P = 0.161）。

2. **Compute residuals**: For each sample, calculate the **gradient** — how wrong the current prediction is and in which direction it should be corrected. For binary classification, this uses the log-loss gradient.

   > **计算残差**：对每个样本，计算**梯度**——当前预测有多错以及应该向哪个方向修正。对于二分类，使用对数损失的梯度。

3. **Fit a new tree to the residuals**: The new tree doesn't predict the target directly — it predicts the *correction needed* to improve the current ensemble.

   > **用新树拟合残差**：新树不直接预测目标——它预测改善当前集成所需的*修正量*。

4. **Add the tree with shrinkage**: `new_prediction = old_prediction + learning_rate × new_tree`. The learning rate (η = 0.03 in production) controls how much each tree contributes. Smaller η = more conservative, needs more trees, but generalizes better.

   > **带收缩率加入新树**：`新预测 = 旧预测 + 学习率 × 新树`。学习率（生产中η = 0.03）控制每棵树的贡献。更小的η = 更保守，需要更多树，但泛化更好。

5. **Regularization**: XGBoost adds L1 (`reg_alpha`) and L2 (`reg_lambda`) penalties to leaf weights, plus tree-level complexity penalties. This prevents individual trees from being too extreme.

   > **正则化**：XGBoost对叶子权重添加L1（`reg_alpha`）和L2（`reg_lambda`）惩罚，以及树级复杂度惩罚。这防止单棵树过于极端。

6. **Repeat** steps 2–5 for `n_estimators` iterations.

   > **重复**步骤2–5，共`n_estimators`次迭代。

### The "Gradient" in Gradient Boosting / "梯度"的含义

The name comes from using **gradient descent** in function space:
- In regular gradient descent, you adjust model *parameters* to minimize loss
- In gradient boosting, you adjust the model *output* by adding new functions (trees)
- Each tree is fitted to the negative gradient of the loss function — it points in the direction of steepest improvement

> 名称来自在函数空间中使用**梯度下降**：
> - 在常规梯度下降中，调整模型*参数*来最小化损失
> - 在梯度提升中，通过添加新函数（树）来调整模型*输出*
> - 每棵树拟合损失函数的负梯度——指向最陡峭的改进方向

XGBoost is "extreme" because it also uses the **second-order gradient** (Hessian), which tells it not just *which direction* to go but *how far* — like having both velocity and acceleration information.

> XGBoost之所以"极端"，是因为它还使用**二阶梯度**（Hessian矩阵），不仅告诉模型*往哪个方向*走，还告诉*走多远*——就像同时拥有速度和加速度信息。

### Key Parameters in This Project / 本项目中的关键参数

| Parameter | Role | Base | Production | Why |
|-----------|------|------|-----------|-----|
| `n_estimators` | Number of boosting rounds | 300 | 500 | More rounds + lower LR = finer fit |
| `max_depth` | Max depth per tree | 5 | 5 | Shallow trees = weak learners (intentional) |
| `learning_rate` | Shrinkage per tree (η) | 0.05 | 0.03 | Slower learning = better generalization |
| `subsample` | Row sampling per tree | 0.8 | 0.8 | Stochastic gradient boosting — reduces overfitting |
| `colsample_bytree` | Feature sampling per tree | 0.8 | 0.7 | Forces diversity, like Random Forest |
| `reg_alpha` | L1 penalty on leaf weights | 0.0 | 0.1 | Sparsifies weak features in production |
| `reg_lambda` | L2 penalty on leaf weights | 1.0 | 1.0 | Smooths leaf weights |
| `scale_pos_weight` | Class imbalance adjustment | auto | auto | = n_neg / n_pos ≈ 5.2 for 16% base rate |

> | 参数 | 作用 | 基础 | 生产 | 原因 |
> |------|------|------|------|------|
> | `n_estimators` | 提升轮数 | 300 | 500 | 更多轮次+更低LR=更精细拟合 |
> | `max_depth` | 每棵树最大深度 | 5 | 5 | 浅树=弱学习器（有意为之） |
> | `learning_rate` | 每棵树的收缩率(η) | 0.05 | 0.03 | 更慢学习=更好泛化 |
> | `subsample` | 每棵树的行采样比 | 0.8 | 0.8 | 随机梯度提升——减少过拟合 |
> | `colsample_bytree` | 每棵树的特征采样比 | 0.8 | 0.7 | 强制多样性，类似随机森林 |
> | `reg_alpha` | 叶子权重L1惩罚 | 0.0 | 0.1 | 生产中稀疏化弱特征 |
> | `reg_lambda` | 叶子权重L2惩罚 | 1.0 | 1.0 | 平滑叶子权重 |
> | `scale_pos_weight` | 类别不平衡调整 | 自动 | 自动 | = 负样本数/正样本数 ≈ 5.2（16%基准率） |

### Strengths / 优点

- **Highest accuracy on tabular data**: Consistently wins Kaggle competitions and benchmarks for structured data
- **Error correction**: Each tree fixes previous mistakes → more expressive than RF
- **Built-in regularization**: L1/L2 penalties, max_depth, min_child_weight — multiple overfitting controls
- **Handles imbalanced classes**: `scale_pos_weight` directly adjusts for rare events
- **Feature importance**: Gain-based, cover-based, and SHAP importance
- **Missing value handling**: Learns optimal direction for missing values at each split
- **Fast**: Optimized C++ core with histogram-based splitting

> - **表格数据最高准确率**：在Kaggle竞赛和结构化数据基准测试中持续胜出
> - **纠错能力**：每棵树修正前面的错误→比RF更具表达力
> - **内置正则化**：L1/L2惩罚、最大深度、最小子节点权重——多重过拟合控制
> - **处理不平衡类别**：`scale_pos_weight`直接调整稀有事件权重
> - **特征重要性**：基于增益、覆盖和SHAP的重要性
> - **缺失值处理**：在每次分裂时学习缺失值的最优方向
> - **快速**：优化的C++核心，基于直方图的分裂

### Weaknesses / 缺点

- **Sequential training**: Trees must be trained one after another (can't parallelize across trees like RF)
- **More hyperparameters**: Requires careful tuning of learning_rate, n_estimators, regularization
- **Overfitting risk**: If learning_rate is too high or n_estimators too large, it memorizes training noise
- **Sensitive to outliers**: Gradient-based learning amplifies the influence of extreme samples

> - **顺序训练**：树必须逐棵训练（不能像RF那样跨树并行化）
> - **更多超参数**：需要仔细调节学习率、轮数、正则化
> - **过拟合风险**：如果学习率过高或轮数过多，会记忆训练噪声
> - **对异常值敏感**：基于梯度的学习放大极端样本的影响

---

## LightGBM / 轻量梯度提升机

**LightGBM** = **Light** **G**radient **B**oosting **M**achine. Developed by Microsoft as a faster, more memory-efficient alternative to XGBoost.

> **LightGBM** = 轻量梯度提升机。由微软开发，作为XGBoost的更快、更省内存的替代方案。

### Core Idea / 核心思想

Same boosting framework as XGBoost, but with two key innovations that make it dramatically faster:

> 与XGBoost相同的提升框架，但有两个关键创新使其速度大幅提升：

### Innovation 1: Leaf-Wise Growth / 创新1：按叶子生长

**XGBoost** grows trees **level-wise** — all nodes at the same depth are split before moving deeper:

> **XGBoost**按**层**生长树——同一深度的所有节点先分裂，再向更深处推进：

```
Level-wise (XGBoost):          Leaf-wise (LightGBM):
       [Root]                        [Root]
      /      \                      /      \
    [A]      [B]                  [A]      [B]
   /   \    /   \                /   \
  [C] [D] [E] [F]             [C]  [D]
                               |
                              [G]  ← grows the leaf with highest loss reduction
```

LightGBM grows the **leaf with the largest loss reduction** first, regardless of depth. This means it can find complex patterns faster with fewer leaves, but is more prone to overfitting on small datasets.

> LightGBM优先生长**损失减少最大的叶子**，不管深度。这意味着它可以用更少的叶子更快地发现复杂模式，但在小数据集上更容易过拟合。

### Innovation 2: Histogram-Based Splitting / 创新2：基于直方图的分裂

Instead of evaluating every possible split point (like XGBoost's exact method), LightGBM **buckets** continuous features into 255 discrete bins first. This reduces the number of split candidates from millions to 255 per feature.

> LightGBM不像XGBoost那样评估每个可能的分裂点，而是先将连续特征**分桶**为255个离散区间。这将每个特征的候选分裂点从数百万减少到255个。

```
Feature: realized_vol_20d
Values:  0.05, 0.08, 0.12, 0.15, 0.18, 0.22, 0.25, 0.31, ...

XGBoost (exact): evaluate split at 0.05, 0.08, 0.12, ... (every unique value)
LightGBM (histogram): bucket into 255 bins → evaluate 255 splits only
```

**Result**: 5–10x faster training with minimal accuracy loss (< 0.1% AUC typically).

> **结果**：训练速度提升5-10倍，准确率损失极小（通常< 0.1% AUC）。

### Additional LightGBM Optimizations / 额外优化

| Technique | What It Does |
|-----------|-------------|
| **GOSS** (Gradient-based One-Side Sampling) | Keeps all high-gradient samples (hard to predict), downsamples low-gradient ones (easy). Focuses training on difficult examples. |
| **EFB** (Exclusive Feature Bundling) | Bundles mutually exclusive features (never both non-zero) into single features. Reduces effective feature count. |
| **Categorical feature support** | Natively handles categorical features without one-hot encoding. Finds optimal split by grouping categories. |

> | 技术 | 作用 |
> |------|------|
> | **GOSS**（基于梯度的单侧采样） | 保留所有高梯度样本（难预测的），下采样低梯度样本（容易的）。集中训练困难样例。 |
> | **EFB**（互斥特征捆绑） | 将互斥特征（不会同时非零）捆绑为单个特征。减少有效特征数。 |
> | **类别特征原生支持** | 原生处理类别特征，无需独热编码。通过分组类别找最优分裂。 |

### Key Parameters in This Project / 本项目中的关键参数

LightGBM uses different parameter names but equivalent concepts:

> LightGBM使用不同的参数名但等价的概念：

| LightGBM | XGBoost Equivalent | Role |
|----------|-------------------|------|
| `num_leaves` | ~2^max_depth | Controls tree complexity (default 31 ≈ depth 5) |
| `n_estimators` | `n_estimators` | Number of boosting rounds |
| `learning_rate` | `learning_rate` | Shrinkage per iteration |
| `subsample` | `subsample` | Row sampling ratio |
| `colsample_bytree` | `colsample_bytree` | Feature sampling ratio |
| `reg_alpha` | `reg_alpha` | L1 regularization |
| `reg_lambda` | `reg_lambda` | L2 regularization |
| `min_child_samples` | `min_child_weight` | Minimum samples in a leaf |

> | LightGBM | XGBoost等价 | 作用 |
> |----------|-----------|------|
> | `num_leaves` | ~2^max_depth | 控制树复杂度（默认31 ≈ 深度5） |
> | `n_estimators` | `n_estimators` | 提升轮数 |
> | `learning_rate` | `learning_rate` | 每次迭代的收缩率 |
> | `subsample` | `subsample` | 行采样比 |
> | `colsample_bytree` | `colsample_bytree` | 特征采样比 |
> | `reg_alpha` | `reg_alpha` | L1正则化 |
> | `reg_lambda` | `reg_lambda` | L2正则化 |
> | `min_child_samples` | `min_child_weight` | 叶子最少样本数 |

### Strengths / 优点

- **Fastest training**: 5–10x faster than XGBoost on large datasets
- **Lower memory**: Histogram-based approach uses far less RAM
- **Competitive accuracy**: Typically within 0.5% AUC of XGBoost
- **Native categorical support**: No preprocessing needed for categorical features
- **Scalable**: Handles datasets with millions of rows efficiently

> - **最快训练速度**：在大数据集上比XGBoost快5-10倍
> - **更低内存**：基于直方图的方法使用远少的RAM
> - **有竞争力的准确率**：通常与XGBoost AUC差距在0.5%以内
> - **原生类别支持**：类别特征无需预处理
> - **可扩展**：高效处理数百万行的数据集

### Weaknesses / 缺点

- **Overfitting on small data**: Leaf-wise growth can create very deep trees on small datasets (like ours with ~5700 training days)
- **Sensitive to `num_leaves`**: Too many leaves → overfitting; must tune carefully
- **Less mature**: Fewer community resources and debugging tools than XGBoost
- **Determinism issues**: Results can vary across platforms due to floating-point ordering

> - **小数据过拟合**：按叶子生长在小数据集上（如我们的约5700训练日）可能创建很深的树
> - **对`num_leaves`敏感**：叶子太多→过拟合；必须仔细调节
> - **成熟度较低**：社区资源和调试工具少于XGBoost
> - **确定性问题**：由于浮点运算顺序，结果可能因平台而异

---

## Comparison / 三者对比

### Architecture Comparison / 架构对比

| Aspect | Random Forest | XGBoost | LightGBM |
|--------|--------------|---------|----------|
| **Training strategy** | Parallel (independent trees) | Sequential (each tree corrects errors) | Sequential (same as XGBoost) |
| **Tree growth** | Level-wise to max depth | Level-wise to max depth | **Leaf-wise** (best-first) |
| **Split finding** | Exact (every value) | Exact or histogram | **Histogram only** (255 bins) |
| **Error correction** | No — trees are independent | Yes — gradient of loss function | Yes — same as XGBoost |
| **Regularization** | Via averaging + randomization | L1/L2 + tree complexity penalty | L1/L2 + tree complexity + GOSS |
| **Missing values** | Imputation needed (sklearn) | Learns optimal branch | Learns optimal branch |

> | 方面 | 随机森林 | XGBoost | LightGBM |
> |------|---------|---------|----------|
> | **训练策略** | 并行（独立树） | 顺序（每棵树纠正错误） | 顺序（同XGBoost） |
> | **树的生长** | 按层到最大深度 | 按层到最大深度 | **按叶子**（最优优先） |
> | **分裂查找** | 精确（每个值） | 精确或直方图 | **仅直方图**（255桶） |
> | **纠错** | 无——树相互独立 | 有——损失函数的梯度 | 有——同XGBoost |
> | **正则化** | 通过平均+随机化 | L1/L2+树复杂度惩罚 | L1/L2+树复杂度+GOSS |
> | **缺失值** | 需要插补（sklearn） | 学习最优分支 | 学习最优分支 |

### Performance Comparison / 性能对比

| Metric | Random Forest | XGBoost | LightGBM |
|--------|--------------|---------|----------|
| **Accuracy (typical)** | Good | **Best** | Very good (≈ XGBoost) |
| **Training speed** | Fast (parallel) | Medium | **Fastest** |
| **Memory usage** | High | Medium | **Lowest** |
| **Overfitting risk** | Low | Medium | Higher (leaf-wise) |
| **Tuning effort** | Low (works OOB) | Medium | Higher (num_leaves sensitive) |
| **Small data (<10K rows)** | **Best** (robust) | Good | Risky (can overfit) |
| **Large data (>1M rows)** | Slow | Good | **Best** |

> | 指标 | 随机森林 | XGBoost | LightGBM |
> |------|---------|---------|----------|
> | **准确率（典型）** | 良好 | **最佳** | 很好（≈ XGBoost） |
> | **训练速度** | 快（并行） | 中等 | **最快** |
> | **内存使用** | 高 | 中等 | **最低** |
> | **过拟合风险** | 低 | 中等 | 较高（按叶子生长） |
> | **调参工作量** | 低（开箱即用） | 中等 | 较高（num_leaves敏感） |
> | **小数据(<10K行)** | **最佳**（鲁棒） | 良好 | 有风险（可能过拟合） |
> | **大数据(>1M行)** | 慢 | 良好 | **最佳** |

### Visual Analogy / 形象类比

| Algorithm | Analogy |
|-----------|---------|
| **Random Forest** | 100 students independently take the same exam, then average their answers. Diverse perspectives cancel out individual mistakes. |
| **XGBoost** | One student takes the exam, reviews mistakes, retakes focusing on missed questions, reviews again, retakes again... 500 iterations of targeted improvement. |
| **LightGBM** | Same as XGBoost, but uses speed-reading techniques and skips easy questions to focus on the hardest ones first. |

> | 算法 | 类比 |
> |------|------|
> | **随机森林** | 100个学生独立参加同一场考试，然后平均他们的答案。多样化视角抵消了个人错误。 |
> | **XGBoost** | 一个学生参加考试，复查错误，针对错题重考，再复查，再重考…500轮定向改进。 |
> | **LightGBM** | 与XGBoost相同，但使用速读技巧并跳过简单题目，优先攻克最难的题。 |

---

## Why XGBoost for This Project / 为什么本项目选择 XGBoost

### 1. Dataset Size / 数据集规模

Our training set has ~5,700 rows — this is **small** by ML standards. XGBoost's level-wise growth and built-in regularization make it the safest choice for small tabular datasets. LightGBM's leaf-wise approach risks overfitting here.

> 我们的训练集约5,700行——按ML标准是**小的**。XGBoost的按层生长和内置正则化使其成为小型表格数据集的最安全选择。LightGBM的按叶子方法在此有过拟合风险。

### 2. Imbalanced Target / 不平衡目标

Only 16.1% of days have >2% moves. XGBoost's `scale_pos_weight` and native log-loss optimization handle this well. RF's bootstrap sampling can dilute rare events even further.

> 仅16.1%的天数有>2%波动。XGBoost的`scale_pos_weight`和原生对数损失优化能良好处理这一情况。RF的自助采样可能进一步稀释稀有事件。

### 3. Feature Interactions / 特征交互

With 122 features including 43 interaction terms, XGBoost's sequential error correction can discover complex conditional patterns (e.g., "high VRP + FOMC + pre-market spike") that RF's independent trees might miss.

> 有122个特征（包括43个交互项），XGBoost的顺序纠错能发现复杂的条件模式（如"高VRP + FOMC + 盘前突增"），RF的独立树可能错过这些。

### 4. Empirical Results / 实证结果

In this project's backtesting:

> 在本项目的回测中：

| Algorithm | AUC (0DTE Range>2%) | Notes |
|-----------|--------------------|----|
| **XGBoost** | **0.826** | Best overall |
| LightGBM | 0.819 | Close, but slightly less stable in walk-forward |
| Random Forest | 0.798 | Decent baseline, lower ceiling |

> | 算法 | AUC（0DTE Range>2%） | 备注 |
> |------|--------------------|----|
> | **XGBoost** | **0.826** | 总体最佳 |
> | LightGBM | 0.819 | 接近，但Walk-forward中稍不稳定 |
> | 随机森林 | 0.798 | 不错的基线，上限较低 |

### 5. Ecosystem / 生态系统

XGBoost has the most mature ecosystem: SHAP integration, extensive documentation, GPU support, and the largest community. This matters for debugging and maintaining production systems.

> XGBoost拥有最成熟的生态系统：SHAP集成、丰富文档、GPU支持和最大社区。这对调试和维护生产系统很重要。

### When to Use Each / 何时使用各算法

| Situation | Recommended |
|-----------|------------|
| Production deployment (this project) | **XGBoost** |
| Quick baseline / sanity check | **Random Forest** |
| Very large dataset (>100K rows) or speed-critical | **LightGBM** |
| Ensemble / model diversity | Train all three, average predictions |

> | 场景 | 推荐 |
> |------|------|
> | 生产部署（本项目） | **XGBoost** |
> | 快速基线/完整性检查 | **随机森林** |
> | 超大数据集(>100K行)或速度敏感 | **LightGBM** |
> | 集成/模型多样性 | 三种都训练，平均预测 |
