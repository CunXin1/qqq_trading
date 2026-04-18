"""Model evaluation metrics and backtest utilities.
模型评估指标和回测工具。

Provides three levels of evaluation / 提供三个层级的评估:

1. evaluate_model():          Summary metrics (AUC, AP, Brier, base rate).
                              汇总指标（AUC、AP、Brier、基准率）。
2. backtest_thresholds():     Hit rate/coverage/lift at multiple thresholds.
                              多阈值下的命中率/覆盖率/提升倍数。
3. find_optimal_threshold():  F1-maximizing threshold search.
                              最大化 F1 的阈值搜索。

Key metrics explained / 关键指标解释:
  AUC (ROC-AUC):  Probability that model ranks a random positive above a random negative.
                  模型将随机正样本排在随机负样本之上的概率。
                  0.5 = random, >0.7 = useful, >0.8 = good.
  AP (Average Precision): Area under Precision-Recall curve. Better than AUC
                          for imbalanced data (low base rates like 5-14%).
                          精确率-召回率曲线下面积。对不平衡数据更好。
  Brier score:    Mean squared error of probability estimates (lower = better).
                  概率估计的均方误差（越低越好）。
  Lift:           hit_rate / base_rate. How many times better than random.
                  命中率 / 基准率。比随机好多少倍。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
)


def evaluate_model(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """Compute standard classification metrics for predicted probabilities.
    计算预测概率的标准分类指标。

    Returns dict with: auc, ap, brier, base_rate, n_samples, n_positive.
    返回字典包含：AUC、AP、Brier、基准率、样本数、正样本数。
    """
    return {
        "auc": roc_auc_score(y_true, y_proba),
        "ap": average_precision_score(y_true, y_proba),
        "brier": brier_score_loss(y_true, y_proba),
        "base_rate": y_true.mean(),
        "n_samples": len(y_true),
        "n_positive": int(y_true.sum()),
    }


def backtest_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Compute hit rate, coverage, lift for each confidence threshold.
    计算每个置信度阈值下的命中率、覆盖率和提升倍数。

    This is the primary tool for choosing trading thresholds.
    这是选择交易阈值的主要工具。
    For straddles: optimize for high precision (hit_rate) at acceptable coverage.
    对于跨式策略：在可接受覆盖率下优化高精确率（命中率）。

    Output columns / 输出列:
      threshold: Model probability cutoff. / 模型概率截断值。
      alerts:    Number of days above threshold (= trading opportunities).
                 超过阈值的天数（= 交易机会数）。
      hits:      Alerts that were actually big moves. / 实际为大波动的信号。
      hit_rate:  hits / alerts = precision. / 命中率 = 精确率。
      coverage:  hits / total_positives = recall. / 覆盖率 = 召回率。
      lift:      hit_rate / base_rate (times better than random).
                 命中率 / 基准率（比随机好多少倍）。
    """
    if thresholds is None:
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    rows = []
    base_rate = y_true.mean()
    total_pos = y_true.sum()

    for t in thresholds:
        mask = y_proba >= t
        n_alerts = mask.sum()
        if n_alerts > 0:
            hits = y_true[mask].sum()
            hit_rate = y_true[mask].mean()
            coverage = hits / total_pos if total_pos > 0 else 0
            lift = hit_rate / base_rate if base_rate > 0 else 0
        else:
            hits = 0
            hit_rate = 0
            coverage = 0
            lift = 0

        rows.append({
            "threshold": t,
            "alerts": int(n_alerts),
            "hits": int(hits),
            "hit_rate": hit_rate,
            "coverage": coverage,
            "lift": lift,
        })

    return pd.DataFrame(rows)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
) -> float:
    """Find threshold that maximizes F1 on validation set.
    找到在验证集上最大化 F1 的阈值。

    Scans thresholds from 0.1 to 0.9 in steps of 0.01.
    F1 = 2 * precision * recall / (precision + recall).
    Note: for trading, you may prefer optimizing precision directly
    (use backtest_thresholds instead).
    在 0.1 到 0.9 之间以 0.01 步长扫描阈值。
    注意：对于交易，你可能更倾向于直接优化精确率
    （改用 backtest_thresholds）。
    """
    best_thresh = 0.5
    best_score = 0

    for t in np.arange(0.1, 0.9, 0.01):
        pred = (y_proba >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        if f1 > best_score:
            best_score = f1
            best_thresh = t

    return best_thresh


def print_classification_report(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> str:
    """Print sklearn classification report at given threshold.
    打印给定阈值下的 sklearn 分类报告。

    Shows precision/recall/F1 for both classes: "No Large Move" and "Large Move".
    展示两个类别的精确率/召回率/F1："无大波动"和"大波动"。
    """
    y_pred = (y_proba >= threshold).astype(int)
    return classification_report(
        y_true, y_pred,
        target_names=["No Large Move", "Large Move"],
    )
