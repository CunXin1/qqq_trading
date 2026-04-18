"""Inference pipeline: load model and predict.
推理管线：加载模型并预测。

This module provides the end-to-end inference path used by both
the live trading pipeline and the CLI predict tool.
本模块提供实盘交易管线和 CLI 预测工具使用的端到端推理路径。

Flow / 流程:
  1. Load trained model + feature list from .joblib + .csv.
     从 .joblib + .csv 加载训练好的模型和特征列表。
  2. Build full feature set from daily_metrics + external_data.
     从 daily_metrics + external_data 构建完整特征集。
  3. Extract latest row (most recent trading day).
     提取最新行（最近交易日）。
  4. Predict probability → classify into signal level.
     预测概率 → 分类为信号级别。

Signal levels / 信号级别:
  HIGH:     prob >= threshold + 0.2 (strong conviction, trade aggressively).
            高确信度，积极交易。
  ELEVATED: prob >= threshold (meets trading criteria).
            达到交易标准。
  MODERATE: prob >= threshold - 0.1 (borderline, monitor closely).
            边界状态，密切关注。
  LOW:      prob < threshold - 0.1 (no trade).
            不交易。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from models.training import load_model
from features.base import engineer_base_features
from features.external import engineer_all_external
from features.interactions import build_interaction_features
from features.registry import get_full_features, get_base_features


@dataclass
class PredictionResult:
    """Container for a single prediction output.
    单次预测输出的容器。"""
    date: str               # Prediction date / 预测日期
    target: str              # Target name (e.g. "0DTE Range>2%") / 目标名称
    probability: float       # Model probability [0, 1] / 模型概率
    signal: str              # Signal level: HIGH/ELEVATED/MODERATE/LOW / 信号级别
    threshold: float         # Decision threshold used / 使用的决策阈值
    model_name: str          # Model filename stem / 模型文件名主干


def _classify_signal(prob: float, threshold: float) -> str:
    """Classify probability into discrete signal level.
    将概率分类为离散信号级别。"""
    if prob >= threshold + 0.2:
        return "HIGH"
    elif prob >= threshold:
        return "ELEVATED"
    elif prob >= threshold - 0.1:
        return "MODERATE"
    return "LOW"


def build_features_for_prediction(
    daily_metrics: pd.DataFrame,
    external_data: pd.DataFrame,
    include_interactions: bool = True,
) -> pd.DataFrame:
    """Run the full feature pipeline for inference.
    运行完整的特征管线用于推理。

    Reproduces the exact same feature engineering as training:
    base (53) + external (26) + interactions (43) = 122 features.
    复制与训练完全相同的特征工程：
    基础 (53) + 外部 (26) + 交互 (43) = 122 个特征。
    """
    df = engineer_base_features(daily_metrics)
    df = engineer_all_external(df, external_data)
    if include_interactions:
        df = build_interaction_features(df)
    return df


def predict(
    model_path: Path,
    daily_metrics: pd.DataFrame,
    external_data: pd.DataFrame,
    threshold: float = 0.5,
    target_name: str = "1DTE |C2C|>2%",
) -> PredictionResult:
    """Load model and make prediction for the latest date.
    加载模型并对最新日期进行预测。

    Args:
        model_path:    Path to .joblib model file. / 模型文件路径。
        daily_metrics: Historical + live daily metrics DataFrame.
                       历史 + 实时日频指标 DataFrame。
        external_data: VIX/VVIX/yields DataFrame. / VIX/VVIX/收益率 DataFrame。
        threshold:     Signal decision threshold (default 0.5).
                       信号决策阈值（默认 0.5）。
        target_name:   Display name for the prediction target.
                       预测目标的显示名称。

    Returns:
        PredictionResult with probability, signal level, and metadata.
        包含概率、信号级别和元数据的 PredictionResult。
    """
    model, feature_cols = load_model(model_path)

    df = build_features_for_prediction(daily_metrics, external_data)

    # Use the latest row
    latest = df.iloc[[-1]]
    date_str = str(latest.index[0].date())

    # Filter to available features
    available = [f for f in feature_cols if f in latest.columns]
    X = latest[available].values

    proba = model.predict_proba(X)[:, 1][0]
    signal = _classify_signal(proba, threshold)

    return PredictionResult(
        date=date_str,
        target=target_name,
        probability=float(proba),
        signal=signal,
        threshold=threshold,
        model_name=model_path.stem,
    )
