"""Unified model training with configurable hyperparameters.
统一模型训练，支持可配置超参数。

Supports three model types / 支持三种模型类型:
  - XGBoost:       Production default. Handles NaN natively, best AUC.
                   生产环境默认。原生处理 NaN，AUC 最优。
  - LightGBM:      Faster training, slightly lower AUC. Good for research.
                   训练更快，AUC 略低。适合研究。
  - RandomForest:  Baseline comparison. Requires NaN imputation.
                   基线对比。需要 NaN 填充。

Model persistence / 模型持久化:
  Each model is saved as two files:
  每个模型保存为两个文件：
    - {name}.joblib:  The trained sklearn/xgb estimator (serialized with joblib).
                      训练好的 sklearn/xgb 估计器（用 joblib 序列化）。
    - {name}.csv:     Feature column names used during training (ensures consistency).
                      训练时使用的特征列名（确保一致性）。

Hyperparameter presets (from config/default.yaml) / 超参数预设：
  base:       n_estimators=300, lr=0.05, colsample=0.8, L1=0.0 (research).
              研究用。
  production: n_estimators=500, lr=0.03, colsample=0.7, L1=0.1 (live trading).
              实盘用。更多正则化以提高鲁棒性。
"""
from __future__ import annotations

from typing import Literal, Optional
import numpy as np
import joblib
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

from config import ModelConfig


def create_model(
    model_type: Literal["xgboost", "lightgbm", "random_forest"] = "xgboost",
    config: Optional[ModelConfig] = None,
    pos_weight: float = 1.0,
    random_state: int = 42,
):
    """Create a classifier with given config and class imbalance weight.
    使用给定配置和类别不平衡权重创建分类器。

    Args:
        model_type:   Algorithm to use. / 使用的算法。
        config:       Hyperparameter preset (base or production). / 超参数预设。
        pos_weight:   Weight for positive class (auto-computed from class ratio).
                      正类权重（从类别比例自动计算）。
        random_state: Random seed for reproducibility. / 随机种子。

    Returns:
        Unfitted sklearn-compatible classifier.
        未拟合的 sklearn 兼容分类器。
    """
    if config is None:
        config = ModelConfig()

    if model_type == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            scale_pos_weight=pos_weight,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            reg_alpha=config.reg_alpha,
            reg_lambda=config.reg_lambda,
            random_state=random_state,
            verbosity=0,
            n_jobs=-1,
        )
    elif model_type == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            is_unbalance=True,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            reg_alpha=config.reg_alpha,
            reg_lambda=config.reg_lambda,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
        )
    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=min(config.max_depth + 3, 10),
            min_samples_leaf=15,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def compute_pos_weight(y: np.ndarray) -> float:
    """Compute positive class weight for imbalanced data.
    计算不平衡数据的正类权重。

    For a 10% base rate: weight = 0.9 / 0.1 = 9.0.
    This tells the model that missing a positive sample costs 9x more.
    对于 10% 的基准率：权重 = 0.9 / 0.1 = 9.0。
    告诉模型漏掉正样本的代价是 9 倍。
    """
    mean = y.mean()
    return max(1.0, (1 - mean) / mean) if mean > 0 else 1.0


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: Literal["xgboost", "lightgbm", "random_forest"] = "xgboost",
    config: Optional[ModelConfig] = None,
    random_state: int = 42,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
):
    """Train a model and return the fitted estimator.
    训练模型并返回已拟合的估计器。

    If X_val/y_val are provided, uses early stopping (XGBoost/LightGBM only):
    stops training when validation AUC doesn't improve for 50 rounds.
    若提供 X_val/y_val，则使用早停（仅 XGBoost/LightGBM）：
    当验证集 AUC 连续 50 轮未提升时停止训练。

    Automatically computes positive class weight from y_train distribution
    and handles NaN (RandomForest gets NaN→-999 imputation).
    自动从 y_train 分布计算正类权重，并处理 NaN
    （RandomForest 将 NaN 替换为 -999）。
    """
    pos_weight = compute_pos_weight(y_train)
    model = create_model(model_type, config, pos_weight, random_state)

    if model_type == "random_forest":
        X_fit = np.nan_to_num(X_train, nan=-999)
        model.fit(X_fit, y_train)
    elif X_val is not None and y_val is not None:
        # Early stopping with validation set
        fit_params = {
            "eval_set": [(X_val, y_val)],
        }
        if model_type == "xgboost":
            fit_params["verbose"] = False
            model.set_params(early_stopping_rounds=50, eval_metric="logloss")
        elif model_type == "lightgbm":
            fit_params["callbacks"] = [lgb.early_stopping(50, verbose=False),
                                       lgb.log_evaluation(period=0)]
            fit_params["eval_metric"] = "binary_logloss"
        model.fit(X_train, y_train, **fit_params)
    else:
        model.fit(X_train, y_train)

    return model


def save_model(model, feature_cols: list[str], path: Path) -> None:
    """Save model (.joblib) and feature list (.csv) as sidecar files.
    将模型 (.joblib) 和特征列表 (.csv) 保存为伴随文件。

    The CSV sidecar ensures that at inference time, features are provided
    in the exact same order as during training.
    CSV 伴随文件确保推理时特征顺序与训练时完全一致。
    """
    joblib.dump(model, path)
    feat_path = path.with_suffix(".csv")
    pd.Series(feature_cols).to_csv(feat_path, index=False, header=["feature"])


def load_model(path: Path):
    """Load model and feature list from disk.
    从磁盘加载模型和特征列表。

    Handles both old-format CSV (header='0') and new-format (header='feature').
    Also checks multiple naming conventions for backward compatibility.
    兼容旧格式 CSV（header='0'）和新格式（header='feature'）。
    同时检查多种命名约定以保持向后兼容性。

    Returns:
        (model, feature_cols): Fitted estimator and list of feature names.
                               If no CSV found, feature_cols is None.
        （模型, 特征列表）：已拟合的估计器和特征名称列表。
                           若未找到 CSV，feature_cols 为 None。
    """
    model = joblib.load(path)

    # Try multiple CSV naming conventions
    candidates = [
        path.with_suffix(".csv"),                                        # new: same stem
        path.parent / f"{path.stem}_feature_columns.csv",                # pattern: X_feature_columns.csv
        path.parent / "interaction_feature_columns.csv",                 # legacy name
        path.parent / "feature_columns.csv",                             # legacy base
    ]

    feature_cols = None
    for feat_path in candidates:
        if feat_path.exists():
            df_feat = pd.read_csv(feat_path)
            col = df_feat.columns[0]  # use whatever the first column is named
            feature_cols = df_feat[col].tolist()
            break

    return model, feature_cols
