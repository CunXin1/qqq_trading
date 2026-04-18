"""Model training, evaluation, and inference.
模型训练、评估与推理。

This package provides the ML model lifecycle:
本包提供 ML 模型的全生命周期：

1. training.py   — Create, train, save, load XGBoost/LightGBM/RandomForest classifiers.
                   创建、训练、保存、加载 XGBoost/LightGBM/RandomForest 分类器。
2. evaluation.py — Compute AUC/AP/Brier, threshold backtesting, optimal threshold search.
                   计算 AUC/AP/Brier、阈值回测、最优阈值搜索。
3. prediction.py — End-to-end inference: load model → build features → predict → classify signal.
                   端到端推理：加载模型 → 构建特征 → 预测 → 分类信号。

All three production models (range_0dte, otc_0dte, c2c_1dte) use XGBoost with
the "production" preset (500 trees, lr=0.03, colsample=0.7, L1=0.1).
三个生产模型均使用 XGBoost 的"production"预设（500 棵树、lr=0.03、colsample=0.7、L1=0.1）。
"""
from models.training import train_model, create_model, load_model, save_model
from models.evaluation import evaluate_model, backtest_thresholds
from models.prediction import predict, PredictionResult
