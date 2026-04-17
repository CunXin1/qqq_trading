"""
qqq_trading — QQQ intraday volatility prediction for 0DTE/1DTE straddle trading.

Quick start:
    from qqq_trading import predict, build_features, train_model, load_config

    # Load config
    config = load_config()

    # Build features from daily metrics + external data
    df = build_features(daily_metrics, external_data)

    # Train a model
    model = train_model(X_train, y_train, "xgboost", config.model.production)

    # Or load a saved model and predict
    result = predict(model_path, daily_metrics, external_data)

Subpackages:
    data        — Data loading (1-min aggregation, VIX/rates, event calendar)
    features    — Feature engineering (base, external, interactions, path)
    models      — Model training, evaluation, and inference
    utils       — Paths, plotting, time-series splits
    cli         — Command-line tools (predict, pipeline)
"""
__version__ = "1.0.0"

from config import Config, load_config
from models.training import train_model, load_model, save_model
from models.evaluation import evaluate_model, backtest_thresholds
from models.prediction import predict, build_features_for_prediction as build_features
from features.registry import get_full_features, get_base_features
