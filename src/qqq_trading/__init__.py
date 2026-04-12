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
    qqq_trading.data        — Data loading (1-min aggregation, VIX/rates, event calendar)
    qqq_trading.features    — Feature engineering (base, external, interactions, path)
    qqq_trading.models      — Model training, evaluation, and inference
    qqq_trading.utils       — Paths, plotting, time-series splits
    qqq_trading.cli         — Command-line tools (predict, pipeline)
"""
__version__ = "1.0.0"

from qqq_trading.config import Config, load_config
from qqq_trading.models.training import train_model, load_model, save_model
from qqq_trading.models.evaluation import evaluate_model, backtest_thresholds
from qqq_trading.models.prediction import predict, build_features_for_prediction as build_features
from qqq_trading.features.registry import get_full_features, get_base_features
