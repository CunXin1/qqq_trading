"""Full retraining pipeline: data -> features -> model -> save."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="QQQ model retraining pipeline")
    parser.add_argument(
        "--model-type", type=str, default="xgboost",
        choices=["xgboost", "lightgbm", "random_forest"],
        help="Model algorithm (default: xgboost)"
    )
    parser.add_argument(
        "--preset", type=str, default="production",
        choices=["base", "production"],
        help="Hyperparameter preset (default: production)"
    )
    parser.add_argument(
        "--train-end", type=str, default="2022-12-31",
        help="Training data cutoff date"
    )
    parser.add_argument(
        "--refresh-external", action="store_true",
        help="Force re-download of external data"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    import numpy as np
    import pandas as pd
    from qqq_trading.utils.paths import OUTPUT_DIR, MODEL_DIR, DATA_DIR
    from qqq_trading.config import load_config
    from qqq_trading.data.daily_metrics import load_1min_data, build_daily_metrics
    from qqq_trading.data.external_data import download_external_data
    from qqq_trading.features.base import engineer_base_features
    from qqq_trading.features.external import engineer_all_external
    from qqq_trading.features.interactions import build_interaction_features
    from qqq_trading.features.registry import get_full_features
    from qqq_trading.models.training import train_model, save_model, compute_pos_weight
    from qqq_trading.models.evaluation import evaluate_model, backtest_thresholds

    config = load_config()
    model_config = getattr(config.model, args.preset)

    # Step 1: Daily metrics
    print("Step 1: Building daily metrics...")
    raw_path = DATA_DIR / "QQQ_1min_adjusted.parquet"
    metrics_path = OUTPUT_DIR / "daily_metrics.parquet"

    if metrics_path.exists():
        print("  Loading cached daily metrics...")
        daily = pd.read_parquet(metrics_path)
        daily.index = pd.to_datetime(daily.index)
    else:
        print("  Computing from 1-min data...")
        raw = load_1min_data(raw_path)
        daily = build_daily_metrics(raw)
        daily.to_parquet(metrics_path)

    # Step 2: External data
    print("Step 2: External data...")
    ext_path = OUTPUT_DIR / "external_data.parquet"
    if args.refresh_external and ext_path.exists():
        ext_path.unlink()
    ext = download_external_data(ext_path)

    # Step 3: Feature engineering
    print("Step 3: Feature engineering...")
    df = engineer_base_features(daily)
    df = engineer_all_external(df, ext)
    df = build_interaction_features(df)

    # Step 4: Train
    print("Step 4: Training model...")
    target_col = "target_next_day_2pct"
    df[target_col] = df["abs_close_to_close_gt_2pct"].shift(-1).astype(float)

    feature_cols = get_full_features(include_interactions=True)
    available = [f for f in feature_cols if f in df.columns]

    valid = df[available + [target_col]].dropna()
    train = valid.loc[:args.train_end]
    test = valid.loc[pd.Timestamp(args.train_end) + pd.Timedelta(days=1):]

    X_train = train[available].values
    y_train = train[target_col].values.astype(int)
    X_test = test[available].values
    y_test = test[target_col].values.astype(int)

    print(f"  Train: {len(train)} samples, Test: {len(test)} samples")
    print(f"  Features: {len(available)}, Preset: {args.preset}")

    model = train_model(X_train, y_train, args.model_type, model_config, config.random_state)

    # Step 5: Evaluate
    print("Step 5: Evaluating...")
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, y_proba)
    bt = backtest_thresholds(y_test, y_proba)

    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  AP:  {metrics['ap']:.4f}")
    print(f"\n  Backtest:")
    print(bt.to_string(index=False))

    # Step 6: Save
    print("\nStep 6: Saving model...")
    save_model(model, available, MODEL_DIR / "interaction_model.joblib")
    print(f"  Saved to {MODEL_DIR / 'interaction_model.joblib'}")

    # Save features parquet
    df.to_parquet(OUTPUT_DIR / "interaction_features.parquet")
    print("Done.")


if __name__ == "__main__":
    main()
