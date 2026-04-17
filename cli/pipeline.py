"""Full retraining pipeline: data -> features -> model -> save."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# target → (source column in daily_metrics, shift, display name)
# shift=0: predict TODAY's outcome (0DTE). shift=-1: predict NEXT day's outcome (1DTE).
TARGETS = {
    "range_0dte": {"col": "intraday_range",     "shift": 0,  "desc": "0DTE intraday range (H-L)/O"},
    "otc_0dte":   {"col": "abs_open_to_close",  "shift": 0,  "desc": "0DTE |O2C|"},
    "c2c_1dte":   {"col": "abs_close_to_close", "shift": -1, "desc": "1DTE next-day |C2C|"},
    "otc_1dte":   {"col": "abs_open_to_close",  "shift": -1, "desc": "1DTE next-day |O2C|"},
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="QQQ model retraining pipeline")
    parser.add_argument(
        "--target", type=str, default="c2c_1dte",
        choices=list(TARGETS.keys()),
        help="Prediction target (default: c2c_1dte — matches legacy interaction_model)"
    )
    parser.add_argument(
        "--thresh", type=float, default=0.02,
        help="Move threshold as decimal (default: 0.02 = 2%%)"
    )
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
    parser.add_argument(
        "--output-name", type=str, default=None,
        help="Override output model filename stem (default: <target>_<thresh>pct)"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    import numpy as np
    import pandas as pd
    from utils.paths import OUTPUT_DIR, MODEL_DIR, DATA_DIR
    from config import load_config
    from data.daily_metrics import load_1min_data, build_daily_metrics
    from data.external_data import download_external_data
    from features.base import engineer_base_features
    from features.external import engineer_all_external
    from features.interactions import build_interaction_features
    from features.registry import get_full_features
    from models.training import train_model, save_model, compute_pos_weight
    from models.evaluation import evaluate_model, backtest_thresholds

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
    tc = TARGETS[args.target]
    thresh_pct = int(round(args.thresh * 100))
    target_col = f"target_{args.target}_{thresh_pct}pct"
    # Build binary target: value > thresh, with appropriate shift (-1 for 1DTE, 0 for 0DTE)
    df[target_col] = (df[tc["col"]] > args.thresh).shift(tc["shift"]).astype(float)
    print(f"  Target: {args.target} ({tc['desc']}) > {args.thresh:.0%}, shift={tc['shift']}")

    feature_cols = get_full_features(include_interactions=True)
    available = [f for f in feature_cols if f in df.columns]

    valid = df[available + [target_col]].dropna()
    train = valid.loc[:args.train_end]
    test = valid.loc[pd.Timestamp(args.train_end) + pd.Timedelta(days=1):]

    X_train = train[available].values
    y_train = train[target_col].values.astype(int)
    X_test = test[available].values
    y_test = test[target_col].values.astype(int)

    print(f"  Train: {len(train)} samples (base rate: {y_train.mean():.2%})")
    print(f"  Test:  {len(test)} samples (base rate: {y_test.mean():.2%})")
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
    stem = args.output_name or f"{args.target}_{thresh_pct}pct_model"
    model_path = MODEL_DIR / f"{stem}.joblib"
    save_model(model, available, model_path)
    print(f"  Saved to {model_path}")

    # Save features parquet (shared across targets — same feature set)
    df.to_parquet(OUTPUT_DIR / "interaction_features.parquet")
    print("Done.")


if __name__ == "__main__":
    main()
