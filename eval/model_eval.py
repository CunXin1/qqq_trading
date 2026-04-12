"""
Evaluate model performance across all targets and thresholds.

Usage:
    python eval/model_eval.py                     # evaluate interaction model
    python eval/model_eval.py --model base        # evaluate base model
    python eval/model_eval.py --test-start 2024-01-01  # different test period
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import pandas as pd
import numpy as np
from qqq_trading.models.training import load_model
from qqq_trading.features.base import engineer_base_features
from qqq_trading.features.external import engineer_all_external
from qqq_trading.features.interactions import build_interaction_features
from qqq_trading.features.registry import get_base_features
from qqq_trading.models.evaluation import evaluate_model, backtest_thresholds
from qqq_trading.utils.paths import OUTPUT_DIR, MODEL_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Model evaluation across targets")
    parser.add_argument("--model", type=str, default="interaction",
                        choices=["interaction", "base"], help="Model to evaluate")
    parser.add_argument("--test-start", type=str, default="2023-01-01")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load data ──
    print("Loading data...")
    daily = pd.read_parquet(OUTPUT_DIR / "daily_metrics.parquet")
    daily.index = pd.to_datetime(daily.index)

    # ── Build features based on model type ──
    print("Building features...")
    df = engineer_base_features(daily)

    if args.model == "interaction":
        ext = pd.read_parquet(OUTPUT_DIR / "external_data.parquet")
        ext.index = pd.to_datetime(ext.index)
        df = engineer_all_external(df, ext)
        df = build_interaction_features(df)
        model_path = MODEL_DIR / "interaction_model.joblib"
    else:
        model_path = MODEL_DIR / "next_day_2pct_model.joblib"

    # ── Load model ──
    print(f"Loading model: {model_path.name}")
    model, feature_cols = load_model(model_path)

    if args.model == "base" and feature_cols is None:
        feature_cols = get_base_features(include_premarket=False)

    available = [f for f in feature_cols if f in df.columns]
    print(f"  Type: {type(model).__name__}, Features: {len(available)}")

    # ── Define targets ──
    targets = {
        "1DTE |C2C|>1%": df["abs_close_to_close_gt_1pct"].shift(-1).astype(float),
        "1DTE |C2C|>2%": df["abs_close_to_close_gt_2pct"].shift(-1).astype(float),
        "1DTE |C2C|>3%": df["abs_close_to_close_gt_3pct"].shift(-1).astype(float),
        "0DTE Range>2%": (df["intraday_range"] > 0.02).astype(float),
        "0DTE Range>3%": (df["intraday_range"] > 0.03).astype(float),
    }

    test = df.loc[args.test_start:]
    print(f"  Test: {args.test_start} to {test.index[-1].date()} ({len(test)} days)")

    # ── Handle NaN for RandomForest ──
    X_test = test[available].values
    if "RandomForest" in type(model).__name__:
        X_test = np.nan_to_num(X_test, nan=-999)

    y_proba = model.predict_proba(X_test)[:, 1]

    # ── Summary table ──
    print()
    print("=" * 105)
    print(f"MODEL EVALUATION: {args.model} ({type(model).__name__})")
    print("=" * 105)
    header = (
        f"{'Target':<20} {'Base%':>6} {'AUC':>7} {'AP':>7} "
        f"{'@0.3':>12} {'@0.4':>12} {'@0.5':>12} {'@0.6':>12} {'@0.7':>12}"
    )
    print(header)
    print("-" * 105)

    for target_name, target_series in targets.items():
        y_test = target_series.loc[test.index].values
        valid_mask = ~np.isnan(y_test)
        if valid_mask.sum() < 50:
            print(f"{target_name:<20} insufficient data")
            continue

        y_valid = y_test[valid_mask].astype(int)
        proba_valid = y_proba[valid_mask]

        metrics = evaluate_model(y_valid, proba_valid)
        bt = backtest_thresholds(y_valid, proba_valid, [0.3, 0.4, 0.5, 0.6, 0.7])

        def fmt(row):
            return f"{int(row['alerts']):>3}/{row['hit_rate']:.0%}"

        bt_strs = [fmt(bt.iloc[i]) for i in range(len(bt))]
        print(
            f"{target_name:<20} {metrics['base_rate']:>5.1%} {metrics['auc']:>7.3f} {metrics['ap']:>7.3f} "
            + " ".join(f"{s:>12}" for s in bt_strs)
        )

    print()
    print("Format: alerts/hit_rate at each confidence threshold")

    # ── Detailed backtest for primary targets ──
    for target_name, target_series in [
        ("1DTE |C2C|>2%", targets["1DTE |C2C|>2%"]),
        ("0DTE Range>2%", targets["0DTE Range>2%"]),
    ]:
        y_test = target_series.loc[test.index].values
        valid_mask = ~np.isnan(y_test)
        y_valid = y_test[valid_mask].astype(int)
        proba_valid = y_proba[valid_mask]

        bt = backtest_thresholds(y_valid, proba_valid, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        base_rate = y_valid.mean()

        print()
        print("=" * 80)
        print(f"DETAILED: {target_name}")
        print("=" * 80)
        print(bt.to_string(index=False))
        print(f"\nBase rate: {base_rate:.1%} ({y_valid.sum()}/{len(y_valid)} days)")


if __name__ == "__main__":
    main()
