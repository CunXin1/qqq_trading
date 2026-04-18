"""
Evaluate model performance across all targets and thresholds.
跨所有目标和阈值评估模型性能。

This script produces a single summary table showing AUC, AP, and
hit rates at multiple confidence thresholds for every target.
本脚本生成一张汇总表，展示每个目标在多个置信度阈值下的AUC、AP和命中率。

It answers the question: "How good is this model, and at which
threshold should I trigger trades?"
它回答的问题是："这个模型有多好，应该在哪个阈值触发交易？"

Targets evaluated / 评估的目标:
    - 1DTE |C2C|>1%  : Tomorrow's absolute close-to-close return > 1%
                        明日收盘绝对收益 > 1%
    - 1DTE |C2C|>2%  : Tomorrow's absolute close-to-close return > 2%
                        明日收盘绝对收益 > 2%
    - 1DTE |C2C|>3%  : Tomorrow's absolute close-to-close return > 3%
                        明日收盘绝对收益 > 3%
    - 0DTE Range>2%  : Today's intraday range > 2%
                        今日日内范围 > 2%
    - 0DTE Range>3%  : Today's intraday range > 3%
                        今日日内范围 > 3%

Usage / 用法:
    python eval/model_eval.py                     # evaluate interaction model / 评估交互模型
    python eval/model_eval.py --model base        # evaluate base model / 评估基础模型
    python eval/model_eval.py --test-start 2024-01-01  # different test period / 不同测试期

Arguments / 参数:
    --model       "interaction" (122 features) or "base" (53 features)
                  "interaction"（122特征）或 "base"（53特征）
    --test-start  Start of test period (default: 2023-01-01)
                  测试期起始日期（默认：2023-01-01）

Output / 输出:
    1. Summary table: Target × Threshold matrix showing alerts/hit_rate
       汇总表：目标×阈值矩阵，显示信号数/命中率
    2. Detailed backtest for primary targets (1DTE |C2C|>2% and 0DTE Range>2%)
       主要目标的详细回测
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import pandas as pd
import numpy as np
from models.training import load_model
from features.base import engineer_base_features
from features.external import engineer_all_external
from features.interactions import build_interaction_features
from features.registry import get_base_features
from models.evaluation import evaluate_model, backtest_thresholds
from utils.paths import OUTPUT_DIR, MODEL_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Model evaluation across targets")
    parser.add_argument("--model", type=str, default="interaction",
                        choices=["interaction", "base"], help="Model to evaluate")
    parser.add_argument("--test-start", type=str, default="2023-01-01")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load data / 加载数据 ──
    print("Loading data...")
    daily = pd.read_parquet(OUTPUT_DIR / "daily_metrics.parquet")
    daily.index = pd.to_datetime(daily.index)

    # ── Build features based on model type / 根据模型类型构建特征 ──
    # interaction model needs external + interaction features
    # base model only needs price/volume features
    # 交互模型需要外部+交互特征；基础模型只需价格/成交量特征
    print("Building features...")
    df = engineer_base_features(daily)

    if args.model == "interaction":
        ext = pd.read_parquet(OUTPUT_DIR / "external_data.parquet")
        ext.index = pd.to_datetime(ext.index)
        df = engineer_all_external(df, ext)
        df = build_interaction_features(df)
        model_path = MODEL_DIR / "interaction_2000_2019.joblib"
    else:
        model_path = MODEL_DIR / "next_day_2pct_2000_2019.joblib"

    # ── Load model / 加载模型 ──
    print(f"Loading model: {model_path.name}")
    model, feature_cols = load_model(model_path)

    if args.model == "base" and feature_cols is None:
        feature_cols = get_base_features(include_premarket=False)

    available = [f for f in feature_cols if f in df.columns]
    print(f"  Type: {type(model).__name__}, Features: {len(available)}")

    # ── Define all evaluation targets / 定义所有评估目标 ──
    # 1DTE targets use shift(-1): predict today, actual outcome is tomorrow
    # 0DTE targets use same-day data: predict and verify on same day
    # 1DTE目标使用shift(-1)：今天预测，明天验证
    # 0DTE目标使用当天数据：当天预测当天验证
    targets = {
        "1DTE |C2C|>1%": df["abs_close_to_close_gt_1pct"].shift(-1).astype(float),
        "1DTE |C2C|>2%": df["abs_close_to_close_gt_2pct"].shift(-1).astype(float),
        "1DTE |C2C|>3%": df["abs_close_to_close_gt_3pct"].shift(-1).astype(float),
        "0DTE Range>2%": (df["intraday_range"] > 0.02).astype(float),
        "0DTE Range>3%": (df["intraday_range"] > 0.03).astype(float),
    }

    test = df.loc[args.test_start:]
    print(f"  Test: {args.test_start} to {test.index[-1].date()} ({len(test)} days)")

    # ── Generate predictions / 生成预测 ──
    # Handle NaN for RandomForest (RF can't handle NaN; XGBoost/LightGBM can)
    # 为RandomForest处理NaN（RF不能处理NaN；XGBoost/LightGBM可以）
    X_test = test[available].values
    if "RandomForest" in type(model).__name__:
        X_test = np.nan_to_num(X_test, nan=-999)

    y_proba = model.predict_proba(X_test)[:, 1]

    # ── Print summary table / 打印汇总表 ──
    # Format: "alerts/hit_rate" at each threshold
    # 格式：每个阈值下的"信号数/命中率"
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

        # evaluate_model returns: auc, ap, brier, base_rate, n_pos, n_total
        # evaluate_model返回：AUC、AP、Brier评分、基准率、正样本数、总样本数
        metrics = evaluate_model(y_valid, proba_valid)

        # backtest_thresholds returns: threshold, alerts, hits, hit_rate, coverage, lift
        # backtest_thresholds返回：阈值、信号数、命中数、命中率、覆盖率、提升倍数
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
    print("格式：每个置信度阈值下的 信号数/命中率")

    # ── Detailed backtest for primary targets / 主要目标的详细回测 ──
    # These are the two most important targets for actual trading
    # 这两个是实际交易中最重要的目标
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
