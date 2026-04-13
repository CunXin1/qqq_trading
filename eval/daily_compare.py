"""
Compare model predictions vs actual daily moves for every trading day.

Shows probability, signal level, and actual market data side by side.
Highlights hits/misses and computes rolling accuracy.

Usage:
    python eval/daily_compare.py                              # all test days
    python eval/daily_compare.py --threshold 0.5              # only show signal days
    python eval/daily_compare.py --start 2025-03-01 --end 2025-04-30  # date range
    python eval/daily_compare.py --threshold 0.3 --csv        # export
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
from qqq_trading.utils.paths import OUTPUT_DIR, MODEL_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Daily prediction vs actual comparison")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Start date")
    parser.add_argument("--end", type=str, default=None, help="End date (default: latest)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Only show days with prob >= threshold (default: show all)")
    parser.add_argument("--target", type=str, default="range2",
                        choices=["range2", "range3", "c2c1", "c2c2", "c2c3"],
                        help="Target to evaluate (default: range2)")
    parser.add_argument("--csv", action="store_true", help="Export to eval/daily_compare.csv")
    return parser.parse_args()


TARGET_CONFIG = {
    "range2": {"name": "0DTE Range>2%", "col": "intraday_range", "thresh": 0.02, "shift": 0},
    "range3": {"name": "0DTE Range>3%", "col": "intraday_range", "thresh": 0.03, "shift": 0},
    "c2c1":   {"name": "1DTE |C2C|>1%", "col": "abs_close_to_close", "thresh": 0.01, "shift": -1},
    "c2c2":   {"name": "1DTE |C2C|>2%", "col": "abs_close_to_close", "thresh": 0.02, "shift": -1},
    "c2c3":   {"name": "1DTE |C2C|>3%", "col": "abs_close_to_close", "thresh": 0.03, "shift": -1},
}


def main():
    args = parse_args()
    tc = TARGET_CONFIG[args.target]

    # ── Load & build ──
    print("Loading data and building features...")
    daily = pd.read_parquet(OUTPUT_DIR / "daily_metrics.parquet")
    daily.index = pd.to_datetime(daily.index)
    ext = pd.read_parquet(OUTPUT_DIR / "external_data.parquet")
    ext.index = pd.to_datetime(ext.index)

    df = engineer_base_features(daily)
    df = engineer_all_external(df, ext)
    df = build_interaction_features(df)

    model, feature_cols = load_model(MODEL_DIR / "interaction_model.joblib")
    available = [f for f in feature_cols if f in df.columns]

    # ── Slice date range ──
    subset = df.loc[args.start:args.end] if args.end else df.loc[args.start:]
    X = subset[available].values
    y_proba = model.predict_proba(X)[:, 1]

    # ── Build comparison table ──
    comp = pd.DataFrame(index=subset.index)
    comp["prob"] = y_proba

    # Signal level
    comp["signal"] = ""
    comp.loc[comp["prob"] >= 0.3, "signal"] = "LOW"
    comp.loc[comp["prob"] >= 0.5, "signal"] = "ELEV"
    comp.loc[comp["prob"] >= 0.7, "signal"] = "HIGH"

    # Actual market data
    comp["open"] = subset["reg_open"]
    comp["high"] = subset["reg_high"]
    comp["low"] = subset["reg_low"]
    comp["close"] = subset["reg_close"]
    comp["range%"] = subset["intraday_range"] * 100
    comp["c2c%"] = subset["close_to_close_ret"] * 100
    comp["o2c%"] = subset["open_to_close_ret"] * 100
    comp["gap%"] = subset["gap_return"] * 100
    comp["maxdd%"] = subset["max_drawdown"] * 100
    comp["maxru%"] = subset["max_runup"] * 100

    if "realized_vol_20d" in subset.columns:
        comp["rv20%"] = subset["realized_vol_20d"] * 100

    # Actual target hit
    if tc["shift"] == 0:
        comp["actual"] = (subset[tc["col"]] > tc["thresh"]).astype(int)
    else:
        comp["actual"] = (subset[tc["col"]].shift(tc["shift"]) > tc["thresh"]).astype(int)

    # Prediction correct?
    def judge(row, threshold):
        predicted = row["prob"] >= threshold
        actual = row["actual"] == 1
        if predicted and actual:
            return "TP"  # true positive: signal + hit
        elif predicted and not actual:
            return "FP"  # false positive: signal but no hit
        elif not predicted and actual:
            return "FN"  # false negative: no signal but hit
        else:
            return "TN"  # true negative: no signal, no hit

    eval_thresh = args.threshold if args.threshold else 0.5
    comp["result"] = comp.apply(lambda r: judge(r, eval_thresh), axis=1)

    # ── Filter if threshold specified ──
    if args.threshold is not None:
        display = comp[comp["prob"] >= args.threshold].copy()
    else:
        display = comp.copy()

    # ── Print ──
    print()
    total = len(comp.dropna(subset=["actual"]))
    actual_pos = (comp["actual"] == 1).sum()
    signals = (comp["prob"] >= eval_thresh).sum()
    tp = (comp["result"] == "TP").sum()
    fp = (comp["result"] == "FP").sum()
    fn = (comp["result"] == "FN").sum()
    tn = (comp["result"] == "TN").sum()
    hit_rate = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print("=" * 130)
    print(f"DAILY COMPARISON: {tc['name']}  |  {comp.index[0].date()} to {comp.index[-1].date()}  |  "
          f"Eval threshold: {eval_thresh}")
    print("=" * 130)
    print(f"Days: {total}  |  Actual big moves: {actual_pos}  |  "
          f"Signals (>={eval_thresh}): {signals}  |  TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}")
    print(f"Precision (hit rate): {hit_rate:.1%}  |  Recall (coverage): {recall:.1%}  |  "
          f"Base rate: {actual_pos/total:.1%}")
    print("=" * 130)
    print()

    if args.threshold is not None:
        label = f"SIGNAL DAYS ONLY (prob >= {args.threshold})"
    else:
        label = "ALL TRADING DAYS"
    print(f"--- {label}: {len(display)} days ---")
    print()

    header = (f"{'Date':<12} {'Prob':>5} {'Sig':>4} "
              f"{'Open':>8} {'High':>8} {'Low':>8} {'Close':>8} "
              f"{'Range%':>7} {'C2C%':>6} {'O2C%':>6} {'Gap%':>6} "
              f"{'MaxDD%':>7} {'MaxRU%':>7} {'RV20%':>6} "
              f"{'Actual':>6} {'Result':>6}")
    print(header)
    print("-" * len(header))

    for date, r in display.iterrows():
        actual_str = "BIG" if r["actual"] == 1 else "-"
        result_str = r["result"]

        # Color hint via marker
        if result_str == "TP":
            marker = "TP"
        elif result_str == "FP":
            marker = "FP"
        elif result_str == "FN":
            marker = "FN"
        else:
            marker = "  "

        rv_str = f"{r['rv20%']:>6.1f}" if "rv20%" in r.index and not pd.isna(r.get("rv20%")) else "   N/A"

        print(
            f"{str(date.date()):<12} {r['prob']:>5.3f} {r['signal']:>4} "
            f"{r['open']:>8.2f} {r['high']:>8.2f} {r['low']:>8.2f} {r['close']:>8.2f} "
            f"{r['range%']:>7.2f} {r['c2c%']:>6.2f} {r['o2c%']:>6.2f} {r['gap%']:>6.2f} "
            f"{r['maxdd%']:>7.2f} {r['maxru%']:>7.2f} {rv_str} "
            f"{actual_str:>6} {marker:>6}"
        )

    # ── Rolling accuracy (20-day window) for signal days ──
    signal_days = comp[comp["prob"] >= eval_thresh].copy()
    if len(signal_days) >= 5:
        print()
        print("=" * 80)
        print(f"ROLLING HIT RATE (last N signals, threshold {eval_thresh})")
        print("=" * 80)
        hits_cum = (signal_days["result"] == "TP").cumsum()
        total_cum = range(1, len(signal_days) + 1)

        print(f"{'After Signal #':<16} {'Date':<12} {'Cumulative Hits':>16} {'Hit Rate':>10}")
        print("-" * 58)
        for i, (date, row) in enumerate(signal_days.iterrows()):
            n = i + 1
            h = int(hits_cum.iloc[i])
            rate = h / n
            if n <= 10 or n % 5 == 0 or n == len(signal_days):
                print(f"{'#' + str(n):<16} {str(date.date()):<12} {f'{h}/{n}':>16} {rate:>9.0%}")

    # ── CSV export ──
    if args.csv:
        csv_path = Path(__file__).parent / "daily_compare.csv"
        display.to_csv(csv_path, float_format="%.4f")
        print(f"\nExported {len(display)} rows to {csv_path}")


if __name__ == "__main__":
    main()
