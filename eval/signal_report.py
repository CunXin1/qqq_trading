"""
Generate detailed signal report: all alerts with OHLCV, events, and missed moves.

Usage:
    python eval/signal_report.py                          # default: prob >= 0.5
    python eval/signal_report.py --threshold 0.6          # higher confidence
    python eval/signal_report.py --threshold 0.3 --csv    # export to CSV
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
    parser = argparse.ArgumentParser(description="Signal report with OHLCV detail")
    parser.add_argument("--threshold", type=float, default=0.5, help="Min probability (default: 0.5)")
    parser.add_argument("--test-start", type=str, default="2023-01-01", help="Test period start")
    parser.add_argument("--miss-range", type=float, default=3.0, help="Range%% to count as missed big move (default: 3.0)")
    parser.add_argument("--csv", action="store_true", help="Export alerts to eval/signals.csv")
    return parser.parse_args()


def load_and_build_features():
    """Load data and build full feature set."""
    daily = pd.read_parquet(OUTPUT_DIR / "daily_metrics.parquet")
    daily.index = pd.to_datetime(daily.index)
    ext = pd.read_parquet(OUTPUT_DIR / "external_data.parquet")
    ext.index = pd.to_datetime(ext.index)

    df = engineer_base_features(daily)
    df = engineer_all_external(df, ext)
    df = build_interaction_features(df)
    return df


def build_signal_table(df, model, feature_cols, test_start):
    """Build DataFrame with predictions and market data for test period."""
    available = [f for f in feature_cols if f in df.columns]
    test = df.loc[test_start:]
    y_proba = model.predict_proba(test[available].values)[:, 1]

    signals = pd.DataFrame({
        "prob": y_proba,
        "open": test["reg_open"].values,
        "high": test["reg_high"].values,
        "low": test["reg_low"].values,
        "close": test["reg_close"].values,
        "range_pct": (test["intraday_range"] * 100).values,
        "c2c_ret": (test["close_to_close_ret"] * 100).values,
        "o2c_ret": (test["open_to_close_ret"] * 100).values,
        "gap_pct": (test["gap_return"] * 100).values,
        "max_dd": (test["max_drawdown"] * 100).values,
        "max_ru": (test["max_runup"] * 100).values,
    }, index=test.index)

    # Event flags
    for col in ["is_fomc_day", "is_nfp_day", "is_earnings_season", "fomc_imminent", "nfp_imminent"]:
        if col in test.columns:
            signals[col] = test[col].values

    # VRP and realized vol
    if "vrp_20d" in test.columns:
        signals["vrp_20d"] = test["vrp_20d"].values
    if "realized_vol_20d" in test.columns:
        signals["rv20"] = (test["realized_vol_20d"] * 100).values

    # Signal level and hit
    signals["signal"] = "NONE"
    signals.loc[signals["prob"] >= 0.5, "signal"] = "ELEV"
    signals.loc[signals["prob"] >= 0.7, "signal"] = "HIGH"
    signals["hit"] = signals["range_pct"] > 2.0

    return signals


def format_events(row):
    """Build event string from flag columns."""
    events = []
    if row.get("is_fomc_day", 0) == 1:
        events.append("FOMC")
    if row.get("is_nfp_day", 0) == 1:
        events.append("NFP")
    if row.get("fomc_imminent", 0) == 1 and row.get("is_fomc_day", 0) != 1:
        events.append("FOMC-1")
    if row.get("nfp_imminent", 0) == 1 and row.get("is_nfp_day", 0) != 1:
        events.append("NFP-1")
    if row.get("is_earnings_season", 0) == 1:
        events.append("EARN")
    return ",".join(events) if events else "-"


def print_alerts(alerts):
    """Print detailed alert table."""
    header = (
        f"{'Date':<12} {'Prob':>5} {'Sig':>4} {'Open':>8} {'High':>8} {'Low':>8} "
        f"{'Close':>8} {'Range%':>7} {'C2C%':>6} {'O2C%':>6} {'Gap%':>6} "
        f"{'MaxDD%':>7} {'MaxRU%':>7} {'RV20%':>6} {'VRP':>6} {'Events':>12} {'Hit':>4}"
    )
    print(header)
    print("-" * len(header))

    for date, r in alerts.iterrows():
        evt = format_events(r)
        hit = "Y" if r["hit"] else "N"
        print(
            f"{str(date.date()):<12} {r['prob']:>5.3f} {r['signal']:>4} "
            f"{r['open']:>8.2f} {r['high']:>8.2f} {r['low']:>8.2f} {r['close']:>8.2f} "
            f"{r['range_pct']:>7.2f} {r['c2c_ret']:>6.2f} {r['o2c_ret']:>6.2f} {r['gap_pct']:>6.2f} "
            f"{r['max_dd']:>7.2f} {r['max_ru']:>7.2f} {r['rv20']:>6.1f} {r['vrp_20d']:>6.3f} "
            f"{evt:>12} {hit:>4}"
        )


def print_monthly_summary(alerts):
    """Print monthly aggregation."""
    alerts = alerts.copy()
    alerts["month"] = alerts.index.to_period("M")
    monthly = alerts.groupby("month").agg(
        signals=("prob", "count"),
        hits=("hit", "sum"),
        avg_prob=("prob", "mean"),
        avg_range=("range_pct", "mean"),
        max_range=("range_pct", "max"),
    ).reset_index()
    monthly["hit_rate"] = monthly["hits"] / monthly["signals"]

    print(f"{'Month':<10} {'Signals':>8} {'Hits':>5} {'HitRate':>8} {'AvgProb':>8} {'AvgRange%':>10} {'MaxRange%':>10}")
    print("-" * 65)
    for _, r in monthly.iterrows():
        print(
            f"{str(r['month']):<10} {r['signals']:>8} {r['hits']:>5.0f} "
            f"{r['hit_rate']:>7.0%} {r['avg_prob']:>8.3f} {r['avg_range']:>10.2f} {r['max_range']:>10.2f}"
        )


def print_missed_moves(signals, threshold, miss_range):
    """Print days with big moves but no signal."""
    missed = signals[(signals["prob"] < threshold) & (signals["range_pct"] > miss_range)]
    print(f"MISSED BIG MOVES (prob < {threshold}, range > {miss_range}%): {len(missed)} days")

    if len(missed) == 0:
        return

    print(f"{'Date':<12} {'Prob':>5} {'Range%':>7} {'C2C%':>6} {'RV20%':>6} {'Events':>12}")
    print("-" * 55)
    for date, r in missed.iterrows():
        evt = format_events(r)
        print(f"{str(date.date()):<12} {r['prob']:>5.3f} {r['range_pct']:>7.2f} {r['c2c_ret']:>6.2f} {r['rv20']:>6.1f} {evt:>12}")


def main():
    args = parse_args()

    print("Loading data and building features...")
    df = load_and_build_features()

    print("Loading model...")
    model, feature_cols = load_model(MODEL_DIR / "interaction_model.joblib")
    print(f"  Model: {type(model).__name__}, Features: {len(feature_cols)}")

    signals = build_signal_table(df, model, feature_cols, args.test_start)
    alerts = signals[signals["prob"] >= args.threshold]
    test_days = len(signals)
    n_hits = alerts["hit"].sum()

    # ── Summary ──
    print()
    print("=" * 100)
    print(f"SIGNAL REPORT  |  Test: {args.test_start} to {signals.index[-1].date()}  |  "
          f"{test_days} days  |  Threshold: {args.threshold}")
    print("=" * 100)
    print(f"Alerts: {len(alerts)}  |  Hits (range>2%): {n_hits}  |  "
          f"Hit rate: {alerts['hit'].mean():.1%}  |  Base rate: {signals['hit'].mean():.1%}  |  "
          f"Lift: {alerts['hit'].mean() / signals['hit'].mean():.1f}x")
    print()

    # ── All alerts ──
    print_alerts(alerts)

    # ── Monthly ──
    print()
    print("=" * 100)
    print("MONTHLY SUMMARY")
    print("=" * 100)
    print_monthly_summary(alerts)

    # ── Missed moves ──
    print()
    print("=" * 100)
    print_missed_moves(signals, args.threshold, args.miss_range)

    # ── CSV export ──
    if args.csv:
        csv_path = Path(__file__).parent / "signals.csv"
        export = alerts[["prob", "signal", "open", "high", "low", "close",
                         "range_pct", "c2c_ret", "o2c_ret", "gap_pct",
                         "max_dd", "max_ru", "rv20", "vrp_20d", "hit"]].copy()
        export["events"] = [format_events(alerts.loc[d]) for d in alerts.index]
        export.to_csv(csv_path)
        print(f"\nExported {len(export)} alerts to {csv_path}")


if __name__ == "__main__":
    main()
