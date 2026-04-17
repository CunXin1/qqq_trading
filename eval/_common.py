"""Shared helpers for task-specific eval scripts.

Three task-specific scripts (eval/eval_range_0dte.py, eval/eval_otc_0dte.py,
eval/eval_c2c_1dte.py) each evaluate their task's true label against any model's
predicted probabilities. This lets you cross-evaluate: e.g., use the c2c_1dte
model to "predict" range_0dte — useful for showing why a task-matched model
wins.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from features.base import engineer_base_features
from features.external import engineer_all_external
from features.interactions import build_interaction_features
from models.training import load_model
from utils.paths import OUTPUT_DIR, MODEL_DIR


# task_name → how to compute the TRUE outcome for that day
# shift=0:  predict & measure on the SAME day (0DTE)
# shift=-1: predict today, measure tomorrow (1DTE) — actual = raw_col.shift(-1)
TASKS = {
    "range_0dte": {
        "col":           "intraday_range",
        "shift":          0,
        "name":          "0DTE Range (H-L)/O",
        "default_model": "range_0dte_2pct_model",
    },
    "otc_0dte": {
        "col":           "abs_open_to_close",
        "shift":          0,
        "name":          "0DTE |O2C|",
        "default_model": "otc_0dte_2pct_model",
    },
    "c2c_1dte": {
        "col":           "abs_close_to_close",
        "shift":         -1,
        "name":          "1DTE next-day |C2C|",
        "default_model": "c2c_1dte_2pct_model",
    },
}


def load_features() -> pd.DataFrame:
    daily = pd.read_parquet(OUTPUT_DIR / "daily_metrics.parquet")
    daily.index = pd.to_datetime(daily.index)
    ext = pd.read_parquet(OUTPUT_DIR / "external_data.parquet")
    ext.index = pd.to_datetime(ext.index)
    df = engineer_base_features(daily)
    df = engineer_all_external(df, ext)
    df = build_interaction_features(df)
    return df


def resolve_model_path(name_or_path: str) -> Path:
    """Accept a model name ('range_0dte_2pct_model'), short alias, or explicit path."""
    p = Path(name_or_path)
    if p.is_absolute() and p.exists():
        return p
    candidates = [
        MODEL_DIR / name_or_path,
        MODEL_DIR / f"{name_or_path}.joblib",
        MODEL_DIR / f"{name_or_path}_model.joblib",
    ]
    for c in candidates:
        if c.suffix == ".joblib" and c.exists():
            return c
    available = sorted(p.stem for p in MODEL_DIR.glob("*.joblib"))
    raise FileNotFoundError(
        f"Model '{name_or_path}' not found in {MODEL_DIR}. Available: {available}"
    )


def build_eval_table(df: pd.DataFrame, task: str, model_name: str,
                     start: str | None = None, end: str | None = None,
                     thresh: float = 0.02) -> tuple[pd.DataFrame, str]:
    """Return a per-day DataFrame with prob, actual, hit flag, and OHLC context.

    thresh defines what counts as a "hit" for this task (e.g., 0.02 for range>2%).
    """
    tc = TASKS[task]
    model_path = resolve_model_path(model_name)
    model, feature_cols = load_model(model_path)
    if feature_cols is None:
        raise ValueError(f"{model_path.name} has no sidecar CSV with feature columns")

    available = [f for f in feature_cols if f in df.columns]
    missing = set(feature_cols) - set(available)
    if missing:
        print(f"WARN: {len(missing)} features in model not in df: {sorted(missing)[:5]}")

    # Build actual label for the task (works with the TRUE outcome, independent of the model)
    # shift=-1 shifts FUTURE values into today's row (target = tomorrow's metric)
    actual_raw = df[tc["col"]]
    if tc["shift"] != 0:
        actual_raw = actual_raw.shift(tc["shift"])
    actual_pct = (actual_raw * 100).rename("actual_pct")

    subset = df.loc[start:end] if (start or end) else df
    feat_mat = subset[available].copy()
    feat_mat = feat_mat.dropna()
    proba = model.predict_proba(feat_mat.values)[:, 1]

    out = pd.DataFrame(index=feat_mat.index)
    out["prob"] = proba
    out["actual_pct"] = actual_pct.loc[feat_mat.index].values
    out["hit"] = out["actual_pct"] > (thresh * 100)

    # Keep OHLC + context columns (from daily_metrics → features), when present
    def attach(col_in, col_out, mult=1.0):
        if col_in in subset.columns:
            out[col_out] = (subset.loc[out.index, col_in].values * mult)

    attach("reg_open",            "open")
    attach("reg_high",            "high")
    attach("reg_low",             "low")
    attach("reg_close",           "close")
    attach("intraday_range",      "range_pct", 100.0)
    attach("close_to_close_ret",  "c2c_pct",   100.0)
    attach("open_to_close_ret",   "o2c_pct",   100.0)
    attach("gap_return",          "gap_pct",   100.0)
    attach("max_drawdown",        "max_dd",    100.0)
    attach("max_runup",           "max_ru",    100.0)
    attach("realized_vol_20d",    "rv20",      100.0)
    attach("vrp_20d",             "vrp_20d")

    for col in ["is_fomc_day", "is_nfp_day", "is_earnings_season",
                "fomc_imminent", "nfp_imminent"]:
        if col in subset.columns:
            out[col] = subset.loc[out.index, col].values

    out.dropna(subset=["actual_pct"], inplace=True)
    return out, model_path.name


def format_events(row) -> str:
    evts = []
    if row.get("is_fomc_day", 0) == 1: evts.append("FOMC")
    if row.get("is_nfp_day", 0) == 1:  evts.append("NFP")
    if row.get("fomc_imminent", 0) == 1 and row.get("is_fomc_day", 0) != 1: evts.append("FOMC-1")
    if row.get("nfp_imminent", 0) == 1 and row.get("is_nfp_day", 0) != 1:   evts.append("NFP-1")
    if row.get("is_earnings_season", 0) == 1: evts.append("EARN")
    return ",".join(evts) if evts else "-"


def _summary_metrics(sig: pd.DataFrame, threshold: float) -> dict:
    y_true = sig["hit"].astype(int)
    y_prob = sig["prob"]
    alerts = sig[sig["prob"] >= threshold]
    tp = int((alerts["hit"]).sum())
    fp = int((~alerts["hit"]).sum())
    actual_pos = int(y_true.sum())
    fn = actual_pos - tp
    tn = len(sig) - tp - fp - fn
    return {
        "n": len(sig),
        "base_rate": y_true.mean(),
        "alerts": len(alerts),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / max(len(alerts), 1),
        "recall":    tp / max(actual_pos, 1),
        "auc":       roc_auc_score(y_true, y_prob) if y_true.nunique() == 2 else float("nan"),
        "ap":        average_precision_score(y_true, y_prob) if y_true.nunique() == 2 else float("nan"),
    }


def print_header(task: str, model_file: str, sig: pd.DataFrame, threshold: float, thresh: float):
    tc = TASKS[task]
    m = _summary_metrics(sig, threshold)
    match = "(task-matched)" if model_file.startswith(tc["default_model"]) else "(CROSS-TASK)"
    print("=" * 118)
    print(f"TASK: {tc['name']} > {thresh:.0%}   |   MODEL: {model_file} {match}")
    print(f"Range: {sig.index.min().date()} to {sig.index.max().date()}   |   Days: {m['n']}   |   Threshold: {threshold}")
    print("=" * 118)
    print(f"AUC: {m['auc']:.3f}  AP: {m['ap']:.3f}  |  Base rate: {m['base_rate']:.1%}  |  "
          f"Alerts: {m['alerts']}  |  TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")
    print(f"Precision: {m['precision']:.1%}  |  Recall: {m['recall']:.1%}  |  "
          f"Lift: {m['precision']/max(m['base_rate'],1e-9):.2f}x")
    print()


def print_alerts(alerts: pd.DataFrame):
    print("--- ALERTS ---")
    hdr = (f"{'Date':<12}{'Prob':>6}{'Actual%':>8}{'Range%':>8}{'C2C%':>7}{'O2C%':>7}"
           f"{'Gap%':>7}{'MaxDD':>7}{'MaxRU':>7}{'RV20':>6}{'VRP':>7}  Events  Hit")
    print(hdr); print("-" * len(hdr))
    for d, r in alerts.iterrows():
        evt = format_events(r)
        hit = "Y" if r["hit"] else "N"
        print(f"{str(d.date()):<12}{r['prob']:>6.3f}{r['actual_pct']:>8.2f}"
              f"{r.get('range_pct', float('nan')):>8.2f}{r.get('c2c_pct', float('nan')):>7.2f}"
              f"{r.get('o2c_pct', float('nan')):>7.2f}{r.get('gap_pct', float('nan')):>7.2f}"
              f"{r.get('max_dd', float('nan')):>7.2f}{r.get('max_ru', float('nan')):>7.2f}"
              f"{r.get('rv20', float('nan')):>6.1f}{r.get('vrp_20d', float('nan')):>7.3f}  {evt:<7}  {hit}")


def print_monthly(alerts: pd.DataFrame):
    a = alerts.copy()
    a["month"] = a.index.to_period("M")
    monthly = a.groupby("month").agg(
        signals=("prob", "count"),
        hits=("hit", "sum"),
        avg_prob=("prob", "mean"),
        avg_actual=("actual_pct", "mean"),
        max_actual=("actual_pct", "max"),
    ).reset_index()
    monthly["hit_rate"] = monthly["hits"] / monthly["signals"]
    print("--- MONTHLY SUMMARY ---")
    print(f"{'Month':<10}{'Signals':>9}{'Hits':>6}{'HitRate':>9}{'AvgProb':>9}{'AvgActual%':>11}{'MaxActual%':>11}")
    print("-" * 65)
    for _, r in monthly.iterrows():
        print(f"{str(r['month']):<10}{r['signals']:>9}{r['hits']:>6.0f}"
              f"{r['hit_rate']:>8.0%}{r['avg_prob']:>9.3f}{r['avg_actual']:>11.2f}{r['max_actual']:>11.2f}")


def print_missed(sig: pd.DataFrame, threshold: float, miss_thresh: float):
    """Days where ACTUAL was big (> miss_thresh) but model didn't signal (prob < threshold)."""
    missed = sig[(sig["prob"] < threshold) & (sig["actual_pct"] > miss_thresh)]
    print(f"--- MISSED BIG MOVES (prob < {threshold}, actual > {miss_thresh}%): {len(missed)} days ---")
    if missed.empty:
        return
    print(f"{'Date':<12}{'Prob':>6}{'Actual%':>8}{'C2C%':>7}{'RV20':>6}  Events")
    print("-" * 55)
    for d, r in missed.iterrows():
        evt = format_events(r)
        print(f"{str(d.date()):<12}{r['prob']:>6.3f}{r['actual_pct']:>8.2f}"
              f"{r.get('c2c_pct', float('nan')):>7.2f}{r.get('rv20', float('nan')):>6.1f}  {evt}")


def print_false_alarms(alerts: pd.DataFrame, thresh_pct: float):
    fp = alerts[alerts["actual_pct"] <= thresh_pct]
    print(f"--- FALSE ALARMS (signal fired, actual <= {thresh_pct}%): {len(fp)} days ---")
    if fp.empty:
        return
    print(f"{'Date':<12}{'Prob':>6}{'Actual%':>8}{'Range%':>8}{'C2C%':>7}{'O2C%':>7}"
          f"{'MaxDD':>7}{'MaxRU':>7}{'RV20':>6}  Events")
    print("-" * 85)
    for d, r in fp.iterrows():
        evt = format_events(r)
        print(f"{str(d.date()):<12}{r['prob']:>6.3f}{r['actual_pct']:>8.2f}"
              f"{r.get('range_pct', float('nan')):>8.2f}{r.get('c2c_pct', float('nan')):>7.2f}"
              f"{r.get('o2c_pct', float('nan')):>7.2f}{r.get('max_dd', float('nan')):>7.2f}"
              f"{r.get('max_ru', float('nan')):>7.2f}{r.get('rv20', float('nan')):>6.1f}  {evt}")


def run_report(task: str, model: str | None = None,
               start: str = "2023-01-01", end: str | None = None,
               thresh: float = 0.02, threshold: float = 0.5,
               miss_thresh: float = 3.0):
    """Entry point used by task-specific eval scripts."""
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Choose from {list(TASKS)}")
    if model is None:
        model = TASKS[task]["default_model"]

    print("Loading features...")
    df = load_features()
    sig, model_file = build_eval_table(df, task, model, start, end, thresh)
    alerts = sig[sig["prob"] >= threshold]

    print_header(task, model_file, sig, threshold, thresh)
    print_alerts(alerts); print()
    print_monthly(alerts); print()
    print_missed(sig, threshold, miss_thresh); print()
    print_false_alarms(alerts, thresh * 100)
