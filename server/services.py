"""Service layer: thin wrappers around existing modules with caching."""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from utils.paths import OUTPUT_DIR, MODEL_DIR

# ── Cache ──
_cache = {
    "model": None,
    "feature_cols": None,
    "features_df": None,
    "dm_mtime": 0,
    "eval_metrics": None,
}

_task_status = {"status": "idle", "message": "", "updated": ""}


def _check_cache():
    """Reload if daily_metrics.parquet changed."""
    dm_path = OUTPUT_DIR / "daily_metrics.parquet"
    if not dm_path.exists():
        return
    mtime = dm_path.stat().st_mtime
    if mtime != _cache["dm_mtime"]:
        _cache["dm_mtime"] = mtime
        _cache["model"] = None
        _cache["features_df"] = None
        _cache["eval_metrics"] = None


def _load_model():
    """Load model (cached)."""
    _check_cache()
    if _cache["model"] is None:
        from models.training import load_model
        model, feat_cols = load_model(MODEL_DIR / "interaction_model.joblib")
        _cache["model"] = model
        _cache["feature_cols"] = feat_cols
    return _cache["model"], _cache["feature_cols"]


def _build_features():
    """Build full feature DataFrame (cached)."""
    _check_cache()
    if _cache["features_df"] is None:
        from features.base import engineer_base_features
        from features.external import engineer_all_external
        from features.interactions import build_interaction_features

        daily = pd.read_parquet(OUTPUT_DIR / "daily_metrics.parquet")
        daily.index = pd.to_datetime(daily.index)
        ext = pd.read_parquet(OUTPUT_DIR / "external_data.parquet")
        ext.index = pd.to_datetime(ext.index)

        df = engineer_base_features(daily)
        df = engineer_all_external(df, ext)
        df = build_interaction_features(df)
        _cache["features_df"] = df
    return _cache["features_df"]


def _predict_latest():
    """Run prediction on latest data row."""
    model, feature_cols = _load_model()
    df = _build_features()
    available = [f for f in feature_cols if f in df.columns]

    latest = df.iloc[-1]
    X = latest[available].values.reshape(1, -1)
    prob = float(model.predict_proba(X)[:, 1][0])

    if prob >= 0.7:
        signal = "HIGH"
    elif prob >= 0.5:
        signal = "ELEVATED"
    elif prob >= 0.3:
        signal = "MODERATE"
    else:
        signal = "LOW"

    # Context
    ctx = {
        "data_date": str(latest.name.date()),
        "prob": prob,
        "signal": signal,
        "close": float(latest.get("reg_close", 0)),
        "range_pct": float(latest.get("intraday_range", 0)) * 100,
        "c2c_pct": float(latest.get("close_to_close_ret", 0)) * 100,
        "o2c_pct": float(latest.get("open_to_close_ret", 0)) * 100,
        "rv20": float(latest.get("realized_vol_20d", 0)) * 100,
        "vrp": float(latest.get("vrp_20d", 0)),
        "max_dd": float(latest.get("max_drawdown", 0)) * 100,
        "max_ru": float(latest.get("max_runup", 0)) * 100,
    }

    # VRP label
    if ctx["vrp"] < -0.05:
        ctx["vrp_label"] = "Complacent"
    elif ctx["vrp"] > 0.05:
        ctx["vrp_label"] = "Fearful"
    else:
        ctx["vrp_label"] = "Neutral"

    # Events
    events = []
    days_to_fomc = latest.get("days_to_fomc", 99)
    days_to_nfp = latest.get("days_to_nfp", 99)
    if days_to_fomc <= 2:
        events.append(f"FOMC in {max(0, int(days_to_fomc)-1)}d")
    if days_to_nfp <= 2:
        events.append(f"NFP in {max(0, int(days_to_nfp)-1)}d")
    if latest.get("is_earnings_season", 0) == 1:
        events.append("Earnings Season")
    ctx["events"] = events

    # Premarket from live cache
    pre_path = OUTPUT_DIR / "live" / "live_premarket.csv"
    if pre_path.exists():
        pre = pd.read_csv(pre_path, index_col=0, parse_dates=True)
        if not pre.empty:
            lp = pre.iloc[-1]
            ctx["pre_range"] = float(lp.get("premarket_range", 0)) * 100
            ctx["pre_ret"] = float(lp.get("premarket_ret", 0)) * 100

    return ctx


# ── Public Service Functions ──

def get_dashboard():
    """Dashboard data: current signal + recent signals."""
    ctx = _predict_latest()

    # Recent 5 signal days (prob >= 0.3)
    model, feature_cols = _load_model()
    df = _build_features()
    available = [f for f in feature_cols if f in df.columns]
    test = df.loc["2023-01-01":]
    proba = model.predict_proba(test[available].values)[:, 1]

    recent_signals = []
    for i in range(len(test) - 1, max(len(test) - 200, -1), -1):
        if proba[i] >= 0.3:
            row = test.iloc[i]
            actual_range = float(row.get("intraday_range", 0)) * 100
            hit = actual_range > 2.0
            recent_signals.append({
                "date": str(test.index[i].date()),
                "prob": float(proba[i]),
                "signal": "HIGH" if proba[i] >= 0.7 else ("ELEV" if proba[i] >= 0.5 else "MOD"),
                "range": actual_range,
                "hit": hit,
            })
            if len(recent_signals) >= 10:
                break

    return {**ctx, "recent_signals": recent_signals, "task_status": _task_status}


def get_signal_detail():
    """Full prediction detail."""
    return _predict_latest()


def get_history(start: str = "", threshold: float = 0.3):
    """Signal history with hit/miss classification."""
    model, feature_cols = _load_model()
    df = _build_features()
    available = [f for f in feature_cols if f in df.columns]

    if not start:
        start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

    test = df.loc[start:]
    if test.empty:
        return {"signals": [], "start": start, "threshold": threshold,
                "total": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0,
                "hit_rate": 0, "recall": 0, "base_rate": 0}

    proba = model.predict_proba(test[available].values)[:, 1]
    actual = (test["intraday_range"] > 0.02).astype(int).values

    signals = []
    tp = fp = fn = tn = 0
    for i in range(len(test)):
        p = float(proba[i])
        a = int(actual[i])
        predicted = p >= threshold
        is_hit = a == 1

        if predicted and is_hit:
            result = "TP"; tp += 1
        elif predicted and not is_hit:
            result = "FP"; fp += 1
        elif not predicted and is_hit:
            result = "FN"; fn += 1
        else:
            result = "TN"; tn += 1

        if predicted or is_hit:  # show signal days and big move days
            row = test.iloc[i]
            signals.append({
                "date": str(test.index[i].date()),
                "prob": p,
                "signal": "HIGH" if p >= 0.7 else ("ELEV" if p >= 0.5 else ("MOD" if p >= 0.3 else "")),
                "range": float(row.get("intraday_range", 0)) * 100,
                "c2c": float(row.get("close_to_close_ret", 0)) * 100,
                "close": float(row.get("reg_close", 0)),
                "result": result,
            })

    total = tp + fp + fn + tn
    hit_rate = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    base_rate = (tp + fn) / total if total > 0 else 0

    return {
        "signals": signals, "start": start, "threshold": threshold,
        "total": total, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "hit_rate": hit_rate, "recall": recall, "base_rate": base_rate,
    }


def get_data_status():
    """Check data file freshness."""
    files = []
    for name, path in [
        ("daily_metrics", OUTPUT_DIR / "daily_metrics.parquet"),
        ("external_data", OUTPUT_DIR / "external_data.parquet"),
        ("interaction_features", OUTPUT_DIR / "interaction_features.parquet"),
        ("live_qqq", OUTPUT_DIR / "live" / "live_qqq.csv"),
        ("live_vix", OUTPUT_DIR / "live" / "live_vix.csv"),
        ("live_premarket", OUTPUT_DIR / "live" / "live_premarket.csv"),
    ]:
        if path.exists():
            stat = path.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime)
            size_kb = stat.st_size / 1024

            # Try to get last date in file
            last_date = ""
            rows = 0
            try:
                if path.suffix == ".parquet":
                    df = pd.read_parquet(path)
                    df.index = pd.to_datetime(df.index)
                    last_date = str(df.index[-1].date())
                    rows = len(df)
                elif path.suffix == ".csv":
                    df = pd.read_csv(path, index_col=0, parse_dates=True)
                    last_date = str(df.index[-1].date())
                    rows = len(df)
            except Exception:
                pass

            age_hours = (datetime.now() - mtime).total_seconds() / 3600
            if age_hours < 12:
                status = "fresh"
            elif age_hours < 48:
                status = "ok"
            else:
                status = "stale"

            files.append({
                "name": name,
                "path": str(path.name),
                "size_kb": f"{size_kb:.0f}",
                "modified": mtime.strftime("%Y-%m-%d %H:%M"),
                "last_date": last_date,
                "rows": rows,
                "status": status,
            })
        else:
            files.append({
                "name": name, "path": str(path.name),
                "size_kb": "-", "modified": "-", "last_date": "-",
                "rows": 0, "status": "missing",
            })

    return {"files": files}


def get_model_info():
    """Model metadata and evaluation metrics."""
    model, feature_cols = _load_model()

    info = {
        "model_type": type(model).__name__,
        "n_features": len(feature_cols) if feature_cols else 0,
        "model_file": "interaction_model.joblib",
        "features": feature_cols[:20] if feature_cols else [],  # show first 20
        "features_total": len(feature_cols) if feature_cols else 0,
    }

    # Cached evaluation
    if _cache["eval_metrics"] is None:
        from models.evaluation import evaluate_model, backtest_thresholds

        df = _build_features()
        available = [f for f in feature_cols if f in df.columns]
        test = df.loc["2023-01-01":]

        y_range = (test["intraday_range"] > 0.02).astype(int).values
        y_proba = model.predict_proba(test[available].values)[:, 1]

        metrics = evaluate_model(y_range, y_proba)
        bt = backtest_thresholds(y_range, y_proba, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        bt_list = bt.to_dict("records")

        _cache["eval_metrics"] = {
            "target": "0DTE Range>2%",
            "test_period": f"2023-01-01 to {test.index[-1].date()}",
            "test_days": len(test),
            "auc": f"{metrics['auc']:.3f}",
            "ap": f"{metrics['ap']:.3f}",
            "brier": f"{metrics['brier']:.3f}",
            "base_rate": f"{metrics['base_rate']:.1%}",
            "backtest": bt_list,
        }

    info.update(_cache["eval_metrics"])
    return info


def trigger_fetch():
    """Background task: fetch + merge data."""
    global _task_status
    _task_status = {"status": "running", "message": "Fetching data...",
                    "updated": datetime.now().strftime("%H:%M:%S")}
    try:
        from live.notify import update_data
        source = update_data()
        _task_status = {"status": "done", "message": f"Data updated ({source})",
                        "updated": datetime.now().strftime("%H:%M:%S")}
        # Invalidate cache
        _cache["dm_mtime"] = 0
    except Exception as e:
        _task_status = {"status": "error", "message": str(e),
                        "updated": datetime.now().strftime("%H:%M:%S")}


def trigger_predict():
    """Background task: run prediction (just invalidates cache)."""
    global _task_status
    _task_status = {"status": "running", "message": "Running prediction...",
                    "updated": datetime.now().strftime("%H:%M:%S")}
    try:
        _cache["dm_mtime"] = 0  # force reload
        _predict_latest()
        _task_status = {"status": "done", "message": "Prediction updated",
                        "updated": datetime.now().strftime("%H:%M:%S")}
    except Exception as e:
        _task_status = {"status": "error", "message": str(e),
                        "updated": datetime.now().strftime("%H:%M:%S")}
