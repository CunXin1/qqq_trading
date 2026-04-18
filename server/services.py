"""
Service layer: thin wrappers around features/models/eval with caching.
服务层：对特征/模型/评估模块的轻量封装，带缓存机制。

Uses file-mtime-based cache invalidation — when daily_metrics.parquet is
updated on disk, all cached model predictions and feature DataFrames are
automatically reloaded on the next request.
采用基于文件修改时间的缓存失效策略 — 当 daily_metrics.parquet 文件更新时，
所有缓存的模型预测和特征 DataFrame 将在下次请求时自动重新加载。

Key public functions / 主要公开函数:
    get_dashboard()      — Current signal + recent 10 alert days for the main page.
                           当前信号 + 最近 10 个告警日，用于主页面。
    get_signal_detail()  — Full prediction context (probability, price, vol, events).
                           完整预测上下文（概率、价格、波动率、事件）。
    get_history()        — Signal history with TP/FP/FN/TN classification.
                           信号历史，包含 TP/FP/FN/TN 分类。
    get_data_status()    — Freshness check for all data files (fresh/ok/stale/missing).
                           所有数据文件的新鲜度检查（fresh/ok/stale/missing）。
    get_model_info()     — Model metadata + cached AUC/AP/Brier evaluation metrics.
                           模型元数据 + 缓存的 AUC/AP/Brier 评估指标。
    trigger_fetch()      — Background task: fetch new data and invalidate cache.
                           后台任务：抓取新数据并使缓存失效。
    trigger_predict()    — Background task: force re-prediction with fresh data.
                           后台任务：使用最新数据强制重新预测。
"""
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
        model, feat_cols = load_model(MODEL_DIR / "range_0dte_2pct_2007_2022.joblib")
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

    # Compute next trading day (skip weekends)
    data_date = latest.name.date()
    next_day = data_date + timedelta(days=1)
    while next_day.weekday() >= 5:  # skip Saturday(5) / Sunday(6)
        next_day += timedelta(days=1)

    # Context
    ctx = {
        "data_date": str(data_date),
        "predict_date": str(next_day),
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
        "model_file": "range_0dte_2pct_2007_2022.joblib",
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


_eval_cache: dict = {"data": None, "params": None, "dm_mtime": 0.0}


def get_eval_report(task: str = "range_0dte", model: str | None = None,
                    start: str = "2023-01-01", end: str | None = None,
                    thresh: float = 0.02, threshold: float = 0.5,
                    miss_thresh: float = 3.0) -> dict:
    """Evaluation report with file-mtime caching.
    带文件修改时间缓存的评估报告。"""
    from eval._common import get_report_data, TASKS, list_models

    params = (task, model, start, end, thresh, threshold, miss_thresh)
    dm_path = OUTPUT_DIR / "daily_metrics.parquet"
    mtime = dm_path.stat().st_mtime if dm_path.exists() else 0

    if (_eval_cache["params"] == params and
            _eval_cache["dm_mtime"] == mtime and
            _eval_cache["data"] is not None):
        data = _eval_cache["data"]
    else:
        data = get_report_data(task, model, start, end, thresh, threshold, miss_thresh)
        _eval_cache.update(data=data, params=params, dm_mtime=mtime)

    # Attach available tasks and models for the filter form
    data["available_tasks"] = [
        {"key": k, "name": v["name"]} for k, v in TASKS.items()
    ]
    data["available_models"] = list_models()
    data["selected_model"] = model or TASKS[task]["default_model"]
    return data


_cross_cache: dict = {"data": None, "dm_mtime": 0.0}


def get_cross_eval() -> dict:
    """Cross-evaluation matrix with caching.
    带缓存的交叉评估矩阵。"""
    from eval._common import get_cross_eval_matrix

    dm_path = OUTPUT_DIR / "daily_metrics.parquet"
    mtime = dm_path.stat().st_mtime if dm_path.exists() else 0

    if _cross_cache["dm_mtime"] == mtime and _cross_cache["data"] is not None:
        return _cross_cache["data"]

    data = get_cross_eval_matrix()
    _cross_cache.update(data=data, dm_mtime=mtime)
    return data


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
