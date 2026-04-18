"""Shared helpers for task-specific eval scripts.
任务特定评估脚本的共享辅助模块。

Three task-specific scripts (eval/eval_range_0dte.py, eval/eval_otc_0dte.py,
eval/eval_c2c_1dte.py) each evaluate their task's true label against any model's
predicted probabilities. This lets you cross-evaluate: e.g., use the c2c_1dte
model to "predict" range_0dte — useful for showing why a task-matched model wins.
三个任务特定脚本各自将其任务的真实标签与任意模型的预测概率进行评估。
这允许交叉评估：例如用 c2c_1dte 模型来"预测" range_0dte——
有助于展示为什么任务匹配的模型更优。

This module provides / 本模块提供:
  1. TASKS dict    — single source of truth defining each prediction task.
                     定义每个预测任务的唯一事实来源。
  2. load_features — rebuild the full 122-feature DataFrame from cached parquets.
                     从缓存的 parquet 文件重建完整的 122 特征 DataFrame。
  3. build_eval_table — compute predictions for any (model, task) pair.
                        为任意（模型, 任务）组合计算预测结果。
  4. Report printers — formatted output for alerts, monthly summary, missed moves,
                       false alarms.
                       格式化输出：警报列表、月度汇总、漏报大波动、假阳性。

Cross-evaluation matrix / 交叉评估矩阵:
  With 3 models × 3 eval targets = 9 combinations, you can build a full
  cross-evaluation matrix to verify that task-matched models consistently
  outperform cross-task models on their own target.
  3 个模型 × 3 个评估目标 = 9 种组合，可构建完整的交叉评估矩阵，
  验证任务匹配模型在其自身目标上始终优于跨任务模型。
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


# Task definitions — single source of truth for all evaluation scripts.
# 任务定义——所有评估脚本的唯一事实来源。
#
# Each entry maps a task_name to:
# 每个条目将 task_name 映射到：
#   col:           Column in daily_metrics containing the raw metric.
#                  daily_metrics 中包含原始指标的列名。
#   shift:         0 for 0DTE (predict & measure same day),
#                  -1 for 1DTE (predict today, measure tomorrow).
#                  0 表示 0DTE（当天预测当天衡量），
#                  -1 表示 1DTE（今天预测明天衡量）。
#   name:          Human-readable display name for reports.
#                  报告中的可读显示名称。
#   default_model: Task-matched model filename stem (without .joblib).
#                  任务匹配模型的文件名主干（不含 .joblib）。
TASKS = {
    "range_0dte": {
        "col":           "intraday_range",        # (H-L)/O — straddle target / 跨式期权目标
        "shift":          0,                       # 0DTE: same-day / 当天
        "name":          "0DTE Range (H-L)/O",
        "default_model": "range_0dte_2pct_2007_2022",
    },
    "otc_0dte": {
        "col":           "abs_open_to_close",     # |O2C| — directional move / 方向性波动
        "shift":          0,                       # 0DTE: same-day / 当天
        "name":          "0DTE |O2C|",
        "default_model": "otc_0dte_2pct_2007_2022",
    },
    "c2c_1dte": {
        "col":           "abs_close_to_close",    # |C2C| — overnight + intraday / 隔夜+日内
        "shift":         -1,                       # 1DTE: predict today → measure tomorrow
                                                   # 今天预测 → 明天衡量
        "name":          "1DTE next-day |C2C|",
        "default_model": "c2c_1dte_2pct_2007_2022",
    },
}


def load_features() -> pd.DataFrame:
    """Load cached data and rebuild the full 122-feature DataFrame.
    加载缓存数据并重建完整的 122 特征 DataFrame。

    Pipeline: daily_metrics.parquet → base features (53)
              + external_data.parquet → external features (26)
              → interaction features (43) = 122 total.
    流程：daily_metrics.parquet → 基础特征 (53)
          + external_data.parquet → 外部特征 (26)
          → 交互特征 (43) = 共 122 个。
    """
    daily = pd.read_parquet(OUTPUT_DIR / "daily_metrics.parquet")
    daily.index = pd.to_datetime(daily.index)
    ext = pd.read_parquet(OUTPUT_DIR / "external_data.parquet")
    ext.index = pd.to_datetime(ext.index)
    df = engineer_base_features(daily)
    df = engineer_all_external(df, ext)
    df = build_interaction_features(df)
    return df


def resolve_model_path(name_or_path: str) -> Path:
    """Accept a model name, stem, or explicit path.
    接受模型名称、文件名主干或完整路径。

    Resolution order / 解析顺序:
      1. Absolute path → use directly / 绝对路径 → 直接使用
      2. MODEL_DIR / name_or_path (if already has .joblib suffix)
      3. MODEL_DIR / name_or_path.joblib

    Raises FileNotFoundError with available models if nothing matches.
    若无匹配则抛出 FileNotFoundError 并列出可用模型。
    """
    p = Path(name_or_path)
    if p.is_absolute() and p.exists():
        return p
    candidates = [
        MODEL_DIR / name_or_path,
        MODEL_DIR / f"{name_or_path}.joblib",
    ]
    for c in candidates:
        if c.suffix == ".joblib" and c.exists():
            return c
    available = sorted(p.stem for p in MODEL_DIR.glob("*.joblib"))
    raise FileNotFoundError(
        f"Model '{name_or_path}' not found in {MODEL_DIR}. Available: {available}"
    )


def list_models() -> list[str]:
    """List all available model stems (without .joblib suffix).
    列出所有可用模型名称（不含 .joblib 后缀）。"""
    return sorted(
        p.stem for p in MODEL_DIR.glob("*.joblib")
        if p.stem != "scaler"
    )


def build_eval_table(df: pd.DataFrame, task: str, model_name: str,
                     start: str | None = None, end: str | None = None,
                     thresh: float = 0.02) -> tuple[pd.DataFrame, str]:
    """Build a per-day evaluation DataFrame with predictions and market context.
    构建逐日评估 DataFrame，包含预测结果和市场上下文。

    This is the core cross-evaluation function. It works with ANY model
    on ANY task — the model provides probabilities, the task defines
    the ground truth label.
    这是核心交叉评估函数。它可以用任意模型评估任意任务——
    模型提供概率值，任务定义真实标签。

    Args:
        df:         Feature DataFrame (from load_features).
                    特征 DataFrame（来自 load_features）。
        task:       Task name (key in TASKS dict).
                    任务名称（TASKS 字典的键）。
        model_name: Model name or path to load.
                    模型名称或路径。
        start/end:  Date range to evaluate. / 评估的日期范围。
        thresh:     Move threshold defining a "hit" (default 0.02 = 2%).
                    定义"命中"的波动阈值（默认 0.02 = 2%）。

    Returns:
        (eval_table, model_filename): DataFrame with columns prob/actual_pct/hit/
        OHLC context, and the resolved model filename string.
        (评估表, 模型文件名): 包含 prob/actual_pct/hit/OHLC 上下文的 DataFrame，
        以及解析后的模型文件名字符串。
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
    """Build a compact event tag string from boolean flag columns.
    从布尔标记列构建紧凑的事件标签字符串。

    Examples / 示例: "FOMC,EARN", "NFP-1", "-" (no events).
    """
    evts = []
    if row.get("is_fomc_day", 0) == 1: evts.append("FOMC")       # FOMC 公布日
    if row.get("is_nfp_day", 0) == 1:  evts.append("NFP")        # 非农公布日
    if row.get("fomc_imminent", 0) == 1 and row.get("is_fomc_day", 0) != 1: evts.append("FOMC-1")  # FOMC 前夜
    if row.get("nfp_imminent", 0) == 1 and row.get("is_nfp_day", 0) != 1:   evts.append("NFP-1")   # 非农前夜
    if row.get("is_earnings_season", 0) == 1: evts.append("EARN") # 财报季
    return ",".join(evts) if evts else "-"


def _summary_metrics(sig: pd.DataFrame, threshold: float) -> dict:
    """Compute confusion matrix and classification metrics for a given threshold.
    计算给定阈值下的混淆矩阵和分类指标。

    Returns dict with: n, base_rate, alerts, tp, fp, fn, tn, precision, recall, auc, ap.
    返回字典包含：样本数、基准率、信号数、TP、FP、FN、TN、精确率、召回率、AUC、AP。
    """
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
    """Print the report header with task info, model info, and summary metrics.
    打印报告头部，包含任务信息、模型信息和汇总指标。"""
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
    """Print detailed alert table: each signal day with prob, actual, OHLC, events, hit.
    打印详细警报表：每个信号日的概率、实际值、OHLC、事件标记、命中状态。"""
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
    """Print monthly aggregation: signals, hits, hit rate, avg/max actual move.
    打印月度聚合：信号数、命中数、命中率、平均/最大实际波动。"""
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
    """Print missed big moves: days where actual exceeded miss_thresh but model probability < threshold.
    打印漏报的大波动：实际值超过 miss_thresh 但模型概率低于 threshold 的日子。

    These are FALSE NEGATIVES — the model's blind spots. Understanding them
    helps identify which market regimes the model struggles with.
    这些是假阴性——模型的盲区。分析它们有助于识别模型在哪些市场状态下表现不佳。
    """
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
    """Print false alarms: signal days where actual move was below the hit threshold.
    打印假阳性：模型发出信号但实际波动低于命中阈值的日子。

    These are FALSE POSITIVES — each one costs a straddle premium or triggers
    an unnecessary trade. Minimizing these is critical for profitability.
    这些是假阳性——每一次都会浪费跨式期权权利金或触发不必要的交易。
    最小化假阳性对盈利能力至关重要。
    """
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
    """Entry point used by task-specific eval scripts.
    任务特定评估脚本使用的入口函数。

    Orchestrates the full evaluation pipeline: load features → build eval table
    → print header, alerts, monthly summary, missed moves, and false alarms.
    编排完整的评估流程：加载特征 → 构建评估表 → 打印头部、警报、月度汇总、
    漏报大波动和假阳性。

    Args:
        task:        Task name from TASKS dict ("range_0dte", "otc_0dte", "c2c_1dte").
                     TASKS 字典中的任务名称。
        model:       Model name or None (uses task-matched default).
                     模型名称或 None（使用任务匹配的默认模型）。
        start/end:   Evaluation date range. / 评估日期范围。
        thresh:      Hit threshold (0.02 = actual > 2% counts as hit).
                     命中阈值（0.02 = 实际值 > 2% 计为命中）。
        threshold:   Signal threshold (prob >= threshold fires a signal).
                     信号阈值（概率 >= threshold 触发信号）。
        miss_thresh: Actual% above which non-signals count as missed (for FN report).
                     非信号日实际值超过此阈值计为漏报（用于假阴性报告）。
    """
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


# ── Structured data API for web UI / 面向 Web UI 的结构化数据 API ──

def _clean_float(v) -> float | None:
    """Replace NaN/Inf with None for JSON/template compatibility.
    将 NaN/Inf 替换为 None 以兼容 JSON 和模板。"""
    if v is None:
        return None
    try:
        if np.isnan(v) or np.isinf(v):
            return None
    except (TypeError, ValueError):
        pass
    return float(v)


def _row_to_dict(row, date) -> dict:
    """Convert a DataFrame row to a clean dict for templates.
    将 DataFrame 行转换为模板友好的字典。"""
    return {
        "date": str(date.date()),
        "prob": _clean_float(row["prob"]),
        "actual_pct": _clean_float(row["actual_pct"]),
        "range_pct": _clean_float(row.get("range_pct")),
        "c2c_pct": _clean_float(row.get("c2c_pct")),
        "o2c_pct": _clean_float(row.get("o2c_pct")),
        "gap_pct": _clean_float(row.get("gap_pct")),
        "max_dd": _clean_float(row.get("max_dd")),
        "max_ru": _clean_float(row.get("max_ru")),
        "rv20": _clean_float(row.get("rv20")),
        "vrp_20d": _clean_float(row.get("vrp_20d")),
        "events": format_events(row),
        "hit": bool(row["hit"]),
    }


def get_report_data(task: str, model: str | None = None,
                    start: str = "2023-01-01", end: str | None = None,
                    thresh: float = 0.02, threshold: float = 0.5,
                    miss_thresh: float = 3.0) -> dict:
    """Return structured evaluation data for web display.
    返回结构化评估数据，用于网页展示。

    Reuses the same core functions (load_features, build_eval_table,
    _summary_metrics) used by the CLI eval scripts, but returns
    dicts/lists instead of printing text.
    复用 CLI 评估脚本使用的核心函数，但返回字典/列表而非打印文本。

    Returns dict with keys: summary, alerts, monthly, missed, false_alarms.
    返回字典，键为：summary, alerts, monthly, missed, false_alarms。
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Choose from {list(TASKS)}")
    if model is None:
        model = TASKS[task]["default_model"]

    df = load_features()
    sig, model_file = build_eval_table(df, task, model, start, end, thresh)
    alerts_df = sig[sig["prob"] >= threshold]

    # Summary metrics
    tc = TASKS[task]
    m = _summary_metrics(sig, threshold)
    summary = {
        "task": task,
        "task_name": tc["name"],
        "model_file": model_file,
        "match": "task-matched" if model_file.startswith(tc["default_model"]) else "CROSS-TASK",
        "date_start": str(sig.index.min().date()),
        "date_end": str(sig.index.max().date()),
        "threshold": threshold,
        "thresh": thresh,
        "miss_thresh": miss_thresh,
        "n": m["n"],
        "base_rate": m["base_rate"],
        "alerts": m["alerts"],
        "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
        "precision": m["precision"],
        "recall": m["recall"],
        "auc": _clean_float(m["auc"]),
        "ap": _clean_float(m["ap"]),
        "lift": _clean_float(m["precision"] / max(m["base_rate"], 1e-9)),
    }

    # Alerts
    alerts_list = [_row_to_dict(r, d) for d, r in alerts_df.iterrows()]

    # Monthly summary
    if not alerts_df.empty:
        a = alerts_df.copy()
        a["month"] = a.index.to_period("M")
        mg = a.groupby("month").agg(
            signals=("prob", "count"),
            hits=("hit", "sum"),
            avg_prob=("prob", "mean"),
            avg_actual=("actual_pct", "mean"),
            max_actual=("actual_pct", "max"),
        ).reset_index()
        mg["hit_rate"] = mg["hits"] / mg["signals"]
        monthly_list = [
            {
                "month": str(r["month"]),
                "signals": int(r["signals"]),
                "hits": int(r["hits"]),
                "hit_rate": float(r["hit_rate"]),
                "avg_prob": float(r["avg_prob"]),
                "avg_actual": float(r["avg_actual"]),
                "max_actual": float(r["max_actual"]),
            }
            for _, r in mg.iterrows()
        ]
    else:
        monthly_list = []

    # Missed moves (false negatives for big moves)
    missed_df = sig[(sig["prob"] < threshold) & (sig["actual_pct"] > miss_thresh)]
    missed_list = [
        {
            "date": str(d.date()),
            "prob": _clean_float(r["prob"]),
            "actual_pct": _clean_float(r["actual_pct"]),
            "c2c_pct": _clean_float(r.get("c2c_pct")),
            "rv20": _clean_float(r.get("rv20")),
            "events": format_events(r),
        }
        for d, r in missed_df.iterrows()
    ]

    # False alarms (false positives)
    fp_df = alerts_df[alerts_df["actual_pct"] <= thresh * 100]
    false_alarms_list = [_row_to_dict(r, d) for d, r in fp_df.iterrows()]

    # Ground truth: actual big moves (TP+FN) + false alarms (FP), excluding TN
    ground_truth_list = []
    for d, r in sig.iterrows():
        signaled = bool(r["prob"] >= threshold)
        hit = bool(r["hit"])
        if not signaled and not hit:
            continue  # skip TN — quiet day, model silent
        row = _row_to_dict(r, d)
        row["signaled"] = signaled
        if signaled and hit:
            row["result"] = "TP"
        elif signaled and not hit:
            row["result"] = "FP"
        else:
            row["result"] = "FN"
        ground_truth_list.append(row)

    return {
        "summary": summary,
        "alerts": alerts_list,
        "monthly": monthly_list,
        "missed": missed_list,
        "false_alarms": false_alarms_list,
        "ground_truth": ground_truth_list,
    }


def get_cross_eval_matrix(start: str = "2023-01-01", end: str | None = None,
                          thresh: float = 0.02) -> dict:
    """Compute AUC for all model x task combinations (3x3 matrix).
    计算所有模型×任务组合的 AUC（3×3 矩阵）。

    Returns a dict with task names and the AUC matrix, showing that
    task-matched models outperform cross-task models.
    返回包含任务名称和 AUC 矩阵的字典，展示任务匹配模型优于跨任务模型。
    """
    df = load_features()
    task_keys = list(TASKS.keys())
    matrix = {}

    for model_task in task_keys:
        model_name = TASKS[model_task]["default_model"]
        try:
            resolve_model_path(model_name)
        except FileNotFoundError:
            continue

        row = {}
        for eval_task in task_keys:
            try:
                sig, _ = build_eval_table(df, eval_task, model_name, start, end, thresh)
                y_true = sig["hit"].astype(int)
                y_prob = sig["prob"]
                auc = roc_auc_score(y_true, y_prob) if y_true.nunique() == 2 else None
                row[eval_task] = _clean_float(auc)
            except Exception:
                row[eval_task] = None

        matrix[model_task] = row

    return {
        "task_keys": task_keys,
        "task_names": {t: TASKS[t]["name"] for t in task_keys},
        "model_names": {t: TASKS[t]["default_model"] for t in task_keys},
        "matrix": matrix,
    }
