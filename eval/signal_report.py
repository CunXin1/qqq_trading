"""
Generate detailed signal report: all alerts with OHLCV, events, and missed moves.
生成详细信号报告：所有警报的OHLCV数据、事件标记和漏报的大波动。

This script focuses on the "trading view" — it shows every day the model
triggered a signal, what happened that day, and critically, which big moves
the model missed entirely.
本脚本聚焦"交易视角"——展示模型触发信号的每一天发生了什么，
以及关键地，模型完全漏报了哪些大波动。

Three sections in the output / 输出包含三个部分:
    1. ALERTS: Every signal day with full market context
       警报列表：每个信号日的完整市场上下文
    2. MONTHLY SUMMARY: Aggregated by month (signals, hits, avg range)
       月度汇总：按月聚合（信号数、命中数、平均范围）
    3. MISSED BIG MOVES: Days with large range but no signal (FN analysis)
       漏报大波动：有大范围波动但无信号的日子（假阴分析）

Usage / 用法:
    python eval/signal_report.py                          # default: prob >= 0.5 / 默认阈值0.5
    python eval/signal_report.py --threshold 0.6          # higher confidence / 更高置信度
    python eval/signal_report.py --threshold 0.3 --csv    # export to CSV / 导出CSV
    python eval/signal_report.py --miss-range 4.0         # only flag misses > 4% / 仅标记>4%的漏报

Arguments / 参数:
    --threshold   Minimum probability to count as signal (default: 0.5)
                  计为信号的最低概率（默认：0.5）
    --test-start  Start of test period (default: 2023-01-01)
                  测试期起始日期（默认：2023-01-01）
    --miss-range  Range% threshold for "missed big move" (default: 3.0)
                  "漏报大波动"的范围%阈值（默认：3.0）
    --csv         Export alerts to eval/signals.csv
                  将警报导出到 eval/signals.csv

Context columns in alert table / 警报表的上下文列:
    RV20%  - 20-day realized vol (vol regime indicator) / 20日已实现波动率（波动率状态指标）
    VRP    - Volatility risk premium (fear gauge) / 波动率风险溢价（恐惧度量）
    Events - FOMC, NFP, FOMC-1 (eve), NFP-1 (eve), EARN (earnings season)
             FOMC、NFP、FOMC-1（前夜）、NFP-1（前夜）、EARN（财报季）
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
from utils.paths import OUTPUT_DIR, MODEL_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Signal report with OHLCV detail")
    parser.add_argument("--threshold", type=float, default=0.5, help="Min probability (default: 0.5)")
    parser.add_argument("--test-start", type=str, default="2023-01-01", help="Test period start")
    parser.add_argument("--miss-range", type=float, default=3.0, help="Range%% to count as missed big move (default: 3.0)")
    parser.add_argument("--csv", action="store_true", help="Export alerts to eval/signals.csv")
    return parser.parse_args()


def load_and_build_features():
    """Load data and build full feature set.
    加载数据并构建完整特征集。"""
    daily = pd.read_parquet(OUTPUT_DIR / "daily_metrics.parquet")
    daily.index = pd.to_datetime(daily.index)
    ext = pd.read_parquet(OUTPUT_DIR / "external_data.parquet")
    ext.index = pd.to_datetime(ext.index)

    df = engineer_base_features(daily)
    df = engineer_all_external(df, ext)
    df = build_interaction_features(df)
    return df


def build_signal_table(df, model, feature_cols, test_start):
    """Build DataFrame with predictions and market data for test period.
    构建测试期内预测结果和市场数据的DataFrame。"""
    available = [f for f in feature_cols if f in df.columns]
    test = df.loc[test_start:]
    y_proba = model.predict_proba(test[available].values)[:, 1]

    signals = pd.DataFrame({
        "prob": y_proba,
        "open": test["reg_open"].values,
        "high": test["reg_high"].values,
        "low": test["reg_low"].values,
        "close": test["reg_close"].values,
        "range_pct": (test["intraday_range"] * 100).values,    # intraday range in % / 日内范围%
        "c2c_ret": (test["close_to_close_ret"] * 100).values,  # close-to-close return % / 收盘收益%
        "o2c_ret": (test["open_to_close_ret"] * 100).values,   # open-to-close return % / 开收盘收益%
        "gap_pct": (test["gap_return"] * 100).values,           # overnight gap % / 隔夜缺口%
        "max_dd": (test["max_drawdown"] * 100).values,          # max intraday drawdown % / 日内最大回撤%
        "max_ru": (test["max_runup"] * 100).values,             # max intraday runup % / 日内最大涨幅%
    }, index=test.index)

    # Event flags — used to understand WHY a signal was generated
    # 事件标记——用于理解信号生成的原因
    for col in ["is_fomc_day", "is_nfp_day", "is_earnings_season", "fomc_imminent", "nfp_imminent"]:
        if col in test.columns:
            signals[col] = test[col].values

    # VRP and realized vol — key regime indicators
    # VRP和已实现波动率——关键状态指标
    if "vrp_20d" in test.columns:
        signals["vrp_20d"] = test["vrp_20d"].values
    if "realized_vol_20d" in test.columns:
        signals["rv20"] = (test["realized_vol_20d"] * 100).values

    # Signal classification and hit determination / 信号分类和命中判定
    signals["signal"] = "NONE"
    signals.loc[signals["prob"] >= 0.5, "signal"] = "ELEV"
    signals.loc[signals["prob"] >= 0.7, "signal"] = "HIGH"
    signals["hit"] = signals["range_pct"] > 2.0  # hit = intraday range > 2% / 命中 = 日内范围 > 2%

    return signals


def format_events(row):
    """Build human-readable event string from flag columns.
    从标记列构建人类可读的事件字符串。"""
    events = []
    if row.get("is_fomc_day", 0) == 1:
        events.append("FOMC")          # FOMC announcement day / FOMC公布日
    if row.get("is_nfp_day", 0) == 1:
        events.append("NFP")           # Non-Farm Payrolls day / 非农公布日
    if row.get("fomc_imminent", 0) == 1 and row.get("is_fomc_day", 0) != 1:
        events.append("FOMC-1")        # Day before FOMC / FOMC前一天
    if row.get("nfp_imminent", 0) == 1 and row.get("is_nfp_day", 0) != 1:
        events.append("NFP-1")         # Day before NFP / NFP前一天
    if row.get("is_earnings_season", 0) == 1:
        events.append("EARN")          # Earnings season / 财报季
    return ",".join(events) if events else "-"


def print_alerts(alerts):
    """Print detailed alert table with OHLCV, vol metrics, events, and hit status.
    打印详细警报表，包含OHLCV、波动率指标、事件和命中状态。"""
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
    """Print monthly aggregation of signal performance.
    打印信号性能的月度聚合。"""
    alerts = alerts.copy()
    alerts["month"] = alerts.index.to_period("M")
    monthly = alerts.groupby("month").agg(
        signals=("prob", "count"),       # number of signals this month / 本月信号数
        hits=("hit", "sum"),             # number of hits / 命中数
        avg_prob=("prob", "mean"),       # average model probability / 平均模型概率
        avg_range=("range_pct", "mean"), # average range on signal days / 信号日平均范围
        max_range=("range_pct", "max"),  # largest range on signal days / 信号日最大范围
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
    """Print days with big moves but no signal — the model's blind spots.
    打印有大波动但无信号的日子——模型的盲区。

    These are FALSE NEGATIVES: real big moves the model failed to predict.
    Understanding why helps improve the model.
    这些是假阴性：模型未能预测的真实大波动。理解原因有助于改进模型。
    """
    missed = signals[(signals["prob"] < threshold) & (signals["range_pct"] > miss_range)]
    print(f"MISSED BIG MOVES (prob < {threshold}, range > {miss_range}%): {len(missed)} days")
    print(f"漏报大波动（概率 < {threshold}，范围 > {miss_range}%）：{len(missed)} 天")

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

    # ── Summary / 汇总 ──
    print()
    print("=" * 100)
    print(f"SIGNAL REPORT  |  Test: {args.test_start} to {signals.index[-1].date()}  |  "
          f"{test_days} days  |  Threshold: {args.threshold}")
    print("=" * 100)
    print(f"Alerts: {len(alerts)}  |  Hits (range>2%): {n_hits}  |  "
          f"Hit rate: {alerts['hit'].mean():.1%}  |  Base rate: {signals['hit'].mean():.1%}  |  "
          f"Lift: {alerts['hit'].mean() / signals['hit'].mean():.1f}x")
    print()

    # ── Section 1: All alerts / 第一部分：所有警报 ──
    print_alerts(alerts)

    # ── Section 2: Monthly summary / 第二部分：月度汇总 ──
    print()
    print("=" * 100)
    print("MONTHLY SUMMARY / 月度汇总")
    print("=" * 100)
    print_monthly_summary(alerts)

    # ── Section 3: Missed moves / 第三部分：漏报 ──
    print()
    print("=" * 100)
    print_missed_moves(signals, args.threshold, args.miss_range)

    # ── CSV export / CSV导出 ──
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
