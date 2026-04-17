"""
Compare model predictions vs actual daily moves for every trading day.
每日预测 vs 实际波动逐日对比。

Shows probability, signal level, and actual market data side by side.
将模型概率、信号级别和实际市场数据并排展示。

Highlights hits/misses (TP/FP/FN/TN) and computes rolling accuracy.
标记命中/失误（真阳/假阳/假阴/真阴）并计算滚动准确率。

This is the most granular evaluation tool — it lets you see EVERY day
and understand exactly when/why the model was right or wrong.
这是最细粒度的评估工具——可以看到每一天的详情，
精确理解模型在何时/为何正确或犯错。

Usage / 用法:
    python eval/daily_compare.py                              # all test days / 所有测试日
    python eval/daily_compare.py --threshold 0.5              # only signal days / 仅信号日
    python eval/daily_compare.py --start 2025-03-01 --end 2025-04-30  # date range / 日期范围
    python eval/daily_compare.py --threshold 0.3 --csv        # export to CSV / 导出CSV

Arguments / 参数:
    --start      Start date for evaluation (default: 2023-01-01)
                 评估起始日期（默认：2023-01-01）
    --end        End date for evaluation (default: latest available)
                 评估截止日期（默认：最新可用）
    --threshold  Only show days with prob >= threshold; if omitted, show all days
                 仅展示概率>=阈值的日子；省略则展示所有日子
    --target     Target to evaluate: range2/range3/c2c1/c2c2/c2c3 (default: range2)
                 评估目标：range2=日内范围>2%, c2c2=|收盘收益|>2% 等
    --csv        Export results to eval/daily_compare.csv
                 将结果导出到 eval/daily_compare.csv

Output columns / 输出列说明:
    Prob    - Model probability (0.0-1.0) / 模型概率
    Sig     - Signal level: LOW(0.3+), ELEV(0.5+), HIGH(0.7+) / 信号级别
    Range%  - Intraday high-low range as % / 日内最高-最低范围百分比
    C2C%    - Close-to-close return / 收盘到收盘收益率
    O2C%    - Open-to-close return / 开盘到收盘收益率
    Gap%    - Overnight gap return / 隔夜缺口收益率
    MaxDD%  - Maximum intraday drawdown / 日内最大回撤
    MaxRU%  - Maximum intraday runup / 日内最大涨幅
    RV20%   - 20-day realized volatility / 20日已实现波动率
    Actual  - BIG if actual move exceeded target, else - / 实际是否超过目标
    Result  - TP/FP/FN/TN classification / 真阳/假阳/假阴/真阴分类
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


# Target configurations / 目标配置
# - name: display name / 显示名称
# - col: source column in daily_metrics / daily_metrics中的源列
# - thresh: move threshold (e.g. 0.02 = 2%) / 波动阈值
# - shift: 0 for same-day (0DTE), -1 for next-day (1DTE) / 0=当日, -1=次日
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

    # ── Load data & build features / 加载数据和构建特征 ──
    print("Loading data and building features...")
    daily = pd.read_parquet(OUTPUT_DIR / "daily_metrics.parquet")
    daily.index = pd.to_datetime(daily.index)
    ext = pd.read_parquet(OUTPUT_DIR / "external_data.parquet")
    ext.index = pd.to_datetime(ext.index)

    df = engineer_base_features(daily)
    df = engineer_all_external(df, ext)
    df = build_interaction_features(df)

    # Load trained model / 加载训练好的模型
    model, feature_cols = load_model(MODEL_DIR / "interaction_model.joblib")
    available = [f for f in feature_cols if f in df.columns]

    # ── Slice date range / 截取日期范围 ──
    subset = df.loc[args.start:args.end] if args.end else df.loc[args.start:]
    X = subset[available].values
    y_proba = model.predict_proba(X)[:, 1]

    # ── Build comparison table / 构建对比表 ──
    comp = pd.DataFrame(index=subset.index)
    comp["prob"] = y_proba

    # Signal level classification / 信号级别分类
    # LOW: 0.3+ (some signal) / 有一定信号
    # ELEV: 0.5+ (elevated, worth attention) / 升高，值得关注
    # HIGH: 0.7+ (high conviction) / 高确信
    comp["signal"] = ""
    comp.loc[comp["prob"] >= 0.3, "signal"] = "LOW"
    comp.loc[comp["prob"] >= 0.5, "signal"] = "ELEV"
    comp.loc[comp["prob"] >= 0.7, "signal"] = "HIGH"

    # Actual market data / 实际市场数据
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

    # Actual target: did the move exceed the threshold? / 实际目标：波动是否超过阈值？
    # shift=0 for 0DTE (same day), shift=-1 for 1DTE (next day's actual vs today's prediction)
    # shift=0用于0DTE（当天），shift=-1用于1DTE（用今天的预测对比明天的实际）
    if tc["shift"] == 0:
        comp["actual"] = (subset[tc["col"]] > tc["thresh"]).astype(int)
    else:
        comp["actual"] = (subset[tc["col"]].shift(tc["shift"]) > tc["thresh"]).astype(int)

    # Confusion matrix classification / 混淆矩阵分类
    # TP (True Positive):  signal fired + actual big move = correct alert / 发信号+实际大波动=正确警报
    # FP (False Positive): signal fired + no big move = false alarm / 发信号+无大波动=虚假警报
    # FN (False Negative): no signal + actual big move = missed move / 无信号+实际大波动=漏报
    # TN (True Negative):  no signal + no big move = correct silence / 无信号+无大波动=正确静默
    def judge(row, threshold):
        predicted = row["prob"] >= threshold
        actual = row["actual"] == 1
        if predicted and actual:
            return "TP"
        elif predicted and not actual:
            return "FP"
        elif not predicted and actual:
            return "FN"
        else:
            return "TN"

    eval_thresh = args.threshold if args.threshold else 0.5
    comp["result"] = comp.apply(lambda r: judge(r, eval_thresh), axis=1)

    # ── Filter if threshold specified / 如指定阈值则过滤 ──
    if args.threshold is not None:
        display = comp[comp["prob"] >= args.threshold].copy()
    else:
        display = comp.copy()

    # ── Print summary statistics / 打印汇总统计 ──
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

    # ── Print daily detail table / 打印逐日详情表 ──
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

    # ── Rolling hit rate for signal days / 信号日的滚动命中率 ──
    # Shows how accuracy evolves over time — useful for detecting model degradation
    # 展示准确率如何随时间变化——有助于检测模型退化
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

    # ── CSV export / CSV导出 ──
    if args.csv:
        csv_path = Path(__file__).parent / "daily_compare.csv"
        display.to_csv(csv_path, float_format="%.4f")
        print(f"\nExported {len(display)} rows to {csv_path}")


if __name__ == "__main__":
    main()
