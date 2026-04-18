"""Evaluate predictions for the 0DTE |O2C| > 2% task.
评估 0DTE 开盘到收盘绝对收益 |O2C| > 2% 任务的预测表现。

This target measures the absolute directional move from open to close,
ignoring the overnight gap. It captures pure intraday directional momentum.
此目标衡量从开盘到收盘的绝对方向性波动，忽略隔夜跳空。
它捕捉纯粹的日内方向性动量。

Unlike range (which counts both up and down swings), |O2C| requires
a sustained directional move — useful for directional option plays
(calls or puts) rather than straddles.
与振幅（计算上下两方向摆幅）不同，|O2C| 需要持续的方向性波动——
适用于方向性期权策略（看涨或看跌）而非跨式。

Default model: otc_0dte_2pct_model (task-matched).
默认模型：otc_0dte_2pct_model（任务匹配）。
Pass --model to cross-evaluate any other trained model on this same target.
传入 --model 可用其他模型在相同目标上进行交叉评估。

Note: This target has the lowest base rate (~4.7%), making precision
especially important — many false alarms at low thresholds.
注意：此目标的基准率最低（约 4.7%），因此精确率尤为重要——
低阈值时会产生大量假阳性。

Usage / 用法:
    python3 -m eval.eval_otc_0dte
    python3 -m eval.eval_otc_0dte --start 2025-01-01 --threshold 0.4
    python3 -m eval.eval_otc_0dte --model range_0dte_2pct_model   # cross-task test / 交叉评估
"""
import argparse
from eval._common import run_report, TASKS


TASK = "otc_0dte"


def main():
    p = argparse.ArgumentParser(description=f"Eval {TASKS[TASK]['name']}")
    p.add_argument("--model", type=str, default=None,
                   help="Model name (without .joblib). Default: task-matched model."
                        " / 模型名称（不含 .joblib）。默认：任务匹配模型。")
    p.add_argument("--start", type=str, default="2023-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--thresh", type=float, default=0.02,
                   help="Hit threshold (default 0.02 = 2%%) / 命中阈值（默认 0.02 = 2%%）")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Signal threshold (default 0.5) / 信号阈值（默认 0.5）")
    p.add_argument("--miss-thresh", type=float, default=3.0,
                   help="Missed-move threshold (default 3.0%%) / 漏报阈值（默认 3.0%%）")
    args = p.parse_args()

    run_report(task=TASK, model=args.model,
               start=args.start, end=args.end,
               thresh=args.thresh, threshold=args.threshold,
               miss_thresh=args.miss_thresh)


if __name__ == "__main__":
    main()
