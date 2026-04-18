"""Evaluate predictions for the 1DTE next-day |C2C| > 2% task.
评估 1DTE 次日收盘对收盘绝对收益 |C2C| > 2% 任务的预测表现。

This is the only 1DTE (next-day) target. Unlike 0DTE models that predict
and verify on the same day, this model predicts TODAY but the ground truth
is TOMORROW's |close-to-close| return.
这是唯一的 1DTE（次日）目标。与当天预测当天验证的 0DTE 模型不同，
此模型在今天预测，但真实标签是明天的 |收盘对收盘| 收益率。

The shift=-1 mechanism: tomorrow's |C2C| value is shifted into today's
training row, so the model learns to predict tomorrow using today's features.
shift=-1 机制：明天的 |C2C| 值被移入今天的训练行，
使模型学会用今天的特征预测明天的波动。

Trading application / 交易应用:
  - Signal today → buy tomorrow's expiry options at today's close or next open.
    今天发信号 → 在今天收盘或明天开盘买入明天到期的期权。
  - Useful for overnight straddles or 1DTE options positioning.
    适用于隔夜跨式或 1DTE 期权布局。

Default model: c2c_1dte_2pct_model (task-matched, AUC ~0.73).
默认模型：c2c_1dte_2pct_model（任务匹配，AUC ~0.73）。
Pass --model to cross-evaluate any other trained model on this same target.
传入 --model 可用其他模型在相同目标上进行交叉评估。

Usage / 用法:
    python3 -m eval.eval_c2c_1dte
    python3 -m eval.eval_c2c_1dte --start 2025-01-01 --threshold 0.5
    python3 -m eval.eval_c2c_1dte --model range_0dte_2pct_model   # cross-task test / 交叉评估
"""
import argparse
from eval._common import run_report, TASKS


TASK = "c2c_1dte"


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
