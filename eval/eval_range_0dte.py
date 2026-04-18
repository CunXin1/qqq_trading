"""Evaluate predictions for the 0DTE Range (H-L)/O > 2% task.
评估 0DTE 日内振幅 (H-L)/O > 2% 任务的预测表现。

This is the most important eval script for straddle strategies:
这是跨式期权策略最重要的评估脚本：
  - The target (intraday range) directly determines straddle profitability.
    目标（日内振幅）直接决定跨式期权的盈利能力。
  - High range = straddle profits; low range = straddle loses premium.
    高振幅 = 跨式盈利；低振幅 = 跨式亏损权利金。

Default model: range_0dte_2pct_model (task-matched, AUC ~0.82).
默认模型：range_0dte_2pct_model（任务匹配，AUC ~0.82）。
Pass --model to cross-evaluate any other trained model on this same target.
传入 --model 可用其他模型在相同目标上进行交叉评估。

Key parameters for trading / 交易关键参数:
  --threshold 0.7~0.8  Recommended for straddles (higher precision, fewer false alarms).
                        跨式策略推荐（更高精确率，更少假阳性）。
  --threshold 0.5       Default, balanced precision/recall.
                        默认值，精确率/召回率均衡。

Usage / 用法:
    python3 -m eval.eval_range_0dte
    python3 -m eval.eval_range_0dte --start 2025-01-01 --threshold 0.6
    python3 -m eval.eval_range_0dte --model c2c_1dte_2pct_model   # cross-task test / 交叉评估
"""
import argparse
from eval._common import run_report, TASKS


TASK = "range_0dte"


def main():
    p = argparse.ArgumentParser(description=f"Eval {TASKS[TASK]['name']}")
    p.add_argument("--model", type=str, default=None,
                   help="Model name (without .joblib). Default: task-matched model."
                        " / 模型名称（不含 .joblib）。默认：任务匹配模型。")
    p.add_argument("--start", type=str, default="2023-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--thresh", type=float, default=0.02,
                   help="Hit threshold for the true target (default 0.02 = 2%%)"
                        " / 真实目标的命中阈值（默认 0.02 = 2%%）")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Model probability threshold to fire a signal (default 0.5)"
                        " / 触发信号的模型概率阈值（默认 0.5）")
    p.add_argument("--miss-thresh", type=float, default=3.0,
                   help="Actual%% above which non-signals count as MISSED (default 3.0)"
                        " / 非信号日实际值超此阈值计为漏报（默认 3.0）")
    args = p.parse_args()

    run_report(task=TASK, model=args.model,
               start=args.start, end=args.end,
               thresh=args.thresh, threshold=args.threshold,
               miss_thresh=args.miss_thresh)


if __name__ == "__main__":
    main()
