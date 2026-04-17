"""Evaluate predictions for the 1DTE next-day |C2C| > 2% task.

Default model: c2c_1dte_2pct_model (task-matched).
Pass --model to cross-evaluate any other trained model on this same target.

Usage:
    python3 -m eval.eval_c2c_1dte
    python3 -m eval.eval_c2c_1dte --start 2025-01-01 --threshold 0.5
    python3 -m eval.eval_c2c_1dte --model range_0dte_2pct_model   # cross-task test
"""
import argparse
from eval._common import run_report, TASKS


TASK = "c2c_1dte"


def main():
    p = argparse.ArgumentParser(description=f"Eval {TASKS[TASK]['name']}")
    p.add_argument("--model", type=str, default=None,
                   help="Model name (without .joblib). Default: task-matched model.")
    p.add_argument("--start", type=str, default="2023-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--thresh", type=float, default=0.02)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--miss-thresh", type=float, default=3.0)
    args = p.parse_args()

    run_report(task=TASK, model=args.model,
               start=args.start, end=args.end,
               thresh=args.thresh, threshold=args.threshold,
               miss_thresh=args.miss_thresh)


if __name__ == "__main__":
    main()
