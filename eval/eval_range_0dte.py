"""Evaluate predictions for the 0DTE Range (H-L)/O > 2% task.

Default model: range_0dte_2pct_model (task-matched).
Pass --model to cross-evaluate any other trained model on this same target.

Usage:
    python3 -m eval.eval_range_0dte
    python3 -m eval.eval_range_0dte --start 2025-01-01 --threshold 0.6
    python3 -m eval.eval_range_0dte --model c2c_1dte_2pct_model   # cross-task test
"""
import argparse
from eval._common import run_report, TASKS


TASK = "range_0dte"


def main():
    p = argparse.ArgumentParser(description=f"Eval {TASKS[TASK]['name']}")
    p.add_argument("--model", type=str, default=None,
                   help="Model name (without .joblib). Default: task-matched model.")
    p.add_argument("--start", type=str, default="2023-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--thresh", type=float, default=0.02,
                   help="Hit threshold for the true target (default 0.02 = 2%%)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Model probability threshold to fire a signal (default 0.5)")
    p.add_argument("--miss-thresh", type=float, default=3.0,
                   help="Actual%% above which non-signals count as MISSED (default 3.0)")
    args = p.parse_args()

    run_report(task=TASK, model=args.model,
               start=args.start, end=args.end,
               thresh=args.thresh, threshold=args.threshold,
               miss_thresh=args.miss_thresh)


if __name__ == "__main__":
    main()
