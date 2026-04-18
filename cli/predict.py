"""CLI entry point for daily QQQ volatility predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="QQQ intraday volatility prediction for 0DTE/1DTE straddles"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Date to predict for (YYYY-MM-DD). Default: latest available."
    )
    parser.add_argument(
        "--model", type=str, default="interaction",
        choices=["interaction", "base"],
        help="Model to use (default: interaction)"
    )
    parser.add_argument(
        "--mode", type=str, default="both",
        choices=["1dte", "0dte", "both"],
        help="Prediction mode (default: both)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--format", type=str, default="text",
        choices=["text", "json"],
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--refresh-external", action="store_true",
        help="Force re-download of external data"
    )
    return parser.parse_args(argv)


def run_prediction(args):
    """Execute prediction pipeline."""
    import pandas as pd
    from utils.paths import OUTPUT_DIR, MODEL_DIR
    from models.prediction import predict

    # Load daily metrics
    daily_path = OUTPUT_DIR / "daily_metrics.parquet"
    if not daily_path.exists():
        print("Error: daily_metrics.parquet not found. Run the pipeline first.", file=sys.stderr)
        sys.exit(1)

    daily = pd.read_parquet(daily_path)
    daily.index = pd.to_datetime(daily.index)

    # Load external data
    ext_path = OUTPUT_DIR / "external_data.parquet"
    if args.refresh_external or not ext_path.exists():
        from data.external_data import download_external_data
        ext = download_external_data(ext_path)
    else:
        ext = pd.read_parquet(ext_path)
        ext.index = pd.to_datetime(ext.index)

    # Select model
    model_map = {
        "interaction": MODEL_DIR / "interaction_2000_2019.joblib",
        "base": MODEL_DIR / "next_day_2pct_2000_2019.joblib",
    }
    model_path = model_map[args.model]
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    # Define targets
    targets = []
    if args.mode in ("1dte", "both"):
        targets.append(("1DTE |C2C|>2%", "> 2% tomorrow"))
    if args.mode in ("0dte", "both"):
        targets.append(("0DTE Range>2%", "Range > 2% today"))

    results = []
    for target_name, description in targets:
        result = predict(
            model_path=model_path,
            daily_metrics=daily,
            external_data=ext,
            threshold=args.threshold,
            target_name=target_name,
        )
        results.append(result)

    return results


def format_text(results, args):
    """Format results as human-readable text."""
    lines = []
    lines.append(f"QQQ Volatility Prediction")
    lines.append(f"Model: {args.model} | Threshold: {args.threshold}")
    lines.append("=" * 50)

    for r in results:
        lines.append(f"\n{r.target}:")
        lines.append(f"  Date:        {r.date}")
        lines.append(f"  Probability: {r.probability:.3f}")
        lines.append(f"  Signal:      {r.signal}")

    return "\n".join(lines)


def format_json(results, args):
    """Format results as JSON."""
    output = {
        "model": args.model,
        "threshold": args.threshold,
        "predictions": [
            {
                "date": r.date,
                "target": r.target,
                "probability": round(r.probability, 4),
                "signal": r.signal,
            }
            for r in results
        ],
    }
    return json.dumps(output, indent=2)


def main(argv=None):
    args = parse_args(argv)
    results = run_prediction(args)

    if args.format == "json":
        print(format_json(results, args))
    else:
        print(format_text(results, args))


if __name__ == "__main__":
    main()
