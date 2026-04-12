"""Tests for CLI predict module."""
from qqq_trading.cli.predict import parse_args, format_text, format_json
from qqq_trading.models.prediction import PredictionResult


def test_parse_args_defaults():
    args = parse_args([])
    assert args.model == "interaction"
    assert args.mode == "both"
    assert args.threshold == 0.5
    assert args.format == "text"


def test_parse_args_custom():
    args = parse_args(["--model", "base", "--threshold", "0.7", "--format", "json"])
    assert args.model == "base"
    assert args.threshold == 0.7
    assert args.format == "json"


def test_format_text():
    results = [
        PredictionResult(
            date="2026-04-10", target="1DTE |C2C|>2%",
            probability=0.65, signal="ELEVATED",
            threshold=0.5, model_name="interaction_model",
        )
    ]
    import argparse
    args = argparse.Namespace(model="interaction", threshold=0.5)
    text = format_text(results, args)
    assert "0.650" in text
    assert "ELEVATED" in text


def test_format_json():
    results = [
        PredictionResult(
            date="2026-04-10", target="1DTE |C2C|>2%",
            probability=0.65, signal="ELEVATED",
            threshold=0.5, model_name="interaction_model",
        )
    ]
    import argparse
    args = argparse.Namespace(model="interaction", threshold=0.5)
    output = format_json(results, args)
    import json
    data = json.loads(output)
    assert data["predictions"][0]["probability"] == 0.65
    assert data["predictions"][0]["signal"] == "ELEVATED"
