"""Configuration management — dataclass with optional YAML override."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class SplitConfig:
    train_end: str = "2019-12-31"
    val_start: str = "2020-01-01"
    val_end: str = "2022-12-31"
    test_start: str = "2023-01-01"


@dataclass
class ModelConfig:
    n_estimators: int = 500
    max_depth: int = 5
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.7
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0


@dataclass
class ModelPresets:
    base: ModelConfig = field(default_factory=lambda: ModelConfig(
        n_estimators=300, learning_rate=0.05, colsample_bytree=0.8,
        reg_alpha=0.0,
    ))
    production: ModelConfig = field(default_factory=ModelConfig)


@dataclass
class PredictionConfig:
    default_model: str = "interaction"
    confidence_threshold: float = 0.5


@dataclass
class Config:
    splits: SplitConfig = field(default_factory=SplitConfig)
    model: ModelPresets = field(default_factory=ModelPresets)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    move_thresholds: list[float] = field(
        default_factory=lambda: [0.01, 0.02, 0.03, 0.05]
    )
    confidence_thresholds: list[float] = field(
        default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    )
    random_state: int = 42


def _merge_dict(target: dict, source: dict) -> dict:
    """Recursively merge source into target."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _merge_dict(target[key], value)
        else:
            target[key] = value
    return target


def load_config(path: Optional[Path] = None) -> Config:
    """Load config with optional YAML override.

    If no path is given, looks for config/default.yaml relative to project root.
    """
    from qqq_trading.utils.paths import PROJECT_ROOT

    config = Config()

    if path is None:
        path = PROJECT_ROOT / "config" / "default.yaml"

    if path.exists():
        with open(path) as f:
            overrides = yaml.safe_load(f) or {}

        if "splits" in overrides:
            for k, v in overrides["splits"].items():
                if hasattr(config.splits, k):
                    setattr(config.splits, k, v)

        if "model" in overrides:
            for preset_name in ("base", "production"):
                if preset_name in overrides["model"]:
                    preset = getattr(config.model, preset_name)
                    for k, v in overrides["model"][preset_name].items():
                        if hasattr(preset, k):
                            setattr(preset, k, v)

        if "prediction" in overrides:
            for k, v in overrides["prediction"].items():
                if hasattr(config.prediction, k):
                    setattr(config.prediction, k, v)

        if "random_state" in overrides:
            config.random_state = overrides["random_state"]

        if "thresholds" in overrides:
            if "move_pcts" in overrides["thresholds"]:
                config.move_thresholds = overrides["thresholds"]["move_pcts"]
            if "confidence" in overrides["thresholds"]:
                config.confidence_thresholds = overrides["thresholds"]["confidence"]

    return config
