"""Tests for config module."""
from qqq_trading.config import Config, load_config, ModelConfig, SplitConfig


def test_default_config():
    config = Config()
    assert config.splits.train_end == "2019-12-31"
    assert config.splits.test_start == "2023-01-01"
    assert config.random_state == 42
    assert 0.01 in config.move_thresholds
    assert 0.5 in config.confidence_thresholds


def test_model_presets():
    config = Config()
    assert config.model.base.n_estimators == 300
    assert config.model.base.learning_rate == 0.05
    assert config.model.production.n_estimators == 500
    assert config.model.production.learning_rate == 0.03


def test_split_config():
    sc = SplitConfig()
    assert sc.train_end < sc.val_start
    assert sc.val_end < sc.test_start


def test_model_config_defaults():
    mc = ModelConfig()
    assert mc.max_depth == 5
    assert mc.subsample == 0.8
    assert mc.reg_alpha == 0.1
