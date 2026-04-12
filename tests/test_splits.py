"""Tests for time-series splitting utilities."""
import pandas as pd
from qqq_trading.utils.splits import date_split, walk_forward_splits


def test_date_split_no_overlap():
    dates = pd.bdate_range("2018-01-01", "2024-12-31")
    df = pd.DataFrame({"x": range(len(dates))}, index=dates)

    splits = date_split(df)
    train_end = splits["train"].index.max()
    val_start = splits["val"].index.min()
    test_start = splits["test"].index.min()

    assert train_end < val_start
    assert splits["val"].index.max() < test_start


def test_date_split_covers_all():
    dates = pd.bdate_range("2018-01-01", "2024-12-31")
    df = pd.DataFrame({"x": range(len(dates))}, index=dates)

    splits = date_split(df)
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == len(df)


def test_walk_forward_splits():
    dates = pd.bdate_range("2005-01-01", "2020-12-31")
    df = pd.DataFrame({"x": range(len(dates))}, index=dates)

    wf = walk_forward_splits(df, test_years=[2015, 2016, 2017], train_window_years=5)
    assert len(wf) == 3
    for split in wf:
        assert "train" in split
        assert "test" in split
        assert "year" in split
        # Train ends before test starts
        assert split["train"].index.max() < split["test"].index.min()
