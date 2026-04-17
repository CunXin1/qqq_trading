"""Tests for event calendar module."""
import pandas as pd
from data.event_calendar import (
    load_fomc_dates, compute_nfp_dates, _compute_eve_dates,
    compute_days_to_event, compute_earnings_season,
)


def test_fomc_dates_loaded():
    dates = load_fomc_dates()
    assert len(dates) > 200
    assert dates[0].year == 2000
    assert dates[-1].year == 2026


def test_fomc_dates_per_year():
    dates = load_fomc_dates()
    for year in range(2002, 2020):
        year_dates = dates[dates.year == year]
        assert 7 <= len(year_dates) <= 10, f"Year {year}: {len(year_dates)} meetings"


def test_nfp_dates():
    dates = compute_nfp_dates(2020, 2020)
    assert len(dates) == 12
    for d in dates:
        assert d.weekday() == 4, f"NFP {d} is not a Friday"
        assert d.day <= 7, f"NFP {d} is not first Friday"


def test_eve_dates_skip_weekends():
    dates = pd.DatetimeIndex([pd.Timestamp("2023-01-30")])  # Monday
    eves = _compute_eve_dates(dates)
    assert eves[0].weekday() == 4  # Friday

    dates2 = pd.DatetimeIndex([pd.Timestamp("2023-01-31")])  # Tuesday
    eves2 = _compute_eve_dates(dates2)
    assert eves2[0].weekday() == 0  # Monday


def test_days_to_event():
    index = pd.DatetimeIndex([
        pd.Timestamp("2023-01-28"),
        pd.Timestamp("2023-01-29"),
        pd.Timestamp("2023-02-01"),
    ])
    events = pd.DatetimeIndex([pd.Timestamp("2023-02-01")])
    days = compute_days_to_event(index, events)
    assert days[0] == 4
    assert days[1] == 3
    assert days[2] == 0


def test_earnings_season():
    dates = pd.bdate_range("2023-01-01", "2023-12-31")
    flags = compute_earnings_season(dates)
    assert flags.sum() > 0
    # January 20-31 should be earnings season
    jan_late = flags.loc["2023-01-20":"2023-01-31"]
    assert jan_late.sum() > 0
    # March should not be earnings season
    mar = flags.loc["2023-03-15":"2023-03-31"]
    assert mar.sum() == 0
