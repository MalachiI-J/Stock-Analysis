from __future__ import annotations

import pandas as pd

from stock_scrapper.market_calendar.nyse_calendar import get_nyse_calendar
from stock_scrapper.market_calendar.session_resolver import SessionResolver


def test_nyse_calendar_accepts_dates_older_than_the_rolling_20_year_default() -> None:
    """Regression test for the 2026-08-03 production incident: exchange_calendars'
    un-bounded ``get_calendar("XNYS")`` defaults to a *rolling* start of
    ``now - 20 years``, which permanently breaks any date older than that as real
    time advances -- exactly the ~20-year-old dates this project's own
    ``historical_lookback_years`` stores. The calendar's start must be a fixed date
    well outside that rolling window, regardless of what "today" is when the test
    runs, or a date this old will eventually raise ``DateOutOfBounds``.
    """
    calendar = get_nyse_calendar()
    assert calendar.first_session <= pd.Timestamp("2000-01-01")
    assert calendar.is_session(pd.Timestamp("2006-08-01")) is True  # an ordinary Tuesday session


def test_nyse_calendar_is_cached_as_a_single_instance() -> None:
    assert get_nyse_calendar() is get_nyse_calendar()


def test_session_resolver_sessions_between_accepts_a_20_year_old_start_date() -> None:
    """Exercises the exact call path that crashed in production
    (``assess_data_health`` -> ``SessionResolver.sessions_between(dates[0], expected)``
    with ``dates[0]`` being a symbol's oldest stored ~20-year-old trade date)."""
    resolver = SessionResolver()
    sessions = resolver.sessions_between("2006-08-01", "2006-08-10")
    assert sessions[0].isoformat() == "2006-08-01"
    assert len(sessions) > 0
