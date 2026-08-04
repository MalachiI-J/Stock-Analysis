from __future__ import annotations

from datetime import date

import pytest

from stock_scrapper.utilities.business_days import add_business_days


def test_add_business_days_zero_returns_the_same_date() -> None:
    assert add_business_days(date(2026, 8, 4), 0) == date(2026, 8, 4)


def test_add_business_days_stays_within_one_week_when_no_weekend_crossed() -> None:
    # 2026-08-04 is a Tuesday; +2 weekdays lands on Thursday, no weekend involved.
    assert add_business_days(date(2026, 8, 4), 2) == date(2026, 8, 6)


def test_add_business_days_skips_a_weekend() -> None:
    # 2026-08-04 is a Tuesday; +5 weekdays crosses one Sat/Sun.
    assert add_business_days(date(2026, 8, 4), 5) == date(2026, 8, 11)


def test_add_business_days_starting_on_a_friday_skips_immediately_to_monday() -> None:
    # 2026-08-07 is a Friday; +1 weekday must land on the following Monday, not Saturday.
    assert add_business_days(date(2026, 8, 7), 1) == date(2026, 8, 10)


def test_add_business_days_starting_on_a_weekend_still_counts_forward_correctly() -> None:
    # 2026-08-08 is a Saturday; +1 weekday lands on Monday.
    assert add_business_days(date(2026, 8, 8), 1) == date(2026, 8, 10)


def test_add_business_days_rejects_negative_sessions() -> None:
    with pytest.raises(ValueError):
        add_business_days(date(2026, 8, 4), -1)
