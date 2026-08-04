"""Approximate business-day (Mon-Fri) date arithmetic.

Not market-holiday-aware -- see ``stock_scrapper.market_calendar`` for the real NYSE
trading calendar used everywhere lookahead-correctness actually matters. This is a
display convenience only, translating a session count into an approximate calendar
date for a report reader, not a new source of truth.
"""

from __future__ import annotations

from datetime import date, timedelta


def add_business_days(start: date, sessions: int) -> date:
    """``start`` plus ``sessions`` weekdays, skipping Saturday/Sunday only."""
    if sessions < 0:
        raise ValueError("sessions must be nonnegative")
    current = start
    remaining = sessions
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Monday=0 .. Friday=4
            remaining -= 1
    return current
