"""NYSE calendar adapter backed by the free exchange-calendars package."""

from __future__ import annotations

from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd

# exchange_calendars defaults to a *rolling* window when no start/end is passed:
# GLOBAL_DEFAULT_START = now - 20 years, GLOBAL_DEFAULT_END = now + 1 year, both
# recomputed from wall-clock time on every call. This project stores up to
# historical_lookback_years (currently 20) of price history, so that rolling floor
# eventually catches up to and passes our own oldest stored date -- exactly what
# happened on 2026-08-03, when `main.py run` started raising
# exchange_calendars.errors.DateOutOfBounds for every symbol's ~20-year-old first
# row. Fixed, generous bounds decouple calendar validity from the current date
# entirely, so this can't recur.
_CALENDAR_START = pd.Timestamp("1990-01-01")
_CALENDAR_END = pd.Timestamp("2100-01-01")


@lru_cache(maxsize=1)
def get_nyse_calendar():
    """Return the process-wide official XNYS calendar."""
    return xcals.get_calendar("XNYS", start=_CALENDAR_START, end=_CALENDAR_END)
