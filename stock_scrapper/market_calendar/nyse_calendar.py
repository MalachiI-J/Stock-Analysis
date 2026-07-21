"""NYSE calendar adapter backed by the free exchange-calendars package."""

from __future__ import annotations

from functools import lru_cache

import exchange_calendars as xcals


@lru_cache(maxsize=1)
def get_nyse_calendar():
    """Return the process-wide official XNYS calendar."""
    return xcals.get_calendar("XNYS")
