"""Resolve completed and adjacent NYSE sessions using timezone-aware times."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from .nyse_calendar import get_nyse_calendar

NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True, slots=True)
class SessionInfo:
    session_date: date
    opens_at: datetime
    closes_at: datetime
    completion_at: datetime
    is_early_close: bool
    is_complete: bool


class SessionResolver:
    """Calendar-backed session resolver; weekdays are never used as a proxy."""

    def __init__(self, provider_delay_minutes: int = 30) -> None:
        if provider_delay_minutes < 0:
            raise ValueError("provider_delay_minutes cannot be negative")
        self.calendar = get_nyse_calendar()
        self.delay = timedelta(minutes=provider_delay_minutes)

    @staticmethod
    def _now(now: datetime | None) -> datetime:
        value = now or datetime.now(timezone.utc)
        if value.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        return value.astimezone(timezone.utc)

    def is_session(self, value: date | str) -> bool:
        return bool(self.calendar.is_session(pd.Timestamp(str(value))))

    def session(self, value: date | str, now: datetime | None = None) -> SessionInfo:
        label = pd.Timestamp(str(value))
        if not self.calendar.is_session(label):
            raise ValueError(f"{value} is not an XNYS session")
        opened = self.calendar.session_open(label).to_pydatetime().astimezone(NEW_YORK)
        closed = self.calendar.session_close(label).to_pydatetime().astimezone(NEW_YORK)
        completion = closed + self.delay
        return SessionInfo(
            label.date(), opened, closed, completion,
            closed.timetz().replace(tzinfo=None).hour < 16,
            self._now(now) >= completion.astimezone(timezone.utc),
        )

    def is_complete(self, value: date | str, now: datetime | None = None) -> bool:
        return self.session(value, now).is_complete if self.is_session(value) else False

    def previous_session(self, value: date | str) -> date:
        label = self.calendar.date_to_session(pd.Timestamp(str(value)), direction="previous")
        if label.date() >= date.fromisoformat(str(value)[:10]):
            label = self.calendar.previous_session(label)
        return label.date()

    def next_session(self, value: date | str) -> date:
        label = self.calendar.date_to_session(pd.Timestamp(str(value)), direction="next")
        if label.date() <= date.fromisoformat(str(value)[:10]):
            label = self.calendar.next_session(label)
        return label.date()

    def previous_completed_session(self, now: datetime | None = None) -> date:
        current = self._now(now).astimezone(NEW_YORK)
        day = current.date()
        if self.is_session(day) and self.is_complete(day, current):
            return day
        return self.calendar.date_to_session(pd.Timestamp(day), direction="previous").date() if not self.is_session(day) else self.calendar.previous_session(pd.Timestamp(day)).date()

    def sessions_between(self, start: date | str, end: date | str) -> list[date]:
        return [value.date() for value in self.calendar.sessions_in_range(pd.Timestamp(str(start)), pd.Timestamp(str(end)))]

    def overlap_start(self, last_completed: date, count: int) -> date:
        if count < 1:
            raise ValueError("overlap session count must be positive")
        sessions = self.calendar.sessions_window(pd.Timestamp(last_completed), -(count - 1))
        return sessions[0].date()
