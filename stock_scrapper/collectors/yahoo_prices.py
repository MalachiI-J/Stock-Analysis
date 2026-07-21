"""yfinance-backed daily price collector."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
import yfinance as yf

from stock_scrapper.collectors.base import BaseCollector, empty_price_frame


class YahooPriceCollector(BaseCollector):
    """Collect adjusted daily price data from yfinance."""

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay_seconds: float = 2.0,
        timeout_seconds: int = 20,
        historical_lookback_years: int = 5,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds cannot be negative")
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be at least 1")
        if isinstance(historical_lookback_years, bool) or not isinstance(historical_lookback_years, int):
            raise TypeError("historical_lookback_years must be an integer")
        if historical_lookback_years < 1:
            raise ValueError("historical_lookback_years must be at least 1")

        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.timeout_seconds = timeout_seconds
        self.historical_lookback_years = historical_lookback_years

    def collect(
        self,
        symbol: str,
        start_date: date | None = None,
        end_date: date | None = None,
        full_refresh: bool = False,
    ) -> pd.DataFrame:
        """Download and normalize daily history through an inclusive end date.

        yfinance treats its ``end`` argument as exclusive. The collector therefore
        requests the calendar day after ``end_date`` and then enforces the intended
        inclusive bounds locally. Weekend and market-holiday end dates naturally
        return the last session supplied by Yahoo; no row is synthesized or filled.
        """
        symbol = symbol.upper().strip()
        if not symbol:
            raise ValueError("A symbol is required")

        inclusive_end = end_date or date.today()
        effective_start = start_date
        if effective_start is None or full_refresh:
            effective_start = _subtract_calendar_years(inclusive_end, self.historical_lookback_years)

        if effective_start > inclusive_end:
            return empty_price_frame()

        exclusive_end = inclusive_end + timedelta(days=1)
        data = self._download(
            symbol=symbol,
            start=effective_start.isoformat(),
            end=exclusive_end.isoformat(),
        )

        if data is None or data.empty:
            return empty_price_frame()

        if isinstance(data, pd.Series):
            data = data.to_frame().T

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [column[0] if isinstance(column, tuple) else column for column in data.columns]

        frame = data.reset_index()
        if "Date" in frame.columns:
            frame = frame.rename(columns={"Date": "trade_date"})
        elif "Datetime" in frame.columns:
            frame = frame.rename(columns={"Datetime": "trade_date"})
        elif "index" in frame.columns:
            frame = frame.rename(columns={"index": "trade_date"})

        if "trade_date" not in frame.columns:
            return empty_price_frame()

        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce", utc=True)
        frame = frame.dropna(subset=["trade_date"])
        frame = frame.loc[
            (frame["trade_date"].dt.date >= effective_start)
            & (frame["trade_date"].dt.date <= inclusive_end)
        ]
        frame = frame.sort_values("trade_date")
        frame = frame.drop_duplicates(subset=["trade_date"], keep="last")
        frame["trade_date"] = frame["trade_date"].dt.strftime("%Y-%m-%d")

        if frame.empty:
            return empty_price_frame()

        collected_at = datetime.now(timezone.utc).isoformat()

        normalized = pd.DataFrame(
            {
                "symbol": symbol,
                "trade_date": frame["trade_date"],
                "open": _numeric_column(frame, "Open"),
                "high": _numeric_column(frame, "High"),
                "low": _numeric_column(frame, "Low"),
                "close": _numeric_column(frame, "Close"),
                "adjusted_close": _numeric_column(frame, "Adj Close"),
                "volume": _numeric_column(frame, "Volume"),
                "dividends": _numeric_column(frame, "Dividends"),
                "stock_splits": _numeric_column(frame, "Stock Splits"),
                "data_source": "yfinance",
                "collected_at": collected_at,
            }
        )
        return normalized.reset_index(drop=True)

    def _download(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Download data with retries and backoff."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "auto_adjust": False,
                    "progress": False,
                    "threads": False,
                    "timeout": self.timeout_seconds,
                    "actions": True,
                    "start": start,
                    "end": end,
                }
                return yf.download(symbol, **kwargs)
            except Exception as exc:  # pragma: no cover - network failure path
                last_error = exc
                if attempt < self.max_retries - 1:
                    import time

                    time.sleep(self.retry_delay_seconds)
        if last_error is not None:
            raise RuntimeError(f"Failed to download {symbol}: {last_error}") from last_error
        return pd.DataFrame()


def _subtract_calendar_years(value: date, years: int) -> date:
    """Return ``value`` shifted back by whole calendar years.

    February 29 maps to February 28 when the target year is not a leap year.
    """
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, month=2, day=28)


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Return a numeric source column while preserving unavailable values."""
    if name not in frame.columns:
        return pd.Series(float("nan"), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[name], errors="coerce")
