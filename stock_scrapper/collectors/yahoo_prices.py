"""yfinance-backed daily price collector."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from stock_scrapper.collectors.base import BaseCollector


class YahooPriceCollector(BaseCollector):
    """Collect adjusted daily price data from yfinance."""

    def __init__(self, max_retries: int = 3, retry_delay_seconds: float = 2.0, timeout_seconds: int = 20) -> None:
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.timeout_seconds = timeout_seconds

    def collect(
        self,
        symbol: str,
        start_date: date | None = None,
        end_date: date | None = None,
        full_refresh: bool = False,
    ) -> pd.DataFrame:
        """Download daily history for a symbol and return a normalized DataFrame."""
        symbol = symbol.upper().strip()
        if not symbol:
            raise ValueError("A symbol is required")

        # For a first run, download a five-year history. Later runs only request the missing period.
        if start_date is None or full_refresh:
            period = "5y"
            data = self._download(symbol=symbol, period=period)
        else:
            if end_date is None:
                end_date = date.today()
            if start_date > end_date:
                return pd.DataFrame(columns=["symbol", "trade_date", "open", "high", "low", "close", "adjusted_close", "volume", "dividends", "stock_splits", "data_source", "collected_at"])
            data = self._download(symbol=symbol, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

        if data is None or data.empty:
            return pd.DataFrame(columns=["symbol", "trade_date", "open", "high", "low", "close", "adjusted_close", "volume", "dividends", "stock_splits", "data_source", "collected_at"])

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
            return pd.DataFrame(columns=["symbol", "trade_date", "open", "high", "low", "close", "adjusted_close", "volume", "dividends", "stock_splits", "data_source", "collected_at"])

        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
        frame = frame.dropna(subset=["trade_date"])
        frame = frame.sort_values("trade_date")
        frame["trade_date"] = frame["trade_date"].dt.strftime("%Y-%m-%d")

        normalized = pd.DataFrame(
            {
                "symbol": symbol,
                "trade_date": frame["trade_date"],
                "open": pd.to_numeric(frame.get("Open"), errors="coerce"),
                "high": pd.to_numeric(frame.get("High"), errors="coerce"),
                "low": pd.to_numeric(frame.get("Low"), errors="coerce"),
                "close": pd.to_numeric(frame.get("Close"), errors="coerce"),
                "adjusted_close": pd.to_numeric(frame.get("Adj Close"), errors="coerce"),
                "volume": pd.to_numeric(frame.get("Volume"), errors="coerce"),
                "dividends": pd.to_numeric(frame.get("Dividends"), errors="coerce"),
                "stock_splits": pd.to_numeric(frame.get("Stock Splits"), errors="coerce"),
                "data_source": "yfinance",
                "collected_at": datetime.utcnow().isoformat(),
            }
        )
        return normalized

    def _download(self, symbol: str, period: str | None = None, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """Download data with retries and backoff."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "auto_adjust": False,
                    "progress": False,
                    "threads": False,
                    "timeout": self.timeout_seconds,
                }
                if period is not None:
                    kwargs["period"] = period
                if start is not None:
                    kwargs["start"] = start
                if end is not None:
                    kwargs["end"] = end
                return yf.download(symbol, **kwargs)
            except Exception as exc:  # pragma: no cover - network failure path
                last_error = exc
                if attempt < self.max_retries - 1:
                    import time

                    time.sleep(self.retry_delay_seconds)
        if last_error is not None:
            raise RuntimeError(f"Failed to download {symbol}: {last_error}") from last_error
        return pd.DataFrame()
