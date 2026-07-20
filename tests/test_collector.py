from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from stock_scrapper.collectors.base import PRICE_HISTORY_COLUMNS
from stock_scrapper.collectors.yahoo_prices import YahooPriceCollector


def _history_frame(*trade_dates: str) -> pd.DataFrame:
    count = len(trade_dates)
    return pd.DataFrame(
        {
            "Open": [100.0 + index for index in range(count)],
            "High": [101.0 + index for index in range(count)],
            "Low": [99.0 + index for index in range(count)],
            "Close": [100.5 + index for index in range(count)],
            "Adj Close": [100.25 + index for index in range(count)],
            "Volume": [1_000 + index for index in range(count)],
            "Dividends": [0.0] * count,
            "Stock Splits": [0.0] * count,
        },
        index=pd.DatetimeIndex(trade_dates, name="Date"),
    )


def _capture_download(monkeypatch: Any, response: pd.DataFrame) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_download(symbol: str, **kwargs: Any) -> pd.DataFrame:
        calls.append((symbol, kwargs))
        return response.copy()

    monkeypatch.setattr("stock_scrapper.collectors.yahoo_prices.yf.download", fake_download)
    return calls


def test_configured_lookback_uses_calendar_years_and_honors_full_refresh(monkeypatch: Any) -> None:
    calls = _capture_download(monkeypatch, _history_frame("2024-02-29"))
    collector = YahooPriceCollector(max_retries=1, historical_lookback_years=3)

    result = collector.collect(" aapl ", start_date=date(2024, 2, 29), end_date=date(2024, 2, 29), full_refresh=True)

    assert calls[0][0] == "AAPL"
    assert calls[0][1]["start"] == "2021-02-28"
    assert calls[0][1]["end"] == "2024-03-01"
    assert "period" not in calls[0][1]
    assert result["trade_date"].tolist() == ["2024-02-29"]


def test_incremental_start_and_inclusive_end_are_forwarded_without_losing_final_row(monkeypatch: Any) -> None:
    calls = _capture_download(monkeypatch, _history_frame("2024-01-03", "2024-01-04", "2024-01-05"))
    collector = YahooPriceCollector(max_retries=1, historical_lookback_years=5)

    result = collector.collect("AAPL", start_date=date(2024, 1, 3), end_date=date(2024, 1, 5))

    assert calls[0][1]["start"] == "2024-01-03"
    assert calls[0][1]["end"] == "2024-01-06"
    assert result["trade_date"].tolist() == ["2024-01-03", "2024-01-04", "2024-01-05"]


def test_weekend_end_date_returns_last_available_session_without_filling(monkeypatch: Any) -> None:
    calls = _capture_download(monkeypatch, _history_frame("2024-01-05"))
    collector = YahooPriceCollector(max_retries=1)

    result = collector.collect("AAPL", start_date=date(2024, 1, 5), end_date=date(2024, 1, 7))

    assert calls[0][1]["end"] == "2024-01-08"
    assert result["trade_date"].tolist() == ["2024-01-05"]


def test_market_holiday_end_date_returns_prior_session_without_filling(monkeypatch: Any) -> None:
    calls = _capture_download(monkeypatch, _history_frame("2024-07-03"))
    collector = YahooPriceCollector(max_retries=1)

    result = collector.collect("AAPL", start_date=date(2024, 7, 3), end_date=date(2024, 7, 4))

    assert calls[0][1]["end"] == "2024-07-05"
    assert result["trade_date"].tolist() == ["2024-07-03"]


def test_rows_outside_requested_bounds_are_not_returned(monkeypatch: Any) -> None:
    _capture_download(monkeypatch, _history_frame("2024-01-02", "2024-01-03", "2024-01-04"))
    collector = YahooPriceCollector(max_retries=1)

    result = collector.collect("AAPL", start_date=date(2024, 1, 3), end_date=date(2024, 1, 3))

    assert result["trade_date"].tolist() == ["2024-01-03"]


def test_start_after_end_returns_canonical_empty_frame_without_download(monkeypatch: Any) -> None:
    calls = _capture_download(monkeypatch, _history_frame("2024-01-03"))
    collector = YahooPriceCollector(max_retries=1)

    result = collector.collect("AAPL", start_date=date(2024, 1, 4), end_date=date(2024, 1, 3))

    assert calls == []
    assert result.empty
    assert tuple(result.columns) == PRICE_HISTORY_COLUMNS


def test_collection_timestamp_is_timezone_aware_utc(monkeypatch: Any) -> None:
    _capture_download(monkeypatch, _history_frame("2024-01-03"))
    collector = YahooPriceCollector(max_retries=1)

    result = collector.collect("AAPL", start_date=date(2024, 1, 3), end_date=date(2024, 1, 3))

    collected_at = datetime.fromisoformat(result.loc[0, "collected_at"])
    assert collected_at.tzinfo is not None
    assert collected_at.utcoffset() == timezone.utc.utcoffset(collected_at)
