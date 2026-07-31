from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from stock_scrapper.database import create_connection, initialize_database
from stock_scrapper.collectors.sec_edgar_fundamentals import (
    fetch_company_facts,
    fetch_ticker_cik_map,
    normalize_company_facts,
    upsert_fundamentals,
)


def _database(tmp_path: Path) -> sqlite3.Connection:
    path = tmp_path / "market.db"
    initialize_database(path)
    return create_connection(path)


def _fact_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "start": "2023-10-01", "end": "2023-12-30", "val": 1000.0,
        "fy": 2024, "fp": "Q1", "form": "10-Q", "filed": "2024-02-01", "accn": "0000320193-24-000001",
    }
    row.update(overrides)
    return row


def _facts_json() -> dict[str, Any]:
    return {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {"units": {"USD": [
                    _fact_row(val=1000.0, filed="2024-02-01"),
                    _fact_row(val=999999.0, form="8-K", filed="2024-01-15"),  # excluded: not 10-Q/10-K
                ]}},
                "Revenues": {"units": {"USD": [
                    _fact_row(val=5000.0, form="10-K", end="2017-12-31", filed="2018-02-01"),
                ]}},
                "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
                    _fact_row(val=6000.0, form="10-Q", end="2023-12-30", filed="2024-02-01"),
                ]}},
                "EarningsPerShareDiluted": {"units": {"USD/shares": [
                    _fact_row(val=1.5, filed="2024-02-01"),
                    _fact_row(val=None, filed="2024-05-01"),  # excluded: missing value
                    {"start": "2024-01-01", "form": "10-Q", "val": 2.0},  # excluded: missing end/filed
                ]}},
                "SomeIrrelevantConcept": {"units": {"USD": [_fact_row(val=1.0)]}},
            }
        }
    }


def test_normalize_company_facts_flattens_allowlisted_concepts_and_merges_aliases() -> None:
    records = normalize_company_facts("aapl", _facts_json())

    assert all(record["symbol"] == "AAPL" for record in records)
    concepts = {record["concept"] for record in records}
    assert concepts == {"net_income", "revenue", "eps_diluted"}

    net_income = [r for r in records if r["concept"] == "net_income"]
    assert len(net_income) == 1  # the 8-K row is excluded
    assert net_income[0]["value"] == 1000.0
    assert net_income[0]["unit"] == "USD"
    assert net_income[0]["period_end"] == "2023-12-30"
    assert net_income[0]["filed_date"] == "2024-02-01"

    revenue = {r["source_tag"]: r for r in records if r["concept"] == "revenue"}
    assert set(revenue) == {"Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"}
    assert revenue["Revenues"]["value"] == 5000.0
    assert revenue["RevenueFromContractWithCustomerExcludingAssessedTax"]["value"] == 6000.0

    eps = [r for r in records if r["concept"] == "eps_diluted"]
    assert len(eps) == 1  # the None-value and missing-end/filed rows are excluded
    assert eps[0]["value"] == 1.5
    assert eps[0]["unit"] == "USD/shares"


def test_normalize_company_facts_handles_missing_sections_gracefully() -> None:
    assert normalize_company_facts("AAPL", {}) == []
    assert normalize_company_facts("AAPL", {"facts": {}}) == []
    assert normalize_company_facts("AAPL", {"facts": {"us-gaap": {}}}) == []


def test_upsert_fundamentals_inserts_then_updates_on_conflict(tmp_path: Path) -> None:
    conn = _database(tmp_path)
    try:
        record = {
            "symbol": "AAPL", "concept": "net_income", "source_tag": "NetIncomeLoss",
            "fiscal_year": 2024, "fiscal_period": "Q1", "form": "10-Q",
            "period_start": "2023-10-01", "period_end": "2023-12-30", "filed_date": "2024-02-01",
            "value": 1000.0, "unit": "USD", "frame": None, "collected_at": "2024-02-02T00:00:00+00:00",
        }
        changed = upsert_fundamentals(conn, [record])
        conn.commit()
        assert changed == 1
        row = conn.execute(
            "SELECT value, collected_at FROM fundamentals WHERE symbol='AAPL' AND concept='net_income'"
        ).fetchone()
        assert tuple(row) == (1000.0, "2024-02-02T00:00:00+00:00")

        revised = dict(record, value=1050.0, collected_at="2024-02-10T00:00:00+00:00")
        changed = upsert_fundamentals(conn, [revised])
        conn.commit()
        assert changed == 1
        rows = conn.execute(
            "SELECT value, collected_at FROM fundamentals WHERE symbol='AAPL' AND concept='net_income'"
        ).fetchall()
        assert [tuple(r) for r in rows] == [(1050.0, "2024-02-10T00:00:00+00:00")]  # updated in place, not duplicated
    finally:
        conn.close()


def test_upsert_fundamentals_keeps_distinct_filed_dates_as_separate_rows(tmp_path: Path) -> None:
    """A later restatement (different filed_date, same period/form) is a distinct
    row, not an overwrite -- point-in-time lookups need both vintages available."""
    conn = _database(tmp_path)
    try:
        base = {
            "symbol": "AAPL", "concept": "net_income", "source_tag": "NetIncomeLoss",
            "fiscal_year": 2024, "fiscal_period": "Q1", "form": "10-Q",
            "period_start": "2023-10-01", "period_end": "2023-12-30", "filed_date": "2024-02-01",
            "value": 1000.0, "unit": "USD", "frame": None, "collected_at": "2024-02-02T00:00:00+00:00",
        }
        restated = dict(base, filed_date="2024-06-01", value=990.0, collected_at="2024-06-02T00:00:00+00:00")
        upsert_fundamentals(conn, [base, restated])
        conn.commit()
        rows = conn.execute(
            "SELECT filed_date, value FROM fundamentals WHERE symbol='AAPL' ORDER BY filed_date"
        ).fetchall()
        assert [tuple(r) for r in rows] == [("2024-02-01", 1000.0), ("2024-06-01", 990.0)]
    finally:
        conn.close()


def test_fetch_ticker_cik_map_parses_and_zero_pads_cik(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
                "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft"},
            }

    calls: list[dict[str, Any]] = []

    def fake_get(url: str, *, headers: dict[str, str], timeout: int) -> _FakeResponse:
        calls.append({"url": url, "headers": headers, "timeout": timeout})
        return _FakeResponse()

    monkeypatch.setattr("stock_scrapper.collectors.sec_edgar_fundamentals.requests.get", fake_get)

    result = fetch_ticker_cik_map(user_agent="Stock Scraper Research test@example.com")

    assert result == {"AAPL": "0000320193", "MSFT": "0000789019"}
    assert calls[0]["headers"]["User-Agent"] == "Stock Scraper Research test@example.com"


def test_fetch_company_facts_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"facts": {"us-gaap": {}}}

    attempts = {"count": 0}

    def flaky_get(url: str, *, headers: dict[str, str], timeout: int) -> _FakeResponse:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ConnectionError("simulated network failure")
        return _FakeResponse()

    monkeypatch.setattr("stock_scrapper.collectors.sec_edgar_fundamentals.requests.get", flaky_get)
    monkeypatch.setattr("stock_scrapper.collectors.sec_edgar_fundamentals.time.sleep", lambda _seconds: None)

    result = fetch_company_facts(
        "320193", user_agent="Stock Scraper Research test@example.com",
        max_retries=3, retry_delay_seconds=0,
    )

    assert result == {"facts": {"us-gaap": {}}}
    assert attempts["count"] == 3


def test_fetch_company_facts_raises_after_exhausting_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_fails(url: str, *, headers: dict[str, str], timeout: int) -> Any:
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr("stock_scrapper.collectors.sec_edgar_fundamentals.requests.get", always_fails)
    monkeypatch.setattr("stock_scrapper.collectors.sec_edgar_fundamentals.time.sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="Failed to fetch"):
        fetch_company_facts(
            "320193", user_agent="Stock Scraper Research test@example.com",
            max_retries=2, retry_delay_seconds=0,
        )


def test_fetch_company_facts_requires_nonempty_user_agent() -> None:
    with pytest.raises(ValueError, match="user_agent"):
        fetch_company_facts("320193", user_agent="  ")
