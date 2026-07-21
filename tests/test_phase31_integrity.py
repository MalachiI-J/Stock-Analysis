from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from stock_scrapper.config import validate_universes
from stock_scrapper.data_health import assess_data_health
from stock_scrapper.database import create_connection, initialize_database, upsert_price_history
from stock_scrapper.market_calendar import SessionResolver
from stock_scrapper.utilities.provenance import source_fingerprint


NY = ZoneInfo("America/New_York")


def test_session_completion_holiday_weekend_early_close_and_dst() -> None:
    resolver = SessionResolver(30)
    assert not resolver.is_session("2026-07-04")
    assert not resolver.is_session("2026-07-05")
    assert not resolver.session("2026-07-20", datetime(2026, 7, 20, 16, 29, tzinfo=NY)).is_complete
    assert resolver.session("2026-07-20", datetime(2026, 7, 20, 16, 30, tzinfo=NY)).is_complete
    early = resolver.session("2026-11-27", datetime(2026, 11, 27, 13, 29, tzinfo=NY))
    assert early.is_early_close and not early.is_complete
    assert resolver.session("2026-11-27", datetime(2026, 11, 27, 13, 30, tzinfo=NY)).is_complete
    assert resolver.session("2026-03-09").opens_at.utcoffset().total_seconds() == -4 * 3600
    assert resolver.session("2026-03-06").opens_at.utcoffset().total_seconds() == -5 * 3600
    assert resolver.previous_session("2026-07-20").isoformat() == "2026-07-17"
    assert resolver.next_session("2026-07-20").isoformat() == "2026-07-21"


def _row(close: float = 10.0) -> dict[str, object]:
    return {"symbol": "AAPL", "trade_date": "2024-01-02", "open": 9.0, "high": 11.0,
            "low": 8.0, "close": close, "adjusted_close": close, "volume": 1000,
            "dividends": 0.0, "stock_splits": 0.0, "data_source": "fixture",
            "collected_at": "2024-01-03T00:00:00+00:00"}


def test_identical_bar_is_unchanged_and_revision_is_audited(tmp_path: Path) -> None:
    db = initialize_database(tmp_path / "market.db")
    conn = create_connection(db)
    try:
        assert upsert_price_history(conn, _row()) == (1, 0)
        assert upsert_price_history(conn, _row()) == (0, 0)
        assert conn.execute("SELECT COUNT(*) FROM price_history_revisions").fetchone()[0] == 0
        assert upsert_price_history(conn, _row(10.5), "collection-test") == (0, 1)
        revision = conn.execute("SELECT * FROM price_history_revisions").fetchone()
        assert revision["collection_run_id"] == "collection-test"
        assert "close" in revision["changed_fields_json"]
        assert conn.execute("SELECT revision_count FROM price_history").fetchone()[0] == 1
    finally:
        conn.close()


def test_incomplete_rows_are_excluded_and_health_warns(tmp_path: Path) -> None:
    db = initialize_database(tmp_path / "market.db")
    conn = create_connection(db)
    try:
        row = {**_row(), "trade_date": "2027-01-04"}
        upsert_price_history(conn, row)
        from stock_scrapper.database import fetch_price_history
        assert fetch_price_history(conn, "AAPL") == []
        assert len(fetch_price_history(conn, "AAPL", include_incomplete=True)) == 1
        assert assess_data_health(conn, ["AAPL"])["status"] == "Warning"
    finally:
        conn.close()


def test_universe_overlap_warning_and_source_fingerprint(tmp_path: Path) -> None:
    assert validate_universes({"candidates": ["SPY"], "benchmark": "SPY"})
    package = tmp_path / "stock_scrapper"; package.mkdir()
    source = package / "sample.py"; source.write_text("VALUE = 1\n", encoding="utf-8")
    first = source_fingerprint(tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert source_fingerprint(tmp_path) != first
