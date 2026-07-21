import sqlite3
from pathlib import Path

from stock_scrapper.database import initialize_database, upsert_price_history


def test_database_creates_schema_and_enforces_unique_symbol_date(tmp_path: Path) -> None:
    db_path = tmp_path / "market.db"
    initialize_database(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='price_history'")
    assert cursor.fetchone() is not None

    row = {
        "symbol": "AAPL",
        "trade_date": "2024-01-02",
        "open": 1.0,
        "high": 2.0,
        "low": 0.5,
        "close": 1.5,
        "adjusted_close": 1.4,
        "volume": 100,
        "dividends": 0.0,
        "stock_splits": 0.0,
        "data_source": "test",
        "collected_at": "2024-01-02T00:00:00",
    }
    inserted, updated = upsert_price_history(conn, row)
    assert inserted == 1
    assert updated == 0

    inserted, updated = upsert_price_history(conn, {**row, "close": 1.6})
    assert inserted == 0
    assert updated == 1

    cursor.execute("SELECT COUNT(*) FROM price_history WHERE symbol = ?", ("AAPL",))
    assert cursor.fetchone()[0] == 1

    conn.close()
