"""SQLite database helpers for Stock Scrapper."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from stock_scrapper.migrations.migration_manager import apply_migrations


def initialize_database(db_path: str | Path) -> Path:
    """Create the SQLite database and all required tables."""
    # The schema is intentionally small and modular so more analytics tables can be added later.
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    apply_migrations(db_file)
    conn = sqlite3.connect(db_file)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS price_history (
                symbol TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adjusted_close REAL,
                volume INTEGER,
                dividends REAL,
                stock_splits REAL,
                data_source TEXT,
                collected_at TEXT,
                PRIMARY KEY (symbol, trade_date)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS collection_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                status TEXT NOT NULL,
                symbols_requested TEXT,
                symbols_updated TEXT,
                symbols_failed TEXT,
                records_inserted INTEGER DEFAULT 0,
                records_updated INTEGER DEFAULT 0,
                error_summary TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS data_quality_issues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                trade_date TEXT,
                issue_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT NOT NULL,
                detected_time TEXT NOT NULL,
                resolved_status INTEGER DEFAULT 0
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_price_history_symbol_date ON price_history(symbol, trade_date)"
        )
        conn.commit()
    finally:
        conn.close()
    return db_file


def create_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def upsert_price_history(conn: sqlite3.Connection, row: Mapping[str, Any]) -> tuple[int, int]:
    """Insert a price row or update an existing one when the symbol/date already exists."""
    symbol = str(row["symbol"]).strip().upper()
    trade_date = str(row["trade_date"])

    existing = conn.execute(
        "SELECT 1 FROM price_history WHERE symbol = ? AND trade_date = ?",
        (symbol, trade_date),
    ).fetchone()

    if existing is not None:
        conn.execute(
            """
            UPDATE price_history
            SET open = ?, high = ?, low = ?, close = ?, adjusted_close = ?, volume = ?, dividends = ?, stock_splits = ?, data_source = ?, collected_at = ?
            WHERE symbol = ? AND trade_date = ?
            """,
            (
                row.get("open"),
                row.get("high"),
                row.get("low"),
                row.get("close"),
                row.get("adjusted_close"),
                row.get("volume"),
                row.get("dividends"),
                row.get("stock_splits"),
                row.get("data_source"),
                row.get("collected_at") or datetime.now(timezone.utc).isoformat(),
                symbol,
                trade_date,
            ),
        )
        return 0, 1

    conn.execute(
        """
        INSERT INTO price_history (
            symbol, trade_date, open, high, low, close, adjusted_close, volume, dividends, stock_splits, data_source, collected_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            symbol,
            trade_date,
            row.get("open"),
            row.get("high"),
            row.get("low"),
            row.get("close"),
            row.get("adjusted_close"),
            row.get("volume"),
            row.get("dividends"),
            row.get("stock_splits"),
            row.get("data_source"),
            row.get("collected_at") or datetime.now(timezone.utc).isoformat(),
        ),
    )
    return 1, 0


def get_latest_trade_date(conn: sqlite3.Connection, symbol: str) -> str | None:
    """Return the latest stored trade date for a symbol, if one exists."""
    row = conn.execute(
        "SELECT trade_date FROM price_history WHERE symbol = ? ORDER BY trade_date DESC LIMIT 1",
        (symbol.upper(),),
    ).fetchone()
    return row["trade_date"] if row is not None else None


def record_quality_issue(conn: sqlite3.Connection, issue: Mapping[str, Any]) -> None:
    """Persist a data quality issue into the database."""
    conn.execute(
        """
        INSERT INTO data_quality_issues (
            symbol, trade_date, issue_type, severity, description, detected_time, resolved_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            issue.get("symbol"),
            issue.get("trade_date"),
            issue.get("issue_type"),
            issue.get("severity"),
            issue.get("description"),
            issue.get("detected_time") or datetime.now(timezone.utc).isoformat(),
            0,
        ),
    )


def insert_collection_run(conn: sqlite3.Connection, payload: Mapping[str, Any]) -> None:
    """Store a collection run summary in the database."""
    conn.execute(
        """
        INSERT INTO collection_runs (
            run_id, start_time, end_time, status, symbols_requested, symbols_updated, symbols_failed, records_inserted, records_updated, error_summary
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload.get("run_id"),
            payload.get("start_time"),
            payload.get("end_time"),
            payload.get("status"),
            payload.get("symbols_requested"),
            payload.get("symbols_updated"),
            payload.get("symbols_failed"),
            payload.get("records_inserted", 0),
            payload.get("records_updated", 0),
            payload.get("error_summary"),
        ),
    )


def fetch_price_history(conn: sqlite3.Connection, symbol: str) -> list[dict[str, Any]]:
    """Load stored price history rows for a single symbol."""
    rows = conn.execute(
        """
        SELECT symbol, trade_date, open, high, low, close, adjusted_close, volume, dividends, stock_splits, data_source, collected_at
        FROM price_history
        WHERE symbol = ?
        ORDER BY trade_date ASC
        """,
        (symbol.upper(),),
    ).fetchall()
    return [dict(row) for row in rows]
