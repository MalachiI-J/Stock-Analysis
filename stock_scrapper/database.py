"""SQLite repositories and lifecycle helpers for Stock Scrapper."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from stock_scrapper.migrations.migration_manager import apply_migrations


PRICE_COLUMNS = (
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "dividends",
    "stock_splits",
    "data_source",
    "collected_at",
)


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp suitable for persistence."""
    return datetime.now(timezone.utc).isoformat()


def initialize_database(db_path: str | Path) -> Path:
    """Create or migrate the database without recreating an existing file."""
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    apply_migrations(db_file)
    return db_file


def create_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with rows and foreign-key checks enabled."""
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def upsert_price_history(conn: sqlite3.Connection, row: Mapping[str, Any]) -> tuple[int, int]:
    """Insert a daily bar or update the same symbol/date without coercing missing fields."""
    symbol = str(row["symbol"]).strip().upper()
    trade_date = str(row["trade_date"])
    existing = conn.execute(
        "SELECT 1 FROM price_history WHERE symbol = ? AND trade_date = ?",
        (symbol, trade_date),
    ).fetchone()
    values = (
        row.get("open"),
        row.get("high"),
        row.get("low"),
        row.get("close"),
        row.get("adjusted_close"),
        row.get("volume"),
        row.get("dividends"),
        row.get("stock_splits"),
        row.get("data_source"),
        row.get("collected_at") or utc_now_iso(),
    )
    if existing is not None:
        conn.execute(
            """
            UPDATE price_history
            SET open = ?, high = ?, low = ?, close = ?, adjusted_close = ?, volume = ?,
                dividends = ?, stock_splits = ?, data_source = ?, collected_at = ?
            WHERE symbol = ? AND trade_date = ?
            """,
            (*values, symbol, trade_date),
        )
        return 0, 1
    conn.execute(
        """
        INSERT INTO price_history (
            symbol, trade_date, open, high, low, close, adjusted_close, volume,
            dividends, stock_splits, data_source, collected_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (symbol, trade_date, *values),
    )
    return 1, 0


def get_latest_trade_date(conn: sqlite3.Connection, symbol: str) -> str | None:
    """Return the latest stored trade date for a symbol."""
    row = conn.execute(
        "SELECT MAX(trade_date) AS trade_date FROM price_history WHERE symbol = ?",
        (symbol.upper(),),
    ).fetchone()
    return str(row["trade_date"]) if row is not None and row["trade_date"] else None


def fetch_price_history(
    conn: sqlite3.Connection,
    symbol: str,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
) -> list[dict[str, Any]]:
    """Load a symbol's daily bars inside an inclusive database-level date range."""
    clauses = ["symbol = ?"]
    parameters: list[Any] = [symbol.strip().upper()]
    if start_date is not None:
        clauses.append("trade_date >= ?")
        parameters.append(start_date.isoformat() if isinstance(start_date, date) else str(start_date))
    if end_date is not None:
        clauses.append("trade_date <= ?")
        parameters.append(end_date.isoformat() if isinstance(end_date, date) else str(end_date))
    sql = (
        f"SELECT {', '.join(PRICE_COLUMNS)} FROM price_history "
        f"WHERE {' AND '.join(clauses)} ORDER BY trade_date ASC"
    )
    rows = conn.execute(sql, parameters).fetchall()
    return [
        dict(row) if isinstance(row, sqlite3.Row) else dict(zip(PRICE_COLUMNS, row, strict=True))
        for row in rows
    ]


def quality_issue_fingerprint(issue: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 identity for the semantic issue fields."""
    payload = {
        "description": str(issue.get("description") or "").strip(),
        "issue_type": str(issue.get("issue_type") or "").strip(),
        "severity": str(issue.get("severity") or "").strip().lower(),
        "symbol": str(issue.get("symbol") or "").strip().upper(),
        "trade_date": str(issue.get("trade_date") or "").strip(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def record_quality_issue(conn: sqlite3.Connection, issue: Mapping[str, Any]) -> int:
    """Insert, redetect, or reopen a quality issue while keeping one unresolved row."""
    fingerprint = quality_issue_fingerprint(issue)
    detected_at = str(issue.get("detected_time") or utc_now_iso())
    now = utc_now_iso()
    existing = conn.execute(
        "SELECT id FROM data_quality_issues WHERE issue_fingerprint = ? "
        "AND resolved_status = 0 ORDER BY id LIMIT 1",
        (fingerprint,),
    ).fetchone()
    if existing is not None:
        issue_id = int(existing["id"] if isinstance(existing, sqlite3.Row) else existing[0])
        conn.execute(
            """
            UPDATE data_quality_issues
            SET last_detected_at = ?, updated_at = ?, severity = ?, description = ?
            WHERE id = ?
            """,
            (
                detected_at,
                now,
                str(issue.get("severity") or "warning").lower(),
                str(issue.get("description") or ""),
                issue_id,
            ),
        )
        return issue_id

    resolved = conn.execute(
        "SELECT id FROM data_quality_issues WHERE issue_fingerprint = ? "
        "AND resolved_status = 1 ORDER BY COALESCE(resolved_at, updated_at) DESC, id DESC LIMIT 1",
        (fingerprint,),
    ).fetchone()
    cursor = conn.execute(
        """
        INSERT INTO data_quality_issues (
            symbol, trade_date, issue_type, severity, description, detected_time,
            resolved_status, issue_fingerprint, first_detected_at, last_detected_at,
            reopened_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)
        """,
        (
            str(issue.get("symbol") or "").strip().upper(),
            issue.get("trade_date"),
            str(issue.get("issue_type") or "").strip(),
            str(issue.get("severity") or "warning").strip().lower(),
            str(issue.get("description") or "").strip(),
            detected_at,
            fingerprint,
            detected_at,
            detected_at,
            detected_at if resolved is not None else None,
            now,
        ),
    )
    return int(cursor.lastrowid)


def resolve_quality_issues_after_validation(
    conn: sqlite3.Connection,
    symbol: str,
    detected_fingerprints: Iterable[str],
    resolved_at: str | None = None,
) -> int:
    """Resolve stale unresolved issues after a complete validation for ``symbol``."""
    active = sorted(set(detected_fingerprints))
    parameters: list[Any] = [resolved_at or utc_now_iso(), utc_now_iso(), symbol.upper()]
    sql = (
        "UPDATE data_quality_issues SET resolved_status = 1, resolved_at = ?, updated_at = ? "
        "WHERE symbol = ? AND resolved_status = 0"
    )
    if active:
        sql += f" AND issue_fingerprint NOT IN ({','.join('?' for _ in active)})"
        parameters.extend(active)
    cursor = conn.execute(sql, parameters)
    return int(cursor.rowcount)


def fetch_quality_issues(
    conn: sqlite3.Connection,
    symbol: str | None = None,
    unresolved_only: bool = True,
    as_of_date: str | date | None = None,
) -> list[dict[str, Any]]:
    """Load quality issues, optionally reconstructing their state at an as-of date."""
    clauses: list[str] = []
    parameters: list[Any] = []
    if symbol is not None:
        clauses.append("symbol = ?")
        parameters.append(symbol.upper())
    if as_of_date is None:
        if unresolved_only:
            clauses.append("resolved_status = 0")
    else:
        as_of = as_of_date.isoformat() if isinstance(as_of_date, date) else str(as_of_date)
        cutoff = datetime.combine(date.fromisoformat(as_of), time.max, tzinfo=timezone.utc).isoformat()
        clauses.append("COALESCE(first_detected_at, detected_time) <= ?")
        parameters.append(cutoff)
        if unresolved_only:
            clauses.append(
                "((reopened_at IS NOT NULL AND reopened_at <= ? AND "
                "(resolved_at IS NULL OR resolved_at < reopened_at OR resolved_at > ?)) OR "
                "((reopened_at IS NULL OR reopened_at > ?) AND "
                "(resolved_at IS NULL OR resolved_at > ?)))"
            )
            parameters.extend([cutoff, cutoff, cutoff, cutoff])
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM data_quality_issues {where} "
        "ORDER BY COALESCE(last_detected_at, detected_time) DESC, id DESC",
        parameters,
    ).fetchall()
    return [dict(row) for row in rows]


def insert_collection_run(conn: sqlite3.Connection, payload: Mapping[str, Any]) -> None:
    """Store a collection-run summary."""
    conn.execute(
        """
        INSERT INTO collection_runs (
            run_id, start_time, end_time, status, symbols_requested, symbols_updated,
            symbols_failed, records_inserted, records_updated, error_summary
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


def list_analysis_runs(conn: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    """Return recent saved analysis runs without recalculating them."""
    rows = conn.execute(
        "SELECT * FROM analysis_runs ORDER BY COALESCE(completed_at, started_at) DESC LIMIT ?",
        (max(1, int(limit)),),
    ).fetchall()
    return [dict(row) for row in rows]


def get_analysis_run(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    """Load one analysis run and every associated persisted result."""
    run = conn.execute(
        "SELECT * FROM analysis_runs WHERE analysis_run_id = ?", (run_id,)
    ).fetchone()
    if run is None:
        return None
    result = dict(run)
    result["analyses"] = [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM stock_analysis WHERE analysis_run_id = ? ORDER BY symbol", (run_id,)
        ).fetchall()
    ]
    regime = conn.execute(
        "SELECT * FROM market_regime_history WHERE analysis_run_id = ?", (run_id,)
    ).fetchone()
    result["regime"] = dict(regime) if regime is not None else None
    return result


def get_latest_analysis_run(
    conn: sqlite3.Connection,
    symbols: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    """Load the latest completed saved run, optionally requiring specified symbols."""
    candidates = conn.execute(
        "SELECT analysis_run_id FROM analysis_runs WHERE status = 'completed' "
        "ORDER BY COALESCE(completed_at, started_at) DESC"
    ).fetchall()
    required = {symbol.upper() for symbol in symbols or []}
    for row in candidates:
        run_id = str(row["analysis_run_id"])
        if required:
            present = {
                str(item["symbol"])
                for item in conn.execute(
                    "SELECT symbol FROM stock_analysis WHERE analysis_run_id = ?", (run_id,)
                ).fetchall()
            }
            if not required.issubset(present):
                continue
        return get_analysis_run(conn, run_id)
    return None
