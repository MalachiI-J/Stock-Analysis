"""SQLite repositories and lifecycle helpers for Stock Scrapper."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from stock_scrapper.migrations.migration_manager import apply_migrations
from stock_scrapper.market_calendar import SessionResolver
from stock_scrapper.revision_policy import compare_price_rows


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
    "bar_status", "is_complete", "session_close_at", "provider_updated_at",
    "first_collected_at", "last_collected_at", "revision_count", "row_fingerprint",
)

FINGERPRINT_FIELDS = ("symbol", "trade_date", "open", "high", "low", "close", "adjusted_close", "volume", "dividends", "stock_splits", "data_source")
ANALYSIS_CRITICAL_FIELDS = {"open", "high", "low", "close", "adjusted_close", "volume", "dividends", "stock_splits"}


def price_row_fingerprint(row: Mapping[str, Any]) -> str:
    """Stable SHA-256 over canonical provider values, independent of timestamps."""
    def normalized(key: str, value: Any) -> Any:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        if key == "symbol": return str(value).strip().upper()
        if key == "trade_date": return str(value)[:10]
        if key == "volume": return int(value)
        if key in {"open", "high", "low", "close", "adjusted_close", "dividends", "stock_splits"}:
            return format(float(value), ".12g")
        return str(value).strip()
    payload = {key: normalized(key, row.get(key)) for key in FINGERPRINT_FIELDS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


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


def upsert_price_history(conn: sqlite3.Connection, row: Mapping[str, Any], collection_run_id: str | None = None, revision_policy: Mapping[str, Any] | None = None) -> tuple[int, int]:
    """Reconcile a bar, preserving an immutable audit record for actual changes."""
    symbol = str(row["symbol"]).strip().upper()
    trade_date = str(row["trade_date"])
    existing = conn.execute(
        "SELECT * FROM price_history WHERE symbol = ? AND trade_date = ?",
        (symbol, trade_date),
    ).fetchone()
    incoming = {**row, "symbol": symbol, "trade_date": trade_date}
    fingerprint = price_row_fingerprint(incoming)
    collected = str(row.get("collected_at") or utc_now_iso())
    resolver = SessionResolver()
    try:
        info = resolver.session(trade_date)
        valid = all(row.get(key) is not None and not (isinstance(row.get(key), float) and math.isnan(row.get(key))) for key in ("open", "high", "low", "close", "volume"))
        complete = info.is_complete and valid
        status = "complete" if complete else ("incomplete" if valid else "invalid")
        session_close = info.closes_at.isoformat()
    except ValueError:
        complete, status, session_close = False, "invalid", None
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
        collected,
    )
    if existing is not None:
        if isinstance(existing, sqlite3.Row):
            previous = dict(existing)
        else:
            names = [str(item[0]) for item in conn.execute("SELECT * FROM price_history LIMIT 0").description]
            previous = dict(zip(names, existing, strict=True))
        previous_fingerprint = previous.get("row_fingerprint") or price_row_fingerprint(previous)
        if previous_fingerprint == fingerprint:
            conn.execute("UPDATE price_history SET last_collected_at=?, provider_updated_at=?, is_complete=?, bar_status=?, session_close_at=?, row_fingerprint=? WHERE symbol=? AND trade_date=?", (collected, collected, int(complete), status, session_close, fingerprint, symbol, trade_date))
            return 0, 0
        comparison = compare_price_rows(previous, incoming, revision_policy)
        changed = list(comparison.exact_differences)
        if not comparison.material_differences:
            conn.execute("UPDATE price_history SET last_collected_at=?, provider_updated_at=?, is_complete=?, bar_status=?, session_close_at=? WHERE symbol=? AND trade_date=?", (collected, collected, int(complete), status, session_close, symbol, trade_date))
            return 0, 0
        conn.execute(
            """INSERT OR IGNORE INTO price_history_revisions
            (symbol,trade_date,detected_at,previous_fingerprint,new_fingerprint,changed_fields_json,previous_values_json,new_values_json,data_source,collection_run_id,reason,analysis_critical,
             revision_class,is_material,absolute_deltas_json,relative_deltas_json,review_status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (symbol, trade_date, utc_now_iso(), previous_fingerprint, fingerprint,
             json.dumps(changed), canonical_values(previous), canonical_values(incoming), row.get("data_source"), collection_run_id,
             "provider_overlap_reconciliation", int(comparison.analysis_critical), comparison.revision_class, 1,
             json.dumps(comparison.absolute_deltas, sort_keys=True), json.dumps(comparison.relative_deltas, sort_keys=True), "unreviewed"),
        )
        conn.execute(
            """
            UPDATE price_history
            SET open = ?, high = ?, low = ?, close = ?, adjusted_close = ?, volume = ?,
                dividends = ?, stock_splits = ?, data_source = ?, collected_at = ?,
                bar_status = ?, is_complete = ?, session_close_at = ?, provider_updated_at = ?,
                first_collected_at = COALESCE(first_collected_at, collected_at), last_collected_at = ?,
                revision_count = revision_count + 1, row_fingerprint = ?
            WHERE symbol = ? AND trade_date = ?
            """,
            (*values, "revised" if complete else status, int(complete), session_close, collected, collected, fingerprint, symbol, trade_date),
        )
        return 0, 1
    conn.execute(
        """
        INSERT INTO price_history (
            symbol, trade_date, open, high, low, close, adjusted_close, volume,
            dividends, stock_splits, data_source, collected_at, bar_status, is_complete,
            session_close_at, provider_updated_at, first_collected_at, last_collected_at,
            revision_count, row_fingerprint
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
        """,
        (symbol, trade_date, *values, status, int(complete), session_close, collected, collected, collected, fingerprint),
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
    include_incomplete: bool = False,
) -> list[dict[str, Any]]:
    """Load a symbol's daily bars inside an inclusive database-level date range."""
    clauses = ["symbol = ?"]
    parameters: list[Any] = [symbol.strip().upper()]
    if not include_incomplete and "is_complete" in {str(r[1]) for r in conn.execute("PRAGMA table_info(price_history)")}:
        # Directly inserted legacy/test fixtures without lifecycle metadata remain
        # readable; migrated/provider rows always have collection metadata and
        # therefore must be explicitly complete.
        clauses.append("(is_complete = 1 OR (bar_status = 'unknown' AND first_collected_at IS NULL AND last_collected_at IS NULL))")
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


def canonical_values(row: Mapping[str, Any]) -> str:
    """Canonical JSON snapshot used by revision audit rows."""
    values = {key: (None if isinstance(row.get(key), float) and math.isnan(row.get(key)) else row.get(key)) for key in FINGERPRINT_FIELDS}
    return json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str, allow_nan=False)


def classify_price_revisions(conn: sqlite3.Connection, policy: Mapping[str, Any] | None = None) -> dict[str, int]:
    """Classify retained legacy revision rows without deleting audit evidence."""
    counts: dict[str, int] = {}
    rows = conn.execute("SELECT revision_id,previous_values_json,new_values_json FROM price_history_revisions").fetchall()
    for row in rows:
        previous = json.loads(row["previous_values_json"] or "{}")
        incoming = json.loads(row["new_values_json"] or "{}")
        result = compare_price_rows(previous, incoming, policy)
        material = int(bool(result.material_differences))
        automatic = result.revision_class in {"precision_noise", "corporate_action_revision"}
        conn.execute("""UPDATE price_history_revisions SET revision_class=?,is_material=?,
          absolute_deltas_json=?,relative_deltas_json=?,review_status=CASE WHEN ? THEN 'automatically_classified' ELSE COALESCE(NULLIF(review_status,''),'unreviewed') END
          WHERE revision_id=?""", (result.revision_class,material,json.dumps(result.absolute_deltas,sort_keys=True),
          json.dumps(result.relative_deltas,sort_keys=True),int(automatic),row["revision_id"]))
        counts[result.revision_class] = counts.get(result.revision_class,0)+1
    return counts


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


def list_analysis_runs(conn: sqlite3.Connection, limit: int = 50, *, scope: str | None = None, as_of_date: str | None = None, canonical_only: bool = False) -> list[dict[str, Any]]:
    """Return recent saved analysis runs without recalculating them."""
    clauses=[]; params: list[Any]=[]
    if scope: clauses.append("analysis_scope=?"); params.append(scope)
    if as_of_date: clauses.append("as_of_date=?"); params.append(as_of_date)
    if canonical_only: clauses.append("is_canonical=1")
    params.append(max(1,int(limit)))
    rows = conn.execute("SELECT * FROM analysis_runs " + (("WHERE "+" AND ".join(clauses)) if clauses else "") + " ORDER BY COALESCE(completed_at, started_at) DESC LIMIT ?", params).fetchall()
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


def get_latest_canonical_analysis_run(conn: sqlite3.Connection, scope: str = "candidate_universe") -> dict[str, Any] | None:
    row=conn.execute("SELECT analysis_run_id FROM analysis_runs WHERE status='completed' AND is_canonical=1 AND analysis_scope=? ORDER BY as_of_date DESC,COALESCE(completed_at,started_at) DESC LIMIT 1",(scope,)).fetchone()
    return get_analysis_run(conn,str(row[0])) if row else None
