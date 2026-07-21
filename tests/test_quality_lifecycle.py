from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Iterator
from uuid import uuid4

import pytest

import stock_scrapper.migrations.migration_manager as migration_manager
from stock_scrapper.database import (
    create_connection,
    fetch_quality_issues,
    initialize_database,
    quality_issue_fingerprint,
    record_quality_issue,
    resolve_quality_issues_after_validation,
)
from stock_scrapper.migrations.migration_manager import apply_migrations


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _issue(
    *,
    symbol: str = "AAPL",
    issue_type: str = "missing_close",
    description: str = "Adjusted close is missing",
    severity: str = "warning",
    trade_date: str = "2024-01-02",
    detected_time: str = "2024-01-02T10:00:00+00:00",
) -> dict[str, str]:
    return {
        "symbol": symbol,
        "trade_date": trade_date,
        "issue_type": issue_type,
        "severity": severity,
        "description": description,
        "detected_time": detected_time,
    }


@pytest.fixture
def workspace_tmp_dir() -> Iterator[Path]:
    path = PROJECT_ROOT / ".agent_test_work" / uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def quality_connection(workspace_tmp_dir: Path) -> Iterator[sqlite3.Connection]:
    db_path = workspace_tmp_dir / "quality.db"
    initialize_database(db_path)
    conn = create_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def test_issue_fingerprint_is_canonical_sha256() -> None:
    issue = _issue(symbol=" aapl ", severity=" Warning ")
    fingerprint = quality_issue_fingerprint(issue)
    canonical_payload = {
        "description": "Adjusted close is missing",
        "issue_type": "missing_close",
        "severity": "warning",
        "symbol": "AAPL",
        "trade_date": "2024-01-02",
    }
    expected = hashlib.sha256(
        json.dumps(
            canonical_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    assert fingerprint == expected
    assert len(fingerprint) == 64
    assert fingerprint == quality_issue_fingerprint(
        _issue(symbol="AAPL", severity="warning")
    )
    assert fingerprint != quality_issue_fingerprint(
        _issue(description="Raw close is missing")
    )


def test_repeated_detection_updates_last_detected_without_duplicate(
    quality_connection: sqlite3.Connection,
) -> None:
    conn = quality_connection
    first = _issue(detected_time="2024-01-02T10:00:00+00:00")
    second = _issue(detected_time="2024-01-03T11:00:00+00:00")

    first_id = record_quality_issue(conn, first)
    second_id = record_quality_issue(conn, second)
    conn.commit()

    assert second_id == first_id
    rows = conn.execute(
        "SELECT * FROM data_quality_issues WHERE issue_fingerprint = ?",
        (quality_issue_fingerprint(first),),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["first_detected_at"] == first["detected_time"]
    assert rows[0]["last_detected_at"] == second["detected_time"]
    assert rows[0]["resolved_status"] == 0


def test_complete_validation_resolves_absent_issues_and_appends_reopen_instance(
    quality_connection: sqlite3.Connection,
) -> None:
    conn = quality_connection
    retained = _issue(issue_type="missing_close", description="Missing close")
    stale = _issue(issue_type="negative_volume", description="Negative volume")
    retained_id = record_quality_issue(conn, retained)
    stale_id = record_quality_issue(conn, stale)
    retained_fingerprint = quality_issue_fingerprint(retained)
    conn.commit()

    resolved = resolve_quality_issues_after_validation(
        conn,
        "AAPL",
        [retained_fingerprint],
        resolved_at="2024-01-05T12:00:00+00:00",
    )
    assert resolved == 1
    assert [row["id"] for row in fetch_quality_issues(conn, "AAPL")] == [
        retained_id
    ]
    assert conn.execute(
        "SELECT resolved_status FROM data_quality_issues WHERE id = ?", (stale_id,)
    ).fetchone()[0] == 1

    assert resolve_quality_issues_after_validation(
        conn,
        "AAPL",
        [],
        resolved_at="2024-01-06T12:00:00+00:00",
    ) == 1
    assert fetch_quality_issues(conn, "AAPL") == []

    reopened = dict(retained)
    reopened["detected_time"] = "2024-01-10T09:00:00+00:00"
    reopened_id = record_quality_issue(conn, reopened)
    conn.commit()
    row = conn.execute(
        "SELECT * FROM data_quality_issues WHERE id = ?", (retained_id,)
    ).fetchone()
    reopened_row = conn.execute(
        "SELECT * FROM data_quality_issues WHERE id = ?", (reopened_id,)
    ).fetchone()
    assert reopened_id != retained_id
    assert row["resolved_status"] == 1
    assert row["resolved_at"] == "2024-01-06T12:00:00+00:00"
    assert reopened_row["resolved_status"] == 0
    assert reopened_row["reopened_at"] == reopened["detected_time"]
    assert reopened_row["last_detected_at"] == reopened["detected_time"]
    assert len(fetch_quality_issues(conn, "AAPL", unresolved_only=True)) == 1


def test_as_of_queries_reconstruct_detected_resolved_and_reopened_periods(
    quality_connection: sqlite3.Connection,
) -> None:
    conn = quality_connection
    issue = _issue(detected_time="2024-01-02T10:00:00+00:00")
    issue_id = record_quality_issue(conn, issue)
    conn.commit()
    resolve_quality_issues_after_validation(
        conn,
        "AAPL",
        [],
        resolved_at="2024-01-05T12:00:00+00:00",
    )
    reopened = dict(issue)
    reopened["detected_time"] = "2024-01-10T09:00:00+00:00"
    reopened_id = record_quality_issue(conn, reopened)
    assert reopened_id != issue_id
    conn.commit()

    assert fetch_quality_issues(
        conn, "AAPL", unresolved_only=True, as_of_date="2024-01-01"
    ) == []
    assert [
        row["id"]
        for row in fetch_quality_issues(
            conn, "AAPL", unresolved_only=True, as_of_date="2024-01-03"
        )
    ] == [issue_id]
    assert fetch_quality_issues(
        conn, "AAPL", unresolved_only=True, as_of_date="2024-01-07"
    ) == []
    assert [
        row["id"]
        for row in fetch_quality_issues(
            conn, "AAPL", unresolved_only=True, as_of_date="2024-01-11"
        )
    ] == [reopened_id]

    resolve_quality_issues_after_validation(
        conn,
        "AAPL",
        [],
        resolved_at="2024-01-12T12:00:00+00:00",
    )
    conn.commit()
    assert fetch_quality_issues(
        conn, "AAPL", unresolved_only=True, as_of_date="2024-01-13"
    ) == []
    assert [
        row["id"]
        for row in fetch_quality_issues(
            conn, "AAPL", unresolved_only=True, as_of_date="2024-01-03"
        )
    ] == [issue_id]
    assert [
        row["id"]
        for row in fetch_quality_issues(
            conn, "AAPL", unresolved_only=False, as_of_date="2024-01-03"
        )
    ] == [issue_id]


def test_partial_unique_index_protects_unresolved_fingerprints(
    quality_connection: sqlite3.Connection,
) -> None:
    conn = quality_connection
    issue = _issue()
    record_quality_issue(conn, issue)
    conn.commit()
    fingerprint = quality_issue_fingerprint(issue)

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO data_quality_issues (
                symbol, trade_date, issue_type, severity, description, detected_time,
                resolved_status, issue_fingerprint
            ) VALUES (?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (
                "AAPL",
                issue["trade_date"],
                issue["issue_type"],
                issue["severity"],
                issue["description"],
                issue["detected_time"],
                fingerprint,
            ),
        )
    conn.rollback()

    conn.execute(
        """
        INSERT INTO data_quality_issues (
            symbol, trade_date, issue_type, severity, description, detected_time,
            resolved_status, issue_fingerprint, resolved_at
        ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
        """,
        (
            "AAPL",
            issue["trade_date"],
            issue["issue_type"],
            issue["severity"],
            issue["description"],
            issue["detected_time"],
            fingerprint,
            "2024-01-04T00:00:00+00:00",
        ),
    )
    conn.commit()
    assert conn.execute(
        "SELECT COUNT(*) FROM data_quality_issues WHERE issue_fingerprint = ? "
        "AND resolved_status = 0",
        (fingerprint,),
    ).fetchone()[0] == 1


def _create_legacy_price_database(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE price_history (
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
            "INSERT INTO price_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "AAPL",
                "2024-01-02",
                100.0,
                102.0,
                99.0,
                101.0,
                100.5,
                1_000_000,
                0.0,
                0.0,
                "legacy",
                "2024-01-02T22:00:00+00:00",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def test_migrations_are_idempotent_and_preserve_legacy_price_values(
    workspace_tmp_dir: Path,
) -> None:
    db_path = workspace_tmp_dir / "legacy.db"
    _create_legacy_price_database(db_path)

    apply_migrations(db_path)
    apply_migrations(db_path)

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT symbol, trade_date, open, high, low, close, adjusted_close, volume, dividends, stock_splits, data_source, collected_at FROM price_history WHERE symbol = 'AAPL' AND trade_date = '2024-01-02'"
        ).fetchone()
        assert row == (
            "AAPL",
            "2024-01-02",
            100.0,
            102.0,
            99.0,
            101.0,
            100.5,
            1_000_000,
            0.0,
            0.0,
            "legacy",
            "2024-01-02T22:00:00+00:00",
        )
        assert [
            value[0]
            for value in conn.execute(
                "SELECT schema_version FROM schema_metadata ORDER BY schema_version"
            )
        ] == [1, 2, 3, 4, 5, 6]
        quality_columns = {
            value[1] for value in conn.execute("PRAGMA table_info(data_quality_issues)")
        }
        assert {
            "issue_fingerprint",
            "first_detected_at",
            "last_detected_at",
            "resolved_at",
            "reopened_at",
            "updated_at",
        } <= quality_columns
        quality_indexes = {
            value[1] for value in conn.execute("PRAGMA index_list(data_quality_issues)")
        }
        regime_indexes = {
            value[1] for value in conn.execute("PRAGMA index_list(market_regime_history)")
        }
        assert "uq_quality_unresolved_fingerprint" in quality_indexes
        assert "uq_market_regime_analysis_run" in regime_indexes
    finally:
        conn.close()


def test_migration_failure_rolls_back_schema_and_data_changes(
    workspace_tmp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = workspace_tmp_dir / "rollback.db"
    initialize_database(db_path)
    conn = create_connection(db_path)
    conn.execute(
        "INSERT INTO price_history (symbol, trade_date, close, adjusted_close) "
        "VALUES ('AAPL', '2024-01-02', 101.0, 100.5)"
    )
    conn.commit()
    conn.close()

    def fail_phase3(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE rollback_probe (id INTEGER PRIMARY KEY)")
        conn.execute(
            "UPDATE price_history SET close = 999.0 WHERE symbol = 'AAPL'"
        )
        raise RuntimeError("forced migration failure")

    monkeypatch.setattr(migration_manager, "_ensure_phase3_tables", fail_phase3)
    with pytest.raises(RuntimeError, match="forced migration failure"):
        apply_migrations(db_path)

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute(
            "SELECT close FROM price_history WHERE symbol = 'AAPL'"
        ).fetchone()[0] == 101.0
        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' "
            "AND name = 'rollback_probe'"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT MAX(schema_version) FROM schema_metadata"
        ).fetchone()[0] == 6
    finally:
        conn.close()
