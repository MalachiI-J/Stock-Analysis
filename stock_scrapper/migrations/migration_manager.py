"""Idempotent schema migration helpers for Phase 2."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def apply_migrations(db_path: str | Path) -> None:
    """Apply Phase 2 migrations in order without destroying existing data."""
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_file)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_metadata (
                schema_version INTEGER NOT NULL,
                applied_at TEXT NOT NULL,
                description TEXT NOT NULL
            )
            """
        )
        current_version = conn.execute("SELECT MAX(schema_version) FROM schema_metadata").fetchone()[0] or 0

        if current_version < 1:
            _apply_v1(conn)
            current_version = 1

        conn.commit()
    finally:
        conn.close()


def _apply_v1(conn: sqlite3.Connection) -> None:
    """Create the Phase 2 analysis tables and add extension columns to quality issues."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_runs (
            analysis_run_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            as_of_date TEXT,
            data_through_date TEXT,
            benchmark_symbol TEXT,
            market_regime TEXT,
            market_regime_confidence REAL,
            symbols_requested TEXT,
            symbols_analyzed TEXT,
            symbols_blocked TEXT,
            status TEXT NOT NULL,
            scoring_version TEXT,
            configuration_hash TEXT,
            error_summary TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_run_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            data_through_date TEXT,
            risk_score REAL,
            opportunity_score REAL,
            confidence_score REAL,
            classification TEXT,
            primary_reason TEXT,
            risk_level TEXT,
            trend_state TEXT,
            eligible_for_scoring INTEGER NOT NULL DEFAULT 0,
            blocking_reasons_json TEXT,
            risk_components_json TEXT,
            opportunity_components_json TEXT,
            confidence_components_json TEXT,
            indicators_json TEXT,
            flags_json TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(analysis_run_id, symbol)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS market_regime_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_run_id TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            regime TEXT NOT NULL,
            confidence REAL,
            benchmark_symbol TEXT,
            metrics_json TEXT,
            reasons_json TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    for column_name, column_sql in {
        "issue_fingerprint": "TEXT",
        "resolved_at": "TEXT",
        "updated_at": "TEXT",
    }.items():
        try:
            conn.execute(f"ALTER TABLE data_quality_issues ADD COLUMN {column_name} {column_sql}")
        except sqlite3.OperationalError:
            pass

    conn.execute(
        "INSERT INTO schema_metadata (schema_version, applied_at, description) VALUES (?, ?, ?)",
        (1, datetime.now(timezone.utc).isoformat(), "Add Phase 2 analysis tables and issue metadata columns"),
    )
