"""Transactional, idempotent SQLite migrations for Stock Scrapper."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


LATEST_SCHEMA_VERSION = 3


def _utc_now() -> str:
    """Return an ISO-8601 timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Return the column names currently present on ``table``."""
    return {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}


def _add_column(conn: sqlite3.Connection, table: str, definition: str) -> None:
    """Add a column only when it is not already present."""
    name = definition.split()[0].strip('"')
    if name not in _columns(conn, table):
        conn.execute(f'ALTER TABLE "{table}" ADD COLUMN {definition}')


def _has_foreign_key(
    conn: sqlite3.Connection,
    table: str,
    child_columns: tuple[str, ...],
    parent_table: str,
    parent_columns: tuple[str, ...],
) -> bool:
    """Return whether one exact (possibly composite) foreign key exists."""
    grouped: dict[int, list[tuple[int, str, str, str]]] = {}
    for row in conn.execute(f'PRAGMA foreign_key_list("{table}")'):
        grouped.setdefault(int(row[0]), []).append(
            (int(row[1]), str(row[2]), str(row[3]), str(row[4]))
        )
    for rows in grouped.values():
        ordered = sorted(rows)
        if (
            ordered
            and ordered[0][1] == parent_table
            and tuple(item[2] for item in ordered) == child_columns
            and tuple(item[3] for item in ordered) == parent_columns
        ):
            return True
    return False


def _column_is_not_null(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Return the declared NOT NULL state for one table column."""
    for row in conn.execute(f'PRAGMA table_info("{table}")'):
        if str(row[1]) == column:
            return bool(row[3])
    return False


def apply_migrations(db_path: str | Path) -> None:
    """Apply every schema migration without replacing stored price data."""
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_file)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN IMMEDIATE")
        _ensure_base_schema(conn)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_metadata (
                schema_version INTEGER NOT NULL,
                applied_at TEXT NOT NULL,
                description TEXT NOT NULL
            )
            """
        )
        current_version = int(
            conn.execute("SELECT COALESCE(MAX(schema_version), 0) FROM schema_metadata").fetchone()[0]
        )
        if current_version < 1:
            _apply_v1(conn)
        else:
            # Old repositories sometimes recorded v1 while missing its extension columns.
            _ensure_phase2_tables(conn)
        if current_version < 2:
            _apply_v2(conn)
        else:
            _ensure_phase2_stabilization(conn)
        if current_version < 3:
            _apply_v3(conn)
        else:
            _ensure_phase3_tables(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _ensure_base_schema(conn: sqlite3.Connection) -> None:
    """Create the Phase 1 tables without changing existing price values."""
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
            resolved_status INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_price_history_symbol_date "
        "ON price_history(symbol, trade_date)"
    )


def _ensure_phase2_tables(conn: sqlite3.Connection) -> None:
    """Create Phase 2 tables and safely extend legacy quality tables."""
    for definition in (
        "issue_fingerprint TEXT",
        "resolved_at TEXT",
        "updated_at TEXT",
    ):
        _add_column(conn, "data_quality_issues", definition)

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
            UNIQUE(analysis_run_id, symbol),
            FOREIGN KEY (analysis_run_id) REFERENCES analysis_runs(analysis_run_id) ON DELETE CASCADE
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
            created_at TEXT NOT NULL,
            FOREIGN KEY (analysis_run_id) REFERENCES analysis_runs(analysis_run_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_quality_symbol_resolved "
        "ON data_quality_issues(symbol, resolved_status)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_quality_fingerprint "
        "ON data_quality_issues(issue_fingerprint)"
    )


def _apply_v1(conn: sqlite3.Connection) -> None:
    """Install the original Phase 2 analysis schema."""
    _ensure_phase2_tables(conn)
    conn.execute(
        "INSERT INTO schema_metadata (schema_version, applied_at, description) VALUES (?, ?, ?)",
        (1, _utc_now(), "Add Phase 2 analysis tables and issue metadata columns"),
    )


def _deduplicate_unresolved_issues(conn: sqlite3.Connection) -> None:
    """Resolve legacy duplicates before adding the partial uniqueness constraint."""
    duplicate_groups = conn.execute(
        """
        SELECT issue_fingerprint
        FROM data_quality_issues
        WHERE resolved_status = 0 AND issue_fingerprint IS NOT NULL
        GROUP BY issue_fingerprint
        HAVING COUNT(*) > 1
        """
    ).fetchall()
    now = _utc_now()
    for (fingerprint,) in duplicate_groups:
        ids = [
            int(row[0])
            for row in conn.execute(
                "SELECT id FROM data_quality_issues WHERE issue_fingerprint = ? "
                "AND resolved_status = 0 ORDER BY id",
                (fingerprint,),
            )
        ]
        for issue_id in ids[1:]:
            conn.execute(
                "UPDATE data_quality_issues SET resolved_status = 1, resolved_at = ?, "
                "updated_at = ? WHERE id = ?",
                (now, now, issue_id),
            )


def _deduplicate_market_regimes(conn: sqlite3.Connection) -> None:
    """Keep the newest legacy regime row when one run has duplicates."""
    conn.execute(
        """
        DELETE FROM market_regime_history
        WHERE id NOT IN (
            SELECT MAX(id)
            FROM market_regime_history
            GROUP BY analysis_run_id
        )
        """
    )


def _rebuild_quality_issues_if_needed(conn: sqlite3.Connection) -> None:
    """Repair the legacy nullable lifecycle flag while retaining every issue."""
    if _column_is_not_null(conn, "data_quality_issues", "resolved_status"):
        return
    conn.execute(
        """
        CREATE TABLE _migration_data_quality_issues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            trade_date TEXT,
            issue_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            description TEXT NOT NULL,
            detected_time TEXT NOT NULL,
            resolved_status INTEGER NOT NULL DEFAULT 0,
            issue_fingerprint TEXT,
            resolved_at TEXT,
            updated_at TEXT,
            first_detected_at TEXT,
            last_detected_at TEXT,
            reopened_at TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO _migration_data_quality_issues (
            id, symbol, trade_date, issue_type, severity, description, detected_time,
            resolved_status, issue_fingerprint, resolved_at, updated_at,
            first_detected_at, last_detected_at, reopened_at
        )
        SELECT id, symbol, trade_date, issue_type, severity, description, detected_time,
               COALESCE(resolved_status, 0), issue_fingerprint, resolved_at, updated_at,
               first_detected_at, last_detected_at, reopened_at
        FROM data_quality_issues
        """
    )
    conn.execute("DROP TABLE data_quality_issues")
    conn.execute("ALTER TABLE _migration_data_quality_issues RENAME TO data_quality_issues")


def _rebuild_stock_analysis_if_needed(conn: sqlite3.Connection) -> None:
    """Retrofit the analysis-run foreign key on legacy databases."""
    if _has_foreign_key(
        conn,
        "stock_analysis",
        ("analysis_run_id",),
        "analysis_runs",
        ("analysis_run_id",),
    ):
        return
    conn.execute(
        """
        CREATE TABLE _migration_stock_analysis (
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
            positive_factors_json TEXT,
            risk_factors_json TEXT,
            confidence_limitations_json TEXT,
            quality_concerns_json TEXT,
            market_regime_effects_json TEXT,
            improvement_conditions_json TEXT,
            weakening_conditions_json TEXT,
            UNIQUE(analysis_run_id, symbol),
            FOREIGN KEY (analysis_run_id) REFERENCES analysis_runs(analysis_run_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        INSERT INTO _migration_stock_analysis (
            id, analysis_run_id, symbol, as_of_date, data_through_date, risk_score,
            opportunity_score, confidence_score, classification, primary_reason,
            risk_level, trend_state, eligible_for_scoring, blocking_reasons_json,
            risk_components_json, opportunity_components_json, confidence_components_json,
            indicators_json, flags_json, created_at, positive_factors_json,
            risk_factors_json, confidence_limitations_json, quality_concerns_json,
            market_regime_effects_json, improvement_conditions_json,
            weakening_conditions_json
        )
        SELECT id, analysis_run_id, symbol, as_of_date, data_through_date, risk_score,
               opportunity_score, confidence_score, classification, primary_reason,
               risk_level, trend_state, eligible_for_scoring, blocking_reasons_json,
               risk_components_json, opportunity_components_json, confidence_components_json,
               indicators_json, flags_json, created_at, positive_factors_json,
               risk_factors_json, confidence_limitations_json, quality_concerns_json,
               market_regime_effects_json, improvement_conditions_json,
               weakening_conditions_json
        FROM stock_analysis
        """
    )
    conn.execute("DROP TABLE stock_analysis")
    conn.execute("ALTER TABLE _migration_stock_analysis RENAME TO stock_analysis")


def _rebuild_market_regimes_if_needed(conn: sqlite3.Connection) -> None:
    """Retrofit the analysis-run foreign key and one-row-per-run constraint."""
    if _has_foreign_key(
        conn,
        "market_regime_history",
        ("analysis_run_id",),
        "analysis_runs",
        ("analysis_run_id",),
    ):
        return
    conn.execute(
        """
        CREATE TABLE _migration_market_regime_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_run_id TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            regime TEXT NOT NULL,
            confidence REAL,
            benchmark_symbol TEXT,
            metrics_json TEXT,
            reasons_json TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(analysis_run_id),
            FOREIGN KEY (analysis_run_id) REFERENCES analysis_runs(analysis_run_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        INSERT INTO _migration_market_regime_history (
            id, analysis_run_id, as_of_date, regime, confidence, benchmark_symbol,
            metrics_json, reasons_json, created_at
        )
        SELECT id, analysis_run_id, as_of_date, regime, confidence, benchmark_symbol,
               metrics_json, reasons_json, created_at
        FROM market_regime_history
        """
    )
    conn.execute("DROP TABLE market_regime_history")
    conn.execute("ALTER TABLE _migration_market_regime_history RENAME TO market_regime_history")


def _ensure_phase2_stabilization(conn: sqlite3.Connection) -> None:
    """Add stable hashes, explanations, issue lifecycle, and regime constraints."""
    _ensure_phase2_tables(conn)
    for definition in (
        "first_detected_at TEXT",
        "last_detected_at TEXT",
        "reopened_at TEXT",
    ):
        _add_column(conn, "data_quality_issues", definition)
    conn.execute(
        "UPDATE data_quality_issues SET first_detected_at = COALESCE(first_detected_at, detected_time), "
        "last_detected_at = COALESCE(last_detected_at, updated_at, detected_time), "
        "updated_at = COALESCE(updated_at, detected_time)"
    )

    for definition in (
        "configuration_snapshot_json TEXT",
        "market_regime_reasons_json TEXT",
    ):
        _add_column(conn, "analysis_runs", definition)

    for definition in (
        "positive_factors_json TEXT",
        "risk_factors_json TEXT",
        "confidence_limitations_json TEXT",
        "quality_concerns_json TEXT",
        "market_regime_effects_json TEXT",
        "improvement_conditions_json TEXT",
        "weakening_conditions_json TEXT",
    ):
        _add_column(conn, "stock_analysis", definition)

    _deduplicate_unresolved_issues(conn)
    _deduplicate_market_regimes(conn)
    _rebuild_quality_issues_if_needed(conn)
    _rebuild_stock_analysis_if_needed(conn)
    _rebuild_market_regimes_if_needed(conn)
    conn.execute(
        """
        INSERT INTO market_regime_history (
            analysis_run_id, as_of_date, regime, confidence, benchmark_symbol,
            metrics_json, reasons_json, created_at
        )
        SELECT
            runs.analysis_run_id,
            COALESCE(
                runs.as_of_date,
                (SELECT MAX(items.as_of_date) FROM stock_analysis AS items
                 WHERE items.analysis_run_id = runs.analysis_run_id),
                SUBSTR(runs.started_at, 1, 10)
            ),
            COALESCE(NULLIF(TRIM(runs.market_regime), ''), 'Insufficient Market Data'),
            runs.market_regime_confidence,
            runs.benchmark_symbol,
            NULL,
            runs.market_regime_reasons_json,
            COALESCE(runs.completed_at, runs.started_at, ?)
        FROM analysis_runs AS runs
        WHERE NOT EXISTS (
            SELECT 1 FROM market_regime_history AS regimes
            WHERE regimes.analysis_run_id = runs.analysis_run_id
        )
        """,
        (_utc_now(),),
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_quality_unresolved_fingerprint "
        "ON data_quality_issues(issue_fingerprint) "
        "WHERE resolved_status = 0 AND issue_fingerprint IS NOT NULL"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_market_regime_analysis_run "
        "ON market_regime_history(analysis_run_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_analysis_runs_as_of_status "
        "ON analysis_runs(as_of_date, status, completed_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_stock_analysis_symbol_run "
        "ON stock_analysis(symbol, analysis_run_id)"
    )


def _apply_v2(conn: sqlite3.Connection) -> None:
    """Install Phase 2 stabilization schema additions."""
    _ensure_phase2_stabilization(conn)
    conn.execute(
        "INSERT INTO schema_metadata (schema_version, applied_at, description) VALUES (?, ?, ?)",
        (2, _utc_now(), "Stabilize Phase 2 lifecycle, hashes, explanations, and regime persistence"),
    )


def _ensure_phase3_tables(conn: sqlite3.Connection) -> None:
    """Create the normalized Phase 3 backtest and walk-forward schema."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_runs (
            run_id TEXT PRIMARY KEY,
            strategy_name TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            status TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            warmup_start_date TEXT,
            benchmark_symbol TEXT NOT NULL,
            initial_cash REAL NOT NULL CHECK(initial_cash > 0),
            ending_equity REAL,
            symbols_json TEXT NOT NULL,
            configuration_hash TEXT NOT NULL,
            configuration_snapshot_json TEXT NOT NULL,
            data_hash TEXT NOT NULL,
            deterministic_result_hash TEXT,
            error_summary TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            signal_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            signal_date TEXT NOT NULL,
            action TEXT NOT NULL,
            classification TEXT,
            opportunity_score REAL,
            risk_score REAL,
            confidence_score REAL,
            market_regime TEXT,
            ranking_json TEXT,
            reason TEXT NOT NULL,
            accepted INTEGER NOT NULL DEFAULT 0,
            rejection_reason TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(run_id, signal_id),
            UNIQUE(run_id, signal_id, symbol),
            UNIQUE(run_id, symbol, signal_date, action, reason),
            FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_orders (
            order_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            signal_id TEXT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL CHECK(side IN ('buy', 'sell')),
            signal_date TEXT NOT NULL,
            scheduled_date TEXT,
            status TEXT NOT NULL,
            quantity REAL,
            reference_price REAL,
            reason TEXT NOT NULL,
            rejection_reason TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(run_id, order_id),
            UNIQUE(run_id, order_id, symbol),
            FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id) ON DELETE CASCADE,
            FOREIGN KEY (run_id, signal_id, symbol) REFERENCES backtest_signals(run_id, signal_id, symbol) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_fills (
            fill_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            order_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            fill_date TEXT NOT NULL,
            side TEXT NOT NULL CHECK(side IN ('buy', 'sell')),
            quantity REAL NOT NULL CHECK(quantity > 0),
            reference_price REAL NOT NULL,
            fill_price REAL NOT NULL,
            commission REAL NOT NULL,
            slippage REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id) ON DELETE CASCADE,
            FOREIGN KEY (run_id, order_id, symbol) REFERENCES backtest_orders(run_id, order_id, symbol) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_trades (
            trade_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            signal_date TEXT NOT NULL,
            entry_date TEXT NOT NULL,
            exit_signal_date TEXT,
            exit_date TEXT,
            quantity REAL NOT NULL,
            entry_reference_price REAL NOT NULL,
            entry_fill_price REAL NOT NULL,
            exit_reference_price REAL,
            exit_fill_price REAL,
            entry_commission REAL NOT NULL,
            exit_commission REAL NOT NULL DEFAULT 0,
            slippage_cost REAL NOT NULL DEFAULT 0,
            realized_pnl REAL,
            return_pct REAL,
            holding_days INTEGER,
            entry_reason TEXT NOT NULL,
            exit_reason TEXT,
            classification TEXT,
            market_regime TEXT,
            opportunity_score REAL,
            risk_score REAL,
            confidence_score REAL,
            ranking_json TEXT,
            ambiguous_daily_bar INTEGER NOT NULL DEFAULT 0,
            strategy_version TEXT NOT NULL,
            configuration_hash TEXT NOT NULL,
            entry_signal_id TEXT,
            entry_order_id TEXT,
            exit_order_id TEXT,
            FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id) ON DELETE CASCADE,
            FOREIGN KEY (run_id, entry_signal_id, symbol) REFERENCES backtest_signals(run_id, signal_id, symbol) ON DELETE CASCADE,
            FOREIGN KEY (run_id, entry_order_id, symbol) REFERENCES backtest_orders(run_id, order_id, symbol) ON DELETE CASCADE,
            FOREIGN KEY (run_id, exit_order_id, symbol) REFERENCES backtest_orders(run_id, order_id, symbol) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_equity_curve (
            run_id TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            cash REAL NOT NULL,
            reserved_cash REAL NOT NULL,
            market_value REAL NOT NULL,
            unrealized_pnl REAL NOT NULL,
            realized_pnl REAL NOT NULL,
            equity REAL NOT NULL,
            gross_exposure REAL NOT NULL,
            position_count INTEGER NOT NULL,
            daily_return REAL,
            benchmark_equity REAL,
            PRIMARY KEY (run_id, trade_date),
            FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_metrics (
            run_id TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL,
            metric_json TEXT,
            PRIMARY KEY (run_id, metric_name),
            FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS walk_forward_runs (
            run_id TEXT PRIMARY KEY,
            strategy_name TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            status TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            configuration_hash TEXT NOT NULL,
            configuration_snapshot_json TEXT NOT NULL,
            benchmark_symbol TEXT,
            symbols_json TEXT,
            error_summary TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS walk_forward_windows (
            window_id TEXT PRIMARY KEY,
            walk_forward_run_id TEXT NOT NULL,
            sequence_number INTEGER NOT NULL,
            window_type TEXT NOT NULL,
            warmup_start TEXT,
            warmup_end TEXT,
            development_start TEXT,
            development_end TEXT,
            validation_start TEXT,
            validation_end TEXT,
            holdout_start TEXT,
            holdout_end TEXT,
            backtest_run_id TEXT,
            status TEXT NOT NULL,
            metrics_json TEXT,
            error_summary TEXT,
            UNIQUE(walk_forward_run_id, sequence_number),
            FOREIGN KEY (walk_forward_run_id) REFERENCES walk_forward_runs(run_id) ON DELETE CASCADE,
            FOREIGN KEY (backtest_run_id) REFERENCES backtest_runs(run_id) ON DELETE SET NULL
        )
        """
    )
    for definition in (
        "benchmark_symbol TEXT",
        "symbols_json TEXT",
        "error_summary TEXT",
    ):
        _add_column(conn, "walk_forward_runs", definition)
    for definition in (
        "entry_signal_id TEXT",
        "entry_order_id TEXT",
        "exit_order_id TEXT",
    ):
        _add_column(conn, "backtest_trades", definition)
    _add_column(conn, "backtest_orders", "rejection_reason TEXT")
    for definition in (
        "window_type TEXT NOT NULL DEFAULT 'validation'",
        "warmup_end TEXT",
        "error_summary TEXT",
    ):
        _add_column(conn, "walk_forward_windows", definition)
    for statement in (
        "CREATE INDEX IF NOT EXISTS idx_backtest_runs_strategy_dates ON backtest_runs(strategy_name, start_date, end_date)",
        "CREATE INDEX IF NOT EXISTS idx_backtest_signals_run_date ON backtest_signals(run_id, signal_date, accepted)",
        "CREATE INDEX IF NOT EXISTS idx_backtest_orders_run_status ON backtest_orders(run_id, status, scheduled_date)",
        "CREATE INDEX IF NOT EXISTS idx_backtest_fills_run_date ON backtest_fills(run_id, fill_date)",
        "CREATE INDEX IF NOT EXISTS idx_backtest_trades_run_symbol ON backtest_trades(run_id, symbol, entry_date)",
        "CREATE INDEX IF NOT EXISTS idx_walk_forward_windows_run ON walk_forward_windows(walk_forward_run_id, sequence_number)",
    ):
        conn.execute(statement)


def _apply_v3(conn: sqlite3.Connection) -> None:
    """Install Phase 3 backtest and walk-forward persistence."""
    _ensure_phase3_tables(conn)
    conn.execute(
        "INSERT INTO schema_metadata (schema_version, applied_at, description) VALUES (?, ?, ?)",
        (3, _utc_now(), "Add Phase 3 portfolio backtest and walk-forward tables"),
    )
