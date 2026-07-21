"""Transactional, idempotent SQLite migrations for Stock Scrapper."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


LATEST_SCHEMA_VERSION = 6


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
        if current_version < 4:
            _apply_v4(conn)
        else:
            _ensure_phase31_tables(conn)
        if current_version < 5:
            _apply_v5(conn)
        else:
            _ensure_phase32_tables(conn)
        if current_version < 6:
            _apply_v6(conn)
        else:
            _ensure_phase33_tables(conn)
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


def _ensure_phase31_tables(conn: sqlite3.Connection) -> None:
    """Add auditable daily-bar lifecycle, revisions, actions, and provenance."""
    for definition in (
        "bar_status TEXT NOT NULL DEFAULT 'unknown' CHECK(bar_status IN ('complete','incomplete','revised','invalid','unknown'))",
        "is_complete INTEGER NOT NULL DEFAULT 0 CHECK(is_complete IN (0,1))",
        "session_close_at TEXT",
        "provider_updated_at TEXT",
        "first_collected_at TEXT",
        "last_collected_at TEXT",
        "revision_count INTEGER NOT NULL DEFAULT 0 CHECK(revision_count >= 0)",
        "row_fingerprint TEXT",
    ):
        _add_column(conn, "price_history", definition)
    conn.execute("UPDATE price_history SET first_collected_at=COALESCE(first_collected_at,collected_at), last_collected_at=COALESCE(last_collected_at,collected_at)")
    # Conservative legacy classification: validated older rows are complete, while
    # every symbol's latest pre-migration row remains unknown pending reconciliation.
    conn.execute(
        """UPDATE price_history AS prices SET bar_status='complete', is_complete=1
        WHERE bar_status='unknown' AND open IS NOT NULL AND high IS NOT NULL
          AND low IS NOT NULL AND close IS NOT NULL AND volume IS NOT NULL
          AND high >= MAX(open, close, low) AND low <= MIN(open, close, high)
          AND trade_date < (SELECT MAX(latest.trade_date) FROM price_history AS latest WHERE latest.symbol=prices.symbol)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS price_history_revisions (
        revision_id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL,
        trade_date TEXT NOT NULL, detected_at TEXT NOT NULL,
        previous_fingerprint TEXT NOT NULL, new_fingerprint TEXT NOT NULL,
        changed_fields_json TEXT NOT NULL, previous_values_json TEXT NOT NULL,
        new_values_json TEXT NOT NULL, data_source TEXT, collection_run_id TEXT,
        reason TEXT NOT NULL, analysis_critical INTEGER NOT NULL DEFAULT 1,
        UNIQUE(symbol, trade_date, previous_fingerprint, new_fingerprint)
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_price_revisions_symbol_date ON price_history_revisions(symbol,trade_date,detected_at)")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS corporate_actions (
        action_id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL,
        action_date TEXT NOT NULL, action_type TEXT NOT NULL CHECK(action_type IN ('dividend','split')),
        dividend_amount REAL, split_ratio REAL, source TEXT NOT NULL,
        collected_at TEXT NOT NULL, provider_reference TEXT,
        UNIQUE(symbol, action_date, action_type, source)
        )"""
    )
    for definition in (
        "application_version TEXT", "scoring_version TEXT", "schema_version INTEGER",
        "git_commit_hash TEXT", "git_dirty INTEGER", "source_fingerprint TEXT",
        "python_version TEXT", "platform_info TEXT", "requested_start_date TEXT",
        "effective_start_date TEXT", "required_warmup_sessions INTEGER",
        "available_warmup_sessions INTEGER", "warmup_policy TEXT",
        "warmup_warning TEXT", "universe_json TEXT", "benchmark_sufficient INTEGER",
    ):
        _add_column(conn, "backtest_runs", definition)
    for definition in (
        "application_version TEXT", "schema_version INTEGER", "git_commit_hash TEXT",
        "git_dirty INTEGER", "source_fingerprint TEXT", "python_version TEXT",
        "platform_info TEXT", "data_health_status TEXT", "universe_json TEXT",
    ):
        _add_column(conn, "analysis_runs", definition)


def _apply_v4(conn: sqlite3.Connection) -> None:
    _ensure_phase31_tables(conn)
    conn.execute(
        "INSERT INTO schema_metadata (schema_version, applied_at, description) VALUES (?, ?, ?)",
        (4, _utc_now(), "Add Phase 3.1 session integrity, revisions, corporate actions, warm-up, and provenance"),
    )


def _ensure_phase32_tables(conn: sqlite3.Connection) -> None:
    for definition in (
        "revision_class TEXT", "is_material INTEGER", "absolute_deltas_json TEXT",
        "relative_deltas_json TEXT", "review_status TEXT NOT NULL DEFAULT 'unreviewed'",
        "review_notes TEXT",
    ): _add_column(conn, "price_history_revisions", definition)
    conn.execute("""CREATE TABLE IF NOT EXISTS corporate_action_coverage (
      symbol TEXT NOT NULL, data_source TEXT NOT NULL, requested_start_date TEXT NOT NULL,
      requested_end_date TEXT NOT NULL, earliest_action_date TEXT, latest_action_date TEXT,
      last_successful_collection_time TEXT, last_attempted_collection_time TEXT NOT NULL,
      collection_status TEXT NOT NULL, error_summary TEXT, coverage_confidence TEXT NOT NULL,
      response_hash TEXT, PRIMARY KEY(symbol,data_source))""")
    for definition in (
        "requested_end_date TEXT", "effective_end_date TEXT", "excluded_symbols_json TEXT",
        "exclusion_reasons_json TEXT", "data_health_snapshot_json TEXT",
        "corporate_action_coverage_json TEXT", "revision_policy_version TEXT",
        "strategy_version_warning TEXT",
    ): _add_column(conn, "backtest_runs", definition)
    _add_column(conn, "analysis_runs", "data_hash TEXT")
    conn.execute("""CREATE TABLE IF NOT EXISTS backtest_symbol_attribution (
      run_id TEXT NOT NULL, symbol TEXT NOT NULL, trades INTEGER NOT NULL, gross_pnl REAL,
      net_pnl REAL, profit_contribution_pct REAL, win_rate REAL, average_trade REAL,
      average_holding_period REAL, commission REAL, slippage REAL,
      PRIMARY KEY(run_id,symbol), FOREIGN KEY(run_id) REFERENCES backtest_runs(run_id) ON DELETE CASCADE)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS backtest_signal_outcomes (
      run_id TEXT NOT NULL, signal_id TEXT NOT NULL, symbol TEXT NOT NULL, signal_date TEXT NOT NULL,
      return_5 REAL, return_21 REAL, return_63 REAL, maximum_favorable_excursion REAL,
      maximum_adverse_excursion REAL, PRIMARY KEY(run_id,signal_id),
      FOREIGN KEY(run_id) REFERENCES backtest_runs(run_id) ON DELETE CASCADE)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS backtest_exit_diagnostics (
      run_id TEXT NOT NULL, trade_id TEXT NOT NULL, symbol TEXT NOT NULL, exit_reason TEXT,
      realized_pnl REAL, maximum_before_exit REAL, maximum_after_exit REAL,
      research_window_return REAL, hold_to_end_return REAL, avoided_later_loss INTEGER,
      exited_before_later_gain INTEGER, PRIMARY KEY(run_id,trade_id),
      FOREIGN KEY(run_id) REFERENCES backtest_runs(run_id) ON DELETE CASCADE)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS backtest_daily_diagnostics (
      run_id TEXT NOT NULL, trade_date TEXT NOT NULL, cash_percentage REAL,
      fully_invested INTEGER, below_maximum_positions INTEGER, no_eligible_candidate INTEGER,
      rejected_eligible_candidates INTEGER, benchmark_return REAL,
      PRIMARY KEY(run_id,trade_date), FOREIGN KEY(run_id) REFERENCES backtest_runs(run_id) ON DELETE CASCADE)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS backtest_benchmark_metrics (
      run_id TEXT NOT NULL, metric_name TEXT NOT NULL, metric_value REAL, limitation TEXT,
      PRIMARY KEY(run_id,metric_name), FOREIGN KEY(run_id) REFERENCES backtest_runs(run_id) ON DELETE CASCADE)""")


def _apply_v5(conn: sqlite3.Connection) -> None:
    _ensure_phase32_tables(conn)
    conn.execute("INSERT INTO schema_metadata(schema_version,applied_at,description) VALUES(?,?,?)",
                 (5,_utc_now(),"Add Phase 3.2 revision calibration, action coverage, run evidence, and diagnostics"))


def _ensure_phase33_tables(conn: sqlite3.Connection) -> None:
    """Add universe-aware analysis identity and exact-run report persistence."""
    for definition in (
        "analysis_scope TEXT", "is_canonical INTEGER NOT NULL DEFAULT 0",
        "requested_symbols_json TEXT", "analyzed_symbols_json TEXT",
        "blocked_symbols_json TEXT", "symbol_count INTEGER",
        "candidate_universe_hash TEXT", "universe_configuration_json TEXT",
        "report_manifest_json TEXT", "supersedes_run_id TEXT",
        "legacy_scope_inferred INTEGER NOT NULL DEFAULT 0",
    ):
        _add_column(conn, "analysis_runs", definition)
    conn.execute("""CREATE TABLE IF NOT EXISTS analysis_reports (
      report_id TEXT PRIMARY KEY, analysis_run_id TEXT NOT NULL,
      generated_at TEXT NOT NULL, scope TEXT NOT NULL, csv_path TEXT,
      html_path TEXT, manifest_path TEXT, csv_sha256 TEXT, html_sha256 TEXT,
      report_version INTEGER NOT NULL DEFAULT 1, status TEXT NOT NULL,
      error_summary TEXT, UNIQUE(analysis_run_id,report_version),
      FOREIGN KEY(analysis_run_id) REFERENCES analysis_runs(analysis_run_id) ON DELETE RESTRICT)""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_analysis_runs_scope_canonical ON analysis_runs(analysis_scope,is_canonical,status,completed_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_analysis_reports_run ON analysis_reports(analysis_run_id,generated_at)")
    candidate = {"AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","JPM","WMT","XOM"}
    all_data = candidate | {"SPY","QQQ","IWM","TLT","GLD"}
    import json
    for row in conn.execute("SELECT analysis_run_id FROM analysis_runs WHERE analysis_scope IS NULL OR requested_symbols_json IS NULL").fetchall():
        run_id = str(row[0])
        symbols = [str(item[0]).upper() for item in conn.execute("SELECT symbol FROM stock_analysis WHERE analysis_run_id=? ORDER BY rowid", (run_id,))]
        symbol_set = set(symbols)
        scope = "candidate_universe" if symbol_set == candidate else ("all_data_symbols" if symbol_set == all_data else "custom")
        conn.execute("""UPDATE analysis_runs SET analysis_scope=?,requested_symbols_json=COALESCE(requested_symbols_json,?),
          analyzed_symbols_json=COALESCE(analyzed_symbols_json,?),blocked_symbols_json=COALESCE(blocked_symbols_json,'[]'),
          symbol_count=COALESCE(symbol_count,?),legacy_scope_inferred=1 WHERE analysis_run_id=?""",
          (scope,json.dumps(symbols),json.dumps(symbols),len(symbols),run_id))
    conn.execute("""UPDATE price_history_revisions SET review_status='automatically_classified'
      WHERE revision_class IN ('precision_noise','corporate_action_revision')
        AND COALESCE(review_status,'unreviewed') IN ('','unreviewed')""")


def _apply_v6(conn: sqlite3.Connection) -> None:
    _ensure_phase33_tables(conn)
    conn.execute("INSERT INTO schema_metadata(schema_version,applied_at,description) VALUES(?,?,?)",
                 (6,_utc_now(),"Add Phase 3.3 universe scope, canonical selection, report persistence, and review classification"))
