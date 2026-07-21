from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from stock_scrapper.database import create_connection, initialize_database
from stock_scrapper.migrations import migration_manager
from stock_scrapper.migrations.migration_manager import apply_migrations


LEGACY_V1_SCHEMA = """
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
);
CREATE TABLE collection_runs (
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
);
CREATE TABLE data_quality_issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    trade_date TEXT,
    issue_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    description TEXT NOT NULL,
    detected_time TEXT NOT NULL,
    resolved_status INTEGER DEFAULT 0,
    issue_fingerprint TEXT,
    resolved_at TEXT,
    updated_at TEXT
);
CREATE TABLE schema_metadata (
    schema_version INTEGER NOT NULL,
    applied_at TEXT NOT NULL,
    description TEXT NOT NULL
);
CREATE TABLE analysis_runs (
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
);
CREATE TABLE stock_analysis (
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
);
CREATE TABLE market_regime_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_run_id TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    regime TEXT NOT NULL,
    confidence REAL,
    benchmark_symbol TEXT,
    metrics_json TEXT,
    reasons_json TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX idx_price_history_symbol_date
    ON price_history(symbol, trade_date);
CREATE INDEX idx_quality_symbol_resolved
    ON data_quality_issues(symbol, resolved_status);
CREATE INDEX idx_quality_fingerprint
    ON data_quality_issues(issue_fingerprint);
"""


PRICE_ROW = (
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


def _create_real_shaped_v1_database(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(LEGACY_V1_SCHEMA)
        conn.execute(
            "INSERT INTO schema_metadata VALUES (1, ?, ?)",
            ("2024-01-03T00:00:00+00:00", "legacy Phase 2 schema"),
        )
        conn.execute(
            "INSERT INTO price_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            PRICE_ROW,
        )
        conn.execute(
            "INSERT INTO analysis_runs (analysis_run_id, started_at, status) VALUES (?, ?, ?)",
            ("analysis-legacy", "2024-01-03T00:00:00+00:00", "completed"),
        )
        conn.execute(
            "INSERT INTO stock_analysis "
            "(analysis_run_id, symbol, as_of_date, eligible_for_scoring, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                "analysis-legacy",
                "AAPL",
                "2024-01-02",
                1,
                "2024-01-03T00:00:00+00:00",
            ),
        )
        conn.execute(
            "INSERT INTO data_quality_issues "
            "(symbol, trade_date, issue_type, severity, description, detected_time, "
            "resolved_status, issue_fingerprint) VALUES (?, ?, ?, ?, ?, ?, NULL, ?)",
            (
                "AAPL",
                "2024-01-02",
                "missing_volume",
                "warning",
                "Legacy nullable lifecycle row",
                "2024-01-03T01:00:00+00:00",
                "legacy-quality-fingerprint",
            ),
        )
        conn.execute(
            "INSERT INTO market_regime_history "
            "(analysis_run_id, as_of_date, regime, confidence, benchmark_symbol, "
            "metrics_json, reasons_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "analysis-legacy",
                "2024-01-02",
                "bear",
                0.55,
                "SPY",
                '{"sequence":1}',
                '["older"]',
                "2024-01-03T01:00:00+00:00",
            ),
        )
        conn.execute(
            "INSERT INTO market_regime_history "
            "(analysis_run_id, as_of_date, regime, confidence, benchmark_symbol, "
            "metrics_json, reasons_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "analysis-legacy",
                "2024-01-02",
                "bull",
                0.75,
                "SPY",
                '{"sequence":2}',
                '["newer"]',
                "2024-01-03T02:00:00+00:00",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _price_row(conn: sqlite3.Connection) -> tuple[object, ...]:
    row = conn.execute(
        "SELECT symbol, trade_date, open, high, low, close, adjusted_close, volume, "
        "dividends, stock_splits, data_source, collected_at FROM price_history"
    ).fetchone()
    assert row is not None
    return tuple(row)


def _foreign_keys(
    conn: sqlite3.Connection, table: str
) -> set[tuple[str, tuple[str, ...], tuple[str, ...], str]]:
    grouped: dict[int, list[sqlite3.Row | tuple[object, ...]]] = {}
    for row in conn.execute(f'PRAGMA foreign_key_list("{table}")'):
        grouped.setdefault(int(row[0]), []).append(row)
    return {
        (
            str(sorted(rows, key=lambda row: int(row[1]))[0][2]),
            tuple(str(row[3]) for row in sorted(rows, key=lambda row: int(row[1]))),
            tuple(str(row[4]) for row in sorted(rows, key=lambda row: int(row[1]))),
            str(sorted(rows, key=lambda row: int(row[1]))[0][6]),
        )
        for rows in grouped.values()
    }


def test_real_shaped_v1_is_repaired_idempotently_without_price_changes(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy-v1.db"
    _create_real_shaped_v1_database(db_path)

    apply_migrations(db_path)
    apply_migrations(db_path)

    conn = create_connection(db_path)
    try:
        assert _price_row(conn) == PRICE_ROW
        assert [
            tuple(row)
            for row in conn.execute(
                "SELECT schema_version FROM schema_metadata ORDER BY schema_version"
            )
        ] == [(1,), (2,), (3,), (4,), (5,), (6,)]
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []

        resolved_status = next(
            row for row in conn.execute("PRAGMA table_info(data_quality_issues)")
            if row[1] == "resolved_status"
        )
        assert resolved_status[3] == 1
        assert conn.execute(
            "SELECT resolved_status FROM data_quality_issues"
        ).fetchone()[0] == 0

        assert (
            "analysis_runs",
            ("analysis_run_id",),
            ("analysis_run_id",),
            "CASCADE",
        ) in _foreign_keys(conn, "stock_analysis")
        assert (
            "analysis_runs",
            ("analysis_run_id",),
            ("analysis_run_id",),
            "CASCADE",
        ) in _foreign_keys(conn, "market_regime_history")

        regimes = conn.execute(
            "SELECT id, regime, confidence, metrics_json, reasons_json, created_at "
            "FROM market_regime_history WHERE analysis_run_id = 'analysis-legacy'"
        ).fetchall()
        assert [tuple(row) for row in regimes] == [
            (
                2,
                "bull",
                0.75,
                '{"sequence":2}',
                '["newer"]',
                "2024-01-03T02:00:00+00:00",
            )
        ]
        assert conn.execute("SELECT COUNT(*) FROM stock_analysis").fetchone()[0] == 1
    finally:
        conn.close()


def test_legacy_constraint_repair_rolls_back_as_one_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "legacy-rollback.db"
    _create_real_shaped_v1_database(db_path)

    def fail_phase3(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE migration_failure_probe (id INTEGER PRIMARY KEY)")
        conn.execute("UPDATE price_history SET close = 999.0")
        raise RuntimeError("forced failure after legacy repairs")

    monkeypatch.setattr(migration_manager, "_ensure_phase3_tables", fail_phase3)
    with pytest.raises(RuntimeError, match="forced failure after legacy repairs"):
        apply_migrations(db_path)

    conn = sqlite3.connect(db_path)
    try:
        assert _price_row(conn) == PRICE_ROW
        assert conn.execute(
            "SELECT schema_version FROM schema_metadata ORDER BY schema_version"
        ).fetchall() == [(1,)]
        assert conn.execute("SELECT COUNT(*) FROM market_regime_history").fetchone()[0] == 2
        assert conn.execute("PRAGMA foreign_key_list(stock_analysis)").fetchall() == []
        assert "first_detected_at" not in {
            row[1] for row in conn.execute("PRAGMA table_info(data_quality_issues)")
        }
        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name = 'migration_failure_probe'"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE '_migration_%'"
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_existing_analysis_run_without_regime_history_is_backfilled_once(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "regime-backfill.db"
    initialize_database(db_path)
    conn = create_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO analysis_runs (
                analysis_run_id, started_at, completed_at, as_of_date,
                benchmark_symbol, market_regime, market_regime_confidence,
                status, market_regime_reasons_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-run",
                "2024-01-03T00:00:00+00:00",
                "2024-01-03T00:01:00+00:00",
                "2024-01-02",
                "SPY",
                "Risk-Off",
                0.65,
                "completed",
                '["Recorded on the legacy run"]',
            ),
        )
        conn.commit()
    finally:
        conn.close()

    apply_migrations(db_path)
    apply_migrations(db_path)

    conn = create_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT as_of_date, regime, confidence, benchmark_symbol, metrics_json, reasons_json "
            "FROM market_regime_history WHERE analysis_run_id='legacy-run'"
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            (
                "2024-01-02",
                "Risk-Off",
                0.65,
                "SPY",
                None,
                '["Recorded on the legacy run"]',
            )
        ]
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def _insert_backtest_run(conn: sqlite3.Connection, run_id: str) -> None:
    conn.execute(
        """
        INSERT INTO backtest_runs (
            run_id, strategy_name, strategy_version, started_at, status, start_date,
            end_date, benchmark_symbol, initial_cash, symbols_json, configuration_hash,
            configuration_snapshot_json, data_hash
        ) VALUES (?, 'score_v1', '1.0', '2024-01-01T00:00:00+00:00', 'completed',
                  '2024-01-01', '2024-12-31', 'SPY', 100000.0, '["AAA","BBB"]',
                  'configuration-hash', '{}', 'data-hash')
        """,
        (run_id,),
    )


def test_phase3_rejects_cross_run_and_cross_symbol_order_fill_links(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "fresh-v3.db"
    initialize_database(db_path)
    conn = create_connection(db_path)
    try:
        _insert_backtest_run(conn, "run-a")
        _insert_backtest_run(conn, "run-b")
        conn.execute(
            "INSERT INTO backtest_signals "
            "(run_id, signal_id, symbol, signal_date, action, reason, created_at) "
            "VALUES ('run-a', 'signal-a', 'AAA', '2024-01-02', 'buy', 'valid', "
            "'2024-01-02T00:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO backtest_signals "
            "(run_id, signal_id, symbol, signal_date, action, reason, created_at) "
            "VALUES ('run-b', 'signal-b', 'AAA', '2024-01-02', 'buy', 'valid', "
            "'2024-01-02T00:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO backtest_orders "
            "(order_id, run_id, signal_id, symbol, side, signal_date, status, reason, created_at) "
            "VALUES ('order-a', 'run-a', 'signal-a', 'AAA', 'buy', '2024-01-02', "
            "'filled', 'valid', '2024-01-02T00:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO backtest_orders "
            "(order_id, run_id, signal_id, symbol, side, signal_date, status, reason, created_at) "
            "VALUES ('order-b', 'run-b', 'signal-b', 'AAA', 'buy', '2024-01-02', "
            "'filled', 'valid', '2024-01-02T00:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO backtest_fills "
            "(fill_id, run_id, order_id, symbol, fill_date, side, quantity, reference_price, "
            "fill_price, commission, slippage, created_at) VALUES "
            "('fill-a', 'run-a', 'order-a', 'AAA', '2024-01-03', 'buy', 1.0, 100.0, "
            "100.0, 0.0, 0.0, '2024-01-03T00:00:00+00:00')"
        )

        invalid_statements = (
            (
                "INSERT INTO backtest_orders "
                "(order_id, run_id, signal_id, symbol, side, signal_date, status, reason, created_at) "
                "VALUES ('order-cross-run', 'run-a', 'signal-b', 'AAA', 'buy', '2024-01-02', "
                "'pending', 'invalid', '2024-01-02T00:00:00+00:00')"
            ),
            (
                "INSERT INTO backtest_orders "
                "(order_id, run_id, signal_id, symbol, side, signal_date, status, reason, created_at) "
                "VALUES ('order-cross-symbol', 'run-a', 'signal-a', 'BBB', 'buy', '2024-01-02', "
                "'pending', 'invalid', '2024-01-02T00:00:00+00:00')"
            ),
            (
                "INSERT INTO backtest_fills "
                "(fill_id, run_id, order_id, symbol, fill_date, side, quantity, "
                "reference_price, fill_price, commission, slippage, created_at) VALUES "
                "('fill-cross-run', 'run-a', 'order-b', 'AAA', '2024-01-03', 'buy', 1.0, "
                "100.0, 100.0, 0.0, 0.0, '2024-01-03T00:00:00+00:00')"
            ),
            (
                "INSERT INTO backtest_fills "
                "(fill_id, run_id, order_id, symbol, fill_date, side, quantity, "
                "reference_price, fill_price, commission, slippage, created_at) VALUES "
                "('fill-cross-symbol', 'run-a', 'order-a', 'BBB', '2024-01-03', 'buy', 1.0, "
                "100.0, 100.0, 0.0, 0.0, '2024-01-03T00:00:00+00:00')"
            ),
        )
        for statement in invalid_statements:
            with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
                conn.execute(statement)

        assert conn.execute("SELECT COUNT(*) FROM backtest_orders").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM backtest_fills").fetchone()[0] == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert (
            "backtest_signals",
            ("run_id", "signal_id", "symbol"),
            ("run_id", "signal_id", "symbol"),
            "CASCADE",
        ) in _foreign_keys(conn, "backtest_orders")
        assert (
            "backtest_orders",
            ("run_id", "order_id", "symbol"),
            ("run_id", "order_id", "symbol"),
            "CASCADE",
        ) in _foreign_keys(conn, "backtest_fills")
    finally:
        conn.close()
