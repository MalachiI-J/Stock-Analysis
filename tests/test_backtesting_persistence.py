from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from stock_scrapper.backtesting.engine import PortfolioBacktestResult, run_portfolio_backtest
from stock_scrapper.backtesting.persistence import (
    list_backtest_runs,
    load_backtest,
    load_walk_forward,
    persist_backtest,
    persist_walk_forward,
)
from stock_scrapper.backtesting.reporting import write_backtest_reports
from stock_scrapper.database import create_connection, initialize_database
from tests.test_backtesting_engine import _analysis_rules, _config, _histories, _install_analysis


TABLES = (
    "backtest_runs",
    "backtest_signals",
    "backtest_orders",
    "backtest_fills",
    "backtest_trades",
    "backtest_equity_curve",
    "backtest_metrics",
)


def _database(tmp_path: Path) -> sqlite3.Connection:
    path = tmp_path / "market.db"
    initialize_database(path)
    return create_connection(path)


def _simulate(
    monkeypatch: pytest.MonkeyPatch,
    *,
    persist_conn: sqlite3.Connection | None = None,
    run_id: str = "persisted-run",
    commit_persistence: bool = True,
) -> PortfolioBacktestResult:
    sessions = ["2024-01-02", "2024-01-03", "2024-01-04"]
    symbols = ["AAA"]
    histories = _histories(
        symbols,
        sessions,
        {
            ("AAA", sessions[1]): {"open_price": 100.0, "high": 106.0, "low": 99.0, "close": 105.0},
            ("AAA", sessions[2]): {"open_price": 109.0, "high": 111.0, "low": 108.0, "close": 110.0},
        },
    )
    config = _config(sessions, commission_bps=10.0, minimum_commission=1.0, slippage_bps=5.0)
    _install_analysis(monkeypatch)
    return run_portfolio_backtest(
        symbols,
        histories,
        _analysis_rules(),
        config,
        persist_conn=persist_conn,
        commit_persistence=commit_persistence,
        run_id=run_id,
    )


def _counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in TABLES}


def test_complete_backtest_persists_every_linked_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    conn = _database(tmp_path)
    try:
        result = _simulate(monkeypatch, persist_conn=conn)
        expected = {
            "backtest_runs": 1,
            "backtest_signals": len(result.signals),
            "backtest_orders": len(result.orders),
            "backtest_fills": len(result.fills),
            "backtest_trades": len(result.trades),
            "backtest_equity_curve": len(result.snapshots),
            "backtest_metrics": len(result.metrics.to_dict()),
        }
        assert _counts(conn) == expected
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []

        run_id = result.run.run_id
        for table in TABLES[1:]:
            assert conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE run_id <> ?", (run_id,)
            ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM backtest_orders AS o "
            "JOIN backtest_signals AS s ON s.run_id=o.run_id AND s.signal_id=o.signal_id "
            "WHERE o.run_id=?",
            (run_id,),
        ).fetchone()[0] == len(result.orders)
        assert conn.execute(
            "SELECT COUNT(*) FROM backtest_fills AS f "
            "JOIN backtest_orders AS o ON o.order_id=f.order_id "
            "WHERE f.run_id=?",
            (run_id,),
        ).fetchone()[0] == len(result.fills)

        saved = load_backtest(conn, run_id)
        assert saved is not None
        assert saved["run_id"] == run_id
        assert saved["deterministic_result_hash"] == result.run.deterministic_result_hash
        assert len(saved["signals"]) == len(result.signals)
        assert len(saved["orders"]) == len(result.orders)
        assert len(saved["fills"]) == len(result.fills)
        assert len(saved["trades"]) == len(result.trades)
        assert len(saved["equity_curve"]) == len(result.snapshots)
        assert saved["metrics"]["ending_equity"] == pytest.approx(result.metrics.ending_equity)
        assert [row["run_id"] for row in list_backtest_runs(conn)] == [run_id]
    finally:
        conn.close()


def test_persistence_failure_rolls_back_the_run_and_all_children(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    conn = _database(tmp_path)
    try:
        result = _simulate(monkeypatch, run_id="source-run")
        run_payload = result.run.to_dict()
        run_payload["run_id"] = "rollback-run"
        bad_fill = {
            "fill_id": "orphan-fill",
            "order_id": "missing-order",
            "symbol": "AAA",
            "execution_date": "2024-01-03",
            "side": "buy",
            "quantity": 1.0,
            "reference_price": 100.0,
            "fill_price": 100.0,
            "commission": 0.0,
            "slippage": 0.0,
        }

        with pytest.raises(sqlite3.IntegrityError):
            persist_backtest(
                conn,
                run_payload,
                signals=[],
                rejected_candidates=[],
                orders=[],
                fills=[bad_fill],
                trades=[],
                snapshots=[],
                metrics={},
            )

        assert _counts(conn) == {table: 0 for table in TABLES}
        assert load_backtest(conn, "rollback-run") is None
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def test_backtest_persistence_can_join_and_roll_back_a_larger_transaction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    conn = _database(tmp_path)
    try:
        conn.execute("BEGIN")
        _simulate(
            monkeypatch,
            persist_conn=conn,
            run_id="deferred-run",
            commit_persistence=False,
        )
        assert conn.execute(
            "SELECT COUNT(*) FROM backtest_runs WHERE run_id='deferred-run'"
        ).fetchone()[0] == 1

        conn.rollback()

        assert _counts(conn) == {table: 0 for table in TABLES}
    finally:
        conn.close()


def test_execution_rejected_order_does_not_mark_signal_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    conn = _database(tmp_path)
    try:
        result = _simulate(monkeypatch, run_id="source-rejection")
        run_payload = result.run.to_dict()
        run_payload["run_id"] = "persisted-rejection"
        order = result.orders[0].to_dict()
        order["status"] = "rejected"
        order["rejection_reason"] = "Adjusted next-session open is unavailable"

        persist_backtest(
            conn,
            run_payload,
            signals=result.signals,
            rejected_candidates=[],
            orders=[order],
            fills=[],
            trades=[],
            snapshots=[],
            metrics={},
        )
        conn.commit()

        signal = conn.execute(
            "SELECT accepted, rejection_reason FROM backtest_signals "
            "WHERE run_id=? AND signal_id=?",
            (run_payload["run_id"], order["signal_id"]),
        ).fetchone()
        saved_order = conn.execute(
            "SELECT status, rejection_reason FROM backtest_orders WHERE run_id=?",
            (run_payload["run_id"],),
        ).fetchone()
        assert tuple(signal) == (0, order["rejection_reason"])
        assert tuple(saved_order) == ("rejected", order["rejection_reason"])
    finally:
        conn.close()


def test_report_regeneration_does_not_duplicate_or_mutate_persisted_simulation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    conn = _database(tmp_path)
    try:
        result = _simulate(monkeypatch, persist_conn=conn, run_id="report-run")
        saved = load_backtest(conn, result.run.run_id)
        assert saved is not None
        baseline_counts = _counts(conn)

        report_dir = tmp_path / "reports"
        first_paths = write_backtest_reports(report_dir, saved)
        first_bytes = {name: path.read_bytes() for name, path in first_paths.items()}
        second_paths = write_backtest_reports(report_dir, saved)

        assert second_paths == first_paths
        assert {name: path.read_bytes() for name, path in second_paths.items()} == first_bytes
        assert len(list(report_dir.iterdir())) == len(first_paths)
        assert _counts(conn) == baseline_counts
        assert conn.execute("SELECT COUNT(*) FROM backtest_runs WHERE run_id=?", (result.run.run_id,)).fetchone()[0] == 1
        assert load_backtest(conn, result.run.run_id) == saved
    finally:
        conn.close()


def _walk_forward_payload(*, duplicate_sequence: bool = False) -> dict[str, object]:
    windows = [
        {
            "window_id": "wf-run-window-0001",
            "walk_forward_run_id": "wf-run",
            "window_number": 1,
            "window_type": "validation",
            "warm_up_start_date": "2023-01-02",
            "warm_up_end_date": "2023-12-29",
            "development_start_date": "2024-01-02",
            "development_end_date": "2024-06-28",
            "evaluation_start_date": "2024-07-01",
            "evaluation_end_date": "2024-12-31",
            "validation_start_date": "2024-07-01",
            "validation_end_date": "2024-12-31",
            "status": "completed",
            "metrics": {"total_return": 0.12},
        },
        {
            "window_id": "wf-run-window-0002",
            "walk_forward_run_id": "wf-run",
            "window_number": 2,
            "window_type": "holdout",
            "warm_up_start_date": "2023-07-03",
            "warm_up_end_date": "2024-06-28",
            "development_start_date": "2024-07-01",
            "development_end_date": "2024-12-31",
            "evaluation_start_date": "2025-01-02",
            "evaluation_end_date": "2025-06-30",
            "holdout_start_date": "2025-01-02",
            "holdout_end_date": "2025-06-30",
            "status": "completed",
            "metrics": {"total_return": 0.04},
        },
    ]
    if duplicate_sequence:
        duplicate = dict(windows[0])
        duplicate["window_id"] = "wf-run-window-duplicate"
        windows.append(duplicate)
    return {
        "walk_forward_run_id": "wf-run",
        "strategy_name": "score_v1",
        "strategy_version": "1.0.0",
        "started_at": "2025-01-01T00:00:00+00:00",
        "completed_at": "2025-01-01T00:01:00+00:00",
        "status": "completed",
        "start_date": "2024-01-02",
        "end_date": "2024-12-31",
        "configuration_hash": "a" * 64,
        "configuration_snapshot": {"strategy_name": "score_v1"},
        "benchmark_symbol": "SPY",
        "symbols": ["AAPL", "SPY"],
        "windows": windows,
    }


def test_walk_forward_persistence_round_trip_and_foreign_keys(tmp_path: Path) -> None:
    conn = _database(tmp_path)
    try:
        persist_walk_forward(conn, _walk_forward_payload())
        conn.commit()

        saved = load_walk_forward(conn, "wf-run")
        assert saved is not None
        assert saved["status"] == "completed"
        assert len(saved["windows"]) == 2
        assert json.loads(saved["windows"][0]["metrics_json"])["total_return"] == 0.12
        assert saved["windows"][1]["validation_start"] is None
        assert saved["windows"][1]["validation_end"] is None
        assert saved["windows"][1]["holdout_start"] == "2025-01-02"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def test_walk_forward_persistence_failure_rolls_back_parent_and_windows(tmp_path: Path) -> None:
    conn = _database(tmp_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            persist_walk_forward(conn, _walk_forward_payload(duplicate_sequence=True))

        assert conn.execute("SELECT COUNT(*) FROM walk_forward_runs").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM walk_forward_windows").fetchone()[0] == 0
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()
