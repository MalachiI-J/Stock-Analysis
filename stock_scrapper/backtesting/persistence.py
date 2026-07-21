"""Transactional persistence for portfolio backtests and reports."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from stock_scrapper.utilities.hashing import canonical_json


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Cannot persist {type(value).__name__}")


def persist_backtest(
    conn: sqlite3.Connection,
    run: Any,
    signals: Sequence[Any],
    rejected_candidates: Sequence[Any],
    orders: Sequence[Any],
    fills: Sequence[Any],
    trades: Sequence[Any],
    snapshots: Sequence[Any],
    metrics: Any,
) -> None:
    """Atomically save one complete simulation and every linked record."""
    payload = _mapping(run)
    order_payloads = [_mapping(item) for item in orders]
    accepted_signal_ids = {
        str(item.get("signal_id"))
        for item in order_payloads
        if item.get("signal_id") is not None and item.get("status") == "filled"
    }
    order_rejections = {
        str(item.get("signal_id")): str(item.get("rejection_reason"))
        for item in order_payloads
        if item.get("signal_id") is not None and item.get("rejection_reason")
    }
    rejected_by_id = {
        str(item.get("signal_id")): item
        for item in (_mapping(candidate) for candidate in rejected_candidates)
        if item.get("signal_id") is not None
    }
    conn.execute("SAVEPOINT persist_backtest")
    try:
        conn.execute(
            """
            INSERT INTO backtest_runs (
                run_id, strategy_name, strategy_version, started_at, completed_at, status,
                start_date, end_date, warmup_start_date, benchmark_symbol, initial_cash,
                ending_equity, symbols_json, configuration_hash, configuration_snapshot_json,
                data_hash, deterministic_result_hash, error_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["run_id"],
                payload["strategy_name"],
                payload["strategy_version"],
                payload.get("started_at") or _utc_now(),
                payload.get("completed_at"),
                payload.get("status", "completed"),
                payload["start_date"],
                payload["end_date"],
                payload.get("warm_up_start_date"),
                payload.get("benchmark_symbol") or payload.get("benchmark"),
                payload["initial_cash"],
                payload.get("ending_equity"),
                canonical_json(payload.get("symbols", [])),
                payload["configuration_hash"],
                canonical_json(payload.get("configuration_snapshot", {})),
                payload.get("price_data_hash") or payload.get("data_hash") or "unavailable",
                payload.get("deterministic_result_hash"),
                payload.get("error_summary"),
            ),
        )

        for signal in signals:
            item = _mapping(signal)
            signal_id = str(item.get("signal_id") or "")
            rejection = rejected_by_id.get(signal_id)
            conn.execute(
                """
                INSERT INTO backtest_signals (
                    run_id, signal_id, symbol, signal_date, action, classification,
                    opportunity_score, risk_score, confidence_score, market_regime,
                    ranking_json, reason, accepted, rejection_reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["run_id"],
                    signal_id,
                    item["symbol"],
                    item["signal_date"],
                    item["action"],
                    item.get("classification"),
                    item.get("opportunity_score"),
                    item.get("risk_score"),
                    item.get("confidence_score"),
                    item.get("market_regime"),
                    canonical_json(item.get("ranking_values", {})),
                    item.get("reason") or "Unspecified signal",
                    int(signal_id in accepted_signal_ids),
                    rejection.get("reason") if rejection else order_rejections.get(signal_id),
                    _utc_now(),
                ),
            )

        for item in order_payloads:
            conn.execute(
                """
                INSERT INTO backtest_orders (
                    order_id, run_id, signal_id, symbol, side, signal_date, scheduled_date,
                    status, quantity, reference_price, reason, rejection_reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item["order_id"],
                    payload["run_id"],
                    item.get("signal_id"),
                    item["symbol"],
                    item["side"],
                    item["signal_date"],
                    item.get("scheduled_execution_date") or item.get("scheduled_date"),
                    item.get("status", "pending"),
                    item.get("quantity"),
                    item.get("reference_price"),
                    item.get("reason") or "Unspecified order",
                    item.get("rejection_reason"),
                    item.get("created_at") or _utc_now(),
                ),
            )

        for fill in fills:
            item = _mapping(fill)
            conn.execute(
                """
                INSERT INTO backtest_fills (
                    fill_id, run_id, order_id, symbol, fill_date, side, quantity,
                    reference_price, fill_price, commission, slippage, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item["fill_id"],
                    payload["run_id"],
                    item["order_id"],
                    item["symbol"],
                    item.get("execution_date") or item.get("fill_date"),
                    item["side"],
                    item["quantity"],
                    item["reference_price"],
                    item["fill_price"],
                    item["commission"],
                    item["slippage"],
                    item.get("created_at") or _utc_now(),
                ),
            )

        for trade in trades:
            item = _mapping(trade)
            conn.execute(
                """
                INSERT INTO backtest_trades (
                    trade_id, run_id, symbol, signal_date, entry_date, exit_signal_date,
                    exit_date, quantity, entry_reference_price, entry_fill_price,
                    exit_reference_price, exit_fill_price, entry_commission, exit_commission,
                    slippage_cost, realized_pnl, return_pct, holding_days, entry_reason,
                    exit_reason, classification, market_regime, opportunity_score, risk_score,
                    confidence_score, ranking_json, ambiguous_daily_bar, strategy_version,
                    configuration_hash, entry_signal_id, entry_order_id, exit_order_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item["trade_id"],
                    payload["run_id"],
                    item["symbol"],
                    item["signal_date"],
                    item.get("execution_date") or item.get("entry_date"),
                    item.get("exit_signal_date"),
                    item.get("exit_execution_date") or item.get("exit_date"),
                    item["quantity"],
                    item.get("reference_price") or item.get("entry_reference_price"),
                    item.get("fill_price") or item.get("entry_fill_price"),
                    item.get("exit_reference_price"),
                    item.get("exit_fill_price"),
                    item.get("commission", 0.0),
                    item.get("exit_commission", 0.0),
                    float(item.get("slippage", 0.0)) + float(item.get("exit_slippage", 0.0)),
                    item.get("realized_pnl"),
                    item.get("return_pct"),
                    item.get("holding_period_days") or item.get("holding_days"),
                    item.get("entry_reason") or "Unspecified entry",
                    item.get("exit_reason"),
                    item.get("classification"),
                    item.get("market_regime"),
                    item.get("opportunity_score"),
                    item.get("risk_score"),
                    item.get("confidence_score"),
                    canonical_json(item.get("ranking_values", {})),
                    int(bool(item.get("ambiguous_daily_bar"))),
                    item["strategy_version"],
                    item["configuration_hash"],
                    item.get("entry_signal_id"),
                    item.get("entry_order_id"),
                    item.get("exit_order_id"),
                ),
            )

        for snapshot in snapshots:
            item = _mapping(snapshot)
            conn.execute(
                """
                INSERT INTO backtest_equity_curve (
                    run_id, trade_date, cash, reserved_cash, market_value, unrealized_pnl,
                    realized_pnl, equity, gross_exposure, position_count, daily_return,
                    benchmark_equity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["run_id"],
                    item.get("snapshot_date") or item.get("trade_date"),
                    item["cash"],
                    item.get("reserved_cash", 0.0),
                    item["market_value"],
                    item.get("unrealized_pnl", 0.0),
                    item.get("realized_pnl", 0.0),
                    item["equity"],
                    item["gross_exposure"],
                    item["position_count"],
                    item.get("daily_return"),
                    item.get("benchmark_equity"),
                ),
            )

        for name, value in _mapping(metrics).items():
            scalar = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
            serialized = None if scalar is not None or value is None else canonical_json(value)
            conn.execute(
                "INSERT INTO backtest_metrics (run_id, metric_name, metric_value, metric_json) "
                "VALUES (?, ?, ?, ?)",
                (payload["run_id"], name, scalar, serialized),
            )
        conn.execute("RELEASE SAVEPOINT persist_backtest")
    except Exception:
        conn.execute("ROLLBACK TO SAVEPOINT persist_backtest")
        conn.execute("RELEASE SAVEPOINT persist_backtest")
        raise


def list_backtest_runs(conn: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    """Return recent persisted simulations."""
    return [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM backtest_runs ORDER BY COALESCE(completed_at, started_at) DESC LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
    ]


def load_backtest(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    """Load one simulation and all linked rows without rerunning it."""
    run = conn.execute("SELECT * FROM backtest_runs WHERE run_id = ?", (run_id,)).fetchone()
    if run is None:
        return None
    result = dict(run)
    for key, table, order in (
        ("signals", "backtest_signals", "signal_date, symbol, id"),
        ("orders", "backtest_orders", "signal_date, symbol, order_id"),
        ("fills", "backtest_fills", "fill_date, symbol, fill_id"),
        ("trades", "backtest_trades", "entry_date, symbol, trade_id"),
        ("equity_curve", "backtest_equity_curve", "trade_date"),
    ):
        result[key] = [
            dict(row)
            for row in conn.execute(
                f"SELECT * FROM {table} WHERE run_id = ? ORDER BY {order}", (run_id,)
            ).fetchall()
        ]
    metrics: dict[str, Any] = {}
    for row in conn.execute(
        "SELECT metric_name, metric_value, metric_json FROM backtest_metrics WHERE run_id = ?",
        (run_id,),
    ).fetchall():
        value = row["metric_value"]
        if value is None and row["metric_json"] is not None:
            value = json.loads(row["metric_json"])
        metrics[str(row["metric_name"])] = value
    result["metrics"] = metrics
    return result


def persist_walk_forward(conn: sqlite3.Connection, run: Any) -> None:
    """Atomically persist a walk-forward run and all fixed windows."""
    payload = _mapping(run)
    windows = list(payload.pop("windows", []))
    conn.execute("SAVEPOINT persist_walk_forward")
    try:
        conn.execute(
            """
            INSERT INTO walk_forward_runs (
                run_id, strategy_name, strategy_version, started_at, completed_at,
                status, start_date, end_date, configuration_hash,
                configuration_snapshot_json, benchmark_symbol, symbols_json, error_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.get("walk_forward_run_id") or payload.get("run_id"),
                payload["strategy_name"],
                payload["strategy_version"],
                payload.get("started_at") or _utc_now(),
                payload.get("completed_at"),
                payload["status"],
                payload["start_date"],
                payload["end_date"],
                payload["configuration_hash"],
                canonical_json(payload.get("configuration_snapshot", {})),
                payload.get("benchmark_symbol"),
                canonical_json(payload.get("symbols", [])),
                payload.get("error_summary"),
            ),
        )
        for window_value in windows:
            window = _mapping(window_value)
            window_type = window.get("window_type", "validation")
            evaluation_start = window.get("evaluation_start_date")
            evaluation_end = window.get("evaluation_end_date")
            validation_start = window.get("validation_start_date") or window.get("validation_start")
            validation_end = window.get("validation_end_date") or window.get("validation_end")
            holdout_start = window.get("holdout_start_date") or window.get("holdout_start")
            holdout_end = window.get("holdout_end_date") or window.get("holdout_end")
            if window_type == "validation":
                validation_start = validation_start or evaluation_start
                validation_end = validation_end or evaluation_end
            elif window_type == "holdout":
                holdout_start = holdout_start or evaluation_start
                holdout_end = holdout_end or evaluation_end
            metrics = window.get("metrics")
            if metrics is not None and not isinstance(metrics, Mapping):
                metrics = _mapping(metrics)
            conn.execute(
                """
                INSERT INTO walk_forward_windows (
                    window_id, walk_forward_run_id, sequence_number, window_type,
                    warmup_start, warmup_end, development_start, development_end,
                    validation_start, validation_end, holdout_start, holdout_end,
                    backtest_run_id, status, metrics_json, error_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    window["window_id"],
                    window.get("walk_forward_run_id") or payload.get("walk_forward_run_id"),
                    window.get("window_number") or window.get("sequence_number"),
                    window_type,
                    window.get("warm_up_start_date") or window.get("warmup_start"),
                    window.get("warm_up_end_date") or window.get("warmup_end"),
                    window.get("development_start_date") or window.get("development_start"),
                    window.get("development_end_date") or window.get("development_end"),
                    validation_start,
                    validation_end,
                    holdout_start,
                    holdout_end,
                    window.get("backtest_run_id"),
                    window.get("status", "completed"),
                    canonical_json(metrics) if metrics is not None else None,
                    window.get("error_summary"),
                ),
            )
        conn.execute("RELEASE SAVEPOINT persist_walk_forward")
    except Exception:
        conn.execute("ROLLBACK TO SAVEPOINT persist_walk_forward")
        conn.execute("RELEASE SAVEPOINT persist_walk_forward")
        raise


def load_walk_forward(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    """Load one saved walk-forward validation and its windows."""
    run = conn.execute("SELECT * FROM walk_forward_runs WHERE run_id = ?", (run_id,)).fetchone()
    if run is None:
        return None
    payload = dict(run)
    payload["windows"] = [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM walk_forward_windows WHERE walk_forward_run_id = ? "
            "ORDER BY sequence_number",
            (run_id,),
        ).fetchall()
    ]
    return payload
