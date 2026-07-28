"""Transactional persistence for predict-v3 runs.

``PredictionRunResult`` was previously print-and-discard (see
stock_scrapper/prediction/service.py) — every run's walk-forward holdout
accuracy/Brier score vanished the moment the process exited, so there was no
way to see whether the model's honest, no-edge-so-far result was improving,
worsening, or holding steady as more history accumulated. This mirrors
``persist_backtest``'s ``(conn, result, ...)`` + run/child-table shape so a
run and its per-fold walk-forward results are one atomic write.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Mapping

from stock_scrapper.prediction.service import PredictionRunResult
from stock_scrapper.utilities.hashing import canonical_json


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def persist_prediction_run(
    conn: sqlite3.Connection,
    run_id: str,
    result: PredictionRunResult,
    *,
    prediction_version: str,
    benchmark_symbol: str | None,
    configuration_snapshot: Mapping[str, Any] | None = None,
    started_at: str | None = None,
) -> None:
    """Atomically save one prediction run and its walk-forward folds."""
    conn.execute("SAVEPOINT persist_prediction_run")
    try:
        conn.execute(
            """
            INSERT INTO prediction_runs (
                run_id, prediction_version, as_of_date, horizon_days, benchmark_symbol,
                status, message, training_samples, holdout_samples, training_start_date,
                training_end_date, positive_label_rate, holdout_accuracy, holdout_brier_score,
                coefficients_json, predictions_json, configuration_snapshot_json,
                started_at, completed_at, error_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                prediction_version,
                result.as_of_date,
                result.horizon_days,
                benchmark_symbol,
                result.status,
                result.message,
                result.training_samples,
                result.holdout_samples,
                result.training_start_date,
                result.training_end_date,
                result.positive_label_rate,
                result.holdout_accuracy,
                result.holdout_brier_score,
                canonical_json([list(pair) for pair in result.coefficients]),
                canonical_json([asdict(prediction) for prediction in result.predictions]),
                canonical_json(dict(configuration_snapshot) if configuration_snapshot else {}),
                started_at or _utc_now(),
                _utc_now(),
                result.message if result.status != "ok" else None,
            ),
        )
        for fold in result.walk_forward_folds:
            conn.execute(
                """
                INSERT INTO prediction_folds (
                    run_id, fold_number, training_samples, test_samples, accuracy, brier_score
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, fold.fold, fold.training_samples, fold.test_samples, fold.accuracy, fold.brier_score),
            )
        conn.execute("RELEASE SAVEPOINT persist_prediction_run")
    except Exception:
        conn.execute("ROLLBACK TO SAVEPOINT persist_prediction_run")
        conn.execute("RELEASE SAVEPOINT persist_prediction_run")
        raise


def list_prediction_runs(conn: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    """Return recent persisted prediction runs, most recent first."""
    return [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM prediction_runs ORDER BY COALESCE(completed_at, started_at) DESC LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
    ]


def load_prediction_run(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    """Load one saved prediction run and its walk-forward folds."""
    run = conn.execute("SELECT * FROM prediction_runs WHERE run_id = ?", (run_id,)).fetchone()
    if run is None:
        return None
    payload = dict(run)
    payload["folds"] = [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM prediction_folds WHERE run_id = ? ORDER BY fold_number", (run_id,)
        ).fetchall()
    ]
    return payload
