"""Transactional persistence for predict-v4 (gradient-boosted regression) runs.

Mirrors ``persistence.py``'s ``(conn, result, ...)`` + run/child-table shape exactly,
against the separate ``gbm_prediction_runs``/``gbm_prediction_folds`` tables (see
``stock_scrapper/migrations/migration_manager.py``'s ``_ensure_gbm_prediction_tables``
for why these are separate from ``prediction_runs``/``prediction_folds`` rather than
overloaded columns on the same tables) — so predict-v4's MSE/information-coefficient
track record accumulates over time exactly the way predict-v3's accuracy/Brier score
already does, and both are equally subject to the "rerun over identical data isn't new
evidence" provenance check.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Mapping

from stock_scrapper.prediction.gbm_service import GbmPredictionRunResult
from stock_scrapper.utilities.hashing import canonical_json


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def persist_gbm_prediction_run(
    conn: sqlite3.Connection,
    run_id: str,
    result: GbmPredictionRunResult,
    *,
    prediction_version: str,
    benchmark_symbol: str | None,
    configuration_snapshot: Mapping[str, Any] | None = None,
    configuration_hash: str | None = None,
    git_commit_hash: str | None = None,
    git_dirty: bool | None = None,
    started_at: str | None = None,
) -> None:
    """Atomically save one predict-v4 run and its walk-forward folds."""
    conn.execute("SAVEPOINT persist_gbm_prediction_run")
    try:
        conn.execute(
            """
            INSERT INTO gbm_prediction_runs (
                run_id, prediction_version, as_of_date, horizon_days, benchmark_symbol,
                status, message, training_samples, holdout_samples, training_start_date,
                training_end_date, holdout_mse, holdout_mean_absolute_error,
                holdout_information_coefficient, baseline_mse, feature_importances_json,
                predictions_json, configuration_snapshot_json,
                started_at, completed_at, error_summary,
                dataset_fingerprint, symbol_universe_hash, feature_set_hash,
                configuration_hash, git_commit_hash, git_dirty
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                result.holdout_mse,
                result.holdout_mean_absolute_error,
                result.holdout_information_coefficient,
                result.baseline_mse,
                canonical_json([list(pair) for pair in result.feature_importances]),
                canonical_json([asdict(prediction) for prediction in result.predictions]),
                canonical_json(dict(configuration_snapshot) if configuration_snapshot else {}),
                started_at or _utc_now(),
                _utc_now(),
                result.message if result.status != "ok" else None,
                result.dataset_fingerprint,
                result.symbol_universe_hash,
                result.feature_set_hash,
                configuration_hash,
                git_commit_hash,
                None if git_dirty is None else int(git_dirty),
            ),
        )
        for fold in result.walk_forward_folds:
            conn.execute(
                """
                INSERT INTO gbm_prediction_folds (
                    run_id, fold_number, training_samples, test_samples,
                    mse, mean_absolute_error, information_coefficient,
                    training_start_date, training_end_date, test_start_date, test_end_date,
                    training_symbol_count, test_symbol_count, purged_samples,
                    baseline_mse, mse_improvement_vs_baseline
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    fold.fold,
                    fold.training_samples,
                    fold.test_samples,
                    fold.mse,
                    fold.mean_absolute_error,
                    fold.information_coefficient,
                    fold.training_start_date,
                    fold.training_end_date,
                    fold.test_start_date,
                    fold.test_end_date,
                    fold.training_symbol_count,
                    fold.test_symbol_count,
                    fold.purged_samples,
                    fold.baseline_mse,
                    fold.mse_improvement_vs_baseline,
                ),
            )
        conn.execute("RELEASE SAVEPOINT persist_gbm_prediction_run")
    except Exception:
        conn.execute("ROLLBACK TO SAVEPOINT persist_gbm_prediction_run")
        conn.execute("RELEASE SAVEPOINT persist_gbm_prediction_run")
        raise


def list_gbm_prediction_runs(conn: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    """Return recent persisted predict-v4 runs, most recent first."""
    return [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM gbm_prediction_runs ORDER BY COALESCE(completed_at, started_at) DESC LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
    ]


def load_gbm_prediction_run(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    """Load one saved predict-v4 run and its walk-forward folds."""
    run = conn.execute("SELECT * FROM gbm_prediction_runs WHERE run_id = ?", (run_id,)).fetchone()
    if run is None:
        return None
    payload = dict(run)
    payload["folds"] = [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM gbm_prediction_folds WHERE run_id = ? ORDER BY fold_number", (run_id,)
        ).fetchall()
    ]
    return payload


def find_gbm_prediction_runs_by_fingerprint(conn: sqlite3.Connection, dataset_fingerprint: str) -> list[dict[str, Any]]:
    """Return every prior predict-v4 run sharing one dataset fingerprint — i.e. reruns
    over identical assembled features/targets, not independent new evidence."""
    return [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM gbm_prediction_runs WHERE dataset_fingerprint = ? "
            "ORDER BY COALESCE(completed_at, started_at) DESC",
            (dataset_fingerprint,),
        ).fetchall()
    ]
