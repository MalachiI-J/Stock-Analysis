"""Transactional persistence for predict-v3 runs.

``PredictionRunResult`` was previously print-and-discard (see
stock_scrapper/prediction/service.py) — every run's walk-forward holdout
accuracy/Brier score vanished the moment the process exited, so there was no
way to see whether the model's honest, no-edge-so-far result was improving,
worsening, or holding steady as more history accumulated. This mirrors
``persist_backtest``'s ``(conn, result, ...)`` + run/child-table shape so a
run and its per-fold walk-forward results are one atomic write.

Also persists evaluation provenance (dataset/symbol/feature/config hashes,
git revision) so two runs over identical data are identifiable as reruns
rather than two independent pieces of accumulating evidence.
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
    configuration_hash: str | None = None,
    git_commit_hash: str | None = None,
    git_dirty: bool | None = None,
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
                started_at, completed_at, error_summary,
                dataset_fingerprint, symbol_universe_hash, feature_set_hash,
                configuration_hash, git_commit_hash, git_dirty
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                INSERT INTO prediction_folds (
                    run_id, fold_number, training_samples, test_samples, accuracy, brier_score,
                    training_start_date, training_end_date, test_start_date, test_end_date,
                    training_symbol_count, test_symbol_count, purged_samples,
                    training_positive_rate, test_positive_rate, baseline_accuracy,
                    baseline_brier_score, accuracy_vs_baseline, brier_improvement_vs_baseline
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    fold.fold,
                    fold.training_samples,
                    fold.test_samples,
                    fold.accuracy,
                    fold.brier_score,
                    fold.training_start_date,
                    fold.training_end_date,
                    fold.test_start_date,
                    fold.test_end_date,
                    fold.training_symbol_count,
                    fold.test_symbol_count,
                    fold.purged_samples,
                    fold.training_positive_rate,
                    fold.test_positive_rate,
                    fold.baseline_accuracy,
                    fold.baseline_brier_score,
                    fold.accuracy_vs_baseline,
                    fold.brier_improvement_vs_baseline,
                ),
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


def find_prediction_runs_by_fingerprint(conn: sqlite3.Connection, dataset_fingerprint: str) -> list[dict[str, Any]]:
    """Return every prior run sharing one dataset fingerprint — i.e. reruns over
    identical assembled features/labels, not independent new evidence."""
    return [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM prediction_runs WHERE dataset_fingerprint = ? "
            "ORDER BY COALESCE(completed_at, started_at) DESC",
            (dataset_fingerprint,),
        ).fetchall()
    ]
