from __future__ import annotations

import sqlite3
from pathlib import Path

from stock_scrapper.database import create_connection, initialize_database
from stock_scrapper.prediction.persistence import (
    find_prediction_runs_by_fingerprint,
    list_prediction_runs,
    load_prediction_run,
    persist_prediction_run,
)
from stock_scrapper.prediction.service import PredictionRunResult, SymbolPrediction, WalkForwardFold


def _database(tmp_path: Path) -> sqlite3.Connection:
    path = tmp_path / "market.db"
    initialize_database(path)
    return create_connection(path)


def _result(**overrides: object) -> PredictionRunResult:
    defaults: dict[str, object] = dict(
        status="ok",
        message=None,
        as_of_date="2026-06-30",
        horizon_days=21,
        training_samples=100,
        holdout_samples=40,
        training_start_date="2023-01-03",
        training_end_date="2026-05-29",
        positive_label_rate=0.53,
        holdout_accuracy=0.491,
        holdout_brier_score=0.2659,
        walk_forward_folds=[
            WalkForwardFold(
                fold=1, training_samples=50, test_samples=20, accuracy=0.46, brier_score=0.31,
                training_start_date="2023-01-03", training_end_date="2024-06-01",
                test_start_date="2024-06-02", test_end_date="2024-12-01",
                training_symbol_count=8, test_symbol_count=6, purged_samples=3,
                training_positive_rate=0.5, test_positive_rate=0.55,
                baseline_accuracy=0.55, baseline_brier_score=0.2475,
                accuracy_vs_baseline=-0.09, brier_improvement_vs_baseline=-0.0625,
            ),
            WalkForwardFold(fold=2, training_samples=70, test_samples=20, accuracy=0.50, brier_score=0.25),
        ],
        coefficients=[("rsi_14", 0.21), ("beta", -0.05)],
        predictions=[
            SymbolPrediction("AAPL", 0.535, None),
            SymbolPrediction("MISSING", None, "One or more required indicators are unavailable"),
        ],
        dataset_fingerprint="fingerprint-abc123",
        symbol_universe_hash="universe-abc123",
        feature_set_hash="features-abc123",
    )
    defaults.update(overrides)
    return PredictionRunResult(**defaults)


def test_persist_prediction_run_saves_run_and_folds(tmp_path: Path) -> None:
    conn = _database(tmp_path)
    try:
        persist_prediction_run(
            conn, "predict-test-001", _result(),
            prediction_version="predict-v3", benchmark_symbol="SPY",
            configuration_snapshot={"horizon_days": 21},
        )
        conn.commit()

        run_count = conn.execute("SELECT COUNT(*) FROM prediction_runs").fetchone()[0]
        fold_count = conn.execute("SELECT COUNT(*) FROM prediction_folds").fetchone()[0]
        assert run_count == 1
        assert fold_count == 2
    finally:
        conn.close()


def test_load_prediction_run_returns_run_with_ordered_folds(tmp_path: Path) -> None:
    conn = _database(tmp_path)
    try:
        persist_prediction_run(
            conn, "predict-test-002", _result(),
            prediction_version="predict-v3", benchmark_symbol="SPY",
        )
        conn.commit()

        loaded = load_prediction_run(conn, "predict-test-002")
        assert loaded is not None
        assert loaded["prediction_version"] == "predict-v3"
        assert loaded["holdout_accuracy"] == 0.491
        assert [fold["fold_number"] for fold in loaded["folds"]] == [1, 2]
        assert loaded["folds"][0]["accuracy"] == 0.46
    finally:
        conn.close()


def test_load_prediction_run_returns_none_for_unknown_id(tmp_path: Path) -> None:
    conn = _database(tmp_path)
    try:
        assert load_prediction_run(conn, "does-not-exist") is None
    finally:
        conn.close()


def test_list_prediction_runs_orders_most_recent_first(tmp_path: Path) -> None:
    conn = _database(tmp_path)
    try:
        persist_prediction_run(
            conn, "predict-older", _result(as_of_date="2026-05-29"),
            prediction_version="predict-v3", benchmark_symbol="SPY", started_at="2026-05-29T00:00:00+00:00",
        )
        conn.commit()
        persist_prediction_run(
            conn, "predict-newer", _result(as_of_date="2026-06-30"),
            prediction_version="predict-v3", benchmark_symbol="SPY", started_at="2026-06-30T00:00:00+00:00",
        )
        conn.commit()

        runs = list_prediction_runs(conn)
        assert [run["run_id"] for run in runs] == ["predict-newer", "predict-older"]
    finally:
        conn.close()


def test_persist_prediction_run_rolls_back_atomically_on_failure(tmp_path: Path) -> None:
    conn = _database(tmp_path)
    try:
        persist_prediction_run(
            conn, "predict-dup", _result(),
            prediction_version="predict-v3", benchmark_symbol="SPY",
        )
        conn.commit()

        try:
            persist_prediction_run(
                conn, "predict-dup", _result(),  # duplicate run_id violates the primary key
                prediction_version="predict-v3", benchmark_symbol="SPY",
            )
        except sqlite3.IntegrityError:
            pass

        run_count = conn.execute("SELECT COUNT(*) FROM prediction_runs").fetchone()[0]
        fold_count = conn.execute("SELECT COUNT(*) FROM prediction_folds").fetchone()[0]
        assert run_count == 1
        assert fold_count == 2  # the failed second attempt must not leave orphaned fold rows
    finally:
        conn.close()


def test_persist_prediction_run_handles_insufficient_data_status(tmp_path: Path) -> None:
    conn = _database(tmp_path)
    try:
        result = PredictionRunResult(
            status="insufficient_data",
            message="Only 5 training sample(s) were available; 200 are required.",
            as_of_date="2026-06-30",
            horizon_days=21,
            training_samples=5,
        )
        persist_prediction_run(
            conn, "predict-insufficient", result,
            prediction_version="predict-v3", benchmark_symbol="SPY",
        )
        conn.commit()

        loaded = load_prediction_run(conn, "predict-insufficient")
        assert loaded is not None
        assert loaded["status"] == "insufficient_data"
        assert loaded["error_summary"] == result.message
        assert loaded["folds"] == []
    finally:
        conn.close()


def test_persist_prediction_run_saves_provenance_and_per_fold_detail(tmp_path: Path) -> None:
    conn = _database(tmp_path)
    try:
        persist_prediction_run(
            conn, "predict-provenance-001", _result(),
            prediction_version="predict-v3", benchmark_symbol="SPY",
            configuration_snapshot={"horizon_days": 21},
            configuration_hash="config-hash-abc",
            git_commit_hash="deadbeef",
            git_dirty=True,
        )
        conn.commit()

        loaded = load_prediction_run(conn, "predict-provenance-001")
        assert loaded is not None
        assert loaded["dataset_fingerprint"] == "fingerprint-abc123"
        assert loaded["symbol_universe_hash"] == "universe-abc123"
        assert loaded["feature_set_hash"] == "features-abc123"
        assert loaded["configuration_hash"] == "config-hash-abc"
        assert loaded["git_commit_hash"] == "deadbeef"
        assert loaded["git_dirty"] == 1

        first_fold = loaded["folds"][0]
        assert first_fold["training_start_date"] == "2023-01-03"
        assert first_fold["test_start_date"] == "2024-06-02"
        assert first_fold["training_symbol_count"] == 8
        assert first_fold["test_symbol_count"] == 6
        assert first_fold["purged_samples"] == 3
        assert first_fold["test_positive_rate"] == 0.55
        assert first_fold["baseline_accuracy"] == 0.55
        assert first_fold["baseline_brier_score"] == 0.2475
        assert first_fold["accuracy_vs_baseline"] == -0.09
        assert first_fold["brier_improvement_vs_baseline"] == -0.0625
    finally:
        conn.close()


def test_persist_prediction_run_stores_null_git_dirty_when_unknown(tmp_path: Path) -> None:
    conn = _database(tmp_path)
    try:
        persist_prediction_run(
            conn, "predict-no-git", _result(),
            prediction_version="predict-v3", benchmark_symbol="SPY",
        )
        conn.commit()

        loaded = load_prediction_run(conn, "predict-no-git")
        assert loaded is not None
        assert loaded["git_dirty"] is None
        assert loaded["git_commit_hash"] is None
        assert loaded["configuration_hash"] is None
    finally:
        conn.close()


def test_find_prediction_runs_by_fingerprint_identifies_reruns_over_identical_data(tmp_path: Path) -> None:
    conn = _database(tmp_path)
    try:
        persist_prediction_run(
            conn, "predict-rerun-1", _result(as_of_date="2026-06-30"),
            prediction_version="predict-v3", benchmark_symbol="SPY", started_at="2026-06-30T00:00:00+00:00",
        )
        conn.commit()
        persist_prediction_run(
            conn, "predict-rerun-2", _result(as_of_date="2026-06-30"),
            prediction_version="predict-v3", benchmark_symbol="SPY", started_at="2026-06-30T01:00:00+00:00",
        )
        conn.commit()
        # A run over genuinely different data gets a different fingerprint and must not
        # be returned as a rerun of the first two.
        persist_prediction_run(
            conn, "predict-different-data", _result(dataset_fingerprint="fingerprint-xyz789"),
            prediction_version="predict-v3", benchmark_symbol="SPY",
        )
        conn.commit()

        reruns = find_prediction_runs_by_fingerprint(conn, "fingerprint-abc123")
        assert {run["run_id"] for run in reruns} == {"predict-rerun-1", "predict-rerun-2"}
        assert [run["run_id"] for run in reruns] == ["predict-rerun-2", "predict-rerun-1"]  # most recent first
    finally:
        conn.close()
