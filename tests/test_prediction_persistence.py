from __future__ import annotations

import sqlite3
from pathlib import Path

from stock_scrapper.database import create_connection, initialize_database
from stock_scrapper.prediction.persistence import (
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
            WalkForwardFold(fold=1, training_samples=50, test_samples=20, accuracy=0.46, brier_score=0.31),
            WalkForwardFold(fold=2, training_samples=70, test_samples=20, accuracy=0.50, brier_score=0.25),
        ],
        coefficients=[("rsi_14", 0.21), ("beta", -0.05)],
        predictions=[
            SymbolPrediction("AAPL", 0.535, None),
            SymbolPrediction("MISSING", None, "One or more required indicators are unavailable"),
        ],
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
