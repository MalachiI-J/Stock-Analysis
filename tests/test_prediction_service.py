from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pytest

from stock_scrapper.models.analysis_models import AnalysisResult
from stock_scrapper.prediction.service import _run_walk_forward, _weighted_average, run_prediction


def _dates(start: str, count: int) -> list[str]:
    first = date.fromisoformat(start)
    return [(first + timedelta(days=index)).isoformat() for index in range(count)]


def _flat_benchmark(trading_dates: list[str], price: float = 200.0) -> dict[str, Any]:
    """A constant-price benchmark series, so beating it is equivalent to a positive raw return."""
    return {"SPY": [{"trade_date": d, "adjusted_close": price} for d in trading_dates]}


class _FakeService:
    def __init__(self, results_by_date_symbol: dict[str, dict[str, AnalysisResult]]) -> None:
        self._results = results_by_date_symbol

    def prime_historical_features(self, histories: Any, snapshot_dates: Any, feature_symbols: Any) -> None:
        return None

    def analyze_loaded_many_as_of(self, symbols: list[str], histories: Any, as_of_date: date, *, persist: bool) -> Any:
        del histories, persist
        by_symbol = self._results.get(as_of_date.isoformat(), {})
        results = [by_symbol[symbol] for symbol in symbols if symbol in by_symbol]
        return type("Batch", (), {"results": results})()


def _result(symbol: str, as_of: str, *, eligible: bool = True, rsi: float | None = 50.0) -> AnalysisResult:
    return AnalysisResult(
        symbol=symbol,
        as_of_date=as_of,
        eligible_for_scoring=eligible,
        indicators={"rsi_14": rsi} if rsi is not None else {},
    )


_RULES = {
    "horizon_days": 3,
    "lookback_years": 10.0,
    "sample_stride_sessions": 1,
    "walk_forward_folds": 2,
    "l2_lambda": 0.01,
    "learning_rate": 0.3,
    "iterations": 50,
    "minimum_training_samples": 5,
    "feature_keys": ["rsi_14"],
}


def test_run_prediction_reports_insufficient_data_when_history_too_short() -> None:
    trading_dates = _dates("2024-01-01", 3)  # not longer than horizon_days=3, so no sample dates exist
    history = {"AAA": [{"trade_date": d, "adjusted_close": 100.0} for d in trading_dates]}
    service = _FakeService({})

    result = run_prediction(
        service, ["AAA"], history, trading_dates,
        as_of_date=trading_dates[-1], rules=_RULES, benchmark_symbol="SPY",
    )

    assert result.status == "insufficient_data"
    assert "trading sessions" in result.message


def test_run_prediction_reports_insufficient_data_below_minimum_samples() -> None:
    trading_dates = _dates("2024-01-01", 15)
    prices = [100.0 + index for index in range(15)]
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    history.update(_flat_benchmark(trading_dates))
    # Only provide results for one training date -> far fewer than minimum_training_samples=5.
    results_by_date = {trading_dates[0]: {"AAA": _result("AAA", trading_dates[0])}}
    service = _FakeService(results_by_date)

    result = run_prediction(
        service, ["AAA"], history, trading_dates,
        as_of_date=trading_dates[-1], rules=_RULES, benchmark_symbol="SPY",
    )

    assert result.status == "insufficient_data"
    assert result.training_samples < _RULES["minimum_training_samples"]


def test_run_prediction_reports_insufficient_data_when_benchmark_price_missing() -> None:
    trading_dates = _dates("2024-01-01", 15)
    prices = [100.0 + index for index in range(15)]
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    # No "SPY" entry at all -> every sample lacks a benchmark price and is dropped.
    results_by_date = {d: {"AAA": _result("AAA", d)} for d in trading_dates}
    service = _FakeService(results_by_date)

    result = run_prediction(
        service, ["AAA"], history, trading_dates,
        as_of_date=trading_dates[-1], rules=_RULES, benchmark_symbol="SPY",
    )

    assert result.status == "insufficient_data"
    assert result.training_samples == 0


def test_run_prediction_succeeds_and_reports_today_predictions() -> None:
    trading_dates = _dates("2024-01-01", 20)
    prices = [100.0 + index for index in range(20)]  # steadily rising
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    history.update(_flat_benchmark(trading_dates))
    as_of = trading_dates[-1]

    results_by_date: dict[str, dict[str, AnalysisResult]] = {
        d: {"AAA": _result("AAA", d, rsi=40.0 + index)} for index, d in enumerate(trading_dates[:-1])
    }
    # Today: AAA has a usable indicator, BBB is missing one, CCC is ineligible.
    results_by_date[as_of] = {
        "AAA": _result("AAA", as_of, rsi=70.0),
        "BBB": _result("BBB", as_of, rsi=None),
        "CCC": _result("CCC", as_of, eligible=False),
    }
    service = _FakeService(results_by_date)

    result = run_prediction(
        service, ["AAA", "BBB", "CCC"], history, trading_dates,
        as_of_date=as_of, rules=_RULES, benchmark_symbol="SPY",
    )

    assert result.status == "ok"
    # 17 embargoed sample dates (20 minus horizon_days=3), all usable -> the final
    # model is fit on all 17, and 2 walk-forward folds cover 11 of them as holdout.
    assert result.training_samples == 17
    assert len(result.walk_forward_folds) == 2
    assert result.holdout_samples == sum(fold.test_samples for fold in result.walk_forward_folds)
    assert result.positive_label_rate == 1.0  # rising price vs a flat benchmark -> always beats it
    assert result.coefficients and result.coefficients[0][0] == "rsi_14"
    by_symbol = {p.symbol: p for p in result.predictions}
    assert by_symbol["AAA"].probability_positive is not None
    assert by_symbol["BBB"].probability_positive is None
    assert "indicators are unavailable" in by_symbol["BBB"].reason
    assert by_symbol["CCC"].probability_positive is None
    assert "Not eligible" in by_symbol["CCC"].reason


def test_run_prediction_handles_too_few_samples_for_any_walk_forward_fold() -> None:
    trading_dates = _dates("2024-01-01", 4)  # horizon_days=3 embargo -> exactly 1 sample date
    prices = [100.0, 101.0, 102.0, 103.0]
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    history.update(_flat_benchmark(trading_dates))
    results_by_date = {d: {"AAA": _result("AAA", d)} for d in trading_dates}
    service = _FakeService(results_by_date)
    rules = dict(_RULES, minimum_training_samples=1)

    result = run_prediction(
        service, ["AAA"], history, trading_dates,
        as_of_date=trading_dates[-1], rules=rules, benchmark_symbol="SPY",
    )

    assert result.status == "ok"
    assert result.training_samples == 1
    assert result.walk_forward_folds == []
    assert result.holdout_accuracy is None
    assert "walk-forward" in result.message


def test_run_prediction_is_deterministic_given_same_inputs() -> None:
    trading_dates = _dates("2024-01-01", 20)
    prices = [100.0 + index for index in range(20)]
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    history.update(_flat_benchmark(trading_dates))
    as_of = trading_dates[-1]
    results_by_date = {
        d: {"AAA": _result("AAA", d, rsi=40.0 + index)} for index, d in enumerate(trading_dates)
    }

    result_a = run_prediction(
        _FakeService(results_by_date), ["AAA"], history, trading_dates,
        as_of_date=as_of, rules=_RULES, benchmark_symbol="SPY",
    )
    result_b = run_prediction(
        _FakeService(results_by_date), ["AAA"], history, trading_dates,
        as_of_date=as_of, rules=_RULES, benchmark_symbol="SPY",
    )

    assert result_a.coefficients == result_b.coefficients
    assert (
        result_a.predictions[0].probability_positive == result_b.predictions[0].probability_positive
    )


def _meta_row(symbol: str, sample_date: str, label_end_date: str) -> dict[str, str]:
    return {"symbol": symbol, "date": sample_date, "label_end_date": label_end_date}


def test_run_walk_forward_never_splits_rows_from_the_same_date_across_a_fold_boundary() -> None:
    # 3 unique dates with different row counts (3, 3, 2) -- a row-count-based boundary
    # would land inside the second date's block; a date-count boundary must not.
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    symbols_by_date = {dates[0]: ["A", "B", "C"], dates[1]: ["A", "B", "C"], dates[2]: ["A", "B"]}
    meta = [
        _meta_row(symbol, d, "2000-01-01")  # far in the past -> never purged
        for d in dates
        for symbol in symbols_by_date[d]
    ]
    features = np.array([[float(index)] for index in range(len(meta))])
    labels = np.array([1.0 if index % 2 == 0 else 0.0 for index in range(len(meta))])

    folds = _run_walk_forward(
        ["x"], features, labels, meta,
        requested_folds=1, l2_lambda=0.0, learning_rate=0.1, iterations=5,
    )

    assert len(folds) == 1
    fold = folds[0]
    assert fold.training_start_date == dates[0]
    assert fold.training_end_date == dates[1]
    assert fold.test_start_date == dates[2]
    assert fold.test_end_date == dates[2]
    assert fold.training_samples == 6  # all of d1 + d2, none split out
    assert fold.test_samples == 2
    assert fold.purged_samples == 0


def test_run_walk_forward_purges_training_rows_whose_label_overlaps_the_test_period() -> None:
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    meta = [
        _meta_row("A", dates[0], dates[0]),  # label resolves before the test period -> kept
        _meta_row("A", dates[1], dates[3]),  # label resolves inside the test period -> purged
        _meta_row("A", dates[2], dates[2]),
        _meta_row("A", dates[3], dates[3]),
    ]
    features = np.array([[1.0], [2.0], [3.0], [4.0]])
    labels = np.array([1.0, 0.0, 1.0, 0.0])

    folds = _run_walk_forward(
        ["x"], features, labels, meta,
        requested_folds=1, l2_lambda=0.0, learning_rate=0.1, iterations=5,
    )

    assert len(folds) == 1
    fold = folds[0]
    assert fold.training_samples == 1
    assert fold.purged_samples == 1
    assert fold.test_samples == 2
    assert fold.training_start_date == dates[0]
    assert fold.training_end_date == dates[0]


def test_run_walk_forward_computes_fold_specific_baselines_from_that_folds_own_test_labels() -> None:
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    # Two symbols per date; labels chosen so each fold's test-period positive rate is
    # distinct from both the other fold's and the whole dataset's rate (3/8 = 0.375).
    labels_by_date = {dates[0]: [0.0, 0.0], dates[1]: [0.0, 0.0], dates[2]: [1.0, 0.0], dates[3]: [1.0, 1.0]}
    meta: list[dict[str, str]] = []
    label_values: list[float] = []
    for d in dates:
        for symbol, label in zip(("X", "Y"), labels_by_date[d]):
            meta.append(_meta_row(symbol, d, "2000-01-01"))
            label_values.append(label)
    features = np.array([[float(index)] for index in range(len(meta))])
    labels = np.array(label_values)

    folds = _run_walk_forward(
        ["x"], features, labels, meta,
        requested_folds=2, l2_lambda=0.0, learning_rate=0.1, iterations=5,
    )

    assert len(folds) == 2
    first, second = folds
    # Fold 1's test period is d3 alone (labels [1, 0]) -> a 50/50 split.
    assert first.test_positive_rate == pytest.approx(0.5)
    assert first.baseline_accuracy == pytest.approx(0.5)
    assert first.baseline_brier_score == pytest.approx(0.25)
    # Fold 2's test period is d4 alone (labels [1, 1]) -> unanimous.
    assert second.test_positive_rate == pytest.approx(1.0)
    assert second.baseline_accuracy == pytest.approx(1.0)
    assert second.baseline_brier_score == pytest.approx(0.0)


def test_weighted_average_differs_from_unweighted_mean_when_fold_sizes_differ() -> None:
    values = [(1.0, 10), (0.0, 90)]  # a small fold that went well, a large one that didn't

    weighted = _weighted_average(values)
    naive = sum(value for value, _ in values) / len(values)

    assert weighted == pytest.approx(0.1)
    assert naive == pytest.approx(0.5)
    assert weighted != pytest.approx(naive)


def test_weighted_average_returns_none_for_empty_or_zero_weight_input() -> None:
    assert _weighted_average([]) is None
    assert _weighted_average([(1.0, 0), (0.5, 0)]) is None


def test_run_prediction_aggregates_holdout_accuracy_by_sample_weight_not_fold_count() -> None:
    trading_dates = _dates("2024-01-01", 30)
    prices = [100.0 + (5 if index % 2 == 0 else -5) for index in range(30)]
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    history.update(_flat_benchmark(trading_dates))
    as_of = trading_dates[-1]
    results_by_date = {d: {"AAA": _result("AAA", d, rsi=40.0 + index)} for index, d in enumerate(trading_dates)}
    service = _FakeService(results_by_date)
    rules = dict(_RULES, walk_forward_folds=3)

    result = run_prediction(
        service, ["AAA"], history, trading_dates,
        as_of_date=as_of, rules=rules, benchmark_symbol="SPY",
    )

    scored_folds = [fold for fold in result.walk_forward_folds if fold.accuracy is not None]
    assert len(scored_folds) >= 2
    # The fold split's remainder distribution (see _fold_boundaries) should leave at
    # least one fold a different size from the others -- otherwise this test wouldn't
    # distinguish weighted from unweighted aggregation at all.
    assert len({fold.test_samples for fold in scored_folds}) > 1

    total_weight = sum(fold.test_samples for fold in scored_folds)
    expected_accuracy = sum(fold.accuracy * fold.test_samples for fold in scored_folds) / total_weight
    expected_brier = sum(fold.brier_score * fold.test_samples for fold in scored_folds) / total_weight
    assert result.holdout_accuracy == pytest.approx(expected_accuracy)
    assert result.holdout_brier_score == pytest.approx(expected_brier)

    baseline_folds = [fold for fold in scored_folds if fold.baseline_accuracy is not None]
    baseline_weight = sum(fold.test_samples for fold in baseline_folds)
    expected_baseline_accuracy = sum(
        fold.baseline_accuracy * fold.test_samples for fold in baseline_folds
    ) / baseline_weight
    assert result.baseline_accuracy == pytest.approx(expected_baseline_accuracy)


def test_run_prediction_exposes_provenance_hashes_and_reruns_match() -> None:
    trading_dates = _dates("2024-01-01", 20)
    prices = [100.0 + index for index in range(20)]
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    history.update(_flat_benchmark(trading_dates))
    as_of = trading_dates[-1]
    results_by_date = {d: {"AAA": _result("AAA", d, rsi=40.0 + index)} for index, d in enumerate(trading_dates)}

    result_a = run_prediction(
        _FakeService(results_by_date), ["AAA"], history, trading_dates,
        as_of_date=as_of, rules=_RULES, benchmark_symbol="SPY",
    )
    result_b = run_prediction(
        _FakeService(results_by_date), ["AAA"], history, trading_dates,
        as_of_date=as_of, rules=_RULES, benchmark_symbol="SPY",
    )

    assert result_a.dataset_fingerprint is not None
    assert result_a.symbol_universe_hash is not None
    assert result_a.feature_set_hash is not None
    # A rerun over identical data/symbols/features must fingerprint identically, so it
    # is identifiable as a rerun rather than independent new evidence.
    assert result_a.dataset_fingerprint == result_b.dataset_fingerprint
    assert result_a.symbol_universe_hash == result_b.symbol_universe_hash
    assert result_a.feature_set_hash == result_b.feature_set_hash

    # A different symbol universe must change the symbol hash but not the feature hash.
    result_c = run_prediction(
        _FakeService(results_by_date), ["AAA", "ZZZ"], history, trading_dates,
        as_of_date=as_of, rules=_RULES, benchmark_symbol="SPY",
    )
    assert result_c.symbol_universe_hash != result_a.symbol_universe_hash
    assert result_c.feature_set_hash == result_a.feature_set_hash
