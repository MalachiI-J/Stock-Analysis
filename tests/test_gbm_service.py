from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pytest

from stock_scrapper.models.analysis_models import AnalysisResult
from stock_scrapper.prediction.gbm_service import _run_gbm_walk_forward, run_gbm_prediction


def _dates(start: str, count: int) -> list[str]:
    first = date.fromisoformat(start)
    return [(first + timedelta(days=index)).isoformat() for index in range(count)]


def _flat_benchmark(trading_dates: list[str], price: float = 200.0) -> dict[str, Any]:
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
    "minimum_training_samples": 5,
    "feature_keys": ["rsi_14"],
    "gbm": {
        "n_estimators": 10,
        "max_depth": 2,
        "learning_rate": 0.3,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "l2_lambda": 0.0,
    },
}


def test_run_gbm_prediction_reports_insufficient_data_when_history_too_short() -> None:
    trading_dates = _dates("2024-01-01", 3)  # not longer than horizon_days=3
    history = {"AAA": [{"trade_date": d, "adjusted_close": 100.0} for d in trading_dates]}
    service = _FakeService({})

    result = run_gbm_prediction(
        service, ["AAA"], history, trading_dates,
        as_of_date=trading_dates[-1], rules=_RULES, benchmark_symbol="SPY",
    )

    assert result.status == "insufficient_data"
    assert "trading sessions" in result.message


def test_run_gbm_prediction_reports_insufficient_data_below_minimum_samples() -> None:
    trading_dates = _dates("2024-01-01", 15)
    prices = [100.0 + index for index in range(15)]
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    history.update(_flat_benchmark(trading_dates))
    results_by_date = {trading_dates[0]: {"AAA": _result("AAA", trading_dates[0])}}
    service = _FakeService(results_by_date)

    result = run_gbm_prediction(
        service, ["AAA"], history, trading_dates,
        as_of_date=trading_dates[-1], rules=_RULES, benchmark_symbol="SPY",
    )

    assert result.status == "insufficient_data"
    assert result.training_samples < _RULES["minimum_training_samples"]


def test_run_gbm_prediction_reports_insufficient_data_when_benchmark_price_missing() -> None:
    trading_dates = _dates("2024-01-01", 15)
    prices = [100.0 + index for index in range(15)]
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    results_by_date = {d: {"AAA": _result("AAA", d)} for d in trading_dates}
    service = _FakeService(results_by_date)

    result = run_gbm_prediction(
        service, ["AAA"], history, trading_dates,
        as_of_date=trading_dates[-1], rules=_RULES, benchmark_symbol="SPY",
    )

    assert result.status == "insufficient_data"
    assert result.training_samples == 0


def test_run_gbm_prediction_succeeds_and_reports_today_predictions() -> None:
    trading_dates = _dates("2024-01-01", 20)
    prices = [100.0 + index for index in range(20)]
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    history.update(_flat_benchmark(trading_dates))
    as_of = trading_dates[-1]

    results_by_date: dict[str, dict[str, AnalysisResult]] = {
        d: {"AAA": _result("AAA", d, rsi=40.0 + index)} for index, d in enumerate(trading_dates[:-1])
    }
    results_by_date[as_of] = {
        "AAA": _result("AAA", as_of, rsi=70.0),
        "BBB": _result("BBB", as_of, rsi=None),
        "CCC": _result("CCC", as_of, eligible=False),
    }
    service = _FakeService(results_by_date)

    result = run_gbm_prediction(
        service, ["AAA", "BBB", "CCC"], history, trading_dates,
        as_of_date=as_of, rules=_RULES, benchmark_symbol="SPY",
    )

    assert result.status == "ok"
    assert result.training_samples == 17  # 20 minus horizon_days=3
    assert len(result.walk_forward_folds) == 2
    assert result.holdout_samples == sum(fold.test_samples for fold in result.walk_forward_folds)
    assert result.feature_importances and result.feature_importances[0][0] == "rsi_14"
    by_symbol = {p.symbol: p for p in result.predictions}
    assert by_symbol["AAA"].predicted_excess_return is not None
    assert by_symbol["BBB"].predicted_excess_return is None
    assert "indicators are unavailable" in by_symbol["BBB"].reason
    assert by_symbol["CCC"].predicted_excess_return is None
    assert "Not eligible" in by_symbol["CCC"].reason


def test_run_gbm_prediction_flags_predictions_far_outside_training_target_spread() -> None:
    """A prediction that extrapolates from a rare, poorly-supported training
    precedent (see the real INTC case in README's "Evaluation honesty") should
    be flagged as low_confidence, not presented with the same apparent
    trustworthiness as an ordinary in-range prediction."""
    trading_dates = _dates("2024-01-01", 40)
    prices = [100.0] * 40
    for i in range(20, 25):
        prices[i] = 100.0 + (i - 19) * 5000.0  # one huge, rare price move
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    history.update(_flat_benchmark(trading_dates))
    as_of = trading_dates[-1]

    results_by_date: dict[str, dict[str, AnalysisResult]] = {}
    for index, d in enumerate(trading_dates[:-1]):
        rsi = 95.0 if index == 19 else 50.0  # rsi=95 occurs exactly once, right before the huge move
        results_by_date[d] = {"AAA": _result("AAA", d, rsi=rsi)}
    results_by_date[as_of] = {
        "AAA": _result("AAA", as_of, rsi=50.0),  # ordinary, well-represented feature value
        "BBB": _result("BBB", as_of, rsi=95.0),  # matches the one rare, extreme precedent
    }
    service = _FakeService(results_by_date)
    rules = dict(_RULES, minimum_training_samples=1)

    result = run_gbm_prediction(
        service, ["AAA", "BBB"], history, trading_dates,
        as_of_date=as_of, rules=rules, benchmark_symbol="SPY",
    )

    assert result.status == "ok"
    by_symbol = {p.symbol: p for p in result.predictions}
    assert by_symbol["AAA"].low_confidence is False
    assert by_symbol["BBB"].low_confidence is True
    assert by_symbol["BBB"].predicted_excess_return is not None  # flagged, not suppressed


def test_run_gbm_prediction_handles_too_few_samples_for_any_walk_forward_fold() -> None:
    trading_dates = _dates("2024-01-01", 4)  # horizon_days=3 embargo -> exactly 1 sample date
    prices = [100.0, 101.0, 102.0, 103.0]
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    history.update(_flat_benchmark(trading_dates))
    results_by_date = {d: {"AAA": _result("AAA", d)} for d in trading_dates}
    service = _FakeService(results_by_date)
    rules = dict(_RULES, minimum_training_samples=1)

    result = run_gbm_prediction(
        service, ["AAA"], history, trading_dates,
        as_of_date=trading_dates[-1], rules=rules, benchmark_symbol="SPY",
    )

    assert result.status == "ok"
    assert result.training_samples == 1
    assert result.walk_forward_folds == []
    assert result.holdout_mse is None
    assert "walk-forward" in result.message


def test_run_gbm_prediction_is_deterministic_given_same_inputs() -> None:
    trading_dates = _dates("2024-01-01", 20)
    prices = [100.0 + index for index in range(20)]
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    history.update(_flat_benchmark(trading_dates))
    as_of = trading_dates[-1]
    results_by_date = {
        d: {"AAA": _result("AAA", d, rsi=40.0 + index)} for index, d in enumerate(trading_dates)
    }

    result_a = run_gbm_prediction(
        _FakeService(results_by_date), ["AAA"], history, trading_dates,
        as_of_date=as_of, rules=_RULES, benchmark_symbol="SPY",
    )
    result_b = run_gbm_prediction(
        _FakeService(results_by_date), ["AAA"], history, trading_dates,
        as_of_date=as_of, rules=_RULES, benchmark_symbol="SPY",
    )

    assert result_a.feature_importances == result_b.feature_importances
    assert (
        result_a.predictions[0].predicted_excess_return == result_b.predictions[0].predicted_excess_return
    )


def _meta_row(symbol: str, sample_date: str, label_end_date: str) -> dict[str, str]:
    return {"symbol": symbol, "date": sample_date, "label_end_date": label_end_date}


_GBM_KWARGS = dict(n_estimators=5, max_depth=2, learning_rate=0.3, min_samples_leaf=1, min_samples_split=2, l2_lambda=0.0)


def test_run_gbm_walk_forward_never_splits_rows_from_the_same_date_across_a_fold_boundary() -> None:
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    symbols_by_date = {dates[0]: ["A", "B", "C"], dates[1]: ["A", "B", "C"], dates[2]: ["A", "B"]}
    meta = [
        _meta_row(symbol, d, "2000-01-01")
        for d in dates
        for symbol in symbols_by_date[d]
    ]
    features = np.array([[float(index)] for index in range(len(meta))])
    targets = np.array([0.01 * index for index in range(len(meta))])

    folds = _run_gbm_walk_forward(
        ["x"], features, targets, meta, requested_folds=1, **_GBM_KWARGS,
    )

    assert len(folds) == 1
    fold = folds[0]
    assert fold.training_start_date == dates[0]
    assert fold.training_end_date == dates[1]
    assert fold.test_start_date == dates[2]
    assert fold.test_end_date == dates[2]
    assert fold.training_samples == 6
    assert fold.test_samples == 2
    assert fold.purged_samples == 0


def test_run_gbm_walk_forward_purges_training_rows_whose_target_overlaps_the_test_period() -> None:
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    meta = [
        _meta_row("A", dates[0], dates[0]),
        _meta_row("A", dates[1], dates[3]),  # resolves inside the test period -> purged
        _meta_row("A", dates[2], dates[2]),
        _meta_row("A", dates[3], dates[3]),
    ]
    features = np.array([[1.0], [2.0], [3.0], [4.0]])
    targets = np.array([0.01, -0.02, 0.03, -0.01])

    folds = _run_gbm_walk_forward(
        ["x"], features, targets, meta, requested_folds=1, **_GBM_KWARGS,
    )

    assert len(folds) == 1
    fold = folds[0]
    assert fold.training_samples == 1
    assert fold.purged_samples == 1
    assert fold.test_samples == 2


def test_run_gbm_walk_forward_computes_fold_specific_baseline_from_test_target_variance() -> None:
    dates = ["2024-01-01", "2024-01-02"]
    meta = [_meta_row("X", dates[0], "2000-01-01"), _meta_row("Y", dates[0], "2000-01-01")]
    meta += [_meta_row("X", dates[1], "2000-01-01"), _meta_row("Y", dates[1], "2000-01-01")]
    targets = np.array([0.0, 0.0, 0.02, -0.02])
    features = np.array([[0.0], [1.0], [2.0], [3.0]])

    folds = _run_gbm_walk_forward(
        ["x"], features, targets, meta, requested_folds=1, **_GBM_KWARGS,
    )

    assert len(folds) == 1
    fold = folds[0]
    test_targets = np.array([0.02, -0.02])
    assert fold.baseline_mse == pytest.approx(float(test_targets.var()))


def test_run_gbm_prediction_exposes_provenance_hashes_and_reruns_match() -> None:
    trading_dates = _dates("2024-01-01", 20)
    prices = [100.0 + index for index in range(20)]
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    history.update(_flat_benchmark(trading_dates))
    as_of = trading_dates[-1]
    results_by_date = {d: {"AAA": _result("AAA", d, rsi=40.0 + index)} for index, d in enumerate(trading_dates)}

    result_a = run_gbm_prediction(
        _FakeService(results_by_date), ["AAA"], history, trading_dates,
        as_of_date=as_of, rules=_RULES, benchmark_symbol="SPY",
    )
    result_b = run_gbm_prediction(
        _FakeService(results_by_date), ["AAA"], history, trading_dates,
        as_of_date=as_of, rules=_RULES, benchmark_symbol="SPY",
    )

    assert result_a.dataset_fingerprint is not None
    assert result_a.dataset_fingerprint == result_b.dataset_fingerprint
    assert result_a.symbol_universe_hash == result_b.symbol_universe_hash
    assert result_a.feature_set_hash == result_b.feature_set_hash

    result_c = run_gbm_prediction(
        _FakeService(results_by_date), ["AAA", "ZZZ"], history, trading_dates,
        as_of_date=as_of, rules=_RULES, benchmark_symbol="SPY",
    )
    assert result_c.symbol_universe_hash != result_a.symbol_universe_hash
    assert result_c.feature_set_hash == result_a.feature_set_hash


def test_run_gbm_prediction_supports_predict_v5_feature_keys_and_fundamentals() -> None:
    """predict_v5 passes its own feature_keys (overriding rules["feature_keys"]) and
    fundamentals_by_symbol through the same function predict-v4 already uses."""
    trading_dates = _dates("2024-01-01", 20)
    prices = [100.0 + index for index in range(20)]
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    history.update(_flat_benchmark(trading_dates))
    as_of = trading_dates[-1]
    results_by_date = {d: {"AAA": _result("AAA", d, rsi=None)} for d in trading_dates}
    service = _FakeService(results_by_date)
    fundamentals_by_symbol = {
        "AAA": [
            {"concept": "eps_diluted", "period_start": "2022-10-01", "period_end": "2022-12-31", "filed_date": "2023-01-15", "value": 1.0},
            {"concept": "eps_diluted", "period_start": "2023-01-01", "period_end": "2023-03-31", "filed_date": "2023-04-15", "value": 1.0},
            {"concept": "eps_diluted", "period_start": "2023-04-01", "period_end": "2023-06-30", "filed_date": "2023-07-15", "value": 1.0},
            {"concept": "eps_diluted", "period_start": "2023-07-01", "period_end": "2023-09-30", "filed_date": "2023-10-15", "value": 1.0},
        ],
    }
    # rules["feature_keys"] deliberately differs from the explicit feature_keys=
    # argument below, to prove the explicit argument wins rather than rules'.
    rules = dict(_RULES, feature_keys=["rsi_14"])

    result = run_gbm_prediction(
        service, ["AAA"], history, trading_dates,
        as_of_date=as_of, rules=rules, benchmark_symbol="SPY",
        feature_keys=["trailing_pe"], fundamentals_by_symbol=fundamentals_by_symbol,
    )

    assert result.status == "ok"
    assert result.feature_importances and result.feature_importances[0][0] == "trailing_pe"
    prediction = next(p for p in result.predictions if p.symbol == "AAA")
    assert prediction.predicted_excess_return is not None


def test_run_gbm_prediction_without_predict_v5_params_matches_predict_v4_exactly() -> None:
    """Omitting feature_keys/gbm_config/fundamentals_by_symbol must reproduce
    predict-v4's existing behavior byte-for-byte -- these are additive, optional
    parameters, not a change to any existing caller."""
    trading_dates = _dates("2024-01-01", 20)
    prices = [100.0 + index for index in range(20)]
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    history.update(_flat_benchmark(trading_dates))
    as_of = trading_dates[-1]
    results_by_date = {d: {"AAA": _result("AAA", d, rsi=40.0 + index)} for index, d in enumerate(trading_dates)}

    baseline = run_gbm_prediction(
        _FakeService(results_by_date), ["AAA"], history, trading_dates,
        as_of_date=as_of, rules=_RULES, benchmark_symbol="SPY",
    )
    explicit = run_gbm_prediction(
        _FakeService(results_by_date), ["AAA"], history, trading_dates,
        as_of_date=as_of, rules=_RULES, benchmark_symbol="SPY",
        feature_keys=None, gbm_config=None, fundamentals_by_symbol=None,
    )

    assert baseline.dataset_fingerprint == explicit.dataset_fingerprint
    assert baseline.feature_importances == explicit.feature_importances
    assert (
        baseline.predictions[0].predicted_excess_return == explicit.predictions[0].predicted_excess_return
    )
