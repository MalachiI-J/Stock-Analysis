from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from stock_scrapper.models.analysis_models import AnalysisResult
from stock_scrapper.prediction.service import run_prediction


def _dates(start: str, count: int) -> list[str]:
    first = date.fromisoformat(start)
    return [(first + timedelta(days=index)).isoformat() for index in range(count)]


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
        service, ["AAA"], history, trading_dates, as_of_date=trading_dates[-1], rules=_RULES
    )

    assert result.status == "insufficient_data"
    assert "trading sessions" in result.message


def test_run_prediction_reports_insufficient_data_below_minimum_samples() -> None:
    trading_dates = _dates("2024-01-01", 15)
    prices = [100.0 + index for index in range(15)]
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    # Only provide results for one training date -> far fewer than minimum_training_samples=5.
    results_by_date = {trading_dates[0]: {"AAA": _result("AAA", trading_dates[0])}}
    service = _FakeService(results_by_date)

    result = run_prediction(
        service, ["AAA"], history, trading_dates, as_of_date=trading_dates[-1], rules=_RULES
    )

    assert result.status == "insufficient_data"
    assert result.training_samples < _RULES["minimum_training_samples"]


def test_run_prediction_succeeds_and_reports_today_predictions() -> None:
    trading_dates = _dates("2024-01-01", 20)
    prices = [100.0 + index for index in range(20)]  # steadily rising
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
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
        service, ["AAA", "BBB", "CCC"], history, trading_dates, as_of_date=as_of, rules=_RULES
    )

    assert result.status == "ok"
    # 17 embargoed sample dates (20 minus horizon_days=3), all usable -> the final
    # model is fit on all 17, and 2 walk-forward folds cover 11 of them as holdout.
    assert result.training_samples == 17
    assert len(result.walk_forward_folds) == 2
    assert result.holdout_samples == sum(fold.test_samples for fold in result.walk_forward_folds)
    assert result.positive_label_rate == 1.0  # prices always rise -> every label is positive
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
    results_by_date = {d: {"AAA": _result("AAA", d)} for d in trading_dates}
    service = _FakeService(results_by_date)
    rules = dict(_RULES, minimum_training_samples=1)

    result = run_prediction(
        service, ["AAA"], history, trading_dates, as_of_date=trading_dates[-1], rules=rules
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
    as_of = trading_dates[-1]
    results_by_date = {
        d: {"AAA": _result("AAA", d, rsi=40.0 + index)} for index, d in enumerate(trading_dates)
    }

    result_a = run_prediction(
        _FakeService(results_by_date), ["AAA"], history, trading_dates, as_of_date=as_of, rules=_RULES
    )
    result_b = run_prediction(
        _FakeService(results_by_date), ["AAA"], history, trading_dates, as_of_date=as_of, rules=_RULES
    )

    assert result_a.coefficients == result_b.coefficients
    assert (
        result_a.predictions[0].probability_positive == result_b.predictions[0].probability_positive
    )
