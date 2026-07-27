from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from stock_scrapper.models.analysis_models import AnalysisResult
from stock_scrapper.prediction.dataset import (
    build_training_dataset,
    feature_value,
    opportunity_percentiles,
    select_sample_dates,
)


def _dates(start: str, count: int) -> list[str]:
    first = date.fromisoformat(start)
    return [(first + timedelta(days=index)).isoformat() for index in range(count)]


class _FakeService:
    """Stands in for AnalysisService: returns caller-controlled results per (symbol, date).

    This isolates dataset-assembly/embargo logic from the real scoring engine,
    which already has its own extensive test coverage elsewhere.
    """

    def __init__(self, results_by_date_symbol: dict[str, dict[str, AnalysisResult]]) -> None:
        self._results = results_by_date_symbol
        self.primed_dates: list[Any] = []

    def prime_historical_features(self, histories: Any, snapshot_dates: Any, feature_symbols: Any) -> None:
        self.primed_dates = list(snapshot_dates)

    def analyze_loaded_many_as_of(self, symbols: list[str], histories: Any, as_of_date: date, *, persist: bool) -> Any:
        del histories, persist
        by_symbol = self._results.get(as_of_date.isoformat(), {})
        results = [by_symbol[symbol] for symbol in symbols if symbol in by_symbol]
        return type("Batch", (), {"results": results})()


def _result(
    symbol: str,
    as_of: str,
    *,
    eligible: bool = True,
    rsi: float | None = 50.0,
    market_regime: str = "Insufficient Market Data",
    opportunity_score: float | None = None,
) -> AnalysisResult:
    return AnalysisResult(
        symbol=symbol,
        as_of_date=as_of,
        eligible_for_scoring=eligible,
        market_regime=market_regime,
        opportunity_score=opportunity_score,
        indicators={"rsi_14": rsi} if rsi is not None else {},
    )


def test_select_sample_dates_embargoes_recent_sessions() -> None:
    trading_dates = _dates("2024-01-01", 40)  # 40 consecutive daily "sessions"
    as_of = trading_dates[-1]
    sampled = select_sample_dates(
        trading_dates, as_of_date=as_of, horizon_days=10, lookback_years=10, stride_sessions=1
    )
    # The last 10 sessions cannot have a known 10-session-forward label.
    assert sampled[-1].isoformat() == trading_dates[-11]
    assert all(day.isoformat() in trading_dates[: len(trading_dates) - 10] for day in sampled)


def test_select_sample_dates_applies_stride() -> None:
    trading_dates = _dates("2024-01-01", 30)
    sampled = select_sample_dates(
        trading_dates, as_of_date=trading_dates[-1], horizon_days=5, lookback_years=10, stride_sessions=5
    )
    assert [day.isoformat() for day in sampled] == [trading_dates[0], trading_dates[5], trading_dates[10], trading_dates[15], trading_dates[20]]


def test_select_sample_dates_returns_empty_when_history_too_short() -> None:
    trading_dates = _dates("2024-01-01", 5)
    sampled = select_sample_dates(
        trading_dates, as_of_date=trading_dates[-1], horizon_days=10, lookback_years=10, stride_sessions=1
    )
    assert sampled == []


def test_build_training_dataset_labels_match_forward_return_sign() -> None:
    trading_dates = _dates("2024-01-01", 20)
    prices = [100.0 + index for index in range(20)]  # steadily rising -> always positive forward return
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    results_by_date = {
        d: {"AAA": _result("AAA", d, rsi=50.0 + index)} for index, d in enumerate(trading_dates)
    }
    service = _FakeService(results_by_date)
    sample_dates = [date.fromisoformat(d) for d in trading_dates[:10]]

    features, labels, meta = build_training_dataset(
        service, ["AAA"], history, sample_dates, horizon_days=5, feature_keys=["rsi_14"]
    )

    assert features.shape[0] == 10
    assert (labels == 1.0).all()  # prices always rise -> always a positive forward return
    assert [row["symbol"] for row in meta] == ["AAA"] * 10


def test_build_training_dataset_drops_rows_missing_features_or_ineligible() -> None:
    trading_dates = _dates("2024-01-01", 10)
    prices = [100.0] * 10
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    results_by_date = {
        trading_dates[0]: {"AAA": _result("AAA", trading_dates[0], eligible=False)},
        trading_dates[1]: {"AAA": _result("AAA", trading_dates[1], rsi=None)},
        trading_dates[2]: {"AAA": _result("AAA", trading_dates[2], rsi=60.0)},
    }
    service = _FakeService(results_by_date)
    sample_dates = [date.fromisoformat(trading_dates[index]) for index in range(3)]

    features, labels, meta = build_training_dataset(
        service, ["AAA"], history, sample_dates, horizon_days=5, feature_keys=["rsi_14"]
    )

    assert features.shape[0] == 1
    assert meta == [{"symbol": "AAA", "date": trading_dates[2]}]


def test_build_training_dataset_excludes_dates_without_enough_forward_history() -> None:
    trading_dates = _dates("2024-01-01", 8)
    prices = [100.0 + index for index in range(8)]
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    # Request a sample date only 2 sessions from the end of the stored history, with horizon=5:
    # index 6 + horizon 5 = 11 >= len(history)=8, so this date must be excluded even though a
    # caller passed it in directly (defense in depth beyond select_sample_dates' own embargo).
    results_by_date = {trading_dates[6]: {"AAA": _result("AAA", trading_dates[6])}}
    service = _FakeService(results_by_date)

    features, labels, meta = build_training_dataset(
        service, ["AAA"], history, [date.fromisoformat(trading_dates[6])], horizon_days=5, feature_keys=["rsi_14"]
    )

    assert features.shape[0] == 0
    assert meta == []


def test_opportunity_percentiles_ranks_by_score_with_symbol_tiebreak() -> None:
    results = [
        _result("AAA", "2024-01-01", opportunity_score=70.0),
        _result("BBB", "2024-01-01", opportunity_score=90.0),
        _result("CCC", "2024-01-01", opportunity_score=50.0),
    ]
    percentiles = opportunity_percentiles(results)
    assert percentiles == {"CCC": 0.0, "AAA": 0.5, "BBB": 1.0}


def test_opportunity_percentiles_falls_back_to_midpoint_for_ambiguous_ranks() -> None:
    assert opportunity_percentiles([_result("AAA", "2024-01-01", opportunity_score=70.0)]) == {"AAA": 0.5}
    assert opportunity_percentiles([_result("AAA", "2024-01-01", opportunity_score=None)]) == {}


def test_feature_value_resolves_derived_and_indicator_features() -> None:
    result = _result("AAA", "2024-01-01", rsi=42.0, market_regime="Risk-On", opportunity_score=80.0)
    percentiles = {"AAA": 0.75}
    assert feature_value(result, "rsi_14", percentiles) == 42.0
    assert feature_value(result, "market_regime_code", percentiles) == 1.0
    assert feature_value(result, "opportunity_score_percentile", percentiles) == 0.75


def test_feature_value_returns_none_for_unmapped_regime() -> None:
    result = _result("AAA", "2024-01-01", market_regime="Insufficient Market Data")
    assert feature_value(result, "market_regime_code", {}) is None


def test_build_training_dataset_drops_rows_with_unmapped_market_regime() -> None:
    trading_dates = _dates("2024-01-01", 10)
    prices = [100.0] * 10
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    results_by_date = {
        trading_dates[0]: {"AAA": _result("AAA", trading_dates[0], market_regime="Insufficient Market Data")},
        trading_dates[1]: {"AAA": _result("AAA", trading_dates[1], market_regime="Risk-On")},
    }
    service = _FakeService(results_by_date)
    sample_dates = [date.fromisoformat(trading_dates[index]) for index in range(2)]

    features, labels, meta = build_training_dataset(
        service, ["AAA"], history, sample_dates,
        horizon_days=5, feature_keys=["rsi_14", "market_regime_code"],
    )

    assert features.shape[0] == 1
    assert meta == [{"symbol": "AAA", "date": trading_dates[1]}]
