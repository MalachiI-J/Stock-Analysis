from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pytest

from stock_scrapper.models.analysis_models import AnalysisResult
from stock_scrapper.prediction.dataset import (
    build_regression_dataset,
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


def _flat_history(symbol: str, trading_dates: list[str], price: float = 200.0) -> dict[str, list[dict[str, Any]]]:
    """A constant-price benchmark series, so beating it is equivalent to a positive raw return."""
    return {symbol: [{"trade_date": d, "adjusted_close": price} for d in trading_dates]}


def test_build_training_dataset_labels_match_excess_return_over_benchmark() -> None:
    trading_dates = _dates("2024-01-01", 20)
    prices = [100.0 + index for index in range(20)]  # steadily rising -> always beats a flat benchmark
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    history.update(_flat_history("SPY", trading_dates))
    results_by_date = {
        d: {"AAA": _result("AAA", d, rsi=50.0 + index)} for index, d in enumerate(trading_dates)
    }
    service = _FakeService(results_by_date)
    sample_dates = [date.fromisoformat(d) for d in trading_dates[:10]]

    features, labels, meta = build_training_dataset(
        service, ["AAA"], history, sample_dates,
        horizon_days=5, feature_keys=["rsi_14"], benchmark_symbol="SPY",
    )

    assert features.shape[0] == 10
    assert (labels == 1.0).all()  # rising price vs a flat benchmark -> always beats it
    assert [row["symbol"] for row in meta] == ["AAA"] * 10


def test_build_training_dataset_drops_rows_missing_features_or_ineligible() -> None:
    trading_dates = _dates("2024-01-01", 10)
    prices = [100.0] * 10
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    history.update(_flat_history("SPY", trading_dates))
    results_by_date = {
        trading_dates[0]: {"AAA": _result("AAA", trading_dates[0], eligible=False)},
        trading_dates[1]: {"AAA": _result("AAA", trading_dates[1], rsi=None)},
        trading_dates[2]: {"AAA": _result("AAA", trading_dates[2], rsi=60.0)},
    }
    service = _FakeService(results_by_date)
    sample_dates = [date.fromisoformat(trading_dates[index]) for index in range(3)]

    features, labels, meta = build_training_dataset(
        service, ["AAA"], history, sample_dates,
        horizon_days=5, feature_keys=["rsi_14"], benchmark_symbol="SPY",
    )

    assert features.shape[0] == 1
    assert meta == [{"symbol": "AAA", "date": trading_dates[2], "label_end_date": trading_dates[2 + 5]}]


def test_build_training_dataset_excludes_dates_without_enough_forward_history() -> None:
    trading_dates = _dates("2024-01-01", 8)
    prices = [100.0 + index for index in range(8)]
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    history.update(_flat_history("SPY", trading_dates))
    # Request a sample date only 2 sessions from the end of the stored history, with horizon=5:
    # index 6 + horizon 5 = 11 >= len(history)=8, so this date must be excluded even though a
    # caller passed it in directly (defense in depth beyond select_sample_dates' own embargo).
    results_by_date = {trading_dates[6]: {"AAA": _result("AAA", trading_dates[6])}}
    service = _FakeService(results_by_date)

    features, labels, meta = build_training_dataset(
        service, ["AAA"], history, [date.fromisoformat(trading_dates[6])],
        horizon_days=5, feature_keys=["rsi_14"], benchmark_symbol="SPY",
    )

    assert features.shape[0] == 0
    assert meta == []


def test_build_training_dataset_drops_rows_missing_benchmark_price() -> None:
    trading_dates = _dates("2024-01-01", 10)
    prices = [100.0 + index for index in range(10)]
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    # No "SPY" entry at all -> every row lacks a benchmark price and must be dropped.
    results_by_date = {trading_dates[0]: {"AAA": _result("AAA", trading_dates[0])}}
    service = _FakeService(results_by_date)

    features, labels, meta = build_training_dataset(
        service, ["AAA"], history, [date.fromisoformat(trading_dates[0])],
        horizon_days=5, feature_keys=["rsi_14"], benchmark_symbol="SPY",
    )

    assert features.shape[0] == 0
    assert meta == []


def test_build_regression_dataset_target_is_continuous_excess_return() -> None:
    trading_dates = _dates("2024-01-01", 20)
    prices = [100.0 + index for index in range(20)]  # +1/session -> known excess return
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    history.update(_flat_history("SPY", trading_dates))  # benchmark return is always 0
    results_by_date = {
        d: {"AAA": _result("AAA", d, rsi=50.0 + index)} for index, d in enumerate(trading_dates)
    }
    service = _FakeService(results_by_date)
    sample_date = date.fromisoformat(trading_dates[0])

    features, targets, meta = build_regression_dataset(
        service, ["AAA"], history, [sample_date],
        horizon_days=5, feature_keys=["rsi_14"], benchmark_symbol="SPY",
    )

    # entry=100, exit=105 (5 sessions later at +1/session) -> symbol_return=0.05;
    # benchmark is flat -> benchmark_return=0.0 -> excess_return=0.05, not a binary 1.0.
    assert features.shape[0] == 1
    assert targets[0] == pytest.approx(0.05)
    assert meta == [{"symbol": "AAA", "date": trading_dates[0], "label_end_date": trading_dates[5]}]


def test_build_regression_dataset_target_can_be_negative() -> None:
    trading_dates = _dates("2024-01-01", 20)
    prices = [100.0 - index for index in range(20)]  # falling price -> negative excess return
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    history.update(_flat_history("SPY", trading_dates))
    results_by_date = {trading_dates[0]: {"AAA": _result("AAA", trading_dates[0])}}
    service = _FakeService(results_by_date)

    features, targets, meta = build_regression_dataset(
        service, ["AAA"], history, [date.fromisoformat(trading_dates[0])],
        horizon_days=5, feature_keys=["rsi_14"], benchmark_symbol="SPY",
    )

    assert features.shape[0] == 1
    assert targets[0] < 0.0


def test_build_regression_dataset_shares_eligibility_and_missing_feature_filtering() -> None:
    trading_dates = _dates("2024-01-01", 10)
    prices = [100.0] * 10
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    history.update(_flat_history("SPY", trading_dates))
    results_by_date = {
        trading_dates[0]: {"AAA": _result("AAA", trading_dates[0], eligible=False)},
        trading_dates[1]: {"AAA": _result("AAA", trading_dates[1], rsi=None)},
        trading_dates[2]: {"AAA": _result("AAA", trading_dates[2], rsi=60.0)},
    }
    service = _FakeService(results_by_date)
    sample_dates = [date.fromisoformat(trading_dates[index]) for index in range(3)]

    features, targets, meta = build_regression_dataset(
        service, ["AAA"], history, sample_dates,
        horizon_days=5, feature_keys=["rsi_14"], benchmark_symbol="SPY",
    )

    assert features.shape[0] == 1
    assert meta == [{"symbol": "AAA", "date": trading_dates[2], "label_end_date": trading_dates[2 + 5]}]


def test_feature_value_falls_back_to_fundamentals_when_indicator_missing() -> None:
    result = _result("AAA", "2024-01-01", rsi=None)
    value = feature_value(result, "trailing_pe", {}, {"trailing_pe": 17.5})
    assert value == 17.5


def test_feature_value_prefers_indicator_over_fundamentals_when_both_present() -> None:
    result = _result("AAA", "2024-01-01", rsi=50.0)
    # "rsi_14" is a real indicator key; a fundamentals dict happening to also carry
    # that key must never override the indicator lookup -- fundamentals only fill
    # in for keys the indicator dict genuinely doesn't have.
    value = feature_value(result, "rsi_14", {}, {"rsi_14": 999.0})
    assert value == 50.0


def test_feature_value_returns_none_when_fundamentals_key_is_unavailable() -> None:
    result = _result("AAA", "2024-01-01", rsi=None)
    assert feature_value(result, "trailing_pe", {}, {"trailing_pe": None}) is None
    assert feature_value(result, "trailing_pe", {}, None) is None


def test_build_regression_dataset_resolves_fundamental_features_from_fundamentals_by_symbol() -> None:
    trading_dates = _dates("2024-01-01", 20)
    prices = [100.0 + index for index in range(20)]
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    history.update(_flat_history("SPY", trading_dates))
    sample_date = trading_dates[10]  # entry_close = 100 + 10 = 110.0
    results_by_date = {sample_date: {"AAA": _result("AAA", sample_date, rsi=None)}}
    service = _FakeService(results_by_date)
    # Four quarterly eps_diluted facts, all filed well before sample_date, so
    # trailing_four_quarter_sum has a full TTM figure to work with: 1+1+1+1 = 4.0.
    fundamentals_by_symbol = {
        "AAA": [
            {"concept": "eps_diluted", "period_start": "2022-10-01", "period_end": "2022-12-31", "filed_date": "2023-01-15", "value": 1.0},
            {"concept": "eps_diluted", "period_start": "2023-01-01", "period_end": "2023-03-31", "filed_date": "2023-04-15", "value": 1.0},
            {"concept": "eps_diluted", "period_start": "2023-04-01", "period_end": "2023-06-30", "filed_date": "2023-07-15", "value": 1.0},
            {"concept": "eps_diluted", "period_start": "2023-07-01", "period_end": "2023-09-30", "filed_date": "2023-10-15", "value": 1.0},
        ],
    }

    features, targets, meta = build_regression_dataset(
        service, ["AAA"], history, [date.fromisoformat(sample_date)],
        horizon_days=5, feature_keys=["trailing_pe"], benchmark_symbol="SPY",
        fundamentals_by_symbol=fundamentals_by_symbol,
    )

    assert features.shape[0] == 1
    assert features[0][0] == pytest.approx(110.0 / 4.0)  # price / TTM eps
    assert meta == [{"symbol": "AAA", "date": sample_date, "label_end_date": trading_dates[15]}]


def test_build_regression_dataset_drops_rows_when_fundamentals_are_not_yet_known() -> None:
    """A fundamentals key with no qualifying quarters on file as of the sample
    date must drop the row -- same missing-feature filtering every other key
    already gets, never a fabricated 0."""
    trading_dates = _dates("2024-01-01", 20)
    prices = [100.0 + index for index in range(20)]
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    history.update(_flat_history("SPY", trading_dates))
    sample_date = trading_dates[10]
    results_by_date = {sample_date: {"AAA": _result("AAA", sample_date, rsi=None)}}
    service = _FakeService(results_by_date)
    fundamentals_by_symbol = {
        "AAA": [
            {
                "concept": "eps_diluted", "period_start": "2023-10-01",
                "period_end": "2023-12-31", "filed_date": "2024-01-01", "value": 5.0,
            },
        ],
    }

    features, targets, meta = build_regression_dataset(
        service, ["AAA"], history, [date.fromisoformat(sample_date)],
        horizon_days=5, feature_keys=["trailing_pe"], benchmark_symbol="SPY",
        fundamentals_by_symbol=fundamentals_by_symbol,
    )

    assert features.shape[0] == 0
    assert meta == []


def test_build_regression_dataset_ignores_fundamentals_by_symbol_when_not_provided() -> None:
    """Every existing caller omits fundamentals_by_symbol -- behavior must be
    byte-for-byte identical to before this parameter existed."""
    trading_dates = _dates("2024-01-01", 20)
    prices = [100.0 + index for index in range(20)]
    history = {"AAA": [{"trade_date": d, "adjusted_close": p} for d, p in zip(trading_dates, prices)]}
    history.update(_flat_history("SPY", trading_dates))
    sample_date = date.fromisoformat(trading_dates[0])
    results_by_date = {trading_dates[0]: {"AAA": _result("AAA", trading_dates[0], rsi=55.0)}}
    service = _FakeService(results_by_date)

    features, targets, meta = build_regression_dataset(
        service, ["AAA"], history, [sample_date],
        horizon_days=5, feature_keys=["rsi_14"], benchmark_symbol="SPY",
    )

    assert features.shape[0] == 1
    assert features[0][0] == 55.0


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
    history.update(_flat_history("SPY", trading_dates))
    results_by_date = {
        trading_dates[0]: {"AAA": _result("AAA", trading_dates[0], market_regime="Insufficient Market Data")},
        trading_dates[1]: {"AAA": _result("AAA", trading_dates[1], market_regime="Risk-On")},
    }
    service = _FakeService(results_by_date)
    sample_dates = [date.fromisoformat(trading_dates[index]) for index in range(2)]

    features, labels, meta = build_training_dataset(
        service, ["AAA"], history, sample_dates,
        horizon_days=5, feature_keys=["rsi_14", "market_regime_code"], benchmark_symbol="SPY",
    )

    assert features.shape[0] == 1
    assert meta == [{"symbol": "AAA", "date": trading_dates[1], "label_end_date": trading_dates[1 + 5]}]
