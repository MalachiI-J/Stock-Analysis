from __future__ import annotations

import pytest

from stock_scrapper.analysis.risk_diagnostics import (
    build_classified_rows,
    investigate_classification,
    pearson_correlation,
    render_risk_diagnostics_text,
    study_driver_features,
)


def _history(dates: list[str], prices: list[float]) -> list[dict[str, object]]:
    return [{"trade_date": date, "adjusted_close": price} for date, price in zip(dates, prices)]


def _dates(count: int) -> list[str]:
    return [f"2024-01-{index + 1:02d}" for index in range(count)]


def _signal(symbol: str, signal_date: str, classification: str, indicators: dict[str, object]) -> dict[str, object]:
    return {
        "symbol": symbol,
        "signal_date": signal_date,
        "classification": classification,
        "indicators": indicators,
    }


def test_pearson_correlation_perfect_positive() -> None:
    assert pearson_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_pearson_correlation_perfect_negative() -> None:
    assert pearson_correlation([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_pearson_correlation_zero_variance_returns_none() -> None:
    assert pearson_correlation([1, 1, 1], [1, 2, 3]) is None


def test_pearson_correlation_single_point_returns_none() -> None:
    assert pearson_correlation([1], [1]) is None


def test_study_driver_features_orders_quintiles_low_to_high_and_computes_means() -> None:
    # 10 rows, feature and excess return both strictly increasing with i -> quintile
    # means should themselves increase monotonically, and correlation should be +1.
    rows = [{"driver": float(i), "excess_return": i * 0.01} for i in range(10)]

    studies = study_driver_features(rows, features=("driver",))

    assert len(studies) == 1
    study = studies[0]
    assert study.n == 10
    assert study.pearson_r == pytest.approx(1.0)
    assert len(study.quintiles) == 5
    means = [bucket.mean_excess_return for bucket in study.quintiles]
    assert means == sorted(means)
    assert study.quintiles[0].mean_excess_return == pytest.approx(0.005)
    assert study.quintiles[-1].mean_excess_return == pytest.approx(0.085)


def test_study_driver_features_skips_feature_with_too_few_rows() -> None:
    rows = [{"driver": float(i), "excess_return": i * 0.01} for i in range(5)]

    studies = study_driver_features(rows, features=("driver",))

    assert studies == []


def test_study_driver_features_skips_missing_values() -> None:
    rows = [{"driver": None, "excess_return": 0.01} for _ in range(10)]

    studies = study_driver_features(rows, features=("driver",))

    assert studies == []


def test_build_classified_rows_pairs_indicators_with_forward_excess_return() -> None:
    dates = _dates(10)
    histories = {
        "SPY": _history(dates, [100.0] * 10),
        "AAA": _history(dates, [100.0, 100, 100, 100, 100, 110, 100, 100, 100, 100]),
    }
    signals = [_signal("AAA", "2024-01-01", "High Risk", {"atr_percentage": 0.05})]

    rows = build_classified_rows(
        signals, histories, classification="High Risk", benchmark_symbol="SPY", horizon_days=5,
        features=("atr_percentage",),
    )

    assert len(rows) == 1
    assert rows[0]["symbol"] == "AAA"
    assert rows[0]["excess_return"] == pytest.approx(0.10)
    assert rows[0]["atr_percentage"] == pytest.approx(0.05)


def test_build_classified_rows_ignores_other_classifications() -> None:
    dates = _dates(10)
    histories = {
        "SPY": _history(dates, [100.0] * 10),
        "AAA": _history(dates, [100.0] * 10),
    }
    signals = [_signal("AAA", "2024-01-01", "Watch", {"atr_percentage": 0.05})]

    rows = build_classified_rows(
        signals, histories, classification="High Risk", benchmark_symbol="SPY", horizon_days=5,
    )

    assert rows == []


def test_investigate_classification_reports_sample_size_and_distinct_symbols() -> None:
    dates = _dates(10)
    histories = {"SPY": _history(dates, [100.0] * 10)}
    signals = []
    for i in range(10):
        symbol = f"SYM{i}"
        # Symbol i's forward return is i * 1%, so excess return over the flat
        # benchmark is also i * 1% — deliberately increasing with the feature value.
        histories[symbol] = _history(dates, [100.0, 100, 100, 100, 100, 100 * (1 + i * 0.01), 100, 100, 100, 100])
        signals.append(_signal(symbol, "2024-01-01", "High Risk", {"atr_percentage": float(i)}))

    result = investigate_classification(
        signals, histories, classification="High Risk", benchmark_symbol="SPY", horizon_days=5,
        features=("atr_percentage",),
    )

    assert result.classification == "High Risk"
    assert result.sample_size == 10
    assert result.distinct_symbols == 10
    assert len(result.driver_studies) == 1
    assert result.driver_studies[0].pearson_r == pytest.approx(1.0)


def test_investigate_classification_empty_when_no_rows_match() -> None:
    dates = _dates(10)
    histories = {"SPY": _history(dates, [100.0] * 10)}

    result = investigate_classification(
        [], histories, classification="High Risk", benchmark_symbol="SPY", horizon_days=5,
    )

    assert result.sample_size == 0
    assert result.distinct_symbols == 0
    assert result.driver_studies == []


def test_render_risk_diagnostics_text_includes_classification_and_sample_size() -> None:
    dates = _dates(10)
    histories = {"SPY": _history(dates, [100.0] * 10)}
    result = investigate_classification(
        [], histories, classification="High Risk", benchmark_symbol="SPY", horizon_days=5,
    )

    text = render_risk_diagnostics_text(result)

    assert "High Risk" in text
    assert "sample size: 0" in text
    assert "no feature had enough non-missing rows to study" in text
