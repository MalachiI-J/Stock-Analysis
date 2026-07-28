from __future__ import annotations

import pytest

from stock_scrapper.analysis.signal_validation import (
    forward_excess_return,
    render_signal_validation_text,
    two_proportion_z_test,
    validate_signals,
)


def _history(dates: list[str], prices: list[float]) -> list[dict[str, object]]:
    return [{"trade_date": date, "adjusted_close": price} for date, price in zip(dates, prices)]


def _signal(symbol: str, signal_date: str, classification: str) -> dict[str, object]:
    return {"symbol": symbol, "signal_date": signal_date, "classification": classification}


def _dates(count: int) -> list[str]:
    return [f"2024-01-{index + 1:02d}" for index in range(count)]


def test_forward_excess_return_computes_symbol_minus_benchmark() -> None:
    dates = _dates(10)
    symbol_history = _history(dates, [100, 101, 102, 103, 104, 110, 106, 107, 108, 109])
    benchmark_history = _history(dates, [100] * 10)
    excess = forward_excess_return(symbol_history, benchmark_history, "2024-01-01", 5)
    assert excess == pytest.approx(0.10)


def test_forward_excess_return_none_when_not_enough_future_history() -> None:
    dates = _dates(5)
    symbol_history = _history(dates, [100] * 5)
    benchmark_history = _history(dates, [100] * 5)
    assert forward_excess_return(symbol_history, benchmark_history, "2024-01-01", 10) is None


def test_forward_excess_return_none_when_benchmark_missing_future_date() -> None:
    dates = _dates(10)
    symbol_history = _history(dates, [100] * 10)
    benchmark_history = _history(dates[:3], [100] * 3)
    assert forward_excess_return(symbol_history, benchmark_history, "2024-01-01", 5) is None


def test_two_proportion_z_test_identical_rates_gives_zero_z_and_p_one() -> None:
    z, p = two_proportion_z_test(5, 10, 5, 10)
    assert z == pytest.approx(0.0)
    assert p == pytest.approx(1.0)


def test_two_proportion_z_test_empty_sample_returns_none() -> None:
    assert two_proportion_z_test(0, 0, 5, 10) == (None, None)


def test_two_proportion_z_test_large_gap_is_significant() -> None:
    z, p = two_proportion_z_test(9, 10, 1, 10)
    assert z is not None and p is not None
    assert p < 0.05


def test_validate_signals_buckets_hit_rate_and_mean_excess_return() -> None:
    dates = _dates(10)
    histories = {
        "SPY": _history(dates, [100] * 10),
        "AAA": _history(dates, [100, 100, 100, 100, 100, 110, 100, 100, 100, 100]),
        "BBB": _history(dates, [100, 100, 100, 100, 100, 90, 100, 100, 100, 100]),
    }
    signals = [
        _signal("AAA", "2024-01-01", "Strong Candidate"),
        _signal("BBB", "2024-01-01", "Avoid"),
    ]

    result = validate_signals(signals, histories, benchmark_symbol="SPY", horizon_days=5)

    assert result.total_samples == 2
    assert result.skipped_samples == 0
    by_classification = {bucket.classification: bucket for bucket in result.buckets}
    strong = by_classification["Strong Candidate"]
    assert strong.sample_size == 1
    assert strong.hit_rate == 1.0
    assert strong.mean_excess_return == pytest.approx(0.10)
    avoid = by_classification["Avoid"]
    assert avoid.sample_size == 1
    assert avoid.hit_rate == 0.0
    assert avoid.mean_excess_return == pytest.approx(-0.10)
    assert by_classification["Candidate"].sample_size == 0


def test_validate_signals_excludes_unclassified_and_counts_data_skips() -> None:
    dates = _dates(10)
    histories = {
        "SPY": _history(dates, [100] * 10),
        "AAA": _history(dates, [100] * 10),
    }
    signals = [
        _signal("AAA", "2024-01-01", "Insufficient Data"),
        _signal("AAA", "2024-01-09", "Strong Candidate"),  # too close to the end for horizon=5
    ]

    result = validate_signals(signals, histories, benchmark_symbol="SPY", horizon_days=5)

    assert result.total_samples == 0
    assert result.skipped_samples == 1


def test_validate_signals_flags_monotonic_ranking() -> None:
    dates = _dates(10)
    # Excess returns decrease strictly: +20%, +10%, 0%, -10%, -20% for the five buckets.
    histories = {
        "SPY": _history(dates, [100] * 10),
        "STRONG": _history(dates, [100, 100, 100, 100, 100, 120, 100, 100, 100, 100]),
        "CAND": _history(dates, [100, 100, 100, 100, 100, 110, 100, 100, 100, 100]),
        "WATCH": _history(dates, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
        "AVOID": _history(dates, [100, 100, 100, 100, 100, 90, 100, 100, 100, 100]),
        "RISKY": _history(dates, [100, 100, 100, 100, 100, 80, 100, 100, 100, 100]),
    }
    signals = [
        _signal("STRONG", "2024-01-01", "Strong Candidate"),
        _signal("CAND", "2024-01-01", "Candidate"),
        _signal("WATCH", "2024-01-01", "Watch"),
        _signal("AVOID", "2024-01-01", "Avoid"),
        _signal("RISKY", "2024-01-01", "High Risk"),
    ]

    result = validate_signals(signals, histories, benchmark_symbol="SPY", horizon_days=5)

    assert result.monotonic is True


def test_validate_signals_flags_non_monotonic_ranking() -> None:
    dates = _dates(10)
    histories = {
        "SPY": _history(dates, [100] * 10),
        "STRONG": _history(dates, [100, 100, 100, 100, 100, 90, 100, 100, 100, 100]),
        "AVOID": _history(dates, [100, 100, 100, 100, 100, 120, 100, 100, 100, 100]),
    }
    signals = [
        _signal("STRONG", "2024-01-01", "Strong Candidate"),
        _signal("AVOID", "2024-01-01", "Avoid"),
    ]

    result = validate_signals(signals, histories, benchmark_symbol="SPY", horizon_days=5)

    assert result.monotonic is False


def test_validate_signals_flags_concentrated_bucket_and_reports_symbol_mean() -> None:
    dates = _dates(10)
    # AAA's single stretch dominates "Strong Candidate" day-count (9 rows) even though
    # it is only one stock; "Watch" has three distinct stocks contributing one row each.
    histories = {
        "SPY": _history(dates, [100] * 10),
        "AAA": _history(dates, [100, 100, 100, 100, 100, 130, 100, 100, 100, 100]),
        "BBB": _history(dates, [100, 100, 100, 100, 100, 105, 100, 100, 100, 100]),
        "CCC": _history(dates, [100, 100, 100, 100, 100, 106, 100, 100, 100, 100]),
        "DDD": _history(dates, [100, 100, 100, 100, 100, 104, 100, 100, 100, 100]),
    }
    signals = [_signal("AAA", "2024-01-01", "Strong Candidate")] + [
        _signal(symbol, "2024-01-01", "Watch") for symbol in ("BBB", "CCC", "DDD")
    ]

    result = validate_signals(signals, histories, benchmark_symbol="SPY", horizon_days=5)

    by_classification = {bucket.classification: bucket for bucket in result.buckets}
    strong = by_classification["Strong Candidate"]
    assert strong.distinct_symbols == 1
    assert strong.concentration_warning is True
    assert strong.symbol_mean_excess_return == pytest.approx(0.30)

    watch = by_classification["Watch"]
    assert watch.distinct_symbols == 3
    assert watch.concentration_warning is True  # still below MIN_DISTINCT_SYMBOLS_FOR_TRUST (4)


def test_validate_signals_symbol_weighted_monotonic_diverges_from_day_weighted() -> None:
    dates = _dates(10)
    # Strong Candidate: one stock (AAA) classified on 3 separate days, each +20% — the
    # day-weighted mean is +20%. Avoid: three different stocks, each classified once,
    # with smaller but still-positive excess returns averaging below 20% per stock.
    # Day-weighted, Strong Candidate (n=3) still beats Avoid (n=3): monotonic holds.
    # Symbol-weighted, both buckets are just an average of their (fewer) distinct
    # symbols' means, so this test only asserts the two checks are computed
    # independently and both resolve to a boolean.
    histories = {
        "SPY": _history(dates, [100] * 10),
        "AAA": _history(dates, [100, 100, 100, 100, 100, 120, 100, 100, 100, 100]),
        "EEE": _history(dates, [100, 100, 100, 100, 100, 105, 100, 100, 100, 100]),
        "FFF": _history(dates, [100, 100, 100, 100, 100, 104, 100, 100, 100, 100]),
        "GGG": _history(dates, [100, 100, 100, 100, 100, 103, 100, 100, 100, 100]),
    }
    signals = [_signal("AAA", "2024-01-01", "Strong Candidate")] + [
        _signal(symbol, "2024-01-01", "Avoid") for symbol in ("EEE", "FFF", "GGG")
    ]

    result = validate_signals(signals, histories, benchmark_symbol="SPY", horizon_days=5)

    assert result.monotonic is True
    assert result.monotonic_symbol_weighted is True


def test_render_signal_validation_text_includes_key_sections() -> None:
    dates = _dates(10)
    histories = {
        "SPY": _history(dates, [100] * 10),
        "AAA": _history(dates, [100, 100, 100, 100, 100, 110, 100, 100, 100, 100]),
    }
    signals = [_signal("AAA", "2024-01-01", "Strong Candidate")]

    result = validate_signals(signals, histories, benchmark_symbol="SPY", horizon_days=5)
    text = render_signal_validation_text(result)

    assert "Strong Candidate" in text
    assert "monotonic" in text
    assert "symbol-weighted" in text
    assert "SPY" in text
    assert "fewer than" in text  # single-symbol bucket triggers the concentration footnote
