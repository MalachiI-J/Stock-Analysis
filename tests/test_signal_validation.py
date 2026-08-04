from __future__ import annotations

from datetime import date, timedelta

import pytest

from stock_scrapper.analysis.signal_validation import (
    MIN_OUTCOME_SAMPLE_SIZE,
    compute_checkpoint_comparisons,
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


def _long_dates(count: int) -> list[str]:
    """Like ``_dates``, but safe past a single month (real calendar arithmetic) --
    for fixtures that need dozens/hundreds of trailing rows."""
    start = date(2020, 1, 1)
    return [(start + timedelta(days=index)).isoformat() for index in range(count)]


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


def test_validate_signals_bonferroni_p_value_corrects_for_comparisons_tested() -> None:
    dates = _dates(10)
    histories = {
        "SPY": _history(dates, [100] * 10),
        "AAA": _history(dates, [100, 100, 100, 100, 100, 110, 100, 100, 100, 100]),
        "BBB": _history(dates, [100, 100, 100, 100, 100, 90, 100, 100, 100, 100]),
    }
    # Exactly two non-empty buckets -> comparisons_tested == 2.
    signals = [
        _signal("AAA", "2024-01-01", "Strong Candidate"),
        _signal("BBB", "2024-01-01", "Avoid"),
    ]

    result = validate_signals(signals, histories, benchmark_symbol="SPY", horizon_days=5)

    by_classification = {bucket.classification: bucket for bucket in result.buckets}
    for classification in ("Strong Candidate", "Avoid"):
        bucket = by_classification[classification]
        assert bucket.p_value is not None
        assert bucket.bonferroni_p_value == pytest.approx(min(1.0, bucket.p_value * 2))


def test_validate_signals_bonferroni_p_value_caps_at_one() -> None:
    dates = _dates(10)
    histories = {
        "SPY": _history(dates, [100] * 10),
        "AAA": _history(dates, [100, 100, 100, 100, 100, 101, 100, 100, 100, 100]),
        "BBB": _history(dates, [100, 100, 100, 100, 100, 99, 100, 100, 100, 100]),
    }
    signals = [
        _signal("AAA", "2024-01-01", "Strong Candidate"),
        _signal("BBB", "2024-01-01", "Avoid"),
    ]

    result = validate_signals(signals, histories, benchmark_symbol="SPY", horizon_days=5)

    for bucket in result.buckets:
        if bucket.bonferroni_p_value is not None:
            assert bucket.bonferroni_p_value <= 1.0


def test_validate_signals_symbol_mean_ci_requires_at_least_two_symbols() -> None:
    dates = _dates(10)
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
    strong = by_classification["Strong Candidate"]  # only 1 distinct symbol -> no CI possible
    assert strong.distinct_symbols == 1
    assert strong.symbol_mean_excess_return_ci_low is None
    assert strong.symbol_mean_excess_return_ci_high is None

    watch = by_classification["Watch"]  # 3 distinct symbols -> a CI can be estimated
    assert watch.distinct_symbols == 3
    assert watch.symbol_mean_excess_return_ci_low is not None
    assert watch.symbol_mean_excess_return_ci_high is not None
    assert watch.symbol_mean_excess_return_ci_low <= watch.symbol_mean_excess_return
    assert watch.symbol_mean_excess_return <= watch.symbol_mean_excess_return_ci_high


def test_render_signal_validation_text_labels_p_values_as_naive_and_shows_ci() -> None:
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
    text = render_signal_validation_text(result)

    assert "naive p" in text
    assert "bonf. p" in text
    assert "descriptive, not validated inferential statistics" in text
    assert "symbol-weighted mean excess return" in text
    assert "95% CI" in text or "CI unavailable" in text


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


def test_validate_signals_leaves_percentiles_none_below_sample_size_floor() -> None:
    dates = _dates(10)
    histories = {
        "SPY": _history(dates, [100] * 10),
        "AAA": _history(dates, [100, 100, 100, 100, 100, 110, 100, 100, 100, 100]),
    }
    signals = [_signal("AAA", "2024-01-01", "Strong Candidate")]

    result = validate_signals(signals, histories, benchmark_symbol="SPY", horizon_days=5)

    bucket = {b.classification: b for b in result.buckets}["Strong Candidate"]
    assert bucket.outcome_sample_size == 1
    assert bucket.p10_excess_return is None
    assert bucket.p90_excess_return is None
    # Mean/median still populate regardless -- only the percentile range needs the floor.
    assert bucket.mean_excess_return == pytest.approx(0.10)


def test_validate_signals_computes_percentile_range_when_sample_size_clears_floor() -> None:
    horizon = 5
    n = 25
    spacing = horizon + 1  # keeps every signal's entry/future window from overlapping
    total_days = n * spacing + horizon + 2
    dates = _long_dates(total_days)
    prices = [100.0] * total_days
    signals = []
    for index in range(n):
        entry_index = index * spacing
        prices[entry_index + horizon] = 100.0 * (1 + index / 100.0)  # 0%, 1%, ..., 24%
        signals.append(_signal("AAA", dates[entry_index], "Strong Candidate"))
    histories = {"SPY": _history(dates, [100.0] * total_days), "AAA": _history(dates, prices)}

    result = validate_signals(signals, histories, benchmark_symbol="SPY", horizon_days=horizon)

    bucket = {b.classification: b for b in result.buckets}["Strong Candidate"]
    assert bucket.outcome_sample_size == 25
    # Linear-interpolation percentile of 0%, 1%, ..., 24%: p10 -> position 2.4, p90 -> position 21.6.
    assert bucket.p10_excess_return == pytest.approx(0.024, abs=1e-6)
    assert bucket.p90_excess_return == pytest.approx(0.216, abs=1e-6)


def _checkpoint_fixture(
    *, better_horizon: int | None, better_return: float = 0.05, base_return: float = 0.01,
    signals_per_symbol: int = 4, symbol_count: int = 5,
) -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    """Builds a fixture where every symbol's every signal has the exact same excess
    return at each checkpoint horizon (zero cross-symbol variance -> the symbol-weighted
    CI degenerates to a single point), so "meaningfully better" reduces to a plain
    greater-than comparison on deterministic numbers instead of a statistical judgment
    call. ``better_horizon`` (if given) gets ``better_return``; every other checkpoint
    horizon gets ``base_return`` for every signal.
    """
    from stock_scrapper.analysis.signal_validation import CHECKPOINT_SESSIONS

    spacing = max(CHECKPOINT_SESSIONS) + 10
    total_days = signals_per_symbol * spacing + max(CHECKPOINT_SESSIONS) + 5
    dates = _long_dates(total_days)
    histories: dict[str, list[dict[str, object]]] = {"SPY": _history(dates, [100.0] * total_days)}
    signals: list[dict[str, object]] = []
    for symbol_index in range(symbol_count):
        symbol = f"SYM{symbol_index}"
        prices = [100.0] * total_days
        for signal_index in range(signals_per_symbol):
            entry_index = signal_index * spacing
            for horizon in CHECKPOINT_SESSIONS:
                return_for_horizon = better_return if horizon == better_horizon else base_return
                prices[entry_index + horizon] = 100.0 * (1 + return_for_horizon)
            signals.append(_signal(symbol, dates[entry_index], "Strong Candidate"))
        histories[symbol] = _history(dates, prices)
    return signals, histories


def test_compute_checkpoint_comparisons_picks_meaningfully_better_checkpoint() -> None:
    signals, histories = _checkpoint_fixture(better_horizon=26)

    results = compute_checkpoint_comparisons(
        signals, histories, benchmark_symbol="SPY", standard_horizon_days=21,
    )

    analysis = results["Strong Candidate"]
    assert analysis.best_checkpoint_meaningfully_better is True
    assert analysis.best_checkpoint_sessions == 26
    sessions_seen = sorted(c.sessions for c in analysis.checkpoints)
    assert sessions_seen == [5, 10, 15, 21, 26, 31]
    winner = next(c for c in analysis.checkpoints if c.sessions == 26)
    assert winner.p10_excess_return == pytest.approx(0.05)
    assert winner.outcome_sample_size == 20


def test_compute_checkpoint_comparisons_no_winner_when_nothing_beats_standard() -> None:
    signals, histories = _checkpoint_fixture(better_horizon=None)

    results = compute_checkpoint_comparisons(
        signals, histories, benchmark_symbol="SPY", standard_horizon_days=21,
    )

    analysis = results["Strong Candidate"]
    assert analysis.best_checkpoint_meaningfully_better is False
    assert analysis.best_checkpoint_sessions is None
    assert len(analysis.checkpoints) == 6


def test_compute_checkpoint_comparisons_requires_outcome_sample_size_floor() -> None:
    # Only 3 signals per symbol per checkpoint (well below MIN_OUTCOME_SAMPLE_SIZE=20
    # when spread across just enough symbols), even though checkpoint 26's numbers
    # would otherwise "win" on the numbers alone.
    signals_per_symbol = 3
    symbol_count = 5
    assert signals_per_symbol * symbol_count < MIN_OUTCOME_SAMPLE_SIZE
    signals, histories = _checkpoint_fixture(
        better_horizon=26, signals_per_symbol=signals_per_symbol, symbol_count=symbol_count,
    )

    results = compute_checkpoint_comparisons(
        signals, histories, benchmark_symbol="SPY", standard_horizon_days=21,
    )

    analysis = results["Strong Candidate"]
    assert analysis.best_checkpoint_meaningfully_better is False
    assert analysis.best_checkpoint_sessions is None


def test_compute_checkpoint_comparisons_folds_in_standard_horizon_even_if_not_a_checkpoint() -> None:
    signals, histories = _checkpoint_fixture(better_horizon=None)

    results = compute_checkpoint_comparisons(
        signals, histories, benchmark_symbol="SPY", standard_horizon_days=17,
        checkpoint_sessions=(5, 10, 15, 21, 26, 31),
    )

    analysis = results["Strong Candidate"]
    assert analysis.standard_horizon_sessions == 17
    sessions_seen = sorted(c.sessions for c in analysis.checkpoints)
    assert sessions_seen == [5, 10, 15, 17, 21, 26, 31]
