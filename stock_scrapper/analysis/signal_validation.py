"""Does score_v1's classification carry real information, independent of the backtester?

The portfolio backtester answers "would this specific sizing/exit/cost configuration
have made money" — a question tangled up with position caps, entry timing, and trade
mechanics. It also cannot be run over more than one independent validation window given
how little history this project has (~5 years), so a single portfolio-level verdict
isn't enough evidence either way.

This module asks a narrower, higher-power question instead: across every day a symbol
was ever classified, regardless of whether the portfolio had room to act on it, did that
classification's forward performance actually differ from the rest? Every classified
instance is a *row* (thousands of them, versus a couple of backtest windows) — but rows
are not independent trials. ``two_proportion_z_test``'s p-values assume every row is an
independent draw, which two things violate here: (1) each row's forward-return window is
``horizon_days`` sessions long and overlaps its neighbors, so adjacent daily rows for the
same symbol mostly restate the same underlying move, and (2) rows cluster by symbol — a
handful of stocks classified for a long stretch can dominate a bucket's day-count even
after the ``MIN_DISTINCT_SYMBOLS_FOR_TRUST`` concentration check passes. The p-values this
module reports are therefore **naive/descriptive, not validated inferential statistics**;
they are surfaced (clearly labeled) alongside a symbol-weighted mean and its confidence
interval — treating each *symbol* as the independent unit — which is the more defensible
read of whether a bucket's apparent difference reflects a broad pattern versus a
concentration-driven mirage. A Bonferroni correction is applied across the five
classification-bucket comparisons so the naive p-values at least aren't treated as if
only one test were run.

Deliberately hand-rolled (``math.erf`` for the normal CDF) rather than adding scipy —
consistent with the rest of this project's no-scipy/no-scikit-learn design.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

# Ordered from most to least favorable — used only to report mean excess return in a
# fixed, readable order and to check monotonicity; "Insufficient Data"/"Data Blocked"
# carry no opportunity signal and are excluded entirely.
CLASSIFICATION_ORDER: tuple[str, ...] = ("Strong Candidate", "Candidate", "Watch", "Avoid", "High Risk")

# Below this many distinct symbols, a bucket's day-level z-test is not trustworthy: with
# horizon_days-long overlapping lookaheads, one symbol's classified stretch produces many
# daily rows that mostly restate the same underlying move, not independent trials. A
# handful of stocks (or one) going on a single big run can then look like a broad,
# statistically "significant" population effect when it is really one or two anecdotes.
MIN_DISTINCT_SYMBOLS_FOR_TRUST = 4

# Below this many individual forward-return observations, a p10/p90 percentile range is
# noisy enough (e.g. the "10th percentile" of 5 values is really just picking one of
# them) that it reads as more precise than it is. Below this floor, the range is left as
# ``None`` rather than shown -- see ClassificationBucket.p10_excess_return/p90_excess_return.
MIN_OUTCOME_SAMPLE_SIZE = 20


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _get(signal: Any, name: str) -> Any:
    return signal.get(name) if isinstance(signal, Mapping) else getattr(signal, name, None)


def _date_index(history: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        str(row.get("trade_date"))[:10]: index
        for index, row in enumerate(history)
        if row.get("trade_date")
    }


def forward_excess_return(
    symbol_history: Sequence[Mapping[str, Any]],
    benchmark_history: Sequence[Mapping[str, Any]],
    signal_date: str,
    horizon_days: int,
    *,
    symbol_index: Mapping[str, int] | None = None,
    benchmark_index: Mapping[str, int] | None = None,
) -> float | None:
    """The symbol's return ``horizon_days`` sessions after ``signal_date``, minus the
    benchmark's return over that identical calendar span. ``None`` if either history
    doesn't reach that far, or the reference/future closes aren't usable — this mirrors
    the exact same index-based lookahead ``stock_scrapper/prediction/dataset.py`` already
    uses (N rows ahead in a bounded, sorted history, not calendar-day arithmetic, which
    sidesteps weekends/holidays for free) so both validations stay directly comparable.
    """
    symbol_index = symbol_index if symbol_index is not None else _date_index(symbol_history)
    benchmark_index = benchmark_index if benchmark_index is not None else _date_index(benchmark_history)

    index = symbol_index.get(signal_date)
    if index is None or index + horizon_days >= len(symbol_history):
        return None
    entry_close = _finite(symbol_history[index].get("adjusted_close"))
    future_row = symbol_history[index + horizon_days]
    future_close = _finite(future_row.get("adjusted_close"))
    if entry_close is None or future_close is None or entry_close <= 0:
        return None

    benchmark_entry_index = benchmark_index.get(signal_date)
    benchmark_future_index = benchmark_index.get(str(future_row.get("trade_date"))[:10])
    if benchmark_entry_index is None or benchmark_future_index is None:
        return None
    benchmark_entry_close = _finite(benchmark_history[benchmark_entry_index].get("adjusted_close"))
    benchmark_future_close = _finite(benchmark_history[benchmark_future_index].get("adjusted_close"))
    if benchmark_entry_close is None or benchmark_future_close is None or benchmark_entry_close <= 0:
        return None

    symbol_return = (future_close - entry_close) / entry_close
    benchmark_return = (benchmark_future_close - benchmark_entry_close) / benchmark_entry_close
    return symbol_return - benchmark_return


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_proportion_z_test(hits_a: int, n_a: int, hits_b: int, n_b: int) -> tuple[float | None, float | None]:
    """Two-sided z-test comparing two hit-rates (e.g. one classification bucket against
    every other classified instance combined). Returns ``(z, p_value)``, or ``(None,
    None)`` when either sample is empty or the pooled rate is degenerate (0 or 1, where
    the normal approximation isn't meaningful).

    This assumes ``n_a``/``n_b`` independent Bernoulli trials — true of a coin flip, not
    true of overlapping-window, symbol-clustered daily rows (see the module docstring).
    The returned p-value is naive/descriptive here, not a validated inferential statistic;
    callers should present it as such (see ``ClassificationBucket.p_value``).
    """
    if n_a == 0 or n_b == 0:
        return None, None
    p_a, p_b = hits_a / n_a, hits_b / n_b
    pooled = (hits_a + hits_b) / (n_a + n_b)
    if pooled <= 0 or pooled >= 1:
        return None, None
    standard_error = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    if standard_error == 0:
        return None, None
    z = (p_a - p_b) / standard_error
    p_value = 2 * (1 - _normal_cdf(abs(z)))
    return z, p_value


def _symbol_mean_confidence_interval(
    per_symbol_means: Sequence[float], *, z: float = 1.96
) -> tuple[float | None, float | None]:
    """A normal-approximation 95% CI around the (equal-weighted-by-symbol) mean of
    ``per_symbol_means``, treating each symbol as one independent observation.

    Hand-rolled and deliberately simple, consistent with the rest of this module — this
    is a normal approximation, not a t-interval, so it understates uncertainty further
    when ``len(per_symbol_means)`` is very small (as low as 2-3 stocks in some buckets).
    Returns ``(None, None)`` when there are fewer than 2 symbols (no variance estimate
    is possible from a single observation).
    """
    n = len(per_symbol_means)
    if n < 2:
        return None, None
    mean = sum(per_symbol_means) / n
    variance = sum((value - mean) ** 2 for value in per_symbol_means) / (n - 1)
    if variance <= 0:
        return mean, mean
    standard_error = math.sqrt(variance / n)
    return mean - z * standard_error, mean + z * standard_error


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Linear-interpolation percentile (the same convention numpy's default uses),
    hand-rolled consistent with the rest of this module. ``sorted_values`` must already
    be sorted ascending and non-empty; ``fraction`` is in ``[0, 1]``.
    """
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    position = fraction * (n - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    weight = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight


@dataclass(slots=True)
class ClassificationBucket:
    """One classification's forward-performance summary, measured against every other
    classified instance combined (not against a flat 50%, and not against a pooled rate
    that includes this bucket's own samples — both would understate real differences).

    ``p_value``/``bonferroni_p_value`` are naive/descriptive (see the module docstring)
    — the symbol-weighted mean and its confidence interval are the more defensible
    read of whether this bucket's apparent difference is a broad pattern or a
    concentration-driven mirage.
    """

    classification: str
    sample_size: int
    hit_rate: float | None = None
    mean_excess_return: float | None = None
    median_excess_return: float | None = None
    z_score: float | None = None
    p_value: float | None = None
    # Bonferroni-corrected across the five classification buckets actually tested
    # (min(1.0, p_value * comparisons_tested)) — still naive/descriptive, just no
    # longer implying only one comparison was run.
    bonferroni_p_value: float | None = None
    distinct_symbols: int = 0
    symbol_mean_excess_return: float | None = None
    symbol_mean_excess_return_ci_low: float | None = None
    symbol_mean_excess_return_ci_high: float | None = None
    concentration_warning: bool = False
    # The realistic spread of *individual* outcomes (10th/90th percentile of the same
    # per-observation excess-return values averaged into mean_excess_return), not a CI
    # on the mean -- a CI answers "how confident are we in the average," this answers
    # "how wide do real outcomes actually get." None below MIN_OUTCOME_SAMPLE_SIZE.
    p10_excess_return: float | None = None
    p90_excess_return: float | None = None
    outcome_sample_size: int = 0


@dataclass(slots=True)
class SignalValidationResult:
    horizon_days: int
    benchmark_symbol: str
    total_samples: int
    skipped_samples: int
    pooled_hit_rate: float | None
    buckets: list[ClassificationBucket] = field(default_factory=list)
    monotonic: bool | None = None
    monotonic_symbol_weighted: bool | None = None


def validate_signals(
    signals: Sequence[Any],
    histories: Mapping[str, list[dict[str, Any]]],
    *,
    benchmark_symbol: str,
    horizon_days: int,
) -> SignalValidationResult:
    """Bucket every classified signal (regardless of the backtester's own entry/exit
    action for it — that's what keeps this independent of position-cap survivorship)
    by classification, and test whether each bucket's hit-rate genuinely differs from
    the rest of the classified population.
    """
    benchmark_history = histories.get(benchmark_symbol.upper(), [])
    benchmark_index = _date_index(benchmark_history)
    symbol_indices: dict[str, dict[str, int]] = {}

    excess_by_classification: dict[str, list[float]] = {name: [] for name in CLASSIFICATION_ORDER}
    # Tracked separately from the flat list above so each bucket's symbol diversity and
    # equal-weighted (one vote per symbol, not per day) mean can be reported alongside
    # the day-level stats — see MIN_DISTINCT_SYMBOLS_FOR_TRUST.
    excess_by_symbol: dict[str, dict[str, list[float]]] = {name: {} for name in CLASSIFICATION_ORDER}
    skipped = 0
    for signal in signals:
        classification = _get(signal, "classification")
        if classification not in excess_by_classification:
            continue
        symbol = str(_get(signal, "symbol") or "").upper()
        signal_date = str(_get(signal, "signal_date") or "")[:10]
        if not symbol or not signal_date:
            skipped += 1
            continue
        symbol_history = histories.get(symbol, [])
        if symbol not in symbol_indices:
            symbol_indices[symbol] = _date_index(symbol_history)
        excess = forward_excess_return(
            symbol_history,
            benchmark_history,
            signal_date,
            horizon_days,
            symbol_index=symbol_indices[symbol],
            benchmark_index=benchmark_index,
        )
        if excess is None:
            skipped += 1
            continue
        excess_by_classification[classification].append(excess)
        excess_by_symbol[classification].setdefault(symbol, []).append(excess)

    all_excess = [value for values in excess_by_classification.values() for value in values]
    total_samples = len(all_excess)
    total_hits = sum(1 for value in all_excess if value > 0)
    pooled_hit_rate = (total_hits / total_samples) if total_samples else None

    # Bonferroni correction denominator: how many of the five buckets actually have a
    # naive p-value computed at all (empty buckets never get one, so they shouldn't
    # inflate the correction).
    comparisons_tested = sum(1 for classification in CLASSIFICATION_ORDER if excess_by_classification[classification])

    buckets: list[ClassificationBucket] = []
    mean_by_classification: dict[str, float] = {}
    for classification in CLASSIFICATION_ORDER:
        values = excess_by_classification[classification]
        n = len(values)
        if n == 0:
            buckets.append(ClassificationBucket(classification, sample_size=0))
            continue
        hits = sum(1 for value in values if value > 0)
        mean_excess = sum(values) / n
        sorted_values = sorted(values)
        midpoint = n // 2
        median_excess = (
            sorted_values[midpoint]
            if n % 2
            else (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2
        )
        rest_hits, rest_n = total_hits - hits, total_samples - n
        z_score, p_value = two_proportion_z_test(hits, n, rest_hits, rest_n)
        bonferroni_p_value = None if p_value is None else min(1.0, p_value * comparisons_tested)
        mean_by_classification[classification] = mean_excess

        per_symbol_means = [sum(values) / len(values) for values in excess_by_symbol[classification].values()]
        distinct_symbols = len(per_symbol_means)
        symbol_mean_excess = sum(per_symbol_means) / distinct_symbols if distinct_symbols else None
        ci_low, ci_high = _symbol_mean_confidence_interval(per_symbol_means)
        p10_excess = p90_excess = None
        if n >= MIN_OUTCOME_SAMPLE_SIZE:
            p10_excess = _percentile(sorted_values, 0.10)
            p90_excess = _percentile(sorted_values, 0.90)

        buckets.append(
            ClassificationBucket(
                classification=classification,
                sample_size=n,
                hit_rate=hits / n,
                mean_excess_return=mean_excess,
                median_excess_return=median_excess,
                z_score=z_score,
                p_value=p_value,
                bonferroni_p_value=bonferroni_p_value,
                distinct_symbols=distinct_symbols,
                symbol_mean_excess_return=symbol_mean_excess,
                symbol_mean_excess_return_ci_low=ci_low,
                symbol_mean_excess_return_ci_high=ci_high,
                concentration_warning=distinct_symbols < MIN_DISTINCT_SYMBOLS_FOR_TRUST,
                p10_excess_return=p10_excess,
                p90_excess_return=p90_excess,
                outcome_sample_size=n,
            )
        )

    ranked_means = [mean_by_classification[name] for name in CLASSIFICATION_ORDER if name in mean_by_classification]
    monotonic = (
        all(current >= following for current, following in zip(ranked_means, ranked_means[1:]))
        if len(ranked_means) >= 2
        else None
    )

    # Same check, but weighting each symbol once regardless of how many days it spent in
    # a classification — the day-weighted version above is exactly what a concentrated
    # bucket (one or two symbols classified for a long overlapping-window stretch) can
    # distort, since that symbol's single outcome then gets counted hundreds of times.
    symbol_weighted_by_classification = {
        bucket.classification: bucket.symbol_mean_excess_return
        for bucket in buckets
        if bucket.symbol_mean_excess_return is not None
    }
    ranked_symbol_means = [
        symbol_weighted_by_classification[name]
        for name in CLASSIFICATION_ORDER
        if name in symbol_weighted_by_classification
    ]
    monotonic_symbol_weighted = (
        all(current >= following for current, following in zip(ranked_symbol_means, ranked_symbol_means[1:]))
        if len(ranked_symbol_means) >= 2
        else None
    )

    return SignalValidationResult(
        horizon_days=horizon_days,
        benchmark_symbol=benchmark_symbol.upper(),
        total_samples=total_samples,
        skipped_samples=skipped,
        pooled_hit_rate=pooled_hit_rate,
        buckets=buckets,
        monotonic=monotonic,
        monotonic_symbol_weighted=monotonic_symbol_weighted,
    )


# Roughly 1, 2, 3, 4 (standard), 5, and 6 weeks out. The standard evaluation horizon
# (usually 21) is folded into whatever set is actually used at call time -- see
# compute_checkpoint_comparisons -- so it's always one of the checkpoints compared,
# never a separate hardcoded number that could silently drift from config.
CHECKPOINT_SESSIONS: tuple[int, ...] = (5, 10, 15, 21, 26, 31)


@dataclass(slots=True)
class CheckpointOutcome:
    """One forward-horizon checkpoint's realized-outcome range and symbol-weighted
    mean/CI -- computed exactly like ClassificationBucket's own fields, just at a
    (possibly) different horizon than the standard one.
    """

    sessions: int
    outcome_sample_size: int = 0
    p10_excess_return: float | None = None
    p90_excess_return: float | None = None
    symbol_mean_excess_return: float | None = None
    symbol_mean_excess_return_ci_low: float | None = None
    symbol_mean_excess_return_ci_high: float | None = None


@dataclass(slots=True)
class BestExitAnalysis:
    """Whether some forward checkpoint meaningfully outperforms the standard horizon
    for one classification, and which one -- see compute_checkpoint_comparisons.
    """

    classification: str
    standard_horizon_sessions: int
    best_checkpoint_sessions: int | None = None
    best_checkpoint_meaningfully_better: bool = False
    checkpoints: list[CheckpointOutcome] = field(default_factory=list)


def compute_checkpoint_comparisons(
    signals: Sequence[Any],
    histories: Mapping[str, list[dict[str, Any]]],
    *,
    benchmark_symbol: str,
    standard_horizon_days: int,
    checkpoint_sessions: Sequence[int] = CHECKPOINT_SESSIONS,
) -> dict[str, BestExitAnalysis]:
    """For every classification, compute the realized-outcome range at each of
    ``checkpoint_sessions`` (plus ``standard_horizon_days``, folded in if not already
    among them), and determine whether any checkpoint's symbol-weighted mean is
    meaningfully better than the standard horizon's.

    "Meaningfully better" requires two things, both deliberately strict: the
    checkpoint's 95% CI must sit entirely above the standard horizon's (its CI low
    above the standard's CI high -- not just a higher point estimate, which is exactly
    the multiple-comparisons trap this function exists to avoid, see the module
    docstring), and the checkpoint's own outcome_sample_size must clear
    MIN_OUTCOME_SAMPLE_SIZE. Among checkpoints that clear both bars, the one with the
    highest symbol-weighted mean wins. If none clear the bar, that's a normal,
    expected result -- ``best_checkpoint_sessions`` stays ``None`` and the standard
    horizon is used as-is.

    Reuses the exact same per-observation gathering ``validate_signals`` does, just
    called once per checkpoint horizon instead of once, since ``forward_excess_return``
    has to be recomputed from scratch at each horizon (a longer horizon needs more
    trailing history to exist, so its usable sample can differ from a shorter one's).
    """
    benchmark_history = histories.get(benchmark_symbol.upper(), [])
    benchmark_index = _date_index(benchmark_history)
    symbol_indices: dict[str, dict[str, int]] = {}

    # Dedupe while preserving order, so the standard horizon is always included exactly
    # once regardless of whether it happens to already be one of checkpoint_sessions.
    horizons = list(dict.fromkeys([*checkpoint_sessions, standard_horizon_days]))

    checkpoints_by_classification: dict[str, list[CheckpointOutcome]] = {
        name: [] for name in CLASSIFICATION_ORDER
    }
    for horizon in horizons:
        excess_by_classification: dict[str, list[float]] = {name: [] for name in CLASSIFICATION_ORDER}
        excess_by_symbol: dict[str, dict[str, list[float]]] = {name: {} for name in CLASSIFICATION_ORDER}
        for signal in signals:
            classification = _get(signal, "classification")
            if classification not in excess_by_classification:
                continue
            symbol = str(_get(signal, "symbol") or "").upper()
            signal_date = str(_get(signal, "signal_date") or "")[:10]
            if not symbol or not signal_date:
                continue
            symbol_history = histories.get(symbol, [])
            if symbol not in symbol_indices:
                symbol_indices[symbol] = _date_index(symbol_history)
            excess = forward_excess_return(
                symbol_history, benchmark_history, signal_date, horizon,
                symbol_index=symbol_indices[symbol], benchmark_index=benchmark_index,
            )
            if excess is None:
                continue
            excess_by_classification[classification].append(excess)
            excess_by_symbol[classification].setdefault(symbol, []).append(excess)

        for classification in CLASSIFICATION_ORDER:
            values = excess_by_classification[classification]
            n = len(values)
            p10_excess = p90_excess = None
            if n >= MIN_OUTCOME_SAMPLE_SIZE:
                sorted_values = sorted(values)
                p10_excess = _percentile(sorted_values, 0.10)
                p90_excess = _percentile(sorted_values, 0.90)
            per_symbol_means = [sum(v) / len(v) for v in excess_by_symbol[classification].values()]
            symbol_mean = sum(per_symbol_means) / len(per_symbol_means) if per_symbol_means else None
            ci_low, ci_high = _symbol_mean_confidence_interval(per_symbol_means)
            checkpoints_by_classification[classification].append(
                CheckpointOutcome(
                    sessions=horizon,
                    outcome_sample_size=n,
                    p10_excess_return=p10_excess,
                    p90_excess_return=p90_excess,
                    symbol_mean_excess_return=symbol_mean,
                    symbol_mean_excess_return_ci_low=ci_low,
                    symbol_mean_excess_return_ci_high=ci_high,
                )
            )

    results: dict[str, BestExitAnalysis] = {}
    for classification in CLASSIFICATION_ORDER:
        checkpoints = checkpoints_by_classification[classification]
        standard = next((c for c in checkpoints if c.sessions == standard_horizon_days), None)
        best_sessions: int | None = None
        meaningfully_better = False
        if standard is not None and standard.symbol_mean_excess_return_ci_high is not None:
            candidates = [
                c for c in checkpoints
                if c.sessions != standard_horizon_days
                and c.outcome_sample_size >= MIN_OUTCOME_SAMPLE_SIZE
                and c.symbol_mean_excess_return_ci_low is not None
                and c.symbol_mean_excess_return_ci_low > standard.symbol_mean_excess_return_ci_high
            ]
            if candidates:
                best = max(candidates, key=lambda c: c.symbol_mean_excess_return)
                best_sessions = best.sessions
                meaningfully_better = True
        results[classification] = BestExitAnalysis(
            classification=classification,
            standard_horizon_sessions=standard_horizon_days,
            best_checkpoint_sessions=best_sessions,
            best_checkpoint_meaningfully_better=meaningfully_better,
            checkpoints=checkpoints,
        )
    return results


def render_signal_validation_text(result: SignalValidationResult) -> str:
    """Plain-text report matching this project's existing CLI-output style (flat
    printed lines with manual column alignment — there is no table-formatting
    dependency anywhere in this codebase to use instead)."""
    lines = [
        f"Signal validation vs {result.benchmark_symbol} "
        f"(forward {result.horizon_days}-session excess return)",
        f"  classified samples: {result.total_samples}  (skipped, insufficient forward history: {result.skipped_samples})",
    ]
    if result.pooled_hit_rate is not None:
        lines.append(f"  pooled hit-rate (beat benchmark): {result.pooled_hit_rate:.1%}")
    lines.append("")
    lines.append(
        f"  {'classification':<16}{'n':>7}{'symbols':>9}{'hit-rate':>10}"
        f"{'mean xs ret':>13}{'median xs ret':>15}{'naive p':>10}{'bonf. p':>10}"
    )
    any_concentration_warning = False
    for bucket in result.buckets:
        if bucket.sample_size == 0:
            lines.append(
                f"  {bucket.classification:<16}{0:>7}{'--':>9}{'--':>10}{'--':>13}{'--':>15}{'--':>10}{'--':>10}"
            )
            continue
        p_text = f"{bucket.p_value:.4f}" if bucket.p_value is not None else "--"
        bonferroni_text = f"{bucket.bonferroni_p_value:.4f}" if bucket.bonferroni_p_value is not None else "--"
        symbols_text = f"{bucket.distinct_symbols}{'*' if bucket.concentration_warning else ' '}"
        any_concentration_warning = any_concentration_warning or bucket.concentration_warning
        lines.append(
            f"  {bucket.classification:<16}{bucket.sample_size:>7}{symbols_text:>9}"
            f"{bucket.hit_rate:>10.1%}{bucket.mean_excess_return:>+13.2%}"
            f"{bucket.median_excess_return:>+15.2%}{p_text:>10}{bonferroni_text:>10}"
        )
    lines.append(
        "  naive/bonf. p-values assume independent daily trials — overlapping "
        f"{result.horizon_days}-session lookaheads and symbol clustering violate that, so treat"
    )
    lines.append(
        "  these as descriptive, not validated inferential statistics. 'bonf.' corrects the naive "
        "p-value for testing all 5 buckets, not just one."
    )
    if any_concentration_warning:
        lines.append(
            f"  * fewer than {MIN_DISTINCT_SYMBOLS_FOR_TRUST} distinct symbols back this bucket — its row-count"
        )
        lines.append(
            "    reflects one or two stocks' overlapping-window history repeated many times, not an"
        )
        lines.append("    independently-sampled population effect. Treat it as anecdote, not evidence.")
    lines.append("")
    lines.append("  symbol-weighted mean excess return (95% CI, one vote per stock regardless of day-count):")
    for bucket in result.buckets:
        if bucket.sample_size == 0 or bucket.symbol_mean_excess_return is None:
            continue
        mean_text = f"{bucket.symbol_mean_excess_return:+.2%}"
        if bucket.symbol_mean_excess_return_ci_low is None:
            ci_text = "CI unavailable (fewer than 2 distinct symbols)"
        else:
            ci_text = (
                f"95% CI [{bucket.symbol_mean_excess_return_ci_low:+.2%}, "
                f"{bucket.symbol_mean_excess_return_ci_high:+.2%}]"
            )
        lines.append(f"    {bucket.classification:<16} mean {mean_text:>8}  {ci_text}")
    lines.append("")
    if result.monotonic is None:
        lines.append("  monotonic (Strong Candidate > ... > High Risk by mean excess return): not enough non-empty buckets to check")
    else:
        lines.append(f"  monotonic (Strong Candidate > ... > High Risk by mean excess return): {'yes' if result.monotonic else 'NO'}")
    if result.monotonic_symbol_weighted is None:
        lines.append("  monotonic (symbol-weighted, one vote per stock): not enough non-empty buckets to check")
    else:
        lines.append(
            "  monotonic (symbol-weighted, one vote per stock): "
            f"{'yes' if result.monotonic_symbol_weighted else 'NO'}"
        )
        lines.append(
            "  a symbol-weighted inversion is not fully explained by day-count concentration alone — "
            "it remains a descriptive finding in this dataset, not a statistically validated one given "
            "the overlapping/clustered observations above."
        )
    return "\n".join(lines)


def load_latest_signal_validation(reports_dir: Path) -> dict[str, Any] | None:
    """Best-effort load of the most recent ``signal_validation_*.json`` artifact
    (written by ``main.py validate-signals``) from ``reports_dir``.

    ``validate-signals`` requires a fresh full-history backtest and is a manual,
    occasional diagnostic (see README's "Evaluation honesty" section) — it is not
    re-run as part of every daily ``recommend``/report. This surfaces its latest known
    result as a dated annotation rather than recomputing it. Returns ``None`` (silently
    — this is supplementary context, never required for a report or a recommendation
    run) if no such artifact exists yet or it can't be parsed.
    """
    try:
        candidates = sorted(
            Path(reports_dir).glob("signal_validation_*.json"),
            key=lambda path: path.stat().st_mtime,
        )
    except OSError:
        return None
    if not candidates:
        return None
    try:
        return json.loads(candidates[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
