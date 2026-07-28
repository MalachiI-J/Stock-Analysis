"""Does score_v1's classification carry real information, independent of the backtester?

The portfolio backtester answers "would this specific sizing/exit/cost configuration
have made money" — a question tangled up with position caps, entry timing, and trade
mechanics. It also cannot be run over more than one independent validation window given
how little history this project has (~5 years), so a single portfolio-level verdict
isn't enough evidence either way.

This module asks a narrower, higher-power question instead: across every day a symbol
was ever classified, regardless of whether the portfolio had room to act on it, did that
classification's forward performance actually differ from the rest? Every classified
instance is a data point (thousands of them, versus a couple of backtest windows), so
this is the more statistically powerful check of whether the label itself means anything.

Deliberately hand-rolled (``math.erf`` for the normal CDF) rather than adding scipy —
consistent with the rest of this project's no-scipy/no-scikit-learn design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

# Ordered from most to least favorable — used only to report mean excess return in a
# fixed, readable order and to check monotonicity; "Insufficient Data"/"Data Blocked"
# carry no opportunity signal and are excluded entirely.
CLASSIFICATION_ORDER: tuple[str, ...] = ("Strong Candidate", "Candidate", "Watch", "Avoid", "High Risk")


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
    """Two-sided z-test comparing two independent hit-rates (e.g. one classification
    bucket against every other classified instance combined). Returns ``(z, p_value)``,
    or ``(None, None)`` when either sample is empty or the pooled rate is degenerate
    (0 or 1, where the normal approximation isn't meaningful).
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


@dataclass(slots=True)
class ClassificationBucket:
    """One classification's forward-performance summary, measured against every other
    classified instance combined (not against a flat 50%, and not against a pooled rate
    that includes this bucket's own samples — both would understate real differences)."""

    classification: str
    sample_size: int
    hit_rate: float | None = None
    mean_excess_return: float | None = None
    median_excess_return: float | None = None
    z_score: float | None = None
    p_value: float | None = None


@dataclass(slots=True)
class SignalValidationResult:
    horizon_days: int
    benchmark_symbol: str
    total_samples: int
    skipped_samples: int
    pooled_hit_rate: float | None
    buckets: list[ClassificationBucket] = field(default_factory=list)
    monotonic: bool | None = None


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

    all_excess = [value for values in excess_by_classification.values() for value in values]
    total_samples = len(all_excess)
    total_hits = sum(1 for value in all_excess if value > 0)
    pooled_hit_rate = (total_hits / total_samples) if total_samples else None

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
        mean_by_classification[classification] = mean_excess
        buckets.append(
            ClassificationBucket(
                classification=classification,
                sample_size=n,
                hit_rate=hits / n,
                mean_excess_return=mean_excess,
                median_excess_return=median_excess,
                z_score=z_score,
                p_value=p_value,
            )
        )

    ranked_means = [mean_by_classification[name] for name in CLASSIFICATION_ORDER if name in mean_by_classification]
    monotonic = (
        all(current >= following for current, following in zip(ranked_means, ranked_means[1:]))
        if len(ranked_means) >= 2
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
    )


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
    lines.append(f"  {'classification':<16}{'n':>7}{'hit-rate':>10}{'mean xs ret':>13}{'median xs ret':>15}{'p-value':>10}")
    for bucket in result.buckets:
        if bucket.sample_size == 0:
            lines.append(f"  {bucket.classification:<16}{0:>7}{'--':>10}{'--':>13}{'--':>15}{'--':>10}")
            continue
        p_text = f"{bucket.p_value:.4f}" if bucket.p_value is not None else "--"
        lines.append(
            f"  {bucket.classification:<16}{bucket.sample_size:>7}"
            f"{bucket.hit_rate:>10.1%}{bucket.mean_excess_return:>+13.2%}"
            f"{bucket.median_excess_return:>+15.2%}{p_text:>10}"
        )
    lines.append("")
    if result.monotonic is None:
        lines.append("  monotonic (Strong Candidate > ... > High Risk by mean excess return): not enough non-empty buckets to check")
    else:
        lines.append(f"  monotonic (Strong Candidate > ... > High Risk by mean excess return): {'yes' if result.monotonic else 'NO'}")
    return "\n".join(lines)
