"""Build a leakage-safe training dataset from already-computed technical indicators.

Every row pairs one symbol's indicators on a historical as-of date with whether
it beat the benchmark's own return over the next ``horizon_days`` trading
sessions — not just whether its own price rose. Raw direction is dominated by
broad market drift (during a sustained Risk-On stretch, most stocks rise most
of the time regardless of anything stock-specific), so a model trained on raw
direction mostly just relearns "is the market going up," which it has no way
to predict better than the base rate. Excess return over the benchmark isolates
the stock-picking signal the rest of this project is actually about.

Both the symbol's and the benchmark's prices are read from ``histories``
bounded no later than the caller's true as-of date, so a training row's label
is only ever computed from data the caller has already loaded — never fetched
separately or extended past that bound. :func:`select_sample_dates`
additionally drops the most recent ``horizon_days`` sessions from the
candidate sample-date pool so every remaining sample date still has enough
later sessions, within that same bounded history, to know its own label.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any, Mapping, Sequence

import numpy as np

from stock_scrapper.analysis.service import AnalysisService


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# Ordinal encoding of the market regime, from most to least favorable. Mirrors the
# direction (though not the exact scale) of the regime's own risk-score mapping in
# stock_scrapper/analysis/risk_score.py, so higher always means "more bullish."
_REGIME_CODES: dict[str, float] = {
    "Risk-On": 1.0,
    "Neutral": 0.0,
    "Risk-Off": -1.0,
    "Stress": -2.0,
}

# Feature keys computed here rather than read from result.indicators, because they
# depend on either a top-level AnalysisResult field (market_regime) or on the whole
# day's batch of results (a cross-sectional rank), not on one symbol in isolation.
DERIVED_FEATURE_KEYS = frozenset({"market_regime_code", "opportunity_score_percentile"})


def opportunity_percentiles(results: Sequence[Any]) -> dict[str, float]:
    """Rank each result's opportunity_score against the rest of that same day's batch.

    Returns a 0..1 percentile per symbol (1.0 = highest opportunity score that day).
    Ties and single-symbol batches fall back to the midpoint (0.5) rather than an
    arbitrary ordering, since ambiguous ranks aren't meaningful signal.
    """
    scored = [
        (result.symbol, score)
        for result in results
        for score in (_finite(getattr(result, "opportunity_score", None)),)
        if score is not None
    ]
    if len(scored) < 2:
        return {symbol: 0.5 for symbol, _ in scored}
    ordered = sorted(scored, key=lambda item: (item[1], item[0]))
    denominator = len(ordered) - 1
    return {symbol: index / denominator for index, (symbol, _) in enumerate(ordered)}


def feature_value(result: Any, key: str, percentiles: Mapping[str, float]) -> float | None:
    """Resolve one feature's value for a result, whether indicator-based or derived."""
    if key == "market_regime_code":
        return _REGIME_CODES.get(getattr(result, "market_regime", None))
    if key == "opportunity_score_percentile":
        return percentiles.get(result.symbol)
    return _finite(result.indicators.get(key))


def select_sample_dates(
    trading_dates: Sequence[str],
    *,
    as_of_date: str,
    horizon_days: int,
    lookback_years: float,
    stride_sessions: int,
) -> list[date]:
    """Pick embargoed, strided historical sample dates from a sorted trading-date list.

    ``trading_dates`` must already be bounded at or before ``as_of_date`` by
    the caller (e.g. via ``fetch_price_history(..., end_date=as_of_date)``).
    """
    earliest = (date.fromisoformat(as_of_date) - timedelta(days=int(lookback_years * 365.25))).isoformat()
    eligible = sorted({str(value)[:10] for value in trading_dates if earliest <= str(value)[:10] <= as_of_date})
    if len(eligible) <= horizon_days:
        return []
    usable = eligible[: len(eligible) - horizon_days]
    return [date.fromisoformat(value) for value in usable[::stride_sessions]]


def _assemble_dataset(
    service: AnalysisService,
    symbols: Sequence[str],
    histories: Mapping[str, list[dict[str, Any]]],
    sample_dates: Sequence[date],
    *,
    horizon_days: int,
    feature_keys: Sequence[str],
    benchmark_symbol: str,
    target_fn: Any,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    """Shared row-assembly for both the binary (``build_training_dataset``) and
    continuous (``build_regression_dataset``) targets — identical eligibility,
    feature, and leakage-safety logic; only how the label/target number itself is
    computed from ``(symbol_return, benchmark_return)`` differs, via ``target_fn``.
    """
    service.prime_historical_features(histories, sample_dates, list(symbols))
    date_index: dict[str, dict[str, int]] = {
        symbol.upper(): {
            str(row["trade_date"])[:10]: index
            for index, row in enumerate(rows)
            if row.get("trade_date")
        }
        for symbol, rows in histories.items()
    }
    benchmark_history = histories.get(benchmark_symbol, [])
    benchmark_date_index = date_index.get(benchmark_symbol.upper(), {})

    feature_rows: list[list[float]] = []
    targets: list[float] = []
    meta: list[dict[str, str]] = []
    for sample_date in sample_dates:
        batch = service.analyze_loaded_many_as_of(list(symbols), histories, sample_date, persist=False)
        percentiles = opportunity_percentiles(batch.results)
        for result in batch.results:
            if not result.eligible_for_scoring:
                continue
            symbol_history = histories.get(result.symbol, [])
            index_map = date_index.get(result.symbol, {})
            index = index_map.get(sample_date.isoformat())
            if index is None or index + horizon_days >= len(symbol_history):
                continue
            entry_close = _finite(symbol_history[index].get("adjusted_close"))
            future_row = symbol_history[index + horizon_days]
            future_close = _finite(future_row.get("adjusted_close"))
            if entry_close is None or future_close is None or entry_close <= 0:
                continue

            benchmark_entry_index = benchmark_date_index.get(sample_date.isoformat())
            benchmark_future_index = benchmark_date_index.get(str(future_row.get("trade_date"))[:10])
            if benchmark_entry_index is None or benchmark_future_index is None:
                continue
            benchmark_entry_close = _finite(benchmark_history[benchmark_entry_index].get("adjusted_close"))
            benchmark_future_close = _finite(benchmark_history[benchmark_future_index].get("adjusted_close"))
            if benchmark_entry_close is None or benchmark_future_close is None or benchmark_entry_close <= 0:
                continue

            values = [feature_value(result, key, percentiles) for key in feature_keys]
            if any(value is None for value in values):
                continue
            symbol_return = (future_close - entry_close) / entry_close
            benchmark_return = (benchmark_future_close - benchmark_entry_close) / benchmark_entry_close
            feature_rows.append(values)  # type: ignore[arg-type]
            targets.append(target_fn(symbol_return, benchmark_return))
            meta.append(
                {
                    "symbol": result.symbol,
                    "date": sample_date.isoformat(),
                    # Where this row's label actually resolves — a training row whose label
                    # depends on prices at or after a later fold's test period start is still
                    # informationally entangled with that test period, even though the row's
                    # *features* only use data up to "date". See _run_walk_forward's purge step.
                    "label_end_date": str(future_row.get("trade_date"))[:10],
                }
            )

    features = np.array(feature_rows, dtype=float) if feature_rows else np.empty((0, len(feature_keys)))
    return features, np.array(targets, dtype=float), meta


def build_training_dataset(
    service: AnalysisService,
    symbols: Sequence[str],
    histories: Mapping[str, list[dict[str, Any]]],
    sample_dates: Sequence[date],
    *,
    horizon_days: int,
    feature_keys: Sequence[str],
    benchmark_symbol: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    """Assemble (features, beat-the-benchmark label) rows for every eligible symbol/date."""
    return _assemble_dataset(
        service, symbols, histories, sample_dates,
        horizon_days=horizon_days, feature_keys=feature_keys, benchmark_symbol=benchmark_symbol,
        target_fn=lambda symbol_return, benchmark_return: 1.0 if symbol_return > benchmark_return else 0.0,
    )


def build_regression_dataset(
    service: AnalysisService,
    symbols: Sequence[str],
    histories: Mapping[str, list[dict[str, Any]]],
    sample_dates: Sequence[date],
    *,
    horizon_days: int,
    feature_keys: Sequence[str],
    benchmark_symbol: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    """Assemble (features, continuous excess-return) rows for every eligible
    symbol/date — predict-v4's target, versus predict-v3's binarized version of the
    same underlying quantity. Keeping the magnitude (rather than just its sign)
    preserves information predict-v3 throws away, which matters for the
    non-monotonic, magnitude-driven patterns ``investigate-risk-inversion`` found
    (see ``stock_scrapper/analysis/risk_diagnostics.py``)."""
    return _assemble_dataset(
        service, symbols, histories, sample_dates,
        horizon_days=horizon_days, feature_keys=feature_keys, benchmark_symbol=benchmark_symbol,
        target_fn=lambda symbol_return, benchmark_return: symbol_return - benchmark_return,
    )
