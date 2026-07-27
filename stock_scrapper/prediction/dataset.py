"""Build a leakage-safe training dataset from already-computed technical indicators.

Every row pairs one symbol's indicators on a historical as-of date with
whether its adjusted close was higher ``horizon_days`` trading sessions later.
Both are read from ``histories`` bounded no later than the caller's true
as-of date, so a training row's label is only ever computed from data the
caller has already loaded — never fetched separately or extended past that
bound. :func:`select_sample_dates` additionally drops the most recent
``horizon_days`` sessions from the candidate sample-date pool so every
remaining sample date still has enough later sessions, within that same
bounded history, to know its own label.
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


def build_training_dataset(
    service: AnalysisService,
    symbols: Sequence[str],
    histories: Mapping[str, list[dict[str, Any]]],
    sample_dates: Sequence[date],
    *,
    horizon_days: int,
    feature_keys: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    """Assemble (features, positive-forward-return label) rows for every eligible symbol/date."""
    service.prime_historical_features(histories, sample_dates, list(symbols))
    date_index: dict[str, dict[str, int]] = {
        symbol.upper(): {
            str(row["trade_date"])[:10]: index
            for index, row in enumerate(rows)
            if row.get("trade_date")
        }
        for symbol, rows in histories.items()
    }

    feature_rows: list[list[float]] = []
    labels: list[float] = []
    meta: list[dict[str, str]] = []
    for sample_date in sample_dates:
        batch = service.analyze_loaded_many_as_of(list(symbols), histories, sample_date, persist=False)
        for result in batch.results:
            if not result.eligible_for_scoring:
                continue
            symbol_history = histories.get(result.symbol, [])
            index_map = date_index.get(result.symbol, {})
            index = index_map.get(sample_date.isoformat())
            if index is None or index + horizon_days >= len(symbol_history):
                continue
            entry_close = _finite(symbol_history[index].get("adjusted_close"))
            future_close = _finite(symbol_history[index + horizon_days].get("adjusted_close"))
            if entry_close is None or future_close is None or entry_close <= 0:
                continue
            values = [_finite(result.indicators.get(key)) for key in feature_keys]
            if any(value is None for value in values):
                continue
            feature_rows.append(values)  # type: ignore[arg-type]
            labels.append(1.0 if future_close > entry_close else 0.0)
            meta.append({"symbol": result.symbol, "date": sample_date.isoformat()})

    features = np.array(feature_rows, dtype=float) if feature_rows else np.empty((0, len(feature_keys)))
    return features, np.array(labels, dtype=float), meta
