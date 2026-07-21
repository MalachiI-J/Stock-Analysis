"""Causal date-indexed feature snapshots for historical simulations.

The live analyzer remains the reference implementation.  This module prepares
the expensive rolling series once, then materializes only requested as-of
dates with the same tail-window operations used by the reference functions.
Every lookup is a right-bounded prefix lookup; future rows are never exposed to
a snapshot.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import date
from math import sqrt
from typing import Any, Mapping, Sequence

import pandas as pd

from stock_scrapper.processing.indicators import (
    TRADING_DAYS_PER_YEAR,
    _empty_result,
    _safe_float,
    _wilder_average,
    classify_status,
)


_NUMERIC_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "dividends",
    "stock_splits",
)
_REQUIRED_OHLCV = ("open", "high", "low", "close", "adjusted_close", "volume")
_AVAILABILITY_FIELDS = (
    "one_month_return",
    "three_month_return",
    "six_month_return",
    "one_year_return",
    "fifty_day_sma",
    "two_hundred_day_sma",
    "twenty_day_volatility",
    "sixty_day_volatility",
    "two_hundred_fifty_two_day_volatility",
    "sixty_day_downside_volatility",
    "rsi_14",
    "atr_14",
    "trend_slope_50",
    "trend_slope_200",
    "time_above_sma50",
    "time_above_sma200",
    "twenty_day_average_dollar_volume",
    "distance_from_52_week_high",
)


def _iso_date(value: str | date) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return date.fromisoformat(str(value)[:10]).isoformat()


def _normalize_history(history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Match the canonical analyzer's sorting, deduplication, and field shape."""
    by_date: dict[str, dict[str, Any]] = {}
    for source in history:
        raw_date = source.get("trade_date")
        try:
            parsed = date.fromisoformat(str(raw_date)[:10])
        except (TypeError, ValueError):
            continue
        row = dict(source)
        row["trade_date"] = parsed.isoformat()
        for field in _NUMERIC_COLUMNS:
            row.setdefault(field, None)
        by_date[parsed.isoformat()] = row
    return [by_date[key] for key in sorted(by_date)]


def _empty_relative_metrics() -> dict[str, Any]:
    return {
        "benchmark_relative_return_21": None,
        "benchmark_relative_return_63": None,
        "benchmark_relative_return_126": None,
        "benchmark_relative_return_252": None,
        "beta_252": None,
        "beta": None,
        "correlation_252": None,
        "correlation": None,
        "relative_strength_trend": None,
        "benchmark_aligned_days": 0,
        "benchmark_alignment_ratio": 0.0,
        "benchmark_available": False,
        "benchmark_data_through_date": None,
    }


@dataclass(frozen=True)
class HistoricalFeatureSnapshot:
    """Canonical indicator inputs for one symbol and one as-of date."""

    base_metrics: dict[str, Any]
    relative_metrics: dict[str, Any]


class _PreparedIndicators:
    def __init__(self, rows: list[dict[str, Any]], symbol: str) -> None:
        self.rows = rows
        self.symbol = symbol
        self.dates = [str(row["trade_date"]) for row in rows]
        frame = pd.DataFrame(rows).copy()
        if frame.empty:
            self.frame = frame
            return
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
        for column in _NUMERIC_COLUMNS:
            if column not in frame.columns:
                frame[column] = float("nan")
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        self.frame = frame
        self.price = frame["adjusted_close"].astype("float64")
        self.raw_close = frame["close"].astype("float64")
        self.volume = frame["volume"].astype("float64")
        adjustment_factor = (
            self.price / self.raw_close.where(self.raw_close != 0)
        ).astype("float64")
        self.adjusted_open = frame["open"] * adjustment_factor
        self.adjusted_high = frame["high"] * adjustment_factor
        self.adjusted_low = frame["low"] * adjustment_factor
        self.previous_adjusted_close = self.price.shift(1)
        self.returns = self.price.pct_change(fill_method=None)
        self.sma20 = self.price.rolling(20, min_periods=20).mean()
        self.sma50 = self.price.rolling(50, min_periods=50).mean()
        self.sma100 = self.price.rolling(100, min_periods=100).mean()
        self.sma200 = self.price.rolling(200, min_periods=200).mean()

        price_changes = self.price.diff()
        self.average_gain = _wilder_average(price_changes.clip(lower=0.0), 14)
        self.average_loss = _wilder_average(-price_changes.clip(upper=0.0), 14)
        true_range = pd.Series(float("nan"), index=frame.index, dtype="float64")
        for index in frame.index:
            high_value = _safe_float(self.adjusted_high.at[index])
            low_value = _safe_float(self.adjusted_low.at[index])
            if high_value is None or low_value is None:
                continue
            if index == frame.index[0]:
                true_range.at[index] = high_value - low_value
                continue
            previous_close = _safe_float(self.previous_adjusted_close.at[index])
            if previous_close is None:
                continue
            true_range.at[index] = max(
                high_value - low_value,
                abs(high_value - previous_close),
                abs(low_value - previous_close),
            )
        self.atr = _wilder_average(true_range, 14)
        daily_dollar_volume = self.price * self.volume
        self.average_volume = self.volume.rolling(20, min_periods=20).mean()
        self.average_dollar_volume = daily_dollar_volume.rolling(
            20, min_periods=20
        ).mean()
        self.median_dollar_volume = daily_dollar_volume.rolling(
            20, min_periods=20
        ).median()

    @staticmethod
    def _period_return(price: pd.Series, index: int, window: int) -> float | None:
        segment = price.iloc[: index + 1].tail(window + 1)
        if (
            len(segment) != window + 1
            or segment.isna().any()
            or segment.iloc[0] == 0
        ):
            return None
        return float(segment.iloc[-1] / segment.iloc[0] - 1.0)

    def _annualized_volatility(self, index: int, window: int) -> float | None:
        segment = self.returns.iloc[: index + 1].tail(window)
        if len(segment) != window or segment.isna().any() or len(segment) < 2:
            return None
        return float(segment.std(ddof=1) * sqrt(TRADING_DAYS_PER_YEAR))

    def _downside_volatility(self, index: int, window: int) -> float | None:
        segment = self.returns.iloc[: index + 1].tail(window)
        if len(segment) != window or segment.isna().any():
            return None
        downside = segment.clip(upper=0.0)
        return float(
            sqrt(float((downside**2).mean())) * sqrt(TRADING_DAYS_PER_YEAR)
        )

    @staticmethod
    def _max_drawdown(
        values: pd.Series, *, require_complete: bool = False
    ) -> float | None:
        if require_complete and values.isna().any():
            return None
        valid = values.dropna()
        if len(valid) < 2:
            return None
        running_high = valid.cummax()
        return float((1.0 - valid / running_high).max())

    def _moving_average_slope(
        self, series: pd.Series, index: int, slope_period: int
    ) -> float | None:
        if index + 1 <= slope_period:
            return None
        current = _safe_float(series.iloc[index])
        earlier = _safe_float(series.iloc[index - slope_period])
        if current is None or earlier in (None, 0.0):
            return None
        return float(current / earlier - 1.0)

    def _time_above_sma(
        self, series: pd.Series, index: int, observation_window: int
    ) -> float | None:
        price = self.price.iloc[: index + 1]
        moving_average = series.iloc[: index + 1]
        valid_mask = price.notna() & moving_average.notna()
        comparisons = (price[valid_mask] > moving_average[valid_mask]).tail(
            observation_window
        )
        if len(comparisons) != observation_window:
            return None
        return float(comparisons.mean())

    def snapshot(self, as_of: str) -> dict[str, Any]:
        index = bisect_right(self.dates, as_of) - 1
        if index < 0 or self.frame.empty:
            return _empty_result(self.symbol)

        price = self.price.iloc[: index + 1]
        volume = self.volume.iloc[: index + 1]
        frame = self.frame.iloc[: index + 1]
        latest_row = frame.iloc[-1]
        latest_price = _safe_float(price.iloc[-1])
        latest_volume = _safe_float(volume.iloc[-1])
        average_volume = _safe_float(self.average_volume.iloc[index])
        average_dollar_volume = _safe_float(self.average_dollar_volume.iloc[index])
        median_dollar_volume = _safe_float(self.median_dollar_volume.iloc[index])
        sma20 = _safe_float(self.sma20.iloc[index])
        sma50 = _safe_float(self.sma50.iloc[index])
        sma100 = _safe_float(self.sma100.iloc[index])
        sma200 = _safe_float(self.sma200.iloc[index])

        latest_gain = _safe_float(self.average_gain.iloc[index])
        latest_loss = _safe_float(self.average_loss.iloc[index])
        rsi_14: float | None = None
        if latest_gain is not None and latest_loss is not None:
            if latest_loss == 0.0:
                rsi_14 = 50.0 if latest_gain == 0.0 else 100.0
            else:
                relative_strength = latest_gain / latest_loss
                rsi_14 = float(100.0 - 100.0 / (1.0 + relative_strength))
        atr_14 = _safe_float(self.atr.iloc[index])

        known_volume = volume.dropna()
        recent_volume = volume.tail(20)
        year_prices = price.tail(252)
        recent_returns = self.returns.iloc[: index + 1].tail(252)
        gaps = (
            self.adjusted_open.iloc[: index + 1]
            / self.previous_adjusted_close.iloc[: index + 1]
            - 1.0
        ).dropna()
        full_drawdown = self._max_drawdown(price)
        fifty_two_week_high: float | None = None
        distance_from_high: float | None = None
        if len(year_prices) == 252 and not year_prices.isna().any():
            fifty_two_week_high = float(year_prices.max())
            if fifty_two_week_high != 0 and latest_price is not None:
                distance_from_high = float(
                    latest_price / fifty_two_week_high - 1.0
                )

        result = _empty_result(self.symbol)
        result.update(
            {
                "history_length": int(index + 1),
                "valid_adjusted_close_count": int(price.notna().sum()),
                "ohlcv_completeness": float(
                    frame[list(_REQUIRED_OHLCV)].notna().to_numpy().mean()
                ),
                "latest_close": latest_price,
                "latest_adjusted_close": latest_price,
                "latest_raw_close": _safe_float(latest_row["close"]),
                "latest_open": _safe_float(latest_row["open"]),
                "latest_high": _safe_float(latest_row["high"]),
                "latest_low": _safe_float(latest_row["low"]),
                "latest_volume": latest_volume,
                "latest_trading_date": latest_row["trade_date"].strftime(
                    "%Y-%m-%d"
                ),
                "one_day_return": _safe_float(self.returns.iloc[index]),
                "five_day_return": self._period_return(price, index, 5),
                "one_month_return": self._period_return(price, index, 21),
                "three_month_return": self._period_return(price, index, 63),
                "six_month_return": self._period_return(price, index, 126),
                "one_year_return": self._period_return(price, index, 252),
                "twenty_day_sma": sma20,
                "fifty_day_sma": sma50,
                "hundred_day_sma": sma100,
                "two_hundred_day_sma": sma200,
                "distance_from_sma50": (
                    None
                    if latest_price is None or sma50 in (None, 0.0)
                    else float(latest_price / sma50 - 1.0)
                ),
                "distance_from_sma200": (
                    None
                    if latest_price is None or sma200 in (None, 0.0)
                    else float(latest_price / sma200 - 1.0)
                ),
                "twenty_day_average_volume": average_volume,
                "volume_relative_to_average": (
                    None
                    if latest_volume is None or average_volume in (None, 0.0)
                    else float(latest_volume / average_volume)
                ),
                "twenty_day_average_dollar_volume": average_dollar_volume,
                "twenty_day_median_dollar_volume": median_dollar_volume,
                "average_dollar_volume": average_dollar_volume,
                "median_dollar_volume": median_dollar_volume,
                "zero_volume_days": (
                    int((known_volume == 0).sum()) if not known_volume.empty else None
                ),
                "twenty_day_zero_volume_days": (
                    int((recent_volume == 0).sum())
                    if len(recent_volume) == 20 and not recent_volume.isna().any()
                    else None
                ),
                "twenty_day_volatility": self._annualized_volatility(index, 20),
                "sixty_day_volatility": self._annualized_volatility(index, 60),
                "two_hundred_fifty_two_day_volatility": self._annualized_volatility(
                    index, 252
                ),
                "sixty_day_downside_volatility": self._downside_volatility(
                    index, 60
                ),
                "fifty_two_week_high": fifty_two_week_high,
                "distance_from_52_week_high": distance_from_high,
                "max_drawdown": full_drawdown,
                "one_year_max_drawdown": (
                    self._max_drawdown(year_prices, require_complete=True)
                    if len(year_prices) == 252
                    else None
                ),
                "full_history_max_drawdown": full_drawdown,
                "worst_one_day_return_last_year": (
                    float(recent_returns.min())
                    if len(recent_returns) == 252
                    and not recent_returns.isna().any()
                    else None
                ),
                "overnight_gap_volatility": (
                    float(gaps.std(ddof=1) * sqrt(TRADING_DAYS_PER_YEAR))
                    if len(gaps) >= 2
                    else None
                ),
                "rsi_14": rsi_14,
                "atr_14": atr_14,
                "atr_percentage": (
                    None
                    if atr_14 is None or latest_price in (None, 0.0)
                    else float(atr_14 / latest_price)
                ),
                "trend_slope_50": self._moving_average_slope(
                    self.sma50, index, 20
                ),
                "trend_slope_200": self._moving_average_slope(
                    self.sma200, index, 20
                ),
                "time_above_sma50": self._time_above_sma(
                    self.sma50, index, 60
                ),
                "time_above_sma200": self._time_above_sma(
                    self.sma200, index, 252
                ),
            }
        )
        result["indicator_availability_ratio"] = sum(
            result.get(field) is not None for field in _AVAILABILITY_FIELDS
        ) / len(_AVAILABILITY_FIELDS)
        status, flags = classify_status(result)
        result["status"] = status
        result["flags"] = flags
        return result


class HistoricalFeatureCache:
    """Precompute causal technical and benchmark features for selected dates."""

    def __init__(
        self,
        histories: Mapping[str, Sequence[Mapping[str, Any]]],
        benchmark_symbol: str,
        snapshot_dates: Sequence[str | date],
        feature_symbols: Sequence[str] | None = None,
    ) -> None:
        self.benchmark_symbol = benchmark_symbol.upper()
        self.snapshot_dates = tuple(sorted({_iso_date(value) for value in snapshot_dates}))
        self.histories = {
            symbol.upper(): _normalize_history(rows)
            for symbol, rows in histories.items()
        }
        self._history_dates = {
            symbol: [str(row["trade_date"]) for row in rows]
            for symbol, rows in self.histories.items()
        }
        self._snapshots: dict[tuple[str, str], HistoricalFeatureSnapshot] = {}
        benchmark_rows = self.histories.get(self.benchmark_symbol, [])
        selected = (
            set(self.histories)
            if feature_symbols is None
            else {symbol.upper() for symbol in feature_symbols}
        )
        for symbol in sorted(selected):
            rows = self.histories.get(symbol, [])
            prepared = _PreparedIndicators(rows, symbol)
            relative = self._relative_snapshots(rows, benchmark_rows)
            for snapshot_date in self.snapshot_dates:
                self._snapshots[(symbol, snapshot_date)] = HistoricalFeatureSnapshot(
                    base_metrics=prepared.snapshot(snapshot_date),
                    relative_metrics=relative[snapshot_date],
                )

    def history_as_of(self, symbol: str, as_of_date: str | date) -> list[dict[str, Any]]:
        """Return a copied, right-bounded normalized prefix for eligibility/context."""
        normalized_symbol = symbol.upper()
        cutoff = _iso_date(as_of_date)
        dates = self._history_dates.get(normalized_symbol, [])
        end = bisect_right(dates, cutoff)
        return [dict(row) for row in self.histories.get(normalized_symbol, [])[:end]]

    def get(
        self, symbol: str, as_of_date: str | date
    ) -> HistoricalFeatureSnapshot | None:
        return self._snapshots.get((symbol.upper(), _iso_date(as_of_date)))

    def _relative_snapshots(
        self,
        stock_history: list[dict[str, Any]],
        benchmark_history: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        stock_by_date = {str(row["trade_date"]): row for row in stock_history}
        benchmark_by_date = {
            str(row["trade_date"]): row for row in benchmark_history
        }
        common_dates = sorted(set(stock_by_date) & set(benchmark_by_date))
        aligned = pd.DataFrame(
            {
                "trade_date": common_dates,
                "stock_price": pd.to_numeric(
                    pd.Series(
                        [stock_by_date[value].get("adjusted_close") for value in common_dates]
                    ),
                    errors="coerce",
                ),
                "benchmark_price": pd.to_numeric(
                    pd.Series(
                        [
                            benchmark_by_date[value].get("adjusted_close")
                            for value in common_dates
                        ]
                    ),
                    errors="coerce",
                ),
            }
        )
        if not aligned.empty:
            aligned = aligned.dropna(subset=["stock_price", "benchmark_price"])
            aligned = aligned[
                (aligned["stock_price"] > 0) & (aligned["benchmark_price"] > 0)
            ].reset_index(drop=True)
        aligned_dates = aligned.get("trade_date", pd.Series(dtype="object")).tolist()
        stock_valid_dates = [
            str(row["trade_date"])
            for row in stock_history
            if pd.notna(pd.to_numeric(row.get("adjusted_close"), errors="coerce"))
        ]
        stock_returns = (
            aligned["stock_price"].pct_change(fill_method=None)
            if not aligned.empty
            else pd.Series(dtype="float64")
        )
        benchmark_returns = (
            aligned["benchmark_price"].pct_change(fill_method=None)
            if not aligned.empty
            else pd.Series(dtype="float64")
        )
        paired = pd.DataFrame(
            {"stock": stock_returns, "benchmark": benchmark_returns}
        )
        snapshots: dict[str, dict[str, Any]] = {}
        for snapshot_date in self.snapshot_dates:
            result = _empty_relative_metrics()
            aligned_end = bisect_right(aligned_dates, snapshot_date)
            valid_stock_days = bisect_right(stock_valid_dates, snapshot_date)
            result["benchmark_aligned_days"] = aligned_end
            result["benchmark_alignment_ratio"] = (
                float(aligned_end / valid_stock_days) if valid_stock_days else 0.0
            )
            result["benchmark_available"] = aligned_end >= 2
            if aligned_end:
                result["benchmark_data_through_date"] = str(
                    aligned.iloc[aligned_end - 1]["trade_date"]
                )
            if aligned_end < 2:
                snapshots[snapshot_date] = result
                continue

            def excess_return(window: int) -> float | None:
                if aligned_end < window + 1:
                    return None
                first = aligned_end - window - 1
                last = aligned_end - 1
                stock_return = float(
                    aligned.iloc[last]["stock_price"]
                    / aligned.iloc[first]["stock_price"]
                    - 1.0
                )
                benchmark_return = float(
                    aligned.iloc[last]["benchmark_price"]
                    / aligned.iloc[first]["benchmark_price"]
                    - 1.0
                )
                return stock_return - benchmark_return

            for window in (21, 63, 126, 252):
                result[f"benchmark_relative_return_{window}"] = excess_return(window)
            if aligned_end >= 64:
                current_ratio = float(
                    aligned.iloc[aligned_end - 1]["stock_price"]
                    / aligned.iloc[aligned_end - 1]["benchmark_price"]
                )
                old_ratio = float(
                    aligned.iloc[aligned_end - 64]["stock_price"]
                    / aligned.iloc[aligned_end - 64]["benchmark_price"]
                )
                if old_ratio != 0:
                    result["relative_strength_trend"] = float(
                        current_ratio / old_ratio - 1.0
                    )

            paired_prefix = paired.iloc[:aligned_end].dropna()
            if len(paired_prefix) >= 252:
                recent = paired_prefix.tail(252)
                benchmark_variance = float(recent["benchmark"].var(ddof=1))
                if benchmark_variance > 0:
                    beta = float(
                        recent["stock"].cov(recent["benchmark"])
                        / benchmark_variance
                    )
                    result["beta_252"] = beta
                    result["beta"] = beta
                stock_std = float(recent["stock"].std(ddof=1))
                benchmark_std = float(recent["benchmark"].std(ddof=1))
                if stock_std > 0 and benchmark_std > 0:
                    correlation = float(
                        recent["stock"].corr(recent["benchmark"])
                    )
                    result["correlation_252"] = correlation
                    result["correlation"] = correlation
            snapshots[snapshot_date] = result
        return snapshots
