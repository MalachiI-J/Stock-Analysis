"""Deterministic daily-bar technical indicators.

All calculations are causal and use only values present on or before the final
row.  Missing OHLCV values are never forward-filled or back-filled.  Returns,
rolling windows, and Wilder recurrences become unavailable when their required
input window is incomplete.
"""

from __future__ import annotations

from math import sqrt
from typing import Any

import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _safe_float(value: Any) -> float | None:
    """Return a finite float-like value, or ``None`` for unavailable data."""
    if value is None or pd.isna(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def _wilder_average(values: pd.Series, period: int) -> pd.Series:
    """Return Wilder's recursive moving average, resetting after missing data."""
    result = pd.Series(float("nan"), index=values.index, dtype="float64")
    seed: list[float] = []
    average: float | None = None
    for index, raw_value in values.items():
        value = _safe_float(raw_value)
        if value is None:
            seed = []
            average = None
            continue
        if average is None:
            seed.append(value)
            if len(seed) < period:
                continue
            if len(seed) > period:
                seed = seed[-period:]
            average = sum(seed) / period
        else:
            average = ((period - 1) * average + value) / period
        result.at[index] = average
    return result


def _empty_result(symbol: str) -> dict[str, Any]:
    """Return the complete indicator shape with unavailable values."""
    return {
        "symbol": symbol,
        "history_length": 0,
        "valid_adjusted_close_count": 0,
        "ohlcv_completeness": 0.0,
        "indicator_availability_ratio": 0.0,
        "latest_close": None,
        "latest_adjusted_close": None,
        "latest_raw_close": None,
        "latest_open": None,
        "latest_high": None,
        "latest_low": None,
        "latest_volume": None,
        "latest_trading_date": None,
        "one_day_return": None,
        "five_day_return": None,
        "one_month_return": None,
        "three_month_return": None,
        "six_month_return": None,
        "one_year_return": None,
        "twenty_day_sma": None,
        "fifty_day_sma": None,
        "hundred_day_sma": None,
        "two_hundred_day_sma": None,
        "distance_from_sma50": None,
        "distance_from_sma200": None,
        "twenty_day_average_volume": None,
        "volume_relative_to_average": None,
        "twenty_day_average_dollar_volume": None,
        "twenty_day_median_dollar_volume": None,
        "average_dollar_volume": None,
        "median_dollar_volume": None,
        "zero_volume_days": None,
        "twenty_day_zero_volume_days": None,
        "twenty_day_volatility": None,
        "sixty_day_volatility": None,
        "two_hundred_fifty_two_day_volatility": None,
        "sixty_day_downside_volatility": None,
        "fifty_two_week_high": None,
        "distance_from_52_week_high": None,
        "max_drawdown": None,
        "one_year_max_drawdown": None,
        "full_history_max_drawdown": None,
        "worst_one_day_return_last_year": None,
        "overnight_gap_volatility": None,
        "rsi_14": None,
        "atr_14": None,
        "atr_percentage": None,
        "trend_slope_50": None,
        "trend_slope_200": None,
        "time_above_sma50": None,
        "time_above_sma200": None,
        "relative_strength_trend": None,
        "benchmark_relative_return_21": None,
        "benchmark_relative_return_63": None,
        "benchmark_relative_return_126": None,
        "benchmark_relative_return_252": None,
        "beta": None,
        "correlation": None,
        "status": "Insufficient Data",
        "flags": ["Not enough price history"],
    }


def calculate_indicators(
    history: list[dict[str, Any]],
    symbol: str,
    *,
    slope_period: int = 20,
) -> dict[str, Any]:
    """Calculate Phase 2 indicators from unmodified daily OHLCV records.

    Prices and returns use reported adjusted close.  ATR and overnight gaps use
    adjusted OHLC derived from ``adjusted_close / close`` so every input is on a
    consistent corporate-action-adjusted basis.  SMA slopes compare the current
    SMA with the same SMA ``slope_period`` trading rows earlier.
    """
    if slope_period <= 0:
        raise ValueError("slope_period must be positive")
    result = _empty_result(symbol)
    if not history:
        return result

    frame = pd.DataFrame(history).copy()
    if "trade_date" not in frame.columns:
        return result
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = (
        frame.dropna(subset=["trade_date"])
        .sort_values("trade_date")
        .drop_duplicates(subset=["trade_date"], keep="last")
        .reset_index(drop=True)
    )
    if frame.empty:
        return result

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "dividends",
        "stock_splits",
    ]
    for column in numeric_columns:
        if column not in frame.columns:
            frame[column] = float("nan")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    price = frame["adjusted_close"].astype("float64")
    raw_close = frame["close"].astype("float64")
    volume = frame["volume"].astype("float64")
    latest_row = frame.iloc[-1]
    latest_price = _safe_float(price.iloc[-1])

    adjustment_factor = (price / raw_close.where(raw_close != 0)).astype("float64")
    adjusted_open = frame["open"] * adjustment_factor
    adjusted_high = frame["high"] * adjustment_factor
    adjusted_low = frame["low"] * adjustment_factor

    returns = price.pct_change(fill_method=None)
    sma20_series = price.rolling(20, min_periods=20).mean()
    sma50_series = price.rolling(50, min_periods=50).mean()
    sma100_series = price.rolling(100, min_periods=100).mean()
    sma200_series = price.rolling(200, min_periods=200).mean()

    def period_return(window: int) -> float | None:
        segment = price.tail(window + 1)
        if len(segment) != window + 1 or segment.isna().any() or segment.iloc[0] == 0:
            return None
        return float(segment.iloc[-1] / segment.iloc[0] - 1.0)

    def latest_rolling_value(series: pd.Series) -> float | None:
        return _safe_float(series.iloc[-1]) if not series.empty else None

    def annualized_volatility(window: int) -> float | None:
        segment = returns.tail(window)
        if len(segment) != window or segment.isna().any() or len(segment) < 2:
            return None
        return float(segment.std(ddof=1) * sqrt(TRADING_DAYS_PER_YEAR))

    def downside_volatility(window: int) -> float | None:
        segment = returns.tail(window)
        if len(segment) != window or segment.isna().any():
            return None
        downside = segment.clip(upper=0.0)
        return float(sqrt(float((downside**2).mean())) * sqrt(TRADING_DAYS_PER_YEAR))

    def max_drawdown(series: pd.Series, *, require_complete: bool = False) -> float | None:
        if require_complete and series.isna().any():
            return None
        valid = series.dropna()
        if len(valid) < 2:
            return None
        running_high = valid.cummax()
        drawdowns = 1.0 - valid / running_high
        return float(drawdowns.max())

    def moving_average_slope(series: pd.Series) -> float | None:
        if len(series) <= slope_period:
            return None
        current = _safe_float(series.iloc[-1])
        earlier = _safe_float(series.iloc[-slope_period - 1])
        if current is None or earlier in (None, 0.0):
            return None
        return float(current / earlier - 1.0)

    def time_above_sma(series: pd.Series, observation_window: int) -> float | None:
        valid_mask = price.notna() & series.notna()
        comparisons = (price[valid_mask] > series[valid_mask]).tail(observation_window)
        if len(comparisons) != observation_window:
            return None
        return float(comparisons.mean())

    # Wilder RSI from adjusted-price changes, not changes in percentage returns.
    price_changes = price.diff()
    gains = price_changes.clip(lower=0.0)
    losses = -price_changes.clip(upper=0.0)
    average_gain = _wilder_average(gains, 14)
    average_loss = _wilder_average(losses, 14)
    latest_gain = latest_rolling_value(average_gain)
    latest_loss = latest_rolling_value(average_loss)
    rsi_14: float | None = None
    if latest_gain is not None and latest_loss is not None:
        if latest_loss == 0.0:
            rsi_14 = 50.0 if latest_gain == 0.0 else 100.0
        else:
            relative_strength = latest_gain / latest_loss
            rsi_14 = float(100.0 - 100.0 / (1.0 + relative_strength))

    # True range is strict except for the first row, where no previous close exists.
    previous_adjusted_close = price.shift(1)
    true_range = pd.Series(float("nan"), index=frame.index, dtype="float64")
    for index in frame.index:
        high_value = _safe_float(adjusted_high.at[index])
        low_value = _safe_float(adjusted_low.at[index])
        if high_value is None or low_value is None:
            continue
        if index == frame.index[0]:
            true_range.at[index] = high_value - low_value
            continue
        previous_close = _safe_float(previous_adjusted_close.at[index])
        if previous_close is None:
            continue
        true_range.at[index] = max(
            high_value - low_value,
            abs(high_value - previous_close),
            abs(low_value - previous_close),
        )
    atr_series = _wilder_average(true_range, 14)
    atr_14 = latest_rolling_value(atr_series)
    atr_percentage = (
        None
        if atr_14 is None or latest_price in (None, 0.0)
        else float(atr_14 / latest_price)
    )

    daily_dollar_volume = price * volume
    average_volume_series = volume.rolling(20, min_periods=20).mean()
    average_dollar_volume_series = daily_dollar_volume.rolling(20, min_periods=20).mean()
    median_dollar_volume_series = daily_dollar_volume.rolling(20, min_periods=20).median()
    twenty_day_average_volume = latest_rolling_value(average_volume_series)
    latest_volume = _safe_float(volume.iloc[-1])
    volume_relative_to_average = (
        None
        if latest_volume is None or twenty_day_average_volume in (None, 0.0)
        else float(latest_volume / twenty_day_average_volume)
    )

    known_volume = volume.dropna()
    zero_volume_days = int((known_volume == 0).sum()) if not known_volume.empty else None
    recent_volume = volume.tail(20)
    twenty_day_zero_volume_days = (
        int((recent_volume == 0).sum())
        if len(recent_volume) == 20 and not recent_volume.isna().any()
        else None
    )

    fifty_two_week_prices = price.tail(252)
    fifty_two_week_high: float | None = None
    distance_from_high: float | None = None
    if len(fifty_two_week_prices) == 252 and not fifty_two_week_prices.isna().any():
        fifty_two_week_high = float(fifty_two_week_prices.max())
        if fifty_two_week_high != 0 and latest_price is not None:
            distance_from_high = float(latest_price / fifty_two_week_high - 1.0)

    one_year_prices = price.tail(252)
    one_year_max_drawdown = (
        max_drawdown(one_year_prices, require_complete=True)
        if len(one_year_prices) == 252
        else None
    )
    full_history_max_drawdown = max_drawdown(price)

    recent_returns = returns.tail(252)
    worst_one_day_return = (
        float(recent_returns.min())
        if len(recent_returns) == 252 and not recent_returns.isna().any()
        else None
    )

    overnight_gaps = adjusted_open / previous_adjusted_close - 1.0
    valid_gaps = overnight_gaps.dropna()
    overnight_gap_volatility = (
        float(valid_gaps.std(ddof=1) * sqrt(TRADING_DAYS_PER_YEAR))
        if len(valid_gaps) >= 2
        else None
    )

    sma20 = latest_rolling_value(sma20_series)
    sma50 = latest_rolling_value(sma50_series)
    sma100 = latest_rolling_value(sma100_series)
    sma200 = latest_rolling_value(sma200_series)
    distance_from_sma50 = (
        None
        if latest_price is None or sma50 in (None, 0.0)
        else float(latest_price / sma50 - 1.0)
    )
    distance_from_sma200 = (
        None
        if latest_price is None or sma200 in (None, 0.0)
        else float(latest_price / sma200 - 1.0)
    )

    required_ohlcv = frame[["open", "high", "low", "close", "adjusted_close", "volume"]]
    ohlcv_completeness = float(required_ohlcv.notna().to_numpy().mean())

    result.update(
        {
            "history_length": int(len(frame)),
            "valid_adjusted_close_count": int(price.notna().sum()),
            "ohlcv_completeness": ohlcv_completeness,
            "latest_close": latest_price,
            "latest_adjusted_close": latest_price,
            "latest_raw_close": _safe_float(latest_row["close"]),
            "latest_open": _safe_float(latest_row["open"]),
            "latest_high": _safe_float(latest_row["high"]),
            "latest_low": _safe_float(latest_row["low"]),
            "latest_volume": latest_volume,
            "latest_trading_date": latest_row["trade_date"].strftime("%Y-%m-%d"),
            "one_day_return": _safe_float(returns.iloc[-1]),
            "five_day_return": period_return(5),
            "one_month_return": period_return(21),
            "three_month_return": period_return(63),
            "six_month_return": period_return(126),
            "one_year_return": period_return(252),
            "twenty_day_sma": sma20,
            "fifty_day_sma": sma50,
            "hundred_day_sma": sma100,
            "two_hundred_day_sma": sma200,
            "distance_from_sma50": distance_from_sma50,
            "distance_from_sma200": distance_from_sma200,
            "twenty_day_average_volume": twenty_day_average_volume,
            "volume_relative_to_average": volume_relative_to_average,
            "twenty_day_average_dollar_volume": latest_rolling_value(
                average_dollar_volume_series
            ),
            "twenty_day_median_dollar_volume": latest_rolling_value(
                median_dollar_volume_series
            ),
            # Backward-compatible aliases now carry the correct 20-session values.
            "average_dollar_volume": latest_rolling_value(average_dollar_volume_series),
            "median_dollar_volume": latest_rolling_value(median_dollar_volume_series),
            "zero_volume_days": zero_volume_days,
            "twenty_day_zero_volume_days": twenty_day_zero_volume_days,
            "twenty_day_volatility": annualized_volatility(20),
            "sixty_day_volatility": annualized_volatility(60),
            "two_hundred_fifty_two_day_volatility": annualized_volatility(252),
            "sixty_day_downside_volatility": downside_volatility(60),
            "fifty_two_week_high": fifty_two_week_high,
            "distance_from_52_week_high": distance_from_high,
            "max_drawdown": full_history_max_drawdown,
            "one_year_max_drawdown": one_year_max_drawdown,
            "full_history_max_drawdown": full_history_max_drawdown,
            "worst_one_day_return_last_year": worst_one_day_return,
            "overnight_gap_volatility": overnight_gap_volatility,
            "rsi_14": rsi_14,
            "atr_14": atr_14,
            "atr_percentage": atr_percentage,
            "trend_slope_50": moving_average_slope(sma50_series),
            "trend_slope_200": moving_average_slope(sma200_series),
            "time_above_sma50": time_above_sma(sma50_series, 60),
            "time_above_sma200": time_above_sma(sma200_series, 252),
        }
    )

    availability_fields = (
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
    result["indicator_availability_ratio"] = sum(
        result.get(field) is not None for field in availability_fields
    ) / len(availability_fields)
    status, flags = classify_status(result)
    result["status"] = status
    result["flags"] = flags
    return result


def classify_status(
    metrics: dict[str, Any],
    has_quality_warning: bool = False,
) -> tuple[str, list[str]]:
    """Assign a descriptive trend status without changing numeric scores."""
    if has_quality_warning:
        return "Data Quality Warning", ["Data quality issues detected"]
    latest_close = metrics.get("latest_close")
    if latest_close is None:
        return "Insufficient Data", ["Not enough price history"]

    flags: list[str] = []
    sma50 = metrics.get("fifty_day_sma")
    sma200 = metrics.get("two_hundred_day_sma")
    if sma50 is not None and sma200 is not None:
        if latest_close > sma50 > sma200:
            status = "Uptrend"
        elif latest_close < sma50 < sma200:
            status = "Downtrend"
        else:
            status = "Mixed Trend"
    else:
        status = "Insufficient Data"

    volatility = metrics.get("twenty_day_volatility")
    if volatility is not None and volatility > 0.35:
        flags.append("High Volatility")
        if status == "Mixed Trend":
            status = "High Volatility"

    distance_from_high = metrics.get("distance_from_52_week_high")
    if distance_from_high is not None and distance_from_high >= -0.05:
        flags.append("Near 52-Week High")
    if status == "Insufficient Data" and not flags:
        flags.append("Not enough price history")
    return status, flags
