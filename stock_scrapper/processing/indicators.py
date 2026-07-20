"""Transparent market indicators and status classification."""

from __future__ import annotations

from math import sqrt
from typing import Any

import pandas as pd


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def calculate_indicators(history: list[dict[str, Any]], symbol: str) -> dict[str, Any]:
    """Calculate a compact set of technical indicators for a symbol using price-based metrics only."""
    empty_result = {
        "symbol": symbol,
        "latest_close": None,
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
        "twenty_day_volatility": None,
        "sixty_day_volatility": None,
        "two_hundred_fifty_two_day_volatility": None,
        "sixty_day_downside_volatility": None,
        "fifty_two_week_high": None,
        "distance_below_52_week_high": None,
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
        "average_dollar_volume": None,
        "median_dollar_volume": None,
        "zero_volume_days": None,
        "relative_strength_trend": None,
        "benchmark_relative_return_21": None,
        "benchmark_relative_return_63": None,
        "benchmark_relative_return_126": None,
        "benchmark_relative_return_252": None,
    }

    if not history:
        return empty_result

    frame = pd.DataFrame(history)
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=["trade_date"]).sort_values("trade_date")

    if frame.empty:
        return empty_result

    for column in ["close", "open", "high", "low", "adjusted_close", "volume"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            frame[column] = pd.Series([None] * len(frame), index=frame.index)

    price_series = frame["adjusted_close"].fillna(frame["close"])
    price_series = price_series.ffill().dropna()
    if price_series.empty:
        return empty_result

    raw_close = frame["close"].ffill().dropna()
    raw_open = frame["open"].ffill().dropna()
    raw_high = frame["high"].ffill().dropna()
    raw_low = frame["low"].ffill().dropna()
    latest_close = float(price_series.iloc[-1])
    latest_date = frame.iloc[-1]["trade_date"].strftime("%Y-%m-%d")
    returns = price_series.pct_change().dropna()

    def _calc_period_return(window: int) -> float | None:
        if len(price_series) < window + 1:
            return None
        return float(price_series.iloc[-1] / price_series.iloc[-window - 1] - 1)

    def _rolling_mean(window: int) -> float | None:
        if len(price_series) < window:
            return None
        return float(price_series.rolling(window).mean().iloc[-1])

    def _rolling_slope(window: int) -> float | None:
        if len(price_series) < window:
            return None
        series = price_series.rolling(window).mean()
        if series.iloc[-1] in [None, 0]:
            return None
        return float((price_series.iloc[-1] / series.iloc[-1] - 1) * 100.0)

    def _time_above_sma(window: int) -> float | None:
        if len(price_series) < window:
            return None
        sma = price_series.rolling(window).mean()
        above = (price_series > sma).sum() / len(price_series)
        return float(above)

    def _compute_rsi(period: int = 14) -> float | None:
        if len(returns) < period + 1:
            return None
        deltas = returns.diff().fillna(0.0)
        gains = deltas.clip(lower=0).rolling(period).mean()
        losses = (-deltas.clip(upper=0)).rolling(period).mean()
        rs = gains / losses.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None

    def _compute_atr(period: int = 14) -> tuple[float | None, float | None]:
        if len(price_series) < period + 1:
            return None, None
        prev_close = price_series.shift(1)
        tr = pd.concat([
            (frame["high"] - frame["low"]).abs(),
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        atr_pct = None if atr is None or latest_close in [None, 0] else float(atr / latest_close)
        return float(atr) if pd.notna(atr) else None, atr_pct

    def _compute_volatility(window: int) -> float | None:
        if len(returns) < window:
            return None
        series = returns.tail(window)
        return float(series.std() * sqrt(252)) if len(series) >= 2 else None

    def _compute_downside_volatility(window: int) -> float | None:
        if len(returns) < window:
            return None
        neg_returns = returns.tail(window).clip(upper=0)
        if neg_returns.empty:
            return None
        return float((neg_returns.pow(2).mean() ** 0.5) * sqrt(252))

    def _compute_max_drawdown(window: int | None = None) -> float | None:
        if window is None:
            series = price_series
        else:
            series = price_series.tail(window)
        if len(series) < 2:
            return None
        cumulative = series / series.iloc[0]
        running_max = cumulative.cummax()
        drawdown = ((running_max - cumulative) / running_max).max()
        return float(drawdown)

    def _compute_worst_day(window: int) -> float | None:
        if len(returns) < window:
            return None
        series = returns.tail(window)
        return float(series.min()) if len(series) >= 1 else None

    def _compute_gap_volatility() -> float | None:
        if len(raw_open) < 2 or len(raw_close) < 2 or len(raw_high) < 2 or len(raw_low) < 2:
            return None
        prev_close = raw_close.shift(1)
        gap = (raw_open / prev_close - 1.0).dropna()
        if gap.empty:
            return None
        return float(gap.std() * sqrt(252)) if len(gap) >= 2 else None

    one_day_return = float(returns.iloc[-1]) if len(returns) >= 1 else None
    five_day_return = _calc_period_return(5)
    one_month_return = _calc_period_return(21)
    three_month_return = _calc_period_return(63)
    six_month_return = _calc_period_return(126)
    one_year_return = _calc_period_return(252)

    twenty_day_sma = _rolling_mean(20)
    fifty_day_sma = _rolling_mean(50)
    hundred_day_sma = _rolling_mean(100)
    two_hundred_day_sma = _rolling_mean(200)

    distance_from_sma50 = None if fifty_day_sma is None else float((latest_close - fifty_day_sma) / fifty_day_sma * 100)
    distance_from_sma200 = None if two_hundred_day_sma is None else float((latest_close - two_hundred_day_sma) / two_hundred_day_sma * 100)

    volume_series = frame["volume"].ffill().dropna()
    twenty_day_average_volume = float(volume_series.rolling(20).mean().iloc[-1]) if len(volume_series) >= 20 else None
    volume_relative_to_average = None if twenty_day_average_volume in [None, 0] else float(volume_series.iloc[-1] / twenty_day_average_volume)
    average_dollar_volume = None if twenty_day_average_volume is None else float(volume_series.iloc[-1] * latest_close)
    median_dollar_volume = None if volume_series.empty else float((volume_series * latest_close).median())
    zero_volume_days = int((volume_series == 0).sum()) if not volume_series.empty else None

    twenty_day_volatility = _compute_volatility(20)
    sixty_day_volatility = _compute_volatility(60)
    two_hundred_fifty_two_day_volatility = _compute_volatility(252)
    sixty_day_downside_volatility = _compute_downside_volatility(60)

    window_size = min(252, len(price_series))
    fifty_two_week_high = float(price_series.tail(window_size).max()) if len(price_series) >= 2 else None
    distance_below_52_week_high = None if fifty_two_week_high in [None, 0] else float((fifty_two_week_high - latest_close) / latest_close * 100)

    one_year_max_drawdown = _compute_max_drawdown(252)
    full_history_max_drawdown = _compute_max_drawdown(None)
    worst_one_day_return_last_year = _compute_worst_day(252)
    overnight_gap_volatility = _compute_gap_volatility()

    rsi_14 = _compute_rsi(14)
    atr_14, atr_percentage = _compute_atr(14)
    trend_slope_50 = _rolling_slope(50)
    trend_slope_200 = _rolling_slope(200)
    time_above_sma50 = _time_above_sma(50)
    time_above_sma200 = _time_above_sma(200)

    return {
        "symbol": symbol,
        "latest_close": latest_close,
        "latest_trading_date": latest_date,
        "one_day_return": one_day_return,
        "five_day_return": five_day_return,
        "one_month_return": one_month_return,
        "three_month_return": three_month_return,
        "six_month_return": six_month_return,
        "one_year_return": one_year_return,
        "twenty_day_sma": twenty_day_sma,
        "fifty_day_sma": fifty_day_sma,
        "hundred_day_sma": hundred_day_sma,
        "two_hundred_day_sma": two_hundred_day_sma,
        "distance_from_sma50": distance_from_sma50,
        "distance_from_sma200": distance_from_sma200,
        "twenty_day_average_volume": twenty_day_average_volume,
        "volume_relative_to_average": volume_relative_to_average,
        "twenty_day_volatility": twenty_day_volatility,
        "sixty_day_volatility": sixty_day_volatility,
        "two_hundred_fifty_two_day_volatility": two_hundred_fifty_two_day_volatility,
        "sixty_day_downside_volatility": sixty_day_downside_volatility,
        "fifty_two_week_high": fifty_two_week_high,
        "distance_below_52_week_high": distance_below_52_week_high,
        "max_drawdown": full_history_max_drawdown if full_history_max_drawdown is not None else None,
        "one_year_max_drawdown": one_year_max_drawdown,
        "full_history_max_drawdown": full_history_max_drawdown,
        "worst_one_day_return_last_year": worst_one_day_return_last_year,
        "overnight_gap_volatility": overnight_gap_volatility,
        "rsi_14": rsi_14,
        "atr_14": atr_14,
        "atr_percentage": atr_percentage,
        "trend_slope_50": trend_slope_50,
        "trend_slope_200": trend_slope_200,
        "time_above_sma50": time_above_sma50,
        "time_above_sma200": time_above_sma200,
        "average_dollar_volume": average_dollar_volume,
        "median_dollar_volume": median_dollar_volume,
        "zero_volume_days": zero_volume_days,
        "relative_strength_trend": None,
        "benchmark_relative_return_21": None,
        "benchmark_relative_return_63": None,
        "benchmark_relative_return_126": None,
        "benchmark_relative_return_252": None,
    }


def classify_status(metrics: dict[str, Any], has_quality_warning: bool = False) -> tuple[str, list[str]]:
    """Assign a simple descriptive status and a set of flags."""
    flags: list[str] = []

    if has_quality_warning:
        return "Data Quality Warning", ["Data quality issues detected"]

    if metrics.get("latest_close") is None:
        return "Insufficient Data", ["Not enough price history"]

    sma50 = metrics.get("fifty_day_sma")
    sma200 = metrics.get("two_hundred_day_sma")
    latest_close = metrics.get("latest_close")

    if sma50 is not None and sma200 is not None:
        if latest_close > sma50 and sma50 > sma200:
            status = "Uptrend"
        elif latest_close < sma50 and sma50 < sma200:
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

    distance_from_high = metrics.get("distance_below_52_week_high")
    if distance_from_high is not None and distance_from_high <= 5:
        flags.append("Near 52-Week High")

    if status == "Insufficient Data" and not flags:
        flags.append("Not enough price history")

    return status, flags
