"""Transparent market indicators and status classification."""

from __future__ import annotations

from math import sqrt
from typing import Any

import pandas as pd


def calculate_indicators(history: list[dict[str, Any]], symbol: str) -> dict[str, Any]:
    """Calculate a compact set of technical indicators for a symbol."""
    if not history:
        return {
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
            "two_hundred_day_sma": None,
            "distance_from_sma50": None,
            "distance_from_sma200": None,
            "twenty_day_average_volume": None,
            "volume_relative_to_average": None,
            "twenty_day_volatility": None,
            "fifty_two_week_high": None,
            "distance_below_52_week_high": None,
            "max_drawdown": None,
        }

    frame = pd.DataFrame(history)
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=["trade_date"]).sort_values("trade_date")

    if frame.empty:
        return {
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
            "two_hundred_day_sma": None,
            "distance_from_sma50": None,
            "distance_from_sma200": None,
            "twenty_day_average_volume": None,
            "volume_relative_to_average": None,
            "twenty_day_volatility": None,
            "fifty_two_week_high": None,
            "distance_below_52_week_high": None,
            "max_drawdown": None,
        }

    # Use adjusted close when available so returns remain consistent after splits and dividends.
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["adjusted_close"] = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")

    price_series = frame["adjusted_close"].fillna(frame["close"])
    price_series = price_series.ffill().dropna()

    if price_series.empty:
        return {
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
            "two_hundred_day_sma": None,
            "distance_from_sma50": None,
            "distance_from_sma200": None,
            "twenty_day_average_volume": None,
            "volume_relative_to_average": None,
            "twenty_day_volatility": None,
            "fifty_two_week_high": None,
            "distance_below_52_week_high": None,
            "max_drawdown": None,
        }

    latest_close = float(price_series.iloc[-1])
    latest_date = frame.iloc[-1]["trade_date"].strftime("%Y-%m-%d")
    returns = price_series.pct_change().dropna()

    def _calc_period_return(window: int) -> float | None:
        if len(price_series) < window + 1:
            return None
        return float(price_series.iloc[-1] / price_series.iloc[-window - 1] - 1)

    one_day_return = float(returns.iloc[-1]) if len(returns) >= 1 else None
    five_day_return = _calc_period_return(5)
    one_month_return = _calc_period_return(21)
    three_month_return = _calc_period_return(63)
    six_month_return = _calc_period_return(126)
    one_year_return = _calc_period_return(252)

    twenty_day_sma = float(price_series.rolling(20).mean().iloc[-1]) if len(price_series) >= 20 else None
    fifty_day_sma = float(price_series.rolling(50).mean().iloc[-1]) if len(price_series) >= 50 else None
    two_hundred_day_sma = float(price_series.rolling(200).mean().iloc[-1]) if len(price_series) >= 200 else None

    distance_from_sma50 = None if fifty_day_sma is None else float((latest_close - fifty_day_sma) / fifty_day_sma * 100)
    distance_from_sma200 = None if two_hundred_day_sma is None else float((latest_close - two_hundred_day_sma) / two_hundred_day_sma * 100)

    volume_series = frame["volume"].ffill().dropna()
    twenty_day_average_volume = float(volume_series.rolling(20).mean().iloc[-1]) if len(volume_series) >= 20 else None
    volume_relative_to_average = None if twenty_day_average_volume in [None, 0] else float(latest_close if False else volume_series.iloc[-1] / twenty_day_average_volume)

    volatility_series = returns.tail(20)
    twenty_day_volatility = float(volatility_series.std() * sqrt(252)) if len(volatility_series) >= 2 else None

    window_size = min(252, len(price_series))
    fifty_two_week_high = float(price_series.tail(window_size).max()) if len(price_series) >= 2 else None
    distance_below_52_week_high = None if fifty_two_week_high in [None, 0] else float((fifty_two_week_high - latest_close) / latest_close * 100)

    cumulative = price_series / price_series.iloc[0]
    running_max = cumulative.cummax()
    max_drawdown = float(((running_max - cumulative) / running_max).max()) if len(cumulative) >= 2 else 0.0

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
        "two_hundred_day_sma": two_hundred_day_sma,
        "distance_from_sma50": distance_from_sma50,
        "distance_from_sma200": distance_from_sma200,
        "twenty_day_average_volume": twenty_day_average_volume,
        "volume_relative_to_average": volume_relative_to_average,
        "twenty_day_volatility": twenty_day_volatility,
        "fifty_two_week_high": fifty_two_week_high,
        "distance_below_52_week_high": distance_below_52_week_high,
        "max_drawdown": max_drawdown,
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
