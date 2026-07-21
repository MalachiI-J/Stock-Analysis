"""One-per-date market context and watchlist breadth calculation."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from statistics import stdev
from typing import Any, Iterable


@dataclass(frozen=True)
class MarketContext:
    """Explainable market-regime result shared by every symbol on one date."""

    regime: str
    confidence: float
    metrics: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def _prices(history: Iterable[dict[str, Any]]) -> list[float | None]:
    """Preserve one adjusted-price slot per row without substituting raw close."""
    values: list[float | None] = []
    for row in history:
        value = row.get("adjusted_close")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            values.append(None)
            continue
        values.append(numeric if numeric > 0 else None)
    return values


def _sma(values: list[float | None], window: int) -> float | None:
    if len(values) < window:
        return None
    tail = values[-window:]
    if any(value is None for value in tail):
        return None
    return sum(float(value) for value in tail if value is not None) / window


def _period_return(values: list[float | None], sessions: int) -> float | None:
    if len(values) < sessions + 1:
        return None
    tail = values[-sessions - 1 :]
    if any(value is None for value in tail) or tail[0] == 0:
        return None
    return float(tail[-1]) / float(tail[0]) - 1.0


def _volatility(values: list[float | None], sessions: int) -> float | None:
    if len(values) < sessions + 1:
        return None
    tail = values[-sessions - 1 :]
    if any(value is None for value in tail):
        return None
    returns = [
        float(tail[index]) / float(tail[index - 1]) - 1.0
        for index in range(1, len(tail))
    ]
    return stdev(returns) * sqrt(252.0) if len(returns) >= 2 else None


def _trend_confirmation(
    history: list[dict[str, Any]], expected_trade_date: str | None
) -> bool | None:
    if (
        not history
        or expected_trade_date is None
        or str(history[-1].get("trade_date") or "")[:10] != expected_trade_date
    ):
        return None
    values = _prices(history)
    sma50 = _sma(values, 50)
    sma200 = _sma(values, 200)
    if not values or values[-1] is None or sma50 is None or sma200 is None:
        return None
    return bool(float(values[-1]) > sma200 and sma50 > sma200)


def calculate_watchlist_breadth(
    histories: dict[str, list[dict[str, Any]]],
    eligible_symbols: Iterable[str] | None = None,
) -> tuple[float | None, int, int]:
    """Return eligible symbols above SMA200 divided by those with a valid SMA200."""
    allowed = {symbol.upper() for symbol in eligible_symbols} if eligible_symbols is not None else None
    numerator = 0
    denominator = 0
    for symbol in sorted(histories):
        if allowed is not None and symbol.upper() not in allowed:
            continue
        values = _prices(histories[symbol])
        sma200 = _sma(values, 200)
        if not values or values[-1] is None or sma200 is None:
            continue
        denominator += 1
        if float(values[-1]) > sma200:
            numerator += 1
    return (numerator / denominator if denominator else None), numerator, denominator


def calculate_market_context(
    benchmark_history: list[dict[str, Any]],
    context_histories: dict[str, list[dict[str, Any]]],
    breadth_ratio: float | None,
    rules: dict[str, Any],
) -> MarketContext:
    """Calculate SPY/QQQ/IWM regime inputs once with deterministic thresholds."""
    spy = _prices(benchmark_history)
    valid_spy_sessions = sum(value is not None for value in spy)
    if len(spy) < 200:
        return MarketContext(
            regime="Insufficient Market Data",
            confidence=0.0,
            metrics={
                "benchmark_sessions": len(spy),
                "benchmark_valid_sessions": valid_spy_sessions,
                "breadth_ratio": breadth_ratio,
                "context_availability_ratio": 0.0,
            },
            reasons=["At least 200 valid benchmark sessions are required"],
        )

    latest = spy[-1]
    sma50 = _sma(spy, 50)
    sma200 = _sma(spy, 200)
    if latest is None or sma50 is None or sma200 is None:
        return MarketContext(
            regime="Insufficient Market Data",
            confidence=0.0,
            metrics={
                "benchmark_sessions": len(spy),
                "benchmark_valid_sessions": valid_spy_sessions,
                "breadth_ratio": breadth_ratio,
                "context_availability_ratio": 0.0,
            },
            reasons=["Current and trailing SPY adjusted-price inputs must be complete"],
        )
    latest = float(latest)
    return63 = _period_return(spy, 63)
    trailing252 = spy[-252:]
    drawdown52 = None
    if len(trailing252) == 252 and not any(value is None for value in trailing252):
        high252 = max(float(value) for value in trailing252 if value is not None)
        drawdown52 = latest / high252 - 1.0 if high252 else None
    volatility20 = _volatility(spy, 20)
    volatility60 = _volatility(spy, 60)
    benchmark_trade_date = (
        str(benchmark_history[-1].get("trade_date") or "")[:10]
        if benchmark_history
        else None
    )
    qqq_confirm = _trend_confirmation(
        context_histories.get("QQQ", []), benchmark_trade_date
    )
    iwm_confirm = _trend_confirmation(
        context_histories.get("IWM", []), benchmark_trade_date
    )
    thresholds = rules.get("market_regime_thresholds", {})
    breadth_threshold = float(thresholds.get("breadth_threshold", 0.5))
    risk_on_minimum_votes = int(thresholds.get("risk_on_minimum_votes", 6))
    risk_off_minimum_votes = int(thresholds.get("risk_off_minimum_votes", 5))
    defensive_drawdown = float(thresholds.get("defensive_drawdown", -0.10))
    stress_volatility = float(
        rules.get("volatility_thresholds", {}).get("stress", thresholds.get("stress_volatility", 0.55))
    )
    stress_long_volatility = float(
        rules.get("volatility_thresholds", {}).get("elevated", 0.35)
    )
    stress_drawdown = float(thresholds.get("stress_drawdown", -0.15))

    availability_values = (
        latest,
        sma50,
        sma200,
        return63,
        drawdown52,
        volatility20,
        volatility60,
        qqq_confirm,
        iwm_confirm,
        breadth_ratio,
    )
    context_availability_ratio = sum(
        value is not None for value in availability_values
    ) / len(availability_values)

    metrics: dict[str, Any] = {
        "benchmark_sessions": len(spy),
        "benchmark_valid_sessions": valid_spy_sessions,
        "spy_latest": latest,
        "spy_sma50": sma50,
        "spy_sma200": sma200,
        "spy_return_63": return63,
        "spy_drawdown_52_week": drawdown52,
        "spy_volatility_20": volatility20,
        "spy_volatility_60": volatility60,
        "qqq_trend_confirmation": qqq_confirm,
        "iwm_trend_confirmation": iwm_confirm,
        "breadth_ratio": breadth_ratio,
        "context_availability_ratio": context_availability_ratio,
    }
    reasons: list[str] = []
    assert sma50 is not None and sma200 is not None

    stressed = bool(
        latest < sma200
        and drawdown52 is not None
        and drawdown52 <= stress_drawdown
        and volatility20 is not None
        and volatility20 >= stress_volatility
        and volatility60 is not None
        and volatility60 >= stress_long_volatility
    )
    if stressed:
        reasons.extend(
            [
                "SPY is below its 200-day moving average",
                "SPY drawdown and short-term volatility exceed configured stress thresholds",
            ]
        )
        return MarketContext("Stress", 0.9, metrics, reasons)

    risk_on_votes: list[bool | None] = [
        latest > sma50,
        latest > sma200,
        sma50 > sma200,
        None if return63 is None else return63 > 0,
        None if drawdown52 is None else drawdown52 > defensive_drawdown,
        qqq_confirm,
        iwm_confirm,
        None if breadth_ratio is None else breadth_ratio >= breadth_threshold,
    ]
    risk_off_votes: list[bool | None] = [
        latest < sma50,
        latest < sma200,
        sma50 < sma200,
        None if return63 is None else return63 < 0,
        None if drawdown52 is None else drawdown52 <= defensive_drawdown,
        None if qqq_confirm is None else not qqq_confirm,
        None if iwm_confirm is None else not iwm_confirm,
        None if breadth_ratio is None else breadth_ratio < breadth_threshold,
    ]
    on_count = sum(vote is True for vote in risk_on_votes)
    off_count = sum(vote is True for vote in risk_off_votes)
    available_votes = sum(vote is not None for vote in risk_on_votes)

    if on_count >= risk_on_minimum_votes and latest > sma200 and sma50 > sma200:
        regime = "Risk-On"
        confidence = min(1.0, on_count / max(1, available_votes))
        reasons.append("SPY trend, momentum, drawdown, and broad-market confirmations are constructive")
    elif off_count >= risk_off_minimum_votes and (latest < sma200 or sma50 < sma200):
        regime = "Risk-Off"
        confidence = min(1.0, off_count / max(1, available_votes))
        reasons.append("SPY trend and multiple market-context measures are defensive")
    else:
        regime = "Neutral"
        confidence = max(0.4, abs(on_count - off_count) / max(1, available_votes))
        reasons.append("Market trend and confirmation measures are mixed")

    if breadth_ratio is None:
        reasons.append("Watchlist breadth was unavailable")
    else:
        reasons.append(
            f"Eligible watchlist breadth was {breadth_ratio:.1%} versus a {breadth_threshold:.1%} threshold"
        )
    if qqq_confirm is None or iwm_confirm is None:
        reasons.append("QQQ or IWM trend confirmation was unavailable")
    return MarketContext(regime, round(confidence, 4), metrics, reasons)
