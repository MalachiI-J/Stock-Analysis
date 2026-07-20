"""Deterministic market-regime analysis."""

from __future__ import annotations

from typing import Any


def _price_from_row(row: dict[str, Any]) -> float | None:
    if row is None:
        return None
    for key in ("adjusted_close", "close"):
        value = row.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def calculate_market_regime(
    benchmark_history: list[dict[str, Any]],
    context_histories: dict[str, list[dict[str, Any]]],
    breadth_ratio: float | None,
    rules: dict[str, Any],
) -> tuple[str, float, list[str], list[str]]:
    """Assign a market regime using actual SPY and context metrics."""
    reasons: list[str] = []
    components: list[str] = []
    if not benchmark_history:
        return "Insufficient Market Data", 0.0, ["Benchmark data unavailable"], ["No benchmark history"]

    prices = [p for p in (_price_from_row(row) for row in benchmark_history) if p is not None]
    if len(prices) < 2:
        return "Insufficient Market Data", 0.0, ["Benchmark data unavailable"], ["No benchmark price"]

    latest_price = prices[-1]
    recent_price = prices[-1]
    sma50 = None
    sma200 = None
    if len(prices) >= 50:
        sma50 = sum(prices[-50:]) / 50.0
    if len(prices) >= 200:
        sma200 = sum(prices[-200:]) / 200.0

    regime = "Neutral"
    confidence = 0.45
    if sma50 is not None and sma200 is not None:
        if recent_price > sma50 and sma50 > sma200:
            regime = "Risk-On"
            confidence = 0.75
            components.append("SPY is above its 50-day and 200-day SMAs")
            reasons.append("SPY is above its 50-day and 200-day moving averages")
        elif recent_price < sma50 and sma50 < sma200:
            regime = "Risk-Off"
            confidence = 0.75
            components.append("SPY is below its 50-day and 200-day SMAs")
            reasons.append("SPY is below its 50-day and 200-day moving averages")
        else:
            components.append("SPY trend is mixed")
            reasons.append("SPY trend is mixed across the 50-day and 200-day moving averages")

    if breadth_ratio is not None:
        if breadth_ratio >= rules.get("market_regime_thresholds", {}).get("breadth_threshold", 0.5):
            components.append("Eligible watchlist breadth is constructive")
        else:
            components.append("Eligible watchlist breadth is weak")

    qqq_history = context_histories.get("QQQ") or []
    iwm_history = context_histories.get("IWM") or []
    if qqq_history and iwm_history:
        qqq_price = _price_from_row(qqq_history[-1])
        iwm_price = _price_from_row(iwm_history[-1])
        if qqq_price is not None and iwm_price is not None:
            components.append("QQQ/IWM context available")
            reasons.append("QQQ and IWM context was available for confirmation")

    if regime == "Risk-On" and breadth_ratio is not None and breadth_ratio < 0.5:
        regime = "Neutral"
        confidence = 0.55
        reasons.append("Breadth was not strong enough to support a full risk-on regime")
    elif regime == "Risk-Off" and breadth_ratio is not None and breadth_ratio > 0.5:
        regime = "Neutral"
        confidence = 0.55
        reasons.append("Breadth was constructive even though the benchmark trend was weak")

    return regime, confidence, components, reasons
