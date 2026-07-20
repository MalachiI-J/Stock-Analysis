"""Deterministic market-regime analysis."""

from __future__ import annotations

from typing import Any


def calculate_market_regime(
    benchmark_history: list[dict[str, Any]],
    context_histories: dict[str, list[dict[str, Any]]],
    breadth_ratio: float | None,
    rules: dict[str, Any],
) -> tuple[str, float, list[str], list[str]]:
    """Assign a simple market regime using deterministic rules."""
    reasons: list[str] = []
    components: list[str] = []
    if not benchmark_history:
        return "Insufficient Market Data", 0.0, ["Benchmark data unavailable"], ["No benchmark history"]

    latest = benchmark_history[-1]
    latest_close = latest.get("close")
    if latest_close is None:
        latest_close = latest.get("adjusted_close")

    if not latest_close:
        return "Insufficient Market Data", 0.0, ["Benchmark data unavailable"], ["No benchmark price"]

    regime = "Neutral"
    confidence = 0.5

    # Use a simple deterministic decision tree that can be configured later.
    if len(benchmark_history) >= 200:
        long_term = benchmark_history[-200].get("close") or benchmark_history[-200].get("adjusted_close")
        recent = benchmark_history[-1].get("close") or benchmark_history[-1].get("adjusted_close")
        if long_term and recent and recent > long_term:
            components.append("SPY above its long-term trend anchor")
            regime = "Risk-On"
            confidence = 0.7
            reasons.append("The benchmark's latest close is above the 200-day anchor")
        else:
            components.append("SPY below its long-term trend anchor")
            regime = "Risk-Off"
            confidence = 0.7
            reasons.append("The benchmark is below its long-term trend anchor")

    if breadth_ratio is not None:
        if breadth_ratio >= rules.get("market_regime_thresholds", {}).get("breadth_threshold", 0.5):
            components.append("Watchlist breadth is positive")
        else:
            components.append("Watchlist breadth is weak")

    if regime == "Risk-On" and confidence > 0.6:
        return regime, confidence, components, reasons
    if regime == "Risk-Off" and confidence > 0.6:
        return regime, confidence, components, reasons
    return "Neutral", 0.5, components, reasons
