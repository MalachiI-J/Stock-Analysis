"""Deterministic opportunity scoring for Phase 2."""

from __future__ import annotations

from typing import Any


def calculate_opportunity_score(metrics: dict[str, Any], rules: dict[str, Any], market_regime: str) -> tuple[float | None, str, dict[str, Any], list[str]]:
    """Calculate a transparent opportunity score from 0 to 100 using actual price-based metrics."""
    if not metrics or metrics.get("latest_close") is None:
        return None, "Unavailable", {}, ["Missing price data"]

    weights = rules.get("opportunity_weights", {})
    if sum(weights.values()) != 100:
        raise ValueError("Opportunity weights must total 100")

    momentum = metrics.get("momentum_score") or 0.0
    momentum_score = min(100.0, max(0.0, momentum * 100.0))

    trend_strength = metrics.get("trend_strength") or 0.0
    trend_strength_score = min(100.0, max(0.0, trend_strength * 100.0))

    relative_strength = metrics.get("relative_strength") or 0.0
    relative_strength_score = min(100.0, max(0.0, relative_strength * 100.0))

    quality = metrics.get("quality_score") or 0.0
    quality_score = min(100.0, max(0.0, quality * 100.0))

    valuation = metrics.get("valuation_score") or 0.0
    valuation_score = min(100.0, max(0.0, valuation * 100.0))

    market_regime_bonus = 15.0 if market_regime in {"Risk-On", "Neutral"} else 0.0

    components = {
        "momentum": momentum_score,
        "trend_strength": trend_strength_score,
        "relative_strength": relative_strength_score,
        "quality": quality_score,
        "valuation": valuation_score,
        "market_regime_bonus": market_regime_bonus,
    }

    weighted = (
        components["momentum"] * weights.get("momentum", 0) / 100.0
        + components["trend_strength"] * weights.get("trend_strength", 0) / 100.0
        + components["relative_strength"] * weights.get("relative_strength", 0) / 100.0
        + components["quality"] * weights.get("quality", 0) / 100.0
        + components["valuation"] * weights.get("valuation", 0) / 100.0
        + components["market_regime_bonus"] * weights.get("market_regime_bonus", 0) / 100.0
    )

    score = round(min(100.0, max(0.0, weighted)), 2)
    if score < 25:
        level = "Low"
    elif score < 45:
        level = "Moderate"
    elif score < 60:
        level = "Elevated"
    elif score < 75:
        level = "High"
    else:
        level = "Very High"
    return score, level, components, ["Momentum and relative-strength factors were weighted into the score"]
