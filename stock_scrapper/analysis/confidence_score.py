"""Deterministic confidence scoring for Phase 2."""

from __future__ import annotations

from typing import Any


def calculate_confidence_score(metrics: dict[str, Any], rules: dict[str, Any], quality_issues: list[dict[str, Any]]) -> tuple[float | None, str, dict[str, Any], list[str]]:
    """Calculate a transparent confidence score from 0 to 100."""
    if not metrics or metrics.get("latest_close") is None:
        return None, "Unavailable", {}, ["Missing price data"]

    base = float(rules.get("confidence_base", 50.0))
    history_bonus = min(20.0, max(0.0, (metrics.get("history_length", 0) / 365.0) * 10.0))
    quality_penalty = 25.0 if quality_issues else 0.0
    data_quality = max(0.0, 100.0 - quality_penalty)
    stability = max(0.0, min(100.0, 100.0 - (metrics.get("twenty_day_volatility", 0.0) * 100.0)))
    score = round(min(100.0, max(0.0, base + history_bonus + data_quality / 10.0 + stability / 10.0)), 2)
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
    return score, level, {"history_bonus": history_bonus, "quality_penalty": quality_penalty, "stability": stability}, ["Confidence is driven by history length and data quality"]
