"""Deterministic confidence scoring for Phase 2."""

from __future__ import annotations

from typing import Any


def calculate_confidence_score(metrics: dict[str, Any], rules: dict[str, Any], quality_issues: list[dict[str, Any]]) -> tuple[float | None, str, dict[str, Any], list[str]]:
    """Calculate a transparent confidence score from 0 to 100 using configured components."""
    if not metrics or metrics.get("latest_close") is None:
        return None, "Unavailable", {}, ["Missing price data"]

    weights = rules.get("confidence_weights") or {}
    if not weights:
        weights = {
            "history_completeness": 30,
            "data_freshness": 20,
            "data_quality": 20,
            "benchmark_availability": 10,
            "indicator_availability": 10,
            "signal_agreement": 10,
        }
    elif sum(weights.values()) != 100:
        raise ValueError("Confidence weights must total 100")

    history_length = max(0, min(1.0, metrics.get("history_length", 0) / 252.0))
    history_completeness = history_length
    freshness = 1.0 if metrics.get("latest_trading_date") else 0.0
    data_quality = 1.0 if not quality_issues else 0.0
    benchmark_availability = 1.0 if metrics.get("benchmark_available", False) else 0.0
    indicator_availability = 1.0 if metrics.get("indicator_availability", 0.0) else 0.0
    signal_agreement = 1.0 if metrics.get("signal_agreement", 0.0) >= 0.5 else 0.0

    components = {
        "history_completeness": history_completeness,
        "data_freshness": freshness,
        "data_quality": data_quality,
        "benchmark_availability": benchmark_availability,
        "indicator_availability": indicator_availability,
        "signal_agreement": signal_agreement,
    }

    score = (
        components["history_completeness"] * weights.get("history_completeness", 0) / 100.0
        + components["data_freshness"] * weights.get("data_freshness", 0) / 100.0
        + components["data_quality"] * weights.get("data_quality", 0) / 100.0
        + components["benchmark_availability"] * weights.get("benchmark_availability", 0) / 100.0
        + components["indicator_availability"] * weights.get("indicator_availability", 0) / 100.0
        + components["signal_agreement"] * weights.get("signal_agreement", 0) / 100.0
    )

    score = round(min(100.0, max(0.0, score * 100.0)), 2)
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
    return score, level, components, ["Confidence is driven by configured quality and agreement factors"]
