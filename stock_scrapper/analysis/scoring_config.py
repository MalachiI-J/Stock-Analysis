"""Validation helpers for canonical scoring configuration."""

from __future__ import annotations

from typing import Any

CANONICAL_OPPORTUNITY_COMPONENTS = (
    "momentum",
    "trend_strength",
    "relative_strength",
    "quality",
    "valuation",
    "market_regime_bonus",
)
CANONICAL_RISK_COMPONENTS = (
    "realized_volatility",
    "drawdown_risk",
    "downside_volatility",
    "atr_gap_risk",
    "beta_sensitivity",
    "trend_deterioration",
    "liquidity_risk",
    "market_regime_risk",
    "data_quality_risk",
)
CANONICAL_CONFIDENCE_COMPONENTS = (
    "history_completeness",
    "data_freshness",
    "data_quality",
    "benchmark_availability",
    "indicator_availability",
    "signal_agreement",
)


def validate_scoring_config(rules: dict[str, Any]) -> dict[str, Any]:
    """Validate the canonical component schema and return the configuration."""
    opportunity_weights = rules.get("opportunity_weights", {})
    risk_weights = rules.get("risk_weights", {})
    confidence_weights = rules.get("confidence_weights", {})

    if set(opportunity_weights.keys()) != set(CANONICAL_OPPORTUNITY_COMPONENTS):
        raise ValueError(
            "opportunity_weights must contain exactly the canonical components: "
            + ", ".join(CANONICAL_OPPORTUNITY_COMPONENTS)
        )
    if sum(opportunity_weights.values()) != 100:
        raise ValueError("opportunity_weights must total 100")

    if set(risk_weights.keys()) != set(CANONICAL_RISK_COMPONENTS):
        raise ValueError(
            "risk_weights must contain exactly the canonical components: "
            + ", ".join(CANONICAL_RISK_COMPONENTS)
        )
    if sum(risk_weights.values()) != 100:
        raise ValueError("risk_weights must total 100")

    if confidence_weights:
        if set(confidence_weights.keys()) != set(CANONICAL_CONFIDENCE_COMPONENTS):
            raise ValueError(
                "confidence_weights must contain exactly the canonical components: "
                + ", ".join(CANONICAL_CONFIDENCE_COMPONENTS)
            )
        if sum(confidence_weights.values()) != 100:
            raise ValueError("confidence_weights must total 100")

    return rules
