"""Validation helpers for the canonical Phase 2 scoring configuration.

The scoring configuration is deliberately strict.  A missing, unknown, negative,
non-numeric, or boolean weight is a configuration error rather than something
the scoring functions should try to repair at runtime.
"""

from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import Any, Mapping, Sequence

CANONICAL_OPPORTUNITY_COMPONENTS = (
    "long_term_trend",
    "multi_period_momentum",
    "relative_strength",
    "trend_quality",
    "volume_participation",
    "breakout_positioning",
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
    "benchmark_alignment",
    "indicator_availability",
    "market_context_availability",
    "signal_agreement",
)

ALLOWED_MARKET_REGIMES = frozenset(
    {"Risk-On", "Neutral", "Risk-Off", "Stress", "Insufficient Market Data"}
)


def validate_weight_group(
    weights: object,
    expected_components: Sequence[str],
    group_name: str,
) -> dict[str, float]:
    """Validate one exact 0-100 weight mapping and return float weights.

    ``bool`` is explicitly rejected even though it is an ``int`` subclass.
    The total is intentionally exact, as required by the project contract.
    """
    if not isinstance(weights, Mapping):
        raise ValueError(f"{group_name} must be a mapping")

    expected = set(expected_components)
    actual = set(weights)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unknown:
            details.append("unknown: " + ", ".join(unknown))
        raise ValueError(
            f"{group_name} must contain exactly the canonical components "
            f"({'; '.join(details)})"
        )

    normalized: dict[str, float] = {}
    for component in expected_components:
        value = weights[component]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{group_name}.{component} must be numeric and not boolean")
        numeric_value = float(value)
        if not isfinite(numeric_value):
            raise ValueError(f"{group_name}.{component} must be finite")
        if numeric_value < 0:
            raise ValueError(f"{group_name}.{component} must be nonnegative")
        normalized[component] = numeric_value

    if sum(normalized.values()) != 100.0:
        raise ValueError(f"{group_name} must total exactly 100")
    return normalized


def _numeric_mapping(
    value: object,
    expected_keys: set[str],
    group_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> dict[str, float]:
    """Validate an exact finite numeric threshold mapping."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{group_name} must be a mapping")
    missing = expected_keys - set(value)
    unknown = set(value) - expected_keys
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown: " + ", ".join(sorted(unknown)))
        raise ValueError(f"{group_name} has invalid keys ({'; '.join(details)})")
    result: dict[str, float] = {}
    for key in sorted(expected_keys):
        raw = value[key]
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise ValueError(f"{group_name}.{key} must be numeric and not boolean")
        number = float(raw)
        if not isfinite(number):
            raise ValueError(f"{group_name}.{key} must be finite")
        if minimum is not None and number < minimum:
            raise ValueError(f"{group_name}.{key} must be at least {minimum}")
        if maximum is not None and number > maximum:
            raise ValueError(f"{group_name}.{key} must be at most {maximum}")
        result[key] = number
    return result


def validate_scoring_config(rules: dict[str, Any]) -> dict[str, Any]:
    """Validate every required scoring weight group and return ``rules``.

    All three groups are mandatory.  This prevents a missing group from being
    silently replaced by application defaults and makes saved configuration
    hashes meaningful.
    """
    if not isinstance(rules, dict):
        raise ValueError("Scoring configuration must be a mapping")

    validate_weight_group(
        rules.get("opportunity_weights"),
        CANONICAL_OPPORTUNITY_COMPONENTS,
        "opportunity_weights",
    )
    validate_weight_group(
        rules.get("risk_weights"),
        CANONICAL_RISK_COMPONENTS,
        "risk_weights",
    )
    validate_weight_group(
        rules.get("confidence_weights"),
        CANONICAL_CONFIDENCE_COMPONENTS,
        "confidence_weights",
    )

    critical_components = rules.get("critical_risk_components")
    if critical_components is not None:
        if not isinstance(critical_components, list) or not all(
            isinstance(component, str) for component in critical_components
        ):
            raise ValueError("critical_risk_components must be a list of component names")
        unknown = sorted(set(critical_components) - set(CANONICAL_RISK_COMPONENTS))
        if unknown:
            raise ValueError(
                "critical_risk_components contains unknown components: " + ", ".join(unknown)
            )

    score_thresholds = _numeric_mapping(
        rules.get("score_thresholds"),
        {
            "strong_candidate",
            "candidate",
            "strong_candidate_confidence",
            "candidate_confidence",
            "watch",
            "high_risk",
            "avoid",
        },
        "score_thresholds",
        minimum=0.0,
        maximum=100.0,
    )
    if not (
        score_thresholds["avoid"]
        <= score_thresholds["watch"]
        <= score_thresholds["candidate"]
        <= score_thresholds["strong_candidate"]
    ):
        raise ValueError(
            "Opportunity classification thresholds must be ordered "
            "avoid <= watch <= candidate <= strong_candidate"
        )
    if not (
        score_thresholds["candidate_confidence"]
        <= score_thresholds["strong_candidate_confidence"]
    ):
        raise ValueError(
            "candidate_confidence must not exceed strong_candidate_confidence"
        )

    risk_levels = _numeric_mapping(
        rules.get("risk_level_thresholds"),
        {"low", "moderate", "elevated", "high"},
        "risk_level_thresholds",
        minimum=0.0,
        maximum=100.0,
    )
    if not (
        risk_levels["low"]
        < risk_levels["moderate"]
        < risk_levels["elevated"]
        < risk_levels["high"]
    ):
        raise ValueError("risk_level_thresholds must be strictly increasing")

    freshness = _numeric_mapping(
        rules.get("freshness_thresholds"),
        {"full_business_days", "stale_business_days"},
        "freshness_thresholds",
        minimum=0.0,
    )
    if not all(number.is_integer() for number in freshness.values()):
        raise ValueError("freshness_thresholds values must be whole business-day counts")
    if freshness["stale_business_days"] <= freshness["full_business_days"]:
        raise ValueError("stale_business_days must exceed full_business_days")

    regime = _numeric_mapping(
        rules.get("market_regime_thresholds"),
        {
            "risk_on_minimum_votes",
            "risk_off_minimum_votes",
            "breadth_threshold",
            "defensive_drawdown",
            "stress_drawdown",
        },
        "market_regime_thresholds",
    )
    for key in ("risk_on_minimum_votes", "risk_off_minimum_votes"):
        if not regime[key].is_integer() or not 1 <= regime[key] <= 8:
            raise ValueError(f"market_regime_thresholds.{key} must be an integer from 1 to 8")
    if not 0.0 <= regime["breadth_threshold"] <= 1.0:
        raise ValueError("market_regime_thresholds.breadth_threshold must be from 0 to 1")
    if not -1.0 <= regime["stress_drawdown"] <= regime["defensive_drawdown"] <= 0.0:
        raise ValueError(
            "Market drawdowns must satisfy -1 <= stress_drawdown <= defensive_drawdown <= 0"
        )

    volatility = _numeric_mapping(
        rules.get("volatility_thresholds"),
        {"low", "elevated", "stress"},
        "volatility_thresholds",
        minimum=0.0,
    )
    if not volatility["low"] < volatility["elevated"] < volatility["stress"]:
        raise ValueError("volatility_thresholds must be strictly increasing")
    _numeric_mapping(
        rules.get("liquidity_thresholds"),
        {"warning_volume_ratio", "warning_dollar_volume"},
        "liquidity_thresholds",
        minimum=0.0,
    )

    classification_rules = rules.get("classification_rules")
    if not isinstance(classification_rules, Mapping) or set(classification_rules) != {
        "candidate_market_regimes"
    }:
        raise ValueError(
            "classification_rules must contain exactly candidate_market_regimes"
        )
    candidate_regimes = classification_rules["candidate_market_regimes"]
    if (
        not isinstance(candidate_regimes, list)
        or not candidate_regimes
        or not all(isinstance(item, str) for item in candidate_regimes)
    ):
        raise ValueError("candidate_market_regimes must be a non-empty list of names")
    unknown_regimes = sorted(set(candidate_regimes) - ALLOWED_MARKET_REGIMES)
    if unknown_regimes:
        raise ValueError(
            "candidate_market_regimes contains unknown regimes: "
            + ", ".join(unknown_regimes)
        )

    return rules
