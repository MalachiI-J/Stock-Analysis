"""Transparent, missingness-aware Phase 2 risk scoring."""

from __future__ import annotations

from math import isfinite, log10
from typing import Any, Callable

from stock_scrapper.analysis.scoring_config import (
    CANONICAL_RISK_COMPONENTS,
    validate_weight_group,
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    return min(maximum, max(minimum, value))


def _scale(value: float, low: float, high: float) -> float:
    """Map a low-risk value to 0 and a high-risk value to 100."""
    if high <= low:
        raise ValueError("Risk scale high bound must exceed low bound")
    return _clamp((value - low) / (high - low) * 100.0)


def _direct_score(metrics: dict[str, Any], component: str) -> float | None:
    value = _number(metrics.get(f"{component}_score"))
    return value if value is not None and 0.0 <= value <= 100.0 else None


def _realized_volatility(metrics: dict[str, Any], rules: dict[str, Any]) -> float | None:
    del rules
    direct = _direct_score(metrics, "realized_volatility")
    if direct is not None:
        return direct
    values = [
        _number(metrics.get("twenty_day_volatility")),
        _number(metrics.get("sixty_day_volatility")),
        _number(metrics.get("two_hundred_fifty_two_day_volatility")),
    ]
    if any(value is None for value in values):
        return None
    return sum(_scale(float(value), 0.10, 0.60) for value in values) / len(values)


def _drawdown_risk(metrics: dict[str, Any], rules: dict[str, Any]) -> float | None:
    del rules
    direct = _direct_score(metrics, "drawdown_risk")
    if direct is not None:
        return direct
    one_year = _number(metrics.get("one_year_max_drawdown"))
    full_history = _number(metrics.get("full_history_max_drawdown"))
    if one_year is None or full_history is None:
        return None
    return _scale(max(abs(one_year), abs(full_history)), 0.05, 0.50)


def _downside_volatility(metrics: dict[str, Any], rules: dict[str, Any]) -> float | None:
    del rules
    direct = _direct_score(metrics, "downside_volatility")
    if direct is not None:
        return direct
    value = _number(metrics.get("sixty_day_downside_volatility"))
    return None if value is None else _scale(value, 0.05, 0.45)


def _atr_gap_risk(metrics: dict[str, Any], rules: dict[str, Any]) -> float | None:
    del rules
    direct = _direct_score(metrics, "atr_gap_risk")
    if direct is not None:
        return direct
    atr_percentage = _number(metrics.get("atr_percentage"))
    gap_volatility = _number(metrics.get("overnight_gap_volatility"))
    if atr_percentage is None or gap_volatility is None:
        return None
    return (
        _scale(atr_percentage, 0.01, 0.08)
        + _scale(gap_volatility, 0.05, 0.50)
    ) / 2.0


def _beta_sensitivity(metrics: dict[str, Any], rules: dict[str, Any]) -> float | None:
    del rules
    direct = _direct_score(metrics, "beta_sensitivity")
    if direct is not None:
        return direct
    beta = _number(metrics.get("beta_252"))
    if beta is None:
        beta = _number(metrics.get("beta"))
    return None if beta is None else _scale(abs(beta), 0.50, 2.00)


def _trend_deterioration(metrics: dict[str, Any], rules: dict[str, Any]) -> float | None:
    del rules
    direct = _direct_score(metrics, "trend_deterioration")
    if direct is not None:
        return direct
    distance200 = _number(metrics.get("distance_from_sma200"))
    slope50 = _number(metrics.get("trend_slope_50"))
    slope200 = _number(metrics.get("trend_slope_200"))
    above50 = _number(metrics.get("time_above_sma50"))
    above200 = _number(metrics.get("time_above_sma200"))
    if None in (distance200, slope50, slope200, above50, above200):
        return None
    sub_scores = (
        _clamp(-distance200 / 0.20 * 100.0),
        _clamp(-slope50 / 0.08 * 100.0),
        _clamp(-slope200 / 0.12 * 100.0),
        _clamp((1.0 - above50) * 100.0),
        _clamp((1.0 - above200) * 100.0),
    )
    return sum(sub_scores) / len(sub_scores)


def _liquidity_risk(metrics: dict[str, Any], rules: dict[str, Any]) -> float | None:
    direct = _direct_score(metrics, "liquidity_risk")
    if direct is not None:
        return direct
    average_dollar = _number(
        metrics.get("twenty_day_average_dollar_volume", metrics.get("average_dollar_volume"))
    )
    median_dollar = _number(
        metrics.get("twenty_day_median_dollar_volume", metrics.get("median_dollar_volume"))
    )
    volume_ratio = _number(metrics.get("volume_relative_to_average"))
    zero_days = _number(
        metrics.get("twenty_day_zero_volume_days", metrics.get("zero_volume_days"))
    )
    if None in (average_dollar, median_dollar, volume_ratio, zero_days):
        return None
    warning_dollar_volume = _number(
        rules.get("liquidity_thresholds", {}).get("warning_dollar_volume")
    )
    warning_dollar_volume = warning_dollar_volume or 1_000_000.0

    def dollar_risk(value: float) -> float:
        if value <= 0:
            return 100.0
        # 100 at/below the warning floor; 0 at ten times that floor.
        return _clamp(100.0 - log10(value / warning_dollar_volume) * 100.0)

    return (
        dollar_risk(average_dollar)
        + dollar_risk(median_dollar)
        + _clamp((1.0 - volume_ratio) * 100.0)
        + _clamp(zero_days / 5.0 * 100.0)
    ) / 4.0


def _market_regime_risk(metrics: dict[str, Any], rules: dict[str, Any]) -> float | None:
    del rules
    direct = _direct_score(metrics, "market_regime_risk")
    if direct is not None:
        return direct
    regime = metrics.get("_market_regime")
    return {
        "Risk-On": 0.0,
        "Neutral": 25.0,
        "Risk-Off": 65.0,
        "Stress": 100.0,
    }.get(str(regime))


def _data_quality_risk(metrics: dict[str, Any], rules: dict[str, Any]) -> float | None:
    del rules
    return _direct_score(metrics, "data_quality_risk")


_COMPONENT_CALCULATORS: dict[
    str,
    Callable[[dict[str, Any], dict[str, Any]], float | None],
] = {
    "realized_volatility": _realized_volatility,
    "drawdown_risk": _drawdown_risk,
    "downside_volatility": _downside_volatility,
    "atr_gap_risk": _atr_gap_risk,
    "beta_sensitivity": _beta_sensitivity,
    "trend_deterioration": _trend_deterioration,
    "liquidity_risk": _liquidity_risk,
    "market_regime_risk": _market_regime_risk,
    "data_quality_risk": _data_quality_risk,
}


def _quality_component(quality_issues: list[dict[str, Any]]) -> tuple[float, bool]:
    severities = {str(issue.get("severity", "")).strip().lower() for issue in quality_issues}
    if "critical" in severities:
        return 100.0, True
    if severities & {"error", "high"}:
        return 75.0, False
    if "warning" in severities:
        return 35.0, False
    return 0.0, False


def _risk_level(score: float, rules: dict[str, Any]) -> str:
    thresholds = rules.get("risk_level_thresholds", {})
    low = _number(thresholds.get("low")) or 25.0
    moderate = _number(thresholds.get("moderate")) or 45.0
    elevated = _number(thresholds.get("elevated")) or 60.0
    high = _number(thresholds.get("high")) or 75.0
    if score < low:
        return "Low"
    if score < moderate:
        return "Moderate"
    if score < elevated:
        return "Elevated"
    if score < high:
        return "High"
    return "Very High"


def calculate_risk_score(
    metrics: dict[str, Any],
    rules: dict[str, Any],
    market_regime: str,
    quality_issues: list[dict[str, Any]],
) -> tuple[float | None, str, dict[str, Any], list[str]]:
    """Calculate measured risk without interpreting missing values as safe.

    Missing critical components block the score.  Missing noncritical components
    remain explicitly unavailable and the available weights are transparently
    renormalized.  A critical data-quality issue always blocks risk scoring.
    """
    weights = validate_weight_group(
        rules.get("risk_weights"),
        CANONICAL_RISK_COMPONENTS,
        "risk_weights",
    )
    critical = set(
        rules.get(
            "critical_risk_components",
            [
                "realized_volatility",
                "drawdown_risk",
                "downside_volatility",
                "atr_gap_risk",
                "beta_sensitivity",
                "liquidity_risk",
                "market_regime_risk",
            ],
        )
    )
    working_metrics = dict(metrics)
    working_metrics["_market_regime"] = market_regime
    quality_value, critical_quality = _quality_component(quality_issues)

    values: dict[str, float | None] = {}
    for component in CANONICAL_RISK_COMPONENTS:
        if component == "data_quality_risk":
            values[component] = quality_value
        else:
            values[component] = _COMPONENT_CALCULATORS[component](working_metrics, rules)

    missing = [component for component, value in values.items() if value is None]
    missing_critical = [component for component in missing if component in critical]
    available_weight = sum(weights[name] for name, value in values.items() if value is not None)
    components: dict[str, Any] = {}
    for component in CANONICAL_RISK_COMPONENTS:
        value = values[component]
        effective_weight = (
            weights[component] * 100.0 / available_weight
            if value is not None and available_weight > 0
            else 0.0
        )
        contribution = value * effective_weight / 100.0 if value is not None else None
        components[component] = {
            "value": round(value, 6) if value is not None else None,
            "available": value is not None,
            "critical": component in critical,
            "weight": weights[component],
            "effective_weight": round(effective_weight, 6),
            "contribution": round(contribution, 6) if contribution is not None else None,
        }

    reasons: list[str] = []
    if missing:
        reasons.append("Unavailable risk components: " + ", ".join(missing))
    if critical_quality:
        reasons.append("Critical unresolved data-quality issue blocks risk scoring")
    if missing_critical:
        reasons.append("Critical risk inputs missing: " + ", ".join(missing_critical))
    if critical_quality or missing_critical or available_weight == 0:
        return None, "Unavailable", components, reasons

    score = round(
        sum(
            float(component["contribution"])
            for component in components.values()
            if component["contribution"] is not None
        ),
        2,
    )
    if not reasons:
        reasons.append("All measured-risk components were available")
    return score, _risk_level(score, rules), components, reasons
