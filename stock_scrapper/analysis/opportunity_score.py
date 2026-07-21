"""Canonical, price-based Phase 2 opportunity scoring."""

from __future__ import annotations

from math import isfinite
from typing import Any, Callable

from stock_scrapper.analysis.scoring_config import (
    CANONICAL_OPPORTUNITY_COMPONENTS,
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


def _symmetric_score(value: float, positive_magnitude: float) -> float:
    """Map ``-magnitude .. +magnitude`` linearly to ``0 .. 100``."""
    return _clamp(50.0 + 50.0 * value / positive_magnitude)


def _direct_score(metrics: dict[str, Any], component: str) -> float | None:
    """Read an optional canonical 0-100 rolling-feature score override."""
    value = _number(metrics.get(f"{component}_score"))
    if value is None or not 0.0 <= value <= 100.0:
        return None
    return value


def _long_term_trend(metrics: dict[str, Any]) -> float | None:
    direct = _direct_score(metrics, "long_term_trend")
    if direct is not None:
        return direct
    latest = _number(metrics.get("latest_close"))
    sma50 = _number(metrics.get("fifty_day_sma"))
    sma200 = _number(metrics.get("two_hundred_day_sma"))
    slope200 = _number(metrics.get("trend_slope_200"))
    time_above200 = _number(metrics.get("time_above_sma200"))
    if None in (latest, sma50, sma200, slope200, time_above200) or sma200 == 0:
        return None
    price_distance = latest / sma200 - 1.0
    average_spread = sma50 / sma200 - 1.0
    return (
        0.35 * _symmetric_score(price_distance, 0.20)
        + 0.25 * _symmetric_score(average_spread, 0.10)
        + 0.20 * _symmetric_score(slope200, 0.10)
        + 0.20 * _clamp(time_above200 * 100.0)
    )


def _multi_period_momentum(metrics: dict[str, Any]) -> float | None:
    direct = _direct_score(metrics, "multi_period_momentum")
    if direct is not None:
        return direct
    periods = (
        ("one_month_return", 0.10),
        ("three_month_return", 0.20),
        ("six_month_return", 0.30),
        ("one_year_return", 0.40),
    )
    values = [(_number(metrics.get(key)), magnitude) for key, magnitude in periods]
    if any(value is None for value, _ in values):
        return None
    numeric_values = [(float(value), magnitude) for value, magnitude in values if value is not None]
    return sum(_symmetric_score(value, magnitude) for value, magnitude in numeric_values) / len(numeric_values)


def _relative_strength(metrics: dict[str, Any]) -> float | None:
    direct = _direct_score(metrics, "relative_strength")
    if direct is not None:
        return direct
    periods = (
        ("benchmark_relative_return_21", 0.05),
        ("benchmark_relative_return_63", 0.10),
        ("benchmark_relative_return_126", 0.15),
        ("benchmark_relative_return_252", 0.25),
        ("relative_strength_trend", 0.10),
    )
    values = [(_number(metrics.get(key)), magnitude) for key, magnitude in periods]
    if any(value is None for value, _ in values):
        return None
    numeric_values = [(float(value), magnitude) for value, magnitude in values if value is not None]
    return sum(_symmetric_score(value, magnitude) for value, magnitude in numeric_values) / len(numeric_values)


def _trend_quality(metrics: dict[str, Any]) -> float | None:
    direct = _direct_score(metrics, "trend_quality")
    if direct is not None:
        return direct
    above50 = _number(metrics.get("time_above_sma50"))
    above200 = _number(metrics.get("time_above_sma200"))
    slope50 = _number(metrics.get("trend_slope_50"))
    slope200 = _number(metrics.get("trend_slope_200"))
    if None in (above50, above200, slope50, slope200):
        return None
    return (
        _clamp(above50 * 100.0)
        + _clamp(above200 * 100.0)
        + _symmetric_score(slope50, 0.08)
        + _symmetric_score(slope200, 0.12)
    ) / 4.0


def _volume_participation(metrics: dict[str, Any]) -> float | None:
    direct = _direct_score(metrics, "volume_participation")
    if direct is not None:
        return direct
    volume_ratio = _number(metrics.get("volume_relative_to_average"))
    daily_return = _number(metrics.get("one_day_return"))
    if volume_ratio is None or daily_return is None or volume_ratio < 0:
        return None
    positive_participation = _clamp(volume_ratio / 2.0 * 100.0)
    if daily_return > 0:
        return positive_participation
    if daily_return < 0:
        return 100.0 - positive_participation
    return 50.0


def _breakout_positioning(metrics: dict[str, Any]) -> float | None:
    direct = _direct_score(metrics, "breakout_positioning")
    if direct is not None:
        return direct
    distance = _number(metrics.get("distance_from_52_week_high"))
    if distance is None:
        return None
    # Zero at 30% or more below the high; 100 at the high.
    return _clamp((distance + 0.30) / 0.30 * 100.0)


_COMPONENT_CALCULATORS: dict[str, Callable[[dict[str, Any]], float | None]] = {
    "long_term_trend": _long_term_trend,
    "multi_period_momentum": _multi_period_momentum,
    "relative_strength": _relative_strength,
    "trend_quality": _trend_quality,
    "volume_participation": _volume_participation,
    "breakout_positioning": _breakout_positioning,
}


def _level(score: float) -> str:
    if score < 25:
        return "Low"
    if score < 45:
        return "Moderate"
    if score < 60:
        return "Elevated"
    if score < 75:
        return "High"
    return "Very High"


def calculate_opportunity_score(
    metrics: dict[str, Any],
    rules: dict[str, Any],
    market_regime: str | None = None,
) -> tuple[float | None, str, dict[str, Any], list[str]]:
    """Calculate the six-component canonical technical opportunity score.

    ``market_regime`` remains in the signature for compatibility but is not a
    component: regime gates belong in classification, not in the technical
    opportunity score.  A missing component makes the score unavailable rather
    than contributing an implicit zero or being reweighted away.
    """
    del market_regime
    weights = validate_weight_group(
        rules.get("opportunity_weights"),
        CANONICAL_OPPORTUNITY_COMPONENTS,
        "opportunity_weights",
    )
    components: dict[str, Any] = {}
    missing: list[str] = []
    for component in CANONICAL_OPPORTUNITY_COMPONENTS:
        value = _COMPONENT_CALCULATORS[component](metrics)
        available = value is not None
        contribution = value * weights[component] / 100.0 if available else None
        components[component] = {
            "value": round(value, 6) if value is not None else None,
            "available": available,
            "weight": weights[component],
            "contribution": round(contribution, 6) if contribution is not None else None,
        }
        if not available:
            missing.append(component)

    if missing:
        return (
            None,
            "Unavailable",
            components,
            ["Unavailable opportunity components: " + ", ".join(missing)],
        )

    score = round(
        sum(float(component["contribution"]) for component in components.values()),
        2,
    )
    return (
        score,
        _level(score),
        components,
        ["All six canonical price-based opportunity components were available"],
    )
