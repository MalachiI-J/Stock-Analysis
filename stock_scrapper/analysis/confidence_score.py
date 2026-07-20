"""Reliability- and completeness-based Phase 2 confidence scoring."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from math import isfinite
from typing import Any, Mapping

from stock_scrapper.analysis.scoring_config import (
    CANONICAL_CONFIDENCE_COMPONENTS,
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


def _fraction_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return 100.0 if value else 0.0
    numeric = _number(value)
    if numeric is None:
        return None
    return _clamp(numeric * 100.0)


def _direct_score(metrics: Mapping[str, Any], component: str) -> float | None:
    value = _number(metrics.get(f"{component}_score"))
    return value if value is not None and 0.0 <= value <= 100.0 else None


def _coerce_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None:
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _business_day_lag(latest: date, as_of: date) -> int | None:
    """Count weekdays after ``latest`` through ``as_of``; holidays are external."""
    if latest > as_of:
        return None
    cursor = latest + timedelta(days=1)
    count = 0
    while cursor <= as_of:
        if cursor.weekday() < 5:
            count += 1
        cursor += timedelta(days=1)
    return count


def _history_completeness(metrics: Mapping[str, Any]) -> float | None:
    direct = _direct_score(metrics, "history_completeness")
    if direct is not None:
        return direct
    explicit = _fraction_score(metrics.get("history_completeness"))
    if explicit is not None:
        return explicit
    valid_count = _number(metrics.get("valid_adjusted_close_count"))
    if valid_count is None:
        valid_count = _number(metrics.get("history_length"))
    expected_count = _number(metrics.get("expected_history_days")) or 252.0
    if valid_count is None or expected_count <= 0:
        return None
    coverage = _clamp(valid_count / expected_count * 100.0)
    field_completeness = _fraction_score(metrics.get("ohlcv_completeness"))
    return coverage if field_completeness is None else (coverage + field_completeness) / 2.0


def _freshness(
    metrics: Mapping[str, Any],
    rules: Mapping[str, Any],
    as_of_date: Any,
) -> tuple[float | None, str | None]:
    direct = _direct_score(metrics, "data_freshness")
    if direct is not None:
        return direct, None
    latest = _coerce_date(metrics.get("latest_trading_date"))
    as_of = _coerce_date(as_of_date) or _coerce_date(metrics.get("as_of_date"))
    if latest is None or as_of is None:
        return None, "Freshness could not be measured without latest and as-of dates"
    lag = _business_day_lag(latest, as_of)
    if lag is None:
        return 0.0, "Latest price date is later than the analysis as-of date"

    thresholds = rules.get("freshness_thresholds", {})
    full_lag = int(_number(thresholds.get("full_business_days")) or 1)
    stale_lag = int(_number(thresholds.get("stale_business_days")) or 5)
    if stale_lag <= full_lag:
        raise ValueError("stale_business_days must exceed full_business_days")
    if lag <= full_lag:
        return 100.0, None
    if lag >= stale_lag:
        return 0.0, f"Price history is stale by {lag} business days"
    score = 100.0 * (stale_lag - lag) / (stale_lag - full_lag)
    return score, f"Price history lags the as-of date by {lag} business days"


def _data_quality(quality_issues: list[dict[str, Any]]) -> tuple[float, str | None]:
    severities = {str(issue.get("severity", "")).strip().lower() for issue in quality_issues}
    if "critical" in severities:
        return 0.0, "Critical unresolved data-quality issue"
    if severities & {"error", "high"}:
        return 25.0, "High-severity unresolved data-quality issue"
    if "warning" in severities:
        return 60.0, "Unresolved data-quality warning"
    return 100.0, None


def _benchmark_alignment(metrics: Mapping[str, Any]) -> float | None:
    direct = _direct_score(metrics, "benchmark_alignment")
    if direct is not None:
        return direct
    ratio = _fraction_score(metrics.get("benchmark_alignment_ratio"))
    if ratio is not None:
        return ratio
    available = metrics.get("benchmark_available")
    return _fraction_score(available) if isinstance(available, bool) else None


def _indicator_availability(metrics: Mapping[str, Any]) -> float | None:
    direct = _direct_score(metrics, "indicator_availability")
    if direct is not None:
        return direct
    ratio = metrics.get("indicator_availability_ratio")
    if ratio is None:
        ratio = metrics.get("indicator_availability")
    return _fraction_score(ratio)


def _market_context_availability(
    metrics: Mapping[str, Any],
    market_context: Mapping[str, Any] | None,
) -> float | None:
    direct = _direct_score(metrics, "market_context_availability")
    if direct is not None:
        return direct
    explicit = _fraction_score(metrics.get("market_context_availability"))
    if explicit is not None:
        return explicit
    if market_context is not None:
        explicit = _fraction_score(market_context.get("availability_ratio"))
        if explicit is not None:
            return explicit
        available = market_context.get("available")
        if isinstance(available, bool):
            return 100.0 if available else 0.0
        regime = market_context.get("regime")
        if regime is not None:
            return 0.0 if regime == "Insufficient Market Data" else 100.0
    regime = metrics.get("market_regime")
    if regime is not None:
        return 0.0 if regime == "Insufficient Market Data" else 100.0
    return None


def _direction(value: float, tolerance: float = 1e-12) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def _signal_agreement(metrics: Mapping[str, Any]) -> float | None:
    direct = _direct_score(metrics, "signal_agreement")
    if direct is not None:
        return direct
    explicit = _fraction_score(metrics.get("signal_agreement"))
    if explicit is not None:
        return explicit

    trend = _number(metrics.get("distance_from_sma200"))
    momentum_values = [
        _number(metrics.get("one_month_return")),
        _number(metrics.get("three_month_return")),
        _number(metrics.get("six_month_return")),
        _number(metrics.get("one_year_return")),
    ]
    relative_values = [
        _number(metrics.get("benchmark_relative_return_21")),
        _number(metrics.get("benchmark_relative_return_63")),
        _number(metrics.get("benchmark_relative_return_126")),
        _number(metrics.get("benchmark_relative_return_252")),
    ]
    if trend is None or any(value is None for value in momentum_values + relative_values):
        return None
    directions = [
        _direction(trend),
        _direction(sum(float(value) for value in momentum_values) / len(momentum_values)),
        _direction(sum(float(value) for value in relative_values) / len(relative_values)),
    ]
    modal_count = max(directions.count(direction) for direction in (-1, 0, 1))
    return modal_count / len(directions) * 100.0


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


def calculate_confidence_score(
    metrics: dict[str, Any],
    rules: dict[str, Any],
    quality_issues: list[dict[str, Any]],
    as_of_date: str | date | datetime | None = None,
    market_context: Mapping[str, Any] | None = None,
) -> tuple[float | None, str, dict[str, Any], list[str]]:
    """Calculate reliability from completeness, freshness, and agreement.

    Unavailable confidence dimensions are explicit and contribute no confidence;
    weights are not renormalized.  This intentionally lowers confidence when the
    analysis cannot establish benchmark, indicator, or market-context coverage.
    """
    weights = validate_weight_group(
        rules.get("confidence_weights"),
        CANONICAL_CONFIDENCE_COMPONENTS,
        "confidence_weights",
    )
    freshness, freshness_reason = _freshness(metrics, rules, as_of_date)
    quality, quality_reason = _data_quality(quality_issues)
    values: dict[str, float | None] = {
        "history_completeness": _history_completeness(metrics),
        "data_freshness": freshness,
        "data_quality": quality,
        "benchmark_alignment": _benchmark_alignment(metrics),
        "indicator_availability": _indicator_availability(metrics),
        "market_context_availability": _market_context_availability(
            metrics, market_context
        ),
        "signal_agreement": _signal_agreement(metrics),
    }

    components: dict[str, Any] = {}
    limitations: list[str] = []
    for component in CANONICAL_CONFIDENCE_COMPONENTS:
        value = values[component]
        contribution = value * weights[component] / 100.0 if value is not None else 0.0
        components[component] = {
            "value": round(value, 6) if value is not None else None,
            "available": value is not None,
            "weight": weights[component],
            "contribution": round(contribution, 6),
        }
        if value is None:
            limitations.append(component.replace("_", " ").title() + " unavailable")

    if freshness_reason:
        limitations.append(freshness_reason)
    if quality_reason:
        limitations.append(quality_reason)
    score = round(sum(component["contribution"] for component in components.values()), 2)
    if not limitations:
        limitations.append("No material confidence limitations detected")
    return score, _level(score), components, limitations
