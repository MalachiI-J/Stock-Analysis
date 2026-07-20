"""Saved-analysis deserialization helpers."""

from __future__ import annotations

import json
from typing import Any, Mapping

from stock_scrapper.models.analysis_models import AnalysisResult


def _json_value(row: Mapping[str, Any], key: str, default: Any) -> Any:
    value = row.get(key)
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def analysis_result_from_row(row: Mapping[str, Any], run: Mapping[str, Any] | None = None) -> AnalysisResult:
    """Reconstruct an ``AnalysisResult`` without recalculating market data."""
    return AnalysisResult(
        symbol=str(row.get("symbol") or ""),
        as_of_date=str(row.get("as_of_date") or (run or {}).get("as_of_date") or ""),
        data_through_date=row.get("data_through_date"),
        market_regime=str((run or {}).get("market_regime") or "Insufficient Market Data"),
        market_regime_confidence=(run or {}).get("market_regime_confidence"),
        risk_score=row.get("risk_score"),
        risk_level=str(row.get("risk_level") or "Unavailable"),
        opportunity_score=row.get("opportunity_score"),
        confidence_score=row.get("confidence_score"),
        classification=str(row.get("classification") or "Insufficient Data"),
        primary_reason=str(row.get("primary_reason") or ""),
        eligible_for_scoring=bool(row.get("eligible_for_scoring")),
        blocking_reasons=_json_value(row, "blocking_reasons_json", []),
        risk_components=_json_value(row, "risk_components_json", {}),
        opportunity_components=_json_value(row, "opportunity_components_json", {}),
        confidence_components=_json_value(row, "confidence_components_json", {}),
        indicators=_json_value(row, "indicators_json", {}),
        flags=_json_value(row, "flags_json", []),
        positive_factors=_json_value(row, "positive_factors_json", []),
        risk_factors=_json_value(row, "risk_factors_json", []),
        confidence_limitations=_json_value(row, "confidence_limitations_json", []),
        quality_concerns=_json_value(row, "quality_concerns_json", []),
        market_regime_effects=_json_value(row, "market_regime_effects_json", []),
        improvement_conditions=_json_value(row, "improvement_conditions_json", []),
        weakening_conditions=_json_value(row, "weakening_conditions_json", []),
        trend_state=str(row.get("trend_state") or "Unknown"),
    )


def results_from_saved_run(run: Mapping[str, Any]) -> list[AnalysisResult]:
    """Deserialize all results embedded by ``get_analysis_run``."""
    return [analysis_result_from_row(row, run) for row in run.get("analyses", [])]
