"""Canonical deterministic Phase 2 analysis and persistence."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any, Mapping

from stock_scrapper.analysis.confidence_score import calculate_confidence_score
from stock_scrapper.analysis.eligibility import evaluate_eligibility
from stock_scrapper.analysis.market_context import MarketContext, calculate_market_context
from stock_scrapper.analysis.opportunity_score import calculate_opportunity_score
from stock_scrapper.analysis.risk_score import calculate_risk_score
from stock_scrapper.analysis.scoring_config import validate_scoring_config
from stock_scrapper.models.analysis_models import AnalysisResult
from stock_scrapper.processing.historical_features import HistoricalFeatureSnapshot
from stock_scrapper.processing.indicators import calculate_indicators
from stock_scrapper.processing.relative_strength import calculate_relative_strength_metrics
from stock_scrapper.utilities.hashing import canonical_json


def _coerce_date(value: str | date | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _normalize_history(
    history: list[dict[str, Any]], as_of_date: date | None = None
) -> list[dict[str, Any]]:
    """Preserve complete daily bars and causally filter/sort them by date."""
    by_date: dict[str, dict[str, Any]] = {}
    for source in history:
        raw_date = source.get("trade_date")
        try:
            parsed = date.fromisoformat(str(raw_date)[:10])
        except (TypeError, ValueError):
            continue
        if as_of_date is not None and parsed > as_of_date:
            continue
        row = dict(source)
        row["trade_date"] = parsed.isoformat()
        # Explicit fields make missing OHLCV visible to indicators; no filling occurs.
        for field in (
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
            "dividends",
            "stock_splits",
        ):
            row.setdefault(field, None)
        by_date[parsed.isoformat()] = row
    return [by_date[key] for key in sorted(by_date)]


def _component_value(component: Any) -> float | None:
    if isinstance(component, Mapping):
        value = component.get("value")
    else:
        value = component
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _classify(
    *,
    eligible: bool,
    eligibility_meta: dict[str, Any],
    risk_score: float | None,
    opportunity_score: float | None,
    confidence_score: float | None,
    market_regime: str,
    rules: dict[str, Any],
) -> str:
    """Apply the documented classification precedence exactly once."""
    if eligibility_meta.get("critical_data_issue") or eligibility_meta.get("invalid_data"):
        return "Data Blocked"
    if not eligible or any(value is None for value in (risk_score, opportunity_score, confidence_score)):
        return "Insufficient Data"

    thresholds = rules.get("score_thresholds", {})
    high_risk = float(thresholds.get("high_risk", 70.0))
    avoid = float(thresholds.get("avoid", 30.0))
    watch = float(thresholds.get("watch", 50.0))
    candidate = float(thresholds.get("candidate", 65.0))
    strong = float(thresholds.get("strong_candidate", 75.0))
    candidate_confidence = float(thresholds.get("candidate_confidence", 60.0))
    strong_confidence = float(thresholds.get("strong_candidate_confidence", 70.0))
    allowed_regimes = set(
        rules.get("classification_rules", {}).get(
            "candidate_market_regimes", ["Risk-On", "Neutral"]
        )
    )
    assert risk_score is not None and opportunity_score is not None and confidence_score is not None
    if risk_score >= high_risk:
        return "High Risk"
    if market_regime == "Stress" or opportunity_score <= avoid:
        return "Avoid"
    if opportunity_score < watch:
        return "Avoid"
    if (
        opportunity_score >= strong
        and confidence_score >= strong_confidence
        and market_regime in allowed_regimes
    ):
        return "Strong Candidate"
    if (
        opportunity_score >= candidate
        and confidence_score >= candidate_confidence
        and market_regime in allowed_regimes
    ):
        return "Candidate"
    return "Watch"


def analyze_symbol(
    symbol: str,
    history: list[dict[str, Any]],
    benchmark_history: list[dict[str, Any]],
    quality_issues: list[dict[str, Any]],
    as_of_date: str | date,
    rules: dict[str, Any],
    minimum_history_days: int,
    minimum_recent_days: int = 20,
    *,
    market_context: MarketContext | None = None,
    context_histories: dict[str, list[dict[str, Any]]] | None = None,
    breadth_ratio: float | None = None,
    feature_snapshot: HistoricalFeatureSnapshot | None = None,
    history_is_normalized: bool = False,
) -> AnalysisResult:
    """Analyze one symbol using the same service contract as historical tests."""
    del minimum_recent_days  # retained for API compatibility; freshness is measured, not imputed.
    validate_scoring_config(rules)
    as_of = _coerce_date(as_of_date) or date.today()
    if history_is_normalized:
        normalized_history = list(history)
        normalized_benchmark = list(benchmark_history)
        normalized_context = {
            key.upper(): list(rows) for key, rows in (context_histories or {}).items()
        }
    else:
        normalized_history = _normalize_history(history, as_of)
        normalized_benchmark = _normalize_history(benchmark_history, as_of)
        normalized_context = {
            key.upper(): _normalize_history(rows, as_of)
            for key, rows in (context_histories or {}).items()
        }

    if feature_snapshot is None:
        base_metrics = calculate_indicators(normalized_history, symbol)
        relative_metrics = calculate_relative_strength_metrics(
            normalized_history, normalized_benchmark
        )
    else:
        base_metrics = dict(feature_snapshot.base_metrics)
        relative_metrics = dict(feature_snapshot.relative_metrics)
    metrics = dict(base_metrics)
    metrics.update(relative_metrics)
    metrics.update(
        {
            "history_length": len(normalized_history),
            "expected_history_days": max(252, minimum_history_days),
            "as_of_date": as_of.isoformat(),
            "quality_issue_count": len(quality_issues),
        }
    )

    if market_context is None:
        market_context = calculate_market_context(
            normalized_benchmark,
            normalized_context,
            breadth_ratio,
            rules,
        )
    metrics["market_regime"] = market_context.regime
    metrics["market_context_availability"] = (
        0.0
        if market_context.regime == "Insufficient Market Data"
        else float(market_context.metrics.get("context_availability_ratio", 0.0))
    )
    metrics["market_context_metrics"] = market_context.metrics

    eligible, blocking_reasons, eligibility_meta = evaluate_eligibility(
        symbol=symbol.upper(),
        history=normalized_history,
        quality_issues=quality_issues,
        as_of_date=as_of,
        minimum_history_days=minimum_history_days,
    )

    risk_score: float | None = None
    risk_level = "Unavailable"
    risk_components: dict[str, Any] = {}
    risk_reasons: list[str] = []
    opportunity_score: float | None = None
    opportunity_components: dict[str, Any] = {}
    opportunity_reasons: list[str] = []
    confidence_score: float | None = None
    confidence_components: dict[str, Any] = {}
    confidence_reasons: list[str] = []

    if eligible:
        risk_score, risk_level, risk_components, risk_reasons = calculate_risk_score(
            metrics, rules, market_context.regime, quality_issues
        )
        opportunity_score, _, opportunity_components, opportunity_reasons = (
            calculate_opportunity_score(metrics, rules, market_context.regime)
        )
        confidence_score, _, confidence_components, confidence_reasons = (
            calculate_confidence_score(
                metrics,
                rules,
                quality_issues,
                as_of_date=as_of,
                market_context={
                    "regime": market_context.regime,
                    "available": market_context.regime != "Insufficient Market Data",
                },
            )
        )

    classification = _classify(
        eligible=eligible,
        eligibility_meta=eligibility_meta,
        risk_score=risk_score,
        opportunity_score=opportunity_score,
        confidence_score=confidence_score,
        market_regime=market_context.regime,
        rules=rules,
    )

    if classification in {"Data Blocked", "Insufficient Data"}:
        primary_reason = "; ".join(blocking_reasons + risk_reasons + opportunity_reasons)
        primary_reason = primary_reason or "Required analysis inputs were unavailable"
    elif classification == "High Risk":
        primary_reason = "Measured risk exceeds the configured high-risk threshold"
    elif classification == "Avoid":
        primary_reason = "Technical opportunity or market conditions are below the avoid threshold"
    elif classification == "Watch":
        primary_reason = "The setup does not yet satisfy every candidate threshold"
    else:
        primary_reason = "Technical opportunity, measured risk, and confidence satisfy candidate rules"

    positive_factors = [
        name.replace("_", " ").title()
        for name, component in opportunity_components.items()
        if (_component_value(component) or -1.0) >= 65.0
    ]
    if market_context.regime == "Risk-On":
        positive_factors.append("Risk-On market regime")
    risk_factors = list(risk_reasons)
    risk_factors.extend(
        name.replace("_", " ").title()
        for name, component in risk_components.items()
        if (_component_value(component) or -1.0) >= 65.0
    )
    quality_concerns = [
        str(issue.get("description") or issue.get("issue_type") or "Quality issue")
        for issue in quality_issues
    ]
    flags = list(base_metrics.get("flags") or [])
    if quality_issues:
        flags.append("Unresolved data-quality issues present")
    if classification in {"Data Blocked", "Insufficient Data"}:
        flags.append("Not eligible for a complete score")

    improvement_conditions = [
        "Opportunity score rises above the configured candidate threshold",
        "Confidence improves through complete, fresh benchmark-aligned data",
        "Measured risk remains below the configured maximum",
    ]
    if blocking_reasons:
        improvement_conditions.insert(0, "Resolve: " + "; ".join(blocking_reasons))
    weakening_conditions = [
        "Opportunity score falls below the configured exit threshold",
        "Risk score reaches the configured high-risk threshold",
        "Price closes below its 200-day moving average",
        "Market regime deteriorates to Stress",
    ]

    return AnalysisResult(
        symbol=symbol.upper(),
        as_of_date=as_of.isoformat(),
        data_through_date=(normalized_history[-1]["trade_date"] if normalized_history else None),
        market_regime=market_context.regime,
        market_regime_confidence=market_context.confidence,
        risk_score=risk_score,
        risk_level=risk_level,
        opportunity_score=opportunity_score,
        confidence_score=confidence_score,
        classification=classification,
        primary_reason=primary_reason,
        eligible_for_scoring=bool(
            eligible
            and risk_score is not None
            and opportunity_score is not None
            and confidence_score is not None
        ),
        blocking_reasons=blocking_reasons,
        risk_components=risk_components,
        opportunity_components=opportunity_components,
        confidence_components=confidence_components,
        indicators=metrics,
        flags=list(dict.fromkeys(flags)),
        positive_factors=list(dict.fromkeys(positive_factors)),
        risk_factors=list(dict.fromkeys(risk_factors)),
        confidence_limitations=confidence_reasons,
        quality_concerns=quality_concerns,
        market_regime_effects=market_context.reasons,
        improvement_conditions=improvement_conditions,
        weakening_conditions=weakening_conditions,
        trend_state=str(base_metrics.get("status") or "Unknown"),
    )


def persist_analysis_results(
    conn: Any,
    analysis_run_id: str,
    results: list[AnalysisResult],
    as_of_date: str,
    data_through_date: str | None,
    benchmark_symbol: str,
    market_regime: str,
    market_regime_confidence: float | None,
    symbols_requested: list[str],
    symbols_analyzed: list[str],
    symbols_blocked: list[str],
    status: str,
    scoring_version: str,
    configuration_hash: str | None = None,
    error_summary: str | None = None,
    *,
    configuration_snapshot: Mapping[str, Any] | None = None,
    market_regime_metrics: Mapping[str, Any] | None = None,
    market_regime_reasons: list[str] | None = None,
    provenance: Mapping[str, Any] | None = None,
    data_health_status: str | None = None,
    universe_snapshot: Mapping[str, Any] | None = None,
    data_hash: str | None = None,
    analysis_scope: str = "custom",
    candidate_universe_hash: str | None = None,
) -> None:
    """Persist a complete run and exactly one shared market-regime row."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO analysis_runs (
            analysis_run_id, started_at, completed_at, as_of_date, data_through_date,
            benchmark_symbol, market_regime, market_regime_confidence, symbols_requested,
            symbols_analyzed, symbols_blocked, status, scoring_version, configuration_hash,
            error_summary, configuration_snapshot_json, market_regime_reasons_json,
            application_version,schema_version,git_commit_hash,git_dirty,source_fingerprint,
            python_version,platform_info,data_health_status,universe_json,data_hash,
            analysis_scope,is_canonical,requested_symbols_json,analyzed_symbols_json,
            blocked_symbols_json,symbol_count,candidate_universe_hash,universe_configuration_json,
            legacy_scope_inferred
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            analysis_run_id,
            now,
            now,
            as_of_date,
            data_through_date,
            benchmark_symbol,
            market_regime,
            market_regime_confidence,
            ",".join(symbols_requested),
            ",".join(symbols_analyzed),
            ",".join(symbols_blocked),
            status,
            scoring_version,
            configuration_hash,
            error_summary,
            canonical_json(configuration_snapshot or {}),
            canonical_json(market_regime_reasons or []),
            (provenance or {}).get("application_version"),(provenance or {}).get("schema_version"),
            (provenance or {}).get("git_commit_hash"),None if (provenance or {}).get("git_dirty") is None else int((provenance or {}).get("git_dirty")),
            (provenance or {}).get("source_fingerprint"),(provenance or {}).get("python_version"),(provenance or {}).get("platform_info"),
            data_health_status,canonical_json(universe_snapshot or {}),data_hash,
            analysis_scope,0,canonical_json(symbols_requested),canonical_json(symbols_analyzed),
            canonical_json(symbols_blocked),len(symbols_requested),candidate_universe_hash,
            canonical_json(universe_snapshot or {}),0,
        ),
    )
    for result in results:
        conn.execute(
            """
            INSERT INTO stock_analysis (
                analysis_run_id, symbol, as_of_date, data_through_date, risk_score,
                opportunity_score, confidence_score, classification, primary_reason,
                risk_level, trend_state, eligible_for_scoring, blocking_reasons_json,
                risk_components_json, opportunity_components_json, confidence_components_json,
                indicators_json, flags_json, created_at, positive_factors_json,
                risk_factors_json, confidence_limitations_json, quality_concerns_json,
                market_regime_effects_json, improvement_conditions_json,
                weakening_conditions_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis_run_id,
                result.symbol,
                result.as_of_date,
                result.data_through_date,
                result.risk_score,
                result.opportunity_score,
                result.confidence_score,
                result.classification,
                result.primary_reason,
                result.risk_level,
                result.trend_state,
                int(result.eligible_for_scoring),
                canonical_json(result.blocking_reasons),
                canonical_json(result.risk_components),
                canonical_json(result.opportunity_components),
                canonical_json(result.confidence_components),
                canonical_json(result.indicators),
                canonical_json(result.flags),
                now,
                canonical_json(result.positive_factors),
                canonical_json(result.risk_factors),
                canonical_json(result.confidence_limitations),
                canonical_json(result.quality_concerns),
                canonical_json(result.market_regime_effects),
                canonical_json(result.improvement_conditions),
                canonical_json(result.weakening_conditions),
            ),
        )
    conn.execute(
        """
        INSERT INTO market_regime_history (
            analysis_run_id, as_of_date, regime, confidence, benchmark_symbol,
            metrics_json, reasons_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            analysis_run_id,
            as_of_date,
            market_regime,
            market_regime_confidence,
            benchmark_symbol,
            canonical_json(market_regime_metrics or {}),
            canonical_json(market_regime_reasons or []),
            now,
        ),
    )
    if analysis_scope == "candidate_universe" and status == "completed" and not symbols_blocked:
        prior = conn.execute("SELECT analysis_run_id FROM analysis_runs WHERE is_canonical=1 AND as_of_date=? AND analysis_run_id<>?", (as_of_date,analysis_run_id)).fetchone()
        conn.execute("UPDATE analysis_runs SET is_canonical=0 WHERE is_canonical=1 AND as_of_date=?", (as_of_date,))
        conn.execute("UPDATE analysis_runs SET is_canonical=1,supersedes_run_id=? WHERE analysis_run_id=?", (str(prior[0]) if prior else None,analysis_run_id))
