"""Deterministic Phase 2 analysis orchestration."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any

from stock_scrapper.analysis.confidence_score import calculate_confidence_score
from stock_scrapper.analysis.eligibility import evaluate_eligibility
from stock_scrapper.analysis.market_regime import calculate_market_regime
from stock_scrapper.analysis.opportunity_score import calculate_opportunity_score
from stock_scrapper.analysis.risk_score import calculate_risk_score
from stock_scrapper.models.analysis_models import AnalysisResult
from stock_scrapper.processing.indicators import calculate_indicators


def _coerce_date(value: str | date | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def _normalize_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in history:
        normalized.append(
            {
                "trade_date": row.get("trade_date"),
                "close": row.get("close"),
                "adjusted_close": row.get("adjusted_close"),
                "volume": row.get("volume"),
            }
        )
    return normalized


def _normalize_return(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, (value + 1.0) / 2.0))


def _build_metrics(
    metrics: dict[str, Any],
    history: list[dict[str, Any]],
    quality_issues: list[dict[str, Any]],
    benchmark_history: list[dict[str, Any]],
) -> dict[str, Any]:
    enriched = dict(metrics)
    enriched["history_length"] = len(history)
    enriched["quality_issue_count"] = len(quality_issues)
    enriched["latest_close"] = metrics.get("latest_close")
    enriched["distance_from_sma50"] = metrics.get("distance_from_sma50") or 0.0
    enriched["distance_from_sma200"] = metrics.get("distance_from_sma200") or 0.0
    enriched["beta"] = 0.35 if benchmark_history else 0.0
    enriched["downside_volatility"] = metrics.get("twenty_day_volatility") or 0.0
    enriched["atr_percentage"] = max(0.0, min(1.0, abs((metrics.get("one_day_return") or 0.0)) + 0.05))
    enriched["trend_deterioration"] = max(0.0, min(1.0, abs(enriched.get("distance_from_sma50", 0.0)) / 100.0 + 0.05))
    enriched["momentum_score"] = _normalize_return(metrics.get("one_year_return"))
    enriched["trend_strength"] = _normalize_return(metrics.get("six_month_return"))
    enriched["relative_strength"] = _normalize_return(metrics.get("one_month_return"))
    enriched["quality_score"] = 0.8 if not quality_issues else 0.45
    enriched["valuation_score"] = 0.65 if (metrics.get("distance_below_52_week_high") is None or metrics.get("distance_below_52_week_high") <= 10) else 0.4
    return enriched


def analyze_symbol(
    symbol: str,
    history: list[dict[str, Any]],
    benchmark_history: list[dict[str, Any]],
    quality_issues: list[dict[str, Any]],
    as_of_date: str | date,
    rules: dict[str, Any],
    minimum_history_days: int,
    minimum_recent_days: int = 20,
) -> AnalysisResult:
    """Run deterministic scoring and build a transparent analysis summary."""
    normalized_history = _normalize_history(history)
    base_metrics = calculate_indicators(normalized_history, symbol)
    enriched_metrics = _build_metrics(base_metrics, normalized_history, quality_issues, benchmark_history)
    as_of = _coerce_date(as_of_date) or date.today()

    eligible, blocking_reasons, eligibility_meta = evaluate_eligibility(
        symbol=symbol,
        history=normalized_history,
        quality_issues=quality_issues,
        as_of_date=as_of,
        minimum_history_days=minimum_history_days,
    )

    regime, regime_confidence, regime_components, regime_reasons = calculate_market_regime(
        benchmark_history=benchmark_history,
        context_histories={},
        breadth_ratio=0.6 if len(normalized_history) >= minimum_recent_days else 0.3,
        rules=rules,
    )

    risk_score = None
    risk_level = "Unavailable"
    risk_components: dict[str, Any] = {}
    risk_reasons: list[str] = []
    opportunity_score = None
    opportunity_level = "Unavailable"
    opportunity_components: dict[str, Any] = {}
    opportunity_reasons: list[str] = []
    confidence_score = None
    confidence_level = "Unavailable"
    confidence_components: dict[str, Any] = {}
    confidence_reasons: list[str] = []

    if eligible:
        risk_score, risk_level, risk_components, risk_reasons = calculate_risk_score(
            enriched_metrics,
            rules,
            regime,
            quality_issues,
        )
        opportunity_score, opportunity_level, opportunity_components, opportunity_reasons = calculate_opportunity_score(
            enriched_metrics,
            rules,
            regime,
        )
        confidence_score, confidence_level, confidence_components, confidence_reasons = calculate_confidence_score(
            enriched_metrics,
            rules,
            quality_issues,
        )

    classification = "Avoid"
    if not eligible:
        classification = "Avoid"
    elif risk_score is not None and opportunity_score is not None:
        thresholds = rules.get("score_thresholds", {})
        strong_candidate = thresholds.get("strong_candidate", 75)
        candidate = thresholds.get("candidate", 65)
        watch = thresholds.get("watch", 50)
        high_risk = thresholds.get("high_risk", 70)
        if opportunity_score >= strong_candidate and risk_score <= high_risk:
            classification = "Strong Candidate"
        elif opportunity_score >= candidate and risk_score <= high_risk:
            classification = "Candidate"
        elif opportunity_score >= watch:
            classification = "Watch"
        else:
            classification = "Avoid"

    if not eligible:
        primary_reason = "; ".join(blocking_reasons) if blocking_reasons else "Insufficient history"
    elif risk_score is not None and risk_score >= 70:
        primary_reason = "Risk profile exceeded the preferred threshold"
    elif opportunity_score is not None and opportunity_score >= 70:
        primary_reason = "Momentum and trend alignment support the setup"
    else:
        primary_reason = "The setup is mixed and should be monitored"

    flags: list[str] = []
    if quality_issues:
        flags.append("Quality issues present")
    if regime == "Risk-Off":
        flags.append("Market regime is defensive")
    if not eligible:
        flags.append("Not eligible for scoring")

    return AnalysisResult(
        symbol=symbol,
        as_of_date=as_of.strftime("%Y-%m-%d"),
        data_through_date=(normalized_history[-1].get("trade_date") if normalized_history else None),
        market_regime=regime,
        market_regime_confidence=regime_confidence,
        risk_score=risk_score,
        risk_level=risk_level,
        opportunity_score=opportunity_score,
        confidence_score=confidence_score,
        classification=classification,
        primary_reason=primary_reason,
        eligible_for_scoring=eligible,
        blocking_reasons=blocking_reasons,
        risk_components=risk_components,
        opportunity_components=opportunity_components,
        confidence_components=confidence_components,
        indicators=enriched_metrics,
        flags=flags,
        positive_factors=regime_components + (["History length supported the analysis"] if eligible else []),
        risk_factors=risk_reasons,
        confidence_limitations=confidence_reasons,
        quality_concerns=[issue.get("description", "") for issue in quality_issues],
        market_regime_effects=regime_reasons,
        improvement_conditions=["Add more price history", "Resolve data quality issues"],
        weakening_conditions=["Risk-Off market regime", "High drawdown profile"],
        trend_state=base_metrics.get("status") if isinstance(base_metrics, dict) else "Unknown",
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
) -> None:
    """Persist analysis runs and per-symbol results into the database."""
    conn.execute(
        """
        INSERT INTO analysis_runs (
            analysis_run_id, started_at, completed_at, as_of_date, data_through_date, benchmark_symbol,
            market_regime, market_regime_confidence, symbols_requested, symbols_analyzed, symbols_blocked,
            status, scoring_version, configuration_hash, error_summary
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            analysis_run_id,
            datetime.now(timezone.utc).isoformat(),
            datetime.now(timezone.utc).isoformat(),
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
        ),
    )

    for result in results:
        conn.execute(
            """
            INSERT INTO stock_analysis (
                analysis_run_id, symbol, as_of_date, data_through_date, risk_score, opportunity_score,
                confidence_score, classification, primary_reason, risk_level, trend_state,
                eligible_for_scoring, blocking_reasons_json, risk_components_json, opportunity_components_json,
                confidence_components_json, indicators_json, flags_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                1 if result.eligible_for_scoring else 0,
                json.dumps(result.blocking_reasons),
                json.dumps(result.risk_components),
                json.dumps(result.opportunity_components),
                json.dumps(result.confidence_components),
                json.dumps(result.indicators),
                json.dumps(result.flags),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
