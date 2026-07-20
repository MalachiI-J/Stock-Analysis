"""Deterministic risk scoring for Phase 2."""

from __future__ import annotations

from typing import Any


def calculate_risk_score(metrics: dict[str, Any], rules: dict[str, Any], market_regime: str, quality_issues: list[dict[str, Any]]) -> tuple[float | None, str, dict[str, Any], list[str]]:
    """Calculate a transparent risk score from 0 to 100 using actual market metrics."""
    if not metrics or metrics.get("latest_close") is None:
        return None, "Unavailable", {}, ["Missing price data"]

    weights = rules.get("risk_weights", {})
    if sum(weights.values()) != 100:
        raise ValueError("Risk weights must total 100")

    volatility = metrics.get("twenty_day_volatility") or 0.0
    volatility_score = min(100.0, max(0.0, volatility * 100.0))

    drawdown = metrics.get("full_history_max_drawdown") or metrics.get("max_drawdown") or 0.0
    drawdown_score = min(100.0, max(0.0, drawdown * 100.0))

    downside = metrics.get("sixty_day_downside_volatility") or 0.0
    downside_score = min(100.0, max(0.0, downside * 100.0))

    atr = metrics.get("atr_percentage") or 0.0
    atr_score = min(100.0, max(0.0, atr * 100.0))

    beta = metrics.get("beta") or 0.0
    beta_score = min(100.0, max(0.0, beta * 25.0))

    trend_deterioration = metrics.get("trend_deterioration") or 0.0
    trend_deterioration_score = min(100.0, max(0.0, trend_deterioration * 100.0))

    liquidity_score = 0.0
    if metrics.get("volume_relative_to_average") is not None:
        liquidity_score = max(0.0, 100.0 - (metrics["volume_relative_to_average"] * 100.0))

    market_regime_penalty = 0.0 if market_regime in {"Risk-On", "Neutral"} else 25.0
    quality_penalty = 10.0 if quality_issues else 0.0

    components = {
        "realized_volatility": volatility_score,
        "drawdown_risk": drawdown_score,
        "downside_volatility": downside_score,
        "atr_gap_risk": atr_score,
        "beta_sensitivity": beta_score,
        "trend_deterioration": trend_deterioration_score,
        "liquidity_risk": liquidity_score,
        "market_regime_risk": market_regime_penalty,
        "data_quality_risk": quality_penalty,
    }

    weighted = (
        components["realized_volatility"] * weights.get("realized_volatility", 0) / 100.0
        + components["drawdown_risk"] * weights.get("drawdown_risk", 0) / 100.0
        + components["downside_volatility"] * weights.get("downside_volatility", 0) / 100.0
        + components["atr_gap_risk"] * weights.get("atr_gap_risk", 0) / 100.0
        + components["beta_sensitivity"] * weights.get("beta_sensitivity", 0) / 100.0
        + components["trend_deterioration"] * weights.get("trend_deterioration", 0) / 100.0
        + components["liquidity_risk"] * weights.get("liquidity_risk", 0) / 100.0
        + components["market_regime_risk"] * weights.get("market_regime_risk", 0) / 100.0
        + components["data_quality_risk"] * weights.get("data_quality_risk", 0) / 100.0
    )

    score = round(min(100.0, max(0.0, weighted)), 2)
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
    return score, level, components, ["Volatility and downside risk were weighted into the score"]
