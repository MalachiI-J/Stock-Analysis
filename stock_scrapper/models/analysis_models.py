"""Typed analysis result models for Phase 2 scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnalysisResult:
    """A deterministic analysis result for one symbol at one as-of date."""

    symbol: str
    as_of_date: str
    data_through_date: str | None = None
    market_regime: str = "Insufficient Market Data"
    market_regime_confidence: float | None = None
    risk_score: float | None = None
    risk_level: str = "Unavailable"
    opportunity_score: float | None = None
    confidence_score: float | None = None
    classification: str = "Insufficient Data"
    primary_reason: str = "Insufficient history"
    eligible_for_scoring: bool = False
    blocking_reasons: list[str] = field(default_factory=list)
    risk_components: dict[str, Any] = field(default_factory=dict)
    opportunity_components: dict[str, Any] = field(default_factory=dict)
    confidence_components: dict[str, Any] = field(default_factory=dict)
    indicators: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    positive_factors: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    confidence_limitations: list[str] = field(default_factory=list)
    quality_concerns: list[str] = field(default_factory=list)
    market_regime_effects: list[str] = field(default_factory=list)
    improvement_conditions: list[str] = field(default_factory=list)
    weakening_conditions: list[str] = field(default_factory=list)
    trend_state: str = "Unknown"
