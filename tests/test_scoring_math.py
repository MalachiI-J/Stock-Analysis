from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from stock_scrapper.analysis.confidence_score import calculate_confidence_score
from stock_scrapper.analysis.opportunity_score import calculate_opportunity_score
from stock_scrapper.analysis.risk_score import calculate_risk_score
from stock_scrapper.analysis.scoring_config import validate_scoring_config


def _project_rules() -> dict:
    path = Path(__file__).resolve().parents[1] / "config" / "scoring_rules.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_actual_project_scoring_yaml_is_canonical_and_valid() -> None:
    rules = _project_rules()
    assert validate_scoring_config(rules) is rules
    assert set(rules["opportunity_weights"]) == {
        "long_term_trend",
        "multi_period_momentum",
        "relative_strength",
        "trend_quality",
        "volume_participation",
        "breakout_positioning",
    }
    assert "quality" not in rules["opportunity_weights"]
    assert "valuation" not in rules["opportunity_weights"]


@pytest.mark.parametrize("bad_value", [True, "20", -1.0, float("inf")])
def test_weight_validation_rejects_non_numeric_boolean_negative_and_nonfinite(
    bad_value: object,
) -> None:
    rules = _project_rules()
    rules["opportunity_weights"]["long_term_trend"] = bad_value
    with pytest.raises(ValueError):
        validate_scoring_config(rules)


def test_weight_validation_rejects_missing_unknown_and_non_100_total() -> None:
    rules = _project_rules()
    missing = deepcopy(rules)
    del missing["confidence_weights"]["signal_agreement"]
    with pytest.raises(ValueError):
        validate_scoring_config(missing)

    unknown = deepcopy(rules)
    unknown["risk_weights"]["placeholder"] = unknown["risk_weights"].pop(
        "beta_sensitivity"
    )
    with pytest.raises(ValueError):
        validate_scoring_config(unknown)

    wrong_total = deepcopy(rules)
    wrong_total["opportunity_weights"]["long_term_trend"] += 1
    with pytest.raises(ValueError):
        validate_scoring_config(wrong_total)


def test_candidate_score_is_mathematically_achievable() -> None:
    rules = _project_rules()
    metrics = {
        f"{component}_score": 70.0 for component in rules["opportunity_weights"]
    }
    score, _, components, _ = calculate_opportunity_score(metrics, rules, "Neutral")

    assert score == 70.0
    assert rules["score_thresholds"]["candidate"] <= score
    assert score < rules["score_thresholds"]["strong_candidate"]
    assert all(component["available"] for component in components.values())


def test_strong_candidate_score_is_achievable_from_raw_technical_metrics() -> None:
    rules = _project_rules()
    metrics = {
        "latest_close": 150.0,
        "fifty_day_sma": 130.0,
        "two_hundred_day_sma": 100.0,
        "trend_slope_50": 0.08,
        "trend_slope_200": 0.12,
        "time_above_sma50": 1.0,
        "time_above_sma200": 1.0,
        "one_day_return": 0.02,
        "one_month_return": 0.10,
        "three_month_return": 0.20,
        "six_month_return": 0.30,
        "one_year_return": 0.40,
        "benchmark_relative_return_21": 0.05,
        "benchmark_relative_return_63": 0.10,
        "benchmark_relative_return_126": 0.15,
        "benchmark_relative_return_252": 0.25,
        "relative_strength_trend": 0.10,
        "volume_relative_to_average": 2.0,
        "distance_from_52_week_high": 0.0,
    }
    score, _, _, _ = calculate_opportunity_score(metrics, rules, "Risk-On")

    assert score == 100.0
    assert score >= rules["score_thresholds"]["strong_candidate"]


def test_opportunity_missing_component_is_explicit_and_blocks_score() -> None:
    rules = _project_rules()
    metrics = {
        f"{component}_score": 70.0 for component in rules["opportunity_weights"]
    }
    del metrics["relative_strength_score"]
    score, level, components, reasons = calculate_opportunity_score(
        metrics, rules, "Neutral"
    )
    assert score is None
    assert level == "Unavailable"
    assert components["relative_strength"]["available"] is False
    assert components["relative_strength"]["value"] is None
    assert "relative_strength" in reasons[0]


def test_risk_missingness_and_critical_quality_are_blocking() -> None:
    rules = _project_rules()
    metrics = {f"{component}_score": 40.0 for component in rules["risk_weights"]}
    score, _, components, _ = calculate_risk_score(metrics, rules, "Neutral", [])
    assert score is not None
    assert all(component["available"] for component in components.values())

    missing = dict(metrics)
    del missing["beta_sensitivity_score"]
    score, level, components, reasons = calculate_risk_score(
        missing, rules, "Neutral", []
    )
    assert score is None
    assert level == "Unavailable"
    assert components["beta_sensitivity"]["available"] is False
    assert any("beta_sensitivity" in reason for reason in reasons)

    score, _, components, reasons = calculate_risk_score(
        metrics,
        rules,
        "Neutral",
        [{"severity": "critical", "issue_type": "missing_close"}],
    )
    assert score is None
    assert components["data_quality_risk"]["value"] == 100.0
    assert any("Critical unresolved" in reason for reason in reasons)


def test_noncritical_missing_risk_component_is_explicitly_reweighted() -> None:
    rules = _project_rules()
    metrics = {f"{component}_score": 40.0 for component in rules["risk_weights"]}
    del metrics["trend_deterioration_score"]
    score, _, components, reasons = calculate_risk_score(metrics, rules, "Neutral", [])

    assert score is not None
    assert components["trend_deterioration"]["available"] is False
    assert components["trend_deterioration"]["contribution"] is None
    effective_total = sum(
        component["effective_weight"]
        for component in components.values()
        if component["available"]
    )
    assert effective_total == pytest.approx(100.0)
    assert any("trend_deterioration" in reason for reason in reasons)


def test_confidence_freshness_and_missing_context_reduce_score_explicitly() -> None:
    rules = _project_rules()
    base_metrics = {
        "latest_close": 100.0,
        "latest_trading_date": "2024-12-31",
        "history_completeness_score": 100.0,
        "benchmark_alignment_score": 100.0,
        "indicator_availability_score": 100.0,
        "market_context_availability_score": 100.0,
        "signal_agreement_score": 100.0,
    }
    score, _, components, _ = calculate_confidence_score(
        base_metrics, rules, [], as_of_date="2024-12-31"
    )
    assert score == 100.0
    assert components["data_freshness"]["value"] == 100.0

    stale_score, _, stale_components, limitations = calculate_confidence_score(
        base_metrics, rules, [], as_of_date="2025-01-10"
    )
    assert stale_score == 85.0
    assert stale_components["data_freshness"]["value"] == 0.0
    assert any("stale" in limitation.lower() for limitation in limitations)

    no_context = dict(base_metrics)
    del no_context["market_context_availability_score"]
    context_score, _, context_components, limitations = calculate_confidence_score(
        no_context, rules, [], as_of_date="2024-12-31"
    )
    assert context_score == 90.0
    assert context_components["market_context_availability"] == {
        "value": None,
        "available": False,
        "weight": 10.0,
        "contribution": 0.0,
    }
    assert any("Market Context Availability" in item for item in limitations)
