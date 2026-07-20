from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from stock_scrapper.models.analysis_models import AnalysisResult
from stock_scrapper.reporting.report_builder import write_phase2_reports


def _history(start: date, count: int, base_price: float) -> list[dict[str, object]]:
    return [
        {
            "trade_date": (start + timedelta(days=index)).isoformat(),
            "adjusted_close": base_price + index * 0.5,
            "close": base_price + index * 0.55,
        }
        for index in range(count)
    ]


def _report_inputs() -> tuple[
    dict[str, object],
    list[AnalysisResult | dict[str, object]],
    dict[str, list[dict[str, object]]],
    list[dict[str, object]],
    dict[str, dict[str, object]],
]:
    metadata: dict[str, object] = {
        "analysis_run_id": "analysis-test-001",
        "as_of_date": "2024-12-31",
        "data_through_date": "2024-08-07",
        "scoring_version": "phase2-v2",
        "configuration_hash": "a" * 64,
        "benchmark_symbol": "SPY",
        "market_regime": "Risk-On",
        "market_regime_confidence": 88.5,
        "market_regime_reasons": ["SPY is above its 200-day average", "Breadth is constructive"],
        "generated_at": "2025-01-01T00:00:00+00:00",
    }
    aapl = AnalysisResult(
        symbol="AAPL",
        as_of_date="2024-12-31",
        data_through_date="2024-08-07",
        market_regime="Risk-On",
        market_regime_confidence=88.5,
        risk_score=20.0,
        risk_level="Low",
        opportunity_score=90.0,
        confidence_score=95.0,
        classification="Strong Candidate",
        primary_reason="Trend and relative strength are aligned",
        eligible_for_scoring=True,
        risk_components={"realized_volatility": {"score": 10.0, "weight": 20}},
        opportunity_components={"long_term_trend": {"score": 95.0, "weight": 25}},
        confidence_components={"history_completeness": {"score": 100.0, "weight": 30}},
        indicators={
            "sma_200": 175.0,
            "relative_strength_63": 0.12,
            "history": [{"trade_date": "2024-01-01", "adjusted_close": 100.0}],
        },
        flags=["Near 52-week high"],
        positive_factors=["Long-term trend is positive"],
        risk_factors=["Short-term volatility is elevated"],
        confidence_limitations=["One benchmark session was unavailable"],
        quality_concerns=["Adjusted close was reviewed"],
        market_regime_effects=["Risk-On supports candidate eligibility"],
        improvement_conditions=["Broader volume participation"],
        weakening_conditions=["Close below the 200-day average"],
        trend_state="Uptrend",
    )
    tsla: dict[str, object] = {
        "symbol": "TSLA",
        "as_of_date": "2024-12-31",
        "data_through_date": "2024-08-07",
        "market_regime": "Risk-On",
        "risk_score": 91.0,
        "risk_level": "Very High",
        "opportunity_score": 35.0,
        "confidence_score": 72.0,
        "classification": "High Risk",
        "primary_reason": "Measured risk exceeded the configured threshold",
        "eligible_for_scoring": True,
        "risk_components": {"drawdown": {"score": 95.0, "weight": 20}},
        "opportunity_components_json": '{"long_term_trend":{"score":35.0,"weight":25}}',
        "confidence_components": {"history_completeness": {"score": 100.0, "weight": 30}},
        "indicators": {"sma_200": 230.0},
        "positive_factors": ["History is complete"],
        "risk_factors": ["Large one-year drawdown"],
        "confidence_limitations": [],
        "quality_concerns": [],
        "market_regime_effects": ["Risk-On does not override symbol risk"],
        "improvement_conditions": ["Risk score below 70"],
        "weakening_conditions": ["Further drawdown"],
        "trend_state": "Mixed Trend",
    }
    spy: dict[str, object] = {
        "symbol": "SPY",
        "as_of_date": "2024-12-31",
        "data_through_date": "2024-08-07",
        "risk_score": 30.0,
        "risk_level": "Moderate",
        "opportunity_score": 58.0,
        "confidence_score": 90.0,
        "classification": "Watch",
        "primary_reason": "Benchmark trend remains constructive",
        "eligible_for_scoring": True,
    }
    histories = {
        "AAPL": _history(date(2024, 1, 1), 220, 100.0)
        + [{"trade_date": "2025-01-02", "adjusted_close": 9999.0, "close": 9999.0}],
        "TSLA": _history(date(2024, 1, 1), 220, 200.0),
        "SPY": _history(date(2024, 1, 1), 220, 400.0),
    }
    quality_issues = [
        {
            "symbol": "AAPL",
            "trade_date": "2024-06-03",
            "issue_type": "adjusted_close_review",
            "severity": "warning",
            "description": "Reviewed <adjusted close>",
        },
        {
            "symbol": "AAPL",
            "trade_date": "2025-01-02",
            "issue_type": "future_issue",
            "severity": "critical",
            "description": "Future issue must not leak",
        },
    ]
    previous = {
        "AAPL": {
            "symbol": "AAPL",
            "classification": "Candidate",
            "risk_score": 25.0,
            "opportunity_score": 80.0,
            "confidence_score": 90.0,
        },
        "TSLA": {
            "symbol": "TSLA",
            "classification": "High Risk",
            "risk_score": 88.0,
            "opportunity_score": 40.0,
            "confidence_score": 75.0,
        },
    }
    return metadata, [aapl, tsla, spy], histories, quality_issues, previous


def test_phase2_report_contains_complete_offline_research_content(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()

    paths = write_phase2_reports(
        tmp_path,
        "2024-12-31",
        metadata,
        results,
        histories,
        issues,
        previous_results=previous,
    )

    assert set(paths) == {"csv", "html"}
    assert paths["csv"].name == "stock_summary_2024-12-31.csv"
    assert paths["html"].name == "stock_summary_2024-12-31.html"

    with paths["csv"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_symbol = {row["symbol"]: row for row in rows}
    assert len(rows) == 3
    assert "history" not in rows[0]
    assert by_symbol["AAPL"]["candidate_rank"] == "1"
    assert by_symbol["TSLA"]["risk_rank"] == "1"
    assert json.loads(by_symbol["AAPL"]["risk_components"])["realized_volatility"]["weight"] == 20
    assert "history" not in json.loads(by_symbol["AAPL"]["indicators"])
    assert json.loads(by_symbol["TSLA"]["opportunity_components"])["long_term_trend"]["score"] == 35.0
    assert float(by_symbol["AAPL"]["opportunity_score_change"]) == 10.0
    assert by_symbol["AAPL"]["classification_changed"] == "True"
    assert "9999" not in paths["csv"].read_text(encoding="utf-8")

    content = paths["html"].read_text(encoding="utf-8")
    for expected in (
        "Run Metadata",
        "As-of date",
        "Data-through date",
        "Scoring version",
        "Configuration hash",
        "Market-regime reasons",
        "Candidate Ranking",
        "Highest-Risk Ranking",
        "Risk components",
        "Opportunity components",
        "Confidence components",
        "Positive factors",
        "Risk factors",
        "Confidence limitations",
        "Data-quality concerns",
        "Improvement conditions",
        "Weakening conditions",
        "Changes From Previous Stored Analysis",
        "Methodology",
        "Research Disclaimer",
        "Historical analysis does not guarantee future performance",
        "SPY is above its 200-day average",
        "Long-term trend is positive",
    ):
        assert expected in content
    for series in ("adjusted-price", "sma-20", "sma-50", "sma-200"):
        assert f'data-series="{series}"' in content
    assert "<svg" in content
    assert "<script" not in content.lower()
    assert "http://" not in content.lower()
    assert "https://" not in content.lower()
    assert "Reviewed &lt;adjusted close&gt;" in content
    assert "Future issue must not leak" not in content
    assert "9999" not in content


def test_phase2_report_is_deterministic_when_generation_metadata_is_fixed(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()

    first = write_phase2_reports(tmp_path / "first", "2024-12-31", metadata, results, histories, issues, previous)
    second = write_phase2_reports(tmp_path / "second", "2024-12-31", metadata, results, histories, issues, previous)

    assert first["csv"].read_bytes() == second["csv"].read_bytes()
    assert first["html"].read_bytes() == second["html"].read_bytes()


def test_phase2_report_bounds_charts_quality_issues_and_data_date_by_as_of_date(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()
    metadata["as_of_date"] = "2024-06-30"
    histories["AAPL"].append(
        {"trade_date": "2024-07-01", "adjusted_close": 7777.0, "close": 7777.0}
    )
    issues.append(
        {
            "symbol": "AAPL",
            "trade_date": "2024-07-01",
            "issue_type": "later_issue",
            "severity": "warning",
            "description": "Later as-of issue",
        }
    )

    paths = write_phase2_reports(
        tmp_path,
        "2024-12-31",
        metadata,
        results,
        histories,
        issues,
        previous,
    )

    content = paths["html"].read_text(encoding="utf-8")
    assert "7777" not in content
    assert "Later as-of issue" not in content
    with paths["csv"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["data_through_date"] <= "2024-06-30" for row in rows)


def test_phase2_report_rejects_invalid_report_date_before_writing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="report_date must be a valid YYYY-MM-DD date"):
        write_phase2_reports(tmp_path, "not-a-date", {}, [], {}, [])

    assert list(tmp_path.iterdir()) == []


def test_symbol_without_history_does_not_inherit_another_symbols_data_date(
    tmp_path: Path,
) -> None:
    metadata, results, histories, issues, previous = _report_inputs()
    results.append(
        {
            "symbol": "MISSING",
            "as_of_date": "2024-12-31",
            "data_through_date": None,
            "classification": "Insufficient Data",
            "eligible_for_scoring": False,
            "primary_reason": "No usable price history",
        }
    )
    histories["MISSING"] = []

    paths = write_phase2_reports(
        tmp_path,
        "2024-12-31",
        metadata,
        results,
        histories,
        issues,
        previous,
    )

    with paths["csv"].open("r", encoding="utf-8", newline="") as handle:
        by_symbol = {row["symbol"]: row for row in csv.DictReader(handle)}
    assert by_symbol["MISSING"]["data_through_date"] == ""
