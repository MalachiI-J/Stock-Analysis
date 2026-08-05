from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import unquote

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
        "Stock Analyzer",
        "vs <span class=\"mono\">SPY</span> benchmark",
        "Run details",
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
        "SPY is above its 200-day average",
        "Long-term trend is positive",
    ):
        assert expected in content
    assert "Research Disclaimer" not in content
    assert "Run Metadata" not in content
    assert "Stock Scrapper Phase 2" not in content
    # Methodology now lives inside the run-details disclosure, not its own section.
    assert "<h5>Methodology</h5>" in content
    assert content.index("<h5>Methodology</h5>") > content.index('id="run-details"')
    for series in ("adjusted-price", "sma-20", "sma-50", "sma-200"):
        assert f'data-series="{series}"' in content
    assert "<svg" in content
    # Exactly two <script> tags are allowed, both narrow and data-free: the
    # theme-preference resolver/toggle (head, before the hero) and the
    # decorative hero animation. Nowhere else may ever have a <script>.
    head_end = content.index("</head>")
    hero_start = content.index('<div class="market-hero"')
    hero_end = content.index('<div class="page">')
    assert content.lower().count("<script") == 2
    assert "<script" in content[:head_end].lower()
    assert "<script" in content[hero_start:hero_end].lower()
    assert content[head_end:hero_start].lower().count("<script") == 0
    assert content[hero_end:].lower().count("<script") == 0
    assert "http://" not in content.lower()
    assert "https://" not in content.lower()
    assert "Reviewed &lt;adjusted close&gt;" in content
    assert "Future issue must not leak" not in content
    assert "9999" not in content


def test_phase2_report_includes_inline_svg_favicon(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    head = content.split("</head>")[0]
    assert '<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,' in head
    href = head.split('href="')[1].split('"')[0]
    decoded = unquote(href)
    assert decoded.startswith("data:image/svg+xml,<svg ")
    assert "#5CF08A" in decoded
    assert "<rect" in decoded and "<polyline" in decoded


def _signal_validation_payload(**bucket_overrides: dict[str, object]) -> dict[str, object]:
    strong_candidate = {
        "classification": "Strong Candidate",
        "sample_size": 4200,
        "distinct_symbols": 20,
        "hit_rate": 0.55,
        "mean_excess_return": 0.0099,
        "symbol_mean_excess_return": 0.0099,
        "symbol_mean_excess_return_ci_low": 0.0001,
        "symbol_mean_excess_return_ci_high": 0.0142,
        "concentration_warning": False,
    }
    strong_candidate.update(bucket_overrides.get("Strong Candidate", {}))
    high_risk = {
        "classification": "High Risk",
        "sample_size": 3100,
        "distinct_symbols": 20,
        "hit_rate": 0.61,
        "mean_excess_return": 0.0324,
        "symbol_mean_excess_return": 0.0324,
        "symbol_mean_excess_return_ci_low": 0.0182,
        "symbol_mean_excess_return_ci_high": 0.0448,
        "concentration_warning": False,
    }
    high_risk.update(bucket_overrides.get("High Risk", {}))
    return {
        "backtest_run_id": "backtest-test-signalvalidation-001",
        "horizon_days": 21,
        "benchmark_symbol": "SPY",
        "buckets": [strong_candidate, high_risk],
    }


def test_phase2_report_omits_signal_validation_notice_when_no_artifact_present(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    assert "Historical signal validation" not in content


def test_phase2_report_surfaces_latest_signal_validation_summary(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()
    (tmp_path / "signal_validation_backtest-test-signalvalidation-001.json").write_text(
        json.dumps(_signal_validation_payload()), encoding="utf-8",
    )

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    assert "Historical signal validation (Strong Candidate)" in content
    assert "Historical signal validation (High Risk)" in content
    assert "+0.99%" in content
    assert "+3.24%" in content
    assert "95% CI [+0.01%, +1.42%]" in content
    assert "95% CI [+1.82%, +4.48%]" in content
    assert "backtest-test-signalvalidation-001" in content
    assert "not a live prediction for the symbols ranked below, and not a recommendation" in content
    # Placed near each ranking section, not just anywhere in the document.
    assert content.index("Candidate Ranking") < content.index("Historical signal validation (Strong Candidate)") < content.index("Highest-Risk Ranking")
    assert content.index("Highest-Risk Ranking") < content.index("Historical signal validation (High Risk)")


def test_phase2_report_signal_validation_notice_flags_concentration_warning(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()
    payload = _signal_validation_payload(**{"High Risk": {"concentration_warning": True, "distinct_symbols": 2}})
    (tmp_path / "signal_validation_backtest-test-signalvalidation-002.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    assert "treat it as anecdote, not a broad pattern" in content


def test_phase2_report_ignores_unparseable_signal_validation_artifact(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()
    (tmp_path / "signal_validation_broken.json").write_text("not json", encoding="utf-8")

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    assert "Historical signal validation" not in content


def _recommendations_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "as_of_date": "2024-12-31",
        "account_value": 10000.0,
        "available_cash": 8500.0,
        "open_position_count": 1,
        "recommendations": [
            {
                "symbol": "NVDA", "action": "BUY", "shares": 10.0, "estimated_dollars": 500.0,
                "reason": "Strong trend", "model_probability": 0.6,
                "predict_v5_excess_return": 0.234, "predict_v5_low_confidence": True,
            },
        ],
        "skipped": ["AAPL: no current price available"],
    }
    payload.update(overrides)
    return payload


def test_phase2_report_nav_bar_links_to_recommendations(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    assert '<a href="#recommendations">Recommendations</a>' in content
    assert '<h2 id="recommendations">' in content


def test_phase2_report_shows_placeholder_when_no_recommendations_artifact_present(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    assert "run <span class=\"mono\">python main.py recommend</span> first" in content


def test_phase2_report_surfaces_latest_recommendations_summary(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()
    (tmp_path / "recommendations_2024-12-31.summary.json").write_text(
        json.dumps(_recommendations_payload()), encoding="utf-8",
    )

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    assert "NVDA" in content
    assert "model 60% beats benchmark" in content
    assert "predict-v5 +23.4%" in content
    assert "LOW CONFIDENCE" in content
    assert "AAPL: no current price available" in content
    assert "Account value $10,000.00" in content
    # Placed right after Market Regime, before Candidate Ranking.
    assert content.index("Market Regime") < content.index("Today's Recommendations") < content.index("Candidate Ranking")


def test_phase2_report_uses_recommendations_matching_this_exact_report_date(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()
    (tmp_path / "recommendations_2024-12-30.summary.json").write_text(
        json.dumps(_recommendations_payload(as_of_date="2024-12-30")), encoding="utf-8",
    )

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    assert "NVDA" not in content.split("Today's Recommendations")[1].split("Candidate Ranking")[0]


def test_phase2_report_ignores_unparseable_recommendations_artifact(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()
    (tmp_path / "recommendations_2024-12-31.summary.json").write_text("not json", encoding="utf-8")

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    assert "run <span class=\"mono\">python main.py recommend</span> first" in content


def test_phase2_report_recommendations_render_as_cards_not_a_table(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()
    (tmp_path / "recommendations_2024-12-31.summary.json").write_text(
        json.dumps(_recommendations_payload()), encoding="utf-8",
    )

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    section = content.split("Today's Recommendations")[1].split("Candidate Ranking")[0]
    assert '<div class="rec-list">' in section
    assert 'class="rec-card"' in section
    assert "<table>" not in section
    assert "NVDA" in section.split('<span class="mono rec-symbol">')[1].split("</span>")[0]
    assert "10 sh · $500.00" in section


def test_phase2_report_recommendations_context_line_colored_by_predict_v5_sign(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()
    payload = _recommendations_payload(recommendations=[
        {
            "symbol": "NVDA", "action": "BUY", "shares": 10.0, "estimated_dollars": 500.0,
            "reason": "Strong trend", "predict_v5_excess_return": 0.05, "predict_v5_low_confidence": False,
        },
        {
            "symbol": "KO", "action": "BUY", "shares": 5.0, "estimated_dollars": 200.0,
            "reason": "Defensive pick", "predict_v5_excess_return": -0.03, "predict_v5_low_confidence": False,
        },
        {
            "symbol": "BAC", "action": "BUY", "shares": 8.0, "estimated_dollars": 300.0,
            "reason": "Financials", "model_probability": 0.55,
        },
    ])
    (tmp_path / "recommendations_2024-12-31.summary.json").write_text(json.dumps(payload), encoding="utf-8")

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    section = content.split("Today's Recommendations")[1].split("Candidate Ranking")[0]
    assert 'class="rec-context mono delta-good">predict-v5 +5.0%' in section
    assert 'class="rec-context mono delta-critical">predict-v5 -3.0%' in section
    assert 'class="rec-context mono delta-neutral">model 55% beats benchmark' in section


def test_phase2_report_includes_resize_widget_when_sizing_rules_are_present(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()
    payload = _recommendations_payload(
        cash_reserve=0.05, max_position_weight=0.15,
        max_trade_dollar_amount=2000.0, min_trade_dollar_amount=100.0,
    )
    (tmp_path / "recommendations_2024-12-31.summary.json").write_text(json.dumps(payload), encoding="utf-8")

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    section = content.split("Today's Recommendations")[1].split("Candidate Ranking")[0]
    assert 'id="rec-adjust-toggle"' in section
    assert 'id="rec-adjust-panel"' in section
    assert "cashReserve: 0.05" in section
    assert "maxPositionWeight: 0.15" in section
    assert "maxTradeDollarAmount: 2000.0" in section
    assert "minTradeDollarAmount: 100.0" in section
    assert 'data-price="50.000000"' in section  # 500.0 / 10.0
    assert 'class="mono rec-sizing-value" data-original="10 sh · $500.00"' in section
    assert 'id="rec-summary-line"' in section


def test_phase2_report_omits_resize_widget_when_sizing_rules_are_missing(tmp_path: Path) -> None:
    """Older recommendations_<date>.summary.json files predate the four sizing-rule
    fields -- the section must degrade to a plain, non-interactive subtitle rather
    than fail or render a broken widget."""
    metadata, results, histories, issues, previous = _report_inputs()
    (tmp_path / "recommendations_2024-12-31.summary.json").write_text(
        json.dumps(_recommendations_payload()), encoding="utf-8",
    )

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    section = content.split("Today's Recommendations")[1].split("Candidate Ranking")[0]
    assert 'id="rec-adjust-toggle"' not in section
    assert "Account value $10,000.00" in section


def test_phase2_report_omits_resize_widget_when_there_are_no_buys(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()
    payload = _recommendations_payload(
        recommendations=[], cash_reserve=0.05, max_position_weight=0.15,
        max_trade_dollar_amount=2000.0, min_trade_dollar_amount=100.0,
    )
    (tmp_path / "recommendations_2024-12-31.summary.json").write_text(json.dumps(payload), encoding="utf-8")

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    section = content.split("Today's Recommendations")[1].split("Candidate Ranking")[0]
    assert 'id="rec-adjust-toggle"' not in section


def test_phase2_report_ranking_tables_use_accent_edge_not_badge_pill(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    candidate_section = content.split('id="candidates"')[1].split('id="highest-risk"')[0]
    risk_section = content.split('id="highest-risk"')[1].split('id="changes"')[0]
    # AAPL is "Strong Candidate" (good) and ranked #1 in both tables' shared pool.
    assert 'data-status="good"' in candidate_section
    assert '<td class="status-good">Strong Candidate</td>' in candidate_section
    # TSLA is "High Risk" (critical).
    assert 'data-status="critical"' in risk_section
    assert '<td class="status-critical">High Risk</td>' in risk_section
    # No more filled badge pill for classification in either ranking table.
    assert "badge-good" not in candidate_section and "badge-critical" not in risk_section


def test_phase2_report_changes_table_classification_badges_are_unchanged(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    changes_section = content.split("Changes From Previous Stored Analysis")[1].split("Data-Quality Concerns")[0]
    assert 'class="badge badge-good">Strong Candidate' in changes_section
    assert "data-status" not in changes_section


def test_phase2_report_omits_data_quality_section_when_there_are_no_issues(tmp_path: Path) -> None:
    metadata, results, histories, _issues, previous = _report_inputs()

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, [], previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    assert "Data-Quality Concerns" not in content
    assert "No data-quality concerns were supplied" not in content


def test_phase2_report_symbol_card_merges_score_and_gauge_into_one_stat_box(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    aapl_card = content.split('id="symbol-AAPL"')[1].split('id="symbol-')[0]
    scores_block = aapl_card.split('<div class="scores">')[1].split('<h4>Adjusted Price')[0]
    # Opportunity/Measured risk each appear as one stat box with an inline gauge —
    # no separate full-width gauge section duplicating the same two numbers.
    assert "primary-gauges" not in aapl_card
    assert scores_block.count("Opportunity") == 1
    assert scores_block.count("Measured risk") == 1
    assert 'class="gauge gauge-good gauge-stat"' in scores_block
    # Candidate/Risk rank moved out of the .scores stat grid into a secondary line.
    assert "Candidate rank" not in aapl_card
    assert "Risk rank" not in aapl_card
    assert 'class="stock-rank-line muted">rank 1 of candidates' in aapl_card


def test_phase2_report_symbol_quickjump_strip_links_and_colors_by_classification(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    strip = content.split('id="symbols"')[1].split('id="symbol-AAPL"')[0]
    assert "Jump to symbol" in strip
    # AAPL is Strong Candidate (good), TSLA is High Risk (critical), SPY is Watch (warning).
    assert '<a class="symbol-chip symbol-chip-good" href="#symbol-AAPL">AAPL</a>' in strip
    assert '<a class="symbol-chip symbol-chip-critical" href="#symbol-TSLA">TSLA</a>' in strip
    assert '<a class="symbol-chip symbol-chip-warning" href="#symbol-SPY">SPY</a>' in strip
    # It's the very first thing under the heading, before any symbol card.
    assert content.index('id="symbols"') < content.index('<div class="quickjump-row">') < content.index('id="symbol-AAPL"')


def test_phase2_report_advisory_only_notice_is_a_single_sentence(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    assert "Sized suggestions only" not in content
    assert "See <span class=\"mono\">python main.py recommend</span>." not in content
    assert "python main.py recommend" in content


def test_phase2_report_signal_validation_notice_collapses_detail_by_default(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()
    (tmp_path / "signal_validation_backtest-test-signalvalidation-001.json").write_text(
        json.dumps(_signal_validation_payload()), encoding="utf-8",
    )

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    notice = content.split('<strong>Historical signal validation (Strong Candidate):</strong>')[1].split("</div>")[0]
    headline, _, detail = notice.partition("<details>")
    assert "mean symbol-weighted forward 21-session excess return vs SPY was +0.99%." in headline
    assert "classified instance(s)" not in headline
    assert "classified instance(s)" in detail


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


def test_phase2_report_shows_historical_outcome_range_with_dollar_equivalent(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()
    payload = _recommendations_payload(recommendations=[
        {
            "symbol": "NVDA", "action": "BUY", "shares": 10.0, "estimated_dollars": 500.0,
            "reason": "Strong trend", "classification": "Strong Candidate",
            "outcome_p10": -0.03, "outcome_p90": 0.087,
            "best_exit_sessions": None, "best_exit_date": "2025-01-28", "checkpoints_compared": [],
        },
    ])
    (tmp_path / "recommendations_2024-12-31.summary.json").write_text(json.dumps(payload), encoding="utf-8")

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    section = content.split("Today's Recommendations")[1].split("Candidate Ranking")[0]
    assert "Historical pattern: -3.0% to +8.7% excess return" in section
    assert "-$15.00 to +$43.50 on this size" in section
    assert 'data-outcome-p10="-0.03"' in section
    assert 'data-outcome-p90="0.087"' in section
    assert 'class="rec-outcome mono rec-outcome-value"' in section


def test_phase2_report_shows_insufficient_history_message_when_outcome_range_missing(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()
    payload = _recommendations_payload(recommendations=[
        {
            "symbol": "NVDA", "action": "BUY", "shares": 10.0, "estimated_dollars": 500.0,
            "reason": "Strong trend", "classification": "Strong Candidate",
            "outcome_p10": None, "outcome_p90": None,
            "best_exit_sessions": None, "best_exit_date": "2025-01-28", "checkpoints_compared": [],
        },
    ])
    (tmp_path / "recommendations_2024-12-31.summary.json").write_text(json.dumps(payload), encoding="utf-8")

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    section = content.split("Today's Recommendations")[1].split("Candidate Ranking")[0]
    assert "Not enough historical data yet for Strong Candidate to show an outcome range." in section
    assert "data-outcome-p10" not in section


def test_phase2_report_best_time_to_sell_shows_good_headline_when_checkpoint_wins(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()
    payload = _recommendations_payload(recommendations=[
        {
            "symbol": "NVDA", "action": "BUY", "shares": 10.0, "estimated_dollars": 500.0,
            "reason": "Strong trend", "classification": "Strong Candidate",
            "outcome_p10": -0.03, "outcome_p90": 0.087,
            "best_exit_sessions": 26, "best_exit_date": "2025-02-04",
            "checkpoints_compared": [
                {"sessions": 21, "date": "2025-01-28", "p10_excess_return": -0.02, "p90_excess_return": 0.06, "outcome_sample_size": 40},
                {"sessions": 26, "date": "2025-02-04", "p10_excess_return": 0.01, "p90_excess_return": 0.09, "outcome_sample_size": 40},
            ],
        },
    ])
    (tmp_path / "recommendations_2024-12-31.summary.json").write_text(json.dumps(payload), encoding="utf-8")

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    section = content.split("Today's Recommendations")[1].split("Candidate Ranking")[0]
    assert '<div class="rec-exit-headline status-good">~5 weeks out &middot; around Feb 4</div>' in section
    assert "This window historically stood out among the checkpoints compared for this setup." in section
    assert "Recalculated every time this report runs" in section
    assert "see all checkpoints compared" in section
    assert "~4 weeks (2025-01-28): -2.0% to +6.0%" in section
    assert "~5 weeks (2025-02-04): +1.0% to +9.0%" in section


def test_phase2_report_best_time_to_sell_shows_hold_headline_when_nothing_wins(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()
    payload = _recommendations_payload(recommendations=[
        {
            "symbol": "NVDA", "action": "BUY", "shares": 10.0, "estimated_dollars": 500.0,
            "reason": "Strong trend", "classification": "Strong Candidate",
            "outcome_p10": -0.02, "outcome_p90": 0.06,
            "best_exit_sessions": None, "best_exit_date": "2025-01-28",
            "checkpoints_compared": [
                {"sessions": 21, "date": "2025-01-28", "p10_excess_return": -0.02, "p90_excess_return": 0.06, "outcome_sample_size": 40},
            ],
        },
    ])
    (tmp_path / "recommendations_2024-12-31.summary.json").write_text(json.dumps(payload), encoding="utf-8")

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    section = content.split("Today's Recommendations")[1].split("Candidate Ranking")[0]
    assert '<div class="rec-exit-headline status-neutral">Hold &middot; check back in about a month</div>' in section
    assert "No shorter window meaningfully outperformed the standard ~1 month horizon." in section


def test_phase2_report_omits_outcome_and_exit_blocks_for_sell_items(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()
    payload = _recommendations_payload(recommendations=[
        {
            "symbol": "AAPL", "action": "SELL", "shares": 5.0, "estimated_dollars": 900.0,
            "reason": "Exit rule triggered", "classification": "Avoid",
            "outcome_p10": -0.05, "outcome_p90": 0.02,
            "best_exit_sessions": 5, "best_exit_date": "2025-01-07",
            "checkpoints_compared": [{"sessions": 5, "date": "2025-01-07", "p10_excess_return": -0.05, "p90_excess_return": 0.02, "outcome_sample_size": 40}],
        },
    ])
    (tmp_path / "recommendations_2024-12-31.summary.json").write_text(json.dumps(payload), encoding="utf-8")

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    section = content.split("Today's Recommendations")[1].split("Candidate Ranking")[0]
    sell_section = section.split("Sell —")[1]
    assert "rec-outcome" not in sell_section
    assert "rec-exit" not in sell_section
    assert "Best time to sell" not in sell_section


def test_phase2_report_resize_widget_js_recomputes_outcome_range(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()
    payload = _recommendations_payload(
        cash_reserve=0.05, max_position_weight=0.15,
        max_trade_dollar_amount=2000.0, min_trade_dollar_amount=100.0,
        recommendations=[
            {
                "symbol": "NVDA", "action": "BUY", "shares": 10.0, "estimated_dollars": 500.0,
                "reason": "Strong trend", "classification": "Strong Candidate",
                "outcome_p10": -0.03, "outcome_p90": 0.087,
                "best_exit_sessions": None, "best_exit_date": "2025-01-28", "checkpoints_compared": [],
            },
        ],
    )
    (tmp_path / "recommendations_2024-12-31.summary.json").write_text(json.dumps(payload), encoding="utf-8")

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    assert "function signedMoney(value)" in content
    assert "function pct(value)" in content
    assert ".rec-outcome-value" in content
    assert 'outcomeSpan.textContent = "—";' in content
    assert "function collapseCallout(item)" in content
    assert "function promoteCallout(item)" in content


def test_phase2_report_promotes_best_time_to_sell_callout_and_collapses_the_rest(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()
    payload = _recommendations_payload(recommendations=[
        {
            "symbol": "NVDA", "action": "BUY", "shares": 10.0, "estimated_dollars": 500.0,
            "reason": "Strong trend", "model_probability": 0.6,
            "predict_v5_excess_return": 0.234, "predict_v5_low_confidence": True,
            "classification": "Strong Candidate",
            "outcome_p10": -0.03, "outcome_p90": 0.087,
            "best_exit_sessions": 26, "best_exit_date": "2025-02-04",
            "checkpoints_compared": [],
        },
    ])
    (tmp_path / "recommendations_2024-12-31.summary.json").write_text(json.dumps(payload), encoding="utf-8")

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    section = content.split("Today's Recommendations")[1].split("Candidate Ranking")[0]
    card = section.split('class="rec-card"')[1]
    # The callout sits between the header and the details toggle -- promoted, not
    # buried behind it.
    head_end = card.index("</div>")
    callout_index = card.index('class="rec-exit-callout')
    details_index = card.index('class="rec-details"')
    assert head_end < callout_index < details_index
    # Reason, model context, historical pattern, and the exit explanation are all
    # folded into the one details toggle rather than sitting always-visible.
    details_body = card[details_index:]
    assert "Strong trend" in details_body
    assert "model 60% beats benchmark" in details_body
    assert "Historical pattern" in details_body
    assert "This window historically stood out" in details_body
    assert "Strong trend" not in card[:details_index]


def test_phase2_report_collapses_callout_when_item_is_not_sized_initially(tmp_path: Path) -> None:
    metadata, results, histories, issues, previous = _report_inputs()
    payload = _recommendations_payload(recommendations=[
        {
            "symbol": "NVDA", "action": "BUY", "shares": None, "estimated_dollars": None,
            "reason": "Strong trend", "classification": "Strong Candidate",
            "outcome_p10": -0.03, "outcome_p90": 0.087,
            "best_exit_sessions": 26, "best_exit_date": "2025-02-04",
            "checkpoints_compared": [],
        },
    ])
    (tmp_path / "recommendations_2024-12-31.summary.json").write_text(json.dumps(payload), encoding="utf-8")

    paths = write_phase2_reports(tmp_path, "2024-12-31", metadata, results, histories, issues, previous_results=previous)

    content = paths["html"].read_text(encoding="utf-8")
    section = content.split("Today's Recommendations")[1].split("Candidate Ranking")[0]
    card = section.split('class="rec-card"')[1]
    details_index = card.index('class="rec-details"')
    callout_index = card.index('class="rec-exit-callout')
    # Not sized -- the callout isn't promoted, it lives inside the details toggle.
    assert callout_index > details_index
