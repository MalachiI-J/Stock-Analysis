from __future__ import annotations

from stock_scrapper.models.analysis_models import AnalysisResult
from stock_scrapper.portfolio import HoldingAssessment
from stock_scrapper.reporting.dashboard_builder import render_dashboard_html
from stock_scrapper.reporting.digest import build_digest
from stock_scrapper.trading.recommendations import RecommendationRunResult, TradeRecommendation


def _result(symbol: str, classification: str, opportunity: float | None = 50.0) -> AnalysisResult:
    return AnalysisResult(
        symbol=symbol, as_of_date="2026-01-15", classification=classification,
        primary_reason=f"{symbol} reason", opportunity_score=opportunity,
        risk_score=30.0, confidence_score=80.0,
    )


def _digest(**overrides: object) -> dict[str, object]:
    base = dict(
        as_of_date="2026-01-15", data_through_date="2026-01-15",
        market_regime="Neutral", market_regime_confidence=80.0,
        results=[_result("AAA", "Strong Candidate")],
    )
    base.update(overrides)
    return build_digest(**base)  # type: ignore[arg-type]


def test_render_dashboard_html_includes_market_regime_and_report_link() -> None:
    html = render_dashboard_html(
        as_of_date="2026-01-15", market_regime="Risk-On", market_regime_confidence=75.0,
        digest=_digest(), recommend=None, phase2_report_href="stock_summary_2026-01-15_candidates_abc.html",
    )
    assert "Risk-On" in html
    assert 'href="stock_summary_2026-01-15_candidates_abc.html"' in html


def test_render_dashboard_html_omits_report_link_when_none_available() -> None:
    html = render_dashboard_html(
        as_of_date="2026-01-15", market_regime="Neutral", market_regime_confidence=None,
        digest=_digest(), recommend=None, phase2_report_href=None,
    )
    assert "full candidate/risk report" not in html


def test_render_dashboard_html_shows_placeholder_when_recommend_is_none() -> None:
    html = render_dashboard_html(
        as_of_date="2026-01-15", market_regime="Neutral", market_regime_confidence=None,
        digest=_digest(), recommend=None, phase2_report_href=None,
    )
    assert "run <span class=\"mono\">python main.py recommend</span> first" in html


def test_render_dashboard_html_renders_buy_recommendation_with_model_context() -> None:
    outcome = RecommendationRunResult(
        as_of_date="2026-01-15", account_value=10000.0, available_cash=8000.0, open_position_count=1,
        recommendations=[
            TradeRecommendation(
                symbol="NVDA", action="BUY", shares=10.0, estimated_dollars=500.0, reason="Strong trend",
                model_probability=0.6, predict_v5_excess_return=0.234, predict_v5_low_confidence=True,
            )
        ],
        skipped=["AAPL: no current price available"],
    )
    html = render_dashboard_html(
        as_of_date="2026-01-15", market_regime="Neutral", market_regime_confidence=None,
        digest=_digest(), recommend=outcome, phase2_report_href=None,
    )
    assert "NVDA" in html
    assert "model 60% beats benchmark" in html
    assert "predict-v5 +23.4%" in html
    assert "LOW CONFIDENCE" in html
    assert "AAPL: no current price available" in html


def test_render_dashboard_html_renders_digest_buckets_and_holdings() -> None:
    holding = HoldingAssessment(
        symbol="AAPL", shares=10.0, average_cost_basis=100.0, latest_price=90.0,
        classification="Avoid", primary_reason="weak", rule_based_exit_reason="Classification-based exit",
        price_stop_reason=None, recommendation="SELL",
    )
    digest = _digest(results=[_result("BBB", "Watch")], holdings=[holding])
    html = render_dashboard_html(
        as_of_date="2026-01-15", market_regime="Neutral", market_regime_confidence=None,
        digest=digest, recommend=None, phase2_report_href=None,
    )
    assert "BBB" in html
    assert "AAPL" in html and "Classification-based exit" in html
    assert 'badge-serious">SELL' in html


def test_render_dashboard_html_escapes_untrusted_text() -> None:
    digest = _digest(results=[
        AnalysisResult(
            symbol="AAA", as_of_date="2026-01-15", classification="Watch",
            primary_reason="<script>alert(1)</script>", opportunity_score=50.0,
        )
    ])
    html = render_dashboard_html(
        as_of_date="2026-01-15", market_regime="Neutral", market_regime_confidence=None,
        digest=digest, recommend=None, phase2_report_href=None,
    )
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
