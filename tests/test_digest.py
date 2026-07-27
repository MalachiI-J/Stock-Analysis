from __future__ import annotations

from stock_scrapper.models.analysis_models import AnalysisResult
from stock_scrapper.portfolio import HoldingAssessment
from stock_scrapper.reporting.digest import build_digest, format_holding_line, render_digest_text


def _result(symbol: str, classification: str, opportunity: float | None = 50.0) -> AnalysisResult:
    return AnalysisResult(
        symbol=symbol,
        as_of_date="2024-12-31",
        data_through_date="2024-12-31",
        classification=classification,
        primary_reason=f"{symbol} reason",
        opportunity_score=opportunity,
        risk_score=30.0,
        confidence_score=80.0,
    )


def test_build_digest_buckets_by_classification() -> None:
    results = [
        _result("AAA", "Strong Candidate", 90.0),
        _result("BBB", "Candidate", 70.0),
        _result("CCC", "Watch"),
        _result("DDD", "Avoid"),
        _result("EEE", "High Risk"),
        _result("FFF", "Insufficient Data"),
    ]
    digest = build_digest(
        as_of_date="2024-12-31",
        data_through_date="2024-12-31",
        market_regime="Neutral",
        market_regime_confidence=80.0,
        results=results,
    )
    assert [entry.symbol for entry in digest["buy"]] == ["AAA", "BBB"]
    assert {entry.symbol for entry in digest["sell"]} == {"DDD", "EEE"}
    assert [entry.symbol for entry in digest["watch"]] == ["CCC"]
    assert [entry.symbol for entry in digest["blocked"]] == ["FFF"]
    assert digest["changes"] == []


def test_build_digest_flags_classification_changes() -> None:
    previous = [_result("AAA", "Watch"), _result("BBB", "Candidate")]
    current = [_result("AAA", "Strong Candidate"), _result("BBB", "Candidate")]
    digest = build_digest(
        as_of_date="2025-01-02",
        data_through_date="2025-01-02",
        market_regime="Risk-On",
        market_regime_confidence=None,
        results=current,
        previous_results=previous,
    )
    changed_symbols = [entry.symbol for entry in digest["changes"]]
    assert changed_symbols == ["AAA"]
    assert digest["symbols"]["AAA"].previous_classification == "Watch"
    assert digest["symbols"]["BBB"].changed is False


def test_render_digest_text_includes_sections_and_disclaimer() -> None:
    digest = build_digest(
        as_of_date="2024-12-31",
        data_through_date="2024-12-31",
        market_regime="Neutral",
        market_regime_confidence=80.0,
        results=[_result("AAA", "Strong Candidate", 90.0), _result("DDD", "Avoid")],
    )
    text = render_digest_text(digest)
    assert "BUY / STRONG" in text
    assert "SELL / AVOID" in text
    assert "AAA" in text and "DDD" in text
    assert "not personalized financial advice" in text


def test_render_digest_text_reports_empty_buckets() -> None:
    digest = build_digest(
        as_of_date="2024-12-31",
        data_through_date=None,
        market_regime="Insufficient Market Data",
        market_regime_confidence=None,
        results=[],
    )
    text = render_digest_text(digest)
    assert "None today." in text
    assert "No classification changes since the previous saved run." in text
    assert "YOUR HOLDINGS — 0 open position(s)" in text
    assert "portfolio-buy" in text


def _holding(**overrides: object) -> HoldingAssessment:
    base = dict(
        symbol="AAPL",
        shares=10.0,
        average_cost_basis=100.0,
        latest_price=110.0,
        classification="Candidate",
        primary_reason="Solid trend",
        rule_based_exit_reason=None,
        price_stop_reason=None,
        recommendation="HOLD",
    )
    base.update(overrides)
    return HoldingAssessment(**base)


def test_build_digest_sorts_holdings_by_symbol() -> None:
    digest = build_digest(
        as_of_date="2024-12-31",
        data_through_date="2024-12-31",
        market_regime="Neutral",
        market_regime_confidence=80.0,
        results=[_result("AAA", "Watch")],
        holdings=[_holding(symbol="MSFT"), _holding(symbol="AAPL")],
    )
    assert [holding.symbol for holding in digest["holdings"]] == ["AAPL", "MSFT"]


def test_render_digest_text_includes_holdings_and_sell_signal() -> None:
    digest = build_digest(
        as_of_date="2024-12-31",
        data_through_date="2024-12-31",
        market_regime="Neutral",
        market_regime_confidence=80.0,
        results=[_result("AAA", "Watch")],
        holdings=[
            _holding(symbol="AAPL"),
            _holding(
                symbol="TSLA",
                classification="High Risk",
                recommendation="SELL",
                rule_based_exit_reason="Classification became High Risk",
                price_stop_reason="Stop loss",
            ),
        ],
    )
    text = render_digest_text(digest)
    assert "YOUR HOLDINGS — 2 open position(s)" in text
    assert "AAPL   10 sh @ avg cost $100.00" in text
    assert "Recommendation: HOLD" in text
    assert "TSLA" in text and "Recommendation: SELL" in text
    assert "Sell signal: Classification became High Risk; Stop loss" in text


def test_format_holding_line_reports_unavailable_price() -> None:
    holding = _holding(latest_price=None, recommendation="UNKNOWN (no current price data)")
    line = format_holding_line(holding)
    assert "latest n/a" in line
    assert "unrealized n/a" in line
