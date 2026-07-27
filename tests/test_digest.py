from __future__ import annotations

from stock_scrapper.models.analysis_models import AnalysisResult
from stock_scrapper.reporting.digest import build_digest, render_digest_text


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
