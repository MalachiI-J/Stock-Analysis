import sqlite3
from pathlib import Path

from stock_scrapper.analysis.engine import analyze_symbol
from stock_scrapper.analysis.scoring_config import validate_scoring_config
from stock_scrapper.database import fetch_price_history, initialize_database, record_quality_issue


def test_opportunity_configuration_validation_rejects_mismatches() -> None:
    rules = {
        "opportunity_weights": {
            "momentum": 20,
            "trend_strength": 20,
            "relative_strength": 20,
            "quality": 20,
            "valuation": 20,
            "market_regime_bonus": 0,
        },
        "risk_weights": {"realized_volatility": 20, "drawdown_risk": 20, "downside_volatility": 15, "atr_gap_risk": 10, "beta_sensitivity": 10, "trend_deterioration": 10, "liquidity_risk": 5, "market_regime_risk": 5, "data_quality_risk": 5},
    }
    validate_scoring_config(rules)

    bad_rules = dict(rules)
    bad_rules["opportunity_weights"] = {"momentum": 20, "trend_strength": 20, "relative_strength": 20, "quality": 20, "wrong_component": 20}
    try:
        validate_scoring_config(bad_rules)
    except ValueError:
        return
    raise AssertionError("Expected validation to fail")


def test_analyze_symbol_can_reach_candidate_thresholds() -> None:
    history = [
        {"trade_date": "2023-01-03", "close": 100.0, "adjusted_close": 100.0, "volume": 2000000},
        {"trade_date": "2023-01-04", "close": 104.0, "adjusted_close": 104.0, "volume": 2200000},
        {"trade_date": "2023-01-05", "close": 108.0, "adjusted_close": 108.0, "volume": 2500000},
    ]
    result = analyze_symbol(
        "AAPL",
        history,
        history,
        [],
        as_of_date="2023-01-05",
        rules={
            "minimum_history_days": 3,
            "market_regime_thresholds": {"breadth_threshold": 0.5},
            "risk_weights": {"realized_volatility": 20, "drawdown_risk": 20, "downside_volatility": 15, "atr_gap_risk": 10, "beta_sensitivity": 10, "trend_deterioration": 10, "liquidity_risk": 5, "market_regime_risk": 5, "data_quality_risk": 5},
            "opportunity_weights": {"momentum": 20, "trend_strength": 20, "relative_strength": 20, "quality": 20, "valuation": 20, "market_regime_bonus": 0},
            "confidence_weights": {"history_completeness": 30, "data_freshness": 20, "data_quality": 20, "benchmark_availability": 10, "indicator_availability": 10, "signal_agreement": 10},
            "score_thresholds": {"strong_candidate": 75, "candidate": 65, "watch": 50, "high_risk": 70, "avoid": 30},
        },
        minimum_history_days=3,
        minimum_recent_days=2,
    )
    assert result.classification in {"Candidate", "Strong Candidate"}


def test_fetch_price_history_honors_as_of_date(tmp_path: Path) -> None:
    db_path = tmp_path / "market.db"
    initialize_database(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("INSERT INTO price_history (symbol, trade_date, close) VALUES (?, ?, ?)", ("AAPL", "2024-01-01", 100.0))
        conn.execute("INSERT INTO price_history (symbol, trade_date, close) VALUES (?, ?, ?)", ("AAPL", "2024-01-02", 101.0))
        conn.execute("INSERT INTO price_history (symbol, trade_date, close) VALUES (?, ?, ?)", ("AAPL", "2024-01-03", 102.0))
        conn.commit()
        rows = fetch_price_history(conn, "AAPL", end_date="2024-01-02")
        assert [row["trade_date"] for row in rows] == ["2024-01-01", "2024-01-02"]
    finally:
        conn.close()


def test_record_quality_issue_deduplicates_unresolved(tmp_path: Path) -> None:
    db_path = tmp_path / "market.db"
    initialize_database(db_path)
    conn = sqlite3.connect(db_path)
    try:
        issue = {"symbol": "AAPL", "trade_date": "2024-01-02", "issue_type": "missing_close", "severity": "warning", "description": "Missing close", "detected_time": "2024-01-02T00:00:00", "resolved_status": 0}
        record_quality_issue(conn, issue)
        record_quality_issue(conn, issue)
        unresolved = conn.execute("SELECT COUNT(*) FROM data_quality_issues WHERE symbol=? AND resolved_status=0", ("AAPL",)).fetchone()[0]
        assert unresolved == 1
    finally:
        conn.close()
