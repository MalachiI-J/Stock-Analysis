import sqlite3
from pathlib import Path

import pytest
import yaml

from stock_scrapper.analysis.engine import analyze_symbol
from stock_scrapper.analysis.opportunity_score import calculate_opportunity_score
from stock_scrapper.analysis.scoring_config import validate_scoring_config
from stock_scrapper.database import fetch_price_history, initialize_database, record_quality_issue


def test_opportunity_configuration_validation_rejects_mismatches() -> None:
    rules = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "scoring_rules.yaml").read_text(
            encoding="utf-8"
        )
    )
    validate_scoring_config(rules)

    bad_rules = dict(rules)
    bad_rules["opportunity_weights"] = dict(rules["opportunity_weights"])
    bad_rules["opportunity_weights"]["wrong_component"] = bad_rules["opportunity_weights"].pop(
        "trend_quality"
    )
    with pytest.raises(ValueError):
        validate_scoring_config(bad_rules)


def test_canonical_opportunity_math_reaches_candidate_threshold() -> None:
    rules = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "scoring_rules.yaml").read_text(
            encoding="utf-8"
        )
    )
    metrics = {f"{name}_score": 70.0 for name in rules["opportunity_weights"]}
    score, _, components, _ = calculate_opportunity_score(metrics, rules, "Neutral")
    assert score == 70.0
    assert rules["score_thresholds"]["candidate"] <= score
    assert all(component["available"] for component in components.values())


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
