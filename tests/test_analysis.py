import csv
import sqlite3
from pathlib import Path

import yaml

from stock_scrapper.analysis.engine import analyze_symbol
from stock_scrapper.database import initialize_database
from stock_scrapper.reporting.report_builder import write_csv_report, write_html_report


def test_symbol_analysis_preserves_unavailable_scores_for_short_history() -> None:
    history = [
        {"trade_date": "2023-01-03", "close": 100.0, "adjusted_close": 100.0, "volume": 1000},
        {"trade_date": "2023-01-04", "close": 101.0, "adjusted_close": 101.0, "volume": 1100},
        {"trade_date": "2023-01-05", "close": 102.0, "adjusted_close": 102.0, "volume": 1200},
    ]
    benchmark_history = [
        {"trade_date": "2022-12-01", "close": 100.0, "adjusted_close": 100.0, "volume": 1000},
        {"trade_date": "2023-01-03", "close": 101.0, "adjusted_close": 101.0, "volume": 1000},
        {"trade_date": "2023-01-05", "close": 103.0, "adjusted_close": 103.0, "volume": 1050},
    ]
    result = analyze_symbol(
        "AAPL",
        history,
        benchmark_history,
        [],
        as_of_date="2023-01-05",
        rules=yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "config" / "scoring_rules.yaml").read_text(
                encoding="utf-8"
            )
        ),
        minimum_history_days=3,
        minimum_recent_days=2,
    )

    assert result.eligible_for_scoring is False
    assert result.risk_score is None
    assert result.opportunity_score is None
    assert result.confidence_score is not None
    assert result.classification == "Insufficient Data"


def test_reports_are_written_without_history_payload_and_without_cdn(tmp_path: Path) -> None:
    csv_path = tmp_path / "summary.csv"
    html_path = tmp_path / "summary.html"
    rows = [{"symbol": "AAPL", "latest_close": 100.0, "history": [{"trade_date": "2023-01-01", "close": 100.0}], "status": "Uptrend"}]

    write_csv_report(csv_path, rows)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        payload = list(csv.DictReader(handle))
    assert "history" not in payload[0]

    write_html_report(html_path, rows, {}, "test", ["AAPL"], [], [])
    content = html_path.read_text(encoding="utf-8")
    assert "plotly" not in content.lower()
    assert "https://cdn.plotly.com" not in content.lower()


def test_database_initialization_creates_phase2_analysis_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "market.db"
    initialize_database(db_path)
    conn = sqlite3.connect(db_path)
    try:
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = {row[0] for row in tables}
        assert "analysis_runs" in table_names
        assert "stock_analysis" in table_names
        assert "market_regime_history" in table_names
    finally:
        conn.close()
