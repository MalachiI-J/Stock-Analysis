from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

import pytest
import yaml

import stock_scrapper.analysis.service as service_module
from stock_scrapper.analysis.engine import _classify
from stock_scrapper.analysis.repository import results_from_saved_run
from stock_scrapper.analysis.service import AnalysisService
from stock_scrapper.database import (
    create_connection,
    fetch_price_history,
    get_analysis_run,
    initialize_database,
    record_quality_issue,
)
from stock_scrapper.utilities.hashing import stable_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WATCHLIST = ["AAPL", "MSFT", "SPY", "QQQ", "IWM"]


def _project_rules() -> dict[str, Any]:
    return yaml.safe_load(
        (PROJECT_ROOT / "config" / "scoring_rules.yaml").read_text(encoding="utf-8")
    )


def _trading_dates(start: date, count: int) -> list[str]:
    dates: list[str] = []
    cursor = start
    while len(dates) < count:
        if cursor.weekday() < 5:
            dates.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return dates


def _next_trading_date(value: str) -> str:
    cursor = date.fromisoformat(value) + timedelta(days=1)
    while cursor.weekday() >= 5:
        cursor += timedelta(days=1)
    return cursor.isoformat()


def _seed_histories(
    conn: sqlite3.Connection,
    sessions: int = 460,
) -> tuple[dict[str, list[dict[str, Any]]], str, dict[str, float]]:
    """Insert complete causal histories with one rising and one falling candidate."""
    dates = _trading_dates(date(2022, 1, 3), sessions)
    prices = {"AAPL": 100.0, "MSFT": 160.0, "SPY": 100.0, "QQQ": 120.0, "IWM": 90.0}
    histories: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in WATCHLIST}
    previous_prices = dict(prices)
    for index, trade_date in enumerate(dates):
        if index:
            spy_return = 0.00070 + ((index % 7) - 3) * 0.00008
            returns = {
                "SPY": spy_return,
                "AAPL": 0.00090 + 1.2 * spy_return,
                "MSFT": -0.00070 + ((index % 5) - 2) * 0.00003,
                "QQQ": 0.00100 + 1.1 * spy_return,
                "IWM": 0.00060 + 0.8 * spy_return,
            }
            previous_prices = dict(prices)
            for symbol in WATCHLIST:
                prices[symbol] *= 1.0 + returns[symbol]

        for symbol in WATCHLIST:
            gap = ((index % 5) - 2) * 0.0002 if index else 0.0
            open_price = previous_prices[symbol] * (1.0 + gap) if index else prices[symbol]
            close = prices[symbol]
            row = {
                "symbol": symbol,
                "trade_date": trade_date,
                "open": open_price,
                "high": max(open_price, close) * 1.005,
                "low": min(open_price, close) * 0.995,
                "close": close,
                "adjusted_close": close,
                "volume": 2_000_000 + index * 1_000 + (index % 11) * 10_000,
                "dividends": 0.0,
                "stock_splits": 0.0,
                "data_source": "test",
                "collected_at": "2024-01-01T00:00:00+00:00",
            }
            histories[symbol].append(row)

    sql = """
        INSERT INTO price_history (
            symbol, trade_date, open, high, low, close, adjusted_close, volume,
            dividends, stock_splits, data_source, collected_at
        ) VALUES (
            :symbol, :trade_date, :open, :high, :low, :close, :adjusted_close,
            :volume, :dividends, :stock_splits, :data_source, :collected_at
        )
    """
    for rows in histories.values():
        conn.executemany(sql, rows)
    conn.commit()
    return histories, dates[-1], dict(prices)


def _insert_extreme_future_rows(
    conn: sqlite3.Connection,
    as_of_date: str,
    last_prices: dict[str, float],
) -> str:
    future_date = _next_trading_date(as_of_date)
    future_prices = {
        "AAPL": 1.0,
        "MSFT": 10_000.0,
        "SPY": 1.0,
        "QQQ": 1.0,
        "IWM": 1.0,
    }
    for symbol, close in future_prices.items():
        previous = last_prices[symbol]
        conn.execute(
            """
            INSERT INTO price_history (
                symbol, trade_date, open, high, low, close, adjusted_close, volume,
                dividends, stock_splits, data_source, collected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 'test', ?)
            """,
            (
                symbol,
                future_date,
                previous,
                max(previous, close) * 1.01,
                min(previous, close) * 0.99,
                close,
                close,
                50_000_000,
                "2025-01-01T00:00:00+00:00",
            ),
        )
    conn.commit()
    return future_date


@pytest.fixture
def workspace_tmp_dir() -> Iterator[Path]:
    path = PROJECT_ROOT / ".agent_test_work" / uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def analysis_database(
    workspace_tmp_dir: Path,
) -> Iterator[tuple[sqlite3.Connection, dict[str, Any], str, dict[str, float]]]:
    db_path = workspace_tmp_dir / "analysis.db"
    initialize_database(db_path)
    conn = create_connection(db_path)
    _, as_of_date, last_prices = _seed_histories(conn)
    try:
        yield conn, _project_rules(), as_of_date, last_prices
    finally:
        conn.close()


def test_database_as_of_filters_stock_benchmark_context_and_future_breadth(
    analysis_database: tuple[
        sqlite3.Connection,
        dict[str, Any],
        str,
        dict[str, float],
    ],
) -> None:
    conn, rules, as_of_date, last_prices = analysis_database
    service = AnalysisService(conn, rules, WATCHLIST)
    before = service.analyze_many_as_of(["AAPL", "MSFT"], as_of_date)

    future_date = _insert_extreme_future_rows(conn, as_of_date, last_prices)
    after = service.analyze_many_as_of(["AAPL", "MSFT"], as_of_date)

    assert future_date > as_of_date
    assert before.data_through_date == as_of_date
    assert before.results[0].data_through_date == as_of_date
    assert before.results[0].indicators["latest_close"] == pytest.approx(
        last_prices["AAPL"]
    )
    assert before.market_context.metrics["spy_latest"] == pytest.approx(
        last_prices["SPY"]
    )
    assert before.market_context.metrics["qqq_trend_confirmation"] is True
    assert before.market_context.metrics["iwm_trend_confirmation"] is True
    assert before.market_context.metrics["breadth_symbols_eligible"] == 2
    assert before.market_context.metrics["breadth_symbols_above_sma200"] == 1
    assert before.market_context.metrics["breadth_ratio"] == pytest.approx(0.5)
    assert before.market_context == after.market_context
    assert [asdict(result) for result in before.results] == [
        asdict(result) for result in after.results
    ]


def test_loaded_history_analysis_matches_database_backed_analysis(
    analysis_database: tuple[
        sqlite3.Connection,
        dict[str, Any],
        str,
        dict[str, float],
    ],
) -> None:
    conn, rules, as_of_date, _ = analysis_database
    database_service = AnalysisService(conn, rules, WATCHLIST)
    database_batch = database_service.analyze_many_as_of(
        ["AAPL", "MSFT"], as_of_date
    )
    all_symbols = set(WATCHLIST) | set(rules["market_context_symbols"])
    unbounded_histories = {
        symbol: fetch_price_history(conn, symbol) for symbol in all_symbols
    }
    loaded_batch = AnalysisService(None, rules, WATCHLIST).analyze_loaded_many_as_of(
        ["AAPL", "MSFT"],
        unbounded_histories,
        as_of_date,
    )

    assert database_batch.configuration_hash == loaded_batch.configuration_hash
    assert database_batch.market_context == loaded_batch.market_context
    assert [asdict(result) for result in database_batch.results] == [
        asdict(result) for result in loaded_batch.results
    ]


def test_market_context_is_calculated_once_and_shared_by_every_symbol(
    analysis_database: tuple[
        sqlite3.Connection,
        dict[str, Any],
        str,
        dict[str, float],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn, rules, as_of_date, _ = analysis_database
    calls: list[tuple[int, float | None]] = []
    original = service_module.calculate_market_context

    def counted_market_context(
        benchmark_history: list[dict[str, Any]],
        context_histories: dict[str, list[dict[str, Any]]],
        breadth_ratio: float | None,
        scoring_rules: dict[str, Any],
    ) -> Any:
        calls.append((len(benchmark_history), breadth_ratio))
        return original(
            benchmark_history,
            context_histories,
            breadth_ratio,
            scoring_rules,
        )

    monkeypatch.setattr(service_module, "calculate_market_context", counted_market_context)
    batch = AnalysisService(conn, rules, WATCHLIST).analyze_many_as_of(
        ["AAPL", "MSFT"], as_of_date
    )

    assert len(calls) == 1
    assert all(result.market_regime == batch.market_context.regime for result in batch.results)
    assert all(
        result.market_regime_confidence == batch.market_context.confidence
        for result in batch.results
    )


@pytest.mark.parametrize(
    (
        "eligible",
        "eligibility_meta",
        "risk",
        "opportunity",
        "confidence",
        "regime",
        "expected",
    ),
    [
        (True, {"critical_data_issue": True}, 90.0, 0.0, 0.0, "Stress", "Data Blocked"),
        (True, {"invalid_data": True}, 90.0, 0.0, 0.0, "Stress", "Data Blocked"),
        (False, {}, 90.0, 0.0, 0.0, "Stress", "Insufficient Data"),
        (True, {}, None, 100.0, 100.0, "Risk-On", "Insufficient Data"),
        (True, {}, 80.0, 0.0, 100.0, "Stress", "High Risk"),
        (True, {}, 20.0, 30.0, 100.0, "Risk-On", "Avoid"),
        (True, {}, 20.0, 70.0, 100.0, "Stress", "Avoid"),
        (True, {}, 20.0, 40.0, 100.0, "Risk-On", "Avoid"),
        (True, {}, 20.0, 55.0, 100.0, "Risk-On", "Watch"),
        (True, {}, 20.0, 70.0, 50.0, "Risk-On", "Watch"),
        (True, {}, 20.0, 70.0, 80.0, "Risk-Off", "Watch"),
        (True, {}, 20.0, 70.0, 80.0, "Neutral", "Candidate"),
        (True, {}, 20.0, 80.0, 80.0, "Risk-On", "Strong Candidate"),
    ],
)
def test_classification_precedence(
    eligible: bool,
    eligibility_meta: dict[str, Any],
    risk: float | None,
    opportunity: float | None,
    confidence: float | None,
    regime: str,
    expected: str,
) -> None:
    assert (
        _classify(
            eligible=eligible,
            eligibility_meta=eligibility_meta,
            risk_score=risk,
            opportunity_score=opportunity,
            confidence_score=confidence,
            market_regime=regime,
            rules=_project_rules(),
        )
        == expected
    )


def test_configuration_hash_is_stable_across_order_and_processes() -> None:
    rules = _project_rules()
    reordered = dict(reversed(list(deepcopy(rules).items())))
    for group in ("opportunity_weights", "risk_weights", "confidence_weights"):
        reordered[group] = dict(reversed(list(reordered[group].items())))

    expected = stable_sha256(rules)
    assert stable_sha256(reordered) == expected
    assert len(expected) == 64
    assert all(character in "0123456789abcdef" for character in expected)

    code = (
        "import json,sys; "
        "from stock_scrapper.utilities.hashing import stable_sha256; "
        "print(stable_sha256(json.loads(sys.stdin.read())))"
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=PROJECT_ROOT,
        input=json.dumps(reordered),
        text=True,
        capture_output=True,
        check=True,
    )
    assert completed.stdout.strip() == expected
    changed = deepcopy(rules)
    changed["scoring_version"] = "different-version"
    assert stable_sha256(changed) != expected


def test_persisted_run_has_one_regime_row_and_restores_explanations(
    analysis_database: tuple[
        sqlite3.Connection,
        dict[str, Any],
        str,
        dict[str, float],
    ],
) -> None:
    conn, rules, as_of_date, _ = analysis_database
    record_quality_issue(
        conn,
        {
            "symbol": "AAPL",
            "trade_date": as_of_date,
            "issue_type": "test_warning",
            "severity": "warning",
            "description": "Deterministic warning for saved explanations",
            "detected_time": f"{as_of_date}T12:00:00+00:00",
        },
    )
    conn.commit()
    batch = AnalysisService(conn, rules, WATCHLIST).analyze_many_as_of(
        ["AAPL", "MSFT"], as_of_date, persist=True
    )
    assert batch.analysis_run_id is not None
    run_id = batch.analysis_run_id

    assert conn.execute(
        "SELECT COUNT(*) FROM analysis_runs WHERE analysis_run_id = ?", (run_id,)
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM stock_analysis WHERE analysis_run_id = ?", (run_id,)
    ).fetchone()[0] == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM market_regime_history WHERE analysis_run_id = ?",
        (run_id,),
    ).fetchone()[0] == 1

    saved_run = get_analysis_run(conn, run_id)
    assert saved_run is not None
    assert saved_run["configuration_hash"] == batch.configuration_hash
    assert json.loads(saved_run["configuration_snapshot_json"]) == rules
    assert saved_run["regime"]["regime"] == batch.market_context.regime
    assert json.loads(saved_run["regime"]["reasons_json"]) == batch.market_context.reasons

    restored = {result.symbol: result for result in results_from_saved_run(saved_run)}
    original = {result.symbol: result for result in batch.results}
    explanation_fields = (
        "primary_reason",
        "blocking_reasons",
        "risk_components",
        "opportunity_components",
        "confidence_components",
        "flags",
        "positive_factors",
        "risk_factors",
        "confidence_limitations",
        "quality_concerns",
        "market_regime_effects",
        "improvement_conditions",
        "weakening_conditions",
    )
    for symbol in original:
        for field in explanation_fields:
            assert getattr(restored[symbol], field) == getattr(original[symbol], field)
    assert original["AAPL"].quality_concerns == [
        "Deterministic warning for saved explanations"
    ]
