from __future__ import annotations

import copy
import csv
from pathlib import Path

import pytest

from stock_scrapper.backtesting.reporting import write_backtest_reports


def _saved_backtest() -> dict[str, object]:
    return {
        "run_id": "backtest/test-001",
        "strategy_name": "score_v1",
        "strategy_version": "1.0.0",
        "started_at": "2026-07-20T12:00:00+00:00",
        "completed_at": "2026-07-20T12:05:00+00:00",
        "status": "completed",
        "start_date": "2024-01-02",
        "end_date": "2024-04-30",
        "warmup_start_date": "2023-01-03",
        "benchmark_symbol": "SPY",
        "initial_cash": 100_000.0,
        "ending_equity": 107_500.0,
        "symbols_json": '["AAPL","MSFT","TSLA"]',
        "configuration_hash": "b" * 64,
        "configuration_snapshot_json": """{
            "strategy_name":"score_v1",
            "strategy_version":"1.0.0",
            "benchmark":"SPY",
            "warm_up_days":252,
            "signal_frequency":"daily",
            "rebalancing_frequency":"weekly",
            "execution_timing":"next_open",
            "position_sizing":"equal_weight",
            "fractional_shares":false,
            "maximum_positions":5,
            "maximum_position_weight":0.25,
            "cash_reserve":0.05,
            "commission_basis_points":1.0,
            "minimum_commission":1.0,
            "slippage_basis_points":5.0,
            "stop_loss":0.1,
            "trailing_stop":0.15,
            "daily_bar_ambiguity_policy":"adverse_first",
            "final_liquidation":{"enabled":true,"timing":"final_close"}
        }""",
        "data_hash": "data-hash",
        "deterministic_result_hash": "result-hash",
        "error_summary": None,
        "signals": [
            {
                "run_id": "backtest/test-001",
                "signal_id": "sig-1",
                "symbol": "AAPL",
                "signal_date": "2024-01-05",
                "action": "BUY",
                "classification": "Strong Candidate",
                "opportunity_score": 88.0,
                "risk_score": 24.0,
                "confidence_score": 92.0,
                "market_regime": "Risk-On",
                "ranking_json": '{"opportunity":88.0}',
                "reason": "Qualified entry",
                "accepted": 1,
                "rejection_reason": None,
            },
            {
                "run_id": "backtest/test-001",
                "signal_id": "sig-2",
                "symbol": "TSLA",
                "signal_date": "2024-01-05",
                "action": "BUY",
                "classification": "Candidate",
                "opportunity_score": 75.0,
                "risk_score": 69.0,
                "confidence_score": 80.0,
                "market_regime": "Risk-On",
                "ranking_json": '{"opportunity":75.0}',
                "reason": "Qualified but capacity constrained",
                "accepted": 0,
                "rejection_reason": "Maximum positions reached",
            },
            {
                "run_id": "backtest/test-001",
                "signal_id": "sig-3",
                "symbol": "MSFT",
                "signal_date": "2024-03-01",
                "action": "BUY",
                "classification": "Candidate",
                "opportunity_score": 78.0,
                "risk_score": 30.0,
                "confidence_score": 87.0,
                "market_regime": "Neutral",
                "ranking_json": '{"opportunity":78.0}',
                "reason": "Qualified entry",
                "accepted": 1,
                "rejection_reason": None,
            },
        ],
        "rejected_candidates": [
            {
                "signal_id": "sig-4",
                "symbol": "NVDA",
                "signal_date": "2024-03-01",
                "action": "BUY",
                "classification": "Candidate",
                "opportunity_score": 74.0,
                "risk_score": 35.0,
                "confidence_score": 82.0,
                "market_regime": "Neutral",
                "reason": "Insufficient cash reserve",
            }
        ],
        "orders": [
            {
                "run_id": "backtest/test-001",
                "order_id": "ord-1",
                "signal_id": "sig-1",
                "symbol": "AAPL",
                "side": "BUY",
                "signal_date": "2024-01-05",
                "scheduled_date": "2024-01-08",
                "status": "filled",
                "quantity": 100,
                "reference_price": 100.0,
                "reason": "Qualified entry",
            },
            {
                "run_id": "backtest/test-001",
                "order_id": "ord-2",
                "signal_id": "sig-3",
                "symbol": "MSFT",
                "side": "BUY",
                "signal_date": "2024-03-01",
                "scheduled_date": "2024-03-04",
                "status": "filled",
                "quantity": 50,
                "reference_price": 200.0,
                "reason": "Qualified entry",
            },
        ],
        "fills": [
            {
                "run_id": "backtest/test-001",
                "fill_id": "fill-1",
                "order_id": "ord-1",
                "symbol": "AAPL",
                "fill_date": "2024-01-08",
                "side": "BUY",
                "quantity": 100,
                "reference_price": 100.0,
                "fill_price": 100.05,
                "commission": 1.0,
                "slippage": 5.0,
            },
            {
                "run_id": "backtest/test-001",
                "fill_id": "fill-2",
                "order_id": "ord-2",
                "symbol": "MSFT",
                "fill_date": "2024-03-04",
                "side": "BUY",
                "quantity": 50,
                "reference_price": 200.0,
                "fill_price": 200.10,
                "commission": 1.0,
                "slippage": 5.0,
            },
        ],
        "trades": [
            {
                "run_id": "backtest/test-001",
                "trade_id": "trade-1",
                "symbol": "AAPL",
                "signal_date": "2024-01-05",
                "entry_date": "2024-01-08",
                "exit_signal_date": "2024-02-09",
                "exit_date": "2024-02-12",
                "quantity": 100,
                "entry_reference_price": 100.0,
                "entry_fill_price": 100.05,
                "exit_reference_price": 110.0,
                "exit_fill_price": 109.95,
                "entry_commission": 1.0,
                "exit_commission": 1.0,
                "slippage_cost": 10.0,
                "realized_pnl": 978.0,
                "return_pct": 0.0978,
                "holding_days": 35,
                "entry_reason": "Strong Candidate",
                "exit_reason": "Opportunity below exit threshold",
                "classification": "Strong Candidate",
                "market_regime": "Risk-On",
                "ambiguous_daily_bar": 0,
                "strategy_version": "1.0.0",
                "configuration_hash": "b" * 64,
            },
            {
                "run_id": "backtest/test-001",
                "trade_id": "trade-2",
                "symbol": "MSFT",
                "signal_date": "2024-03-01",
                "entry_date": "2024-03-04",
                "exit_signal_date": "2024-04-12",
                "exit_date": "2024-04-15",
                "quantity": 50,
                "entry_reference_price": 200.0,
                "entry_fill_price": 200.10,
                "exit_reference_price": 195.0,
                "exit_fill_price": 194.90,
                "entry_commission": 1.0,
                "exit_commission": 1.0,
                "slippage_cost": 10.0,
                "realized_pnl": -262.0,
                "return_pct": -0.0262,
                "holding_days": 42,
                "entry_reason": "Candidate",
                "exit_reason": "Stop loss",
                "classification": "Candidate",
                "market_regime": "Neutral",
                "ambiguous_daily_bar": 1,
                "strategy_version": "1.0.0",
                "configuration_hash": "b" * 64,
            },
        ],
        "equity_curve": [
            {"run_id": "backtest/test-001", "trade_date": "2024-01-02", "cash": 100000.0, "reserved_cash": 0.0, "market_value": 0.0, "unrealized_pnl": 0.0, "realized_pnl": 0.0, "equity": 100000.0, "gross_exposure": 0.0, "position_count": 0, "daily_return": None, "benchmark_equity": 100000.0},
            {"run_id": "backtest/test-001", "trade_date": "2024-02-01", "cash": 89994.0, "reserved_cash": 0.0, "market_value": 10500.0, "unrealized_pnl": 495.0, "realized_pnl": 0.0, "equity": 100494.0, "gross_exposure": 0.1045, "position_count": 1, "daily_return": 0.00494, "benchmark_equity": 101000.0},
            {"run_id": "backtest/test-001", "trade_date": "2024-03-01", "cash": 100966.0, "reserved_cash": 0.0, "market_value": 0.0, "unrealized_pnl": 0.0, "realized_pnl": 978.0, "equity": 100966.0, "gross_exposure": 0.0, "position_count": 0, "daily_return": 0.0047, "benchmark_equity": 99500.0},
            {"run_id": "backtest/test-001", "trade_date": "2024-04-30", "cash": 107500.0, "reserved_cash": 0.0, "market_value": 0.0, "unrealized_pnl": 0.0, "realized_pnl": 716.0, "equity": 107500.0, "gross_exposure": 0.0, "position_count": 0, "daily_return": 0.0647, "benchmark_equity": 103000.0},
        ],
        "metrics": {
            "starting_equity": 100000.0,
            "ending_equity": 107500.0,
            "net_profit": 7500.0,
            "total_return": 0.075,
            "cagr": 0.24,
            "annualized_volatility": 0.14,
            "maximum_drawdown": -0.03,
            "drawdown_duration": 12,
            "sharpe_ratio": 1.2,
            "sortino_ratio": 1.6,
            "calmar_ratio": 8.0,
            "exposure": 0.45,
            "turnover": 0.8,
            "number_of_trades": 2,
            "win_rate": 0.5,
            "commission_cost": 4.0,
            "slippage_cost": 20.0,
            "benchmark_total_return": 0.03,
            "benchmark_maximum_drawdown": -0.045,
            "return_vs_benchmark": 0.045,
            "drawdown_vs_benchmark": 0.015,
            "cash_total_return": 0.0,
            "monthly_returns": {"2024-01": 0.01, "2024-02": -0.005, "2024-03": 0.02},
            "annual_returns": {"2024": 0.075},
            "limitations": ["Static universe supplied by the user"],
        },
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_backtest_reports_emit_all_offline_html_and_csv_outputs(tmp_path: Path) -> None:
    saved = _saved_backtest()
    original = copy.deepcopy(saved)

    paths = write_backtest_reports(tmp_path, saved)

    assert set(paths) == {
        "html", "summary", "trades", "signals", "rejected_candidates",
        "orders_and_fills", "equity_curve", "monthly_returns", "annual_returns",
    }
    assert all(path.exists() for path in paths.values())
    assert paths["html"].name == "backtest_backtest-test-001.html"
    assert len(_read_csv(paths["summary"])) == 1
    assert len(_read_csv(paths["trades"])) == 2
    assert len(_read_csv(paths["signals"])) == 4
    rejected = _read_csv(paths["rejected_candidates"])
    assert {row["symbol"] for row in rejected} == {"TSLA", "NVDA"}
    order_fills = _read_csv(paths["orders_and_fills"])
    assert [row["record_type"] for row in order_fills].count("order") == 2
    assert [row["record_type"] for row in order_fills].count("fill") == 2
    assert len(_read_csv(paths["equity_curve"])) == 4
    assert len(_read_csv(paths["monthly_returns"])) == 3
    assert len(_read_csv(paths["annual_returns"])) == 1
    assert saved == original

    content = paths["html"].read_text(encoding="utf-8")
    for heading in (
        "Strategy Assumptions",
        "Date Range",
        "Warm-Up Range",
        "Candidate Universe",
        "Excluded Symbols",
        "Execution Assumptions",
        "Commission and Slippage",
        "Performance Summary",
        "SPY Comparison",
        "Equity Curve",
        "Drawdown Chart",
        "Monthly Returns",
        "Annual Returns",
        "Complete Trade Log",
        "Rejected Signals",
        "Performance by Symbol",
        "Performance by Market Regime",
        "Survivorship-bias warning",
        "Static-watchlist warning",
        "Educational Disclaimer",
    ):
        assert heading in content
    for series in (
        "portfolio-equity",
        "benchmark-equity",
        "portfolio-drawdown",
        "benchmark-drawdown",
    ):
        assert f'data-series="{series}"' in content
    assert "Maximum positions reached" in content
    assert "Opportunity below exit threshold" in content
    assert "Historical and simulated performance does not guarantee future results" in content
    assert "<svg" in content
    assert "<script" not in content.lower()
    assert "http://" not in content.lower()
    assert "https://" not in content.lower()


def test_backtest_report_regeneration_is_deterministic_and_overwrites(tmp_path: Path) -> None:
    saved = _saved_backtest()
    first_paths = write_backtest_reports(tmp_path, saved)
    first_bytes = {key: path.read_bytes() for key, path in first_paths.items()}
    first_paths["html"].write_text("stale", encoding="utf-8")
    first_paths["summary"].write_text("stale", encoding="utf-8")

    second_paths = write_backtest_reports(tmp_path, saved)

    assert first_paths == second_paths
    assert {key: path.read_bytes() for key, path in second_paths.items()} == first_bytes


def test_backtest_report_requires_persisted_run_identifier(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty run_id"):
        write_backtest_reports(tmp_path, {"metrics": {}})

    assert list(tmp_path.iterdir()) == []
