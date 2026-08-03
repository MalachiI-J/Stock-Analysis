from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

import main as cli
from stock_scrapper.exceptions import ExitCode, InvalidConfigurationError, MissingDataError
from stock_scrapper.backtesting.walk_forward import InsufficientWalkForwardDataError
from stock_scrapper.models.analysis_models import AnalysisResult


class _Logger:
    def info(self, *args: Any, **kwargs: Any) -> None:
        pass

    def warning(self, *args: Any, **kwargs: Any) -> None:
        pass

    def exception(self, *args: Any, **kwargs: Any) -> None:
        pass


class _Connection:
    def __init__(self) -> None:
        self.closed = False
        self.commits = 0
        self.rollbacks = 0
        self.executed: list[str] = []

    def execute(self, sql: str, params: Any = ()) -> None:
        self.executed.append(sql)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True


@dataclass
class _BacktestConfigStub:
    benchmark: str = "SPY"
    warm_up_days: int = 99
    strategy_version: str = "1.1.0"

    @property
    def walk_forward(self) -> SimpleNamespace:
        return SimpleNamespace(warm_up_days=7)

    def to_dict(self) -> dict[str, Any]:
        return {"benchmark": self.benchmark, "strategy_name": "score_v1"}

    def with_overrides(self, **overrides: Any) -> SimpleNamespace:
        merged = {"benchmark": self.benchmark, "strategy_name": "score_v1", **overrides}
        result = SimpleNamespace(**merged)
        result.to_dict = lambda: merged
        return result


def _config(tmp_path: Path) -> dict[str, Any]:
    return {
        "database_path": str(tmp_path / "market.db"),
        "watchlist_path": str(tmp_path / "watchlist.csv"),
        "raw_data_dir": str(tmp_path / "raw"),
        "processed_data_dir": str(tmp_path / "processed"),
        "reports_dir": str(tmp_path / "reports"),
        "logs_dir": str(tmp_path / "logs"),
        "historical_lookback_years": 5,
        "retry_count": 3,
        "retry_delay_seconds": 0,
    }


def _install_startup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    watchlist: list[str] | None = None,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda _base_dir: config)
    monkeypatch.setattr(cli, "ensure_directories", lambda _config: None)
    monkeypatch.setattr(cli, "load_watchlist", lambda _path: list(watchlist or ["AAA"]))
    monkeypatch.setattr(cli, "setup_logging", lambda _config, run_id: _Logger())


@pytest.mark.parametrize(
    "argv",
    [
        ["backtest", "--strategy", "unknown"],
        ["walk-forward", "--strategy", "unknown"],
    ],
)
def test_parser_rejects_unknown_strategies(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.build_parser().parse_args(argv)
    assert exc_info.value.code == int(ExitCode.INVALID_ARGUMENTS)


@pytest.mark.parametrize(
    "argv",
    [
        ["scores", "--run-id", "saved-run", "--recalculate"],
        ["explain", "AAPL", "--run-id", "saved-run", "--recalculate"],
    ],
)
def test_parser_rejects_ambiguous_saved_analysis_options(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.build_parser().parse_args(argv)
    assert exc_info.value.code == int(ExitCode.INVALID_ARGUMENTS)


@pytest.mark.parametrize(
    "argv",
    [
        ["scores", "--as-of-date", "2024-12-31"],
        ["explain", "AAPL", "--as-of-date", "2024-12-31"],
    ],
)
def test_saved_analysis_date_requires_recalculation(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(argv)
    assert exc_info.value.code == int(ExitCode.INVALID_ARGUMENTS)


def test_exact_saved_run_requires_every_requested_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _Connection()
    saved = {
        "analysis_run_id": "saved-run",
        "as_of_date": "2024-12-31",
        "market_regime": "Neutral",
        "analyses": [{"symbol": "AAPL", "classification": "Watch"}],
    }
    monkeypatch.setattr(cli, "initialize_database", lambda _path: None)
    monkeypatch.setattr(cli, "create_connection", lambda _path: connection)
    monkeypatch.setattr(cli, "get_analysis_run", lambda _conn, _run_id: saved)

    with pytest.raises(MissingDataError, match="MSFT"):
        cli._load_saved_results(
            {"database_path": "unused.db"},
            "saved-run",
            ["AAPL", "MSFT"],
        )

    assert connection.closed is True


def _install_walk_forward(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    window_statuses: list[str],
    captured_run_ids: list[str],
) -> list[_Connection]:
    _install_startup(monkeypatch, tmp_path)
    connections: list[_Connection] = []

    def create_connection(_path: str) -> _Connection:
        connection = _Connection()
        connections.append(connection)
        return connection

    monkeypatch.setattr(cli, "initialize_database", lambda _path: None)
    monkeypatch.setattr(cli, "create_connection", create_connection)
    monkeypatch.setattr(
        cli,
        "_backtest_config",
        lambda _base_dir, _args: _BacktestConfigStub(),
    )
    monkeypatch.setattr(cli, "load_scoring_rules", lambda _base_dir: {})
    monkeypatch.setattr(
        cli,
        "_load_backtest_inputs",
        lambda *_args: ({"SPY": [{"trade_date": "2024-01-02"}]}, {}),
    )
    monkeypatch.setattr(cli, "persist_walk_forward", lambda _conn, _result: None)

    def fake_run_walk_forward(
        _config: Any,
        _trading_dates: list[str],
        _executor: Any,
        *,
        symbols: list[str],
        walk_forward_run_id: str,
    ) -> SimpleNamespace:
        del symbols
        captured_run_ids.append(walk_forward_run_id)
        windows = [
            SimpleNamespace(
                window_type="validation",
                evaluation_start_date="2024-01-02",
                evaluation_end_date="2024-01-02",
                status=status,
                backtest_run_id=None,
            )
            for status in window_statuses
        ]
        overall = (
            "completed"
            if all(status == "completed" for status in window_statuses)
            else "completed_with_errors"
        )
        return SimpleNamespace(
            walk_forward_run_id=walk_forward_run_id,
            status=overall,
            windows=windows,
            benchmark_symbol=None,
            symbols=[],
            configuration_snapshot={},
        )

    monkeypatch.setattr(cli, "run_walk_forward", fake_run_walk_forward)
    return connections


@pytest.mark.parametrize(
    ("window_statuses", "expected"),
    [
        (["completed", "completed"], ExitCode.SUCCESS),
        (["completed", "failed"], ExitCode.PARTIAL_FAILURE),
        (["failed", "failed"], ExitCode.OPERATION_FAILED),
    ],
)
def test_walk_forward_exit_code_reflects_window_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    window_statuses: list[str],
    expected: ExitCode,
) -> None:
    captured: list[str] = []
    connections = _install_walk_forward(monkeypatch, tmp_path, window_statuses, captured)

    assert cli.main(["walk-forward", "--symbols", "AAA"]) == int(expected)
    assert len(captured) == 1
    assert len(connections) == 1
    assert connections[0].commits == 1
    assert connections[0].executed == ["BEGIN"]
    assert connections[0].rollbacks == 0
    assert connections[0].closed is True


def test_repeated_walk_forward_commands_receive_distinct_execution_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[str] = []
    connections = _install_walk_forward(monkeypatch, tmp_path, ["completed"], captured)
    hex_values: Iterator[str] = iter(("a" * 32, "b" * 32))
    monkeypatch.setattr(cli, "uuid4", lambda: SimpleNamespace(hex=next(hex_values)))

    assert cli.main(["walk-forward", "--symbols", "AAA"]) == int(
        ExitCode.SUCCESS
    )
    assert cli.main(["walk-forward", "--symbols", "AAA"]) == int(
        ExitCode.SUCCESS
    )

    assert len(captured) == 2
    assert captured[0] != captured[1]
    assert captured[0].endswith("-aaaaaaaa")
    assert captured[1].endswith("-bbbbbbbb")
    assert all(connection.closed for connection in connections)


def test_walk_forward_parent_persistence_failure_rolls_back_children(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[str] = []
    connections = _install_walk_forward(monkeypatch, tmp_path, ["completed"], captured)
    monkeypatch.setattr(
        cli,
        "persist_walk_forward",
        lambda _conn, _result: (_ for _ in ()).throw(sqlite3.IntegrityError("parent failed")),
    )

    assert cli.main(["walk-forward", "--symbols", "AAA"]) == int(ExitCode.DATABASE_FAILURE)
    assert connections[0].commits == 0
    assert connections[0].rollbacks == 1
    assert connections[0].closed is True


def test_walk_forward_reports_per_window_active_return_and_flags_single_window_ceiling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_startup(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "initialize_database", lambda _path: None)
    monkeypatch.setattr(cli, "create_connection", lambda _path: _Connection())
    monkeypatch.setattr(cli, "_backtest_config", lambda _base_dir, _args: _BacktestConfigStub())
    monkeypatch.setattr(cli, "load_scoring_rules", lambda _base_dir: {})
    monkeypatch.setattr(
        cli, "_load_backtest_inputs", lambda *_args: ({"SPY": [{"trade_date": "2024-01-02"}]}, {})
    )
    monkeypatch.setattr(cli, "persist_walk_forward", lambda _conn, _result: None)

    validation_metrics = SimpleNamespace(active_return=0.05, sharpe_ratio=1.2, benchmark_sharpe_ratio=0.9)
    holdout_metrics = SimpleNamespace(active_return=-0.02, sharpe_ratio=0.5, benchmark_sharpe_ratio=1.1)

    def fake_run_walk_forward(
        _config: Any, _trading_dates: list[str], _executor: Any, *, symbols: list[str], walk_forward_run_id: str
    ) -> SimpleNamespace:
        del symbols
        windows = [
            SimpleNamespace(
                window_type="validation", evaluation_start_date="2024-01-02", evaluation_end_date="2024-06-01",
                status="completed", backtest_run_id="bt-1", metrics=validation_metrics,
            ),
            SimpleNamespace(
                window_type="holdout", evaluation_start_date="2024-06-02", evaluation_end_date="2024-12-01",
                status="completed", backtest_run_id="bt-2", metrics=holdout_metrics,
            ),
        ]
        return SimpleNamespace(
            walk_forward_run_id=walk_forward_run_id, status="completed", windows=windows,
            benchmark_symbol=None, symbols=[], configuration_snapshot={},
        )

    monkeypatch.setattr(cli, "run_walk_forward", fake_run_walk_forward)

    assert cli.main(["walk-forward", "--symbols", "AAA"]) == int(ExitCode.SUCCESS)

    output = capsys.readouterr().out
    assert "validation [2024-01-02..2024-06-01]: active_return=+5.00% sharpe=1.20 (benchmark sharpe=0.90) -> beat benchmark" in output
    assert "holdout [2024-06-02..2024-12-01]: active_return=-2.00% sharpe=0.50 (benchmark sharpe=1.10) -> trailed benchmark" in output
    assert "not a statistically powered walk-forward" in output


def test_insufficient_walk_forward_history_uses_missing_data_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[str] = []
    connections = _install_walk_forward(monkeypatch, tmp_path, ["completed"], captured)
    monkeypatch.setattr(
        cli,
        "run_walk_forward",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            InsufficientWalkForwardDataError("not enough sessions")
        ),
    )

    assert cli.main(["walk-forward", "--symbols", "AAA"]) == int(ExitCode.MISSING_DATA)
    assert connections[0].rollbacks == 1
    assert connections[0].closed is True


def test_signal_capture_config_disables_early_exits_and_forces_holding_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _CapturingStub(_BacktestConfigStub):
        def with_overrides(self, **overrides: Any) -> SimpleNamespace:
            captured.update(overrides)
            return super().with_overrides(**overrides)

    monkeypatch.setattr(cli, "_backtest_config", lambda _base_dir, _args: _CapturingStub())

    cli._signal_capture_config(Path("."), SimpleNamespace(strategy="score_v1"), 21)

    assert captured["entry_thresholds"]["classifications"] == ["Strong Candidate"]
    assert captured["entry_thresholds"]["minimum_opportunity_score"] == 0.0
    assert captured["entry_thresholds"]["minimum_average_dollar_volume"] == 0.0
    assert captured["exit_thresholds"]["classifications"] == ["Data Blocked"]
    assert captured["exit_thresholds"]["maximum_risk_score"] == 100.0
    assert captured["exit_thresholds"]["exit_below_sma200"] is False
    assert captured["exit_thresholds"]["exit_on_stress"] is False
    assert captured["stop_loss"] is None
    assert captured["trailing_stop"] is None
    assert captured["profit_target"] is None
    assert captured["maximum_holding_period"] == 21
    assert captured["minimum_confidence"] == 0.0
    assert captured["maximum_risk"] == 100.0
    assert "Risk-Off" in captured["allowed_market_regimes"]
    assert captured["strategy_version"] == "1.1.0-signal-capture-diagnostic"


def test_signal_capture_config_passes_real_schema_validation() -> None:
    """Exercises the REAL BacktestConfig.with_overrides/validate_backtesting_config —
    the stubbed tests above never run real validation, and an earlier version of
    _signal_capture_config passed an empty exit_thresholds.classifications list that
    only a real validation pass ever caught (ValueError: must be a non-empty list)."""
    repo_root = Path(__file__).resolve().parent.parent
    args = SimpleNamespace(strategy="score_v1", symbols=None, start=None, end=None)

    typed = cli._signal_capture_config(repo_root, args, 21)

    assert typed.entry_thresholds.classifications == ("Strong Candidate",)
    assert typed.exit_thresholds.classifications == ("Data Blocked",)
    assert typed.maximum_holding_period == 21
    assert typed.stop_loss is None
    assert typed.trailing_stop is None


def test_signal_capture_test_command_reuses_walk_forward_machinery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: list[str] = []
    connections = _install_walk_forward(monkeypatch, tmp_path, ["completed", "completed"], captured)

    assert cli.main(["signal-capture-test", "--symbols", "AAA"]) == int(ExitCode.SUCCESS)

    output = capsys.readouterr().out
    assert "Signal-capture diagnostic" in output
    assert "Strong-Candidate-only entry" in output
    assert len(captured) == 1
    assert captured[0].startswith("wf-signalcapture-")
    assert connections[0].commits == 1
    assert connections[0].closed is True


def test_walk_forward_children_use_walk_forward_warmup_and_deferred_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_ids: list[str] = []
    _install_walk_forward(monkeypatch, tmp_path, ["completed"], captured_ids)
    observed: dict[str, Any] = {}

    def run_child(_symbols, _histories, _rules, child_config, **kwargs):
        observed["config"] = child_config
        observed["kwargs"] = kwargs
        return SimpleNamespace(
            run=SimpleNamespace(run_id=kwargs["run_id"]),
            metrics=SimpleNamespace(),
        )

    def run_windows(
        config,
        _trading_dates,
        executor,
        *,
        symbols,
        walk_forward_run_id,
    ):
        del symbols
        window = SimpleNamespace(
            window_id=f"{walk_forward_run_id}-window-0001",
            window_type="validation",
            evaluation_start_date="2024-02-01",
            evaluation_end_date="2024-03-01",
            status="completed",
            backtest_run_id=None,
        )
        outcome = executor(window, config)
        window.backtest_run_id = outcome.backtest_run_id
        return SimpleNamespace(
            walk_forward_run_id=walk_forward_run_id,
            status="completed",
            windows=[window],
            benchmark_symbol=None,
            symbols=[],
            configuration_snapshot={},
        )

    monkeypatch.setattr(cli, "run_portfolio_backtest", run_child)
    monkeypatch.setattr(cli, "run_walk_forward", run_windows)

    assert cli.main(["walk-forward", "--symbols", "AAA"]) == int(ExitCode.SUCCESS)
    assert observed["config"].start_date == "2024-02-01"
    assert observed["config"].end_date == "2024-03-01"
    assert observed["config"].warm_up_days == 7
    assert observed["kwargs"]["commit_persistence"] is False


def test_run_uses_one_report_connection_and_closes_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_startup(monkeypatch, tmp_path)
    connections: list[_Connection] = []

    def create_connection(_path: str) -> _Connection:
        connection = _Connection()
        connections.append(connection)
        return connection

    batch = SimpleNamespace(
        analysis_run_id="analysis-run",
        as_of_date="2024-12-31",
        data_through_date="2024-12-31",
        configuration_hash="config-hash",
        market_context=SimpleNamespace(regime="Neutral", confidence=80.0, reasons=[]),
        results=[],
    )
    monkeypatch.setattr(
        cli,
        "update_symbols",
        lambda *_args, **_kwargs: (["AAA"], [], 0, 0),
    )
    monkeypatch.setattr(cli, "validate_database", lambda *_args: [])
    monkeypatch.setattr(cli, "_held_portfolio_symbols", lambda _config: [])
    monkeypatch.setattr(cli, "_analysis_batch", lambda *_args, **_kwargs: batch)
    monkeypatch.setattr(
        cli,
        "load_scoring_rules",
        lambda _base_dir: {
            "scoring_version": "test",
            "benchmark_symbol": "SPY",
        },
    )
    monkeypatch.setattr(cli, "create_connection", create_connection)
    monkeypatch.setattr(cli, "fetch_price_history", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cli, "fetch_quality_issues", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cli, "_previous_analysis", lambda *_args: [])
    monkeypatch.setattr(
        cli,
        "write_phase2_reports",
        lambda *_args: {"csv": "report.csv", "html": "report.html"},
    )

    assert cli.main(["run", "--symbols", "AAA"]) == int(ExitCode.SUCCESS)
    assert len(connections) == 1
    assert connections[0].closed is True


def test_build_reports_closes_connection_when_history_loading_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    connection = _Connection()
    config = _config(tmp_path)
    config["base_dir"] = str(tmp_path)
    result = AnalysisResult(
        symbol="AAA",
        as_of_date="2024-12-31",
        data_through_date="2024-12-31",
    )
    monkeypatch.setattr(cli, "initialize_database", lambda _path: None)
    monkeypatch.setattr(cli, "create_connection", lambda _path: connection)
    monkeypatch.setattr(
        cli,
        "fetch_price_history",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("history failed")),
    )

    with pytest.raises(RuntimeError, match="history failed"):
        cli.build_reports(
            config,
            _Logger(),
            ["AAA"],
            report_date="2024-12-31",
            analysis_results=[result],
        )

    assert connection.closed is True


def _install_digest_saved_run(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    saved = {
        "analysis_run_id": "saved-run",
        "as_of_date": "2024-12-31",
        "data_through_date": "2024-12-31",
        "market_regime": "Neutral",
        "market_regime_confidence": 80.0,
    }
    results = [
        AnalysisResult(
            symbol="AAA",
            as_of_date="2024-12-31",
            data_through_date="2024-12-31",
            classification="Strong Candidate",
            primary_reason="Strong trend",
            opportunity_score=90.0,
            risk_score=20.0,
            confidence_score=85.0,
        ),
        AnalysisResult(
            symbol="BBB",
            as_of_date="2024-12-31",
            data_through_date="2024-12-31",
            classification="Avoid",
            primary_reason="Deteriorating trend",
            opportunity_score=10.0,
            risk_score=80.0,
            confidence_score=60.0,
        ),
    ]
    monkeypatch.setattr(
        cli, "_load_saved_results", lambda *_args, **_kwargs: (saved, results)
    )
    monkeypatch.setattr(cli, "_previous_analysis", lambda *_args: [])
    return saved


def test_digest_command_writes_digest_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_startup(monkeypatch, tmp_path)
    _install_digest_saved_run(monkeypatch)

    assert cli.main(["digest", "--run-id", "saved-run"]) == int(ExitCode.SUCCESS)

    target = tmp_path / "reports" / "digest_2024-12-31.txt"
    assert target.exists()
    text = target.read_text(encoding="utf-8")
    assert "AAA" in text and "Strong Candidate" in text
    assert "BBB" in text and "Avoid" in text
    captured = capsys.readouterr()
    assert "BUY / STRONG" in captured.out

    summary_path = tmp_path / "reports" / "digest_2024-12-31.summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["buy_count"] == 1
    assert summary["sell_count"] == 1
    assert summary["top_buy_symbols"] == ["AAA"]


def test_digest_command_no_save_skips_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_startup(monkeypatch, tmp_path)
    _install_digest_saved_run(monkeypatch)

    assert cli.main(["digest", "--run-id", "saved-run", "--no-save"]) == int(
        ExitCode.SUCCESS
    )

    assert not (tmp_path / "reports" / "digest_2024-12-31.txt").exists()
    assert not (tmp_path / "reports" / "digest_2024-12-31.summary.json").exists()


def test_dashboard_command_writes_html_without_recommend_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_startup(monkeypatch, tmp_path)
    _install_digest_saved_run(monkeypatch)

    assert cli.main(["dashboard", "--run-id", "saved-run"]) == int(ExitCode.SUCCESS)

    target = tmp_path / "reports" / "dashboard_2024-12-31.html"
    assert target.exists()
    html = target.read_text(encoding="utf-8")
    assert "AAA" in html and "BBB" in html
    assert "python main.py recommend</span> first" in html
    captured = capsys.readouterr()
    assert "Dashboard:" in captured.out


def test_dashboard_command_reads_existing_recommend_summary_without_recomputing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_startup(monkeypatch, tmp_path)
    _install_digest_saved_run(monkeypatch)

    def _fail_if_called(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("dashboard must not recompute recommend/predict-v5")

    monkeypatch.setattr(cli, "build_recommendations", _fail_if_called)
    monkeypatch.setattr(cli, "run_gbm_prediction", _fail_if_called)

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "as_of_date": "2024-12-31", "account_value": 10000.0, "available_cash": 8500.0,
        "open_position_count": 1,
        "recommendations": [
            {
                "symbol": "NVDA", "action": "BUY", "shares": 10.0, "estimated_dollars": 500.0,
                "reason": "Strong trend", "model_probability": 0.6,
                "predict_v5_excess_return": 0.234, "predict_v5_low_confidence": True,
            }
        ],
        "skipped": [],
    }
    (reports_dir / "recommendations_2024-12-31.summary.json").write_text(
        json.dumps(summary_payload), encoding="utf-8"
    )

    assert cli.main(["dashboard", "--run-id", "saved-run"]) == int(ExitCode.SUCCESS)

    html = (reports_dir / "dashboard_2024-12-31.html").read_text(encoding="utf-8")
    assert "NVDA" in html
    assert "predict-v5 +23.4%" in html
    assert "LOW CONFIDENCE" in html


def test_dashboard_command_includes_resize_widget_when_summary_has_sizing_rules(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_startup(monkeypatch, tmp_path)
    _install_digest_saved_run(monkeypatch)

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "as_of_date": "2024-12-31", "account_value": 10000.0, "available_cash": 8500.0,
        "open_position_count": 1,
        "cash_reserve": 0.05, "max_position_weight": 0.15,
        "max_trade_dollar_amount": 2000.0, "min_trade_dollar_amount": 100.0,
        "recommendations": [
            {
                "symbol": "NVDA", "action": "BUY", "shares": 10.0, "estimated_dollars": 500.0,
                "reason": "Strong trend",
            }
        ],
        "skipped": [],
    }
    (reports_dir / "recommendations_2024-12-31.summary.json").write_text(
        json.dumps(summary_payload), encoding="utf-8"
    )

    assert cli.main(["dashboard", "--run-id", "saved-run"]) == int(ExitCode.SUCCESS)

    html = (reports_dir / "dashboard_2024-12-31.html").read_text(encoding="utf-8")
    assert 'id="rec-adjust-toggle"' in html
    assert 'data-price="50.000000"' in html


def test_dashboard_command_links_to_existing_phase2_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_startup(monkeypatch, tmp_path)
    _install_digest_saved_run(monkeypatch)

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "stock_summary_2024-12-31_candidates_abc123.html").write_text("<html></html>", encoding="utf-8")

    assert cli.main(["dashboard", "--run-id", "saved-run"]) == int(ExitCode.SUCCESS)

    html = (reports_dir / "dashboard_2024-12-31.html").read_text(encoding="utf-8")
    assert 'href="stock_summary_2024-12-31_candidates_abc123.html"' in html


def test_digest_command_includes_holdings_section(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_startup(monkeypatch, tmp_path)
    _install_digest_saved_run(monkeypatch)
    assert cli.main(
        ["portfolio-buy", "--symbol", "AAA", "--shares", "10", "--price", "50", "--date", "2024-11-01"]
    ) == int(ExitCode.SUCCESS)

    assert cli.main(["digest", "--run-id", "saved-run"]) == int(ExitCode.SUCCESS)

    text = (tmp_path / "reports" / "digest_2024-12-31.txt").read_text(encoding="utf-8")
    assert "YOUR HOLDINGS — 1 open position(s)" in text
    assert "AAA" in text
    assert "Classification: Strong Candidate" in text


def test_portfolio_buy_command_records_lot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_startup(monkeypatch, tmp_path)

    assert cli.main(
        ["portfolio-buy", "--symbol", "aapl", "--shares", "10", "--price", "150.25", "--date", "2026-01-05"]
    ) == int(ExitCode.SUCCESS)

    captured = capsys.readouterr()
    assert "10 sh AAPL" in captured.out

    assert cli.main(["portfolio-show"]) == int(ExitCode.SUCCESS)
    captured = capsys.readouterr()
    assert "OPEN POSITIONS — 1 symbol(s)" in captured.out
    assert "AAPL" in captured.out


def test_portfolio_buy_rejects_nonpositive_shares(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_startup(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["portfolio-buy", "--symbol", "AAPL", "--shares", "0", "--price", "10", "--date", "2026-01-05"])
    assert exc_info.value.code == int(ExitCode.INVALID_ARGUMENTS)


def test_portfolio_sell_command_closes_lot_and_reports_realized_pnl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_startup(monkeypatch, tmp_path)
    assert cli.main(
        ["portfolio-buy", "--symbol", "AAPL", "--shares", "10", "--price", "100", "--date", "2026-01-05"]
    ) == int(ExitCode.SUCCESS)

    assert cli.main(
        ["portfolio-sell", "--symbol", "AAPL", "--shares", "4", "--price", "120", "--date", "2026-02-05"]
    ) == int(ExitCode.SUCCESS)
    captured = capsys.readouterr()
    assert "realized P&L=+80.00" in captured.out

    assert cli.main(["portfolio-show", "--closed"]) == int(ExitCode.SUCCESS)
    captured = capsys.readouterr()
    assert "AAPL" in captured.out
    assert "CLOSED LOTS" in captured.out


def test_portfolio_sell_command_reports_insufficient_holdings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_startup(monkeypatch, tmp_path)
    assert cli.main(
        ["portfolio-buy", "--symbol", "AAPL", "--shares", "5", "--price", "100", "--date", "2026-01-05"]
    ) == int(ExitCode.SUCCESS)

    assert cli.main(
        ["portfolio-sell", "--symbol", "AAPL", "--shares", "6", "--price", "120", "--date", "2026-02-05"]
    ) == int(ExitCode.OPERATION_FAILED)


def test_cleanup_logs_deletes_only_files_older_than_retention(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_startup(monkeypatch, tmp_path)
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True)
    old_log = logs_dir / "stock_scrapper_old.log"
    new_log = logs_dir / "stock_scrapper_new.log"
    old_log.write_text("old", encoding="utf-8")
    new_log.write_text("new", encoding="utf-8")
    old_timestamp = (datetime.now(timezone.utc) - timedelta(days=45)).timestamp()
    os.utime(old_log, (old_timestamp, old_timestamp))

    assert cli.main(["cleanup-logs", "--days", "30"]) == int(ExitCode.SUCCESS)

    assert not old_log.exists()
    assert new_log.exists()
    captured = capsys.readouterr()
    assert "Deleted 1 log file(s)" in captured.out


def test_cleanup_logs_uses_configured_retention_when_no_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config["logs_retention_days"] = 7
    monkeypatch.setattr(cli, "load_config", lambda _base_dir: config)
    monkeypatch.setattr(cli, "ensure_directories", lambda _config: None)
    monkeypatch.setattr(cli, "load_watchlist", lambda _path: ["AAA"])
    monkeypatch.setattr(cli, "setup_logging", lambda _config, run_id: _Logger())
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True)
    old_log = logs_dir / "daily_run_old.log"
    old_log.write_text("old", encoding="utf-8")
    old_timestamp = (datetime.now(timezone.utc) - timedelta(days=10)).timestamp()
    os.utime(old_log, (old_timestamp, old_timestamp))

    assert cli.main(["cleanup-logs"]) == int(ExitCode.SUCCESS)

    assert not old_log.exists()


def test_cleanup_logs_rejects_nonpositive_days(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_startup(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["cleanup-logs", "--days", "0"])
    assert exc_info.value.code == int(ExitCode.INVALID_ARGUMENTS)


def test_cleanup_logs_include_reports_deletes_only_unreferenced_patterns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_startup(monkeypatch, tmp_path)
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)

    old_timestamp = (datetime.now(timezone.utc) - timedelta(days=45)).timestamp()
    unreferenced_old = [
        reports_dir / "digest_2024-01-01.txt",
        reports_dir / "digest_2024-01-01.summary.json",
        reports_dir / "recommendations_2024-01-01.txt",
        reports_dir / "recommendations_2024-01-01.summary.json",
        reports_dir / "dashboard_2024-01-01.html",
        reports_dir / "data_health_2024-01-01.json",
        reports_dir / "data_health_2024-01-01.html",
        reports_dir / "stock_summary_2024-01-01_screen-abc12345.csv",
        reports_dir / "stock_summary_2024-01-01_screen-abc12345.html",
    ]
    for path in unreferenced_old:
        path.write_text("x", encoding="utf-8")
        os.utime(path, (old_timestamp, old_timestamp))

    # Persisted-report-style filenames must never be deleted, even when old.
    protected_old = [
        reports_dir / "stock_summary_2024-01-01_candidates_abc12345.csv",
        reports_dir / "stock_summary_2024-01-01_candidates_abc12345.manifest.json",
        reports_dir / "stock_summary_2024-01-01_custom-AAPL_abc12345.html",
    ]
    for path in protected_old:
        path.write_text("x", encoding="utf-8")
        os.utime(path, (old_timestamp, old_timestamp))

    recent_digest = reports_dir / "digest_2026-01-01.txt"
    recent_digest.write_text("x", encoding="utf-8")

    assert cli.main(["cleanup-logs", "--days", "30", "--include-reports"]) == int(ExitCode.SUCCESS)

    for path in unreferenced_old:
        assert not path.exists(), f"{path} should have been deleted"
    for path in protected_old:
        assert path.exists(), f"{path} must never be deleted"
    assert recent_digest.exists()
    captured = capsys.readouterr()
    assert "Deleted 9 unreferenced report file(s)" in captured.out


def test_cleanup_logs_without_include_reports_leaves_reports_alone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_startup(monkeypatch, tmp_path)
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)
    old_digest = reports_dir / "digest_2024-01-01.txt"
    old_digest.write_text("x", encoding="utf-8")
    old_timestamp = (datetime.now(timezone.utc) - timedelta(days=45)).timestamp()
    os.utime(old_digest, (old_timestamp, old_timestamp))

    assert cli.main(["cleanup-logs", "--days", "30"]) == int(ExitCode.SUCCESS)

    assert old_digest.exists()


def _seed_price_row(symbol: str, trade_date: str, price: float) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "trade_date": trade_date,
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "adjusted_close": price,
        "volume": 1_000_000,
        "dividends": 0.0,
        "stock_splits": 0.0,
        "data_source": "test",
        "collected_at": f"{trade_date}T22:00:00+00:00",
    }


def test_portfolio_compare_reports_shadow_benchmark_comparison(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from stock_scrapper.database import create_connection, initialize_database, upsert_price_history

    _install_startup(monkeypatch, tmp_path)
    config = _config(tmp_path)
    initialize_database(config["database_path"])
    conn = create_connection(config["database_path"])
    try:
        for trade_date, aapl_price, spy_price in [
            ("2020-01-02", 100.0, 50.0),
            ("2020-06-30", 120.0, 60.0),
        ]:
            upsert_price_history(conn, _seed_price_row("AAPL", trade_date, aapl_price))
            upsert_price_history(conn, _seed_price_row("SPY", trade_date, spy_price))
        conn.commit()
    finally:
        conn.close()

    assert cli.main(
        ["portfolio-buy", "--symbol", "AAPL", "--shares", "10", "--price", "100", "--date", "2020-01-02"]
    ) == int(ExitCode.SUCCESS)
    capsys.readouterr()

    assert cli.main(["portfolio-compare", "--as-of-date", "2020-06-30"]) == int(ExitCode.SUCCESS)
    captured = capsys.readouterr()
    assert "Total invested:        $1,000.00" in captured.out
    assert "Unrealized P&L:        $+200.00" in captured.out
    assert "Total P&L:             $+200.00 (+20.0%)" in captured.out
    assert "SPY shadow P&L:      $+200.00 (+20.0%)" in captured.out
    assert "Excess vs SPY:         +0.0%" in captured.out
    assert not summary_has_unpriced(captured.out)


def summary_has_unpriced(output: str) -> bool:
    return "excluded from" in output


def test_portfolio_compare_reports_none_recorded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_startup(monkeypatch, tmp_path)

    assert cli.main(["portfolio-compare"]) == int(ExitCode.SUCCESS)
    captured = capsys.readouterr()
    assert "No portfolio lots recorded" in captured.out


def _install_recommend_saved_run(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    saved = {
        "analysis_run_id": "saved-run",
        "as_of_date": "2024-12-31",
        "data_through_date": "2024-12-31",
        "market_regime": "Neutral",
        "market_regime_confidence": 80.0,
    }
    results = [
        AnalysisResult(
            symbol="AAA", as_of_date="2024-12-31", classification="Strong Candidate",
            primary_reason="Strong trend", opportunity_score=90.0,
        ),
        AnalysisResult(
            symbol="BBB", as_of_date="2024-12-31", classification="Avoid",
            primary_reason="Weak trend", opportunity_score=10.0,
        ),
    ]
    monkeypatch.setattr(cli, "_load_saved_results", lambda *_args, **_kwargs: (saved, results))
    return saved


def _install_recommend_trading_rules(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> dict[str, Any]:
    rules = {
        "trading_rules_version": "test",
        "account_value": 1000.0,
        "available_cash": 1000.0,
        "max_position_weight": 0.5,
        "cash_reserve": 0.0,
        "max_open_positions": 10,
        "max_trade_dollar_amount": 1000.0,
        "min_trade_dollar_amount": 10.0,
        "max_new_buys_per_run": 3,
        "require_strong_candidate_for_buy": True,
        "auto_execute": False,
    }
    rules.update(overrides)
    monkeypatch.setattr(cli, "load_trading_rules", lambda _base_dir: rules)
    return rules


def test_recommend_command_writes_sized_buy_recommendation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from stock_scrapper.database import create_connection, initialize_database, upsert_price_history

    _install_startup(monkeypatch, tmp_path)
    _install_recommend_saved_run(monkeypatch)
    _install_recommend_trading_rules(monkeypatch)
    config = _config(tmp_path)
    initialize_database(config["database_path"])
    conn = create_connection(config["database_path"])
    try:
        upsert_price_history(conn, _seed_price_row("AAA", "2024-12-31", 50.0))
        conn.commit()
    finally:
        conn.close()

    assert cli.main(["recommend", "--run-id", "saved-run", "--no-model"]) == int(ExitCode.SUCCESS)

    captured = capsys.readouterr()
    assert "BUY — 1" in captured.out
    assert "SELL — 0" in captured.out
    assert "AAA" in captured.out and "500.00" in captured.out
    assert "BBB" not in captured.out.split("BUY — 1")[1].split("Considered")[0]

    target = tmp_path / "reports" / "recommendations_2024-12-31.txt"
    assert target.exists()
    summary_path = tmp_path / "reports" / "recommendations_2024-12-31.summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["recommendations"] == [
        {
            "symbol": "AAA", "action": "BUY", "shares": 10.0,
            "estimated_dollars": 500.0, "reason": "Strong trend", "model_probability": None,
            "predict_v5_excess_return": None, "predict_v5_low_confidence": False,
        }
    ]
    # Sizing-rule constants a report needs for its client-side "what if" resize widget.
    assert summary["cash_reserve"] == 0.0
    assert summary["max_position_weight"] == 0.5
    assert summary["max_trade_dollar_amount"] == 1000.0
    assert summary["min_trade_dollar_amount"] == 10.0


def test_recommend_command_attaches_predict_v5_context_to_buy_recommendations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from stock_scrapper.database import create_connection, initialize_database, upsert_price_history
    from stock_scrapper.prediction.service import PredictionRunResult
    from stock_scrapper.prediction.gbm_service import GbmPredictionRunResult, SymbolExcessReturnPrediction

    _install_startup(monkeypatch, tmp_path)
    _install_recommend_saved_run(monkeypatch)
    _install_recommend_trading_rules(monkeypatch)
    config = _config(tmp_path)
    initialize_database(config["database_path"])
    conn = create_connection(config["database_path"])
    try:
        upsert_price_history(conn, _seed_price_row("AAA", "2024-12-31", 50.0))
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(
        cli, "load_prediction_rules",
        lambda _base_dir: {"predict_v5": {"feature_keys": ["rsi_14"], "gbm": None}},
    )
    monkeypatch.setattr(cli, "load_universes", lambda _config: {
        "candidates": ["AAA", "BBB"], "benchmark": "SPY", "market_context": ["SPY"], "defensive_context": [],
    })
    monkeypatch.setattr(cli, "load_scoring_rules", lambda _base_dir: {"scoring_version": "test"})
    monkeypatch.setattr(cli, "AnalysisService", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        cli, "run_prediction",
        lambda *_args, **_kwargs: PredictionRunResult(
            status="ok", message=None, as_of_date="2024-12-31", horizon_days=21,
        ),
    )
    monkeypatch.setattr(
        cli, "run_gbm_prediction",
        lambda *_args, **_kwargs: GbmPredictionRunResult(
            status="ok", message=None, as_of_date="2024-12-31", horizon_days=21,
            training_samples=10, holdout_samples=5,
            predictions=[SymbolExcessReturnPrediction("AAA", 0.234, None, True)],
        ),
    )

    assert cli.main(["recommend", "--run-id", "saved-run"]) == int(ExitCode.SUCCESS)

    captured = capsys.readouterr()
    assert "predict-v5: +23.4% predicted excess return" in captured.out
    assert "[LOW CONFIDENCE]" in captured.out

    summary_path = tmp_path / "reports" / "recommendations_2024-12-31.summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["recommendations"][0]["predict_v5_excess_return"] == 0.234
    assert summary["recommendations"][0]["predict_v5_low_confidence"] is True


def test_recommend_command_no_save_skips_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_startup(monkeypatch, tmp_path)
    _install_recommend_saved_run(monkeypatch)
    _install_recommend_trading_rules(monkeypatch)

    assert cli.main(["recommend", "--run-id", "saved-run", "--no-model", "--no-save"]) == int(ExitCode.SUCCESS)

    assert not (tmp_path / "reports" / "recommendations_2024-12-31.txt").exists()
    assert not (tmp_path / "reports" / "recommendations_2024-12-31.summary.json").exists()


def test_recommend_command_respects_max_trade_dollar_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from stock_scrapper.database import create_connection, initialize_database, upsert_price_history

    _install_startup(monkeypatch, tmp_path)
    _install_recommend_saved_run(monkeypatch)
    _install_recommend_trading_rules(monkeypatch, max_trade_dollar_amount=100.0)
    config = _config(tmp_path)
    initialize_database(config["database_path"])
    conn = create_connection(config["database_path"])
    try:
        upsert_price_history(conn, _seed_price_row("AAA", "2024-12-31", 50.0))
        conn.commit()
    finally:
        conn.close()

    assert cli.main(["recommend", "--run-id", "saved-run", "--no-model"]) == int(ExitCode.SUCCESS)

    summary_path = tmp_path / "reports" / "recommendations_2024-12-31.summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["recommendations"][0]["shares"] == 2.0  # $100 cap / $50 price
    assert summary["recommendations"][0]["estimated_dollars"] == 100.0


def test_recommend_review_reports_hit_rate_against_actual_prices(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from stock_scrapper.database import create_connection, initialize_database, upsert_price_history

    _install_startup(monkeypatch, tmp_path)
    config = _config(tmp_path)
    reports_dir = Path(config["reports_dir"])
    reports_dir.mkdir(parents=True)
    summary_path = reports_dir / "recommendations_2024-12-31.summary.json"
    summary_path.write_text(
        json.dumps({
            "as_of_date": "2024-12-31",
            "recommendations": [
                {
                    "symbol": "AAPL", "action": "BUY", "shares": 10.0, "estimated_dollars": 1000.0,
                    "reason": "test", "model_probability": None,
                },
            ],
        }),
        encoding="utf-8",
    )

    initialize_database(config["database_path"])
    conn = create_connection(config["database_path"])
    try:
        upsert_price_history(conn, _seed_price_row("SPY", "2024-12-31", 400.0))
        upsert_price_history(conn, _seed_price_row("SPY", "2025-01-31", 408.0))  # SPY +2%
        upsert_price_history(conn, _seed_price_row("AAPL", "2025-01-31", 110.0))  # entry 100 -> +10%
        conn.commit()
    finally:
        conn.close()

    assert cli.main(
        ["recommend-review", "--recommendation-date", "2024-12-31", "--as-of-date", "2025-01-31"]
    ) == int(ExitCode.SUCCESS)

    captured = capsys.readouterr()
    assert "AAPL" in captured.out and "beat the benchmark" in captured.out
    assert "BUY hit rate" in captured.out and "100%" in captured.out


def test_recommend_review_reports_missing_data_when_no_file_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_startup(monkeypatch, tmp_path)

    assert cli.main(
        ["recommend-review", "--recommendation-date", "2024-12-31"]
    ) == int(ExitCode.MISSING_DATA)


def test_update_trading_rules_file_patches_matching_lines_preserving_comments(tmp_path: Path) -> None:
    path = tmp_path / "trading_rules.yaml"
    path.write_text(
        "trading_rules_version: \"trade-v1\"\n"
        "account_value: 10000.0\n"
        "available_cash: 10000.0\n"
        "max_position_weight: 0.15\n"
        "# Placeholder comment that must survive.\n"
        "auto_execute: false\n",
        encoding="utf-8",
    )

    cli._update_trading_rules_file(path, {"account_value": 5000.0, "available_cash": 2000.0})

    text = path.read_text(encoding="utf-8")
    assert "account_value: 5000.0" in text
    assert "available_cash: 2000.0" in text
    assert "# Placeholder comment that must survive." in text
    assert "max_position_weight: 0.15" in text
    assert "auto_execute: false" in text


def test_update_trading_rules_file_appends_keys_that_do_not_exist_yet(tmp_path: Path) -> None:
    path = tmp_path / "trading_rules.yaml"
    path.write_text("trading_rules_version: \"trade-v1\"\n", encoding="utf-8")

    cli._update_trading_rules_file(path, {"account_value": 5000.0})

    assert "account_value: 5000.0" in path.read_text(encoding="utf-8")


def test_account_set_writes_valid_values_and_keeps_other_fields(tmp_path: Path) -> None:
    path = tmp_path / "trading_rules.yaml"
    path.write_text(
        "trading_rules_version: \"trade-v1\"\n"
        "account_value: 10000.0\n"
        "available_cash: 10000.0\n"
        "max_position_weight: 0.15\n"
        "cash_reserve: 0.05\n"
        "max_open_positions: 10\n"
        "max_trade_dollar_amount: 2000.0\n"
        "min_trade_dollar_amount: 100.0\n"
        "max_new_buys_per_run: 3\n"
        "require_strong_candidate_for_buy: true\n"
        "auto_execute: false\n",
        encoding="utf-8",
    )

    cli._account_set(path, 20000.0, 5000.0)

    text = path.read_text(encoding="utf-8")
    assert "account_value: 20000.0" in text
    assert "available_cash: 5000.0" in text
    assert "max_position_weight: 0.15" in text  # untouched


def test_account_set_rejects_available_cash_exceeding_account_value(tmp_path: Path) -> None:
    path = tmp_path / "trading_rules.yaml"
    path.write_text(
        "trading_rules_version: \"trade-v1\"\n"
        "account_value: 10000.0\n"
        "available_cash: 10000.0\n"
        "max_position_weight: 0.15\n"
        "cash_reserve: 0.05\n"
        "max_open_positions: 10\n"
        "max_trade_dollar_amount: 2000.0\n"
        "min_trade_dollar_amount: 100.0\n"
        "max_new_buys_per_run: 3\n"
        "require_strong_candidate_for_buy: true\n"
        "auto_execute: false\n",
        encoding="utf-8",
    )
    original = path.read_text(encoding="utf-8")

    with pytest.raises(InvalidConfigurationError, match="must not exceed account_value"):
        cli._account_set(path, 1000.0, 5000.0)

    assert path.read_text(encoding="utf-8") == original  # rejected before any write


def test_account_set_raises_when_trading_rules_file_is_missing(tmp_path: Path) -> None:
    with pytest.raises(InvalidConfigurationError, match="missing"):
        cli._account_set(tmp_path / "does-not-exist.yaml", 10000.0, 5000.0)


def test_account_set_command_rejects_negative_account_value() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["account-set", "--account-value", "-1", "--available-cash", "0"])
    assert exc_info.value.code == int(ExitCode.INVALID_ARGUMENTS)


def test_account_set_command_rejects_available_cash_exceeding_account_value() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["account-set", "--account-value", "1000", "--available-cash", "2000"])
    assert exc_info.value.code == int(ExitCode.INVALID_ARGUMENTS)


def test_account_set_command_calls_account_set_and_prints_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_startup(monkeypatch, tmp_path)
    captured_args: dict[str, Any] = {}

    def _fake_account_set(path: Path, account_value: float, available_cash: float) -> None:
        captured_args.update(path=path, account_value=account_value, available_cash=available_cash)

    monkeypatch.setattr(cli, "_account_set", _fake_account_set)

    assert cli.main(
        ["account-set", "--account-value", "20000", "--available-cash", "5000"]
    ) == int(ExitCode.SUCCESS)

    assert captured_args["account_value"] == 20000.0
    assert captured_args["available_cash"] == 5000.0
    assert captured_args["path"].name == "trading_rules.yaml"
    captured = capsys.readouterr()
    assert "account_value=$20,000.00" in captured.out
    assert "available_cash=$5,000.00" in captured.out


def test_new_screening_symbols_excludes_tracked_and_dedupes() -> None:
    result = cli._new_screening_symbols(["aapl", "MSFT", "ZZZZ", "zzzz", "jpm"], ["AAPL", "JPM"])
    assert result == ["MSFT", "ZZZZ"]


def _install_startup_with_real_watchlist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    watchlist: list[str],
) -> dict[str, Any]:
    # Unlike _install_startup, this leaves load_watchlist unstubbed so a
    # separately loaded screening-universe CSV is read for real too (the
    # blanket stub would otherwise intercept both calls identically).
    config = _config(tmp_path)
    Path(config["watchlist_path"]).write_text(
        "symbol\n" + "\n".join(watchlist) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(cli, "load_config", lambda _base_dir: config)
    monkeypatch.setattr(cli, "ensure_directories", lambda _config: None)
    monkeypatch.setattr(cli, "setup_logging", lambda _config, run_id: _Logger())
    return config


def test_screen_command_reports_no_new_symbols(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_startup_with_real_watchlist(monkeypatch, tmp_path, ["AAPL", "MSFT"])
    universe_path = tmp_path / "screening_universe.csv"
    universe_path.write_text("symbol\nAAPL\nMSFT\n", encoding="utf-8")

    assert cli.main(["screen", "--universe-path", str(universe_path)]) == int(ExitCode.SUCCESS)
    captured = capsys.readouterr()
    assert "No new symbols to screen" in captured.out


def test_screen_command_reports_new_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_startup_with_real_watchlist(monkeypatch, tmp_path, ["AAPL"])
    universe_path = tmp_path / "screening_universe.csv"
    universe_path.write_text("symbol\nUNH\nJNJ\n", encoding="utf-8")

    batch = SimpleNamespace(
        results=[
            AnalysisResult(
                symbol="UNH", as_of_date="2024-12-31", data_through_date="2024-12-31",
                classification="Strong Candidate", primary_reason="Strong trend",
                opportunity_score=88.0, risk_score=20.0, confidence_score=90.0,
            ),
            AnalysisResult(
                symbol="JNJ", as_of_date="2024-12-31", data_through_date="2024-12-31",
                classification="Watch", primary_reason="Mixed signals",
                opportunity_score=55.0, risk_score=40.0, confidence_score=70.0,
            ),
        ],
        as_of_date="2024-12-31",
        data_through_date="2024-12-31",
        configuration_hash="config-hash",
        market_context=SimpleNamespace(regime="Neutral", confidence=80.0, reasons=[]),
    )
    monkeypatch.setattr(cli, "_analysis_batch", lambda *_args, **_kwargs: batch)
    monkeypatch.setattr(cli, "load_scoring_rules", lambda _base_dir: {"scoring_version": "test", "benchmark_symbol": "SPY"})
    monkeypatch.setattr(cli, "initialize_database", lambda _path: None)
    monkeypatch.setattr(cli, "create_connection", lambda _path: _Connection())
    monkeypatch.setattr(cli, "fetch_price_history", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cli, "fetch_quality_issues", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cli, "write_phase2_reports", lambda *_args, **_kwargs: {"csv": "s.csv", "html": "s.html"})

    assert cli.main(["screen", "--universe-path", str(universe_path)]) == int(ExitCode.SUCCESS)
    captured = capsys.readouterr()
    assert "Screened 2 symbol(s)" in captured.out
    assert "New Candidate/Strong Candidate symbol(s): 1" in captured.out
    assert "UNH" in captured.out and "Strong Candidate" in captured.out
    assert "Screener report: s.html" in captured.out


def test_warn_if_screening_universe_stale_warns_past_threshold(
    capsys: pytest.CaptureFixture[str],
) -> None:
    stale_date = (date.today() - timedelta(days=200)).isoformat()
    cli._warn_if_screening_universe_stale(
        {"screening": {"universe_last_verified": stale_date, "staleness_warning_days": 180}}
    )
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "screening_universe.csv" in captured.err
    assert stale_date in captured.err


def test_warn_if_screening_universe_stale_silent_within_threshold(
    capsys: pytest.CaptureFixture[str],
) -> None:
    recent_date = (date.today() - timedelta(days=10)).isoformat()
    cli._warn_if_screening_universe_stale(
        {"screening": {"universe_last_verified": recent_date, "staleness_warning_days": 180}}
    )
    captured = capsys.readouterr()
    assert captured.err == ""


def test_warn_if_screening_universe_stale_silent_when_setting_missing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._warn_if_screening_universe_stale({})
    captured = capsys.readouterr()
    assert captured.err == ""


def test_warn_if_screening_universe_stale_silent_when_malformed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._warn_if_screening_universe_stale({"screening": {"universe_last_verified": "not-a-date"}})
    captured = capsys.readouterr()
    assert captured.err == ""


def test_screen_command_prints_staleness_warning_when_configured_stale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _install_startup_with_real_watchlist(monkeypatch, tmp_path, ["AAPL", "MSFT"])
    stale_date = (date.today() - timedelta(days=200)).isoformat()
    config["screening"] = {"universe_last_verified": stale_date, "staleness_warning_days": 180}

    universe_path = tmp_path / "screening_universe.csv"
    universe_path.write_text("symbol\nAAPL\nMSFT\n", encoding="utf-8")

    assert cli.main(["screen", "--universe-path", str(universe_path)]) == int(ExitCode.SUCCESS)
    captured = capsys.readouterr()
    assert "WARNING" in captured.err and "screening_universe.csv" in captured.err


def test_screen_command_update_failure_raises_operation_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_startup(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "load_universes", lambda _config: {"candidates": []})
    universe_path = tmp_path / "screening_universe.csv"
    universe_path.write_text("symbol\nUNH\n", encoding="utf-8")
    monkeypatch.setattr(cli, "update_symbols", lambda *_args, **_kwargs: ([], ["UNH"], 0, 0))

    assert cli.main(["screen", "--universe-path", str(universe_path), "--update"]) == int(
        ExitCode.OPERATION_FAILED
    )


def test_screen_command_rejects_missing_universe_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_startup(monkeypatch, tmp_path)
    missing = tmp_path / "does-not-exist.csv"

    assert cli.main(["screen", "--universe-path", str(missing)]) == int(
        ExitCode.INVALID_CONFIGURATION
    )


def _install_predict_startup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_startup(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "load_universes", lambda _config: {
        "candidates": ["AAPL"], "benchmark": "SPY", "market_context": ["SPY"], "defensive_context": [],
    })
    monkeypatch.setattr(cli, "load_scoring_rules", lambda _base_dir: {"scoring_version": "test"})
    monkeypatch.setattr(cli, "initialize_database", lambda _path: None)
    monkeypatch.setattr(cli, "create_connection", lambda _path: _Connection())
    monkeypatch.setattr(cli, "fetch_price_history", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cli, "fetch_fundamentals", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cli, "AnalysisService", lambda *_args, **_kwargs: object())


def test_predict_command_reports_insufficient_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from stock_scrapper.prediction.service import PredictionRunResult

    _install_predict_startup(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cli, "run_prediction",
        lambda *_args, **_kwargs: PredictionRunResult(
            status="insufficient_data", message="Not enough history.", as_of_date="2026-01-01", horizon_days=21,
        ),
    )

    assert cli.main(["predict", "--symbols", "AAPL"]) == int(ExitCode.MISSING_DATA)


def test_predict_command_reports_coefficients_and_ranked_predictions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from stock_scrapper.prediction.service import PredictionRunResult, SymbolPrediction, WalkForwardFold

    _install_predict_startup(monkeypatch, tmp_path)
    result = PredictionRunResult(
        status="ok",
        message=None,
        as_of_date="2026-01-01",
        horizon_days=21,
        training_samples=400,
        holdout_samples=100,
        training_start_date="2023-01-01",
        training_end_date="2025-11-01",
        positive_label_rate=0.52,
        holdout_accuracy=0.55,
        holdout_brier_score=0.24,
        walk_forward_folds=[WalkForwardFold(1, 200, 50, 0.55, 0.24), WalkForwardFold(2, 300, 50, 0.55, 0.24)],
        coefficients=[("rsi_14", 0.42), ("beta", -0.13), ("atr_percentage", 0.001)],
        predictions=[
            SymbolPrediction("AAPL", 0.71, None),
            SymbolPrediction("MSFT", 0.30, None),
            SymbolPrediction("ZZZZ", None, "One or more required indicators are unavailable"),
        ],
    )
    monkeypatch.setattr(cli, "run_prediction", lambda *_args, **_kwargs: result)

    assert cli.main(["predict", "--symbols", "AAPL", "MSFT", "ZZZZ"]) == int(ExitCode.SUCCESS)
    captured = capsys.readouterr()
    assert "EXPERIMENTAL STATISTICAL FORECAST" in captured.err
    assert "Walk-forward holdout accuracy: 55.0%" in captured.out
    assert "dataset-wide positive rate 52.0%" in captured.out
    assert "fold 1: train" in captured.out and "(200 samples, 0 symbols, 0 purged" in captured.out
    assert "rsi_14" in captured.out and "+0.4200" in captured.out
    assert "Near-zero influence" in captured.out and "atr_percentage" in captured.out
    lines = captured.out.splitlines()
    aapl_line = next(line for line in lines if line.strip().startswith("AAPL"))
    msft_line = next(line for line in lines if line.strip().startswith("MSFT"))
    zzzz_line = next(line for line in lines if line.strip().startswith("ZZZZ"))
    assert lines.index(aapl_line) < lines.index(msft_line) < lines.index(zzzz_line)
    assert "71.0%" in aapl_line
    assert "unavailable" in zzzz_line


def test_predict_command_horizon_days_override_replaces_config_value(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from stock_scrapper.prediction.service import PredictionRunResult

    _install_predict_startup(monkeypatch, tmp_path)
    captured_rules: dict[str, Any] = {}

    def _fake_run_prediction(*_args: Any, **kwargs: Any) -> PredictionRunResult:
        captured_rules.update(kwargs["rules"])
        return PredictionRunResult(
            status="insufficient_data", message="x", as_of_date="2026-01-01",
            horizon_days=kwargs["rules"]["horizon_days"],
        )

    monkeypatch.setattr(cli, "run_prediction", _fake_run_prediction)

    cli.main(["predict", "--symbols", "AAPL", "--horizon-days", "5"])

    assert captured_rules["horizon_days"] == 5


def test_predict_command_rejects_nonpositive_horizon_days(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_predict_startup(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["predict", "--symbols", "AAPL", "--horizon-days", "0"])
    assert exc_info.value.code == int(ExitCode.INVALID_ARGUMENTS)


def test_predict_v4_command_reports_insufficient_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from stock_scrapper.prediction.gbm_service import GbmPredictionRunResult

    _install_predict_startup(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cli, "run_gbm_prediction",
        lambda *_args, **_kwargs: GbmPredictionRunResult(
            status="insufficient_data", message="Not enough history.", as_of_date="2026-01-01", horizon_days=21,
        ),
    )

    assert cli.main(["predict-v4", "--symbols", "AAPL"]) == int(ExitCode.MISSING_DATA)


def test_predict_v4_command_reports_feature_importances_and_ranked_predictions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from stock_scrapper.prediction.gbm_service import (
        GbmPredictionRunResult,
        GbmWalkForwardFold,
        SymbolExcessReturnPrediction,
    )

    _install_predict_startup(monkeypatch, tmp_path)
    result = GbmPredictionRunResult(
        status="ok",
        message=None,
        as_of_date="2026-01-01",
        horizon_days=21,
        training_samples=400,
        holdout_samples=100,
        training_start_date="2023-01-01",
        training_end_date="2025-11-01",
        holdout_mse=0.018,
        holdout_mean_absolute_error=0.09,
        holdout_information_coefficient=0.04,
        baseline_mse=0.021,
        walk_forward_folds=[
            GbmWalkForwardFold(1, 200, 50, mse=0.018, mean_absolute_error=0.09, information_coefficient=0.03),
            GbmWalkForwardFold(2, 300, 50, mse=0.017, mean_absolute_error=0.08, information_coefficient=0.05),
        ],
        feature_importances=[("six_month_return", 0.62), ("beta", 0.28), ("atr_percentage", 0.10)],
        predictions=[
            SymbolExcessReturnPrediction("AAPL", 0.021, None),
            SymbolExcessReturnPrediction("MSFT", -0.005, None),
            SymbolExcessReturnPrediction("ZZZZ", None, "One or more required indicators are unavailable"),
        ],
    )
    monkeypatch.setattr(cli, "run_gbm_prediction", lambda *_args, **_kwargs: result)

    assert cli.main(["predict-v4", "--symbols", "AAPL", "MSFT", "ZZZZ"]) == int(ExitCode.SUCCESS)
    captured = capsys.readouterr()
    assert "EXPERIMENTAL STATISTICAL FORECAST" in captured.err
    assert "Walk-forward holdout MSE: 0.018000" in captured.out
    assert "fold 1: train" in captured.out
    assert "six_month_return" in captured.out and "62.0%" in captured.out
    lines = captured.out.splitlines()
    aapl_line = next(line for line in lines if line.strip().startswith("AAPL"))
    msft_line = next(line for line in lines if line.strip().startswith("MSFT"))
    zzzz_line = next(line for line in lines if line.strip().startswith("ZZZZ"))
    assert lines.index(aapl_line) < lines.index(msft_line) < lines.index(zzzz_line)
    assert "+2.10%" in aapl_line
    assert "unavailable" in zzzz_line


def test_predict_v4_command_horizon_days_override_replaces_config_value(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from stock_scrapper.prediction.gbm_service import GbmPredictionRunResult

    _install_predict_startup(monkeypatch, tmp_path)
    captured_rules: dict[str, Any] = {}

    def _fake_run_gbm_prediction(*_args: Any, **kwargs: Any) -> GbmPredictionRunResult:
        captured_rules.update(kwargs["rules"])
        return GbmPredictionRunResult(
            status="insufficient_data", message="x", as_of_date="2026-01-01",
            horizon_days=kwargs["rules"]["horizon_days"],
        )

    monkeypatch.setattr(cli, "run_gbm_prediction", _fake_run_gbm_prediction)

    cli.main(["predict-v4", "--symbols", "AAPL", "--horizon-days", "5"])

    assert captured_rules["horizon_days"] == 5


def test_predict_v4_command_rejects_nonpositive_horizon_days(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_predict_startup(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["predict-v4", "--symbols", "AAPL", "--horizon-days", "0"])
    assert exc_info.value.code == int(ExitCode.INVALID_ARGUMENTS)


def test_predict_v5_command_reports_insufficient_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from stock_scrapper.prediction.gbm_service import GbmPredictionRunResult

    _install_predict_startup(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cli, "run_gbm_prediction",
        lambda *_args, **_kwargs: GbmPredictionRunResult(
            status="insufficient_data", message="Not enough history.", as_of_date="2026-01-01", horizon_days=21,
        ),
    )

    assert cli.main(["predict-v5", "--symbols", "AAPL"]) == int(ExitCode.MISSING_DATA)


def test_predict_v5_command_flags_low_confidence_predictions_in_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from stock_scrapper.prediction.gbm_service import GbmPredictionRunResult, SymbolExcessReturnPrediction

    _install_predict_startup(monkeypatch, tmp_path)
    result = GbmPredictionRunResult(
        status="ok", message=None, as_of_date="2026-01-01", horizon_days=252,
        training_samples=10, holdout_samples=5,
        predictions=[
            SymbolExcessReturnPrediction("AAPL", 0.02, None, low_confidence=False),
            SymbolExcessReturnPrediction("INTC", 2.6611, None, low_confidence=True),
        ],
    )
    monkeypatch.setattr(cli, "run_gbm_prediction", lambda *_args, **_kwargs: result)

    exit_code = cli.main(["predict-v5", "--symbols", "AAPL", "INTC"])

    assert exit_code == int(ExitCode.SUCCESS)
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    aapl_line = next(line for line in lines if line.strip().startswith("AAPL"))
    intc_line = next(line for line in lines if line.strip().startswith("INTC"))
    assert "LOW CONFIDENCE" not in aapl_line
    assert "+266.11%" in intc_line and "LOW CONFIDENCE" in intc_line


def test_predict_v5_command_passes_predict_v5_feature_keys_and_fundamentals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from stock_scrapper.prediction.gbm_service import GbmPredictionRunResult, SymbolExcessReturnPrediction

    _install_predict_startup(monkeypatch, tmp_path)
    requested_symbols: list[str] = []

    def _fake_fetch_fundamentals(_conn: Any, symbol: str) -> list[Any]:
        requested_symbols.append(symbol)
        return []

    monkeypatch.setattr(cli, "fetch_fundamentals", _fake_fetch_fundamentals)
    captured_kwargs: dict[str, Any] = {}

    def _fake_run_gbm_prediction(*_args: Any, **kwargs: Any) -> GbmPredictionRunResult:
        captured_kwargs.update(kwargs)
        return GbmPredictionRunResult(
            status="ok", message=None, as_of_date="2026-01-01", horizon_days=21,
            training_samples=10, holdout_samples=5,
            predictions=[SymbolExcessReturnPrediction("AAPL", 0.01, None)],
        )

    monkeypatch.setattr(cli, "run_gbm_prediction", _fake_run_gbm_prediction)

    exit_code = cli.main(["predict-v5", "--symbols", "AAPL"])

    assert exit_code == int(ExitCode.SUCCESS)
    # predict-v5's own widened feature list (from config/prediction_rules.yaml's
    # predict_v5.feature_keys) must be passed explicitly, not the base feature_keys.
    assert "trailing_pe" in captured_kwargs["feature_keys"]
    assert captured_kwargs["fundamentals_by_symbol"] == {"AAPL": []}
    assert requested_symbols == ["AAPL"]
    # predict-v5's own (heavier) regularization from config/prediction_rules.yaml's
    # predict_v5.gbm section must be passed explicitly, not the shared gbm section.
    assert captured_kwargs["gbm_config"]["min_samples_leaf"] == 200
    captured = capsys.readouterr()
    assert "model=predict-v5" in captured.out
    assert "EXPERIMENTAL STATISTICAL FORECAST" in captured.err


def test_predict_v5_command_horizon_days_override_replaces_config_value(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from stock_scrapper.prediction.gbm_service import GbmPredictionRunResult

    _install_predict_startup(monkeypatch, tmp_path)
    captured_rules: dict[str, Any] = {}

    def _fake_run_gbm_prediction(*_args: Any, **kwargs: Any) -> GbmPredictionRunResult:
        captured_rules.update(kwargs["rules"])
        return GbmPredictionRunResult(
            status="insufficient_data", message="x", as_of_date="2026-01-01",
            horizon_days=kwargs["rules"]["horizon_days"],
        )

    monkeypatch.setattr(cli, "run_gbm_prediction", _fake_run_gbm_prediction)

    cli.main(["predict-v5", "--symbols", "AAPL", "--horizon-days", "5"])

    assert captured_rules["horizon_days"] == 5


def test_predict_v5_command_rejects_nonpositive_horizon_days(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_predict_startup(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["predict-v5", "--symbols", "AAPL", "--horizon-days", "0"])
    assert exc_info.value.code == int(ExitCode.INVALID_ARGUMENTS)


def test_collect_fundamentals_command_persists_records_and_reports_partial_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from stock_scrapper.database import create_connection

    config = _config(tmp_path)
    config["edgar"] = {
        "user_agent": "Stock Scraper Research test@example.com",
        "timeout_seconds": 5, "max_retries": 1, "retry_delay_seconds": 0,
    }
    monkeypatch.setattr(cli, "load_config", lambda _base_dir: config)
    monkeypatch.setattr(cli, "ensure_directories", lambda _config: None)
    monkeypatch.setattr(cli, "load_watchlist", lambda _path: ["AAPL", "MSFT"])
    monkeypatch.setattr(cli, "setup_logging", lambda _config, run_id: _Logger())

    # AAPL resolves to a CIK and returns one usable fact; MSFT has no CIK on file
    # (mirrors a real symbol SEC doesn't recognize) and must be counted as failed,
    # not silently skipped.
    monkeypatch.setattr(cli, "fetch_ticker_cik_map", lambda **_kwargs: {"AAPL": "0000320193"})

    def fake_fetch_company_facts(cik: str, **_kwargs: Any) -> dict[str, Any]:
        assert cik == "0000320193"
        return {
            "facts": {
                "us-gaap": {
                    "NetIncomeLoss": {"units": {"USD": [
                        {
                            "start": "2023-10-01", "end": "2023-12-30", "val": 1000.0,
                            "fy": 2024, "fp": "Q1", "form": "10-Q", "filed": "2024-02-01",
                        },
                    ]}},
                },
            },
        }

    monkeypatch.setattr(cli, "fetch_company_facts", fake_fetch_company_facts)

    exit_code = cli.main(["collect-fundamentals", "--symbols", "AAPL", "MSFT"])

    assert exit_code == int(ExitCode.PARTIAL_FAILURE)
    captured = capsys.readouterr()
    assert "fundamentals_upserted=1 symbols=2 failed=1" in captured.out
    assert "failed_symbols=MSFT" in captured.out


def test_collect_fundamentals_command_treats_zero_facts_as_a_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A resolved CIK that returns no usable facts at all (e.g. SEC's ticker file
    once mapped "XOM" to an unrelated shell entity) must be surfaced as a failure,
    not silently reported as a successful, empty collection."""
    config = _config(tmp_path)
    config["edgar"] = {
        "user_agent": "Stock Scraper Research test@example.com",
        "timeout_seconds": 5, "max_retries": 1, "retry_delay_seconds": 0,
    }
    monkeypatch.setattr(cli, "load_config", lambda _base_dir: config)
    monkeypatch.setattr(cli, "ensure_directories", lambda _config: None)
    monkeypatch.setattr(cli, "load_watchlist", lambda _path: ["XOM"])
    monkeypatch.setattr(cli, "setup_logging", lambda _config, run_id: _Logger())
    monkeypatch.setattr(cli, "fetch_ticker_cik_map", lambda **_kwargs: {"XOM": "0002115436"})
    monkeypatch.setattr(cli, "fetch_company_facts", lambda *_args, **_kwargs: {"facts": {"us-gaap": {}}})

    exit_code = cli.main(["collect-fundamentals", "--symbols", "XOM"])

    assert exit_code == int(ExitCode.PARTIAL_FAILURE)
    captured = capsys.readouterr()
    assert "fundamentals_upserted=0 symbols=1 failed=1" in captured.out
    assert "failed_symbols=XOM" in captured.out
