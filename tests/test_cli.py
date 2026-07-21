from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

import main as cli
from stock_scrapper.exceptions import ExitCode, MissingDataError
from stock_scrapper.backtesting.walk_forward import InsufficientWalkForwardDataError
from stock_scrapper.models.analysis_models import AnalysisResult


class _Logger:
    def info(self, *args: Any, **kwargs: Any) -> None:
        pass

    def exception(self, *args: Any, **kwargs: Any) -> None:
        pass


class _Connection:
    def __init__(self) -> None:
        self.closed = False
        self.commits = 0
        self.rollbacks = 0
        self.executed: list[str] = []

    def execute(self, sql: str) -> None:
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

    @property
    def walk_forward(self) -> SimpleNamespace:
        return SimpleNamespace(warm_up_days=7)

    def to_dict(self) -> dict[str, Any]:
        return {"benchmark": self.benchmark, "strategy_name": "score_v1"}

    def with_overrides(self, **overrides: Any) -> SimpleNamespace:
        return SimpleNamespace(benchmark=self.benchmark, **overrides)


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
