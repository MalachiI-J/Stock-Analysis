from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date, timedelta
from pathlib import Path

import pytest

from stock_scrapper.backtesting.config import load_backtesting_config, validate_backtesting_config
from stock_scrapper.backtesting.walk_forward import (
    InsufficientWalkForwardDataError,
    WalkForwardExecutionResult,
    generate_walk_forward_windows,
    normalize_trading_dates,
    run_walk_forward,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config(*, start_date: str | None = None, end_date: str | None = None):
    payload = load_backtesting_config(PROJECT_ROOT).to_dict()
    payload.update(start_date=start_date, end_date=end_date)
    payload["walk_forward"] = {
        "warm_up_days": 2,
        "development_days": 3,
        "validation_days": 2,
        "final_holdout_days": 2,
        "step_days": 2,
    }
    return validate_backtesting_config(payload)


def _dates(count: int, start: date = date(2024, 1, 2)) -> list[str]:
    return [(start + timedelta(days=index)).isoformat() for index in range(count)]


def test_window_boundaries_reserve_a_non_overlapping_final_holdout() -> None:
    trading_dates = _dates(11)
    config = _config(start_date=trading_dates[2], end_date=trading_dates[-1])

    windows = generate_walk_forward_windows(
        config,
        trading_dates,
        walk_forward_run_id="wf-boundaries",
    )

    assert [window.window_type for window in windows] == ["validation", "validation", "holdout"]
    first, second, holdout = windows
    assert (first.warm_up_start_date, first.warm_up_end_date) == (trading_dates[0], trading_dates[1])
    assert (first.development_start_date, first.development_end_date) == (trading_dates[2], trading_dates[4])
    assert (first.validation_start_date, first.validation_end_date) == (trading_dates[5], trading_dates[6])
    assert (second.validation_start_date, second.validation_end_date) == (trading_dates[7], trading_dates[8])
    assert (holdout.holdout_start_date, holdout.holdout_end_date) == (trading_dates[9], trading_dates[10])
    assert holdout.validation_start_date is None
    assert max(window.validation_end_date for window in windows if window.validation_end_date) < holdout.holdout_start_date


def test_window_generation_rejects_insufficient_or_unsorted_sessions() -> None:
    dates = _dates(8)
    config = _config(start_date=dates[2], end_date=dates[-1])
    with pytest.raises(InsufficientWalkForwardDataError, match="need 7, have 6"):
        generate_walk_forward_windows(config, dates)

    enough_dates = _dates(9)
    missing_warmup = _config(start_date=enough_dates[1], end_date=enough_dates[-1])
    with pytest.raises(InsufficientWalkForwardDataError, match="prior warm-up"):
        generate_walk_forward_windows(missing_warmup, enough_dates)

    with pytest.raises(ValueError, match="strictly increasing"):
        normalize_trading_dates(["2024-01-02", "2024-01-02"])
    with pytest.raises(ValueError, match="strictly increasing"):
        normalize_trading_dates(["2024-01-03", "2024-01-02"])


def test_callback_receives_one_fixed_config_without_optimization() -> None:
    trading_dates = _dates(11)
    config = _config(start_date=trading_dates[2], end_date=trading_dates[-1])
    observed_hashes: list[str] = []
    observed_config_ids: list[int] = []
    observed_window_types: list[str] = []

    def execute(window, callback_config):
        observed_hashes.append(callback_config.configuration_hash)
        observed_config_ids.append(id(callback_config))
        observed_window_types.append(window.window_type)
        with pytest.raises(FrozenInstanceError):
            callback_config.maximum_risk = 99
        return WalkForwardExecutionResult(backtest_run_id=f"bt-{window.window_number}")

    result = run_walk_forward(
        config,
        trading_dates,
        execute,
        walk_forward_run_id="wf-fixed",
        symbols=["SPY", "AAPL", "AAPL"],
        timestamp="2025-01-01T00:00:00+00:00",
    )

    assert result.status == "completed"
    assert observed_window_types == ["validation", "validation", "holdout"]
    assert observed_hashes == [config.configuration_hash] * 3
    assert observed_config_ids == [id(config)] * 3
    assert result.symbols == ["AAPL", "SPY"]
    assert [window.backtest_run_id for window in result.windows] == ["bt-1", "bt-2", "bt-3"]


def test_window_generation_and_runner_are_reproducible() -> None:
    trading_dates = _dates(11)
    config = _config(start_date=trading_dates[2], end_date=trading_dates[-1])

    first_windows = generate_walk_forward_windows(config, trading_dates, walk_forward_run_id="wf-repeat")
    second_windows = generate_walk_forward_windows(config, trading_dates, walk_forward_run_id="wf-repeat")
    assert [window.to_dict() for window in first_windows] == [window.to_dict() for window in second_windows]

    def execute(window, callback_config):
        return WalkForwardExecutionResult(backtest_run_id=f"backtest-{window.window_number}")

    first_run = run_walk_forward(
        config,
        trading_dates,
        execute,
        walk_forward_run_id="wf-repeat",
        timestamp="2025-01-01T00:00:00+00:00",
    )
    second_run = run_walk_forward(
        config,
        trading_dates,
        execute,
        walk_forward_run_id="wf-repeat",
        timestamp="2025-01-01T00:00:00+00:00",
    )
    assert first_run.to_dict() == second_run.to_dict()


def test_future_sessions_after_configured_end_do_not_change_windows_or_ids() -> None:
    in_range_dates = _dates(11)
    config = _config(start_date=in_range_dates[2], end_date=in_range_dates[-1])
    extended_dates = _dates(15)

    original = generate_walk_forward_windows(config, in_range_dates)
    with_future_rows = generate_walk_forward_windows(config, extended_dates)

    assert [window.to_dict() for window in original] == [window.to_dict() for window in with_future_rows]


def test_callback_failure_is_retained_and_does_not_erase_other_windows() -> None:
    trading_dates = _dates(11)
    config = _config(start_date=trading_dates[2], end_date=trading_dates[-1])

    def execute(window, callback_config):
        if window.window_number == 2:
            raise RuntimeError("deterministic test failure")
        return WalkForwardExecutionResult(backtest_run_id=f"bt-{window.window_number}")

    result = run_walk_forward(config, trading_dates, execute, timestamp="2025-01-01T00:00:00+00:00")
    assert result.status == "completed_with_errors"
    assert [window.status for window in result.windows] == ["completed", "failed", "completed"]
    assert "deterministic test failure" in (result.windows[1].error_summary or "")
