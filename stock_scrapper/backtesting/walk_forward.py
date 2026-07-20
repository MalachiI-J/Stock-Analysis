"""Deterministic fixed-window walk-forward validation.

This module deliberately does not optimize thresholds. It partitions sorted
trading sessions using the immutable :class:`BacktestConfig`, then invokes a
caller-supplied executor with that exact configuration for each validation
window and one final, non-overlapping holdout window.
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from typing import Callable, Sequence

from stock_scrapper.backtesting.config import BacktestConfig
from stock_scrapper.backtesting.models import PerformanceMetrics, WalkForwardRun, WalkForwardWindow


class InsufficientWalkForwardDataError(ValueError):
    """Raised when fixed windows cannot be formed from available sessions."""


@dataclass(frozen=True, slots=True)
class WalkForwardExecutionResult:
    """Outcome returned by a callback for one fixed evaluation window."""

    backtest_run_id: str | None = None
    metrics: PerformanceMetrics | None = None
    status: str = "completed"
    error_summary: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"completed", "failed"}:
            raise ValueError("Walk-forward callback status must be 'completed' or 'failed'")
        if self.status == "failed" and not self.error_summary:
            raise ValueError("A failed walk-forward callback result requires error_summary")


WindowExecutor = Callable[[WalkForwardWindow, BacktestConfig], WalkForwardExecutionResult | None]


def _coerce_trading_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"Trading date must use YYYY-MM-DD format: {value}") from exc
    raise ValueError(f"Unsupported trading date value: {value!r}")


def normalize_trading_dates(trading_dates: Sequence[str | date | datetime]) -> tuple[date, ...]:
    """Validate a strictly increasing, duplicate-free trading calendar."""
    normalized = tuple(_coerce_trading_date(value) for value in trading_dates)
    if not normalized:
        raise InsufficientWalkForwardDataError("No trading dates were supplied")
    for previous, current in zip(normalized, normalized[1:]):
        if current <= previous:
            raise ValueError("Trading dates must be strictly increasing and contain no duplicates")
    return normalized


def _evaluation_bounds(config: BacktestConfig, dates: tuple[date, ...]) -> tuple[int, int]:
    rules = config.walk_forward
    if config.start_date is None:
        evaluation_start = rules.warm_up_days
    else:
        evaluation_start = bisect_left(dates, config.start_date)
    evaluation_end = len(dates) if config.end_date is None else bisect_right(dates, config.end_date)

    if evaluation_start >= evaluation_end:
        raise InsufficientWalkForwardDataError("No trading sessions fall within the configured start/end range")
    if evaluation_start < rules.warm_up_days:
        raise InsufficientWalkForwardDataError(
            f"Walk-forward start requires {rules.warm_up_days} prior warm-up sessions; "
            f"only {evaluation_start} are available"
        )
    evaluation_sessions = evaluation_end - evaluation_start
    minimum_evaluation_sessions = (
        rules.development_days + rules.validation_days + rules.final_holdout_days
    )
    if evaluation_sessions < minimum_evaluation_sessions:
        raise InsufficientWalkForwardDataError(
            "Insufficient evaluation sessions for one development, validation, and holdout sequence: "
            f"need {minimum_evaluation_sessions}, have {evaluation_sessions}"
        )
    return evaluation_start, evaluation_end


def _default_run_id(config: BacktestConfig, dates: tuple[date, ...], symbols: Sequence[str] = ()) -> str:
    evaluation_start, evaluation_end = _evaluation_bounds(config, dates)
    relevant_start = evaluation_start - config.walk_forward.warm_up_days
    relevant_dates = dates[relevant_start:evaluation_end]
    payload = {
        "configuration_hash": config.configuration_hash,
        "first_trading_date": relevant_dates[0].isoformat(),
        "last_trading_date": relevant_dates[-1].isoformat(),
        "trading_date_count": len(relevant_dates),
        "symbols": sorted({symbol.strip().upper() for symbol in symbols if symbol.strip()}),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return f"wf-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:24]}"


def generate_walk_forward_windows(
    config: BacktestConfig,
    trading_dates: Sequence[str | date | datetime],
    *,
    walk_forward_run_id: str | None = None,
    symbols: Sequence[str] = (),
) -> list[WalkForwardWindow]:
    """Generate fixed rolling validation windows and one reserved holdout.

    ``start_date`` is the first development session; warm-up sessions may occur
    before it. ``end_date`` is inclusive. When either is null, the available
    trading calendar determines that boundary.
    """
    dates = normalize_trading_dates(trading_dates)
    evaluation_start, evaluation_end = _evaluation_bounds(config, dates)
    rules = config.walk_forward
    run_id = walk_forward_run_id or _default_run_id(config, dates, symbols)
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("walk_forward_run_id must be a non-empty string")
    run_id = run_id.strip()

    holdout_start = evaluation_end - rules.final_holdout_days
    windows: list[WalkForwardWindow] = []
    development_start = evaluation_start
    sequence_number = 1

    while True:
        development_end = development_start + rules.development_days - 1
        validation_start = development_end + 1
        validation_end = validation_start + rules.validation_days - 1
        if validation_end >= holdout_start:
            break

        warmup_start = development_start - rules.warm_up_days
        warmup_end = development_start - 1
        windows.append(
            WalkForwardWindow(
                window_id=f"{run_id}-window-{sequence_number:04d}",
                walk_forward_run_id=run_id,
                window_number=sequence_number,
                window_type="validation",
                warm_up_start_date=dates[warmup_start].isoformat(),
                evaluation_start_date=dates[validation_start].isoformat(),
                evaluation_end_date=dates[validation_end].isoformat(),
                status="pending",
                warm_up_end_date=dates[warmup_end].isoformat(),
                development_start_date=dates[development_start].isoformat(),
                development_end_date=dates[development_end].isoformat(),
                validation_start_date=dates[validation_start].isoformat(),
                validation_end_date=dates[validation_end].isoformat(),
            )
        )
        sequence_number += 1
        development_start += rules.step_days

    if not windows:
        # _evaluation_bounds normally prevents this; retain a direct guard in
        # case future configuration adds spacing rules.
        raise InsufficientWalkForwardDataError("No complete validation window fits before the final holdout")

    holdout_development_end = holdout_start - 1
    holdout_development_start = holdout_start - rules.development_days
    holdout_warmup_end = holdout_development_start - 1
    holdout_warmup_start = holdout_development_start - rules.warm_up_days
    if holdout_warmup_start < 0:
        raise InsufficientWalkForwardDataError("Insufficient warm-up history for the final holdout")

    windows.append(
        WalkForwardWindow(
            window_id=f"{run_id}-window-{sequence_number:04d}",
            walk_forward_run_id=run_id,
            window_number=sequence_number,
            window_type="holdout",
            warm_up_start_date=dates[holdout_warmup_start].isoformat(),
            evaluation_start_date=dates[holdout_start].isoformat(),
            evaluation_end_date=dates[evaluation_end - 1].isoformat(),
            status="pending",
            warm_up_end_date=dates[holdout_warmup_end].isoformat(),
            development_start_date=dates[holdout_development_start].isoformat(),
            development_end_date=dates[holdout_development_end].isoformat(),
            holdout_start_date=dates[holdout_start].isoformat(),
            holdout_end_date=dates[evaluation_end - 1].isoformat(),
        )
    )
    return windows


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_timestamp(value: str | None) -> str:
    if value is None:
        return _utc_timestamp()
    if not isinstance(value, str) or not value.strip():
        raise ValueError("timestamp must be a non-empty ISO-8601 string")
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("timestamp must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc).isoformat()


def run_walk_forward(
    config: BacktestConfig,
    trading_dates: Sequence[str | date | datetime],
    executor: WindowExecutor,
    *,
    walk_forward_run_id: str | None = None,
    symbols: Sequence[str] = (),
    timestamp: str | None = None,
) -> WalkForwardRun:
    """Execute each fixed window with one immutable, unoptimized config.

    Callback exceptions are retained on the affected window and execution
    continues, producing a ``completed_with_errors`` run rather than silently
    discarding validation evidence.
    """
    if not callable(executor):
        raise ValueError("executor must be callable")
    dates = normalize_trading_dates(trading_dates)
    normalized_symbols = sorted({symbol.strip().upper() for symbol in symbols if symbol.strip()})
    run_id = walk_forward_run_id or _default_run_id(config, dates, normalized_symbols)
    windows = generate_walk_forward_windows(
        config,
        dates,
        walk_forward_run_id=run_id,
        symbols=normalized_symbols,
    )
    recorded_at = _normalize_timestamp(timestamp)

    expected_config_hash = config.configuration_hash
    completed_windows: list[WalkForwardWindow] = []
    for window in windows:
        callback_window = replace(window)
        boundary_state = (
            callback_window.window_id,
            callback_window.window_number,
            callback_window.window_type,
            callback_window.warm_up_start_date,
            callback_window.warm_up_end_date,
            callback_window.development_start_date,
            callback_window.development_end_date,
            callback_window.validation_start_date,
            callback_window.validation_end_date,
            callback_window.holdout_start_date,
            callback_window.holdout_end_date,
            callback_window.evaluation_start_date,
            callback_window.evaluation_end_date,
        )
        try:
            outcome = executor(callback_window, config)
            if outcome is None:
                outcome = WalkForwardExecutionResult()
            if not isinstance(outcome, WalkForwardExecutionResult):
                raise TypeError("executor must return WalkForwardExecutionResult or None")
            current_boundary_state = (
                callback_window.window_id,
                callback_window.window_number,
                callback_window.window_type,
                callback_window.warm_up_start_date,
                callback_window.warm_up_end_date,
                callback_window.development_start_date,
                callback_window.development_end_date,
                callback_window.validation_start_date,
                callback_window.validation_end_date,
                callback_window.holdout_start_date,
                callback_window.holdout_end_date,
                callback_window.evaluation_start_date,
                callback_window.evaluation_end_date,
            )
            if current_boundary_state != boundary_state:
                raise RuntimeError("executor must not mutate fixed walk-forward window boundaries")
            if config.configuration_hash != expected_config_hash:
                raise RuntimeError("executor must not optimize or mutate the fixed backtesting configuration")
            completed_windows.append(
                replace(
                    window,
                    status=outcome.status,
                    backtest_run_id=outcome.backtest_run_id,
                    metrics=outcome.metrics,
                    error_summary=outcome.error_summary,
                )
            )
        except Exception as exc:  # the failure is retained as validation evidence
            completed_windows.append(
                replace(
                    window,
                    status="failed",
                    error_summary=f"{type(exc).__name__}: {exc}",
                )
            )

    status = "completed" if all(window.status == "completed" for window in completed_windows) else "completed_with_errors"
    return WalkForwardRun(
        walk_forward_run_id=run_id,
        strategy_name=config.strategy_name,
        strategy_version=config.strategy_version,
        configuration_hash=config.configuration_hash,
        start_date=completed_windows[0].development_start_date or completed_windows[0].evaluation_start_date,
        end_date=completed_windows[-1].evaluation_end_date,
        status=status,
        windows=completed_windows,
        started_at=recorded_at,
        completed_at=recorded_at,
        benchmark_symbol=config.benchmark,
        symbols=normalized_symbols,
        configuration_snapshot=config.to_dict(),
    )


# Explicit name emphasizes that no parameter search occurs.
run_fixed_walk_forward = run_walk_forward
