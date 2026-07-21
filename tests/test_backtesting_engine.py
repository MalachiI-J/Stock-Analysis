from __future__ import annotations

import copy
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import pytest
import yaml

from stock_scrapper.analysis.market_context import MarketContext
from stock_scrapper.analysis.service import AnalysisBatch, AnalysisService
from stock_scrapper.backtesting.config import BacktestConfig, load_backtesting_config, validate_backtesting_config
from stock_scrapper.backtesting.engine import PortfolioBacktestResult, run_portfolio_backtest
from stock_scrapper.models.analysis_models import AnalysisResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config(
    sessions: list[str],
    *,
    initial_cash: float = 1000.0,
    maximum_positions: int = 1,
    maximum_position_weight: float = 0.5,
    cash_reserve: float = 0.1,
    commission_bps: float = 0.0,
    minimum_commission: float = 0.0,
    slippage_bps: float = 0.0,
    stop_loss: float | None = None,
    trailing_stop: float | None = None,
    profit_target: float | None = None,
    final_liquidation: bool = True,
) -> BacktestConfig:
    payload = load_backtesting_config(PROJECT_ROOT).to_dict()
    payload.update(
        initial_cash=initial_cash,
        warm_up_days=2,
        warmup_policy="allow_with_warning",
        start_date=sessions[0],
        end_date=sessions[-1],
        signal_frequency="daily",
        rebalancing_frequency="daily",
        minimum_confidence=0.0,
        maximum_risk=100.0,
        maximum_positions=maximum_positions,
        maximum_position_weight=maximum_position_weight,
        cash_reserve=cash_reserve,
        fractional_shares=True,
        position_sizing="equal_weight",
        commission_basis_points=commission_bps,
        minimum_commission=minimum_commission,
        slippage_basis_points=slippage_bps,
        stop_loss=stop_loss,
        trailing_stop=trailing_stop,
        profit_target=profit_target,
        maximum_holding_period=None,
        execution_timing="next_open",
    )
    payload["entry_thresholds"] = {
        "classifications": ["Candidate", "Strong Candidate"],
        "minimum_opportunity_score": 0.0,
        "minimum_average_dollar_volume": 0.0,
    }
    payload["exit_thresholds"] = {
        "classifications": ["Data Blocked", "Insufficient Data", "High Risk", "Avoid"],
        "minimum_opportunity_score": 0.0,
        "minimum_confidence_score": 0.0,
        "maximum_risk_score": 100.0,
        "exit_below_sma200": True,
        "exit_on_stress": True,
    }
    payload["allowed_market_regimes"] = ["Risk-On", "Neutral"]
    payload["final_liquidation"] = {
        "enabled": final_liquidation,
        "timing": "final_close",
        "apply_costs": True,
    }
    return validate_backtesting_config(payload)


def _row(
    symbol: str,
    trade_date: str,
    *,
    open_price: float | None = 100.0,
    high: float | None = None,
    low: float | None = None,
    close: float = 100.0,
    volume: int = 1_000_000,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "trade_date": trade_date,
        "open": open_price,
        "high": high if high is not None else max(close, open_price or close) + 1.0,
        "low": low if low is not None else min(close, open_price or close) - 1.0,
        "close": close,
        "adjusted_close": close,
        "volume": volume,
        "dividends": 0.0,
        "stock_splits": 0.0,
    }


def _histories(
    symbols: list[str],
    sessions: list[str],
    overrides: dict[tuple[str, str], dict[str, object]] | None = None,
) -> dict[str, list[dict[str, object]]]:
    overrides = overrides or {}
    result: dict[str, list[dict[str, object]]] = {}
    for symbol in [*symbols, "SPY"]:
        rows: list[dict[str, object]] = []
        for session in sessions:
            values = overrides.get((symbol, session), {})
            rows.append(_row(symbol, session, **values))
        result[symbol] = rows
    return result


ResultProvider = Callable[[str, str, list[dict[str, object]]], AnalysisResult]


def _candidate_result(symbol: str, as_of: str, history: list[dict[str, object]]) -> AnalysisResult:
    available = [row for row in history if str(row["trade_date"]) <= as_of]
    latest = available[-1]
    return AnalysisResult(
        symbol=symbol,
        as_of_date=as_of,
        data_through_date=str(latest["trade_date"]),
        market_regime="Risk-On",
        market_regime_confidence=100.0,
        risk_score=20.0,
        risk_level="Low",
        opportunity_score=80.0,
        confidence_score=90.0,
        classification="Candidate",
        primary_reason="Synthetic canonical candidate",
        eligible_for_scoring=True,
        indicators={
            "latest_close": latest["adjusted_close"],
            "distance_from_sma200": 0.1,
            "benchmark_relative_return_252": 0.2,
            "twenty_day_average_dollar_volume": 100_000_000.0,
            "twenty_day_volatility": 0.2,
        },
    )


def _install_analysis(
    monkeypatch: pytest.MonkeyPatch,
    provider: ResultProvider = _candidate_result,
) -> None:
    def analyze_loaded_many_as_of(
        self,
        symbols,
        histories,
        as_of_date,
        *,
        quality_by_symbol=None,
        persist=False,
    ):
        as_of = str(as_of_date)[:10]
        results = [provider(symbol, as_of, histories[symbol]) for symbol in symbols]
        return AnalysisBatch(
            results=results,
            market_context=MarketContext("Risk-On", 100.0, {"breadth_ratio": 1.0}, ["Synthetic context"]),
            as_of_date=as_of,
            data_through_date=max(result.data_through_date for result in results if result.data_through_date),
            configuration_hash=self.configuration_hash,
        )

    monkeypatch.setattr(AnalysisService, "analyze_loaded_many_as_of", analyze_loaded_many_as_of)


def _run(
    monkeypatch: pytest.MonkeyPatch,
    sessions: list[str],
    *,
    symbols: list[str] | None = None,
    histories: dict[str, list[dict[str, object]]] | None = None,
    config: BacktestConfig | None = None,
    run_id: str = "test-run",
) -> PortfolioBacktestResult:
    symbols = symbols or ["AAA"]
    _install_analysis(monkeypatch)
    return run_portfolio_backtest(
        symbols,
        histories or _histories(symbols, sessions),
        _analysis_rules(),
        config or _config(sessions),
        run_id=run_id,
    )


def _analysis_rules() -> dict[str, object]:
    with (PROJECT_ROOT / "config" / "scoring_rules.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _business_dates(start: date, count: int) -> list[str]:
    values: list[str] = []
    current = start
    while len(values) < count:
        if current.weekday() < 5:
            values.append(current.isoformat())
        current += timedelta(days=1)
    return values


def _trend_history(symbol: str, sessions: list[str], start_price: float, daily_gain: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    price = start_price
    for session in sessions:
        price *= 1.0 + daily_gain
        rows.append(
            _row(
                symbol,
                session,
                open_price=price * 0.999,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=2_000_000,
            )
        )
    return rows


def _core_records(result: PortfolioBacktestResult) -> dict[str, object]:
    return {
        "signals": [
            {key: value for key, value in item.to_dict().items() if key not in {"run_id", "signal_id", "created_at"}}
            for item in result.signals
        ],
        "orders": [
            {
                key: value
                for key, value in item.to_dict().items()
                if key not in {"run_id", "order_id", "signal_id", "created_at", "cancelled_at"}
            }
            for item in result.orders
        ],
        "fills": [
            {key: value for key, value in item.to_dict().items() if key not in {"run_id", "fill_id", "order_id", "created_at"}}
            for item in result.fills
        ],
        "trades": [
            {
                key: value
                for key, value in item.to_dict().items()
                if key
                not in {
                    "run_id",
                    "trade_id",
                    "entry_signal_id",
                    "entry_order_id",
                    "exit_order_id",
                }
            }
            for item in result.trades
        ],
        "snapshots": [
            {key: value for key, value in item.to_dict().items() if key != "run_id"}
            for item in result.snapshots
        ],
        "metrics": result.metrics.to_dict() if result.metrics else None,
    }


def test_close_signal_executes_at_next_available_open_after_weekend_and_holiday(monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = ["2024-01-05", "2024-01-09", "2024-01-10"]  # Friday, then Tuesday
    histories = _histories(
        ["AAA"],
        sessions,
        {("AAA", "2024-01-09"): {"open_price": 103.0, "high": 105.0, "low": 102.0, "close": 104.0}},
    )
    result = _run(monkeypatch, sessions, histories=histories)

    entry_order = next(order for order in result.orders if order.side == "buy")
    entry_fill = next(fill for fill in result.fills if fill.side == "buy")
    trade = result.trades[0]
    assert entry_order.signal_date == "2024-01-05"
    assert entry_order.scheduled_execution_date == "2024-01-09"
    assert entry_fill.execution_date == "2024-01-09"
    assert entry_fill.reference_price == 103.0
    assert trade.signal_date == "2024-01-05"
    assert trade.execution_date == "2024-01-09"
    assert trade.execution_date > trade.signal_date


def test_missing_next_open_rejects_order_without_a_fill(monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = ["2024-01-05", "2024-01-09"]
    histories = _histories(
        ["AAA"],
        sessions,
        {("AAA", "2024-01-09"): {"open_price": None, "high": 102.0, "low": 99.0, "close": 101.0}},
    )
    result = _run(monkeypatch, sessions, histories=histories)

    assert len(result.orders) == 1
    assert result.orders[0].status == "rejected"
    assert result.orders[0].rejection_reason == "Adjusted next-session open is unavailable"
    assert result.fills == []
    assert result.trades == []
    assert any("open is unavailable" in candidate.reason for candidate in result.rejected_candidates)


def test_slippage_commission_and_portfolio_accounting(monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = ["2024-01-02", "2024-01-03", "2024-01-04"]
    histories = _histories(
        ["AAA"],
        sessions,
        {
            ("AAA", "2024-01-03"): {"open_price": 100.0, "high": 106.0, "low": 99.0, "close": 105.0},
            ("AAA", "2024-01-04"): {"open_price": 109.0, "high": 111.0, "low": 108.0, "close": 110.0},
        },
    )
    config = _config(sessions, commission_bps=100.0, minimum_commission=2.0, slippage_bps=100.0)
    result = _run(monkeypatch, sessions, histories=histories, config=config)

    buy, sell = result.fills
    assert buy.fill_price == pytest.approx(buy.reference_price * 1.01)
    assert sell.fill_price == pytest.approx(sell.reference_price * 0.99)
    assert buy.commission == pytest.approx(buy.notional * 0.01)
    assert sell.commission == pytest.approx(sell.notional * 0.01)
    assert buy.slippage > 0 and sell.slippage > 0

    holding_snapshot = result.snapshots[1]
    final_snapshot = result.snapshots[-1]
    assert holding_snapshot.unrealized_pnl == pytest.approx(
        holding_snapshot.market_value - buy.quantity * buy.fill_price
    )
    assert holding_snapshot.equity == pytest.approx(holding_snapshot.cash + holding_snapshot.market_value)
    assert final_snapshot.position_count == 0
    assert final_snapshot.realized_pnl == pytest.approx(result.trades[0].realized_pnl)
    assert final_snapshot.equity == pytest.approx(config.initial_cash + result.trades[0].realized_pnl)
    assert all(snapshot.cash >= -1e-9 for snapshot in result.snapshots)
    assert all(snapshot.gross_exposure <= 1.0 + 1e-9 for snapshot in result.snapshots)


@pytest.mark.parametrize(
    ("kind", "overrides", "expected_reason"),
    [
        (
            "stop",
            {
                "config": {"stop_loss": 0.10},
                "day2": {"open_price": 100.0, "high": 102.0, "low": 95.0, "close": 100.0},
                "day3": {"open_price": 95.0, "high": 101.0, "low": 85.0, "close": 94.0},
            },
            "Stop loss",
        ),
        (
            "ambiguous",
            {
                "config": {"stop_loss": 0.10, "profit_target": 0.10},
                "day2": {"open_price": 100.0, "high": 102.0, "low": 95.0, "close": 100.0},
                "day3": {"open_price": 100.0, "high": 115.0, "low": 85.0, "close": 105.0},
            },
            "Stop loss",
        ),
    ],
)
def test_stop_and_ambiguous_bar_use_conservative_fill(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    overrides: dict[str, object],
    expected_reason: str,
) -> None:
    sessions = ["2024-01-02", "2024-01-03", "2024-01-04"]
    histories = _histories(
        ["AAA"],
        sessions,
        {
            ("AAA", sessions[1]): overrides["day2"],
            ("AAA", sessions[2]): overrides["day3"],
        },
    )
    config = _config(sessions, **overrides["config"])
    result = _run(monkeypatch, sessions, histories=histories, config=config)
    trade = result.trades[0]

    assert trade.exit_reason == expected_reason
    assert trade.exit_reference_price == pytest.approx(90.0)
    assert trade.exit_fill_price == pytest.approx(90.0)
    assert trade.ambiguous_daily_bar is (kind == "ambiguous")
    assert trade.ambiguity_policy == ("adverse_first" if kind == "ambiguous" else None)


def test_trailing_stop_tracks_high_water_mark(monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    histories = _histories(
        ["AAA"],
        sessions,
        {
            ("AAA", sessions[1]): {"open_price": 100.0, "high": 110.0, "low": 95.0, "close": 108.0},
            ("AAA", sessions[2]): {"open_price": 110.0, "high": 120.0, "low": 105.0, "close": 118.0},
            ("AAA", sessions[3]): {"open_price": 115.0, "high": 116.0, "low": 107.0, "close": 110.0},
        },
    )
    config = _config(sessions, trailing_stop=0.10)
    result = _run(monkeypatch, sessions, histories=histories, config=config)
    trade = result.trades[0]

    assert trade.exit_reason == "Trailing stop"
    assert trade.exit_reference_price == pytest.approx(108.0)
    assert trade.exit_execution_date == sessions[-1]


def test_final_liquidation_records_reason_and_closes_all_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = ["2024-01-02", "2024-01-03", "2024-01-04"]
    result = _run(monkeypatch, sessions)

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "Final liquidation"
    assert result.trades[0].exit_execution_date == sessions[-1]
    assert result.snapshots[-1].position_count == 0
    assert result.snapshots[-1].market_value == 0


def test_shared_cash_limits_reserve_weights_ranking_and_tie_break(monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = ["2024-01-02", "2024-01-03", "2024-01-04"]
    symbols = ["CCC", "BBB", "AAA"]
    config = _config(
        sessions,
        maximum_positions=2,
        maximum_position_weight=0.4,
        cash_reserve=0.2,
    )
    result = _run(monkeypatch, sessions, symbols=symbols, config=config)

    first_day_ranked = [candidate for candidate in result.ranked_candidates if candidate.signal_date == sessions[0]]
    assert [candidate.symbol for candidate in first_day_ranked] == ["AAA", "BBB", "CCC"]
    assert [candidate.rank for candidate in first_day_ranked] == [1, 2, 3]
    assert any(
        candidate.symbol == "CCC" and "position limit" in candidate.reason.lower()
        for candidate in result.rejected_candidates
    )

    signal_snapshot = result.snapshots[0]
    holding_snapshot = result.snapshots[1]
    assert signal_snapshot.reserved_cash == pytest.approx(800.0)
    assert holding_snapshot.position_count == 2
    assert holding_snapshot.cash == pytest.approx(200.0)
    assert holding_snapshot.market_value == pytest.approx(800.0)
    assert holding_snapshot.equity == pytest.approx(1000.0)
    assert holding_snapshot.gross_exposure == pytest.approx(0.8)
    assert all(fill.notional / config.initial_cash <= config.maximum_position_weight for fill in result.fills if fill.side == "buy")
    assert all(snapshot.cash >= 0 for snapshot in result.snapshots)
    assert all(snapshot.market_value <= snapshot.equity + 1e-9 for snapshot in result.snapshots)


def test_gap_up_is_resized_at_fill_to_respect_maximum_position_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = ["2024-01-02", "2024-01-03", "2024-01-04"]
    histories = _histories(
        ["AAA"],
        sessions,
        {
            ("AAA", sessions[1]): {
                "open_price": 200.0,
                "high": 202.0,
                "low": 198.0,
                "close": 200.0,
            }
        },
    )
    config = _config(
        sessions,
        maximum_position_weight=0.5,
        cash_reserve=0.0,
    )
    result = _run(monkeypatch, sessions, histories=histories, config=config)

    entry_fill = next(fill for fill in result.fills if fill.side == "buy")
    assert entry_fill.quantity == pytest.approx(2.5)
    assert entry_fill.notional <= config.initial_cash * config.maximum_position_weight + 1e-9
    assert result.snapshots[1].gross_exposure <= config.maximum_position_weight + 1e-9


def test_distinct_signal_and_rebalance_schedules_still_create_month_end_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = _business_dates(date(2024, 1, 2), 24)
    payload = _config(sessions).to_dict()
    payload.update(signal_frequency="weekly", rebalancing_frequency="monthly")
    config = validate_backtesting_config(payload)
    result = _run(monkeypatch, sessions, config=config)

    entry_order = next(order for order in result.orders if order.side == "buy")
    assert entry_order.signal_date == "2024-01-31"
    assert entry_order.scheduled_execution_date == "2024-02-01"


def test_configured_volatility_lookback_controls_position_sizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = ["2024-01-02", "2024-01-03", "2024-01-04"]

    def provider(symbol: str, as_of: str, history: list[dict[str, object]]) -> AnalysisResult:
        result = _candidate_result(symbol, as_of, history)
        result.indicators["sixty_day_volatility"] = 0.4
        result.indicators.pop("twenty_day_volatility")
        return result

    payload = _config(sessions).to_dict()
    payload.update(position_sizing="volatility_adjusted", volatility_lookback_days=60)
    config = validate_backtesting_config(payload)
    _install_analysis(monkeypatch, provider)
    result = run_portfolio_backtest(
        ["AAA"],
        _histories(["AAA"], sessions),
        _analysis_rules(),
        config,
        run_id="volatility-lookback",
    )

    assert any(fill.side == "buy" for fill in result.fills)


def test_final_liquidation_fails_loudly_when_adjusted_close_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = ["2024-01-02", "2024-01-03", "2024-01-04"]
    histories = _histories(["AAA"], sessions)
    histories["AAA"][-1]["adjusted_close"] = None

    with pytest.raises(ValueError, match="Final liquidation failed for AAA"):
        _run(monkeypatch, sessions, histories=histories)


def test_unaffordable_order_is_rejected_before_cash_can_go_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = ["2024-01-02", "2024-01-03"]
    config = _config(sessions, minimum_commission=700.0)
    result = _run(monkeypatch, sessions, config=config)

    assert result.orders == []
    assert result.fills == []
    assert result.trades == []
    assert any("unaffordable" in candidate.reason.lower() for candidate in result.rejected_candidates)
    assert all(snapshot.cash == config.initial_cash for snapshot in result.snapshots)


def test_real_analysis_and_engine_ignore_future_context_and_breadth_rows() -> None:
    sessions = _business_dates(date(2023, 1, 2), 265)
    evaluation = sessions[-4:]
    symbols = ["AAA", "BBB"]
    histories = {
        "AAA": _trend_history("AAA", sessions, 50.0, 0.0010),
        "BBB": _trend_history("BBB", sessions, 70.0, 0.0008),
        "SPY": _trend_history("SPY", sessions, 100.0, 0.0006),
        "QQQ": _trend_history("QQQ", sessions, 90.0, 0.0007),
        "IWM": _trend_history("IWM", sessions, 80.0, 0.0004),
    }
    future_sessions = _business_dates(date.fromisoformat(sessions[-1]) + timedelta(days=1), 3)
    with_future = copy.deepcopy(histories)
    for symbol, price in (("SPY", 10.0), ("QQQ", 5.0), ("IWM", 4.0), ("BBB", 1.0)):
        with_future[symbol].extend(
            [_row(symbol, session, open_price=price, high=price, low=price, close=price) for session in future_sessions]
        )

    rules = _analysis_rules()
    service = AnalysisService(None, rules, list(histories))
    baseline_batch = service.analyze_loaded_many_as_of(symbols, histories, evaluation[-1])
    future_batch = service.analyze_loaded_many_as_of(symbols, with_future, evaluation[-1])
    assert asdict(baseline_batch.market_context) == asdict(future_batch.market_context)
    assert [asdict(result) for result in baseline_batch.results] == [asdict(result) for result in future_batch.results]

    config = _config(evaluation, stop_loss=None, trailing_stop=None)
    baseline_run = run_portfolio_backtest(symbols, histories, rules, config, run_id="baseline")
    future_run = run_portfolio_backtest(symbols, with_future, rules, config, run_id="with-future")
    assert baseline_run.run.deterministic_result_hash == future_run.run.deterministic_result_hash
    assert _core_records(baseline_run) == _core_records(future_run)


def test_repeated_engine_runs_are_reproducible_across_run_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = ["2024-01-02", "2024-01-03", "2024-01-04"]
    symbols = ["AAA", "BBB"]
    histories = _histories(symbols, sessions)
    config = _config(sessions, maximum_positions=2, maximum_position_weight=0.4, cash_reserve=0.2)
    _install_analysis(monkeypatch)

    first = run_portfolio_backtest(symbols, histories, _analysis_rules(), config, run_id="run-one")
    second = run_portfolio_backtest(symbols, histories, _analysis_rules(), config, run_id="run-two")

    assert first.run.run_id != second.run.run_id
    assert first.run.deterministic_result_hash == second.run.deterministic_result_hash
    assert _core_records(first) == _core_records(second)
