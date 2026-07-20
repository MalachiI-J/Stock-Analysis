from __future__ import annotations

from pathlib import Path

import pytest

from stock_scrapper.backtesting.config import (
    BacktestConfig,
    canonical_config_json,
    configuration_hash,
    load_backtesting_config,
    validate_backtesting_config,
)
from stock_scrapper.backtesting.models import (
    BacktestRun,
    PortfolioSnapshot,
    Position,
    Trade,
    WalkForwardRun,
    WalkForwardWindow,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_actual_backtesting_yaml_loads_and_contains_conservative_defaults() -> None:
    config = load_backtesting_config(PROJECT_ROOT)

    assert isinstance(config, BacktestConfig)
    assert config.strategy_name == "score_v1"
    assert config.benchmark == "SPY"
    assert config.execution_timing == "next_open"
    assert config.daily_bar_ambiguity_policy == "adverse_first"
    assert config.maximum_positions == 10
    assert 0 < config.maximum_position_weight <= 1 - config.cash_reserve
    assert config.entry_thresholds.minimum_opportunity_score >= config.exit_thresholds.minimum_opportunity_score
    assert len(config.configuration_hash) == 64


def test_backtesting_configuration_hash_is_stable_and_sensitive() -> None:
    config = load_backtesting_config(PROJECT_ROOT)
    payload = config.to_dict()
    reordered = dict(reversed(list(payload.items())))

    assert configuration_hash(payload) == configuration_hash(reordered)
    assert canonical_config_json(payload) == canonical_config_json(reordered)
    assert config.with_overrides(initial_cash=200000).configuration_hash != config.configuration_hash


def test_backtesting_configuration_rejects_unknown_missing_and_unsafe_values() -> None:
    payload = load_backtesting_config(PROJECT_ROOT).to_dict()

    unknown = dict(payload)
    unknown["mystery_setting"] = True
    with pytest.raises(ValueError, match="unknown mystery_setting"):
        validate_backtesting_config(unknown)

    missing = dict(payload)
    missing.pop("execution_timing")
    with pytest.raises(ValueError, match="missing execution_timing"):
        validate_backtesting_config(missing)

    same_close = dict(payload)
    same_close["execution_timing"] = "same_close"
    with pytest.raises(ValueError, match="execution_timing"):
        validate_backtesting_config(same_close)

    invalid_dates = dict(payload)
    invalid_dates.update(start_date="2025-01-01", end_date="2024-01-01")
    with pytest.raises(ValueError, match="start_date"):
        validate_backtesting_config(invalid_dates)

    invalid_reserve = dict(payload)
    invalid_reserve["cash_reserve"] = 1.0
    with pytest.raises(ValueError, match="cash_reserve"):
        validate_backtesting_config(invalid_reserve)

    unsupported_liquidation = dict(payload)
    unsupported_liquidation["final_liquidation"] = {
        "enabled": True,
        "timing": "next_open",
        "apply_costs": True,
    }
    with pytest.raises(ValueError, match="final_liquidation.timing"):
        validate_backtesting_config(unsupported_liquidation)

    unsupported_volatility = dict(payload)
    unsupported_volatility["volatility_lookback_days"] = 30
    with pytest.raises(ValueError, match="volatility_lookback_days must be one of"):
        validate_backtesting_config(unsupported_volatility)


def test_typed_position_and_trade_retain_required_audit_fields() -> None:
    position = Position(
        symbol="AAPL",
        quantity=10,
        average_cost=100,
        entry_date="2024-01-02",
        market_price=110,
        highest_price=112,
    )
    assert position.market_value == 1100
    assert position.unrealized_pnl == 100

    trade = Trade(
        trade_id="trade-1",
        run_id="run-1",
        symbol="AAPL",
        signal_date="2024-01-02",
        execution_date="2024-01-03",
        opportunity_score=75,
        risk_score=30,
        confidence_score=80,
        classification="Candidate",
        market_regime="Risk-On",
        ranking_values={"rank": 1},
        quantity=10,
        reference_price=100,
        fill_price=100.05,
        commission=1,
        slippage=0.5,
        entry_reason="Ranked first",
        exit_reason="Opportunity fell below threshold",
        strategy_version="1.0.0",
        configuration_hash="a" * 64,
        exit_execution_date="2024-02-01",
        exit_fill_price=110,
        exit_commission=1,
        exit_slippage=0.5,
    )
    payload = trade.to_dict()
    assert payload["signal_date"] == "2024-01-02"
    assert payload["execution_date"] == "2024-01-03"
    assert payload["ranking_values"] == {"rank": 1}
    assert trade.total_commission == 2
    assert trade.total_slippage == 1


def test_run_snapshot_and_walk_forward_models_map_to_persistence_columns() -> None:
    run = BacktestRun(
        run_id="run-1",
        strategy_name="score_v1",
        strategy_version="1.0.0",
        configuration_hash="a" * 64,
        benchmark_symbol="SPY",
        start_date="2024-01-01",
        end_date="2024-12-31",
        warm_up_start_date="2023-01-01",
        initial_cash=100000,
        symbols=["AAPL", "SPY"],
        status="completed",
        configuration_snapshot={"strategy_name": "score_v1"},
        started_at="2024-01-01T00:00:00+00:00",
        price_data_hash="b" * 64,
        deterministic_result_hash="c" * 64,
    )
    run_record = run.to_persistence_record()
    assert run_record["data_hash"] == "b" * 64
    assert run_record["deterministic_result_hash"] == "c" * 64
    assert run_record["warmup_start_date"] == "2023-01-01"

    snapshot = PortfolioSnapshot(
        run_id="run-1",
        snapshot_date="2024-01-02",
        cash=50000,
        reserved_cash=0,
        market_value=51000,
        equity=101000,
        gross_exposure=0.505,
        position_count=1,
        realized_pnl=0,
        unrealized_pnl=1000,
        commissions=1,
        slippage=5,
        daily_return=0.01,
        benchmark_equity=100500,
    )
    assert snapshot.to_persistence_record()["benchmark_equity"] == 100500

    window = WalkForwardWindow(
        window_id="window-1",
        walk_forward_run_id="wf-1",
        window_number=1,
        window_type="validation",
        warm_up_start_date="2022-01-01",
        evaluation_start_date="2024-01-01",
        evaluation_end_date="2024-12-31",
        status="completed",
        development_start_date="2023-01-01",
        development_end_date="2023-12-31",
        validation_start_date="2024-01-01",
        validation_end_date="2024-12-31",
    )
    assert window.to_persistence_record()["sequence_number"] == 1
    assert window.to_persistence_record()["validation_start"] == "2024-01-01"

    holdout = WalkForwardWindow(
        window_id="window-2",
        walk_forward_run_id="wf-1",
        window_number=2,
        window_type="holdout",
        warm_up_start_date="2023-01-01",
        evaluation_start_date="2025-01-01",
        evaluation_end_date="2025-12-31",
        status="completed",
    )
    holdout_record = holdout.to_persistence_record()
    assert holdout_record["validation_start"] is None
    assert holdout_record["validation_end"] is None
    assert holdout_record["holdout_start"] == "2025-01-01"
    assert holdout_record["holdout_end"] == "2025-12-31"

    walk_run = WalkForwardRun(
        walk_forward_run_id="wf-1",
        strategy_name="score_v1",
        strategy_version="1.0.0",
        configuration_hash="a" * 64,
        start_date="2023-01-01",
        end_date="2024-12-31",
        status="completed",
        windows=[window],
        started_at="2025-01-01T00:00:00+00:00",
        configuration_snapshot={"strategy_name": "score_v1"},
    )
    assert walk_run.to_persistence_record()["run_id"] == "wf-1"
