from __future__ import annotations

import pytest

from stock_scrapper.backtesting.config import (
    BacktestConfig,
    EntryThresholds,
    ExitThresholds,
    FinalLiquidationRules,
    WalkForwardRules,
)
from stock_scrapper.models.analysis_models import AnalysisResult
from stock_scrapper.portfolio import (
    HoldingAssessment,
    PortfolioPosition,
    aggregate_open_lots,
    evaluate_holding,
)


def _config(**overrides: object) -> BacktestConfig:
    base = dict(
        strategy_name="score_v1",
        strategy_version="1.1.0",
        benchmark="SPY",
        initial_cash=100000.0,
        warm_up_days=252,
        warmup_policy="shift_start",
        start_date=None,
        end_date=None,
        signal_frequency="daily",
        rebalancing_frequency="daily",
        entry_thresholds=EntryThresholds(
            classifications=("Candidate", "Strong Candidate"),
            minimum_opportunity_score=65.0,
            minimum_average_dollar_volume=1_000_000.0,
        ),
        exit_thresholds=ExitThresholds(
            classifications=("Data Blocked", "Insufficient Data", "High Risk", "Avoid"),
            minimum_opportunity_score=50.0,
            minimum_confidence_score=50.0,
            maximum_risk_score=70.0,
            exit_below_sma200=True,
            exit_on_stress=True,
        ),
        allowed_market_regimes=("Risk-On", "Neutral"),
        minimum_confidence=60.0,
        maximum_risk=70.0,
        maximum_positions=10,
        maximum_position_weight=0.15,
        cash_reserve=0.05,
        fractional_shares=True,
        position_sizing="equal_weight",
        volatility_lookback_days=20,
        commission_basis_points=1.0,
        minimum_commission=0.0,
        slippage_basis_points=5.0,
        stop_loss=0.10,
        trailing_stop=0.15,
        profit_target=None,
        maximum_holding_period=252,
        execution_timing="next_open",
        final_liquidation=FinalLiquidationRules(enabled=True, timing="final_close", apply_costs=True),
        risk_free_rate=0.0,
        annualization_factor=252,
        daily_bar_ambiguity_policy="adverse_first",
        walk_forward=WalkForwardRules(
            warm_up_days=252, development_days=480, validation_days=252, final_holdout_days=252, step_days=252
        ),
    )
    base.update(overrides)
    return BacktestConfig(**base)


def _result(**overrides: object) -> AnalysisResult:
    base = dict(
        symbol="AAPL",
        as_of_date="2026-07-27",
        classification="Candidate",
        market_regime="Neutral",
        opportunity_score=80.0,
        risk_score=30.0,
        confidence_score=90.0,
        primary_reason="Solid trend",
    )
    base.update(overrides)
    return AnalysisResult(**base)


def test_aggregate_open_lots_computes_weighted_average_cost() -> None:
    lots = [
        {"symbol": "aapl", "remaining_shares": 5, "cost_basis_per_share": 100.0, "opened_date": "2026-01-01"},
        {"symbol": "AAPL", "remaining_shares": 10, "cost_basis_per_share": 130.0, "opened_date": "2026-03-01"},
        {"symbol": "AAPL", "remaining_shares": 0, "cost_basis_per_share": 999.0, "opened_date": "2026-04-01"},
    ]
    positions = aggregate_open_lots(lots)
    assert len(positions) == 1
    position = positions[0]
    assert position.symbol == "AAPL"
    assert position.shares == 15.0
    assert position.average_cost_basis == pytest.approx((5 * 100.0 + 10 * 130.0) / 15.0)
    assert position.earliest_opened_date == "2026-01-01"


def test_aggregate_open_lots_groups_multiple_symbols_and_skips_empty() -> None:
    lots = [
        {"symbol": "AAPL", "remaining_shares": 5, "cost_basis_per_share": 100.0, "opened_date": "2026-01-01"},
        {"symbol": "MSFT", "remaining_shares": 2, "cost_basis_per_share": 300.0, "opened_date": "2026-02-01"},
        {"symbol": "TSLA", "remaining_shares": 0, "cost_basis_per_share": 200.0, "opened_date": "2026-02-01"},
    ]
    positions = aggregate_open_lots(lots)
    assert [position.symbol for position in positions] == ["AAPL", "MSFT"]


def test_evaluate_holding_recommends_hold_when_no_exit_signal() -> None:
    position = PortfolioPosition(symbol="AAPL", shares=10, average_cost_basis=100.0, earliest_opened_date="2026-01-01")
    assessment = evaluate_holding(
        position,
        result=_result(),
        backtest_config=_config(),
        latest_price=110.0,
        peak_price_since_entry=115.0,
        holding_period_sessions=20,
    )
    assert assessment.recommendation == "HOLD"
    assert assessment.rule_based_exit_reason is None
    assert assessment.price_stop_reason is None
    assert assessment.unrealized_pnl == pytest.approx(100.0)
    assert assessment.unrealized_pnl_pct == pytest.approx(0.10)


def test_evaluate_holding_recommends_sell_on_avoid_classification() -> None:
    position = PortfolioPosition(symbol="AAPL", shares=10, average_cost_basis=100.0, earliest_opened_date="2026-01-01")
    assessment = evaluate_holding(
        position,
        result=_result(classification="Avoid"),
        backtest_config=_config(),
        latest_price=105.0,
        peak_price_since_entry=105.0,
        holding_period_sessions=20,
    )
    assert assessment.recommendation == "SELL"
    assert "Avoid" in assessment.rule_based_exit_reason


def test_evaluate_holding_recommends_sell_on_stop_loss_even_without_saved_analysis() -> None:
    position = PortfolioPosition(symbol="ZZZZ", shares=10, average_cost_basis=100.0, earliest_opened_date="2026-01-01")
    assessment = evaluate_holding(
        position,
        result=None,
        backtest_config=_config(),
        latest_price=89.0,
        peak_price_since_entry=100.0,
        holding_period_sessions=20,
    )
    assert assessment.recommendation == "SELL"
    assert assessment.rule_based_exit_reason is None
    assert assessment.price_stop_reason == "Stop loss"


def test_evaluate_holding_recommends_sell_on_trailing_stop() -> None:
    position = PortfolioPosition(symbol="AAPL", shares=10, average_cost_basis=100.0, earliest_opened_date="2026-01-01")
    assessment = evaluate_holding(
        position,
        result=_result(),
        backtest_config=_config(),
        latest_price=127.0,
        peak_price_since_entry=150.0,
        holding_period_sessions=20,
    )
    # 150 * (1 - 0.15) = 127.5 trailing-stop trigger; 127.0 <= 127.5 -> triggered.
    assert assessment.recommendation == "SELL"
    assert assessment.price_stop_reason == "Trailing stop"


def test_evaluate_holding_reports_unknown_recommendation_without_price() -> None:
    position = PortfolioPosition(symbol="ZZZZ", shares=10, average_cost_basis=100.0, earliest_opened_date="2026-01-01")
    assessment = evaluate_holding(
        position,
        result=None,
        backtest_config=_config(),
        latest_price=None,
        peak_price_since_entry=None,
        holding_period_sessions=0,
    )
    assert assessment.recommendation == "UNKNOWN (no current price data)"
    assert assessment.market_value is None
    assert assessment.unrealized_pnl is None
