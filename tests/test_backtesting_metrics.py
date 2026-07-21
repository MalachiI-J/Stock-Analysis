from __future__ import annotations

from math import isclose, sqrt
from statistics import mean, stdev

from stock_scrapper.backtesting.metrics import (
    calculate_annualized_volatility,
    calculate_cagr,
    calculate_calmar_ratio,
    calculate_drawdown,
    calculate_performance_metrics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_total_return,
)


def test_core_return_drawdown_and_annualized_metrics() -> None:
    returns = [0.01, -0.005, 0.015, -0.01]

    assert isclose(calculate_total_return(100, 121) or 0, 0.21)
    assert isclose(calculate_cagr(100, 121, 2) or 0, 0.1, rel_tol=1e-12)
    assert isclose(
        calculate_annualized_volatility(returns) or 0,
        stdev(returns) * sqrt(252),
        rel_tol=1e-12,
    )
    assert isclose(
        calculate_sharpe_ratio(returns) or 0,
        mean(returns) / stdev(returns) * sqrt(252),
        rel_tol=1e-12,
    )
    downside = sqrt(mean(min(value, 0) ** 2 for value in returns))
    assert isclose(
        calculate_sortino_ratio(returns) or 0,
        mean(returns) / downside * sqrt(252),
        rel_tol=1e-12,
    )

    maximum_drawdown, duration = calculate_drawdown([100, 120, 90, 95, 121, 100])
    assert isclose(maximum_drawdown or 0, 0.25)
    assert duration == 2
    assert isclose(calculate_calmar_ratio(0.10, 0.25) or 0, 0.4)


def test_complete_performance_metrics_trade_stats_periods_and_benchmark() -> None:
    equity = [
        {"date": "2023-12-29", "equity": 100.0, "gross_exposure": 0.0},
        {"date": "2024-01-02", "equity": 110.0, "gross_exposure": 0.5},
        {"date": "2024-01-31", "equity": 105.0, "gross_exposure": 0.8},
        {"date": "2024-02-29", "equity": 120.0, "gross_exposure": 0.7},
        {"date": "2024-12-31", "equity": 121.0, "gross_exposure": 0.4},
    ]
    benchmark = [
        ("2023-12-29", 100.0),
        ("2024-01-02", 102.0),
        ("2024-01-31", 101.0),
        ("2024-02-29", 105.0),
        ("2024-12-31", 110.0),
    ]
    trades = [
        {
            "quantity": 2,
            "fill_price": 100,
            "exit_fill_price": 110,
            "realized_pnl": 20,
            "commission": 1,
            "exit_commission": 1,
            "slippage": 0.5,
            "exit_slippage": 0.5,
            "holding_period_days": 10,
        },
        {
            "quantity": 1,
            "fill_price": 120,
            "exit_fill_price": 110,
            "realized_pnl": -10,
            "commission": 1,
            "exit_commission": 1,
            "slippage": 0.25,
            "exit_slippage": 0.25,
            "holding_period_days": 20,
        },
        {
            "quantity": 1,
            "fill_price": 100,
            "exit_fill_price": 105,
            "realized_pnl": 5,
            "commission": 1,
            "exit_commission": 1,
            "slippage": 0.25,
            "exit_slippage": 0.25,
            "holding_period_days": 15,
        },
    ]

    metrics = calculate_performance_metrics(equity, trades, benchmark_curve=benchmark)

    assert metrics.starting_equity == 100
    assert metrics.ending_equity == 121
    assert metrics.net_profit == 21
    assert isclose(metrics.total_return or 0, 0.21)
    assert isclose(metrics.maximum_drawdown or 0, (110 - 105) / 110)
    assert metrics.drawdown_duration == 1
    assert metrics.number_of_trades == 3
    assert isclose(metrics.win_rate or 0, 2 / 3)
    assert metrics.average_win == 12.5
    assert metrics.average_loss == -10
    assert metrics.best_trade == 20
    assert metrics.worst_trade == -10
    assert metrics.profit_factor == 2.5
    assert metrics.expectancy == 5
    assert metrics.average_holding_period == 15
    assert metrics.consecutive_wins == 1
    assert metrics.consecutive_losses == 1
    assert metrics.commission_cost == 6
    assert metrics.slippage_cost == 2
    assert isclose(metrics.exposure or 0, 0.48)
    assert isclose(metrics.monthly_returns["2024-01"], 0.05)
    assert isclose(metrics.monthly_returns["2024-02"], 120 / 105 - 1)
    assert isclose(metrics.annual_returns["2024"], 0.21)
    assert isclose(metrics.benchmark_total_return or 0, 0.10)
    assert isclose(metrics.return_vs_benchmark or 0, 0.11)


def test_metrics_handle_zero_denominators_without_infinities() -> None:
    assert calculate_total_return(0, 100) is None
    assert calculate_cagr(0, 100, 1) is None
    assert calculate_cagr(100, 100, 0) is None
    assert calculate_sharpe_ratio([0.0, 0.0, 0.0]) is None
    assert calculate_sortino_ratio([0.01, 0.02]) is None
    assert calculate_calmar_ratio(0.1, 0.0) is None

    metrics = calculate_performance_metrics([100.0, 100.0, 100.0])
    assert metrics.total_return == 0
    assert metrics.maximum_drawdown == 0
    assert metrics.sharpe_ratio is None
    assert metrics.sortino_ratio is None
    assert metrics.calmar_ratio is None
    assert metrics.win_rate is None
    assert metrics.profit_factor is None
    assert metrics.turnover == 0


def test_snapshot_benchmark_equity_is_used_when_no_separate_curve_is_passed() -> None:
    snapshots = [
        {"date": "2024-01-02", "equity": 100, "gross_exposure": 0, "benchmark_equity": 100},
        {"date": "2024-01-03", "equity": 110, "gross_exposure": 1, "benchmark_equity": 105},
    ]
    metrics = calculate_performance_metrics(snapshots)
    assert isclose(metrics.benchmark_total_return or 0, 0.05)
    assert isclose(metrics.return_vs_benchmark or 0, 0.05)


def test_fill_costs_and_turnover_include_positions_that_remain_open() -> None:
    equity = [
        {"date": "2024-01-02", "equity": 100.0, "gross_exposure": 0.0},
        {"date": "2024-01-03", "equity": 99.0, "gross_exposure": 0.5},
    ]
    fills = [
        {
            "quantity": 0.5,
            "fill_price": 100.0,
            "notional": 50.0,
            "commission": 1.0,
            "slippage": 0.5,
        }
    ]

    metrics = calculate_performance_metrics(equity, fills=fills)

    assert metrics.number_of_trades == 0
    assert metrics.commission_cost == 1.0
    assert metrics.slippage_cost == 0.5
    assert metrics.turnover == 50.0 / mean([100.0, 99.0])
