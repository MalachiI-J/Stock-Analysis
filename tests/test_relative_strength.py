from __future__ import annotations

from datetime import date, timedelta

import pytest

from stock_scrapper.processing.relative_strength import (
    align_series,
    calculate_relative_strength_metrics,
)


def _prices_from_returns(start: float, returns: list[float]) -> list[float]:
    prices = [start]
    for daily_return in returns:
        prices.append(prices[-1] * (1.0 + daily_return))
    return prices


def test_aligned_benchmark_returns_beta_correlation_and_rs_trend() -> None:
    benchmark_returns = [0.001 + (index % 7 - 3) * 0.0004 for index in range(260)]
    stock_returns = [0.0005 + 1.5 * value for value in benchmark_returns]
    benchmark_prices = _prices_from_returns(100.0, benchmark_returns)
    stock_prices = _prices_from_returns(50.0, stock_returns)
    stock_history = []
    benchmark_history = []
    for index, (stock_price, benchmark_price) in enumerate(
        zip(stock_prices, benchmark_prices)
    ):
        trade_date = str(date(2023, 1, 1) + timedelta(days=index))
        stock_history.append({"trade_date": trade_date, "adjusted_close": stock_price})
        benchmark_history.append(
            {"trade_date": trade_date, "adjusted_close": benchmark_price}
        )

    metrics = calculate_relative_strength_metrics(stock_history, benchmark_history)

    assert metrics["benchmark_aligned_days"] == len(stock_prices)
    assert metrics["benchmark_alignment_ratio"] == 1.0
    assert metrics["benchmark_available"] is True
    for window in (21, 63, 126, 252):
        stock_return = stock_prices[-1] / stock_prices[-window - 1] - 1.0
        benchmark_return = benchmark_prices[-1] / benchmark_prices[-window - 1] - 1.0
        assert metrics[f"benchmark_relative_return_{window}"] == pytest.approx(
            stock_return - benchmark_return
        )
    assert metrics["beta_252"] == pytest.approx(1.5)
    assert metrics["beta"] == pytest.approx(1.5)
    assert metrics["correlation_252"] == pytest.approx(1.0)
    ratio_now = stock_prices[-1] / benchmark_prices[-1]
    ratio_then = stock_prices[-64] / benchmark_prices[-64]
    assert metrics["relative_strength_trend"] == pytest.approx(ratio_now / ratio_then - 1.0)


def test_alignment_does_not_fill_missing_dates_or_prices() -> None:
    stock = [
        {"trade_date": "2024-01-01", "adjusted_close": 100.0},
        {"trade_date": "2024-01-02", "adjusted_close": None},
        {"trade_date": "2024-01-03", "adjusted_close": 102.0},
    ]
    benchmark = [
        {"trade_date": "2024-01-01", "adjusted_close": 200.0},
        {"trade_date": "2024-01-02", "adjusted_close": 201.0},
        {"trade_date": "2024-01-04", "adjusted_close": 203.0},
    ]

    aligned_stock, aligned_benchmark = align_series(stock, benchmark)
    metrics = calculate_relative_strength_metrics(stock, benchmark)

    assert [row["trade_date"] for row in aligned_stock] == ["2024-01-01", "2024-01-02"]
    assert [row["trade_date"] for row in aligned_benchmark] == [
        "2024-01-01",
        "2024-01-02",
    ]
    assert aligned_stock[-1]["adjusted_close"] is None
    assert metrics["benchmark_aligned_days"] == 1
    assert metrics["benchmark_available"] is False
    assert metrics["benchmark_relative_return_21"] is None
