from __future__ import annotations

from math import isclose

from stock_scrapper.backtesting.corporate_actions import (
    adjust_position_for_split,
    build_adjusted_ohlc,
    calculate_adjustment_factor,
    cash_dividend_credit,
    position_market_value,
    reconcile_position_for_price_basis,
)


def test_adjusted_ohlc_uses_reported_close_factor_consistently() -> None:
    bar = build_adjusted_ohlc(
        {
            "symbol": "AAPL",
            "trade_date": "2024-01-02",
            "open": 99,
            "high": 102,
            "low": 98,
            "close": 100,
            "adjusted_close": 50,
            "volume": 1000,
            "dividends": 0,
            "stock_splits": 2,
        }
    )

    assert bar.adjustment_factor == 0.5
    assert bar.adjusted_open == 49.5
    assert bar.adjusted_high == 51
    assert bar.adjusted_low == 49
    assert bar.adjusted_close == 50
    assert bar.adjustment_available is True
    assert bar.execution_price_available is True


def test_missing_adjustment_factor_never_fabricates_an_execution_open() -> None:
    assert calculate_adjustment_factor(100, None) is None
    assert calculate_adjustment_factor(0, 100) is None

    missing_adjusted = build_adjusted_ohlc(
        {"symbol": "AAPL", "trade_date": "2024-01-02", "open": 99, "close": 100, "adjusted_close": None}
    )
    assert missing_adjusted.adjustment_factor is None
    assert missing_adjusted.adjusted_open is None
    assert missing_adjusted.execution_price_available is False

    missing_raw = build_adjusted_ohlc(
        {"symbol": "AAPL", "trade_date": "2024-01-02", "open": 99, "close": None, "adjusted_close": 100}
    )
    assert missing_raw.adjustment_factor is None
    assert missing_raw.adjusted_open is None
    assert missing_raw.adjusted_close == 100


def test_forward_and_reverse_splits_preserve_cost_basis_and_raw_market_value() -> None:
    forward = adjust_position_for_split(10, 100, 2)
    assert forward.applied is True
    assert forward.adjusted_quantity == 20
    assert forward.adjusted_average_cost == 50
    assert forward.original_cost_basis == forward.adjusted_cost_basis == 1000
    assert position_market_value(10, 100) == position_market_value(forward.adjusted_quantity, 50)

    reverse = adjust_position_for_split(10, 100, 0.25)
    assert reverse.applied is True
    assert reverse.adjusted_quantity == 2.5
    assert reverse.adjusted_average_cost == 400
    assert reverse.original_cost_basis == reverse.adjusted_cost_basis == 1000
    assert position_market_value(10, 100) == position_market_value(reverse.adjusted_quantity, 400)


def test_adjusted_price_basis_does_not_double_count_splits_or_dividends() -> None:
    adjusted_position = reconcile_position_for_price_basis(10, 50, 2, price_basis="adjusted")
    assert adjusted_position.applied is False
    assert adjusted_position.adjusted_quantity == 10
    assert adjusted_position.adjusted_average_cost == 50
    assert position_market_value(10, 50) == position_market_value(adjusted_position.adjusted_quantity, 50)

    assert cash_dividend_credit(10, 1, price_basis="adjusted") == 0
    assert cash_dividend_credit(10, 1, price_basis="raw") == 10
    assert cash_dividend_credit(10, None, price_basis="raw") is None

    before_dividend = build_adjusted_ohlc({"close": 100, "adjusted_close": 99, "open": 100})
    after_dividend = build_adjusted_ohlc({"close": 99, "adjusted_close": 99, "open": 99, "dividends": 1})
    assert isclose(before_dividend.adjusted_close or 0, after_dividend.adjusted_close or 0)


def test_zero_or_missing_split_is_no_action_not_a_destroyed_position() -> None:
    zero = adjust_position_for_split(10, 100, 0)
    missing = adjust_position_for_split(10, 100, None)
    assert zero.applied is False
    assert zero.adjusted_quantity == 10
    assert missing.applied is False
    assert missing.adjusted_quantity == 10

