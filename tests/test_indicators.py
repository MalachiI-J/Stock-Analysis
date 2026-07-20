import pytest

from stock_scrapper.processing.indicators import calculate_indicators, classify_status


def test_indicator_calculations_and_status() -> None:
    history = [
        {"trade_date": "2024-01-01", "close": 100.0, "adjusted_close": 100.0, "volume": 1000},
        {"trade_date": "2024-01-02", "close": 102.0, "adjusted_close": 102.0, "volume": 1200},
        {"trade_date": "2024-01-03", "close": 101.0, "adjusted_close": 101.0, "volume": 1100},
        {"trade_date": "2024-01-04", "close": 105.0, "adjusted_close": 105.0, "volume": 1300},
        {"trade_date": "2024-01-05", "close": 108.0, "adjusted_close": 108.0, "volume": 1400},
        {"trade_date": "2024-01-08", "close": 110.0, "adjusted_close": 110.0, "volume": 900},
    ]

    metrics = calculate_indicators(history, "AAPL")
    assert metrics["latest_close"] == 110.0
    assert metrics["one_day_return"] == pytest.approx(0.018518518518518517)
    assert metrics["five_day_return"] == pytest.approx(0.1)
    assert metrics["twenty_day_sma"] is None
    assert metrics["max_drawdown"] == pytest.approx(0.00980392156862746)

    status, flags = classify_status(metrics, has_quality_warning=False)
    assert status in {"Insufficient Data", "Uptrend", "Mixed Trend", "Downtrend"}
    assert isinstance(flags, list)
