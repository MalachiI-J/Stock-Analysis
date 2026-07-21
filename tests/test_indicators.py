from __future__ import annotations

from datetime import date, timedelta
from math import sqrt
from statistics import median, stdev

import pytest

from stock_scrapper.processing.indicators import calculate_indicators, classify_status


def _history(
    closes: list[float],
    *,
    volumes: list[float] | None = None,
    gaps: list[float] | None = None,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    volumes = volumes or [1_000.0] * len(closes)
    gaps = gaps or [0.0] * len(closes)
    for index, close in enumerate(closes):
        previous = closes[index - 1] if index else close
        open_price = previous * (1.0 + gaps[index])
        rows.append(
            {
                "trade_date": str(date(2022, 1, 1) + timedelta(days=index)),
                "open": open_price,
                "high": max(open_price, close) + 1.0,
                "low": min(open_price, close) - 1.0,
                "close": close,
                "adjusted_close": close,
                "volume": volumes[index],
                "dividends": 0.0,
                "stock_splits": 0.0,
            }
        )
    return rows


def test_wilder_rsi_and_atr_use_price_changes_and_true_range() -> None:
    closes = [
        44.34,
        44.09,
        44.15,
        43.61,
        44.33,
        44.83,
        45.10,
        45.42,
        45.84,
        46.08,
        45.89,
        46.03,
        45.61,
        46.28,
        46.28,
    ]
    rows = _history(closes)
    for row in rows:
        row["high"] = float(row["close"]) + 1.0
        row["low"] = float(row["close"]) - 1.0
        row["open"] = row["close"]

    metrics = calculate_indicators(rows, "TEST")

    assert metrics["rsi_14"] == pytest.approx(70.4641350211)
    assert metrics["atr_14"] == pytest.approx(2.0)
    assert metrics["atr_percentage"] == pytest.approx(2.0 / closes[-1])


def test_missing_latest_ohlcv_is_not_forward_filled() -> None:
    rows = _history([100.0 + index for index in range(80)])
    rows.append(
        {
            "trade_date": "2024-12-31",
            "open": None,
            "high": None,
            "low": None,
            "close": None,
            "adjusted_close": None,
            "volume": None,
        }
    )

    metrics = calculate_indicators(rows, "TEST")

    assert metrics["latest_close"] is None
    assert metrics["one_day_return"] is None
    assert metrics["twenty_day_average_volume"] is None
    assert metrics["volume_relative_to_average"] is None
    assert metrics["atr_14"] is None
    assert metrics["ohlcv_completeness"] < 1.0
    assert rows[-1]["close"] is None


def test_slopes_time_above_and_twenty_day_dollar_volume() -> None:
    closes = [100.0 * (1.001**index) for index in range(500)]
    volumes = [1_000.0 + index for index in range(500)]
    metrics = calculate_indicators(_history(closes, volumes=volumes), "TEST")

    expected_sma50_slope = (
        (sum(closes[-50:]) / 50.0) / (sum(closes[-70:-20]) / 50.0) - 1.0
    )
    expected_sma200_slope = (
        (sum(closes[-200:]) / 200.0)
        / (sum(closes[-220:-20]) / 200.0)
        - 1.0
    )
    dollar_volumes = [price * volume for price, volume in zip(closes[-20:], volumes[-20:])]

    assert metrics["trend_slope_50"] == pytest.approx(expected_sma50_slope)
    assert metrics["trend_slope_200"] == pytest.approx(expected_sma200_slope)
    assert metrics["time_above_sma50"] == 1.0
    assert metrics["time_above_sma200"] == 1.0
    assert metrics["twenty_day_average_volume"] == pytest.approx(sum(volumes[-20:]) / 20)
    assert metrics["twenty_day_average_dollar_volume"] == pytest.approx(
        sum(dollar_volumes) / 20
    )
    assert metrics["twenty_day_median_dollar_volume"] == pytest.approx(
        median(dollar_volumes)
    )
    assert metrics["distance_from_52_week_high"] == pytest.approx(0.0)


def test_volatility_downside_gap_drawdown_and_liquidity_math() -> None:
    daily_returns = [0.012, -0.007, 0.004, -0.003, 0.009] * 104
    closes = [100.0]
    for daily_return in daily_returns:
        closes.append(closes[-1] * (1.0 + daily_return))
    gaps = [0.0] + ([0.01, -0.005, 0.002, -0.008, 0.004] * 104)
    volumes = [0.0 if index in {510, 515} else 2_000.0 + index for index in range(len(closes))]
    metrics = calculate_indicators(
        _history(closes, volumes=volumes, gaps=gaps),
        "TEST",
    )

    last_20 = daily_returns[-20:]
    last_60 = daily_returns[-60:]
    last_252 = daily_returns[-252:]
    downside = [min(value, 0.0) for value in last_60]
    expected_gap_volatility = stdev(gaps[1:]) * sqrt(252)
    running_high = closes[0]
    expected_drawdown = 0.0
    for close in closes:
        running_high = max(running_high, close)
        expected_drawdown = max(expected_drawdown, 1.0 - close / running_high)

    assert metrics["twenty_day_volatility"] == pytest.approx(stdev(last_20) * sqrt(252))
    assert metrics["sixty_day_volatility"] == pytest.approx(stdev(last_60) * sqrt(252))
    assert metrics["two_hundred_fifty_two_day_volatility"] == pytest.approx(
        stdev(last_252) * sqrt(252)
    )
    assert metrics["sixty_day_downside_volatility"] == pytest.approx(
        sqrt(sum(value * value for value in downside) / 60.0) * sqrt(252)
    )
    assert metrics["worst_one_day_return_last_year"] == pytest.approx(min(last_252))
    assert metrics["overnight_gap_volatility"] == pytest.approx(expected_gap_volatility)
    assert metrics["full_history_max_drawdown"] == pytest.approx(expected_drawdown)
    assert metrics["zero_volume_days"] == 2
    assert metrics["twenty_day_zero_volume_days"] == 2
    assert metrics["distance_from_52_week_high"] <= 0.0


def test_status_uses_negative_distance_from_high_convention() -> None:
    status, flags = classify_status(
        {
            "latest_close": 99.0,
            "fifty_day_sma": 95.0,
            "two_hundred_day_sma": 90.0,
            "twenty_day_volatility": 0.10,
            "distance_from_52_week_high": -0.01,
        }
    )
    assert status == "Uptrend"
    assert "Near 52-Week High" in flags
