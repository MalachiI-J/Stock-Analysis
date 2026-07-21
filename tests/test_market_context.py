from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from stock_scrapper.analysis.market_context import (
    calculate_market_context,
    calculate_watchlist_breadth,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _rules() -> dict[str, object]:
    with (PROJECT_ROOT / "config" / "scoring_rules.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        return yaml.safe_load(handle)


def _history(symbol: str, count: int = 260, gain: float = 0.001) -> list[dict[str, object]]:
    price = 100.0
    rows: list[dict[str, object]] = []
    for index in range(count):
        price *= 1.0 + gain
        rows.append(
            {
                "symbol": symbol,
                "trade_date": f"2024-{index // 28 + 1:02d}-{index % 28 + 1:02d}",
                "close": price,
                "adjusted_close": price,
            }
        )
    return rows


def test_market_context_never_substitutes_raw_close_for_missing_adjusted_close() -> None:
    spy = _history("SPY")
    spy[-1]["adjusted_close"] = None
    spy[-1]["close"] = 9999.0

    context = calculate_market_context(
        spy,
        {"QQQ": _history("QQQ"), "IWM": _history("IWM")},
        1.0,
        _rules(),
    )

    assert context.regime == "Insufficient Market Data"
    assert context.metrics["context_availability_ratio"] == 0.0
    assert "complete" in context.reasons[0].lower()


def test_breadth_requires_current_and_complete_trailing_adjusted_prices() -> None:
    complete = _history("AAA")
    missing_latest = _history("BBB")
    missing_latest[-1]["adjusted_close"] = None
    missing_inside_window = _history("CCC")
    missing_inside_window[-50]["adjusted_close"] = None

    breadth, above, eligible = calculate_watchlist_breadth(
        {
            "AAA": complete,
            "BBB": missing_latest,
            "CCC": missing_inside_window,
        },
        ["AAA", "BBB", "CCC"],
    )

    assert breadth == pytest.approx(1.0)
    assert above == 1
    assert eligible == 1


def test_market_vote_thresholds_and_partial_context_availability_are_configured() -> None:
    rules = _rules()
    context = calculate_market_context(
        _history("SPY"),
        {"QQQ": _history("QQQ"), "IWM": []},
        1.0,
        rules,
    )
    assert context.metrics["context_availability_ratio"] == pytest.approx(0.9)

    strict = _rules()
    strict["market_regime_thresholds"]["risk_on_minimum_votes"] = 9
    strict_context = calculate_market_context(
        _history("SPY"),
        {"QQQ": _history("QQQ"), "IWM": _history("IWM")},
        1.0,
        strict,
    )
    assert strict_context.regime == "Neutral"
