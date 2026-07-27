from __future__ import annotations

from stock_scrapper.trading.review import evaluate_recommendation_outcomes, render_review_text


def _prices(table: dict[tuple[str, str], float]):
    def price_at(symbol: str, day: str) -> float | None:
        return table.get((symbol, day))

    return price_at


def test_buy_that_beats_benchmark_is_graded_a_hit() -> None:
    recs = [{"symbol": "AAPL", "action": "BUY", "shares": 10.0, "estimated_dollars": 1000.0}]
    price_at = _prices({
        ("SPY", "2026-01-01"): 400.0, ("SPY", "2026-02-01"): 408.0,  # SPY +2%
        ("AAPL", "2026-02-01"): 110.0,  # entry 100 -> exit 110, AAPL +10%
    })

    result = evaluate_recommendation_outcomes(
        recs, recommendation_date="2026-01-01", review_date="2026-02-01",
        benchmark_symbol="SPY", price_at=price_at,
    )

    outcome = result.outcomes[0]
    assert outcome.realized_return_pct == 0.10
    assert outcome.benchmark_return_pct == 0.02
    assert round(outcome.excess_return_pct, 4) == 0.08
    assert outcome.outcome == "beat the benchmark"
    assert result.buy_hit_rate == 1.0
    assert round(result.average_buy_excess_return_pct, 4) == 0.08


def test_buy_that_trails_benchmark_is_graded_a_miss() -> None:
    recs = [{"symbol": "AAPL", "action": "BUY", "shares": 10.0, "estimated_dollars": 1000.0}]
    price_at = _prices({
        ("SPY", "2026-01-01"): 400.0, ("SPY", "2026-02-01"): 440.0,  # SPY +10%
        ("AAPL", "2026-02-01"): 105.0,  # AAPL +5%, below benchmark
    })

    result = evaluate_recommendation_outcomes(
        recs, recommendation_date="2026-01-01", review_date="2026-02-01",
        benchmark_symbol="SPY", price_at=price_at,
    )

    assert result.outcomes[0].outcome == "trailed the benchmark"
    assert result.buy_hit_rate == 0.0


def test_sell_where_price_fell_is_graded_validated() -> None:
    recs = [{"symbol": "TSLA", "action": "SELL", "shares": 5.0, "estimated_dollars": 500.0}]
    price_at = _prices({("TSLA", "2026-02-01"): 80.0})  # entry 100 -> exit 80

    result = evaluate_recommendation_outcomes(
        recs, recommendation_date="2026-01-01", review_date="2026-02-01",
        benchmark_symbol="SPY", price_at=price_at,
    )

    outcome = result.outcomes[0]
    assert outcome.realized_return_pct == -0.20
    assert outcome.outcome == "selling looks validated (price fell after)"
    assert result.sell_avoided_loss_rate == 1.0


def test_sell_where_price_rose_is_graded_a_missed_gain() -> None:
    recs = [{"symbol": "TSLA", "action": "SELL", "shares": 5.0, "estimated_dollars": 500.0}]
    price_at = _prices({("TSLA", "2026-02-01"): 120.0})  # entry 100 -> exit 120

    result = evaluate_recommendation_outcomes(
        recs, recommendation_date="2026-01-01", review_date="2026-02-01",
        benchmark_symbol="SPY", price_at=price_at,
    )

    assert result.outcomes[0].outcome == "would have missed a gain by selling"
    assert result.sell_avoided_loss_rate == 0.0


def test_missing_exit_price_is_reported_as_unpriced() -> None:
    recs = [{"symbol": "AAPL", "action": "BUY", "shares": 10.0, "estimated_dollars": 1000.0}]
    price_at = _prices({})  # no data anywhere

    result = evaluate_recommendation_outcomes(
        recs, recommendation_date="2026-01-01", review_date="2026-02-01",
        benchmark_symbol="SPY", price_at=price_at,
    )

    assert result.outcomes[0].outcome == "no price data available for the review date"
    assert result.unpriced_symbols == ["AAPL"]
    assert result.buy_hit_rate is None


def test_aggregate_stats_average_across_multiple_buys() -> None:
    recs = [
        {"symbol": "AAA", "action": "BUY", "shares": 10.0, "estimated_dollars": 1000.0},
        {"symbol": "BBB", "action": "BUY", "shares": 10.0, "estimated_dollars": 1000.0},
    ]
    price_at = _prices({
        ("SPY", "2026-01-01"): 100.0, ("SPY", "2026-02-01"): 100.0,  # flat benchmark
        ("AAA", "2026-02-01"): 110.0,  # +10% excess
        ("BBB", "2026-02-01"): 90.0,  # -10% excess
    })

    result = evaluate_recommendation_outcomes(
        recs, recommendation_date="2026-01-01", review_date="2026-02-01",
        benchmark_symbol="SPY", price_at=price_at,
    )

    assert result.buy_hit_rate == 0.5
    assert round(result.average_buy_excess_return_pct, 4) == 0.0


def test_render_review_text_handles_empty_recommendations() -> None:
    result = evaluate_recommendation_outcomes(
        [], recommendation_date="2026-01-01", review_date="2026-02-01",
        benchmark_symbol="SPY", price_at=_prices({}),
    )
    text = render_review_text(result)
    assert "No recommendations were recorded" in text


def test_render_review_text_includes_outcomes_and_aggregates() -> None:
    recs = [{"symbol": "AAPL", "action": "BUY", "shares": 10.0, "estimated_dollars": 1000.0}]
    price_at = _prices({
        ("SPY", "2026-01-01"): 400.0, ("SPY", "2026-02-01"): 408.0,
        ("AAPL", "2026-02-01"): 110.0,
    })
    result = evaluate_recommendation_outcomes(
        recs, recommendation_date="2026-01-01", review_date="2026-02-01",
        benchmark_symbol="SPY", price_at=price_at,
    )

    text = render_review_text(result)

    assert "RECOMMENDATION REVIEW" in text
    assert "AAPL" in text and "beat the benchmark" in text
    assert "BUY hit rate" in text and "100%" in text
    assert "not investment advice" in text.lower()
