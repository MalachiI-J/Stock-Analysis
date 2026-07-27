from __future__ import annotations

from stock_scrapper.models.analysis_models import AnalysisResult
from stock_scrapper.portfolio import HoldingAssessment, PortfolioPosition
from stock_scrapper.trading.recommendations import (
    build_recommendations,
    compute_available_cash,
    render_recommendations_text,
)


def _rules(**overrides: object) -> dict[str, object]:
    base = dict(
        starting_capital=10000.0,
        max_position_weight=0.15,
        cash_reserve=0.05,
        max_open_positions=10,
        max_trade_dollar_amount=2000.0,
        min_trade_dollar_amount=100.0,
        max_new_buys_per_run=3,
        require_strong_candidate_for_buy=True,
    )
    base.update(overrides)
    return base


def _result(symbol: str, classification: str, opportunity_score: float | None = 80.0) -> AnalysisResult:
    return AnalysisResult(
        symbol=symbol, as_of_date="2026-01-15", classification=classification,
        opportunity_score=opportunity_score, primary_reason=f"{symbol} looks good",
        eligible_for_scoring=True,
    )


def _lot(symbol: str, shares: float, cost: float, lot_id: str = "lot-1") -> dict[str, object]:
    return {"lot_id": lot_id, "symbol": symbol, "shares": shares, "cost_basis_per_share": cost}


def _sale(symbol: str, shares: float, price: float, lot_id: str = "lot-1") -> dict[str, object]:
    return {"lot_id": lot_id, "symbol": symbol, "shares": shares, "sale_price": price}


def test_compute_available_cash_subtracts_lots_and_adds_sale_proceeds() -> None:
    lots = [_lot("AAPL", 10, 100.0), _lot("MSFT", 5, 200.0)]
    sales = [_sale("AAPL", 4, 150.0)]
    assert compute_available_cash(lots, sales, starting_capital=10000.0) == 10000.0 - 2000.0 + 600.0


def test_build_recommendations_sells_a_holding_that_triggered_an_exit_rule() -> None:
    holding = HoldingAssessment(
        symbol="AAPL", shares=10.0, average_cost_basis=100.0, latest_price=90.0,
        classification="Avoid", primary_reason="weak", rule_based_exit_reason="Classification-based exit",
        price_stop_reason=None, recommendation="SELL",
    )
    position = PortfolioPosition(symbol="AAPL", shares=10.0, average_cost_basis=100.0, earliest_opened_date="2026-01-01")

    result = build_recommendations(
        as_of_date="2026-01-15", results_by_symbol={}, holdings=[holding], positions=[position],
        lots=[_lot("AAPL", 10, 100.0)], sales=[], latest_price_by_symbol={"AAPL": 90.0}, rules=_rules(),
    )

    assert len(result.recommendations) == 1
    sell = result.recommendations[0]
    assert sell.action == "SELL" and sell.symbol == "AAPL" and sell.shares == 10.0
    assert sell.estimated_dollars == 900.0
    assert sell.reason == "Classification-based exit"


def test_build_recommendations_does_not_sell_a_healthy_holding() -> None:
    holding = HoldingAssessment(
        symbol="AAPL", shares=10.0, average_cost_basis=100.0, latest_price=110.0,
        classification="Candidate", primary_reason="fine", rule_based_exit_reason=None,
        price_stop_reason=None, recommendation="HOLD",
    )
    position = PortfolioPosition(symbol="AAPL", shares=10.0, average_cost_basis=100.0, earliest_opened_date="2026-01-01")

    result = build_recommendations(
        as_of_date="2026-01-15", results_by_symbol={}, holdings=[holding], positions=[position],
        lots=[_lot("AAPL", 10, 100.0)], sales=[], latest_price_by_symbol={"AAPL": 110.0}, rules=_rules(),
    )

    assert result.recommendations == []


def test_build_recommendations_sizes_a_buy_within_position_weight_and_dollar_cap() -> None:
    results = {"NVDA": _result("NVDA", "Strong Candidate")}
    result = build_recommendations(
        as_of_date="2026-01-15", results_by_symbol=results, holdings=[], positions=[],
        lots=[], sales=[], latest_price_by_symbol={"NVDA": 50.0}, rules=_rules(),
    )

    # account_value = 10000 cash; max_position_weight=0.15 -> $1500 cap, below the $2000
    # max_trade_dollar_amount cap, so $1500 / $50 = 30 shares.
    assert len(result.recommendations) == 1
    buy = result.recommendations[0]
    assert buy.action == "BUY" and buy.symbol == "NVDA" and buy.shares == 30.0
    assert buy.estimated_dollars == 1500.0


def test_build_recommendations_respects_max_trade_dollar_amount_cap() -> None:
    results = {"NVDA": _result("NVDA", "Strong Candidate")}
    rules = _rules(max_position_weight=0.9, cash_reserve=0.0, max_trade_dollar_amount=500.0)

    result = build_recommendations(
        as_of_date="2026-01-15", results_by_symbol=results, holdings=[], positions=[],
        lots=[], sales=[], latest_price_by_symbol={"NVDA": 50.0}, rules=rules,
    )

    assert result.recommendations[0].estimated_dollars == 500.0
    assert result.recommendations[0].shares == 10.0


def test_build_recommendations_skips_symbols_already_held() -> None:
    results = {"NVDA": _result("NVDA", "Strong Candidate")}
    position = PortfolioPosition(symbol="NVDA", shares=5.0, average_cost_basis=40.0, earliest_opened_date="2026-01-01")

    result = build_recommendations(
        as_of_date="2026-01-15", results_by_symbol=results, holdings=[], positions=[position],
        lots=[_lot("NVDA", 5, 40.0)], sales=[], latest_price_by_symbol={"NVDA": 50.0}, rules=_rules(),
    )

    assert result.recommendations == []


def test_build_recommendations_excludes_candidate_when_strong_required() -> None:
    results = {"NVDA": _result("NVDA", "Candidate")}
    result = build_recommendations(
        as_of_date="2026-01-15", results_by_symbol=results, holdings=[], positions=[],
        lots=[], sales=[], latest_price_by_symbol={"NVDA": 50.0}, rules=_rules(require_strong_candidate_for_buy=True),
    )
    assert result.recommendations == []
    assert result.skipped == []  # excluded by classification filter, never considered a "skip"


def test_build_recommendations_includes_candidate_when_not_required() -> None:
    results = {"NVDA": _result("NVDA", "Candidate")}
    result = build_recommendations(
        as_of_date="2026-01-15", results_by_symbol=results, holdings=[], positions=[],
        lots=[], sales=[], latest_price_by_symbol={"NVDA": 50.0}, rules=_rules(require_strong_candidate_for_buy=False),
    )
    assert len(result.recommendations) == 1 and result.recommendations[0].symbol == "NVDA"


def test_build_recommendations_respects_max_new_buys_per_run() -> None:
    results = {
        symbol: _result(symbol, "Strong Candidate", opportunity_score=score)
        for symbol, score in [("AAA", 90.0), ("BBB", 85.0), ("CCC", 80.0), ("DDD", 75.0)]
    }
    prices = {symbol: 50.0 for symbol in results}
    rules = _rules(max_new_buys_per_run=2, max_trade_dollar_amount=200.0, max_position_weight=0.9, cash_reserve=0.0)

    result = build_recommendations(
        as_of_date="2026-01-15", results_by_symbol=results, holdings=[], positions=[],
        lots=[], sales=[], latest_price_by_symbol=prices, rules=rules,
    )

    bought = [rec.symbol for rec in result.recommendations]
    assert bought == ["AAA", "BBB"]  # ranked by opportunity_score, capped at 2
    assert any("max_new_buys_per_run" in entry for entry in result.skipped)


def test_build_recommendations_respects_max_open_positions() -> None:
    position = PortfolioPosition(symbol="ZZZ", shares=1.0, average_cost_basis=10.0, earliest_opened_date="2026-01-01")
    results = {"NVDA": _result("NVDA", "Strong Candidate")}

    result = build_recommendations(
        as_of_date="2026-01-15", results_by_symbol=results, holdings=[], positions=[position],
        lots=[_lot("ZZZ", 1, 10.0)], sales=[], latest_price_by_symbol={"NVDA": 50.0},
        rules=_rules(max_open_positions=1),
    )

    assert result.recommendations == []
    assert any("max_open_positions" in entry for entry in result.skipped)


def test_build_recommendations_stops_when_cash_is_exhausted() -> None:
    results = {
        symbol: _result(symbol, "Strong Candidate", opportunity_score=score)
        for symbol, score in [("AAA", 90.0), ("BBB", 85.0)]
    }
    prices = {"AAA": 50.0, "BBB": 50.0}
    # Only $50 of spendable cash after the reserve -> below min_trade_dollar_amount for both.
    rules = _rules(starting_capital=100.0, cash_reserve=0.5, min_trade_dollar_amount=100.0, max_position_weight=0.95)

    result = build_recommendations(
        as_of_date="2026-01-15", results_by_symbol=results, holdings=[], positions=[],
        lots=[], sales=[], latest_price_by_symbol=prices, rules=rules,
    )

    assert result.recommendations == []
    assert len(result.skipped) == 1  # breaks after the first once cash is confirmed exhausted


def test_build_recommendations_attaches_model_probability_to_buys() -> None:
    results = {"NVDA": _result("NVDA", "Strong Candidate")}
    result = build_recommendations(
        as_of_date="2026-01-15", results_by_symbol=results, holdings=[], positions=[],
        lots=[], sales=[], latest_price_by_symbol={"NVDA": 50.0}, rules=_rules(),
        model_probability_by_symbol={"NVDA": 0.63},
    )
    assert result.recommendations[0].model_probability == 0.63


def test_build_recommendations_reports_account_value_and_available_cash() -> None:
    holding = HoldingAssessment(
        symbol="AAPL", shares=10.0, average_cost_basis=100.0, latest_price=110.0,
        classification="Candidate", primary_reason="fine", rule_based_exit_reason=None,
        price_stop_reason=None, recommendation="HOLD",
    )
    position = PortfolioPosition(symbol="AAPL", shares=10.0, average_cost_basis=100.0, earliest_opened_date="2026-01-01")

    result = build_recommendations(
        as_of_date="2026-01-15", results_by_symbol={}, holdings=[holding], positions=[position],
        lots=[_lot("AAPL", 10, 100.0)], sales=[], latest_price_by_symbol={"AAPL": 110.0}, rules=_rules(),
    )

    assert result.available_cash == 10000.0 - 1000.0  # starting capital minus the one lot's cost basis
    assert result.account_value == (10000.0 - 1000.0) + 1100.0  # cash + current market value of the holding
    assert result.open_position_count == 1


def test_render_recommendations_text_includes_buys_sells_and_disclaimer() -> None:
    results = {"NVDA": _result("NVDA", "Strong Candidate")}
    result = build_recommendations(
        as_of_date="2026-01-15", results_by_symbol=results, holdings=[], positions=[],
        lots=[], sales=[], latest_price_by_symbol={"NVDA": 50.0}, rules=_rules(),
        model_probability_by_symbol={"NVDA": 0.6},
    )

    text = render_recommendations_text(result)

    assert "TRADE RECOMMENDATIONS" in text
    assert "BUY — 1" in text and "NVDA" in text and "model: 60% positive" in text
    assert "SELL — 0" in text and "None" in text
    assert "not investment advice" in text.lower()


def test_render_recommendations_text_lists_skipped_candidates() -> None:
    results = {"NVDA": _result("NVDA", "Candidate")}
    result = build_recommendations(
        as_of_date="2026-01-15", results_by_symbol=results, holdings=[], positions=[],
        lots=[], sales=[], latest_price_by_symbol={}, rules=_rules(require_strong_candidate_for_buy=False),
    )

    text = render_recommendations_text(result)

    assert "Considered but not recommended" in text
    assert "no current price" in text
