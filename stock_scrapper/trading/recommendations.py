"""Turn today's scores and real holdings into sized, restriction-bounded trade recommendations.

Advisory only: nothing here places an order. SELL signals are exactly the ones
:func:`stock_scrapper.portfolio.evaluate_holding` already computes (the same rules a
backtest would exit on), so a live sell recommendation and a backtest exit always agree.
BUY candidates come from score_v1's own classification (Strong Candidate / Candidate) —
the experimental prediction model's probability is attached to each candidate purely as
displayed context, never as a gate, since its accuracy doesn't yet warrant driving a
real trade decision (see stock_scrapper/prediction/).

There is no persisted cash ledger. Available cash is derived from the existing
portfolio_lots/portfolio_sales tables plus one user-set "starting_capital" figure:
capital in, minus every dollar ever committed to a lot, plus every dollar a sale ever
returned. This intentionally does not need a new migration or a new table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from stock_scrapper.models.analysis_models import AnalysisResult
from stock_scrapper.portfolio import HoldingAssessment, PortfolioPosition

_BUY_ELIGIBLE_CLASSIFICATIONS_STRICT = ("Strong Candidate",)
_BUY_ELIGIBLE_CLASSIFICATIONS_RELAXED = ("Strong Candidate", "Candidate")


def compute_available_cash(
    lots: Sequence[Mapping[str, Any]],
    sales: Sequence[Mapping[str, Any]],
    *,
    starting_capital: float,
) -> float:
    """Derive spendable cash without a ledger: capital minus lots' cost basis plus sale proceeds."""
    invested = sum(float(lot["shares"]) * float(lot["cost_basis_per_share"]) for lot in lots)
    proceeds = sum(float(sale["shares"]) * float(sale["sale_price"]) for sale in sales)
    return starting_capital - invested + proceeds


@dataclass(slots=True)
class TradeRecommendation:
    """One sized, actionable suggestion — the human still decides whether to place it."""

    symbol: str
    action: str  # "BUY" or "SELL"
    shares: float
    estimated_dollars: float
    reason: str
    model_probability: float | None = None


@dataclass(slots=True)
class RecommendationRunResult:
    """Complete outcome of one recommendation run: sizing context plus every suggestion."""

    as_of_date: str
    account_value: float
    available_cash: float
    open_position_count: int
    recommendations: list[TradeRecommendation] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def _holding_value(holding: HoldingAssessment) -> float:
    value = holding.market_value
    return value if value is not None else holding.shares * holding.average_cost_basis


def build_recommendations(
    *,
    as_of_date: str,
    results_by_symbol: Mapping[str, AnalysisResult],
    holdings: Sequence[HoldingAssessment],
    positions: Sequence[PortfolioPosition],
    lots: Sequence[Mapping[str, Any]],
    sales: Sequence[Mapping[str, Any]],
    latest_price_by_symbol: Mapping[str, float],
    rules: Mapping[str, Any],
    model_probability_by_symbol: Mapping[str, float] | None = None,
) -> RecommendationRunResult:
    """Compute SELL signals for held positions and sized BUY candidates for the rest."""
    model_probability_by_symbol = model_probability_by_symbol or {}
    held_symbols = {position.symbol for position in positions}
    available_cash = compute_available_cash(lots, sales, starting_capital=float(rules["starting_capital"]))
    holdings_value = sum(_holding_value(holding) for holding in holdings)
    account_value = available_cash + holdings_value

    recommendations: list[TradeRecommendation] = []
    skipped: list[str] = []

    for holding in holdings:
        reason = holding.rule_based_exit_reason or holding.price_stop_reason
        if reason is None:
            continue
        price = latest_price_by_symbol.get(holding.symbol)
        estimated_dollars = holding.shares * price if price is not None else 0.0
        recommendations.append(
            TradeRecommendation(
                symbol=holding.symbol, action="SELL", shares=holding.shares,
                estimated_dollars=estimated_dollars, reason=reason,
            )
        )

    open_position_count = len(positions)
    cash_reserve = float(rules["cash_reserve"])
    max_position_dollars = account_value * float(rules["max_position_weight"])
    max_trade_dollar_amount = float(rules["max_trade_dollar_amount"])
    min_trade_dollar_amount = float(rules["min_trade_dollar_amount"])
    max_open_positions = int(rules["max_open_positions"])
    max_new_buys = int(rules["max_new_buys_per_run"])
    eligible_classifications = (
        _BUY_ELIGIBLE_CLASSIFICATIONS_STRICT
        if rules["require_strong_candidate_for_buy"]
        else _BUY_ELIGIBLE_CLASSIFICATIONS_RELAXED
    )
    spendable = max(0.0, available_cash - account_value * cash_reserve)

    candidates = sorted(
        (
            result for symbol, result in results_by_symbol.items()
            if symbol not in held_symbols and result.classification in eligible_classifications
        ),
        key=lambda result: (
            0 if result.classification == "Strong Candidate" else 1,
            -(result.opportunity_score if result.opportunity_score is not None else -1.0),
            result.symbol,
        ),
    )

    new_buys = 0
    for result in candidates:
        if new_buys >= max_new_buys:
            skipped.append(f"{result.symbol}: already reached max_new_buys_per_run ({max_new_buys}) for this run")
            continue
        if open_position_count >= max_open_positions:
            skipped.append(f"{result.symbol}: already at max_open_positions ({max_open_positions})")
            continue
        price = latest_price_by_symbol.get(result.symbol)
        if price is None or price <= 0:
            skipped.append(f"{result.symbol}: no current price available")
            continue
        target_dollars = min(max_trade_dollar_amount, max_position_dollars, spendable)
        if target_dollars < min_trade_dollar_amount:
            skipped.append(
                f"{result.symbol}: affordable size (${target_dollars:,.2f}) is below "
                f"min_trade_dollar_amount (${min_trade_dollar_amount:,.2f})"
            )
            if spendable < min_trade_dollar_amount:
                break  # no cash left for anyone further down the list either
            continue
        shares = math.floor(target_dollars / price)
        if shares < 1:
            skipped.append(f"{result.symbol}: price (${price:,.2f}) is too high for the affordable size")
            continue
        estimated_dollars = shares * price
        recommendations.append(
            TradeRecommendation(
                symbol=result.symbol, action="BUY", shares=float(shares),
                estimated_dollars=estimated_dollars, reason=result.primary_reason,
                model_probability=model_probability_by_symbol.get(result.symbol),
            )
        )
        spendable -= estimated_dollars
        open_position_count += 1
        new_buys += 1

    return RecommendationRunResult(
        as_of_date=as_of_date,
        account_value=account_value,
        available_cash=available_cash,
        open_position_count=len(positions),
        recommendations=recommendations,
        skipped=skipped,
    )


def render_recommendations_text(result: RecommendationRunResult) -> str:
    """Render a recommendation run to a plain-text report, digest-style."""
    lines = [
        f"TRADE RECOMMENDATIONS — as of {result.as_of_date}",
        f"Account value: ${result.account_value:,.2f}  |  Available cash: ${result.available_cash:,.2f}  |  "
        f"Open positions: {result.open_position_count}",
        "",
    ]
    sells = [rec for rec in result.recommendations if rec.action == "SELL"]
    buys = [rec for rec in result.recommendations if rec.action == "BUY"]

    lines.append(f"SELL — {len(sells)}")
    if sells:
        for rec in sells:
            lines.append(f"  {rec.symbol:<6} {rec.shares:g} sh (~${rec.estimated_dollars:,.2f}) — {rec.reason}")
    else:
        lines.append("  None")

    lines.append("")
    lines.append(f"BUY — {len(buys)}")
    if buys:
        for rec in buys:
            probability_text = (
                "" if rec.model_probability is None else f" | model: {rec.model_probability:.0%} positive"
            )
            lines.append(
                f"  {rec.symbol:<6} {rec.shares:g} sh (~${rec.estimated_dollars:,.2f}) — {rec.reason}{probability_text}"
            )
    else:
        lines.append("  None")

    if result.skipped:
        lines.append("")
        lines.append("Considered but not recommended:")
        for entry in result.skipped:
            lines.append(f"  {entry}")

    lines.append("")
    lines.append(
        "These are sized suggestions only — nothing here has been bought or sold. Review each "
        "before placing any order yourself. Educational research output, not investment advice."
    )
    return "\n".join(lines)
