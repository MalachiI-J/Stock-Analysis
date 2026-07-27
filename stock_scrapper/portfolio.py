"""Aggregate real owned lots into positions and assess them against score_v1 exit rules.

This module is deliberately DB-free and pure: callers (the CLI, the digest)
load lots and price history from SQLite and pass plain values in. It reuses
the same rule-based exit logic the backtester uses
(:mod:`stock_scrapper.backtesting.exit_rules`) so a live "consider selling"
signal is judged by exactly the rules that would have closed the position in
a backtest, plus a close-price stop-loss/trailing-stop check.

It also compares real trades with a "SPY shadow portfolio": what the same
dollars, invested in the benchmark on the same days, would be worth now. This
is a fair like-for-like comparison for irregular real-world buy/sell timing,
unlike a naive total-cost-vs-current-value ratio that ignores when the
benchmark itself moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from stock_scrapper.backtesting.config import BacktestConfig
from stock_scrapper.backtesting.exit_rules import evaluate_price_stop, evaluate_rule_based_exit
from stock_scrapper.models.analysis_models import AnalysisResult


@dataclass(slots=True)
class PortfolioPosition:
    """One symbol's real holding, aggregated across every open lot (FIFO cost basis)."""

    symbol: str
    shares: float
    average_cost_basis: float
    earliest_opened_date: str
    lots: list[dict[str, Any]] = field(default_factory=list)

    @property
    def cost_basis(self) -> float:
        return self.shares * self.average_cost_basis


def aggregate_open_lots(lots: Sequence[Mapping[str, Any]]) -> list[PortfolioPosition]:
    """Group open lots by symbol into one weighted-average-cost position each."""
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for lot in lots:
        remaining = float(lot.get("remaining_shares") or 0.0)
        if remaining <= 0:
            continue
        symbol = str(lot["symbol"]).upper()
        by_symbol.setdefault(symbol, []).append(dict(lot))

    positions: list[PortfolioPosition] = []
    for symbol, symbol_lots in by_symbol.items():
        total_shares = sum(float(lot["remaining_shares"]) for lot in symbol_lots)
        weighted_cost = sum(
            float(lot["remaining_shares"]) * float(lot["cost_basis_per_share"]) for lot in symbol_lots
        )
        earliest_opened_date = min(str(lot["opened_date"]) for lot in symbol_lots)
        positions.append(
            PortfolioPosition(
                symbol=symbol,
                shares=total_shares,
                average_cost_basis=weighted_cost / total_shares if total_shares else 0.0,
                earliest_opened_date=earliest_opened_date,
                lots=symbol_lots,
            )
        )
    return sorted(positions, key=lambda position: position.symbol)


@dataclass(slots=True)
class HoldingAssessment:
    """A held position's current value plus a rules-based hold/sell recommendation."""

    symbol: str
    shares: float
    average_cost_basis: float
    latest_price: float | None
    classification: str | None
    primary_reason: str | None
    rule_based_exit_reason: str | None
    price_stop_reason: str | None
    recommendation: str

    @property
    def market_value(self) -> float | None:
        return None if self.latest_price is None else self.shares * self.latest_price

    @property
    def unrealized_pnl(self) -> float | None:
        value = self.market_value
        return None if value is None else value - self.shares * self.average_cost_basis

    @property
    def unrealized_pnl_pct(self) -> float | None:
        if self.latest_price is None or self.average_cost_basis <= 0:
            return None
        return (self.latest_price - self.average_cost_basis) / self.average_cost_basis


def evaluate_holding(
    position: PortfolioPosition,
    *,
    result: AnalysisResult | None,
    backtest_config: BacktestConfig,
    latest_price: float | None,
    peak_price_since_entry: float | None,
    holding_period_sessions: int,
) -> HoldingAssessment:
    """Assess one real position: current value plus a rules-based sell signal.

    ``result`` is the latest saved analysis for the symbol, if any — a held
    symbol outside the analyzed universe simply has no rule-based signal, only
    the price-based stop/trailing-stop check (when price history exists).
    """
    rule_based_exit_reason = (
        evaluate_rule_based_exit(backtest_config, result, holding_period_sessions)
        if result is not None
        else None
    )
    price_stop_reason = None
    if latest_price is not None and peak_price_since_entry is not None and position.average_cost_basis > 0:
        price_stop_reason = evaluate_price_stop(
            backtest_config,
            average_cost_basis=position.average_cost_basis,
            highest_price_since_entry=peak_price_since_entry,
            latest_price=latest_price,
        )
    if rule_based_exit_reason or price_stop_reason:
        recommendation = "SELL"
    elif latest_price is None:
        recommendation = "UNKNOWN (no current price data)"
    else:
        recommendation = "HOLD"
    return HoldingAssessment(
        symbol=position.symbol,
        shares=position.shares,
        average_cost_basis=position.average_cost_basis,
        latest_price=latest_price,
        classification=result.classification if result is not None else None,
        primary_reason=result.primary_reason if result is not None else None,
        rule_based_exit_reason=rule_based_exit_reason,
        price_stop_reason=price_stop_reason,
        recommendation=recommendation,
    )


@dataclass(slots=True)
class PortfolioPerformanceSummary:
    """Real trading performance plus a fair, like-for-like benchmark comparison."""

    total_invested: float
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    total_return_pct: float | None
    benchmark_shadow_pnl: float | None
    benchmark_shadow_invested: float | None
    benchmark_shadow_return_pct: float | None
    excess_return_pct: float | None
    unpriced_lot_ids: list[str]
    unpriced_symbols: list[str]


def evaluate_portfolio_performance(
    lots: Sequence[Mapping[str, Any]],
    sales: Sequence[Mapping[str, Any]],
    holdings: Sequence[HoldingAssessment],
    *,
    benchmark_price_at: Callable[[str], float | None],
    benchmark_price_as_of: float | None,
) -> PortfolioPerformanceSummary:
    """Compare real realized+unrealized P&L with a same-dollars, same-days SPY shadow portfolio.

    For each lot, ``benchmark_price_at(lot's opened_date)`` gives the shadow
    number of benchmark shares that lot's cash would have bought; each sale
    (matched to its originating lot via ``lot_id``) and any still-open
    remainder is priced the same way at its own date. A lot only contributes
    to the shadow comparison if every benchmark price point it needs is
    available — partial lots are not mixed into the totals, so the returned
    ``benchmark_shadow_invested`` always matches the dollars actually behind
    ``benchmark_shadow_pnl``. Lots missing any required price are listed in
    ``unpriced_lot_ids`` and simply excluded rather than guessed at.

    ``holdings`` supplies today's actual unrealized P&L (already computed
    against real market prices in :func:`evaluate_holding`); symbols with no
    current price are listed in ``unpriced_symbols`` and excluded from
    ``unrealized_pnl``/``total_pnl`` rather than treated as zero.
    """
    sales_by_lot: dict[str, list[Mapping[str, Any]]] = {}
    for sale in sales:
        sales_by_lot.setdefault(str(sale["lot_id"]), []).append(sale)

    total_invested = sum(float(lot["shares"]) * float(lot["cost_basis_per_share"]) for lot in lots)
    realized_pnl = sum(float(sale["realized_pnl"]) for sale in sales)
    unrealized_by_symbol = {holding.symbol: holding.unrealized_pnl for holding in holdings}
    unrealized_pnl = sum(value for value in unrealized_by_symbol.values() if value is not None)
    unpriced_symbols = sorted(
        symbol for symbol, value in unrealized_by_symbol.items() if value is None
    )

    benchmark_shadow_pnl = 0.0
    benchmark_shadow_invested = 0.0
    unpriced_lot_ids: list[str] = []
    for lot in lots:
        lot_id = str(lot["lot_id"])
        lot_shares = float(lot["shares"])
        cost_per_share = float(lot["cost_basis_per_share"])
        entry_price = benchmark_price_at(str(lot["opened_date"]))
        if entry_price is None or entry_price <= 0 or lot_shares <= 0:
            unpriced_lot_ids.append(lot_id)
            continue
        shadow_shares_total = (lot_shares * cost_per_share) / entry_price

        lot_ok = True
        lot_shadow_pnl = 0.0
        lot_invested = 0.0
        for sale in sales_by_lot.get(lot_id, []):
            exit_price = benchmark_price_at(str(sale["sale_date"]))
            if exit_price is None:
                lot_ok = False
                break
            sale_shares = float(sale["shares"])
            fraction = sale_shares / lot_shares
            dollars_invested = sale_shares * cost_per_share
            lot_shadow_pnl += shadow_shares_total * fraction * exit_price - dollars_invested
            lot_invested += dollars_invested

        remaining = float(lot.get("remaining_shares") or 0.0)
        if lot_ok and remaining > 0:
            if benchmark_price_as_of is None:
                lot_ok = False
            else:
                fraction = remaining / lot_shares
                dollars_invested = remaining * cost_per_share
                lot_shadow_pnl += shadow_shares_total * fraction * benchmark_price_as_of - dollars_invested
                lot_invested += dollars_invested

        if lot_ok:
            benchmark_shadow_pnl += lot_shadow_pnl
            benchmark_shadow_invested += lot_invested
        else:
            unpriced_lot_ids.append(lot_id)

    total_pnl = realized_pnl + unrealized_pnl
    total_return_pct = total_pnl / total_invested if total_invested > 0 else None
    has_shadow_data = benchmark_shadow_invested > 0
    benchmark_shadow_return_pct = (
        benchmark_shadow_pnl / benchmark_shadow_invested if has_shadow_data else None
    )
    excess_return_pct = (
        total_return_pct - benchmark_shadow_return_pct
        if total_return_pct is not None and benchmark_shadow_return_pct is not None
        else None
    )
    return PortfolioPerformanceSummary(
        total_invested=total_invested,
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        total_pnl=total_pnl,
        total_return_pct=total_return_pct,
        benchmark_shadow_pnl=benchmark_shadow_pnl if has_shadow_data else None,
        benchmark_shadow_invested=benchmark_shadow_invested if has_shadow_data else None,
        benchmark_shadow_return_pct=benchmark_shadow_return_pct,
        excess_return_pct=excess_return_pct,
        unpriced_lot_ids=sorted(set(unpriced_lot_ids)),
        unpriced_symbols=unpriced_symbols,
    )
