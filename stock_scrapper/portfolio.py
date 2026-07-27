"""Aggregate real owned lots into positions and assess them against score_v1 exit rules.

This module is deliberately DB-free and pure: callers (the CLI, the digest)
load lots and price history from SQLite and pass plain values in. It reuses
the same rule-based exit logic the backtester uses
(:mod:`stock_scrapper.backtesting.exit_rules`) so a live "consider selling"
signal is judged by exactly the rules that would have closed the position in
a backtest, plus a close-price stop-loss/trailing-stop check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

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
