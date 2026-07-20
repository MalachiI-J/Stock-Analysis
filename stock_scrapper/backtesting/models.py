"""Typed models for the Phase 3 backtesting scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class BacktestTrade:
    """A single executed trade in the backtest."""

    symbol: str
    entry_date: str
    entry_price: float
    exit_date: str | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    return_pct: float = 0.0
    side: str = "buy"


@dataclass(slots=True)
class BacktestResult:
    """A per-symbol backtest summary used by the CLI and reporting layers."""

    symbol: str
    initial_cash: float
    final_cash: float
    final_value: float
    total_return: float
    trade_count: int
    winning_trades: int
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    equity_curve: list[float] = field(default_factory=list)
    trades: list[BacktestTrade] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "initial_cash": self.initial_cash,
            "final_cash": self.final_cash,
            "final_value": self.final_value,
            "total_return": self.total_return,
            "trade_count": self.trade_count,
            "winning_trades": self.winning_trades,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "equity_curve": self.equity_curve,
            "notes": self.notes,
        }
