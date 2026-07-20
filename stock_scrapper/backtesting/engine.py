"""Deterministic long-only backtesting helpers for Phase 3."""

from __future__ import annotations

from typing import Any

from stock_scrapper.backtesting.models import BacktestResult, BacktestTrade
from stock_scrapper.processing.indicators import calculate_indicators


def _to_history_rows(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "trade_date": row.get("trade_date"),
            "close": row.get("close"),
            "adjusted_close": row.get("adjusted_close"),
            "volume": row.get("volume"),
        }
        for row in history
    ]


def run_symbol_backtest(symbol: str, history: list[dict[str, Any]], initial_cash: float = 100000.0) -> BacktestResult:
    """Run a simple long-only backtest using rolling trend signals from the existing indicators."""
    if not history:
        return BacktestResult(
            symbol=symbol,
            initial_cash=initial_cash,
            final_cash=initial_cash,
            final_value=initial_cash,
            total_return=0.0,
            trade_count=0,
            winning_trades=0,
            win_rate=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            notes=["No history available"],
        )

    cash = initial_cash
    holdings = 0.0
    equity_curve: list[float] = []
    trades: list[BacktestTrade] = []
    daily_returns: list[float] = []
    peak_value = initial_cash
    max_drawdown = 0.0
    last_signal = "hold"
    last_close = None

    for index, row in enumerate(history):
        price = float(row.get("adjusted_close") or row.get("close") or 0.0)
        if price <= 0:
            continue

        window_history = history[: index + 1]
        metrics = calculate_indicators(_to_history_rows(window_history), symbol)
        one_day_return = metrics.get("one_day_return") or 0.0
        distance_from_sma50 = metrics.get("distance_from_sma50") or 0.0

        signal = "hold"
        if distance_from_sma50 > 0 and one_day_return > 0:
            signal = "buy"
        elif distance_from_sma50 < 0 or one_day_return < 0:
            signal = "sell"

        if signal == "buy" and holdings == 0 and cash > 0:
            holdings = cash / price
            cash = 0.0
            last_signal = "buy"
            trades.append(BacktestTrade(symbol=symbol, entry_date=row.get("trade_date", ""), entry_price=price, side="buy"))
        elif signal == "sell" and holdings > 0:
            realized_value = holdings * price
            pnl = realized_value - (holdings * trades[-1].entry_price if trades else 0.0)
            cash = realized_value
            holdings = 0.0
            if trades:
                trade = trades[-1]
                trade.exit_date = row.get("trade_date", "")
                trade.exit_price = price
                trade.pnl = pnl
                trade.return_pct = (price / trade.entry_price - 1.0) if trade.entry_price else 0.0
            last_signal = "sell"

        portfolio_value = cash + holdings * price
        equity_curve.append(portfolio_value)

        if portfolio_value > peak_value:
            peak_value = portfolio_value
        current_drawdown = (peak_value - portfolio_value) / peak_value if peak_value else 0.0
        max_drawdown = max(max_drawdown, current_drawdown)

        if index > 0:
            prior_value = equity_curve[-2]
            if prior_value:
                daily_returns.append((portfolio_value - prior_value) / prior_value)

        last_close = price

    if holdings > 0 and last_close is not None:
        entry_shares = holdings
        final_position_value = holdings * last_close
        cash = final_position_value
        holdings = 0.0
        if trades:
            trade = trades[-1]
            trade.exit_date = history[-1].get("trade_date", "")
            trade.exit_price = last_close
            trade.pnl = final_position_value - (entry_shares * trade.entry_price)
            trade.return_pct = (last_close / trade.entry_price - 1.0) if trade.entry_price else 0.0

    final_value = cash + holdings * (last_close or 0.0)
    total_return = (final_value / initial_cash) - 1.0 if initial_cash else 0.0

    completed_trades = [trade for trade in trades if trade.exit_date is not None and trade.exit_price is not None]
    winning_trades = sum(1 for trade in completed_trades if trade.return_pct > 0.0)
    win_rate = (winning_trades / len(completed_trades)) if completed_trades else 0.0
    if daily_returns:
        mean_return = sum(daily_returns) / len(daily_returns)
        variance = sum((value - mean_return) ** 2 for value in daily_returns) / len(daily_returns)
        sharpe_ratio = (mean_return / (variance ** 0.5)) if variance > 0 else 0.0
    else:
        sharpe_ratio = 0.0

    return BacktestResult(
        symbol=symbol,
        initial_cash=initial_cash,
        final_cash=cash,
        final_value=final_value,
        total_return=total_return,
        trade_count=len(completed_trades),
        winning_trades=winning_trades,
        win_rate=win_rate,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        equity_curve=equity_curve,
        trades=completed_trades,
        notes=["Simple long-only swing strategy using moving-average and recent-return signals"],
    )


def run_backtest(symbols: list[str], histories: dict[str, list[dict[str, Any]]], initial_cash: float = 100000.0) -> list[BacktestResult]:
    """Backtest each requested symbol independently and return a per-symbol summary."""
    return [run_symbol_backtest(symbol, histories.get(symbol, []), initial_cash=initial_cash) for symbol in symbols]
