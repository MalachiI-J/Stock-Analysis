"""Canonical score_v1 shared-portfolio historical simulation."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Mapping, Sequence
from uuid import uuid4

from stock_scrapper.analysis.service import AnalysisService
from stock_scrapper.backtesting.config import BacktestConfig
from stock_scrapper.backtesting.corporate_actions import AdjustedOHLC, build_adjusted_ohlc
from stock_scrapper.backtesting.metrics import calculate_performance_metrics
from stock_scrapper.backtesting.models import (
    BacktestRun,
    Fill,
    Order,
    PerformanceMetrics,
    PortfolioSnapshot,
    Position,
    RankedCandidate,
    RejectedCandidate,
    Signal,
    Trade,
)
from stock_scrapper.backtesting.persistence import persist_backtest
from stock_scrapper.models.analysis_models import AnalysisResult
from stock_scrapper.utilities.hashing import stable_sha256


@dataclass(slots=True)
class PortfolioBacktestResult:
    """Complete in-memory output for one reproducible portfolio simulation."""

    run: BacktestRun
    signals: list[Signal] = field(default_factory=list)
    ranked_candidates: list[RankedCandidate] = field(default_factory=list)
    rejected_candidates: list[RejectedCandidate] = field(default_factory=list)
    orders: list[Order] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    snapshots: list[PortfolioSnapshot] = field(default_factory=list)
    metrics: PerformanceMetrics | None = None

    @property
    def total_return(self) -> float:
        return float(self.metrics.total_return or 0.0) if self.metrics else 0.0

    @property
    def final_value(self) -> float:
        return float(self.metrics.ending_equity) if self.metrics else self.run.initial_cash

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        return float(self.metrics.win_rate or 0.0) if self.metrics else 0.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _effective_date(value: date | None, fallback: str) -> date:
    return value if value is not None else date.fromisoformat(fallback)


def _frequency_dates(sessions: Sequence[str], frequency: str) -> set[str]:
    """Select final available sessions for daily, ISO-weekly, or monthly signals."""
    if frequency == "daily":
        return set(sessions)
    selected: dict[tuple[int, int] | tuple[int, int, int], str] = {}
    for text in sessions:
        current = date.fromisoformat(text)
        if frequency == "weekly":
            iso = current.isocalendar()
            key: tuple[int, int] | tuple[int, int, int] = (iso.year, iso.week)
        elif frequency == "monthly":
            key = (current.year, current.month, 0)
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
        selected[key] = text
    return set(selected.values())


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


class _PortfolioSimulator:
    """Stateful event loop implementing costs, limits, stops, and T+1 fills."""

    def __init__(
        self,
        symbols: Sequence[str],
        histories: Mapping[str, list[dict[str, Any]]],
        analysis_rules: dict[str, Any],
        config: BacktestConfig,
        quality_by_symbol: Mapping[str, list[dict[str, Any]]] | None,
        run_id: str,
    ) -> None:
        self.symbols = list(dict.fromkeys(symbol.upper() for symbol in symbols))
        self.histories = {symbol.upper(): list(rows) for symbol, rows in histories.items()}
        self.analysis_rules = analysis_rules
        self.config = config
        self.quality_by_symbol = {
            symbol.upper(): list(issues)
            for symbol, issues in (quality_by_symbol or {}).items()
        }
        self.run_id = run_id
        self.analysis = AnalysisService(
            None,
            analysis_rules,
            list(dict.fromkeys([*self.symbols, *self.histories.keys()])),
        )
        self.bars: dict[str, dict[str, AdjustedOHLC]] = {}
        self.symbol_sessions: dict[str, list[str]] = {}
        for symbol, rows in self.histories.items():
            bars: dict[str, AdjustedOHLC] = {}
            for row in rows:
                bar = build_adjusted_ohlc(row)
                if bar.trade_date:
                    bars[str(bar.trade_date)[:10]] = bar
            self.bars[symbol] = bars
            self.symbol_sessions[symbol] = sorted(bars)

        benchmark_sessions = self.symbol_sessions.get(config.benchmark.upper(), [])
        universe_sessions = sorted(
            {session for symbol in self.symbols for session in self.symbol_sessions.get(symbol, [])}
        )
        available_sessions = benchmark_sessions or universe_sessions
        if not available_sessions:
            raise ValueError("No stored trading sessions are available for the requested universe")
        start = _effective_date(config.start_date, available_sessions[0])
        end = _effective_date(config.end_date, available_sessions[-1])
        self.sessions = [session for session in available_sessions if start.isoformat() <= session <= end.isoformat()]
        if not self.sessions:
            raise ValueError("No stored trading sessions fall inside the requested backtest dates")
        self.start_date = start.isoformat()
        self.end_date = end.isoformat()
        self.signal_dates = _frequency_dates(self.sessions, config.signal_frequency)
        self.rebalance_dates = _frequency_dates(self.sessions, config.rebalancing_frequency)
        self.analysis_dates = self.signal_dates | self.rebalance_dates
        self.analysis.prime_historical_features(
            self.histories,
            sorted(self.analysis_dates),
            self.symbols,
        )

        self.cash = float(config.initial_cash)
        self.positions: dict[str, Position] = {}
        self.open_trades: dict[str, Trade] = {}
        self.pending_orders: list[Order] = []
        self.signals: list[Signal] = []
        self.ranked_candidates: list[RankedCandidate] = []
        self.rejected: list[RejectedCandidate] = []
        self.orders: list[Order] = []
        self.fills: list[Fill] = []
        self.trades: list[Trade] = []
        self.snapshots: list[PortfolioSnapshot] = []
        self.cumulative_realized = 0.0
        self.cumulative_commission = 0.0
        self.cumulative_slippage = 0.0
        self.previous_equity = float(config.initial_cash)
        self.signal_counter = 0
        self.order_counter = 0
        self.fill_counter = 0
        self.trade_counter = 0
        self.last_results: dict[str, AnalysisResult] = {}
        self.benchmark_quantity: float | None = None
        self.benchmark_last_price: float | None = None

    @property
    def reserved_cash(self) -> float:
        return sum(order.reserved_cash for order in self.pending_orders if order.status == "pending" and order.side == "buy")

    def _identifier(self, kind: str) -> str:
        if kind == "signal":
            self.signal_counter += 1
            counter = self.signal_counter
        elif kind == "order":
            self.order_counter += 1
            counter = self.order_counter
        elif kind == "fill":
            self.fill_counter += 1
            counter = self.fill_counter
        else:
            self.trade_counter += 1
            counter = self.trade_counter
        return f"{self.run_id}-{kind}-{counter:08d}"

    def _new_signal(
        self,
        result: AnalysisResult,
        signal_date: str,
        action: str,
        reason: str,
        ranking: dict[str, Any] | None = None,
    ) -> Signal:
        signal = Signal(
            symbol=result.symbol,
            signal_date=signal_date,
            action=action,
            reason=reason,
            classification=result.classification,
            market_regime=result.market_regime,
            opportunity_score=result.opportunity_score,
            risk_score=result.risk_score,
            confidence_score=result.confidence_score,
            strategy_version=self.config.strategy_version,
            configuration_hash=self.config.configuration_hash,
            signal_id=self._identifier("signal"),
            run_id=self.run_id,
            reference_price=_number(result.indicators.get("latest_close")),
            ranking_values=ranking or {},
            indicators=dict(result.indicators),
            critical_quality_issue=any(
                "critical" in reason_text.lower() for reason_text in result.blocking_reasons
            ),
        )
        self.signals.append(signal)
        return signal

    def _next_session(self, symbol: str, after_date: str) -> str | None:
        return next(
            (
                session
                for session in self.symbol_sessions.get(symbol, [])
                if session > after_date and session <= self.end_date
            ),
            None,
        )

    def _commission(self, notional: float) -> float:
        return max(
            float(self.config.minimum_commission),
            notional * float(self.config.commission_basis_points) / 10_000.0,
        )

    def _adverse_price(self, reference: float, side: str, apply_costs: bool = True) -> float:
        if not apply_costs:
            return reference
        direction = 1.0 if side == "buy" else -1.0
        return reference * (
            1.0 + direction * float(self.config.slippage_basis_points) / 10_000.0
        )

    def _portfolio_equity(self) -> float:
        return self.cash + sum(position.market_value for position in self.positions.values())

    def _reject_signal(
        self,
        signal: Signal,
        reason: str,
        *,
        rank: int | None = None,
        requested_notional: float | None = None,
    ) -> None:
        self.rejected.append(
            RejectedCandidate(
                symbol=signal.symbol,
                signal_date=signal.signal_date,
                reason=reason,
                opportunity_score=signal.opportunity_score,
                confidence_score=signal.confidence_score,
                risk_score=signal.risk_score,
                classification=signal.classification,
                market_regime=signal.market_regime,
                rank=rank,
                available_cash=max(0.0, self.cash - self.reserved_cash),
                requested_notional=requested_notional,
                ranking_values=dict(signal.ranking_values),
                signal_id=signal.signal_id,
            )
        )

    def _schedule_order(
        self,
        signal: Signal,
        side: str,
        quantity: float,
        reason: str,
        reserved_cash: float = 0.0,
    ) -> Order | None:
        execution_date = self._next_session(signal.symbol, signal.signal_date)
        if execution_date is None:
            self._reject_signal(signal, "No next trading session is available")
            return None
        order = Order(
            order_id=self._identifier("order"),
            run_id=self.run_id,
            symbol=signal.symbol,
            side=side,
            signal_date=signal.signal_date,
            scheduled_execution_date=execution_date,
            quantity=float(quantity),
            reference_price=signal.reference_price,
            reason=reason,
            status="pending",
            reserved_cash=max(0.0, reserved_cash),
            created_at=_utc_now(),
            signal_id=signal.signal_id,
        )
        self.orders.append(order)
        self.pending_orders.append(order)
        return order

    def _execute_order(
        self,
        order: Order,
        execution_date: str,
        *,
        reference_override: float | None = None,
        ambiguous: bool = False,
        apply_costs: bool = True,
    ) -> bool:
        bar = self.bars.get(order.symbol, {}).get(execution_date)
        reference = reference_override
        if reference is None and bar is not None:
            reference = bar.adjusted_open
        if reference is None or reference <= 0:
            order.status = "rejected"
            order.rejection_reason = "Adjusted next-session open is unavailable"
            signal = next((item for item in self.signals if item.signal_id == order.signal_id), None)
            if signal is not None:
                self._reject_signal(signal, order.rejection_reason)
            return False

        quantity = float(order.quantity)
        if quantity <= 0:
            order.status = "rejected"
            order.rejection_reason = "Order quantity is not positive"
            return False
        fill_price = self._adverse_price(reference, order.side, apply_costs)
        notional = quantity * fill_price
        commission = self._commission(notional) if apply_costs else 0.0
        slippage_cost = quantity * abs(fill_price - reference)

        if order.side == "buy":
            if order.symbol in self.positions:
                order.status = "rejected"
                order.rejection_reason = "A position is already open"
                return False
            if len(self.positions) >= self.config.maximum_positions:
                order.status = "rejected"
                order.rejection_reason = "Maximum position limit reached"
                return False
            # A next-session gap can make the quantity sized at the signal close
            # exceed the configured cap.  Re-size against the actual adverse fill
            # and post-commission equity before committing cash.
            equity_before_fill = self._portfolio_equity()
            maximum_notional = equity_before_fill * float(self.config.maximum_position_weight)
            for _ in range(3):
                cap_commission = self._commission(maximum_notional) if apply_costs else 0.0
                maximum_notional = max(
                    0.0,
                    (equity_before_fill - cap_commission)
                    * float(self.config.maximum_position_weight),
                )
            maximum_quantity = maximum_notional / fill_price
            if not self.config.fractional_shares:
                maximum_quantity = float(math.floor(maximum_quantity))
            quantity = min(quantity, maximum_quantity)
            order.quantity = quantity
            if quantity <= 0:
                order.status = "rejected"
                order.rejection_reason = "Fill-time position cap cannot purchase one share"
                signal = next((item for item in self.signals if item.signal_id == order.signal_id), None)
                if signal is not None:
                    self._reject_signal(signal, order.rejection_reason)
                return False
            notional = quantity * fill_price
            commission = self._commission(notional) if apply_costs else 0.0
            slippage_cost = quantity * abs(fill_price - reference)
            reserve_floor = self._portfolio_equity() * float(self.config.cash_reserve)
            required = notional + commission
            if self.cash - required < reserve_floor - 1e-8:
                order.status = "rejected"
                order.rejection_reason = "Order is unaffordable after the cash reserve"
                signal = next((item for item in self.signals if item.signal_id == order.signal_id), None)
                if signal is not None:
                    self._reject_signal(signal, order.rejection_reason, requested_notional=required)
                return False
            self.cash -= required
            position = Position(
                symbol=order.symbol,
                quantity=quantity,
                average_cost=fill_price,
                entry_date=execution_date,
                market_price=fill_price,
                highest_price=fill_price,
                entry_commission=commission,
                entry_slippage=slippage_cost,
                entry_signal_id=order.signal_id,
                entry_order_id=order.order_id,
            )
            self.positions[order.symbol] = position
            signal = next(item for item in self.signals if item.signal_id == order.signal_id)
            trade = Trade(
                trade_id=self._identifier("trade"),
                run_id=self.run_id,
                symbol=order.symbol,
                signal_date=signal.signal_date,
                execution_date=execution_date,
                opportunity_score=signal.opportunity_score,
                risk_score=signal.risk_score,
                confidence_score=signal.confidence_score,
                classification=signal.classification,
                market_regime=signal.market_regime,
                ranking_values=dict(signal.ranking_values),
                quantity=quantity,
                reference_price=reference,
                fill_price=fill_price,
                commission=commission,
                slippage=slippage_cost,
                entry_reason=order.reason,
                exit_reason="",
                strategy_version=self.config.strategy_version,
                configuration_hash=self.config.configuration_hash,
                ambiguous_daily_bar=ambiguous,
                ambiguity_policy=(self.config.daily_bar_ambiguity_policy if ambiguous else None),
                entry_signal_id=order.signal_id,
                entry_order_id=order.order_id,
            )
            self.open_trades[order.symbol] = trade
        else:
            position = self.positions.get(order.symbol)
            if position is None:
                order.status = "rejected"
                order.rejection_reason = "No open position is available to sell"
                return False
            quantity = min(quantity, position.quantity)
            notional = quantity * fill_price
            commission = self._commission(notional) if apply_costs else 0.0
            slippage_cost = quantity * abs(fill_price - reference)
            proceeds = notional - commission
            self.cash += proceeds
            trade = self.open_trades.pop(order.symbol)
            realized = (
                quantity * (fill_price - trade.fill_price)
                - trade.commission
                - commission
            )
            self.cumulative_realized += realized
            trade.exit_signal_date = order.signal_date
            trade.exit_execution_date = execution_date
            trade.exit_reference_price = reference
            trade.exit_fill_price = fill_price
            trade.exit_commission = commission
            trade.exit_slippage = slippage_cost
            trade.realized_pnl = realized
            invested = quantity * trade.fill_price + trade.commission
            trade.return_pct = realized / invested if invested > 0 else None
            trade.holding_period_days = position.holding_period_days
            trade.exit_reason = order.reason
            trade.ambiguous_daily_bar = trade.ambiguous_daily_bar or ambiguous
            if ambiguous:
                trade.ambiguity_policy = self.config.daily_bar_ambiguity_policy
            trade.exit_order_id = order.order_id
            self.trades.append(trade)
            del self.positions[order.symbol]

        self.cumulative_commission += commission
        self.cumulative_slippage += slippage_cost
        fill = Fill(
            fill_id=self._identifier("fill"),
            order_id=order.order_id,
            run_id=self.run_id,
            symbol=order.symbol,
            side=order.side,
            execution_date=execution_date,
            quantity=quantity,
            reference_price=reference,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage_cost,
            notional=quantity * fill_price,
            ambiguous_daily_bar=ambiguous,
            ambiguity_policy=(self.config.daily_bar_ambiguity_policy if ambiguous else None),
            created_at=_utc_now(),
        )
        self.fills.append(fill)
        order.status = "filled"
        return True

    def _execute_pending(self, session: str) -> None:
        due = [
            order
            for order in self.pending_orders
            if order.status == "pending" and order.scheduled_execution_date == session
        ]
        for order in sorted(due, key=lambda item: (0 if item.side == "sell" else 1, item.symbol, item.order_id)):
            self._execute_order(order, session)
        self.pending_orders = [order for order in self.pending_orders if order.status == "pending"]

    def _stop_events(self, session: str) -> None:
        for symbol in sorted(list(self.positions)):
            position = self.positions.get(symbol)
            bar = self.bars.get(symbol, {}).get(session)
            trade = self.open_trades.get(symbol)
            if position is None or bar is None or trade is None:
                continue
            low = bar.adjusted_low
            high = bar.adjusted_high
            open_price = bar.adjusted_open
            if low is None or high is None:
                continue
            stop_levels: list[tuple[str, float]] = []
            if self.config.stop_loss is not None:
                stop_levels.append(("Stop loss", trade.fill_price * (1.0 - self.config.stop_loss)))
            if self.config.trailing_stop is not None:
                stop_levels.append(("Trailing stop", position.highest_price * (1.0 - self.config.trailing_stop)))
            active_stop = max(stop_levels, key=lambda item: item[1]) if stop_levels else None
            stop_hit = active_stop is not None and low <= active_stop[1]
            target_level = (
                trade.fill_price * (1.0 + self.config.profit_target)
                if self.config.profit_target is not None
                else None
            )
            target_hit = target_level is not None and high >= target_level
            ambiguous = bool(stop_hit and target_hit)
            if ambiguous:
                trade.ambiguous_daily_bar = True
                trade.ambiguity_policy = self.config.daily_bar_ambiguity_policy
                if self.config.daily_bar_ambiguity_policy == "skip_bar":
                    position.highest_price = max(position.highest_price, high)
                    continue
                choose_stop = self.config.daily_bar_ambiguity_policy == "adverse_first"
            else:
                choose_stop = bool(stop_hit)
            if stop_hit or target_hit:
                if choose_stop and active_stop is not None:
                    reason, trigger = active_stop
                    reference = min(open_price, trigger) if open_price is not None else trigger
                else:
                    reason = "Profit target"
                    assert target_level is not None
                    reference = max(open_price, target_level) if open_price is not None else target_level
                result = self.last_results.get(symbol) or AnalysisResult(
                    symbol=symbol,
                    as_of_date=session,
                    classification=trade.classification,
                    market_regime=trade.market_regime,
                    opportunity_score=trade.opportunity_score,
                    risk_score=trade.risk_score,
                    confidence_score=trade.confidence_score,
                )
                signal = self._new_signal(result, session, "exit", reason)
                order = Order(
                    order_id=self._identifier("order"),
                    run_id=self.run_id,
                    symbol=symbol,
                    side="sell",
                    signal_date=session,
                    scheduled_execution_date=session,
                    quantity=position.quantity,
                    reference_price=reference,
                    reason=reason,
                    created_at=_utc_now(),
                    signal_id=signal.signal_id,
                )
                self.orders.append(order)
                self._execute_order(
                    order,
                    session,
                    reference_override=reference,
                    ambiguous=ambiguous,
                )
            elif symbol in self.positions:
                position.highest_price = max(position.highest_price, high)

    def _entry_rejection_reason(self, result: AnalysisResult) -> str | None:
        if result.classification not in set(self.config.entry_thresholds.classifications):
            return f"Classification {result.classification} is not entry-eligible"
        if result.opportunity_score is None or result.opportunity_score < self.config.entry_thresholds.minimum_opportunity_score:
            return "Opportunity score is below the entry threshold"
        if result.risk_score is None or result.risk_score > self.config.maximum_risk:
            return "Risk score is unavailable or above the maximum"
        if result.confidence_score is None or result.confidence_score < self.config.minimum_confidence:
            return "Confidence score is unavailable or below the minimum"
        if result.market_regime not in set(self.config.allowed_market_regimes):
            return f"Market regime {result.market_regime} is not allowed"
        if any("critical" in text.lower() for text in result.blocking_reasons):
            return "A critical data-quality issue blocks entry"
        liquidity = _number(result.indicators.get("twenty_day_average_dollar_volume"))
        if liquidity is None:
            return "Minimum liquidity cannot be established"
        if liquidity < self.config.entry_thresholds.minimum_average_dollar_volume:
            return "Average dollar volume is below the minimum"
        if result.symbol in self.positions:
            return "A position is already held"
        if any(order.symbol == result.symbol and order.status == "pending" for order in self.pending_orders):
            return "An order is already pending"
        return None

    def _exit_reason(self, result: AnalysisResult, position: Position) -> str | None:
        thresholds = self.config.exit_thresholds
        if thresholds.exit_on_stress and result.market_regime == "Stress":
            return "Market entered Stress"
        if result.classification in set(thresholds.classifications):
            return f"Classification became {result.classification}"
        if result.risk_score is None or result.risk_score > thresholds.maximum_risk_score:
            return "Risk score exceeded the exit maximum or became unavailable"
        if result.opportunity_score is None or result.opportunity_score < thresholds.minimum_opportunity_score:
            return "Opportunity score fell below the exit threshold"
        if result.confidence_score is None or result.confidence_score < thresholds.minimum_confidence_score:
            return "Confidence score fell below the exit threshold"
        distance200 = _number(result.indicators.get("distance_from_sma200"))
        if thresholds.exit_below_sma200 and distance200 is not None and distance200 < 0:
            return "Price closed below the 200-day moving average"
        if (
            self.config.maximum_holding_period is not None
            and position.holding_period_days >= self.config.maximum_holding_period
        ):
            return "Maximum holding period reached"
        return None

    def _rank(self, result: AnalysisResult) -> tuple[Any, ...]:
        relative = _number(result.indicators.get("benchmark_relative_return_252"))
        liquidity = _number(result.indicators.get("twenty_day_average_dollar_volume"))
        return (
            -(result.opportunity_score or -math.inf),
            -(result.confidence_score or -math.inf),
            result.risk_score if result.risk_score is not None else math.inf,
            -(relative if relative is not None else -math.inf),
            -(liquidity if liquidity is not None else -math.inf),
            result.symbol,
        )

    def _analyze_close(self, session: str) -> None:
        batch = self.analysis.analyze_loaded_many_as_of(
            self.symbols,
            self.histories,
            session,
            quality_by_symbol=self.quality_by_symbol,
            persist=False,
        )
        results = {result.symbol: result for result in batch.results}
        self.last_results = results

        pending_sell_symbols = {
            order.symbol for order in self.pending_orders if order.status == "pending" and order.side == "sell"
        }
        for symbol in sorted(self.positions):
            if symbol in pending_sell_symbols:
                continue
            result = results[symbol]
            reason = self._exit_reason(result, self.positions[symbol])
            if reason:
                signal = self._new_signal(result, session, "exit", reason)
                self._schedule_order(
                    signal,
                    "sell",
                    self.positions[symbol].quantity,
                    reason,
                )

        if session not in self.rebalance_dates:
            return
        eligible: list[AnalysisResult] = []
        for symbol in sorted(results):
            result = results[symbol]
            reason = self._entry_rejection_reason(result)
            if reason is None:
                eligible.append(result)
            else:
                signal = self._new_signal(result, session, "hold", reason)
                self._reject_signal(signal, reason)
        eligible.sort(key=self._rank)
        pending_buys = sum(
            order.status == "pending" and order.side == "buy" for order in self.pending_orders
        )
        slots = max(0, self.config.maximum_positions - len(self.positions) - pending_buys)
        selected = eligible[:slots]
        for rank, result in enumerate(eligible, start=1):
            ranking = {
                "opportunity": result.opportunity_score,
                "confidence": result.confidence_score,
                "risk": result.risk_score,
                "relative_strength": result.indicators.get("benchmark_relative_return_252"),
                "liquidity": result.indicators.get("twenty_day_average_dollar_volume"),
                "rank": rank,
            }
            ranked = RankedCandidate(
                symbol=result.symbol,
                signal_date=session,
                rank=rank,
                opportunity_score=float(result.opportunity_score),
                confidence_score=float(result.confidence_score),
                risk_score=float(result.risk_score),
                relative_strength=_number(ranking["relative_strength"]),
                liquidity=_number(ranking["liquidity"]),
                classification=result.classification,
                market_regime=result.market_regime,
                ranking_values=ranking,
            )
            self.ranked_candidates.append(ranked)
            signal = self._new_signal(result, session, "entry", "score_v1 entry rules satisfied", ranking)
            ranked.signal_id = signal.signal_id
            if result not in selected:
                self._reject_signal(signal, "Maximum position limit left no available slot", rank=rank)
                continue

            equity = self._portfolio_equity()
            if self.config.position_sizing == "volatility_adjusted":
                volatility_key = {
                    20: "twenty_day_volatility",
                    60: "sixty_day_volatility",
                    252: "two_hundred_fifty_two_day_volatility",
                }[self.config.volatility_lookback_days]
                volatilities = [
                    _number(item.indicators.get(volatility_key)) for item in selected
                ]
                if any(value is None or value <= 0 for value in volatilities):
                    self._reject_signal(signal, "Volatility-adjusted sizing requires valid volatility", rank=rank)
                    continue
                inverse = 1.0 / float(result.indicators[volatility_key])
                inverse_total = sum(1.0 / float(value) for value in volatilities if value is not None)
                target_weight = min(
                    self.config.maximum_position_weight,
                    (1.0 - self.config.cash_reserve) * inverse / inverse_total,
                )
            else:
                target_weight = min(
                    self.config.maximum_position_weight,
                    (1.0 - self.config.cash_reserve) / self.config.maximum_positions,
                )
            reference = _number(result.indicators.get("latest_close"))
            if reference is None or reference <= 0:
                self._reject_signal(signal, "Reference close is unavailable", rank=rank)
                continue
            target_notional = equity * target_weight
            quantity = target_notional / reference
            if not self.config.fractional_shares:
                quantity = float(math.floor(quantity))
            if quantity <= 0:
                self._reject_signal(signal, "Target allocation cannot purchase one share", rank=rank)
                continue
            estimated_fill = self._adverse_price(reference, "buy")
            estimated_cost = quantity * estimated_fill + self._commission(quantity * estimated_fill)
            reserve_floor = equity * self.config.cash_reserve
            available = self.cash - self.reserved_cash - reserve_floor
            if estimated_cost > available + 1e-8:
                self._reject_signal(
                    signal,
                    "Order is unaffordable after pending orders and cash reserve",
                    rank=rank,
                    requested_notional=estimated_cost,
                )
                continue
            self._schedule_order(
                signal,
                "buy",
                quantity,
                "score_v1 entry rules satisfied",
                reserved_cash=estimated_cost,
            )

    def _mark(self, session: str) -> None:
        for symbol, position in self.positions.items():
            bar = self.bars.get(symbol, {}).get(session)
            if bar is not None and bar.adjusted_close is not None:
                position.market_price = bar.adjusted_close
            position.holding_period_days += 1

        benchmark_bar = self.bars.get(self.config.benchmark.upper(), {}).get(session)
        if benchmark_bar is not None and benchmark_bar.adjusted_close is not None:
            self.benchmark_last_price = benchmark_bar.adjusted_close
            if self.benchmark_quantity is None:
                self.benchmark_quantity = self.config.initial_cash / benchmark_bar.adjusted_close
        benchmark_equity = (
            self.benchmark_quantity * self.benchmark_last_price
            if self.benchmark_quantity is not None and self.benchmark_last_price is not None
            else None
        )
        market_value = sum(position.market_value for position in self.positions.values())
        unrealized = sum(position.unrealized_pnl for position in self.positions.values())
        equity = self.cash + market_value
        daily_return = equity / self.previous_equity - 1.0 if self.previous_equity > 0 else None
        snapshot = PortfolioSnapshot(
            run_id=self.run_id,
            snapshot_date=session,
            cash=self.cash,
            reserved_cash=self.reserved_cash,
            market_value=market_value,
            equity=equity,
            gross_exposure=(market_value / equity if equity > 0 else 0.0),
            position_count=len(self.positions),
            realized_pnl=self.cumulative_realized,
            unrealized_pnl=unrealized,
            commissions=self.cumulative_commission,
            slippage=self.cumulative_slippage,
            daily_return=daily_return,
            benchmark_equity=benchmark_equity,
        )
        self.snapshots.append(snapshot)
        self.previous_equity = equity

    def _final_liquidation(self, session: str) -> None:
        if not self.config.final_liquidation.enabled:
            return
        for symbol in sorted(list(self.positions)):
            position = self.positions[symbol]
            bar = self.bars.get(symbol, {}).get(session)
            reference = bar.adjusted_close if bar is not None else None
            trade = self.open_trades[symbol]
            result = self.last_results.get(symbol) or AnalysisResult(
                symbol=symbol,
                as_of_date=session,
                classification=trade.classification,
                market_regime=trade.market_regime,
                opportunity_score=trade.opportunity_score,
                risk_score=trade.risk_score,
                confidence_score=trade.confidence_score,
            )
            signal = self._new_signal(result, session, "exit", "Final liquidation")
            order = Order(
                order_id=self._identifier("order"),
                run_id=self.run_id,
                symbol=symbol,
                side="sell",
                signal_date=session,
                scheduled_execution_date=session,
                quantity=position.quantity,
                reference_price=reference,
                reason="Final liquidation",
                created_at=_utc_now(),
                signal_id=signal.signal_id,
            )
            self.orders.append(order)
            filled = self._execute_order(
                order,
                session,
                reference_override=reference,
                apply_costs=self.config.final_liquidation.apply_costs,
            )
            if not filled:
                raise ValueError(
                    f"Final liquidation failed for {symbol} on {session}: "
                    f"{order.rejection_reason or 'no executable adjusted close'}"
                )

    def run(self) -> PortfolioBacktestResult:
        final_session = self.sessions[-1]
        for session in self.sessions:
            self._execute_pending(session)
            self._stop_events(session)
            if session in self.analysis_dates and session != final_session:
                self._analyze_close(session)
            if session == final_session:
                self._final_liquidation(session)
            self._mark(session)
            if self.cash < -1e-7:
                raise AssertionError("Portfolio cash became negative")

        if self.config.final_liquidation.enabled and self.positions:
            raise AssertionError("Final liquidation completed with open positions")

        metrics = calculate_performance_metrics(
            self.snapshots,
            self.trades,
            fills=self.fills,
            risk_free_rate=self.config.risk_free_rate,
            annualization_factor=self.config.annualization_factor,
        )
        warmup_sessions = [
            session
            for session in self.symbol_sessions.get(self.config.benchmark.upper(), [])
            if session < self.start_date
        ]
        warmup_start = (
            warmup_sessions[-self.config.warm_up_days]
            if len(warmup_sessions) >= self.config.warm_up_days
            else (warmup_sessions[0] if warmup_sessions else None)
        )
        run = BacktestRun(
            run_id=self.run_id,
            strategy_name=self.config.strategy_name,
            strategy_version=self.config.strategy_version,
            configuration_hash=self.config.configuration_hash,
            benchmark_symbol=self.config.benchmark,
            start_date=self.start_date,
            end_date=self.end_date,
            warm_up_start_date=warmup_start,
            initial_cash=self.config.initial_cash,
            symbols=self.symbols,
            status="completed",
            configuration_snapshot=self.config.to_dict(),
            started_at=_utc_now(),
            completed_at=_utc_now(),
            ending_equity=metrics.ending_equity,
            price_data_hash=stable_sha256(
                {
                    symbol: [
                        row
                        for row in self.histories.get(symbol, [])
                        if str(row.get("trade_date", ""))[:10] <= self.end_date
                    ]
                    for symbol in sorted(self.histories)
                }
            ),
        )
        normalized = {
            "signals": [
                {key: value for key, value in signal.to_dict().items() if key not in {"run_id", "signal_id"}}
                for signal in self.signals
            ],
            "orders": [
                {
                    key: value
                    for key, value in order.to_dict().items()
                    if key not in {"run_id", "order_id", "signal_id", "created_at", "cancelled_at"}
                }
                for order in self.orders
            ],
            "trades": [
                {
                    key: value
                    for key, value in trade.to_dict().items()
                    if key not in {
                        "run_id",
                        "trade_id",
                        "entry_signal_id",
                        "entry_order_id",
                        "exit_order_id",
                    }
                }
                for trade in self.trades
            ],
            "equity": [
                {key: value for key, value in snapshot.to_dict().items() if key != "run_id"}
                for snapshot in self.snapshots
            ],
            "metrics": metrics.to_dict(),
        }
        run.deterministic_result_hash = stable_sha256(normalized)
        return PortfolioBacktestResult(
            run=run,
            signals=self.signals,
            ranked_candidates=self.ranked_candidates,
            rejected_candidates=self.rejected,
            orders=self.orders,
            fills=self.fills,
            trades=self.trades,
            snapshots=self.snapshots,
            metrics=metrics,
        )


def run_portfolio_backtest(
    symbols: Sequence[str],
    histories: Mapping[str, list[dict[str, Any]]],
    analysis_rules: dict[str, Any],
    config: BacktestConfig,
    *,
    quality_by_symbol: Mapping[str, list[dict[str, Any]]] | None = None,
    persist_conn: sqlite3.Connection | None = None,
    commit_persistence: bool = True,
    run_id: str | None = None,
) -> PortfolioBacktestResult:
    """Run score_v1 with one cash account and optionally persist it atomically.

    ``commit_persistence=False`` lets a caller compose several persisted runs
    inside a larger transaction, as the walk-forward workflow does.
    """
    if config.strategy_name != "score_v1":
        raise ValueError(f"Unknown strategy: {config.strategy_name}")
    effective_run_id = run_id or (
        "backtest-"
        + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        + "-"
        + uuid4().hex[:8]
    )
    simulator = _PortfolioSimulator(
        symbols,
        histories,
        analysis_rules,
        config,
        quality_by_symbol,
        effective_run_id,
    )
    result = simulator.run()
    if persist_conn is not None:
        persist_backtest(
            persist_conn,
            result.run,
            result.signals,
            result.rejected_candidates,
            result.orders,
            result.fills,
            result.trades,
            result.snapshots,
            result.metrics,
        )
        if commit_persistence:
            persist_conn.commit()
    return result


# Concise public alias used by CLI and tests.
run_backtest = run_portfolio_backtest
