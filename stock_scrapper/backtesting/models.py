"""Typed records used throughout deterministic portfolio backtesting."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


def _json_payload(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


class DictSerializable:
    """Dataclass mixin providing a persistence-friendly dictionary."""

    __slots__ = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type]


@dataclass(slots=True)
class Signal(DictSerializable):
    """A decision made after a session close using only available data."""

    symbol: str
    signal_date: str
    action: str
    reason: str
    classification: str
    market_regime: str
    opportunity_score: float | None
    risk_score: float | None
    confidence_score: float | None
    strategy_version: str
    configuration_hash: str
    signal_id: str | None = None
    run_id: str | None = None
    reference_price: float | None = None
    ranking_values: dict[str, Any] = field(default_factory=dict)
    indicators: dict[str, Any] = field(default_factory=dict)
    critical_quality_issue: bool = False
    accepted: bool = False
    rejection_reason: str | None = None
    created_at: str | None = None

    @property
    def config_hash(self) -> str:
        return self.configuration_hash

    def to_persistence_record(self) -> dict[str, Any]:
        """Map this signal to the ``backtest_signals`` columns."""
        return {
            "run_id": self.run_id,
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "signal_date": self.signal_date,
            "action": self.action,
            "classification": self.classification,
            "opportunity_score": self.opportunity_score,
            "risk_score": self.risk_score,
            "confidence_score": self.confidence_score,
            "market_regime": self.market_regime,
            "ranking_json": _json_payload(self.ranking_values),
            "reason": self.reason,
            "accepted": 1 if self.accepted else 0,
            "rejection_reason": self.rejection_reason,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class RankedCandidate(DictSerializable):
    """An eligible entry candidate with a deterministic portfolio rank."""

    symbol: str
    signal_date: str
    rank: int
    opportunity_score: float
    confidence_score: float
    risk_score: float
    relative_strength: float | None
    liquidity: float | None
    classification: str
    market_regime: str
    ranking_values: dict[str, Any] = field(default_factory=dict)
    signal_id: str | None = None


@dataclass(slots=True)
class RejectedCandidate(DictSerializable):
    """A candidate that could not enter, retaining the exact reason."""

    symbol: str
    signal_date: str
    reason: str
    opportunity_score: float | None = None
    confidence_score: float | None = None
    risk_score: float | None = None
    classification: str | None = None
    market_regime: str | None = None
    rank: int | None = None
    available_cash: float | None = None
    requested_notional: float | None = None
    ranking_values: dict[str, Any] = field(default_factory=dict)
    signal_id: str | None = None


@dataclass(slots=True)
class Order(DictSerializable):
    """An order scheduled for a future eligible trading session."""

    order_id: str
    run_id: str
    symbol: str
    side: str
    signal_date: str
    scheduled_execution_date: str | None
    quantity: float
    reference_price: float | None
    reason: str
    status: str = "pending"
    reserved_cash: float = 0.0
    created_at: str | None = None
    cancelled_at: str | None = None
    rejection_reason: str | None = None
    signal_id: str | None = None

    @property
    def scheduled_date(self) -> str | None:
        return self.scheduled_execution_date

    def to_persistence_record(self) -> dict[str, Any]:
        """Map this order to the ``backtest_orders`` columns."""
        return {
            "order_id": self.order_id,
            "run_id": self.run_id,
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "side": self.side,
            "signal_date": self.signal_date,
            "scheduled_date": self.scheduled_execution_date,
            "status": self.status,
            "quantity": self.quantity,
            "reference_price": self.reference_price,
            "reason": self.reason,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class Fill(DictSerializable):
    """An executed order with explicit friction and reference pricing."""

    fill_id: str
    order_id: str
    run_id: str
    symbol: str
    side: str
    execution_date: str
    quantity: float
    reference_price: float
    fill_price: float
    commission: float
    slippage: float
    notional: float
    ambiguous_daily_bar: bool = False
    ambiguity_policy: str | None = None
    created_at: str | None = None

    @property
    def fill_date(self) -> str:
        return self.execution_date

    def to_persistence_record(self) -> dict[str, Any]:
        """Map this fill to the ``backtest_fills`` columns."""
        return {
            "fill_id": self.fill_id,
            "run_id": self.run_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "fill_date": self.execution_date,
            "side": self.side,
            "quantity": self.quantity,
            "reference_price": self.reference_price,
            "fill_price": self.fill_price,
            "commission": self.commission,
            "slippage": self.slippage,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class Position(DictSerializable):
    """An open long position in the one shared portfolio."""

    symbol: str
    quantity: float
    average_cost: float
    entry_date: str
    market_price: float
    highest_price: float
    realized_pnl: float = 0.0
    entry_commission: float = 0.0
    entry_slippage: float = 0.0
    holding_period_days: int = 0
    entry_signal_id: str | None = None
    entry_order_id: str | None = None

    @property
    def market_value(self) -> float:
        return self.quantity * self.market_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.average_cost

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            market_value=self.market_value,
            cost_basis=self.cost_basis,
            unrealized_pnl=self.unrealized_pnl,
        )
        return payload


@dataclass(slots=True)
class Trade(DictSerializable):
    """A completed position with a fully reproducible decision trail."""

    trade_id: str
    run_id: str
    symbol: str
    signal_date: str
    execution_date: str
    opportunity_score: float | None
    risk_score: float | None
    confidence_score: float | None
    classification: str
    market_regime: str
    ranking_values: dict[str, Any]
    quantity: float
    reference_price: float
    fill_price: float
    commission: float
    slippage: float
    entry_reason: str
    exit_reason: str
    strategy_version: str
    configuration_hash: str
    exit_signal_date: str | None = None
    exit_execution_date: str | None = None
    exit_reference_price: float | None = None
    exit_fill_price: float | None = None
    exit_commission: float = 0.0
    exit_slippage: float = 0.0
    realized_pnl: float | None = None
    return_pct: float | None = None
    holding_period_days: int | None = None
    ambiguous_daily_bar: bool = False
    ambiguity_policy: str | None = None
    entry_signal_id: str | None = None
    entry_order_id: str | None = None
    exit_order_id: str | None = None

    @property
    def total_commission(self) -> float:
        return self.commission + self.exit_commission

    @property
    def total_slippage(self) -> float:
        return self.slippage + self.exit_slippage

    @property
    def config_hash(self) -> str:
        return self.configuration_hash

    @property
    def entry_date(self) -> str:
        return self.execution_date

    @property
    def exit_date(self) -> str | None:
        return self.exit_execution_date

    @property
    def holding_days(self) -> int | None:
        return self.holding_period_days

    def to_persistence_record(self) -> dict[str, Any]:
        """Map this completed trade to the ``backtest_trades`` columns."""
        return {
            "trade_id": self.trade_id,
            "run_id": self.run_id,
            "symbol": self.symbol,
            "signal_date": self.signal_date,
            "entry_date": self.execution_date,
            "exit_signal_date": self.exit_signal_date,
            "exit_date": self.exit_execution_date,
            "quantity": self.quantity,
            "entry_reference_price": self.reference_price,
            "entry_fill_price": self.fill_price,
            "exit_reference_price": self.exit_reference_price,
            "exit_fill_price": self.exit_fill_price,
            "entry_commission": self.commission,
            "exit_commission": self.exit_commission,
            "slippage_cost": self.total_slippage,
            "realized_pnl": self.realized_pnl,
            "return_pct": self.return_pct,
            "holding_days": self.holding_period_days,
            "entry_reason": self.entry_reason,
            "exit_reason": self.exit_reason,
            "classification": self.classification,
            "market_regime": self.market_regime,
            "opportunity_score": self.opportunity_score,
            "risk_score": self.risk_score,
            "confidence_score": self.confidence_score,
            "ranking_json": _json_payload(self.ranking_values),
            "ambiguous_daily_bar": 1 if self.ambiguous_daily_bar else 0,
            "strategy_version": self.strategy_version,
            "configuration_hash": self.configuration_hash,
        }


@dataclass(slots=True)
class PortfolioSnapshot(DictSerializable):
    """End-of-session accounting state for the shared portfolio."""

    run_id: str
    snapshot_date: str
    cash: float
    reserved_cash: float
    market_value: float
    equity: float
    gross_exposure: float
    position_count: int
    realized_pnl: float
    unrealized_pnl: float
    commissions: float
    slippage: float
    daily_return: float | None
    benchmark_equity: float | None = None

    @property
    def trade_date(self) -> str:
        return self.snapshot_date

    def to_persistence_record(self) -> dict[str, Any]:
        """Map this snapshot to the ``backtest_equity_curve`` columns."""
        return {
            "run_id": self.run_id,
            "trade_date": self.snapshot_date,
            "cash": self.cash,
            "reserved_cash": self.reserved_cash,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "equity": self.equity,
            "gross_exposure": self.gross_exposure,
            "position_count": self.position_count,
            "daily_return": self.daily_return,
            "benchmark_equity": self.benchmark_equity,
        }


@dataclass(slots=True)
class BacktestRun(DictSerializable):
    """Metadata and reproducibility inputs for one simulation."""

    run_id: str
    strategy_name: str
    strategy_version: str
    configuration_hash: str
    benchmark_symbol: str
    start_date: str
    end_date: str
    warm_up_start_date: str | None
    initial_cash: float
    symbols: list[str]
    status: str
    configuration_snapshot: dict[str, Any]
    started_at: str
    completed_at: str | None = None
    ending_equity: float | None = None
    price_data_hash: str | None = None
    deterministic_result_hash: str | None = None
    error_summary: str | None = None
    application_version: str | None = None
    scoring_version: str | None = None
    schema_version: int | None = None
    git_commit_hash: str | None = None
    git_dirty: bool | None = None
    source_fingerprint: str | None = None
    python_version: str | None = None
    platform_info: str | None = None
    requested_start_date: str | None = None
    effective_start_date: str | None = None
    requested_end_date: str | None = None
    effective_end_date: str | None = None
    required_warmup_sessions: int | None = None
    available_warmup_sessions: int | None = None
    warmup_policy: str | None = None
    warmup_warning: str | None = None
    benchmark_sufficient: bool | None = None
    excluded_symbols: list[str] = field(default_factory=list)
    exclusion_reasons: dict[str, str] = field(default_factory=dict)
    universe_snapshot: dict[str, Any] = field(default_factory=dict)
    strategy_version_warning: str | None = None

    @property
    def config_hash(self) -> str:
        return self.configuration_hash

    @property
    def data_hash(self) -> str | None:
        return self.price_data_hash

    def to_persistence_record(self) -> dict[str, Any]:
        """Map this run to the ``backtest_runs`` columns."""
        return {
            "run_id": self.run_id,
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "warmup_start_date": self.warm_up_start_date,
            "benchmark_symbol": self.benchmark_symbol,
            "initial_cash": self.initial_cash,
            "ending_equity": self.ending_equity,
            "symbols_json": _json_payload(self.symbols),
            "configuration_hash": self.configuration_hash,
            "configuration_snapshot_json": _json_payload(self.configuration_snapshot),
            "data_hash": self.price_data_hash,
            "deterministic_result_hash": self.deterministic_result_hash,
            "error_summary": self.error_summary,
            "application_version": self.application_version,
            "scoring_version": self.scoring_version,
            "schema_version": self.schema_version,
            "git_commit_hash": self.git_commit_hash,
            "git_dirty": self.git_dirty,
            "source_fingerprint": self.source_fingerprint,
            "python_version": self.python_version,
            "platform_info": self.platform_info,
            "requested_start_date": self.requested_start_date,
            "effective_start_date": self.effective_start_date,
            "requested_end_date": self.requested_end_date,
            "effective_end_date": self.effective_end_date,
            "required_warmup_sessions": self.required_warmup_sessions,
            "available_warmup_sessions": self.available_warmup_sessions,
            "warmup_policy": self.warmup_policy,
            "warmup_warning": self.warmup_warning,
            "benchmark_sufficient": self.benchmark_sufficient,
            "excluded_symbols": self.excluded_symbols,
            "exclusion_reasons": self.exclusion_reasons,
            "universe_snapshot": self.universe_snapshot,
            "strategy_version_warning": self.strategy_version_warning,
        }


@dataclass(slots=True)
class PerformanceMetrics(DictSerializable):
    """Portfolio and benchmark measurements for a completed backtest."""

    starting_equity: float
    ending_equity: float
    net_profit: float
    total_return: float | None
    cagr: float | None
    annualized_volatility: float | None
    maximum_drawdown: float | None
    drawdown_duration: int | None
    sharpe_ratio: float | None
    sortino_ratio: float | None
    calmar_ratio: float | None
    exposure: float | None
    turnover: float | None
    number_of_trades: int
    win_rate: float | None
    average_win: float | None
    average_loss: float | None
    best_trade: float | None
    worst_trade: float | None
    profit_factor: float | None
    expectancy: float | None
    average_holding_period: float | None
    consecutive_wins: int | None
    consecutive_losses: int | None
    commission_cost: float | None
    slippage_cost: float | None
    monthly_returns: dict[str, float]
    annual_returns: dict[str, float]
    benchmark_total_return: float | None = None
    benchmark_maximum_drawdown: float | None = None
    return_vs_benchmark: float | None = None
    drawdown_vs_benchmark: float | None = None
    benchmark_cagr: float | None = None
    benchmark_annualized_volatility: float | None = None
    benchmark_sharpe_ratio: float | None = None
    benchmark_sortino_ratio: float | None = None
    benchmark_calmar_ratio: float | None = None
    active_return: float | None = None
    tracking_error: float | None = None
    information_ratio: float | None = None
    upside_capture: float | None = None
    downside_capture: float | None = None
    positive_benchmark_sessions_captured: float | None = None
    beta_to_benchmark: float | None = None
    correlation_to_benchmark: float | None = None
    cash_total_return: float = 0.0
    limitations: list[str] = field(default_factory=list)

    @property
    def return_vs_spy(self) -> float | None:
        return self.return_vs_benchmark

    @property
    def drawdown_vs_spy(self) -> float | None:
        return self.drawdown_vs_benchmark

    @property
    def trade_count(self) -> int:
        return self.number_of_trades

    def to_persistence_records(self, run_id: str) -> list[dict[str, Any]]:
        """Return normalized rows for the ``backtest_metrics`` key/value table."""
        records: list[dict[str, Any]] = []
        for name, value in self.to_dict().items():
            if isinstance(value, (dict, list)):
                records.append({"run_id": run_id, "metric_name": name, "metric_value": None, "metric_json": _json_payload(value)})
            else:
                records.append({"run_id": run_id, "metric_name": name, "metric_value": value, "metric_json": None})
        return records


@dataclass(slots=True)
class WalkForwardWindow(DictSerializable):
    """One fixed development, validation, or holdout evaluation window."""

    window_id: str
    walk_forward_run_id: str
    window_number: int
    window_type: str
    warm_up_start_date: str
    evaluation_start_date: str
    evaluation_end_date: str
    status: str
    backtest_run_id: str | None = None
    metrics: PerformanceMetrics | None = None
    error_summary: str | None = None
    warm_up_end_date: str | None = None
    development_start_date: str | None = None
    development_end_date: str | None = None
    validation_start_date: str | None = None
    validation_end_date: str | None = None
    holdout_start_date: str | None = None
    holdout_end_date: str | None = None

    @property
    def sequence_number(self) -> int:
        return self.window_number

    def to_persistence_record(self) -> dict[str, Any]:
        """Map explicit window boundaries to ``walk_forward_windows`` columns."""
        validation_start = self.validation_start_date
        validation_end = self.validation_end_date
        holdout_start = self.holdout_start_date
        holdout_end = self.holdout_end_date
        if self.window_type == "validation":
            validation_start = validation_start or self.evaluation_start_date
            validation_end = validation_end or self.evaluation_end_date
        elif self.window_type == "holdout":
            holdout_start = holdout_start or self.evaluation_start_date
            holdout_end = holdout_end or self.evaluation_end_date
        return {
            "window_id": self.window_id,
            "walk_forward_run_id": self.walk_forward_run_id,
            "sequence_number": self.window_number,
            "warmup_start": self.warm_up_start_date,
            "development_start": self.development_start_date,
            "development_end": self.development_end_date,
            "validation_start": validation_start,
            "validation_end": validation_end,
            "holdout_start": holdout_start,
            "holdout_end": holdout_end,
            "backtest_run_id": self.backtest_run_id,
            "status": self.status,
            "metrics_json": _json_payload(self.metrics.to_dict()) if self.metrics is not None else None,
        }


@dataclass(slots=True)
class WalkForwardRun(DictSerializable):
    """Metadata for a complete fixed-window validation exercise."""

    walk_forward_run_id: str
    strategy_name: str
    strategy_version: str
    configuration_hash: str
    start_date: str
    end_date: str
    status: str
    windows: list[WalkForwardWindow] = field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None
    error_summary: str | None = None
    benchmark_symbol: str | None = None
    symbols: list[str] = field(default_factory=list)
    configuration_snapshot: dict[str, Any] = field(default_factory=dict)

    @property
    def config_hash(self) -> str:
        return self.configuration_hash

    @property
    def run_id(self) -> str:
        return self.walk_forward_run_id

    def to_persistence_record(self) -> dict[str, Any]:
        """Map this run to the ``walk_forward_runs`` columns."""
        return {
            "run_id": self.walk_forward_run_id,
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "configuration_hash": self.configuration_hash,
            "configuration_snapshot_json": _json_payload(self.configuration_snapshot),
            "error_summary": self.error_summary,
        }


# Compatibility records retained for the original prototype engine. New code
# should use Trade and PerformanceMetrics above.
@dataclass(slots=True)
class BacktestTrade(DictSerializable):
    """Legacy per-symbol trade record used by the prototype engine."""

    symbol: str
    entry_date: str
    entry_price: float
    exit_date: str | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    return_pct: float = 0.0
    side: str = "buy"


@dataclass(slots=True)
class BacktestResult(DictSerializable):
    """Legacy per-symbol result retained until portfolio-engine integration."""

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
