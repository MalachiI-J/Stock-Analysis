"""Mathematically explicit portfolio-performance metrics.

Ratios with an unavailable denominator return ``None``. No missing value is
silently replaced by zero, while legitimate zero activity/cost remains zero.
Daily statistics use sample standard deviation and 252-session annualization
by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite, sqrt
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

from stock_scrapper.backtesting.models import PerformanceMetrics


@dataclass(frozen=True, slots=True)
class _EquityPoint:
    point_date: date | None
    value: float


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _get(item: Any, *names: str) -> Any:
    if isinstance(item, Mapping):
        for name in names:
            if name in item:
                return item[name]
        return None
    for name in names:
        if hasattr(item, name):
            return getattr(item, name)
    return None


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError as exc:
            raise ValueError(f"Invalid equity-curve date: {value}") from exc
    raise ValueError(f"Unsupported equity-curve date: {value!r}")


def _normalize_equity_curve(curve: Sequence[Any]) -> list[_EquityPoint]:
    points: list[_EquityPoint] = []
    for item in curve:
        point_date: date | None
        value: Any
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            point_date = None
            value = item
        elif isinstance(item, tuple) and len(item) == 2:
            point_date = _parse_date(item[0])
            value = item[1]
        else:
            point_date = _parse_date(_get(item, "snapshot_date", "date", "trade_date"))
            value = _get(item, "equity", "portfolio_equity", "value", "close")
        numeric_value = _number(value, "equity")
        if numeric_value < 0:
            raise ValueError("equity cannot be negative in a long-only, no-leverage backtest")
        points.append(_EquityPoint(point_date, numeric_value))

    dated = [point.point_date for point in points if point.point_date is not None]
    if dated and len(dated) != len(points):
        raise ValueError("equity-curve dates must be present for every point or for none")
    if dated and any(current <= previous for previous, current in zip(dated, dated[1:])):
        raise ValueError("equity-curve dates must be strictly increasing")
    return points


def calculate_total_return(starting_equity: float, ending_equity: float) -> float | None:
    """Return simple total return, or ``None`` when starting equity is zero."""
    start = _number(starting_equity, "starting_equity")
    end = _number(ending_equity, "ending_equity")
    if start <= 0:
        return None
    return end / start - 1.0


def calculate_cagr(starting_equity: float, ending_equity: float, years: float) -> float | None:
    """Return compound annual growth over a positive duration."""
    start = _number(starting_equity, "starting_equity")
    end = _number(ending_equity, "ending_equity")
    duration = _number(years, "years")
    if start <= 0 or end < 0 or duration <= 0:
        return None
    if end == 0:
        return -1.0
    return (end / start) ** (1.0 / duration) - 1.0


def calculate_annualized_volatility(returns: Sequence[float], annualization_factor: int = 252) -> float | None:
    """Annualize sample standard deviation of daily returns."""
    values = [_number(value, "return") for value in returns]
    if len(values) < 2 or annualization_factor <= 0:
        return None
    return stdev(values) * sqrt(annualization_factor)


def calculate_sharpe_ratio(
    returns: Sequence[float], risk_free_rate: float = 0.0, annualization_factor: int = 252
) -> float | None:
    """Calculate annualized Sharpe using a geometrically converted daily risk-free rate."""
    values = [_number(value, "return") for value in returns]
    if len(values) < 2 or annualization_factor <= 0 or risk_free_rate <= -1.0:
        return None
    daily_risk_free = (1.0 + risk_free_rate) ** (1.0 / annualization_factor) - 1.0
    excess = [value - daily_risk_free for value in values]
    deviation = stdev(excess)
    if deviation == 0:
        return None
    return mean(excess) / deviation * sqrt(annualization_factor)


def calculate_sortino_ratio(
    returns: Sequence[float], risk_free_rate: float = 0.0, annualization_factor: int = 252
) -> float | None:
    """Calculate annualized Sortino using all periods in downside deviation."""
    values = [_number(value, "return") for value in returns]
    if not values or annualization_factor <= 0 or risk_free_rate <= -1.0:
        return None
    daily_risk_free = (1.0 + risk_free_rate) ** (1.0 / annualization_factor) - 1.0
    excess = [value - daily_risk_free for value in values]
    downside_deviation = sqrt(mean(min(value, 0.0) ** 2 for value in excess))
    if downside_deviation == 0:
        return None
    return mean(excess) / downside_deviation * sqrt(annualization_factor)


def calculate_drawdown(equity_values: Sequence[float]) -> tuple[float | None, int | None]:
    """Return maximum drawdown magnitude and longest underwater duration in bars."""
    values = [_number(value, "equity") for value in equity_values]
    if not values:
        return None, None
    if any(value < 0 for value in values):
        raise ValueError("equity cannot be negative")

    peak = values[0]
    maximum_drawdown = 0.0
    current_duration = 0
    maximum_duration = 0
    for value in values[1:]:
        if value >= peak:
            peak = value
            current_duration = 0
            continue
        current_duration += 1
        maximum_duration = max(maximum_duration, current_duration)
        if peak > 0:
            maximum_drawdown = max(maximum_drawdown, (peak - value) / peak)
    return maximum_drawdown, maximum_duration


def calculate_maximum_drawdown(equity_values: Sequence[float]) -> float | None:
    """Return maximum peak-to-trough loss as a positive fraction."""
    return calculate_drawdown(equity_values)[0]


def calculate_drawdown_duration(equity_values: Sequence[float]) -> int | None:
    """Return the maximum number of consecutive bars below a prior peak."""
    return calculate_drawdown(equity_values)[1]


def calculate_calmar_ratio(cagr: float | None, maximum_drawdown: float | None) -> float | None:
    """Return CAGR divided by maximum drawdown when both are available."""
    if cagr is None or maximum_drawdown is None or maximum_drawdown <= 0:
        return None
    return cagr / maximum_drawdown


def _daily_returns(values: Sequence[float]) -> tuple[list[float], bool]:
    returns: list[float] = []
    unavailable = False
    for previous, current in zip(values, values[1:]):
        if previous == 0:
            unavailable = True
            continue
        returns.append(current / previous - 1.0)
    return returns, unavailable


def calculate_daily_returns(equity_values: Sequence[float]) -> list[float]:
    """Calculate all defined simple period returns from an equity sequence."""
    values = [_number(value, "equity") for value in equity_values]
    return _daily_returns(values)[0]


def _period_returns(points: Sequence[_EquityPoint], yearly: bool) -> dict[str, float]:
    if not points or any(point.point_date is None for point in points):
        return {}
    period_ends: dict[str, float] = {}
    for point in points:
        assert point.point_date is not None
        key = point.point_date.strftime("%Y" if yearly else "%Y-%m")
        period_ends[key] = point.value
    results: dict[str, float] = {}
    prior_value = points[0].value
    for key, end_value in period_ends.items():
        if prior_value == 0:
            prior_value = end_value
            continue
        results[key] = end_value / prior_value - 1.0
        prior_value = end_value
    return results


def calculate_monthly_returns(equity_curve: Sequence[Any]) -> dict[str, float]:
    """Return end-of-month returns keyed by ``YYYY-MM``."""
    return _period_returns(_normalize_equity_curve(equity_curve), yearly=False)


def calculate_annual_returns(equity_curve: Sequence[Any]) -> dict[str, float]:
    """Return end-of-year returns keyed by ``YYYY``."""
    return _period_returns(_normalize_equity_curve(equity_curve), yearly=True)


def _trade_pnl(trade: Any) -> float | None:
    value = _get(trade, "realized_pnl", "pnl")
    if value is not None:
        return _number(value, "trade pnl")
    quantity = _get(trade, "quantity")
    entry = _get(trade, "fill_price", "entry_price")
    exit_price = _get(trade, "exit_fill_price", "exit_price")
    if quantity is None or entry is None or exit_price is None:
        return None
    return _number(quantity, "trade quantity") * (
        _number(exit_price, "exit fill price") - _number(entry, "entry fill price")
    )


def _trade_holding_period(trade: Any) -> float | None:
    value = _get(trade, "holding_period_days")
    if value is not None:
        return _number(value, "holding period")
    entry = _parse_date(_get(trade, "execution_date", "entry_date"))
    exit_date = _parse_date(_get(trade, "exit_execution_date", "exit_date"))
    if entry is None or exit_date is None:
        return None
    return float((exit_date - entry).days)


def _trade_cost(trade: Any, total_name: str, entry_name: str, exit_name: str) -> float | None:
    total = _get(trade, total_name)
    if total is not None:
        value = _number(total, total_name)
        if value < 0:
            raise ValueError(f"{total_name} cannot be negative")
        return value
    entry = _get(trade, entry_name)
    exit_value = _get(trade, exit_name)
    if entry is None or exit_value is None:
        return None
    total_value = _number(entry, entry_name) + _number(exit_value, exit_name)
    if total_value < 0:
        raise ValueError(f"{total_name} cannot be negative")
    return total_value


def _trade_statistics(trades: Sequence[Any]) -> dict[str, Any]:
    pnl_by_trade = [_trade_pnl(trade) for trade in trades]
    pnl_complete = all(value is not None for value in pnl_by_trade)
    pnl_values = [value for value in pnl_by_trade if value is not None]
    wins = [value for value in pnl_values if value > 0]
    losses = [value for value in pnl_values if value < 0]
    maximum_wins = 0
    maximum_losses = 0
    current_wins = 0
    current_losses = 0
    for pnl in pnl_values:
        if pnl > 0:
            current_wins += 1
            current_losses = 0
        elif pnl < 0:
            current_losses += 1
            current_wins = 0
        else:
            current_wins = 0
            current_losses = 0
        maximum_wins = max(maximum_wins, current_wins)
        maximum_losses = max(maximum_losses, current_losses)

    period_by_trade = [_trade_holding_period(trade) for trade in trades]
    periods_complete = all(value is not None for value in period_by_trade)
    periods = [value for value in period_by_trade if value is not None]
    commissions = [_trade_cost(trade, "total_commission", "commission", "exit_commission") for trade in trades]
    slippage = [_trade_cost(trade, "total_slippage", "slippage", "exit_slippage") for trade in trades]
    gross_loss = abs(sum(losses))
    limitations: list[str] = []
    if trades and not pnl_complete:
        limitations.append("Trade outcome metrics are unavailable because one or more trades lack realized P&L")
    if trades and not periods_complete:
        limitations.append("Average holding period is unavailable because one or more trades lack dates")
    if trades and any(value is None for value in commissions):
        limitations.append("Commission cost is unavailable because one or more trades lack cost data")
    if trades and any(value is None for value in slippage):
        limitations.append("Slippage cost is unavailable because one or more trades lack cost data")
    return {
        "number_of_trades": len(trades),
        "win_rate": len(wins) / len(pnl_values) if pnl_values and pnl_complete else None,
        "average_win": mean(wins) if wins and pnl_complete else None,
        "average_loss": mean(losses) if losses and pnl_complete else None,
        "best_trade": max(pnl_values) if pnl_values and pnl_complete else None,
        "worst_trade": min(pnl_values) if pnl_values and pnl_complete else None,
        "profit_factor": sum(wins) / gross_loss if gross_loss > 0 and pnl_complete else None,
        "expectancy": mean(pnl_values) if pnl_values and pnl_complete else None,
        "average_holding_period": mean(periods) if periods and periods_complete else None,
        "consecutive_wins": maximum_wins if pnl_complete else None,
        "consecutive_losses": maximum_losses if pnl_complete else None,
        "commission_cost": sum(value for value in commissions if value is not None) if all(value is not None for value in commissions) else None,
        "slippage_cost": sum(value for value in slippage if value is not None) if all(value is not None for value in slippage) else None,
        "limitations": limitations,
    }


def _turnover_notional(trades: Sequence[Any]) -> float | None:
    total = 0.0
    for trade in trades:
        quantity = _get(trade, "quantity")
        entry_price = _get(trade, "fill_price", "entry_price")
        exit_price = _get(trade, "exit_fill_price", "exit_price")
        if quantity is None or entry_price is None:
            return None
        quantity_value = abs(_number(quantity, "trade quantity"))
        total += quantity_value * abs(_number(entry_price, "entry fill price"))
        if exit_price is not None:
            total += quantity_value * abs(_number(exit_price, "exit fill price"))
    return total


def _fill_statistics(fills: Sequence[Any]) -> tuple[float | None, float | None, float | None]:
    """Return commission, slippage, and traded notional for every fill."""
    commission_total = 0.0
    slippage_total = 0.0
    notional_total = 0.0
    for fill in fills:
        commission = _get(fill, "commission")
        slippage = _get(fill, "slippage")
        notional = _get(fill, "notional")
        if notional is None:
            quantity = _get(fill, "quantity")
            fill_price = _get(fill, "fill_price")
            if quantity is None or fill_price is None:
                notional_total = -1.0
            elif notional_total >= 0:
                notional_total += abs(_number(quantity, "fill quantity")) * abs(
                    _number(fill_price, "fill price")
                )
        elif notional_total >= 0:
            notional_total += abs(_number(notional, "fill notional"))

        if commission is None:
            commission_total = -1.0
        elif commission_total >= 0:
            value = _number(commission, "fill commission")
            if value < 0:
                raise ValueError("fill commission cannot be negative")
            commission_total += value

        if slippage is None:
            slippage_total = -1.0
        elif slippage_total >= 0:
            value = _number(slippage, "fill slippage")
            if value < 0:
                raise ValueError("fill slippage cannot be negative")
            slippage_total += value

    return (
        None if commission_total < 0 else commission_total,
        None if slippage_total < 0 else slippage_total,
        None if notional_total < 0 else notional_total,
    )


def _curve_exposure(curve: Sequence[Any], explicit: Sequence[float] | None) -> float | None:
    if explicit is not None:
        if len(explicit) != len(curve):
            raise ValueError("exposure_values must have one value per equity point")
        values = [_number(value, "exposure") for value in explicit]
    else:
        values = []
        for item in curve:
            value = _get(item, "gross_exposure")
            if value is None:
                return None
            values.append(_number(value, "gross_exposure"))
    if not values:
        return None
    if any(value < 0 for value in values):
        raise ValueError("gross exposure cannot be negative")
    return mean(values)


def _aligned_values(
    strategy: Sequence[_EquityPoint], benchmark: Sequence[_EquityPoint]
) -> tuple[list[float], list[float]]:
    if not strategy or not benchmark:
        return [], []
    if strategy[0].point_date is not None and benchmark[0].point_date is not None:
        benchmark_by_date = {point.point_date: point.value for point in benchmark}
        pairs = [(point.value, benchmark_by_date[point.point_date]) for point in strategy if point.point_date in benchmark_by_date]
    elif strategy[0].point_date is None and benchmark[0].point_date is None:
        pairs = [(strategy[index].value, benchmark[index].value) for index in range(min(len(strategy), len(benchmark)))]
    else:
        raise ValueError("strategy and benchmark curves must use the same date convention")
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def _embedded_benchmark_curve(equity_curve: Sequence[Any]) -> list[Any] | None:
    """Extract benchmark equity carried by portfolio snapshots, when complete."""
    if not equity_curve:
        return None
    embedded: list[Any] = []
    for item in equity_curve:
        benchmark_equity = _get(item, "benchmark_equity")
        if benchmark_equity is None:
            return None
        point_date = _get(item, "snapshot_date", "date", "trade_date")
        embedded.append((point_date, benchmark_equity) if point_date is not None else benchmark_equity)
    return embedded


def calculate_performance_metrics(
    equity_curve: Sequence[Any],
    trades: Sequence[Any] = (),
    *,
    fills: Sequence[Any] = (),
    benchmark_curve: Sequence[Any] | None = None,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
    exposure_values: Sequence[float] | None = None,
    turnover_notional: float | None = None,
) -> PerformanceMetrics:
    """Calculate the complete tested metric set for one portfolio run."""
    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive")
    if risk_free_rate <= -1.0:
        raise ValueError("risk_free_rate must be greater than -1")
    points = _normalize_equity_curve(equity_curve)
    if not points:
        raise ValueError("equity_curve must contain at least one point")

    values = [point.value for point in points]
    starting_equity = values[0]
    ending_equity = values[-1]
    daily_returns, unavailable_return = _daily_returns(values)
    maximum_drawdown, drawdown_duration = calculate_drawdown(values)

    if len(points) >= 2 and points[0].point_date is not None and points[-1].point_date is not None:
        assert points[0].point_date is not None and points[-1].point_date is not None
        years = (points[-1].point_date - points[0].point_date).days / 365.2425
    else:
        years = (len(points) - 1) / annualization_factor

    total_return = calculate_total_return(starting_equity, ending_equity)
    cagr = calculate_cagr(starting_equity, ending_equity, years)
    volatility = None if unavailable_return else calculate_annualized_volatility(daily_returns, annualization_factor)
    sharpe = None if unavailable_return else calculate_sharpe_ratio(daily_returns, risk_free_rate, annualization_factor)
    sortino = None if unavailable_return else calculate_sortino_ratio(daily_returns, risk_free_rate, annualization_factor)
    calmar = calculate_calmar_ratio(cagr, maximum_drawdown)
    trade_stats = _trade_statistics(trades)
    trade_limitations = trade_stats.pop("limitations")

    fill_notional: float | None = None
    if fills:
        fill_commission, fill_slippage, fill_notional = _fill_statistics(fills)
        trade_stats["commission_cost"] = fill_commission
        trade_stats["slippage_cost"] = fill_slippage
        if fill_commission is None:
            trade_limitations.append("Commission cost is unavailable because one or more fills lack cost data")
        if fill_slippage is None:
            trade_limitations.append("Slippage cost is unavailable because one or more fills lack cost data")

    notional = turnover_notional
    if notional is None:
        notional = fill_notional if fills else _turnover_notional(trades)
    elif notional < 0:
        raise ValueError("turnover_notional cannot be negative")
    average_equity = mean(values)
    turnover = None if notional is None or average_equity <= 0 else notional / average_equity

    effective_benchmark_curve = benchmark_curve if benchmark_curve is not None else _embedded_benchmark_curve(equity_curve)
    benchmark_total_return: float | None = None
    benchmark_drawdown: float | None = None
    return_vs_benchmark: float | None = None
    drawdown_vs_benchmark: float | None = None
    benchmark_cagr=benchmark_volatility=benchmark_sharpe=benchmark_sortino=benchmark_calmar=None
    tracking_error=information_ratio=upside_capture=downside_capture=positive_capture=beta=correlation=None
    if effective_benchmark_curve is not None:
        benchmark_points = _normalize_equity_curve(effective_benchmark_curve)
        aligned_strategy, aligned_benchmark = _aligned_values(points, benchmark_points)
        if aligned_strategy and aligned_benchmark:
            benchmark_total_return = calculate_total_return(aligned_benchmark[0], aligned_benchmark[-1])
            benchmark_drawdown = calculate_maximum_drawdown(aligned_benchmark)
            aligned_strategy_return = calculate_total_return(aligned_strategy[0], aligned_strategy[-1])
            aligned_strategy_drawdown = calculate_maximum_drawdown(aligned_strategy)
            if aligned_strategy_return is not None and benchmark_total_return is not None:
                return_vs_benchmark = aligned_strategy_return - benchmark_total_return
            if aligned_strategy_drawdown is not None and benchmark_drawdown is not None:
                drawdown_vs_benchmark = aligned_strategy_drawdown - benchmark_drawdown
            benchmark_cagr=calculate_cagr(aligned_benchmark[0],aligned_benchmark[-1],years)
            strategy_daily,_=_daily_returns(aligned_strategy); benchmark_daily,_=_daily_returns(aligned_benchmark)
            benchmark_volatility=calculate_annualized_volatility(benchmark_daily,annualization_factor)
            benchmark_sharpe=calculate_sharpe_ratio(benchmark_daily,risk_free_rate,annualization_factor)
            benchmark_sortino=calculate_sortino_ratio(benchmark_daily,risk_free_rate,annualization_factor)
            benchmark_calmar=calculate_calmar_ratio(benchmark_cagr,benchmark_drawdown)
            if len(strategy_daily)==len(benchmark_daily) and len(strategy_daily)>=2:
                active=[s-b for s,b in zip(strategy_daily,benchmark_daily)]
                active_std=stdev(active); tracking_error=active_std*sqrt(annualization_factor)
                information_ratio=(mean(active)/active_std*sqrt(annualization_factor)) if active_std else None
                up=[i for i,b in enumerate(benchmark_daily) if b>0]; down=[i for i,b in enumerate(benchmark_daily) if b<0]
                up_den=sum(benchmark_daily[i] for i in up); down_den=sum(benchmark_daily[i] for i in down)
                upside_capture=sum(strategy_daily[i] for i in up)/up_den if up and up_den else None
                downside_capture=sum(strategy_daily[i] for i in down)/down_den if down and down_den else None
                positive_capture=sum(strategy_daily[i]>0 for i in up)/len(up) if up else None
                bmean=mean(benchmark_daily); smean=mean(strategy_daily)
                covariance=sum((s-smean)*(b-bmean) for s,b in zip(strategy_daily,benchmark_daily))/(len(strategy_daily)-1)
                bvar=sum((b-bmean)**2 for b in benchmark_daily)/(len(benchmark_daily)-1)
                beta=covariance/bvar if bvar else None
                sdev=stdev(strategy_daily); bdev=stdev(benchmark_daily)
                correlation=covariance/(sdev*bdev) if sdev and bdev else None

    limitations: list[str] = []
    limitations.extend(trade_limitations)
    if unavailable_return:
        limitations.append("Returns following zero equity are unavailable")
    if volatility is None:
        limitations.append("Annualized volatility requires at least two defined returns")
    if sharpe is None:
        limitations.append("Sharpe ratio is unavailable without nonzero return dispersion")
    if sortino is None:
        limitations.append("Sortino ratio is unavailable without downside dispersion")
    if turnover is None:
        limitations.append("Turnover is unavailable without fill notionals and positive average equity")
    exposure = _curve_exposure(equity_curve, exposure_values)
    if exposure is None:
        limitations.append("Exposure is unavailable because the curve has no exposure values")
    if effective_benchmark_curve is None:
        limitations.append("Benchmark comparison was not supplied")
    elif benchmark_total_return is None:
        limitations.append("Benchmark comparison requires at least one aligned point with positive starting equity")

    return PerformanceMetrics(
        starting_equity=starting_equity,
        ending_equity=ending_equity,
        net_profit=ending_equity - starting_equity,
        total_return=total_return,
        cagr=cagr,
        annualized_volatility=volatility,
        maximum_drawdown=maximum_drawdown,
        drawdown_duration=drawdown_duration,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        exposure=exposure,
        turnover=turnover,
        monthly_returns=_period_returns(points, yearly=False),
        annual_returns=_period_returns(points, yearly=True),
        benchmark_total_return=benchmark_total_return,
        benchmark_maximum_drawdown=benchmark_drawdown,
        return_vs_benchmark=return_vs_benchmark,
        drawdown_vs_benchmark=drawdown_vs_benchmark,
        benchmark_cagr=benchmark_cagr,benchmark_annualized_volatility=benchmark_volatility,
        benchmark_sharpe_ratio=benchmark_sharpe,benchmark_sortino_ratio=benchmark_sortino,
        benchmark_calmar_ratio=benchmark_calmar,active_return=return_vs_benchmark,
        tracking_error=tracking_error,information_ratio=information_ratio,upside_capture=upside_capture,
        downside_capture=downside_capture,positive_benchmark_sessions_captured=positive_capture,
        beta_to_benchmark=beta,correlation_to_benchmark=correlation,
        limitations=limitations,
        **trade_stats,
    )


# Short aliases are useful in notebooks and keep compatibility with common
# metric naming conventions.
total_return = calculate_total_return
cagr = calculate_cagr
annualized_volatility = calculate_annualized_volatility
maximum_drawdown = calculate_maximum_drawdown
sharpe_ratio = calculate_sharpe_ratio
sortino_ratio = calculate_sortino_ratio
calmar_ratio = calculate_calmar_ratio
