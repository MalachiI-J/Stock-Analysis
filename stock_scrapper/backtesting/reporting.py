"""Self-contained reporting for persisted historical backtests."""

from __future__ import annotations

import csv
import html
import io
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


RUN_FIELDS: tuple[str, ...] = (
    "run_id",
    "strategy_name",
    "strategy_version",
    "status",
    "started_at",
    "completed_at",
    "start_date",
    "end_date",
    "warmup_start_date",
    "benchmark_symbol",
    "initial_cash",
    "ending_equity",
    "symbols",
    "configuration_hash",
    "data_hash",
    "deterministic_result_hash",
    "error_summary",
)

SIGNAL_FIELDS: tuple[str, ...] = (
    "run_id",
    "signal_id",
    "symbol",
    "signal_date",
    "action",
    "classification",
    "opportunity_score",
    "risk_score",
    "confidence_score",
    "market_regime",
    "ranking_json",
    "reason",
    "accepted",
    "rejection_reason",
)

TRADE_FIELDS: tuple[str, ...] = (
    "run_id",
    "trade_id",
    "symbol",
    "signal_date",
    "entry_date",
    "exit_signal_date",
    "exit_date",
    "quantity",
    "entry_reference_price",
    "entry_fill_price",
    "exit_reference_price",
    "exit_fill_price",
    "entry_commission",
    "exit_commission",
    "slippage_cost",
    "realized_pnl",
    "return_pct",
    "holding_days",
    "entry_reason",
    "exit_reason",
    "classification",
    "market_regime",
    "opportunity_score",
    "risk_score",
    "confidence_score",
    "ranking_json",
    "ambiguous_daily_bar",
    "strategy_version",
    "configuration_hash",
)

ORDER_FILL_FIELDS: tuple[str, ...] = (
    "record_type",
    "run_id",
    "order_id",
    "fill_id",
    "signal_id",
    "symbol",
    "side",
    "signal_date",
    "scheduled_date",
    "fill_date",
    "status",
    "quantity",
    "reference_price",
    "fill_price",
    "commission",
    "slippage",
    "reason",
)

EQUITY_FIELDS: tuple[str, ...] = (
    "run_id",
    "trade_date",
    "cash",
    "reserved_cash",
    "market_value",
    "unrealized_pnl",
    "realized_pnl",
    "equity",
    "gross_exposure",
    "position_count",
    "daily_return",
    "benchmark_equity",
)


def _field_union(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    fields: list[str]=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(str(key))
    return tuple(fields or ("status",))


def write_backtest_reports(output_dir: str | Path, saved_run: Mapping[str, Any]) -> dict[str, Path]:
    """Render reports from a previously persisted backtest without rerunning it.

    The expected mapping is the result of
    :func:`stock_scrapper.backtesting.persistence.load_backtest`. Optional
    ``rejected_candidates`` rows are also accepted; otherwise rejected rows are
    derived from the persisted signals table.
    """
    payload = _mapping(saved_run)
    run_id = str(payload.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("saved_run must contain a non-empty run_id")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = f"backtest_{_filename_token(run_id)}"

    config = _json_mapping(payload.get("configuration_snapshot_json") or payload.get("configuration_snapshot"))
    symbols = _json_list(payload.get("symbols_json") or payload.get("symbols"))
    metrics = _json_mapping(payload.get("metrics"))
    orders = _sorted_records(payload.get("orders"), ("signal_date", "symbol", "order_id"))
    fills = _sorted_records(payload.get("fills"), ("fill_date", "symbol", "fill_id"))
    trades = _sorted_records(payload.get("trades"), ("entry_date", "symbol", "trade_id"))
    equity_curve = _sorted_records(payload.get("equity_curve"), ("trade_date",))
    signals, rejected = _prepare_signals(payload, orders)
    order_fill_rows = [dict(item, record_type="order") for item in orders] + [
        dict(item, record_type="fill") for item in fills
    ]
    order_fill_rows.sort(
        key=lambda item: (
            str(item.get("signal_date") or item.get("fill_date") or ""),
            str(item.get("symbol") or ""),
            str(item.get("record_type") or ""),
            str(item.get("order_id") or item.get("fill_id") or ""),
        )
    )

    monthly_returns = _period_return_rows(metrics.get("monthly_returns"), "month")
    annual_returns = _period_return_rows(metrics.get("annual_returns"), "year")
    summary_row = _summary_row(payload, symbols, config, metrics)

    paths = {
        "html": output_path / f"{stem}.html",
        "summary": output_path / f"{stem}_summary.csv",
        "trades": output_path / f"{stem}_trades.csv",
        "signals": output_path / f"{stem}_signals.csv",
        "rejected_candidates": output_path / f"{stem}_rejected_candidates.csv",
        "orders_and_fills": output_path / f"{stem}_orders_and_fills.csv",
        "equity_curve": output_path / f"{stem}_equity_curve.csv",
        "monthly_returns": output_path / f"{stem}_monthly_returns.csv",
        "annual_returns": output_path / f"{stem}_annual_returns.csv",
        "configuration": output_path / f"{stem}_configuration.json",
        "provenance": output_path / f"{stem}_provenance.csv",
        "data_health": output_path / f"{stem}_data_health.csv",
        "benchmark_metrics": output_path / f"{stem}_benchmark_metrics.csv",
        "symbol_attribution": output_path / f"{stem}_symbol_attribution.csv",
        "signal_outcomes": output_path / f"{stem}_signal_outcomes.csv",
        "exit_diagnostics": output_path / f"{stem}_exit_diagnostics.csv",
        "daily_diagnostics": output_path / f"{stem}_daily_diagnostics.csv",
    }

    _write_csv(paths["summary"], [summary_row], RUN_FIELDS)
    _write_csv(paths["trades"], trades, TRADE_FIELDS)
    _write_csv(paths["signals"], signals, SIGNAL_FIELDS)
    _write_csv(paths["rejected_candidates"], rejected, SIGNAL_FIELDS)
    _write_csv(paths["orders_and_fills"], order_fill_rows, ORDER_FILL_FIELDS)
    _write_csv(paths["equity_curve"], equity_curve, EQUITY_FIELDS)
    _write_csv(paths["monthly_returns"], monthly_returns, ("month", "return"))
    _write_csv(paths["annual_returns"], annual_returns, ("year", "return"))
    _atomic_write_text(paths["configuration"],json.dumps(config,sort_keys=True,indent=2,ensure_ascii=False))
    provenance={key:payload.get(key) for key in ("application_version","strategy_name","strategy_version","scoring_version","schema_version","git_commit_hash","git_dirty","source_fingerprint","python_version","platform_info","configuration_hash","data_hash","deterministic_result_hash")}
    _write_csv(paths["provenance"],[provenance],tuple(provenance))
    health=_json_mapping(payload.get("data_health_snapshot_json")); _write_csv(paths["data_health"],health.get("symbols",[]),_field_union(health.get("symbols",[])))
    benchmark_rows=[{"metric":k,"value":v} for k,v in metrics.items() if "benchmark" in k or k in {"active_return","tracking_error","information_ratio","upside_capture","downside_capture","beta_to_benchmark","correlation_to_benchmark"}]
    _write_csv(paths["benchmark_metrics"],benchmark_rows,("metric","value"))
    for key in ("symbol_attribution","signal_outcomes","exit_diagnostics","daily_diagnostics"):
        values=[_mapping(v) for v in payload.get(key,[])]; _write_csv(paths[key],values,_field_union(values))

    html_content = _render_html(
        payload,
        symbols,
        config,
        metrics,
        signals,
        rejected,
        trades,
        equity_curve,
        monthly_returns,
        annual_returns,
    )
    _atomic_write_text(paths["html"], html_content)
    return paths


def _prepare_signals(
    payload: Mapping[str, Any],
    orders: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted_signal_ids = {str(order.get("signal_id")) for order in orders if order.get("signal_id") is not None}
    signals = [_mapping(item) for item in payload.get("signals") or []]
    for signal in signals:
        if signal.get("accepted") is None:
            signal["accepted"] = str(signal.get("signal_id")) in accepted_signal_ids

    existing_ids = {str(signal.get("signal_id")) for signal in signals if signal.get("signal_id") is not None}
    for raw_rejection in payload.get("rejected_candidates") or []:
        rejection = _mapping(raw_rejection)
        signal_id = str(rejection.get("signal_id") or "")
        rejection["accepted"] = False
        rejection.setdefault("rejection_reason", rejection.get("reason"))
        if signal_id and signal_id in existing_ids:
            continue
        signals.append(rejection)
        if signal_id:
            existing_ids.add(signal_id)

    signals.sort(key=lambda item: (str(item.get("signal_date") or ""), str(item.get("symbol") or ""), str(item.get("signal_id") or "")))
    rejected = [signal for signal in signals if not _truthy(signal.get("accepted"))]
    return signals, rejected


def _summary_row(
    payload: Mapping[str, Any],
    symbols: list[Any],
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    row = {field: payload.get(field) for field in RUN_FIELDS}
    row["symbols"] = _canonical_json(symbols)
    row["warmup_start_date"] = payload.get("warmup_start_date") or payload.get("warm_up_start_date")
    row["data_hash"] = payload.get("data_hash") or payload.get("price_data_hash")
    for name, value in sorted(metrics.items()):
        if name not in {"monthly_returns", "annual_returns"}:
            row[name] = value
    return row


def _period_return_rows(value: Any, period_name: str) -> list[dict[str, Any]]:
    values = _json_mapping(value)
    return [{period_name: period, "return": values[period]} for period in sorted(values)]


def _render_html(
    run: Mapping[str, Any],
    symbols: list[Any],
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    signals: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    equity_curve: list[dict[str, Any]],
    monthly_returns: list[dict[str, Any]],
    annual_returns: list[dict[str, Any]],
) -> str:
    run_id = str(run.get("run_id"))
    benchmark = run.get("benchmark_symbol") or config.get("benchmark") or "SPY"
    excluded_symbols = _json_list(run.get("excluded_symbols") or config.get("excluded_symbols"))
    assumptions = {
        "run.configuration_hash": run.get("configuration_hash"),
        "run.data_hash": run.get("data_hash") or run.get("price_data_hash"),
        "run.deterministic_result_hash": run.get("deterministic_result_hash"),
        **_flatten_mapping(config),
    }
    execution_keys = (
        "execution_timing",
        "signal_frequency",
        "rebalancing_frequency",
        "position_sizing",
        "fractional_shares",
        "maximum_positions",
        "maximum_position_weight",
        "cash_reserve",
        "stop_loss",
        "trailing_stop",
        "profit_target",
        "maximum_holding_period",
        "daily_bar_ambiguity_policy",
        "final_liquidation",
    )
    execution = {key: config.get(key) for key in execution_keys if key in config}
    costs = {
        "Commission basis points": config.get("commission_basis_points"),
        "Minimum commission": config.get("minimum_commission"),
        "Slippage basis points": config.get("slippage_basis_points"),
        "Recorded commission cost": metrics.get("commission_cost"),
        "Recorded slippage cost": metrics.get("slippage_cost"),
    }

    performance_by_symbol = _aggregate_trade_performance(trades, "symbol")
    performance_by_regime = _aggregate_trade_performance(trades, "market_regime")
    equity_svg = _equity_chart(equity_curve, benchmark)
    drawdown_svg = _drawdown_chart(equity_curve, benchmark)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stock Scrapper Backtest Report — {_escape(run_id)}</title>
  <style>
    :root {{ color-scheme:light; --ink:#172033; --muted:#5f6b7a; --line:#d6dce5; --panel:#f7f9fc; }}
    body {{ max-width:1250px; margin:0 auto; padding:24px; color:var(--ink); font:15px/1.48 Arial,sans-serif; }}
    h1,h2,h3 {{ line-height:1.2; }} h2 {{ margin-top:34px; padding-bottom:6px; border-bottom:2px solid var(--line); }}
    table {{ width:100%; border-collapse:collapse; margin:12px 0 24px; }} th,td {{ border:1px solid var(--line); padding:7px 9px; text-align:left; vertical-align:top; }} th {{ background:#eef2f7; }}
    .notice {{ padding:12px 14px; margin:14px 0; border-left:5px solid #b45309; background:#fff7ed; }}
    .warning {{ padding:12px 14px; margin:14px 0; border-left:5px solid #b91c1c; background:#fef2f2; }}
    .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(175px,1fr)); gap:10px; }} .metric {{ padding:10px; background:var(--panel); border:1px solid var(--line); border-radius:6px; }} .metric strong {{ display:block; font-size:1.2rem; }}
    .chart-wrap {{ overflow-x:auto; }} .chart {{ width:100%; min-width:680px; height:auto; background:white; }} .muted {{ color:var(--muted); }} code {{ overflow-wrap:anywhere; }}
  </style>
</head>
<body>
  <h1>Stock Scrapper Historical Backtest Report</h1>
  <div class="notice"><strong>Run:</strong> {_escape(run_id)} &nbsp; <strong>Status:</strong> {_display(run.get('status'))} &nbsp; <strong>Strategy:</strong> {_display(run.get('strategy_name'))} {_display(run.get('strategy_version'))}</div>

  <h2>Strategy Assumptions</h2>
  {_key_value_table(assumptions, 'No configuration snapshot was persisted.')}
  <h2>Date Range</h2>
  {_key_value_table({'Evaluation start': run.get('start_date'), 'Evaluation end': run.get('end_date'), 'Completed at': run.get('completed_at')}, 'No date range was recorded.')}
  <h2>Warm-Up Range</h2>
  {_key_value_table({'Warm-up start': run.get('warmup_start_date') or run.get('warm_up_start_date'), 'Signals begin': run.get('start_date'), 'Configured warm-up sessions': config.get('warm_up_days')}, 'No warm-up range was recorded.')}
  <h2>Candidate Universe</h2>
  {_render_list(symbols, 'No candidate universe was persisted.')}
  <h2>Excluded Symbols</h2>
  {_render_list(excluded_symbols, 'No excluded symbols were recorded by this persisted run.')}
  <h2>Execution Assumptions</h2>
  {_key_value_table(execution, 'No execution assumptions were persisted.')}
  <h2>Commission and Slippage</h2>
  {_key_value_table(costs, 'No cost assumptions were persisted.')}

  <h2>Performance Summary</h2>
  {_metrics_grid(metrics)}
  <h2>SPY Comparison</h2>
  <p>Persisted benchmark: <strong>{_escape(benchmark)}</strong></p>
  {_key_value_table({'Strategy total return': _percent_text(metrics.get('total_return')), f'{benchmark} total return': _percent_text(metrics.get('benchmark_total_return')), 'Return versus benchmark': _percent_text(_first_defined(metrics.get('return_vs_benchmark'), metrics.get('return_vs_spy'))), 'Strategy maximum drawdown': _percent_text(metrics.get('maximum_drawdown')), f'{benchmark} maximum drawdown': _percent_text(metrics.get('benchmark_maximum_drawdown')), 'Drawdown versus benchmark': _percent_text(_first_defined(metrics.get('drawdown_vs_benchmark'), metrics.get('drawdown_vs_spy'))), 'Cash total return': _percent_text(metrics.get('cash_total_return'))}, 'No benchmark comparison was persisted.')}

  <h2>Equity Curve</h2>{equity_svg}
  <h2>Drawdown Chart</h2>{drawdown_svg}
  <h2>Monthly Returns</h2>{_returns_table(monthly_returns, 'month')}
  <h2>Annual Returns</h2>{_returns_table(annual_returns, 'year')}
  <h2>Complete Trade Log</h2>{_trade_table(trades)}
  <h2>Rejected Signals</h2>{_signal_table(rejected)}
  <h2>Performance by Symbol</h2>{_aggregate_table(performance_by_symbol, 'Symbol')}
  <h2>Performance by Market Regime</h2>{_aggregate_table(performance_by_regime, 'Market regime')}

  <h2>Data-Health Summary</h2>{_key_value_table(_json_mapping(run.get('data_health_snapshot_json')), 'No data-health snapshot was persisted.')}
  <h2>Requested and Effective Dates</h2>{_key_value_table({'Requested start':run.get('requested_start_date'),'Effective start':run.get('effective_start_date'),'Requested end':run.get('requested_end_date'),'Effective end':run.get('effective_end_date')},'No date evidence persisted.')}
  <h2>Warm-Up Sufficiency</h2>{_key_value_table({'Policy':run.get('warmup_policy'),'Required sessions':run.get('required_warmup_sessions'),'Available sessions':run.get('available_warmup_sessions'),'Benchmark sufficient':run.get('benchmark_sufficient'),'Warning':run.get('warmup_warning')},'No warm-up evidence persisted.')}
  <h2>Software Provenance</h2>{_key_value_table({k:run.get(k) for k in ('application_version','strategy_version','scoring_version','schema_version','git_commit_hash','git_dirty','source_fingerprint','python_version','platform_info')},'No provenance persisted.')}
  <h2>Benchmark Risk-Adjusted Comparison</h2>{_key_value_table({k:metrics.get(k) for k in ('benchmark_cagr','benchmark_annualized_volatility','benchmark_sharpe_ratio','benchmark_sortino_ratio','benchmark_calmar_ratio','active_return','tracking_error','information_ratio','upside_capture','downside_capture','beta_to_benchmark','correlation_to_benchmark')},'No benchmark diagnostics persisted.')}
  <h2>Symbol Contribution and Profit Concentration</h2>{_key_value_table({'Attribution rows':len(run.get('symbol_attribution') or []),'Signal outcome rows':len(run.get('signal_outcomes') or []),'Daily diagnostic rows':len(run.get('daily_diagnostics') or [])},'No diagnostics persisted.')}
  <h2>Opportunity-Cost, Signal, and Exit Diagnostics</h2><p>These are post-simulation counterfactual research diagnostics and never alter the historical decisions in this run.</p>

  <h2>Limitations and Warnings</h2>
  <div class="warning"><strong>Survivorship-bias warning:</strong> The candidate universe may omit securities that delisted, merged, or otherwise left the available dataset. Results can therefore overstate historical robustness.</div>
  <div class="warning"><strong>Static-watchlist warning:</strong> This simulation uses a fixed supplied universe rather than the investable universe known on each historical date.</div>
  {_render_list(metrics.get('limitations'), 'No additional metric limitations were recorded.')}
  <h2>Educational Disclaimer</h2>
  <p>This report is educational research, not personalized financial advice or a recommendation to trade. Historical and simulated performance does not guarantee future results. Free market data can be delayed, revised, incomplete, or inconsistent. Commission, slippage, daily-bar ambiguity, corporate-action handling, and execution assumptions can materially affect results.</p>
</body>
</html>
"""


def _equity_chart(curve: list[dict[str, Any]], benchmark: Any) -> str:
    dates, equity, benchmark_equity = _curve_series(curve)
    return _line_chart(
        "Equity curve",
        dates,
        {"portfolio-equity": equity, "benchmark-equity": benchmark_equity},
        {"portfolio-equity": "Portfolio", "benchmark-equity": str(benchmark)},
        {"portfolio-equity": "#1d4ed8", "benchmark-equity": "#64748b"},
        percent_axis=False,
    )


def _drawdown_chart(curve: list[dict[str, Any]], benchmark: Any) -> str:
    dates, equity, benchmark_equity = _curve_series(curve)
    return _line_chart(
        "Drawdown chart",
        dates,
        {
            "portfolio-drawdown": _drawdown_series(equity),
            "benchmark-drawdown": _drawdown_series(benchmark_equity),
        },
        {"portfolio-drawdown": "Portfolio drawdown", "benchmark-drawdown": f"{benchmark} drawdown"},
        {"portfolio-drawdown": "#b91c1c", "benchmark-drawdown": "#64748b"},
        percent_axis=True,
    )


def _curve_series(curve: list[dict[str, Any]]) -> tuple[list[str], list[float | None], list[float | None]]:
    by_date: dict[str, tuple[float | None, float | None]] = {}
    for item in curve:
        trade_date = str(item.get("trade_date") or item.get("snapshot_date") or "")
        if trade_date:
            by_date[trade_date] = (_number(item.get("equity")), _number(item.get("benchmark_equity")))
    dates = sorted(by_date)
    return dates, [by_date[item][0] for item in dates], [by_date[item][1] for item in dates]


def _drawdown_series(values: list[float | None]) -> list[float | None]:
    result: list[float | None] = []
    peak: float | None = None
    for value in values:
        if value is None or value <= 0:
            result.append(None)
            continue
        peak = value if peak is None else max(peak, value)
        result.append(value / peak - 1.0)
    return result


def _line_chart(
    title: str,
    dates: list[str],
    series: Mapping[str, list[float | None]],
    labels: Mapping[str, str],
    colors: Mapping[str, str],
    percent_axis: bool,
) -> str:
    available = [value for values in series.values() for value in values if value is not None]
    if not dates or not available:
        return '<p class="muted">No persisted curve data was available.</p>'
    width, height = 920.0, 330.0
    left, right, top, bottom = 65.0, 18.0, 24.0, 44.0
    plot_width, plot_height = width - left - right, height - top - bottom
    minimum, maximum = min(available), max(available)
    padding = (maximum - minimum) * 0.05 if maximum != minimum else max(abs(maximum) * 0.05, 0.01)
    minimum -= padding
    maximum += padding

    def x(index: int) -> float:
        return left if len(dates) == 1 else left + index * plot_width / (len(dates) - 1)

    def y(value: float) -> float:
        return top + (maximum - value) * plot_height / (maximum - minimum)

    grid = []
    for tick in range(5):
        fraction = tick / 4
        y_value = top + fraction * plot_height
        value = maximum - fraction * (maximum - minimum)
        label = f"{value:.1%}" if percent_axis else f"{value:,.2f}"
        grid.append(f'<line x1="{left}" y1="{y_value:.2f}" x2="{width-right}" y2="{y_value:.2f}" stroke="#e5e7eb"/><text x="{left-8}" y="{y_value+4:.2f}" text-anchor="end" font-size="11" fill="#4b5563">{_escape(label)}</text>')
    paths = []
    legend = []
    for legend_index, (name, values) in enumerate(series.items()):
        for segment in _segments(values):
            points = " ".join(f"{x(index):.2f},{y(value):.2f}" for index, value in segment)
            if len(segment) == 1:
                index, value = segment[0]
                paths.append(f'<circle data-series="{_escape(name)}" cx="{x(index):.2f}" cy="{y(value):.2f}" r="2" fill="{colors[name]}"/>')
            else:
                paths.append(f'<polyline data-series="{_escape(name)}" points="{points}" fill="none" stroke="{colors[name]}" stroke-width="2.2"/>')
        legend_x = left + legend_index * 190
        legend.append(f'<line x1="{legend_x}" y1="{height-11}" x2="{legend_x+24}" y2="{height-11}" stroke="{colors[name]}" stroke-width="3"/><text x="{legend_x+30}" y="{height-7}" font-size="12">{_escape(labels[name])}</text>')
    chart_id = _filename_token(title)
    return f'''<div class="chart-wrap"><svg class="chart" viewBox="0 0 {int(width)} {int(height)}" role="img" aria-labelledby="{chart_id}-title"><title id="{chart_id}-title">{_escape(title)}</title><rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#fff" stroke="#d1d5db"/>{''.join(grid)}{''.join(paths)}<text x="{left}" y="{height-bottom+18}" font-size="11">{_escape(dates[0])}</text><text x="{width-right}" y="{height-bottom+18}" text-anchor="end" font-size="11">{_escape(dates[-1])}</text>{''.join(legend)}</svg></div>'''


def _segments(values: list[float | None]) -> list[list[tuple[int, float]]]:
    result: list[list[tuple[int, float]]] = []
    current: list[tuple[int, float]] = []
    for index, value in enumerate(values):
        if value is None:
            if current:
                result.append(current)
                current = []
        else:
            current.append((index, value))
    if current:
        result.append(current)
    return result


def _aggregate_trade_performance(trades: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        grouped[str(trade.get(field) or "Unavailable")].append(trade)
    rows = []
    for group in sorted(grouped):
        items = grouped[group]
        pnl = [_number(item.get("realized_pnl")) for item in items]
        returns = [_number(item.get("return_pct")) for item in items]
        defined_pnl = [value for value in pnl if value is not None]
        defined_returns = [value for value in returns if value is not None]
        wins = sum(1 for value in defined_pnl if value > 0)
        rows.append(
            {
                "group": group,
                "trades": len(items),
                "realized_pnl": sum(defined_pnl),
                "average_return": sum(defined_returns) / len(defined_returns) if defined_returns else None,
                "win_rate": wins / len(defined_pnl) if defined_pnl else None,
            }
        )
    return rows


def _metrics_grid(metrics: Mapping[str, Any]) -> str:
    excluded = {"monthly_returns", "annual_returns", "limitations"}
    items = [(name, value) for name, value in sorted(metrics.items()) if name not in excluded]
    if not items:
        return '<p class="muted">No performance metrics were persisted.</p>'
    return '<div class="metrics">' + "".join(
        f'<div class="metric">{_escape(_label(name))}<strong>{_metric_display(name, value)}</strong></div>'
        for name, value in items
    ) + "</div>"


def _returns_table(rows: list[dict[str, Any]], period_name: str) -> str:
    return _html_table(
        rows,
        ((period_name, _label(period_name)), ("return", "Return")),
        "No period returns were persisted.",
        percent_fields={"return"},
    )


def _trade_table(trades: list[dict[str, Any]]) -> str:
    columns = (
        ("trade_id", "Trade"),
        ("symbol", "Symbol"),
        ("signal_date", "Signal date"),
        ("entry_date", "Entry date"),
        ("exit_signal_date", "Exit signal date"),
        ("exit_date", "Exit date"),
        ("quantity", "Quantity"),
        ("entry_reference_price", "Entry reference"),
        ("entry_fill_price", "Entry fill"),
        ("exit_reference_price", "Exit reference"),
        ("exit_fill_price", "Exit fill"),
        ("entry_commission", "Entry commission"),
        ("exit_commission", "Exit commission"),
        ("slippage_cost", "Slippage cost"),
        ("realized_pnl", "Realized P&L"),
        ("return_pct", "Return"),
        ("holding_days", "Holding days"),
        ("entry_reason", "Entry reason"),
        ("exit_reason", "Exit reason"),
        ("classification", "Classification"),
        ("market_regime", "Regime"),
        ("opportunity_score", "Opportunity"),
        ("risk_score", "Risk"),
        ("confidence_score", "Confidence"),
        ("ranking_json", "Ranking values"),
        ("ambiguous_daily_bar", "Ambiguous bar"),
        ("strategy_version", "Strategy version"),
        ("configuration_hash", "Configuration hash"),
    )
    table = _html_table(
        trades,
        columns,
        "No completed trades were persisted.",
        percent_fields={"return_pct"},
    )
    return f'<div class="chart-wrap">{table}</div>' if trades else table


def _signal_table(signals: list[dict[str, Any]]) -> str:
    columns = (
        ("signal_id", "Signal"), ("symbol", "Symbol"), ("signal_date", "Date"),
        ("action", "Action"), ("classification", "Classification"),
        ("opportunity_score", "Opportunity"), ("risk_score", "Risk"),
        ("confidence_score", "Confidence"), ("market_regime", "Regime"),
        ("rejection_reason", "Rejection reason"),
    )
    return _html_table(signals, columns, "No rejected signals were persisted.")


def _aggregate_table(rows: list[dict[str, Any]], group_label: str) -> str:
    columns = (
        ("group", group_label), ("trades", "Trades"), ("realized_pnl", "Realized P&L"),
        ("average_return", "Average return"), ("win_rate", "Win rate"),
    )
    return _html_table(rows, columns, "No completed trades were available for aggregation.", percent_fields={"average_return", "win_rate"})


def _html_table(
    rows: list[dict[str, Any]],
    columns: Sequence[tuple[str, str]],
    empty_message: str,
    percent_fields: set[str] | None = None,
) -> str:
    if not rows:
        return f'<p class="muted">{_escape(empty_message)}</p>'
    percent_fields = percent_fields or set()
    header = "".join(f"<th>{_escape(label)}</th>" for _, label in columns)
    body = []
    for row in rows:
        cells = []
        for field, _ in columns:
            value = row.get(field)
            cells.append(f"<td>{_percent(value) if field in percent_fields else _display(value)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _key_value_table(values: Mapping[str, Any], empty_message: str) -> str:
    present = [(key, value) for key, value in values.items() if value is not None]
    if not present:
        return f'<p class="muted">{_escape(empty_message)}</p>'
    rows = "".join(f"<tr><th>{_escape(_label(str(key)))}</th><td>{_display_value(value)}</td></tr>" for key, value in present)
    return f"<table><tbody>{rows}</tbody></table>"


def _flatten_mapping(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in sorted(value.items()):
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            result.update(_flatten_mapping(item, name))
        else:
            result[name] = item
    return result


def _render_list(values: Any, empty_message: str) -> str:
    items = _json_list(values)
    if not items:
        return f'<p class="muted">{_escape(empty_message)}</p>'
    return "<ul>" + "".join(f"<li>{_display_value(item)}</li>" for item in items) + "</ul>"


def _metric_display(name: str, value: Any) -> str:
    percent_names = {
        "total_return", "cagr", "annualized_volatility", "maximum_drawdown", "exposure",
        "turnover", "win_rate", "benchmark_total_return", "benchmark_maximum_drawdown",
        "return_vs_benchmark", "return_vs_spy", "drawdown_vs_benchmark", "drawdown_vs_spy",
        "cash_total_return",
    }
    return _percent(value) if name in percent_names else _display(value)


def _display_value(value: Any) -> str:
    if isinstance(value, (Mapping, list, tuple, set)):
        return f"<code>{_escape(_canonical_json(value))}</code>"
    return _display(value)


def _percent(value: Any) -> str:
    number = _number(value)
    return '<span class="muted">Unavailable</span>' if number is None else f"{number:.2%}"


def _percent_text(value: Any) -> str | None:
    number = _number(value)
    return None if number is None else f"{number:.2%}"


def _display(value: Any) -> str:
    if value is None or value == "":
        return '<span class="muted">Unavailable</span>'
    if isinstance(value, float):
        return f"{value:,.4f}"
    return _escape(value)


def _label(value: str) -> str:
    return value.replace("_", " ").replace(".", " › ").strip().title()


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _write_csv(path: Path, rows: list[dict[str, Any]], preferred_fields: Sequence[str]) -> None:
    field_set = {str(key) for row in rows for key in row}
    fields = list(dict.fromkeys([*preferred_fields, *sorted(field_set - set(preferred_fields))]))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(row.get(field)) for field in fields})
    _atomic_write_text(path, buffer.getvalue())


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary_name = handle.name
            handle.write(content)
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def _csv_value(value: Any) -> Any:
    return _canonical_json(value) if isinstance(value, (Mapping, list, tuple, set)) else value


def _sorted_records(value: Any, sort_fields: Sequence[str]) -> list[dict[str, Any]]:
    records = [_mapping(item) for item in value or []]
    return sorted(records, key=lambda item: tuple(str(item.get(field) or "") for field in sort_fields))


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "keys"):
        return {key: value[key] for key in value.keys()}
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"Expected a mapping-like value, got {type(value).__name__}")


def _json_mapping(value: Any) -> dict[str, Any]:
    decoded = _decode_json(value)
    return dict(decoded) if isinstance(decoded, Mapping) else {}


def _json_list(value: Any) -> list[Any]:
    decoded = _decode_json(value)
    if decoded is None:
        return []
    if isinstance(decoded, list):
        return decoded
    if isinstance(decoded, (tuple, set)):
        return list(decoded)
    return [decoded]


def _decode_json(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "{")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "accepted"}
    return bool(value)


def _first_defined(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _filename_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return token or "backtest"
