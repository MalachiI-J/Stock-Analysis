"""Build self-contained offline Phase 2 research reports."""

from __future__ import annotations

import csv
import html
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


PHASE2_CSV_FIELDS: tuple[str, ...] = (
    "report_date",
    "analysis_run_id",
    "as_of_date",
    "data_through_date",
    "scoring_version",
    "configuration_hash",
    "benchmark_symbol",
    "market_regime",
    "market_regime_confidence",
    "market_regime_reasons",
    "symbol",
    "candidate_rank",
    "risk_rank",
    "classification",
    "eligible_for_scoring",
    "risk_score",
    "risk_level",
    "opportunity_score",
    "confidence_score",
    "trend_state",
    "primary_reason",
    "blocking_reasons",
    "risk_components",
    "opportunity_components",
    "confidence_components",
    "indicators",
    "flags",
    "positive_factors",
    "risk_factors",
    "confidence_limitations",
    "quality_concerns",
    "market_regime_effects",
    "improvement_conditions",
    "weakening_conditions",
    "previous_classification",
    "previous_risk_score",
    "previous_opportunity_score",
    "previous_confidence_score",
    "risk_score_change",
    "opportunity_score_change",
    "confidence_score_change",
    "classification_changed",
    "change_summary",
)

_LIST_FIELDS = (
    "blocking_reasons",
    "flags",
    "positive_factors",
    "risk_factors",
    "confidence_limitations",
    "quality_concerns",
    "market_regime_effects",
    "improvement_conditions",
    "weakening_conditions",
)

_COMPONENT_FIELDS = (
    "risk_components",
    "opportunity_components",
    "confidence_components",
    "indicators",
)

_HISTORY_PAYLOAD_KEYS = frozenset(
    {"history", "price_history", "historical_prices", "chart_history", "price_series", "date_series", "dates"}
)


def write_phase2_reports(
    output_dir: str | Path,
    report_date: str | date,
    run_metadata: Mapping[str, Any] | Any,
    results: Iterable[Mapping[str, Any] | Any] | Mapping[str, Any],
    histories: Mapping[str, Any],
    quality_issues: Iterable[Mapping[str, Any] | Any],
    previous_results: Iterable[Mapping[str, Any] | Any] | Mapping[str, Any] | None = None,
    report_identity: str | None = None,
) -> dict[str, Path]:
    """Write Phase 2 summary CSV and HTML reports.

    ``results`` may contain :class:`AnalysisResult` instances, mappings loaded
    from SQLite, or equivalent objects. Histories are chart-only inputs and are
    never serialized into the CSV. Chart data is defensively bounded by the
    report/as-of date even when a caller supplies later rows.
    """
    report_date_text = _normalize_date(report_date, field_name="report_date")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    normalized_results = _normalize_results(results)
    normalized_previous = {
        item["symbol"].upper(): item
        for item in _normalize_results(previous_results or [])
        if item.get("symbol")
    }
    normalized_histories = {str(symbol).upper(): _history_rows(rows) for symbol, rows in histories.items()}
    metadata_input = _object_to_dict(run_metadata)
    scope_date = _earliest_date_text(report_date_text, metadata_input.get("as_of_date")) or report_date_text
    normalized_issues = [
        normalized
        for issue in quality_issues
        if _issue_is_in_scope(normalized := _object_to_dict(issue), scope_date)
    ]
    metadata = _prepare_metadata(
        metadata_input,
        report_date_text,
        normalized_results,
        normalized_histories,
    )

    candidate_order = sorted(
        (
            item
            for item in normalized_results
            if item.get("classification") in {"Candidate", "Strong Candidate"}
        ),
        key=lambda item: (
            -_sortable_number(item.get("opportunity_score"), default=-math.inf),
            -_sortable_number(item.get("confidence_score"), default=-math.inf),
            _sortable_number(item.get("risk_score"), default=math.inf),
            str(item.get("symbol", "")),
        ),
    )
    risk_order = sorted(
        (item for item in normalized_results if _finite_number(item.get("risk_score")) is not None),
        key=lambda item: (
            -_sortable_number(item.get("risk_score"), default=-math.inf),
            -_sortable_number(item.get("opportunity_score"), default=-math.inf),
            str(item.get("symbol", "")),
        ),
    )
    candidate_ranks = {str(item["symbol"]).upper(): rank for rank, item in enumerate(candidate_order, 1)}
    risk_ranks = {str(item["symbol"]).upper(): rank for rank, item in enumerate(risk_order, 1)}

    entries: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for result in sorted(normalized_results, key=lambda item: str(item.get("symbol", ""))):
        symbol = str(result.get("symbol", "")).upper()
        cutoff = _earliest_date_text(
            report_date_text,
            metadata.get("as_of_date"),
            result.get("as_of_date"),
        )
        result["data_through_date"] = _bounded_date_text(
            result.get("data_through_date"),
            cutoff or report_date_text,
        )
        previous = normalized_previous.get(symbol)
        symbol_issues = _issues_for_symbol(normalized_issues, symbol)
        quality_concerns = _deduplicate(
            _as_list(result.get("quality_concerns"))
            + [str(issue.get("description") or issue.get("issue_type") or "Data-quality issue") for issue in symbol_issues]
        )
        change = _analysis_change(result, previous)
        entry = {
            "result": result,
            "previous": previous,
            "history": normalized_histories.get(symbol, []),
            "quality_issues": symbol_issues,
            "quality_concerns": quality_concerns,
            "candidate_rank": candidate_ranks.get(symbol),
            "risk_rank": risk_ranks.get(symbol),
            "change": change,
            "cutoff": cutoff,
        }
        entries.append(entry)
        csv_rows.append(_phase2_csv_row(report_date_text, metadata, entry))

    suffix = f"_{report_identity}" if report_identity else ""
    csv_path = output_path / f"stock_summary_{report_date_text}{suffix}.csv"
    html_path = output_path / f"stock_summary_{report_date_text}{suffix}.html"
    _write_phase2_csv(csv_path, csv_rows)
    html_path.write_text(
        _render_phase2_html(report_date_text, metadata, entries, candidate_order, risk_order, normalized_issues),
        encoding="utf-8",
    )
    return {"csv": csv_path, "html": html_path}


def _prepare_metadata(
    run_metadata: Mapping[str, Any] | Any,
    report_date: str,
    results: list[dict[str, Any]],
    histories: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    metadata = _object_to_dict(run_metadata)
    first = results[0] if results else {}
    metadata["as_of_date"] = _earliest_date_text(metadata.get("as_of_date"), report_date) or report_date
    metadata["market_regime"] = metadata.get("market_regime") or first.get("market_regime") or "Unavailable"
    metadata["market_regime_confidence"] = metadata.get("market_regime_confidence", first.get("market_regime_confidence"))

    reasons = metadata.get("market_regime_reasons")
    if reasons is None:
        reasons = metadata.get("regime_reasons")
    if reasons is None:
        reasons = metadata.get("reasons_json") or metadata.get("market_regime_reasons_json")
    if reasons is None:
        reasons = first.get("market_regime_effects")
    metadata["market_regime_reasons"] = _as_list(reasons)

    scope_date = str(metadata["as_of_date"])
    latest_history_date: str | None = None
    for rows in histories.values():
        for row in rows:
            candidate = _bounded_date_text(row.get("trade_date"), scope_date)
            if candidate is not None and (latest_history_date is None or candidate > latest_history_date):
                latest_history_date = candidate
    if latest_history_date is None:
        for result in results:
            candidate = _bounded_date_text(result.get("data_through_date"), scope_date)
            if candidate is not None and (latest_history_date is None or candidate > latest_history_date):
                latest_history_date = candidate
    supplied_data_date = _bounded_date_text(metadata.get("data_through_date"), scope_date)
    metadata["data_through_date"] = latest_history_date or supplied_data_date

    generated_at = metadata.get("generated_at") or metadata.get("completed_at")
    metadata["generated_at"] = generated_at or datetime.now(timezone.utc).isoformat()
    return metadata


def _phase2_csv_row(report_date: str, metadata: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    result = entry["result"]
    previous = entry["previous"] or {}
    change = entry["change"]
    row: dict[str, Any] = {
        "report_date": report_date,
        "analysis_run_id": metadata.get("analysis_run_id") or metadata.get("run_id"),
        "as_of_date": metadata.get("as_of_date"),
        "data_through_date": _bounded_date_text(
            result.get("data_through_date"), report_date
        ),
        "scoring_version": metadata.get("scoring_version"),
        "configuration_hash": metadata.get("configuration_hash"),
        "benchmark_symbol": metadata.get("benchmark_symbol"),
        "market_regime": metadata.get("market_regime"),
        "market_regime_confidence": metadata.get("market_regime_confidence"),
        "market_regime_reasons": _canonical_json(metadata.get("market_regime_reasons", [])),
        "symbol": result.get("symbol"),
        "candidate_rank": entry.get("candidate_rank"),
        "risk_rank": entry.get("risk_rank"),
        "classification": result.get("classification"),
        "eligible_for_scoring": result.get("eligible_for_scoring"),
        "risk_score": result.get("risk_score"),
        "risk_level": result.get("risk_level"),
        "opportunity_score": result.get("opportunity_score"),
        "confidence_score": result.get("confidence_score"),
        "trend_state": result.get("trend_state"),
        "primary_reason": result.get("primary_reason"),
        "quality_concerns": _canonical_json(entry.get("quality_concerns", [])),
        "previous_classification": previous.get("classification"),
        "previous_risk_score": previous.get("risk_score"),
        "previous_opportunity_score": previous.get("opportunity_score"),
        "previous_confidence_score": previous.get("confidence_score"),
        "risk_score_change": change.get("risk_score_change"),
        "opportunity_score_change": change.get("opportunity_score_change"),
        "confidence_score_change": change.get("confidence_score_change"),
        "classification_changed": change.get("classification_changed"),
        "change_summary": change.get("summary"),
    }
    for field in _LIST_FIELDS:
        if field != "quality_concerns":
            row[field] = _canonical_json(result.get(field, []))
    for field in _COMPONENT_FIELDS:
        row[field] = _canonical_json(result.get(field, {}))
    return row


def _write_phase2_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PHASE2_CSV_FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _render_phase2_html(
    report_date: str,
    metadata: dict[str, Any],
    entries: list[dict[str, Any]],
    candidate_order: list[dict[str, Any]],
    risk_order: list[dict[str, Any]],
    quality_issues: list[dict[str, Any]],
) -> str:
    candidate_rank = {str(item.get("symbol", "")).upper(): rank for rank, item in enumerate(candidate_order, 1)}
    risk_rank = {str(item.get("symbol", "")).upper(): rank for rank, item in enumerate(risk_order, 1)}
    regime_reasons = _render_list(metadata.get("market_regime_reasons"), "No regime reasons were recorded.")

    metadata_rows = [
        ("Report date", report_date),
        ("As-of date", metadata.get("as_of_date")),
        ("Data-through date", metadata.get("data_through_date")),
        ("Analysis run", metadata.get("analysis_run_id") or metadata.get("run_id")),
        ("Scoring version", metadata.get("scoring_version")),
        ("Configuration hash", metadata.get("configuration_hash")),
        ("Benchmark", metadata.get("benchmark_symbol")),
        ("Generated at", metadata.get("generated_at")),
    ]
    metadata_html = "".join(
        f"<tr><th>{_escape(label)}</th><td>{_display(value)}</td></tr>" for label, value in metadata_rows
    )

    candidate_html = _ranking_table(candidate_order, candidate_rank, empty_message="No Candidate or Strong Candidate results.")
    risk_html = _ranking_table(risk_order, risk_rank, empty_message="No measured risk scores were available.")
    detail_html = "".join(_result_section(entry) for entry in entries)
    changes_html = _changes_table(entries)
    quality_html = _quality_table(quality_issues)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stock Scrapper Phase 2 Report — {_escape(report_date)}</title>
  <style>
    :root {{ color-scheme: light; --ink:#172033; --muted:#5f6b7a; --line:#d6dce5; --panel:#f7f9fc; }}
    body {{ max-width:1200px; margin:0 auto; padding:24px; color:var(--ink); font:15px/1.5 Arial,sans-serif; }}
    h1,h2,h3,h4 {{ line-height:1.2; }} h2 {{ border-bottom:2px solid var(--line); padding-bottom:6px; margin-top:34px; }}
    table {{ width:100%; border-collapse:collapse; margin:12px 0 22px; }}
    th,td {{ border:1px solid var(--line); padding:7px 9px; text-align:left; vertical-align:top; }}
    th {{ background:#eef2f7; }} .metadata th {{ width:210px; }}
    .notice {{ padding:12px 14px; border-left:5px solid #b45309; background:#fff7ed; margin:18px 0; }}
    .regime {{ padding:14px; background:#eef6ff; border:1px solid #bfdbfe; }}
    .stock {{ border:1px solid var(--line); border-radius:8px; padding:18px; margin:20px 0; }}
    .scores {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:12px 0; }}
    .score {{ background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:10px; }}
    .score strong {{ display:block; font-size:1.3rem; }}
    .chart-wrap {{ overflow-x:auto; }} .price-chart {{ width:100%; min-width:660px; height:auto; background:white; }}
    .legend {{ font-size:12px; }} .muted {{ color:var(--muted); }}
    .lists {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:14px; }}
    .lists section {{ background:var(--panel); padding:10px 14px; }} ul {{ margin-top:6px; }}
    code {{ overflow-wrap:anywhere; }}
  </style>
</head>
<body>
  <h1>Stock Scrapper Phase 2 Research Report</h1>
  <div class="notice"><strong>Research disclaimer:</strong> Educational research only; not personalized financial advice. Scores and classifications do not guarantee investment performance.</div>
  <h2>Run Metadata</h2>
  <table class="metadata"><tbody>{metadata_html}</tbody></table>
  <h2>Market Regime</h2>
  <div class="regime"><strong>{_display(metadata.get('market_regime'))}</strong> — confidence {_score(metadata.get('market_regime_confidence'))}<h3>Market-regime reasons</h3>{regime_reasons}</div>
  <h2>Candidate Ranking</h2>
  {candidate_html}
  <h2>Highest-Risk Ranking</h2>
  {risk_html}
  <h2>Changes From Previous Stored Analysis</h2>
  {changes_html}
  <h2>Data-Quality Concerns</h2>
  {quality_html}
  <h2>Symbol Analysis</h2>
  {detail_html or '<p>No symbol results were available.</p>'}
  <h2>Methodology</h2>
  <p>This report presents deterministic, explainable Phase 2 classifications using data available through the stated as-of date. Opportunity, measured risk, and confidence are separate 0–100 scales. Missing inputs remain unavailable rather than being silently treated as zero.</p>
  <p>Charts use adjusted closing prices supplied to the report and trailing, non-centered 20-, 50-, and 200-session simple moving averages. Rows later than the report/as-of date are excluded from charts. Candidate ranking uses higher opportunity, then higher confidence, lower risk, and symbol as a deterministic tie-breaker. Highest-risk ranking is descending by measured risk.</p>
  <h2>Research Disclaimer</h2>
  <p>This software is for educational and research use only. It does not provide personalized financial advice or recommend trades. Historical analysis does not guarantee future performance. Free market data may be delayed, revised, incomplete, or affected by survivorship and static-watchlist bias.</p>
</body>
</html>
"""


def _ranking_table(
    results: list[dict[str, Any]],
    ranks: Mapping[str, int],
    empty_message: str,
) -> str:
    if not results:
        return f"<p>{_escape(empty_message)}</p>"
    rows = []
    for result in results:
        symbol = str(result.get("symbol", "")).upper()
        rows.append(
            "<tr>"
            f"<td>{ranks.get(symbol, '')}</td><td>{_escape(symbol)}</td>"
            f"<td>{_display(result.get('classification'))}</td>"
            f"<td>{_score(result.get('opportunity_score'))}</td>"
            f"<td>{_score(result.get('risk_score'))}</td>"
            f"<td>{_score(result.get('confidence_score'))}</td></tr>"
        )
    return "<table><thead><tr><th>Rank</th><th>Symbol</th><th>Classification</th><th>Opportunity</th><th>Risk</th><th>Confidence</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _result_section(entry: dict[str, Any]) -> str:
    result = entry["result"]
    symbol = str(result.get("symbol", ""))
    chart = _price_chart_svg(symbol, entry.get("history", []), entry.get("cutoff"))
    component_html = "".join(
        _component_table(title, result.get(field))
        for title, field in (
            ("Risk components", "risk_components"),
            ("Opportunity components", "opportunity_components"),
            ("Confidence components", "confidence_components"),
            ("Indicator snapshot", "indicators"),
        )
    )
    lists = (
        ("Positive factors", result.get("positive_factors")),
        ("Risk factors", result.get("risk_factors")),
        ("Confidence limitations", result.get("confidence_limitations")),
        ("Data-quality concerns", entry.get("quality_concerns")),
        ("Market-regime effects", result.get("market_regime_effects")),
        ("Improvement conditions", result.get("improvement_conditions")),
        ("Weakening conditions", result.get("weakening_conditions")),
        ("Blocking reasons", result.get("blocking_reasons")),
        ("Flags", result.get("flags")),
    )
    list_html = "".join(
        f"<section><h4>{_escape(title)}</h4>{_render_list(values, 'None recorded.')}</section>"
        for title, values in lists
    )
    return f"""<article class="stock">
<h3>{_escape(symbol)} — {_display(result.get('classification'))}</h3>
<p><strong>Primary reason:</strong> {_display(result.get('primary_reason'))}<br />
<strong>Data through:</strong> {_display(result.get('data_through_date'))} &nbsp; <strong>Trend state:</strong> {_display(result.get('trend_state'))}</p>
<div class="scores"><div class="score">Opportunity<strong>{_score(result.get('opportunity_score'))}</strong></div><div class="score">Measured risk<strong>{_score(result.get('risk_score'))}</strong><span>{_display(result.get('risk_level'))}</span></div><div class="score">Confidence<strong>{_score(result.get('confidence_score'))}</strong></div><div class="score">Candidate rank<strong>{_display(entry.get('candidate_rank'))}</strong></div><div class="score">Risk rank<strong>{_display(entry.get('risk_rank'))}</strong></div></div>
<h4>Adjusted Price and Moving Averages</h4>{chart}
<div class="lists">{list_html}</div>
<h4>Changes from previous stored analysis</h4><p>{_display(entry['change'].get('summary'))}</p>
{component_html}
</article>"""


def _component_table(title: str, components: Any) -> str:
    values = _as_dict(components)
    if not values:
        return f"<h4>{_escape(title)}</h4><p class=\"muted\">Unavailable.</p>"
    rows = "".join(
        f"<tr><th>{_escape(key)}</th><td>{_display_component(value)}</td></tr>"
        for key, value in sorted(values.items(), key=lambda item: str(item[0]))
    )
    return f"<h4>{_escape(title)}</h4><table><tbody>{rows}</tbody></table>"


def _changes_table(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "<p>No results were available for comparison.</p>"
    rows = "".join(
        "<tr>"
        f"<td>{_display(entry['result'].get('symbol'))}</td>"
        f"<td>{_display((entry.get('previous') or {}).get('classification'))}</td>"
        f"<td>{_display(entry['result'].get('classification'))}</td>"
        f"<td>{_score_change(entry['change'].get('opportunity_score_change'))}</td>"
        f"<td>{_score_change(entry['change'].get('risk_score_change'))}</td>"
        f"<td>{_score_change(entry['change'].get('confidence_score_change'))}</td>"
        f"<td>{_display(entry['change'].get('summary'))}</td></tr>"
        for entry in entries
    )
    return "<table><thead><tr><th>Symbol</th><th>Previous</th><th>Current</th><th>Opportunity Δ</th><th>Risk Δ</th><th>Confidence Δ</th><th>Summary</th></tr></thead><tbody>" + rows + "</tbody></table>"


def _quality_table(issues: list[dict[str, Any]]) -> str:
    if not issues:
        return "<p>No data-quality concerns were supplied for this report.</p>"
    rows = "".join(
        "<tr>"
        f"<td>{_display(issue.get('symbol'))}</td><td>{_display(issue.get('trade_date'))}</td>"
        f"<td>{_display(issue.get('severity'))}</td><td>{_display(issue.get('issue_type'))}</td>"
        f"<td>{_display(issue.get('description'))}</td></tr>"
        for issue in sorted(issues, key=lambda item: (str(item.get("symbol", "")), str(item.get("trade_date", "")), str(item.get("issue_type", ""))))
    )
    return "<table><thead><tr><th>Symbol</th><th>Date</th><th>Severity</th><th>Type</th><th>Description</th></tr></thead><tbody>" + rows + "</tbody></table>"


def _price_chart_svg(symbol: str, history: list[dict[str, Any]], cutoff: str | None) -> str:
    points = _chart_points(history, cutoff)
    if not points or all(value is None for _, value in points):
        return '<p class="muted">No adjusted-price history was available for this chart.</p>'

    prices = [value for _, value in points]
    series = {
        "adjusted-price": prices,
        "sma-20": _rolling_average(prices, 20),
        "sma-50": _rolling_average(prices, 50),
        "sma-200": _rolling_average(prices, 200),
    }
    available_values = [value for values in series.values() for value in values if value is not None]
    if not available_values:
        return '<p class="muted">No adjusted-price history was available for this chart.</p>'

    width, height = 900.0, 320.0
    left, right, top, bottom = 58.0, 18.0, 24.0, 42.0
    plot_width, plot_height = width - left - right, height - top - bottom
    minimum, maximum = min(available_values), max(available_values)
    padding = (maximum - minimum) * 0.05 if maximum != minimum else max(abs(maximum) * 0.05, 1.0)
    minimum -= padding
    maximum += padding

    def x_position(index: int) -> float:
        return left if len(points) == 1 else left + index * plot_width / (len(points) - 1)

    def y_position(value: float) -> float:
        return top + (maximum - value) * plot_height / (maximum - minimum)

    colors = {
        "adjusted-price": "#1d4ed8",
        "sma-20": "#d97706",
        "sma-50": "#15803d",
        "sma-200": "#7e22ce",
    }
    labels = {"adjusted-price": "Adjusted price", "sma-20": "SMA20", "sma-50": "SMA50", "sma-200": "SMA200"}
    grid = []
    for tick in range(5):
        fraction = tick / 4
        y = top + fraction * plot_height
        value = maximum - fraction * (maximum - minimum)
        grid.append(f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{width-right:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/><text x="{left-7:.1f}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#4b5563">{value:.2f}</text>')

    paths = []
    for name, values in series.items():
        for segment in _contiguous_segments(values):
            coordinates = " ".join(f"{x_position(index):.2f},{y_position(value):.2f}" for index, value in segment)
            if len(segment) == 1:
                index, value = segment[0]
                paths.append(f'<circle data-series="{name}" cx="{x_position(index):.2f}" cy="{y_position(value):.2f}" r="2" fill="{colors[name]}"/>')
            else:
                paths.append(f'<polyline data-series="{name}" points="{coordinates}" fill="none" stroke="{colors[name]}" stroke-width="{2.2 if name == "adjusted-price" else 1.7}"/>')

    legend = []
    for index, name in enumerate(series):
        x = left + index * 150
        legend.append(f'<line x1="{x:.1f}" y1="{height-12:.1f}" x2="{x+22:.1f}" y2="{height-12:.1f}" stroke="{colors[name]}" stroke-width="3"/><text class="legend" x="{x+28:.1f}" y="{height-8:.1f}" fill="#374151">{labels[name]}</text>')

    safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", symbol).strip("-") or "symbol"
    first_date, last_date = points[0][0], points[-1][0]
    return f'''<div class="chart-wrap"><svg class="price-chart" viewBox="0 0 {int(width)} {int(height)}" role="img" aria-labelledby="chart-{_escape(safe_id)}-title"><title id="chart-{_escape(safe_id)}-title">{_escape(symbol)} adjusted price with 20-, 50-, and 200-session moving averages</title><rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#fff" stroke="#d1d5db"/>{''.join(grid)}{''.join(paths)}<text x="{left:.1f}" y="{height-bottom+18:.1f}" font-size="11">{_escape(first_date)}</text><text x="{width-right:.1f}" y="{height-bottom+18:.1f}" text-anchor="end" font-size="11">{_escape(last_date)}</text>{''.join(legend)}</svg></div>'''


def _chart_points(history: list[dict[str, Any]], cutoff: str | None) -> list[tuple[str, float | None]]:
    cutoff_date = _parse_date(cutoff)
    by_date: dict[str, float | None] = {}
    for raw_row in history:
        row = _object_to_dict(raw_row)
        trade_date = _parse_date(row.get("trade_date"))
        if trade_date is None or (cutoff_date is not None and trade_date > cutoff_date):
            continue
        by_date[trade_date.isoformat()] = _finite_number(row.get("adjusted_close"))
    return sorted(by_date.items())


def _rolling_average(values: list[float | None], window: int) -> list[float | None]:
    result: list[float | None] = []
    for index in range(len(values)):
        if index + 1 < window:
            result.append(None)
            continue
        trailing = values[index - window + 1 : index + 1]
        result.append(sum(trailing) / window if all(value is not None for value in trailing) else None)
    return result


def _contiguous_segments(values: list[float | None]) -> list[list[tuple[int, float]]]:
    segments: list[list[tuple[int, float]]] = []
    current: list[tuple[int, float]] = []
    for index, value in enumerate(values):
        if value is None:
            if current:
                segments.append(current)
                current = []
        else:
            current.append((index, value))
    if current:
        segments.append(current)
    return segments


def _analysis_change(current: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    if previous is None:
        return {
            "risk_score_change": None,
            "opportunity_score_change": None,
            "confidence_score_change": None,
            "classification_changed": None,
            "summary": "No previous stored analysis was available.",
        }
    changes = {
        "risk_score_change": _numeric_change(current.get("risk_score"), previous.get("risk_score")),
        "opportunity_score_change": _numeric_change(current.get("opportunity_score"), previous.get("opportunity_score")),
        "confidence_score_change": _numeric_change(current.get("confidence_score"), previous.get("confidence_score")),
        "classification_changed": current.get("classification") != previous.get("classification"),
    }
    parts = []
    if changes["classification_changed"]:
        parts.append(f"Classification changed from {previous.get('classification') or 'Unavailable'} to {current.get('classification') or 'Unavailable'}")
    for label, field in (("opportunity", "opportunity_score_change"), ("risk", "risk_score_change"), ("confidence", "confidence_score_change")):
        value = changes[field]
        if value is not None:
            parts.append(f"{label.capitalize()} {value:+.2f}")
    changes["summary"] = "; ".join(parts) if parts else "No material score or classification change was available."
    return changes


def _numeric_change(current: Any, previous: Any) -> float | None:
    current_number = _finite_number(current)
    previous_number = _finite_number(previous)
    return None if current_number is None or previous_number is None else current_number - previous_number


def _normalize_results(values: Iterable[Mapping[str, Any] | Any] | Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(values, Mapping):
        if "symbol" in values:
            source_values: Iterable[Any] = [values]
        else:
            expanded = []
            for symbol, value in values.items():
                item = _object_to_dict(value)
                item.setdefault("symbol", symbol)
                expanded.append(item)
            source_values = expanded
    else:
        source_values = values
    return [_normalize_result(value) for value in source_values]


def _normalize_result(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    result = _object_to_dict(value)
    for field in _LIST_FIELDS:
        source = result.get(field)
        if source in (None, ""):
            source = result.get(f"{field}_json")
        result[field] = _as_list(source)
    for field in _COMPONENT_FIELDS:
        source = result.get(field)
        if source in (None, ""):
            source = result.get(f"{field}_json")
        result[field] = _strip_history_payloads(_as_dict(source))
    if result.get("symbol") is not None:
        result["symbol"] = str(result["symbol"]).upper()
    return result


def _history_rows(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if hasattr(value, "to_dict") and not isinstance(value, Mapping):
        try:
            value = value.to_dict(orient="records")
        except TypeError:
            value = value.to_dict()
    if isinstance(value, Mapping):
        value = [value]
    return [_object_to_dict(row) for row in value]


def _object_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "keys"):
        try:
            return {key: value[key] for key in value.keys()}
        except (KeyError, TypeError):
            pass
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"Expected a mapping or result-like object, got {type(value).__name__}")


def _as_list(value: Any) -> list[Any]:
    decoded = _decode_json(value)
    if decoded is None:
        return []
    if isinstance(decoded, list):
        return decoded
    if isinstance(decoded, (tuple, set)):
        return list(decoded)
    return [decoded]


def _as_dict(value: Any) -> dict[str, Any]:
    decoded = _decode_json(value)
    return dict(decoded) if isinstance(decoded, Mapping) else {}


def _decode_json(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "{")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _issues_for_symbol(issues: list[dict[str, Any]], symbol: str) -> list[dict[str, Any]]:
    return [
        issue
        for issue in issues
        if not issue.get("symbol") or str(issue.get("symbol", "")).upper() == symbol
    ]


def _issue_is_in_scope(issue: dict[str, Any], report_date: str) -> bool:
    issue_date = _parse_date(issue.get("trade_date"))
    maximum = _parse_date(report_date)
    return issue_date is None or maximum is None or issue_date <= maximum


def _deduplicate(values: Sequence[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _normalize_date(value: str | date, field_name: str) -> str:
    parsed = _parse_date(value)
    if parsed is None:
        raise ValueError(f"{field_name} must be a valid YYYY-MM-DD date")
    return parsed.isoformat()


def _parse_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _earliest_date_text(*values: Any) -> str | None:
    parsed = [item for item in (_parse_date(value) for value in values) if item is not None]
    return min(parsed).isoformat() if parsed else None


def _bounded_date_text(value: Any, maximum: str) -> str | None:
    parsed = _parse_date(value)
    max_date = _parse_date(maximum)
    if parsed is None or max_date is None:
        return None
    return parsed.isoformat() if parsed <= max_date else None


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sortable_number(value: Any, default: float) -> float:
    number = _finite_number(value)
    return default if number is None else number


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _strip_history_payloads(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_history_payloads(item)
            for key, item in value.items()
            if str(key).lower() not in _HISTORY_PAYLOAD_KEYS
        }
    if isinstance(value, list):
        return [_strip_history_payloads(item) for item in value]
    return value


def _render_list(values: Any, empty_message: str) -> str:
    items = _as_list(values)
    if not items:
        return f'<p class="muted">{_escape(empty_message)}</p>'
    return "<ul>" + "".join(f"<li>{_display(item)}</li>" for item in items) + "</ul>"


def _display_component(value: Any) -> str:
    if isinstance(value, (Mapping, list, tuple, set)):
        return f"<code>{_escape(_canonical_json(value))}</code>"
    return _display(value)


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _display(value: Any) -> str:
    if value is None or value == "":
        return "<span class=\"muted\">Unavailable</span>"
    return _escape(value)


def _score(value: Any) -> str:
    number = _finite_number(value)
    return '<span class="muted">Unavailable</span>' if number is None else f"{number:.2f}"


def _score_change(value: Any) -> str:
    number = _finite_number(value)
    return '<span class="muted">Unavailable</span>' if number is None else f"{number:+.2f}"


def _flatten_row_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    """Convert legacy report rows into a CSV-safe shape."""
    flattened: dict[str, Any] = {}
    for key, value in row.items():
        if key == "history":
            continue
        flattened[key] = _canonical_json(value) if isinstance(value, (list, dict, tuple, set)) else value
    return flattened


def write_csv_report(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write the legacy stock-summary CSV without historical arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in _flatten_row_for_csv(row).keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _flatten_row_for_csv(row).get(key) for key in fieldnames})
    return path


def write_html_report(
    path: Path,
    summary_rows: list[dict[str, Any]],
    config: dict[str, Any],
    data_source: str,
    successful_symbols: list[str],
    failed_symbols: list[str],
    quality_issues: list[dict[str, Any]],
) -> Path:
    """Write the legacy Phase 1-compatible HTML summary."""
    _ = config
    path.parent.mkdir(parents=True, exist_ok=True)
    market_rows = "".join(
        "<tr>"
        f"<td>{_escape(row.get('symbol'))}</td><td>{_escape(row.get('status'))}</td>"
        f"<td>{_escape(row.get('latest_close'))}</td><td>{_escape(row.get('latest_trading_date'))}</td>"
        f"<td>{_escape(row.get('twenty_day_volatility'))}</td></tr>"
        for row in summary_rows
    )
    sections = []
    for row in summary_rows:
        history = row.get("history", [])
        chart_text = f"{len(history)} price points stored locally." if history else "No price history available."
        sections.append(
            f"<section><h3>{_escape(row.get('symbol'))}</h3>"
            f"<p><strong>Status:</strong> {_escape(row.get('status'))}<br/>"
            f"<strong>Flags:</strong> {_escape(', '.join(str(flag) for flag in row.get('flags', [])))}<br/>"
            f"<strong>Latest close:</strong> {_escape(row.get('latest_close'))}<br/>"
            f"<strong>One-day return:</strong> {_escape(row.get('one_day_return'))}<br/>"
            f"<strong>20-day volatility:</strong> {_escape(row.get('twenty_day_volatility'))}</p>"
            f"<p>{_escape(chart_text)}</p></section>"
        )
    quality_items = "".join(
        f"<li>{_escape(issue.get('symbol'))} - {_escape(issue.get('issue_type'))} - {_escape(issue.get('severity'))}</li>"
        for issue in quality_issues
    ) or "<li>No warnings recorded.</li>"
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    content = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/><title>Stock Scrapper Report</title><style>body{{font-family:Arial,sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%;margin-bottom:24px}}th,td{{border:1px solid #d1d5db;padding:8px;text-align:left}}th{{background:#f3f4f6}}.note{{background:#fef3c7;padding:12px;border-left:4px solid #f59e0b}}</style></head><body>
<h1>Stock Scrapper</h1><p><strong>Generated:</strong> {generated_at} UTC</p><p><strong>Data source:</strong> {_escape(data_source)}</p><p><strong>Symbols analyzed:</strong> {len(summary_rows)}</p><p><strong>Successful:</strong> {_escape(', '.join(successful_symbols) or 'None')}</p><p><strong>Failed:</strong> {_escape(', '.join(failed_symbols) or 'None')}</p><div class="note">This report is for research and educational purposes only. It is not financial advice.</div>
<h2>Market Summary</h2><table><thead><tr><th>Symbol</th><th>Status</th><th>Latest Close</th><th>Latest Date</th><th>20d Volatility</th></tr></thead><tbody>{market_rows}</tbody></table><h2>Data Quality Warnings</h2><ul>{quality_items}</ul><h2>Stock Details</h2>{''.join(sections)}<h2>Statistic Explanations</h2><ul><li>Latest close: the most recent closing price in the local database.</li><li>One-day return: the percentage change from the previous trading day.</li><li>Moving averages describe trailing price trends.</li><li>Volatility estimates recent price variability.</li></ul></body></html>"""
    path.write_text(content, encoding="utf-8")
    return path
