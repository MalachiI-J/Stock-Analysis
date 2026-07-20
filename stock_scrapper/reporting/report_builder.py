"""Generate CSV and HTML reports for the Phase 1 market summary."""

from __future__ import annotations

import csv
import html
from datetime import datetime
from pathlib import Path
from typing import Any


def _flatten_row_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    """Convert report rows into a CSV-safe shape by removing nested payloads."""
    flattened: dict[str, Any] = {}
    for key, value in row.items():
        if key == "history":
            continue
        if isinstance(value, (list, dict)):
            flattened[key] = ",".join(str(item) for item in value) if isinstance(value, list) else str(value)
        else:
            flattened[key] = value
    return flattened


def write_csv_report(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write the stock summary rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in _flatten_row_for_csv(row).keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _flatten_row_for_csv(row).get(key) for key in fieldnames})
    return path


def write_html_report(path: Path, summary_rows: list[dict[str, Any]], config: dict[str, Any], data_source: str, successful_symbols: list[str], failed_symbols: list[str], quality_issues: list[dict[str, Any]]) -> Path:
    """Create a browser-friendly HTML summary with simple charts."""
    # The HTML report is self-contained so it can be opened directly in a browser without a server.
    path.parent.mkdir(parents=True, exist_ok=True)

    def _safe_text(value: Any) -> str:
        return "" if value is None else str(value)

    market_rows = []
    for row in summary_rows:
        market_rows.append(
            "<tr>"
            f"<td>{html.escape(_safe_text(row.get('symbol')))}</td>"
            f"<td>{html.escape(_safe_text(row.get('status')))}</td>"
            f"<td>{html.escape(_safe_text(row.get('latest_close')))}</td>"
            f"<td>{html.escape(_safe_text(row.get('latest_trading_date')))}</td>"
            f"<td>{html.escape(_safe_text(row.get('twenty_day_volatility')))}</td>"
            f"</tr>"
        )

    sections = []
    for row in summary_rows:
        symbol = row.get("symbol")
        history = row.get("history", [])
        if history:
            prices = [item.get("close") for item in history if item.get("close") is not None]
            chart_html = f"<p>Offline chart available for {html.escape(_safe_text(symbol))}: {len(prices)} price points stored locally.</p>"
        else:
            chart_html = "<p>No price history available.</p>"

        sections.append(
            f"<section><h3>{html.escape(_safe_text(symbol))}</h3>"
            f"<p><strong>Status:</strong> {html.escape(_safe_text(row.get('status')))}<br/>"
            f"<strong>Flags:</strong> {html.escape(_safe_text(', '.join(row.get('flags', []))))}<br/>"
            f"<strong>Latest close:</strong> {html.escape(_safe_text(row.get('latest_close')))}<br/>"
            f"<strong>One-day return:</strong> {html.escape(_safe_text(row.get('one_day_return')))}<br/>"
            f"<strong>20-day volatility:</strong> {html.escape(_safe_text(row.get('twenty_day_volatility')))}</p>"
            f"{chart_html}</section>"
        )

    quality_items = []
    for issue in quality_issues:
        quality_items.append(
            f"<li>{html.escape(_safe_text(issue.get('symbol')))} - {html.escape(_safe_text(issue.get('issue_type')))} - {html.escape(_safe_text(issue.get('severity')))}</li>"
        )

    html_content = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Stock Scrapper Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h1, h2, h3 {{ color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px; text-align: left; }}
    th {{ background-color: #f3f4f6; }}
    .note {{ background-color: #fef3c7; padding: 12px; border-left: 4px solid #f59e0b; }}
  </style>
</head>
<body>
  <h1>Stock Scrapper</h1>
  <p><strong>Generated:</strong> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
  <p><strong>Data source:</strong> {html.escape(_safe_text(data_source))}</p>
  <p><strong>Symbols analyzed:</strong> {len(summary_rows)}</p>
  <p><strong>Successful:</strong> {', '.join(successful_symbols) or 'None'}</p>
  <p><strong>Failed:</strong> {', '.join(failed_symbols) or 'None'}</p>
  <div class=\"note\">This report is for research and educational purposes only. It is not financial advice.</div>
  <h2>Market Summary</h2>
  <table>
    <thead><tr><th>Symbol</th><th>Status</th><th>Latest Close</th><th>Latest Date</th><th>20d Volatility</th></tr></thead>
    <tbody>{''.join(market_rows)}</tbody>
  </table>
  <h2>Data Quality Warnings</h2>
  <ul>{''.join(quality_items) if quality_items else '<li>No warnings recorded.</li>'}</ul>
  <h2>Stock Details</h2>
  {''.join(sections)}
  <h2>Statistic Explanations</h2>
  <ul>
    <li>Latest close: the most recent closing price in the local database.</li>
    <li>One-day return: the percentage change from the previous trading day.</li>
    <li>Moving averages: smooth recent price action and are used to describe simple trend direction.</li>
    <li>Volatility: a basic estimate of recent price variability.</li>
  </ul>
</body>
</html>
"""
    path.write_text(html_content, encoding="utf-8")
    return path
