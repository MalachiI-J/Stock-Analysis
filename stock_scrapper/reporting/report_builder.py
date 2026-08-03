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

# Status-color mapping for badges: reuses the report's fixed status palette
# (good/warning/serious/critical/neutral) rather than inventing new colors per
# field, so "Strong Candidate" and "Low risk" read as the same kind of good.
_CLASSIFICATION_STATUS: dict[str, str] = {
    "Strong Candidate": "good",
    "Candidate": "good",
    "Watch": "warning",
    "Avoid": "serious",
    "High Risk": "critical",
    "Insufficient Data": "neutral",
    "Data Blocked": "neutral",
}
_RISK_LEVEL_STATUS: dict[str, str] = {
    "Low": "good",
    "Moderate": "warning",
    "Elevated": "serious",
    "High": "critical",
    "Unavailable": "neutral",
}
_REGIME_STATUS: dict[str, str] = {
    "Risk-On": "good",
    "Neutral": "neutral",
    "Risk-Off": "serious",
    "Stress": "critical",
    "Insufficient Market Data": "neutral",
}


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
    signal_validation = _latest_signal_validation_summary(output_path)
    recommendations_summary = _latest_recommendations_summary(output_path, report_date_text)
    html_path.write_text(
        _render_phase2_html(
            report_date_text, metadata, entries, candidate_order, risk_order, normalized_issues,
            signal_validation=signal_validation,
            recommendations_summary=recommendations_summary,
        ),
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


_REPORT_STYLES = """\
    /* Data-terminal theme: dark by default, with a light counterpart selected
       by [data-theme] on <html> (set by the theme script in <head>, before
       paint, so there's no flash of the wrong theme). Every color used below
       is one of these custom properties — never a hardcoded hex scattered
       through a component rule — so switching theme is one attribute flip,
       not a per-component conditional. The hero banner never reads these
       specific tokens (it uses its own literal colors, recolored by its own
       [data-theme="light"] overrides further down) — same mechanism, its own
       palette, since the hero's colors were never meant to match the body's. */
    :root, [data-theme="dark"] {
      color-scheme: dark;
      /* Opaque now (was rgba(255,255,255,0.05)) — a translucent surface let
         the page texture show through every card. This solid hex is chosen
         to match that old composited appearance, so nothing else needed to
         be re-tuned to compensate. */
      --page:#10130f; --surface:#1c1f1b; --border:rgba(255,255,255,0.14);
      --ink:#e7e5df; --ink-2:#8a8d87; --muted:#8a8d87;
      --line:rgba(255,255,255,0.10); --th-bg:rgba(255,255,255,0.035);
      --page-grid-a:rgba(150,180,200,0.05); --page-grid-b:rgba(150,180,200,0.04);
      /* A lightening highlight reads as a soft glow against a near-black
         page; the same white value would be a no-op against light mode's
         near-white page (see the light block below for that variant). */
      --page-glow:rgba(255,255,255,0.10);
      /* Same color as --page in both themes — a separate name purely so
         component rules can say "recessed instrument-panel surface" instead
         of "the page color, again," now that --surface itself is a real,
         opaque card color and no longer doubles for both roles. */
      --inset-surface:var(--page);
      --track-bg:rgba(255,255,255,0.10); --chip-bg:rgba(255,255,255,0.06);
      --hover-bg:rgba(255,255,255,0.06);
      --card-shadow:0 12px 28px rgba(0,0,0,0.35);
      /* Deliberately DARKER/recessed than --surface (used for the active
         circle below) — that contrast is what reads as "raised," not flat. */
      --toggle-track:rgba(255,255,255,0.03); --toggle-shadow:0 1px 3px rgba(0,0,0,.45);
      --mono:"JetBrains Mono","IBM Plex Mono",ui-monospace,"SFMono-Regular",Menlo,Consolas,monospace;
      --sans:system-ui,-apple-system,"Segoe UI",sans-serif;
      /* Three semantic accents only — color appears exclusively where it means
         something (a classification, a risk level, a delta), never decoratively. */
      --good-fg:#5DCAA5; --good-bg:rgba(93,202,165,0.12); --good-border:rgba(93,202,165,0.35);
      --warning-fg:#FAC775; --warning-bg:rgba(250,199,117,0.12); --warning-border:rgba(250,199,117,0.35);
      --serious-fg:#F09595; --serious-bg:rgba(240,149,149,0.12); --serious-border:rgba(240,149,149,0.35);
      --critical-fg:#F09595; --critical-bg:rgba(240,149,149,0.12); --critical-border:rgba(240,149,149,0.35);
      --neutral-fg:#8a8d87; --neutral-bg:rgba(138,141,135,0.12); --neutral-border:rgba(138,141,135,0.30);
      --good-accent-bar:var(--good-border); --warning-accent-bar:var(--warning-border);
      --serious-accent-bar:var(--serious-border); --critical-accent-bar:var(--serious-border);
      --chart-blue:#3987e5; --chart-orange:#d95926; --chart-aqua:#199e70; --chart-yellow:#eda100;
    }
    [data-theme="light"] {
      color-scheme: light;
      --page:#F7F6F2; --surface:#FFFFFF; --border:rgba(0,0,0,0.10);
      --ink:#1a1c18; --ink-2:#6b6f68; --muted:#6b6f68;
      --line:rgba(0,0,0,0.08); --th-bg:rgba(0,0,0,0.025);
      --page-grid-a:rgba(90,110,130,0.05); --page-grid-b:rgba(90,110,130,0.04);
      /* A white glow would be invisible against a near-white page — a very
         soft, low-alpha dark tint gives the same "a bit more presence near
         the top" effect without needing a lighter-than-page color to exist. */
      --page-glow:rgba(20,22,18,0.05);
      --inset-surface:var(--page);
      --track-bg:rgba(0,0,0,0.08); --chip-bg:rgba(0,0,0,0.045);
      --hover-bg:rgba(0,0,0,0.045);
      --card-shadow:0 10px 24px rgba(0,0,0,0.10);
      --toggle-track:rgba(0,0,0,0.06); --toggle-shadow:0 1px 2px rgba(0,0,0,.18);
      --good-fg:#27500A; --good-bg:#EAF3DE; --good-border:rgba(39,80,10,0.30);
      --warning-fg:#633806; --warning-bg:#FAEEDA; --warning-border:rgba(99,56,6,0.30);
      --serious-fg:#791F1F; --serious-bg:#FCEBEB; --serious-border:rgba(121,31,31,0.28);
      --critical-fg:#791F1F; --critical-bg:#FCEBEB; --critical-border:rgba(121,31,31,0.28);
      --neutral-fg:#6b6f68; --neutral-bg:rgba(0,0,0,0.06); --neutral-border:rgba(0,0,0,0.18);
      --good-accent-bar:#639922; --warning-accent-bar:#BA7517;
      --serious-accent-bar:#BA3B3B; --critical-accent-bar:#BA3B3B;
      --chart-blue:#2a78d6; --chart-orange:#eb6834; --chart-aqua:#1baf7a; --chart-yellow:#c98500;
    }
    * { box-sizing:border-box; }
    /* Dot-grid page texture: a sparse field of ~1px dots on a 28px pitch,
       two interleaved layers (offset by half a cell) from the same
       --page-grid-a/-b tokens used everywhere else, so it keeps differing
       correctly between themes automatically. `.page-fade` (first child of
       body — see markup) is a second, document-anchored copy of the same
       pattern, mask-faded to nothing over roughly one viewport height: near
       the hero the two layers overlap (denser), and past that only this
       steady base layer remains (dimmer, but never zero — no "recede to
       nothing" for long reports). `.page-glow` is a separate, small radial
       accent near the top center, driven by its own --page-glow token (a
       lightening highlight in dark mode; a white glow would be invisible on
       light mode's near-white page, so that token holds a soft darkening
       tint there instead) — same idea in both themes, independent of the
       fade's own masking so its lifecycle stays simple to reason about. Once
       a card/table has an
       opaque background (see --surface, step 1), it naturally paints over
       whichever part of this texture would otherwise show behind it — no
       extra "stop at the card edge" rule needed. Both grid layers use plain
       `scroll` attachment (no `fixed`) on purpose: a `fixed` layer is
       viewport-anchored while the fade overlay is document-anchored, and
       mixing the two would drift the two grids out of phase with each other
       as the page scrolls. */
    body { margin:0; color:var(--ink); position:relative;
      font:15px/1.6 var(--sans);
      background-color:var(--page);
      background-image:
        radial-gradient(circle, var(--page-grid-a) 1px, transparent 1.5px),
        radial-gradient(circle, var(--page-grid-b) 1px, transparent 1.5px);
      background-size:28px 28px, 28px 28px;
      background-position:0 0, 14px 14px; }
    .page-fade { position:absolute; top:0; left:0; right:0; height:100vh; z-index:0; pointer-events:none;
      background-image:
        radial-gradient(circle, var(--page-grid-a) 1px, transparent 1.5px),
        radial-gradient(circle, var(--page-grid-b) 1px, transparent 1.5px);
      background-size:28px 28px, 28px 28px;
      background-position:0 0, 14px 14px;
      -webkit-mask-image:linear-gradient(to bottom, black, transparent);
      mask-image:linear-gradient(to bottom, black, transparent); }
    .page-glow { position:absolute; top:0; left:0; right:0; height:60vh; z-index:0; pointer-events:none;
      background:radial-gradient(ellipse at 50% 0%, var(--page-glow), transparent 60%); }
    .page { max-width:1200px; margin:0 auto; padding:8px 24px 56px; }
    h1,h2,h3,h4 { line-height:1.25; font-weight:600; }
    h1 { font-size:1.7rem; margin-bottom:2px; }
    .subtitle { color:var(--ink-2); margin-top:0; margin-bottom:18px; font-size:14px; }
    h2 { border-bottom:1px solid var(--line); padding-bottom:8px; margin-top:56px; scroll-margin-top:52px; }
    h2:target { color:var(--good-fg); transition:color 1.8s ease; }
    h4 { margin-bottom:6px; }
    table { width:100%; border-collapse:collapse; margin:12px 0 22px; }
    th,td { padding:11px 14px; text-align:left; vertical-align:top; }
    td { border-bottom:1px solid var(--line); }
    tbody tr:last-child td { border-bottom:none; }
    table:not(.metadata) tbody tr:nth-child(even) td { background:var(--th-bg); }
    table:not(.metadata) tbody tr:hover td { background:var(--hover-bg); }
    th { background:var(--th-bg); color:var(--ink-2); font-weight:600;
      font-size:12px; text-transform:uppercase; letter-spacing:.04em; border-bottom:1px solid var(--border); }
    td.num, th.num { text-align:right; }
    /* Accent-edge rows (Candidate/Highest-Risk ranking): the row's left border
       plus plain colored classification text do the job a filled badge pill
       used to do alone — see _ranking_table(). */
    tbody tr[data-status] { border-left:3px solid var(--neutral-fg); }
    tbody tr[data-status="good"] { border-left-color:var(--good-fg); }
    tbody tr[data-status="warning"] { border-left-color:var(--warning-fg); }
    tbody tr[data-status="serious"] { border-left-color:var(--serious-fg); }
    tbody tr[data-status="critical"] { border-left-color:var(--critical-fg); }
    .status-good { color:var(--good-fg); } .status-warning { color:var(--warning-fg); }
    .status-serious { color:var(--serious-fg); } .status-critical { color:var(--critical-fg); }
    .status-neutral { color:var(--neutral-fg); }
    .mono, td.num, .stat-value, .delta, code, .kv dd, .metadata td { font-family:var(--mono); font-variant-numeric:tabular-nums; }
    /* Shared key/value table style (used by the footer's run-details
       disclosure): hairline row separators only, no visible cell borders. */
    table.metadata { border:none; margin:0; }
    table.metadata tr { border-bottom:1px solid var(--line); }
    table.metadata tr:last-child { border-bottom:none; }
    table.metadata th, table.metadata td { border:none; padding:9px 4px; }
    table.metadata th { background:transparent; width:200px; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
    /* Elevation lives only on major panels (.card, .stock) — a single drop
       shadow, no inset highlight, no colored glow. Recessed/secondary
       elements (.stat, .lists section, details.raw, .notice, the chart)
       deliberately don't get it — they should read as set INTO the page,
       not floating above it. */
    .card { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px 18px; margin:14px 0 22px;
      box-shadow:var(--card-shadow); }
    .card > table { margin:0; }
    .notice { padding:14px 16px; border:1px solid var(--border); border-left:3px solid var(--muted);
      background:var(--surface); color:var(--ink-2); border-radius:8px; margin:18px 0; }
    .notice-good { border-left-color:var(--good-border); }
    .notice-critical { border-left-color:var(--critical-border); }
    /* Collapsed-by-default detail inside a .notice — same ▸/▾ marker convention
       as details.raw, but inline text-sized rather than a boxed panel. */
    .notice details { margin-top:6px; font-size:12.5px; }
    .notice details > summary { cursor:pointer; color:var(--muted); list-style:none; user-select:none; }
    .notice details > summary::-webkit-details-marker { display:none; }
    .notice details > summary::before { content:"▸ "; }
    .notice details[open] > summary::before { content:"▾ "; }
    .notice details p { margin:6px 0 0; }
    .regime-head { display:flex; align-items:center; gap:12px; }
    .regime-confidence { color:var(--ink-2); font-size:13px; font-family:var(--mono); }
    .card ul { list-style:none; margin:6px 0 0; padding:0; }
    .card ul li { position:relative; padding-left:16px; margin:4px 0; color:var(--ink); }
    .card ul li::before { content:"–"; position:absolute; left:0; color:var(--muted); }
    /* One card per recommendation row (Buy/Sell) instead of a spreadsheet-style
       table — no elevation on the individual cards, since they already sit
       inside the section's own .card container. */
    .rec-list { display:flex; flex-direction:column; gap:8px; }
    .rec-card { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:12px 16px; }
    .rec-card-head { display:flex; justify-content:space-between; align-items:baseline; gap:10px; }
    .rec-symbol { font-size:15px; font-weight:600; }
    .rec-reason { color:var(--muted); margin-top:4px; }
    .rec-context { font-size:12.5px; margin-top:6px; }
    /* Client-side "what if" resize of the already-chosen BUY list only — never a
       server round-trip, never re-runs candidate selection, never saves anything.
       See _account_adjust_widget_html() in report_builder.py. */
    .rec-adjust { margin:10px 0 4px; }
    .rec-adjust-toggle, .rec-adjust-btn {
      font-family:var(--sans); font-size:13px; font-weight:600; cursor:pointer;
      border:1px solid var(--border); border-radius:8px; padding:7px 14px;
      background:var(--surface); color:var(--ink); transition:background 0.15s ease;
    }
    .rec-adjust-toggle:hover, .rec-adjust-btn:hover { background:var(--hover-bg); }
    .rec-adjust-btn-secondary { color:var(--ink-2); }
    .rec-adjust-panel { margin-top:10px; padding:14px 16px; border:1px solid var(--border);
      border-radius:10px; background:var(--inset-surface); display:flex; flex-wrap:wrap;
      gap:16px; align-items:flex-end; }
    .rec-adjust-field { display:flex; flex-direction:column; gap:5px; }
    .rec-adjust-field label { font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
    .rec-adjust-field input { font-family:var(--mono); font-size:14px; background:var(--surface);
      border:1px solid var(--border); border-radius:6px; padding:7px 10px; color:var(--ink); width:150px; }
    .rec-adjust-field input:focus-visible { outline:2px solid var(--good-fg); outline-offset:1px; }
    .rec-adjust-actions { display:flex; gap:8px; }
    .rec-adjust-note { margin-top:8px; font-size:12.5px; }
    .rec-item-excluded { opacity:0.5; border-style:dashed; }
    .badge { display:inline-flex; align-items:center; gap:5px; padding:3px 11px; border-radius:999px;
      font-size:13px; font-weight:600; border:1px solid transparent; white-space:nowrap; }
    .badge-good { color:var(--good-fg); background:var(--good-bg); border-color:var(--good-border); }
    .badge-warning { color:var(--warning-fg); background:var(--warning-bg); border-color:var(--warning-border); }
    .badge-serious { color:var(--serious-fg); background:var(--serious-bg); border-color:var(--serious-border); }
    .badge-critical { color:var(--critical-fg); background:var(--critical-bg); border-color:var(--critical-border); }
    .badge-neutral { color:var(--neutral-fg); background:var(--neutral-bg); border-color:var(--neutral-border); }
    /* Symbol quick-jump strip above Symbol Analysis: same-page links styled like
       .badge (same status colors, no new ones) so the strip doubles as an
       at-a-glance overview of the whole set's classifications. */
    .quickjump-label { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; margin-bottom:8px; }
    .quickjump-row { display:flex; flex-wrap:wrap; gap:8px; }
    .symbol-chip { display:inline-flex; align-items:center; padding:3px 11px; border-radius:999px; font-size:13px;
      font-weight:600; font-family:var(--mono); text-decoration:none; border:1px solid transparent; white-space:nowrap; }
    .symbol-chip:hover { text-decoration:underline; }
    .symbol-chip-good { color:var(--good-fg); background:var(--good-bg); border-color:var(--good-border); }
    .symbol-chip-warning { color:var(--warning-fg); background:var(--warning-bg); border-color:var(--warning-border); }
    .symbol-chip-serious { color:var(--serious-fg); background:var(--serious-bg); border-color:var(--serious-border); }
    .symbol-chip-critical { color:var(--critical-fg); background:var(--critical-bg); border-color:var(--critical-border); }
    .symbol-chip-neutral { color:var(--neutral-fg); background:var(--neutral-bg); border-color:var(--neutral-border); }
    /* Score gauge: a thin 0-100 bar so relative strength reads at a glance
       without comparing two numbers by eye. */
    .gauge-row { display:inline-flex; align-items:center; gap:8px; }
    .gauge { position:relative; width:56px; height:5px; border-radius:999px; background:var(--track-bg); overflow:hidden; flex:0 0 auto; }
    /* Compact variant for inside a .stat box: full width of the box, slimmer
       than the old full-size gauge it replaces (see .scores below). */
    .gauge.gauge-stat { width:100%; height:4px; margin-top:7px; }
    .gauge .gauge-fill { position:absolute; top:0; left:0; bottom:0; border-radius:999px; }
    .gauge-good .gauge-fill { background:var(--good-fg); }
    .gauge-warning .gauge-fill { background:var(--warning-fg); }
    .gauge-serious .gauge-fill, .gauge-critical .gauge-fill { background:var(--serious-fg); }
    .gauge-neutral .gauge-fill { background:var(--neutral-fg); }
    /* Signed deltas: color + caret instead of plain +/- text. Risk uses an
       inverted color rule (a drop in risk is good) — see _delta_html(). */
    .delta { font-weight:600; white-space:nowrap; }
    .delta-good { color:var(--good-fg); } .delta-critical { color:var(--critical-fg); } .delta-neutral { color:var(--muted); }
    .chip { display:inline-flex; align-items:center; padding:2px 9px; border-radius:999px; font-size:11.5px;
      color:var(--ink-2); background:var(--chip-bg); border:1px solid var(--border); white-space:nowrap; }
    .stock { border:1px solid var(--border); background:var(--surface); border-radius:10px; padding:20px; margin:20px 0; scroll-margin-top:52px;
      box-shadow:var(--card-shadow); }
    .stock-head { display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }
    .stock-head h3 { margin:0; font-family:var(--mono); }
    .stock-rank-line { margin:2px 0 0; font-size:12px; }
    .scores { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:14px 0; }
    .stat { background:var(--inset-surface); border:1px solid var(--border); border-top:2px solid var(--border); border-radius:8px; padding:12px 14px; }
    .stat.stat-good { border-top-color:var(--good-accent-bar); }
    .stat.stat-warning { border-top-color:var(--warning-accent-bar); }
    .stat.stat-serious, .stat.stat-critical { border-top-color:var(--serious-accent-bar); }
    .stat .stat-label { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }
    .stat .stat-value { font-size:1.5rem; font-weight:600; margin-top:3px; }
    .stat .stat-sub { margin-top:5px; }
    .chart-wrap { overflow-x:auto; }
    .price-chart { width:100%; min-width:660px; height:auto; background:var(--inset-surface); }
    .legend { font-size:12px; font-family:var(--mono); } .muted { color:var(--muted); }
    .lists { display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:14px; }
    .lists section { background:var(--inset-surface); border:1px solid var(--border); border-radius:8px; padding:10px 14px; }
    .lists section h5 { margin:10px 0 2px; font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
    .lists section h5:first-child { margin-top:0; }
    ul { margin-top:6px; padding-left:18px; } ul li { color:var(--ink); } ul li::marker { color:var(--muted); }
    code { overflow-wrap:anywhere; }
    details.raw { margin:14px 0 0; border:1px solid var(--border); border-radius:8px; background:var(--inset-surface); }
    details.raw > summary { cursor:pointer; padding:10px 14px; font-weight:600; color:var(--ink-2);
      list-style:none; user-select:none; }
    details.raw > summary::-webkit-details-marker { display:none; }
    details.raw > summary::before { content:"▸ "; }
    details.raw[open] > summary::before { content:"▾ "; }
    details.raw .raw-body { padding:0 14px 14px; }
    details.raw table { margin:8px 0 14px; }

    /* Footer run-details disclosure: a hairline rule + generous top margin so it
       reads as trailing, optional content set apart from the symbol cards above
       it. Chevron rotates via a plain CSS transform on [open] — no JS. Also
       houses the Methodology note (see _render_phase2_html), folded in here
       rather than given its own always-visible section. */
    details.run-footer { margin-top:52px; padding-top:20px; border-top:1px solid var(--line); scroll-margin-top:52px; }
    details.run-footer > summary { cursor:pointer; list-style:none; user-select:none;
      display:inline-flex; align-items:center; gap:6px; color:var(--muted);
      font-size:12px; text-transform:uppercase; letter-spacing:.06em; }
    details.run-footer > summary::-webkit-details-marker { display:none; }
    details.run-footer > summary .chevron { display:inline-block; transition:transform .2s ease; }
    details.run-footer[open] > summary .chevron { transform:rotate(180deg); }
    details.run-footer .run-footer-body { margin-top:12px; max-width:480px; }
    details.run-footer .run-footer-body h5 { margin:0 0 6px; font-size:11px; text-transform:uppercase;
      letter-spacing:.05em; color:var(--muted); }
    details.run-footer .run-footer-body p { margin:0 0 12px; font-size:13px; }
    details.run-footer table.metadata { margin:0; }

    /* Sticky jump nav — CSS-only "current section" cue via :target on the
       headings themselves (see h2:target above); a real scroll-spy needs JS,
       which this report intentionally limits to the decorative hero only. */
    /* Same elevated-surface language as .card (opaque --surface + hairline border +
       --card-shadow) rather than the old translucent/blurred nav-bg — the bar now
       reads as a distinct floating panel above both the hero and the scrolling body,
       instead of a faint tint that barely separated from either. */
    .term-nav { position:sticky; top:0; z-index:5; display:flex; flex-wrap:wrap; align-items:center;
      justify-content:space-between; gap:8px;
      background:var(--surface); border-bottom:1px solid var(--border); box-shadow:var(--card-shadow);
      padding:8px 24px; }
    .term-nav .term-nav-links { display:flex; flex-wrap:wrap; gap:2px 4px; }
    .term-nav a { color:var(--ink-2); text-decoration:none; font-family:var(--mono); font-size:12px;
      text-transform:uppercase; letter-spacing:.06em; padding:6px 12px; border-radius:6px;
      background:var(--inset-surface); border:1px solid var(--border); }
    .term-nav a:hover { color:var(--ink); background:var(--hover-bg); border-color:var(--ink-2); }
    /* Same-page :target cue as h2:target above, reflected onto the matching nav chip.
       :has() lets the nav (which sits before the headings in the DOM) react to a
       descendant's :target state anywhere in the document — a plain sibling
       combinator can't reach backward like this. Progressive enhancement only: where
       :has() isn't supported, the chips just behave as plain nav links. */
    body:has(#top:target) .term-nav a[href="#top"],
    body:has(#candidates:target) .term-nav a[href="#candidates"],
    body:has(#highest-risk:target) .term-nav a[href="#highest-risk"],
    body:has(#changes:target) .term-nav a[href="#changes"],
    body:has(#symbols:target) .term-nav a[href="#symbols"],
    body:has(#run-details:target) .term-nav a[href="#run-details"] {
      color:var(--good-fg); background:var(--good-bg); border-color:var(--good-border);
    }
    /* Theme toggle: both icons always shown (never just one implying "click to
       switch") — the active mode sits on a raised circle, the inactive one is
       dimmed, so current state is unambiguous without reading anything. */
    .theme-toggle { display:inline-flex; align-items:center; gap:2px; background:var(--toggle-track);
      border:1px solid var(--border); border-radius:999px; padding:3px; flex:0 0 auto; }
    .theme-btn { display:inline-flex; align-items:center; justify-content:center; width:26px; height:26px;
      border-radius:50%; border:none; background:transparent; color:var(--muted); cursor:pointer;
      font-size:13px; line-height:1; padding:0; }
    .theme-btn:hover { color:var(--ink); }
    .theme-btn.is-active { background:var(--surface); color:var(--ink); box-shadow:var(--toggle-shadow); }
    .theme-btn:focus-visible { outline:2px solid var(--good-fg); outline-offset:2px; }
    @media print { .term-nav { display:none; } }

    /* Animated hero banner (decorative only; aria-hidden). Everything runs off one
       shared 60s timeline: a bull dwell (0-25s), a 5s falling transition that carries
       the numbers through zero into negative (25-30s), a bear dwell (30-55s), and a
       5s rising transition back through zero (55-60s) — so the mood change is a
       story, not a hard cut. Independently, the trend line "draws" toward its own
       arrowhead every 1.6s (motion concentrated at the tip, never the whole shape
       moving), and the bars/grid keep a fast ambient pulse/drift underneath it all. */
    /* No background of its own — .market-hero is a direct sibling of .page-fade/
       .page-glow in the body (see markup), appearing after both in DOM order, so it
       already paints above them at the same z-index:auto/0 stacking level without any
       explicit z-index needed. With no opaque background of its own, the shared page
       background (body's dot-grid, plus .page-fade/.page-glow's denser near-the-top
       overlay) shows straight through it. A dedicated hero glow (two stacked
       var(--page-glow) radial-gradient layers) was tried here and reverted twice now —
       the hero already generates its own visual activity (drifting grid, pulsing bars,
       glowing trend line, ticker numbers), and stacking a third white-based wash on
       top of .page-glow's own accent flattened it into a lit grey field instead of a
       calm, dark backdrop those elements read clearly against. Keeping the hero fully
       dark also gives the bold, opaque .term-nav bar sitting right below it (see
       further down) more contrast to stand out against, which is the effect actually
       wanted here. .page-glow is left to do the ambient-light job everywhere it's
       actually needed — the calmer body sections below the hero — rather than
       duplicating it in the one place that benefits from staying dark. */
    .market-hero { position:relative; overflow:hidden; height:240px; }
    .market-hero .grid { position:absolute; inset:0;
      /* Same dot texture as the body (var(--page-grid-a/-b), 28px pitch, two layers
         offset by half a cell) rather than its own cross-hatch pattern, so the hero
         reads as the same visual language as the rest of the page — the drift
         animation and bottom mask-fade below are what still make it feel "alive". */
      background-image:
        radial-gradient(circle, var(--page-grid-a) 1px, transparent 1.5px),
        radial-gradient(circle, var(--page-grid-b) 1px, transparent 1.5px);
      background-size:28px 28px, 28px 28px;
      background-position:0 0, 14px 14px;
      -webkit-mask-image:linear-gradient(to bottom, black, transparent 82%);
      mask-image:linear-gradient(to bottom, black, transparent 82%);
      animation:hero-grid-drift 26s linear infinite; }
    /* Drift distances are multiples of the 28px tile (168=6x28, 140=5x28) so each
       layer loops seamlessly; the second layer keeps its 14px offset constant while
       drifting horizontally, the first drifts vertically — same two-layer parallax
       idea as before, just retuned for the dot tile size instead of the old
       40px/70px cross-hatch repeat sizes. */
    @keyframes hero-grid-drift { 0% { background-position:0 0, 14px 14px; } 100% { background-position:0 168px, 154px 14px; } }
    .market-hero .bars-layer { position:absolute; inset:0; display:flex; align-items:flex-end;
      gap:12px; padding:0 26px 0 26px; }
    .market-hero .bar { flex:0 0 20px; width:20px; border-radius:3px 3px 0 0; transform-origin:bottom;
      animation-name:hero-bar-pulse; animation-duration:7.5s; animation-timing-function:ease-in-out;
      animation-iteration-count:infinite; }
    .market-hero .bars-layer.bull .bar { background:linear-gradient(to top, rgba(20,90,40,0) 0%, rgba(46,222,110,.65) 100%);
      box-shadow:0 0 16px rgba(46,222,110,.35); }
    .market-hero .bars-layer.bear .bar { background:linear-gradient(to top, rgba(90,20,20,0) 0%, rgba(230,70,70,.65) 100%);
      box-shadow:0 0 16px rgba(230,70,70,.35); }
    /* Subtle only — this used to swing to 1.12 on a 4.2s cycle, which read as
       distracting bouncing; a small, slow breathing motion still says "alive"
       without competing with the trend line for attention. */
    @keyframes hero-bar-pulse { 0%,100% { transform:scaleY(1); } 50% { transform:scaleY(1.035); } }
    /* The trend line is the one genuinely procedural piece (see hero script below):
       a real, continuously-extending random-walk series drawn to a canvas, not a
       fixed shape looping through CSS states — that's what an actually-evolving
       line needs, and CSS keyframes fundamentally can't express it. */
    .market-hero .trend-canvas { position:absolute; inset:0; width:100%; height:100%; }
    .market-hero .bars-layer.bull { animation:hero-bull-cycle 60s ease-in-out infinite; }
    .market-hero .bars-layer.bear { animation:hero-bear-cycle 60s ease-in-out infinite; }
    @keyframes hero-bull-cycle { 0%,41.667% { opacity:1; } 50%,91.667% { opacity:0; } 100% { opacity:1; } }
    @keyframes hero-bear-cycle { 0%,41.667% { opacity:0; } 50%,91.667% { opacity:1; } 100% { opacity:0; } }

    /* Ticker numbers: each position is a single element whose text/color the hero
       script (below) rewrites every ~380ms — driven by the same shared value and
       phase timeline as the trend line, so the numbers change constantly and
       roughly track the arrow instead of jumping between a handful of baked
       keyframes every several seconds. Positive/green during the bull dwell,
       negative/red during the bear dwell, sweeping through zero during the two
       transitions — see tickerUpdate() in the script for the actual shape. */
    .market-hero .tick { position:absolute; left:0; top:0; font:600 15px/1 ui-monospace,"SFMono-Regular",Menlo,Consolas,monospace;
      letter-spacing:.02em; white-space:nowrap; }
    .market-hero .tick.tick-pos { color:#5CF08A; text-shadow:0 0 10px rgba(92,240,138,.75); }
    .market-hero .tick.tick-neg { color:#FF6B6B; text-shadow:0 0 10px rgba(255,107,107,.75); }

    /* Light-mode hero: same composition, recolored for a light backdrop. A
       bright neon glow that reads as "alive" against near-black looks washed
       out (bars) or garish (ticker text) on a light surface — light mode
       trades glow for saturation/depth instead. Colors only; the cycle,
       timing, and layout above are untouched. */
    [data-theme="light"] .market-hero .bars-layer.bull .bar {
      background:linear-gradient(to top, rgba(20,90,40,0) 0%, rgba(29,122,58,.82) 100%);
      box-shadow:none;
    }
    [data-theme="light"] .market-hero .bars-layer.bear .bar {
      background:linear-gradient(to top, rgba(110,20,20,0) 0%, rgba(178,42,42,.82) 100%);
      box-shadow:none;
    }
    [data-theme="light"] .market-hero .tick.tick-pos { color:#27500A; text-shadow:none; }
    [data-theme="light"] .market-hero .tick.tick-neg { color:#791F1F; text-shadow:none; }

    @media (prefers-reduced-motion: reduce) {
      .market-hero .grid, .market-hero .bar { animation:none; }
      .market-hero .bars-layer.bull { animation:none; opacity:1; }
      .market-hero .bars-layer.bear { animation:none; opacity:0; }
    }
    /* The hero script itself checks prefers-reduced-motion, skips the ticker
       text updates, and stops redrawing after one frame — this covers only the
       CSS-driven bars/grid. */
    @media print { .market-hero { display:none; } }
"""

_MARKET_HERO_TEMPLATE = """\
  <div class="market-hero" id="top" aria-hidden="true">
    <div class="grid"></div>
    <div class="bars-layer bull">
      <div class="bar" style="height:34px; animation-delay:0.0s"></div><div class="bar" style="height:58px; animation-delay:.3s"></div>
      <div class="bar" style="height:44px; animation-delay:.6s"></div><div class="bar" style="height:72px; animation-delay:.9s"></div>
      <div class="bar" style="height:52px; animation-delay:1.2s"></div><div class="bar" style="height:90px; animation-delay:1.5s"></div>
      <div class="bar" style="height:68px; animation-delay:1.8s"></div><div class="bar" style="height:106px; animation-delay:2.1s"></div>
      <div class="bar" style="height:82px; animation-delay:2.4s"></div><div class="bar" style="height:122px; animation-delay:2.7s"></div>
      <div class="bar" style="height:96px; animation-delay:3.0s"></div><div class="bar" style="height:138px; animation-delay:3.3s"></div>
      <div class="bar" style="height:110px; animation-delay:3.6s"></div><div class="bar" style="height:152px; animation-delay:3.9s"></div>
      <div class="bar" style="height:124px; animation-delay:4.2s"></div><div class="bar" style="height:166px; animation-delay:4.5s"></div>
    </div>
    <div class="bars-layer bear">
      <div class="bar" style="height:34px; animation-delay:0.0s"></div><div class="bar" style="height:58px; animation-delay:.3s"></div>
      <div class="bar" style="height:44px; animation-delay:.6s"></div><div class="bar" style="height:72px; animation-delay:.9s"></div>
      <div class="bar" style="height:52px; animation-delay:1.2s"></div><div class="bar" style="height:90px; animation-delay:1.5s"></div>
      <div class="bar" style="height:68px; animation-delay:1.8s"></div><div class="bar" style="height:106px; animation-delay:2.1s"></div>
      <div class="bar" style="height:82px; animation-delay:2.4s"></div><div class="bar" style="height:122px; animation-delay:2.7s"></div>
      <div class="bar" style="height:96px; animation-delay:3.0s"></div><div class="bar" style="height:138px; animation-delay:3.3s"></div>
      <div class="bar" style="height:110px; animation-delay:3.6s"></div><div class="bar" style="height:152px; animation-delay:3.9s"></div>
      <div class="bar" style="height:124px; animation-delay:4.2s"></div><div class="bar" style="height:166px; animation-delay:4.5s"></div>
    </div>
    <canvas class="trend-canvas" aria-hidden="true"></canvas>
{ticker_html}  </div>
  <script>
  // Decorative only: draws a genuinely evolving random-walk line onto the hero
  // canvas above. Nothing here reads or touches report data — this is the only
  // scripted element anywhere in the document. Silently no-ops if canvas isn't
  // supported, and stops redrawing (leaving one static
  // frame) when the visitor prefers reduced motion.
  (function () {
    try {
      var canvas = document.querySelector(".market-hero .trend-canvas");
      if (!canvas || !canvas.getContext) return;
      var ctx = canvas.getContext("2d");
      var reduceMotion = window.matchMedia &&
        window.matchMedia("(prefers-reduced-motion: reduce)").matches;

      var CYCLE_MS = 60000, BULL_MS = 25000, TRANSITION_MS = 5000;
      // Trail samples carry their own timestamp so x-position is a continuous
      // function of "now" (recomputed every animation frame) instead of a step
      // that only moves once per append — that discretization was the stutter.
      var VISIBLE_MS = 20000, APPEND_EVERY_MS = 220, SAMPLE_EVERY_MS = 70;
      var dpr = window.devicePixelRatio || 1;
      // `targetValue` is the underlying stochastic walk (updated in slow, decisive
      // steps by the leg system below). `displayValue` is the ONE continuously
      // eased quantity actually drawn — every recorded trail sample and the live
      // tip both come from it, so there is never a seam between "history" and
      // "now" the way there was when the tip had its own separate smoothing
      // layered on top of already-smoothed historical points (that mismatch is
      // what caused the line to visibly dip under/over the arrowhead).
      var trail = [];
      var targetValue = 100;
      var startedAt = Date.now();
      var lastTargetUpdate = -Infinity;
      var lastSampleAt = -Infinity;
      var lastFrameAt = 0;
      var displayValue = null;
      var displayMin = null, displayMax = null;
      var displayAngle = null;
      // Dwell phases (the 25s bull/bear stretches) run a chain of ~5s legs, each
      // randomly up or down. Speed (not direction odds) is what makes the phase
      // "generally" trend the right way: a leg that runs WITH the phase's color
      // (up during green, down during red) moves fast; a leg AGAINST it (a dip
      // during green, a relief bounce during red) moves slowly. Direction is a
      // genuine 50/50 coin flip each leg — the fast/slow asymmetry alone is what
      // keeps the net motion trending the phase's way.
      var legDir = 1, legSpeed = 1, legRemainingMs = 0;
      var dwellKind = null;
      // Snapshot of targetValue at the moment the current dwell phase began —
      // the ticker numbers (below) use targetValue-minus-this as their organic
      // "chop", i.e. the exact same leg-driven wobble the trend line is riding,
      // so the numbers roughly track the arrow instead of moving independently.
      var dwellAnchorValue = 0;
      var tickerEls = null;
      var lastTickerUpdate = -Infinity;
      var TICKER_UPDATE_MS = 380;

      function dwellDrift(biasUp) {
        if (legRemainingMs <= 0) {
          legRemainingMs = 4000 + Math.random() * 2400;
          legDir = Math.random() < 0.5 ? 1 : -1;
          var withPhase = (legDir === 1) === biasUp;
          legSpeed = withPhase ? 1.8 + Math.random() * 1.2 : 0.25 + Math.random() * 0.35;
        }
        legRemainingMs -= APPEND_EVERY_MS;
        return legDir * legSpeed;
      }

      // Frame-rate independent easing: converges toward `target` at a fixed
      // half-life regardless of how far apart two draw() calls land in time,
      // so motion looks the same on a 60Hz or 144Hz display.
      function ease(current, target, halfLifeMs, dtMs) {
        if (current === null) return target;
        var rate = 1 - Math.pow(0.5, dtMs / halfLifeMs);
        return current + (target - current) * rate;
      }

      function resize() {
        var rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = Math.max(1, Math.round(rect.width * dpr));
        canvas.height = Math.max(1, Math.round(rect.height * dpr));
      }

      function phaseOf(elapsedMs) {
        var t = elapsedMs % CYCLE_MS;
        if (t < BULL_MS) return { dwell: true, bull: true, stage: "bull" };
        if (t < BULL_MS + TRANSITION_MS) {
          return { dwell: false, drift: -3.2, bull: t < BULL_MS + TRANSITION_MS / 2, stage: "crash" };
        }
        if (t < BULL_MS + TRANSITION_MS + BULL_MS) return { dwell: true, bull: false, stage: "bear" };
        return { dwell: false, drift: 3.2, bull: t > CYCLE_MS - TRANSITION_MS / 2, stage: "recovery" };
      }

      function updateTarget(elapsedMs) {
        var phase = phaseOf(elapsedMs);
        if (phase.dwell) {
          var kindKey = phase.bull ? "bull" : "bear";
          if (dwellKind !== kindKey) {
            dwellKind = kindKey;
            legRemainingMs = 0;
            dwellAnchorValue = targetValue;
          }
        } else {
          dwellKind = null;
        }
        var drift = phase.dwell ? dwellDrift(phase.bull) : phase.drift;
        // Small noise only — legs are the story now, so texture must stay well
        // under even the slow leg's own speed or it'll read as wiggle again.
        targetValue += drift + (Math.random() - 0.5) * 0.35;
      }

      // Below this, the arrow reads as flat/sideways rather than trending —
      // the numbers freeze in place while it's this level, same as a real
      // ticker showing no move when a stock just isn't doing anything.
      var TICKER_FLAT_ANGLE = 0.06;

      function tickerUpdate(elapsedMs) {
        if (!tickerEls || !tickerEls.length) return;
        if (displayAngle !== null && Math.abs(displayAngle) < TICKER_FLAT_ANGLE) return;
        var phase = phaseOf(elapsedMs);
        var t = elapsedMs % CYCLE_MS;
        // The crash/recovery sweep is deterministic from the cycle clock alone
        // (unlike the dwell wobble, which rides the live random walk) — this is
        // what guarantees the numbers genuinely cross zero on schedule, matching
        // the arrow going negative in the red phase.
        var chop = targetValue - dwellAnchorValue;
        for (var i = 0; i < tickerEls.length; i++) {
          var el = tickerEls[i];
          var base = parseFloat(el.getAttribute("data-base"));
          var scale = parseFloat(el.getAttribute("data-scale"));
          var shown;
          if (phase.stage === "crash") {
            var fracC = (t - BULL_MS) / TRANSITION_MS;
            shown = (75 - 165 * fracC) + chop * 0.35 * scale;
          } else if (phase.stage === "recovery") {
            var fracR = (t - (BULL_MS + TRANSITION_MS + BULL_MS)) / TRANSITION_MS;
            shown = (-75 + 165 * fracR) + chop * 0.35 * scale;
          } else if (phase.bull) {
            // Bull dwell: rising chop (an up-leg, arrow trending up) makes the
            // positive number bigger; a down-leg shrinks it back toward zero.
            shown = Math.max(base * 0.4, base + chop * scale);
          } else {
            // Bear dwell: the sign is flipped from the bull case on purpose.
            // The arrow trending UP here means recovering, so the loss should
            // shrink toward zero (less negative) — not grow more negative the
            // way a naive "+chop" would (that was the bug: the number was
            // using the same math as the positive side, so it went the wrong
            // way whenever the arrow ticked up during a red phase). Trending
            // DOWN (further decline) correctly grows the magnitude instead.
            shown = -Math.max(base * 0.4, base - chop * scale);
          }
          var prev = parseFloat(el.getAttribute("data-prev"));
          var up = isNaN(prev) ? shown >= 0 : shown >= prev;
          el.setAttribute("data-prev", shown.toFixed(2));
          el.className = "tick " + (shown >= 0 ? "tick-pos" : "tick-neg");
          el.textContent = shown.toFixed(2) + " " + (up ? "▲" : "▼");
        }
      }

      function draw(elapsedMs, dtMs) {
        var w = canvas.width, h = canvas.height;
        ctx.clearRect(0, 0, w, h);
        if (trail.length < 2) return;

        var rawValues = trail.map(function (p) { return p.v; }).concat([displayValue]);
        var rawMin = Math.min.apply(null, rawValues), rawMax = Math.max.apply(null, rawValues);
        var pad = (rawMax - rawMin) * 0.18 || 6;
        rawMin -= pad; rawMax += pad;
        // Ease the vertical scale itself too — recomputing min/max from the raw
        // window every frame otherwise makes the whole chart "breathe"/rescale
        // as points slide in and out, which reads as unsteady even once the
        // line itself is smooth.
        displayMin = ease(displayMin, rawMin, 600, dtMs);
        displayMax = ease(displayMax, rawMax, 600, dtMs);
        var min = displayMin, max = displayMax;

        var pxPerMs = w / VISIBLE_MS;
        var bull = phaseOf(elapsedMs).bull;
        // Canvas drawing can't read CSS variables, so the theme is checked here
        // directly — cheap enough every frame, and it's how the line repaints
        // correctly the moment someone flips the toggle, no reload needed. Dark
        // keeps its original near-white/glow look; light swaps to deeper,
        // saturated strokes (matching --good-fg/--serious-fg's light-mode hex)
        // with the glow removed, since contrast on white comes from a darker
        // mark, not a bloom.
        var isLightTheme = document.documentElement.getAttribute("data-theme") === "light";
        var color, glow, fillTop, fillBottom, lineShadowBlur;
        if (isLightTheme) {
          color = bull ? "#27500A" : "#791F1F";
          glow = bull ? "rgba(39,80,10,.35)" : "rgba(121,31,31,.35)";
          fillTop = bull ? "rgba(39,80,10,0.18)" : "rgba(121,31,31,0.18)";
          fillBottom = bull ? "rgba(39,80,10,0)" : "rgba(121,31,31,0)";
          lineShadowBlur = 0;
        } else {
          color = bull ? "#eafff0" : "#fff0ee";
          glow = bull ? "rgba(92,240,138,.85)" : "rgba(255,107,107,.85)";
          fillTop = bull ? "rgba(92,240,138,.22)" : "rgba(255,107,107,.22)";
          fillBottom = bull ? "rgba(92,240,138,0)" : "rgba(255,107,107,0)";
          lineShadowBlur = 10 * dpr;
        }
        // The tip sits well clear of the canvas edge — proportional to width so
        // the gap reads consistently at any report/window size — and only its y
        // eases, so the line never touches or snaps against the border.
        var tipX = w - Math.max(70 * dpr, w * 0.09);

        function xAt(p) { return tipX - (elapsedMs - p.t) * pxPerMs; }
        function yAt(v) { return h - ((v - min) / (max - min)) * h; }

        // `trail` is a rolling record of the SAME continuously-eased displayValue
        // drawn as the tip below — not a separate, noisier source — so the last
        // history sample and the tip always sit on the same smooth trajectory.
        var xs = trail.map(xAt).concat([tipX]);
        var ys = trail.map(function (p) { return yAt(p.v); }).concat([yAt(displayValue)]);
        var startIdx = 0;
        while (startIdx < xs.length - 2 && xs[startIdx + 1] < -20 * dpr) startIdx++;

        var lastI = xs.length - 1;
        var tipY = ys[lastI];
        // The arrowhead's angle comes from a point just under half a second
        // back — far enough that one noisy sample can't flip it (the earlier
        // flicker bug), but close enough that it can never meaningfully diverge
        // from the curve actually drawn into the tip. (A longer 1.4s lookback,
        // plus forcing the line's own geometry to bend onto that angle, is what
        // produced the disconnected-looking "two separate pieces" artifact —
        // the forced bend and the real data could point different directions.)
        var lookbackTime = elapsedMs - 420;
        var refIdx = 0;
        for (var li = trail.length - 1; li >= 0; li--) {
          if (trail[li].t <= lookbackTime) { refIdx = li; break; }
        }
        var refX = xs[refIdx], refY = ys[refIdx];
        var targetAngle = Math.atan2(tipY - refY, tipX - refX);
        displayAngle = ease(displayAngle, targetAngle, 320, dtMs);

        function tracePath() {
          ctx.beginPath();
          ctx.moveTo(xs[startIdx], ys[startIdx]);
          // Quadratic-through-midpoints: draws a smooth curve through the data
          // instead of sharp straight segments between every jittery sample.
          for (var i = startIdx + 1; i < xs.length - 1; i++) {
            var midX = (xs[i] + xs[i + 1]) / 2, midY = (ys[i] + ys[i + 1]) / 2;
            ctx.quadraticCurveTo(xs[i], ys[i], midX, midY);
          }
          ctx.lineTo(xs[xs.length - 1], ys[xs.length - 1]);
        }

        ctx.save();
        tracePath();
        ctx.lineTo(xs[xs.length - 1], h);
        ctx.lineTo(xs[startIdx], h);
        ctx.closePath();
        var fillGrad = ctx.createLinearGradient(0, 0, 0, h);
        fillGrad.addColorStop(0, fillTop);
        fillGrad.addColorStop(1, fillBottom);
        ctx.fillStyle = fillGrad;
        ctx.fill();
        ctx.restore();

        ctx.save();
        ctx.strokeStyle = color;
        ctx.shadowColor = glow;
        ctx.shadowBlur = lineShadowBlur;
        ctx.lineWidth = 4 * dpr;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        tracePath();
        ctx.stroke();

        ctx.translate(tipX, tipY);
        ctx.rotate(displayAngle);
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(14 * dpr, 0);
        ctx.lineTo(-8 * dpr, -7 * dpr);
        ctx.lineTo(-8 * dpr, 7 * dpr);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
      }

      function frame() {
        var elapsedMs = Date.now() - startedAt;
        // Backgrounded/inactive tabs throttle or fully pause requestAnimationFrame,
        // but Date.now() keeps advancing — so the first frame after switching back
        // can see a jump of seconds, minutes, or more. Left alone, that huge jump
        // blows most of `trail` past the cutoff below (a real gap that size looks
        // like "all this history is stale, drop it") and leaves the arrow's
        // lookback angle referencing whatever sparse, stale points survive —
        // which is what read as the arrow snapping to point the wrong way. Treat
        // any gap bigger than one background-throttle tick as time that never
        // happened: fold it into startedAt so elapsedMs picks back up exactly
        // where it left off, instead of jumping.
        if (lastFrameAt && elapsedMs - lastFrameAt > 1000) {
          startedAt += elapsedMs - lastFrameAt;
          elapsedMs = lastFrameAt;
        }
        var dtMs = lastFrameAt ? Math.min(elapsedMs - lastFrameAt, 100) : 16.7;
        lastFrameAt = elapsedMs;

        if (elapsedMs - lastTargetUpdate >= APPEND_EVERY_MS) {
          lastTargetUpdate = elapsedMs;
          updateTarget(elapsedMs);
        }
        // The single easing pass: every trail sample recorded below, and the
        // live tip drawn in draw(), both read this same value.
        displayValue = ease(displayValue, targetValue, 180, dtMs);

        if (elapsedMs - lastSampleAt >= SAMPLE_EVERY_MS) {
          lastSampleAt = elapsedMs;
          trail.push({ t: elapsedMs, v: displayValue });
          var cutoff = elapsedMs - VISIBLE_MS - SAMPLE_EVERY_MS * 2;
          while (trail.length > 2 && trail[0].t < cutoff) trail.shift();
        }

        // Reduced motion: leave the server-rendered ticker text alone rather
        // than rewriting it once and freezing on an arbitrary in-between value.
        if (!reduceMotion && elapsedMs - lastTickerUpdate >= TICKER_UPDATE_MS) {
          lastTickerUpdate = elapsedMs;
          tickerUpdate(elapsedMs);
        }

        draw(elapsedMs, dtMs);
        if (!reduceMotion) requestAnimationFrame(frame);
      }

      resize();
      window.addEventListener("resize", resize);
      tickerEls = document.querySelectorAll(".market-hero .tick");
      for (var seed = -Math.ceil(VISIBLE_MS / APPEND_EVERY_MS); seed <= 0; seed++) {
        var seedT = seed * APPEND_EVERY_MS;
        updateTarget(seedT);
        trail.push({ t: seedT, v: targetValue });
      }
      displayValue = targetValue;
      lastTargetUpdate = 0;
      lastSampleAt = 0;
      requestAnimationFrame(frame);
    } catch (err) {
      // Decorative animation only; never let it disrupt the report itself.
    }
  })();
  </script>
"""

# Each position's baseline dwell magnitude and a small per-position multiplier
# applied to the shared organic wobble (see tickerUpdate() in the hero script)
# so the six numbers don't all move in perfect lockstep even though they share
# one underlying timeline. The rendered text below is only the pre-JS static
# fallback (also what's shown under prefers-reduced-motion) — the script
# rewrites it continuously from there.
_HERO_TICKER_POSITIONS: tuple[tuple[str, str, float, float], ...] = (
    ("6%", "30px", 109.25, 1.00),
    ("2%", "96px", 77.61, 0.85),
    ("38%", "18px", 104.46, 1.15),
    ("33%", "132px", 104.61, 0.95),
    ("68%", "70px", 127.83, 1.05),
    ("85%", "112px", 142.10, 0.90),
)


def _hero_ticker_pos_html(left: str, top: str, base: float, scale: float) -> str:
    return (
        f'    <span class="tick tick-pos" style="left:{left}; top:{top};" '
        f'data-base="{base:.2f}" data-scale="{scale:.2f}">{base:.2f} &#9650;</span>\n'
    )


def _market_hero_html() -> str:
    ticker_html = "".join(
        _hero_ticker_pos_html(left, top, base, scale) for left, top, base, scale in _HERO_TICKER_POSITIONS
    )
    # A plain string replace, not str.format(): the template's inline <script>
    # is full of literal JS braces that str.format() would misparse as fields.
    return _MARKET_HERO_TEMPLATE.replace("{ticker_html}", ticker_html)


# Resolves and persists the report's light/dark theme. This is the one other
# scripted element in the document besides the hero animation — like that one,
# it touches only a UI preference (never report data), runs synchronously in
# <head> before <body> paints (so there's no flash of the wrong theme), and is
# wrapped in try/catch so a failure here can never break the report itself.
#
# Default is always dark — deliberately NOT OS-detected via prefers-color-scheme
# — matching the hero banner's mood until the visitor opts into light mode
# themselves. localStorage is what makes that choice stick across later
# reports: each report is written to its own new HTML file (a fresh
# stock_summary_<date>_<hash>.html), so nothing server-side can remember a
# preference between them — the browser's storage for the page's origin is
# the only thing that carries over. In practice that means it reliably
# persists across reports opened the same way in the same browser profile;
# whether that key is shared across separate local files at all depends on
# how that browser partitions storage for file:// pages, which is outside
# what a static report can control.
_THEME_SCRIPT = """\
  <script>
  (function () {
    try {
      var root = document.documentElement;
      var STORAGE_KEY = "stockScrapperReportTheme";

      function storedTheme() {
        try { return window.localStorage.getItem(STORAGE_KEY); } catch (err) { return null; }
      }

      function applyTheme(theme) {
        root.setAttribute("data-theme", theme);
        var buttons = document.querySelectorAll(".theme-btn");
        for (var i = 0; i < buttons.length; i++) {
          var btn = buttons[i];
          var active = btn.getAttribute("data-set-theme") === theme;
          btn.classList.toggle("is-active", active);
          btn.setAttribute("aria-pressed", active ? "true" : "false");
        }
      }

      var initialTheme = storedTheme() || "dark";
      applyTheme(initialTheme);

      document.addEventListener("DOMContentLoaded", function () {
        applyTheme(initialTheme);
        var toggle = document.querySelector(".theme-toggle");
        if (!toggle) return;
        toggle.addEventListener("click", function (event) {
          var btn = event.target.closest ? event.target.closest(".theme-btn") : null;
          if (!btn) return;
          var theme = btn.getAttribute("data-set-theme");
          applyTheme(theme);
          try { window.localStorage.setItem(STORAGE_KEY, theme); } catch (err) {}
        });
      });
    } catch (err) {
      // Decorative preference only; never let it disrupt the report itself.
    }
  })();
  </script>
"""


def _latest_signal_validation_summary(reports_dir: Path) -> dict[str, Any] | None:
    """Best-effort load of the most recent ``signal_validation_*.json`` artifact
    (written by ``main.py validate-signals``) from the same reports directory.

    ``validate-signals`` requires a fresh full-history backtest and is a manual,
    occasional diagnostic (see README's "Evaluation honesty" section) — it is not
    re-run as part of every daily report. This surfaces its latest known result as
    a dated annotation rather than recomputing it, so the daily pipeline stays
    fast. Returns ``None`` (silently — this is supplementary context, not core
    report content) if no such artifact exists yet or it can't be parsed.
    """
    try:
        candidates = sorted(
            Path(reports_dir).glob("signal_validation_*.json"),
            key=lambda path: path.stat().st_mtime,
        )
    except OSError:
        return None
    if not candidates:
        return None
    try:
        return json.loads(candidates[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _signal_validation_notice_html(summary: Mapping[str, Any] | None, classification: str) -> str:
    """A dated callout summarizing one classification bucket's historical forward
    excess return from the latest ``validate-signals`` run, if one is on file.

    Framed deliberately as a descriptive, dataset-wide historical pattern — not a
    live prediction for the specific symbols in today's ranking, and not a
    recommendation. Mirrors the caveats ``render_signal_validation_text`` already
    prints at the CLI (naive/descriptive p-values, concentration warnings).
    """
    if not isinstance(summary, Mapping):
        return ""
    bucket = next(
        (
            item
            for item in summary.get("buckets") or []
            if isinstance(item, Mapping) and item.get("classification") == classification
        ),
        None,
    )
    if not bucket or not bucket.get("sample_size"):
        return ""
    symbol_mean = bucket.get("symbol_mean_excess_return")
    if symbol_mean is None:
        return ""
    ci_low = bucket.get("symbol_mean_excess_return_ci_low")
    ci_high = bucket.get("symbol_mean_excess_return_ci_high")
    ci_text = (
        f"95% CI [{ci_low:+.2%}, {ci_high:+.2%}]"
        if ci_low is not None and ci_high is not None
        else "95% CI unavailable (fewer than 2 distinct symbols)"
    )
    horizon = summary.get("horizon_days")
    benchmark = summary.get("benchmark_symbol")
    run_id = summary.get("backtest_run_id")
    status = "notice-critical" if _CLASSIFICATION_STATUS.get(classification) == "critical" else "notice-good"
    concentration_note = ""
    if bucket.get("concentration_warning"):
        concentration_note = (
            f" Fewer than {bucket.get('distinct_symbols')} distinct symbols back this figure — "
            "treat it as anecdote, not a broad pattern."
        )
    return (
        f'<div class="notice {status}">'
        f"<strong>Historical signal validation ({_escape(classification)}):</strong> mean symbol-weighted "
        f"forward {_escape(str(horizon))}-session excess return vs {_escape(str(benchmark))} was "
        f"{symbol_mean:+.2%}."
        '<details><summary>details</summary><p>'
        f"Across {bucket.get('sample_size')} classified instance(s) ({bucket.get('distinct_symbols')} distinct "
        f"symbol(s)) in the full backtested history ({ci_text})."
        f"{concentration_note} This is a descriptive historical pattern across the whole dataset, from "
        f"<span class=\"mono\">{_escape(str(run_id))}</span> — not a live prediction for the symbols ranked "
        "below, and not a recommendation. See README's \"Evaluation honesty\" section."
        "</p></details>"
        "</div>"
    )


def _latest_recommendations_summary(reports_dir: Path, report_date: str) -> dict[str, Any] | None:
    """Best-effort load of ``recommendations_<report_date>.summary.json`` (written by
    ``main.py recommend``) from the same reports directory, for this exact report date.

    Not recomputed here — ``recommend`` trains predict/predict-v5 from scratch, which
    is comparatively slow, and duplicating that per report would meaningfully slow the
    daily pipeline for no new information. Returns ``None`` (silently — this is
    supplementary context, not core report content) if ``recommend`` hasn't been run
    yet for this date or the file can't be parsed.
    """
    path = Path(reports_dir) / f"recommendations_{report_date}.summary.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _account_adjust_widget_html(
    *,
    account_value: float,
    available_cash: float,
    open_position_count: Any,
    cash_reserve: float | None,
    max_position_weight: float | None,
    max_trade_dollar_amount: float | None,
    min_trade_dollar_amount: float | None,
    has_buys: bool,
) -> str:
    """The account-value/cash subtitle line, plus (when the sizing-rule constants are
    available and there is at least one BUY to resize) a button opening a small panel
    that recalculates each already-chosen BUY's shares/dollars entirely client-side.

    This is a preview only: it never re-runs candidate selection (that needs the full
    scoring/eligibility pipeline this static file doesn't have), and it never persists
    anything — closing the page forgets the edit. Making that change for real requires
    ``main.py account-set``, which is what the note under the panel points to. The rule
    constants are ``None`` for older ``recommendations_<date>.summary.json`` files that
    predate this feature; the widget silently degrades to a plain, non-interactive
    subtitle in that case rather than guessing.
    """
    # A real "·" character throughout (never an &middot;/&nbsp; entity): this same
    # string is later restored verbatim via JS textContent (see data-original below),
    # which does not decode HTML entities — using one plain-text representation
    # everywhere avoids that mismatch entirely rather than requiring two variants.
    open_positions_text = "Unavailable" if open_position_count is None else str(open_position_count)
    summary_line = (
        f"Account value ${account_value:,.2f}  ·  "
        f"Available cash ${available_cash:,.2f}  ·  "
        f"Open positions {open_positions_text}"
    )
    can_resize = (
        has_buys
        and cash_reserve is not None
        and max_position_weight is not None
        and max_trade_dollar_amount is not None
        and min_trade_dollar_amount is not None
    )
    if not can_resize:
        return f'<p class="subtitle">{_escape(summary_line)}</p>'

    subtitle = (
        f'<p class="subtitle" id="rec-summary-line" data-original="{_escape(summary_line)}" '
        f'data-open-positions="{_escape(open_positions_text)}">{_escape(summary_line)}</p>'
    )
    widget = f"""\
  <div class="rec-adjust">
    <button type="button" class="rec-adjust-toggle" id="rec-adjust-toggle" aria-expanded="false" aria-controls="rec-adjust-panel">Adjust account value / cash</button>
    <div class="rec-adjust-panel" id="rec-adjust-panel" hidden>
      <div class="rec-adjust-field">
        <label for="rec-account-value">Account value</label>
        <input type="number" id="rec-account-value" value="{account_value:.2f}" step="0.01" min="0" />
      </div>
      <div class="rec-adjust-field">
        <label for="rec-available-cash">Available cash</label>
        <input type="number" id="rec-available-cash" value="{available_cash:.2f}" step="0.01" min="0" />
      </div>
      <div class="rec-adjust-actions">
        <button type="button" class="rec-adjust-btn" id="rec-adjust-apply">Apply</button>
        <button type="button" class="rec-adjust-btn rec-adjust-btn-secondary" id="rec-adjust-reset">Reset</button>
      </div>
    </div>
    <p class="muted rec-adjust-note" id="rec-adjust-note" hidden>Resized preview only — recalculates today's
    already-chosen candidates, does not change which symbols are recommended, and nothing here is saved.
    Run <span class="mono">python main.py account-set --account-value &lt;amount&gt; --available-cash
    &lt;amount&gt;</span> to persist a change for real.</p>
  </div>
  <script>
  (function () {{
    var accountValueInput = document.getElementById("rec-account-value");
    var availableCashInput = document.getElementById("rec-available-cash");
    var toggle = document.getElementById("rec-adjust-toggle");
    var panel = document.getElementById("rec-adjust-panel");
    var note = document.getElementById("rec-adjust-note");
    var applyBtn = document.getElementById("rec-adjust-apply");
    var resetBtn = document.getElementById("rec-adjust-reset");
    var summaryLine = document.getElementById("rec-summary-line");
    if (!toggle || !panel || !accountValueInput || !availableCashInput || !applyBtn || !resetBtn) return;

    var rules = {{
      cashReserve: {cash_reserve!r},
      maxPositionWeight: {max_position_weight!r},
      maxTradeDollarAmount: {max_trade_dollar_amount!r},
      minTradeDollarAmount: {min_trade_dollar_amount!r}
    }};

    function money(value) {{
      return "$" + value.toLocaleString("en-US", {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
    }}

    function applyResize() {{
      var accountValue = parseFloat(accountValueInput.value);
      var availableCash = parseFloat(availableCashInput.value);
      if (!isFinite(accountValue) || accountValue < 0 || !isFinite(availableCash) || availableCash < 0) return;
      var maxPositionDollars = accountValue * rules.maxPositionWeight;
      var spendable = Math.max(0, availableCash - accountValue * rules.cashReserve);
      var exhausted = false;
      var items = document.querySelectorAll("[data-price]");
      items.forEach(function (item) {{
        var valueSpan = item.querySelector(".rec-sizing-value");
        if (!valueSpan) return;
        var price = parseFloat(item.getAttribute("data-price"));
        item.classList.remove("rec-item-excluded");
        if (exhausted) {{
          item.classList.add("rec-item-excluded");
          valueSpan.textContent = "Not sized — no cash left at this level";
          return;
        }}
        var targetDollars = Math.min(rules.maxTradeDollarAmount, maxPositionDollars, spendable);
        if (targetDollars < rules.minTradeDollarAmount) {{
          item.classList.add("rec-item-excluded");
          var bottleneck;
          if (targetDollars === maxPositionDollars) {{
            bottleneck = Math.round(rules.maxPositionWeight * 100) + "% position cap (" +
              money(maxPositionDollars) + " of " + money(accountValue) + " account value)";
          }} else if (targetDollars === spendable) {{
            bottleneck = "spendable cash (" + money(spendable) + ")";
          }} else {{
            bottleneck = "max_trade_dollar_amount (" + money(rules.maxTradeDollarAmount) + ")";
          }}
          valueSpan.textContent = "Not sized — " + bottleneck + " is below the " +
            money(rules.minTradeDollarAmount) + " minimum trade";
          if (spendable < rules.minTradeDollarAmount) exhausted = true;
          return;
        }}
        var shares = Math.floor(targetDollars / price);
        if (shares < 1) {{
          item.classList.add("rec-item-excluded");
          valueSpan.textContent = "Not sized — price too high for the affordable size";
          return;
        }}
        var estimatedDollars = shares * price;
        valueSpan.textContent = shares + " sh \\u00B7 " + money(estimatedDollars);
        spendable -= estimatedDollars;
      }});
      if (summaryLine) {{
        summaryLine.textContent = "Account value " + money(accountValue) + "  \\u00B7  Available cash " +
          money(availableCash) + "  \\u00B7  Open positions " + summaryLine.getAttribute("data-open-positions");
      }}
      note.hidden = false;
    }}

    function reset() {{
      accountValueInput.value = {account_value!r};
      availableCashInput.value = {available_cash!r};
      document.querySelectorAll("[data-price]").forEach(function (item) {{
        item.classList.remove("rec-item-excluded");
        var valueSpan = item.querySelector(".rec-sizing-value");
        if (valueSpan) {{
          var original = valueSpan.getAttribute("data-original");
          if (original !== null) valueSpan.textContent = original;
        }}
      }});
      if (summaryLine) summaryLine.textContent = summaryLine.getAttribute("data-original");
      note.hidden = true;
    }}

    toggle.addEventListener("click", function () {{
      var expanded = toggle.getAttribute("aria-expanded") === "true";
      toggle.setAttribute("aria-expanded", String(!expanded));
      panel.hidden = expanded;
    }});
    applyBtn.addEventListener("click", applyResize);
    resetBtn.addEventListener("click", reset);
  }})();
  </script>"""
    return subtitle + widget


def _recommendations_section_html(summary: Mapping[str, Any] | None) -> str:
    """Today's sized BUY/SELL suggestions from the latest ``recommend`` run, if one is
    on file for this report date. Advisory only — nothing here has been bought or
    sold, and model context (``predict``/``predict-v5``) is displayed information,
    never a gate on any suggestion (see README's "Phase 6a" section).
    """
    if not isinstance(summary, Mapping):
        return (
            '<p class="muted">No recommendations_&lt;date&gt;.summary.json found for this date — '
            'run <span class="mono">python main.py recommend</span> first.</p>'
        )

    def rows(items: list[Mapping[str, Any]], *, resizable: bool = False) -> str:
        if not items:
            return "<p>None today.</p>"
        cards = []
        for item in items:
            context: list[str] = []
            probability = item.get("model_probability")
            if probability is not None:
                context.append(f"model {probability:.0%} beats benchmark")
            excess_return = item.get("predict_v5_excess_return")
            if excess_return is not None:
                flag = (
                    ' <span class="badge badge-warning">LOW CONFIDENCE</span>'
                    if item.get("predict_v5_low_confidence") else ""
                )
                context.append(f"predict-v5 {excess_return:+.1%}{flag}")
            shares = _finite_number(item.get("shares"))
            dollars = _finite_number(item.get("estimated_dollars"))
            shares_text = "n/a" if shares is None else f"{shares:g}"
            dollars_text = "n/a" if dollars is None else f"${dollars:,.2f}"
            if excess_return is None or excess_return == 0:
                context_status = "neutral"
            else:
                context_status = "good" if excess_return > 0 else "critical"
            context_html = (
                f'<div class="rec-context mono delta-{context_status}">{" &middot; ".join(context)}</div>'
                if context else ""
            )
            # A real "·" (not &middot;) and a data-original attribute: the client-side
            # resize widget restores this exact text via JS textContent on Reset, which
            # does not decode HTML entities, so the displayed and restorable forms must
            # already be identical plain text.
            sizing_text = f"{shares_text} sh · {dollars_text}"
            price_attr = ""
            if resizable and shares is not None and dollars is not None and shares > 0:
                price_attr = f' data-price="{dollars / shares:.6f}"'
            cards.append(
                f'<div class="rec-card"{price_attr}>'
                '<div class="rec-card-head">'
                f'<span class="mono rec-symbol">{_display(item.get("symbol"))}</span>'
                f'<span class="mono rec-sizing-value" data-original="{_escape(sizing_text)}">{_escape(sizing_text)}</span>'
                "</div>"
                f'<div class="rec-reason">{_display(item.get("reason"))}</div>'
                f"{context_html}"
                "</div>"
            )
        return '<div class="rec-list">' + "".join(cards) + "</div>"

    recommendations = summary.get("recommendations") or []
    buys = [item for item in recommendations if isinstance(item, Mapping) and item.get("action") == "BUY"]
    sells = [item for item in recommendations if isinstance(item, Mapping) and item.get("action") == "SELL"]
    skipped = summary.get("skipped") or []
    skipped_html = (
        "<h4>Considered but not recommended</h4><ul>"
        + "".join(f"<li>{_escape(entry)}</li>" for entry in skipped)
        + "</ul>"
        if skipped else ""
    )
    account_value = _finite_number(summary.get("account_value"))
    available_cash = _finite_number(summary.get("available_cash"))
    subtitle = (
        _account_adjust_widget_html(
            account_value=account_value,
            available_cash=available_cash,
            open_position_count=summary.get("open_position_count"),
            cash_reserve=_finite_number(summary.get("cash_reserve")),
            max_position_weight=_finite_number(summary.get("max_position_weight")),
            max_trade_dollar_amount=_finite_number(summary.get("max_trade_dollar_amount")),
            min_trade_dollar_amount=_finite_number(summary.get("min_trade_dollar_amount")),
            has_buys=bool(buys),
        )
        if account_value is not None and available_cash is not None else ""
    )
    return (
        subtitle
        + f'<h3>Buy — {len(buys)}</h3><div class="card">{rows(buys, resizable=True)}</div>'
        + f'<h3>Sell — {len(sells)}</h3><div class="card">{rows(sells)}</div>'
        + skipped_html
    )


def _render_phase2_html(
    report_date: str,
    metadata: dict[str, Any],
    entries: list[dict[str, Any]],
    candidate_order: list[dict[str, Any]],
    risk_order: list[dict[str, Any]],
    quality_issues: list[dict[str, Any]],
    signal_validation: Mapping[str, Any] | None = None,
    recommendations_summary: Mapping[str, Any] | None = None,
) -> str:
    candidate_rank = {str(item.get("symbol", "")).upper(): rank for rank, item in enumerate(candidate_order, 1)}
    risk_rank = {str(item.get("symbol", "")).upper(): rank for rank, item in enumerate(risk_order, 1)}
    regime_reasons = _render_list(metadata.get("market_regime_reasons"), "No regime reasons were recorded.")
    recommendations_html = _recommendations_section_html(recommendations_summary)

    # Report date, As-of date, and Benchmark live in the header subtitle now —
    # this footer only carries the run-identification fields that don't fit
    # there, collapsed by default since they're reference detail, not headline.
    footer_rows = [
        ("Data-through date", metadata.get("data_through_date")),
        ("Analysis run", metadata.get("analysis_run_id") or metadata.get("run_id")),
        ("Scoring version", metadata.get("scoring_version")),
        ("Configuration hash", metadata.get("configuration_hash")),
        ("Generated at", metadata.get("generated_at")),
    ]
    footer_html = "".join(
        f"<tr><th>{_escape(label)}</th><td>{_display(value)}</td></tr>" for label, value in footer_rows
    )

    candidate_html = _ranking_table(candidate_order, candidate_rank, empty_message="No Candidate or Strong Candidate results.")
    risk_html = _ranking_table(risk_order, risk_rank, empty_message="No measured risk scores were available.")
    candidate_validation_html = _signal_validation_notice_html(signal_validation, "Strong Candidate")
    risk_validation_html = _signal_validation_notice_html(signal_validation, "High Risk")
    quickjump_html = _symbol_quickjump_html(entries)
    detail_html = "".join(_result_section(entry) for entry in entries)
    changes_html = _changes_table(entries)
    quality_section_html = (
        f"<h2>Data-Quality Concerns</h2>{_quality_table(quality_issues)}" if quality_issues else ""
    )
    regime_badge = _badge(metadata.get("market_regime"), _REGIME_STATUS)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stock Analyzer — {_escape(report_date)}</title>
  <style>
{_REPORT_STYLES}
  </style>
{_THEME_SCRIPT}
</head>
<body>
  <div class="page-glow" aria-hidden="true"></div>
  <div class="page-fade" aria-hidden="true"></div>
{_market_hero_html()}
  <nav class="term-nav" aria-label="Report sections">
    <div class="term-nav-links">
    <a href="#top">Home</a><a href="#recommendations">Recommendations</a><a href="#candidates">Candidates</a>
    <a href="#highest-risk">Highest risk</a><a href="#changes">Changes</a><a href="#symbols">Symbols</a>
    <a href="#run-details">Run details</a>
    </div>
    <div class="theme-toggle" role="group" aria-label="Theme">
      <button type="button" class="theme-btn" data-set-theme="light" aria-pressed="false" title="Light mode" aria-label="Light mode">&#9728;</button>
      <button type="button" class="theme-btn is-active" data-set-theme="dark" aria-pressed="true" title="Dark mode" aria-label="Dark mode">&#9789;</button>
    </div>
  </nav>
  <div class="page">
  <h1>Stock Analyzer</h1>
  <p class="subtitle">As of {_escape(report_date)} &nbsp;·&nbsp; vs <span class="mono">{_display(metadata.get('benchmark_symbol'))}</span> benchmark</p>
  <div class="notice"><strong>Research disclaimer:</strong> Educational research only; not personalized financial advice. Scores and classifications do not guarantee investment performance.</div>
  <h2 id="market-regime">Market Regime</h2>
  <div class="card regime"><div class="regime-head">{regime_badge}<span class="regime-confidence">confidence {_score(metadata.get('market_regime_confidence'))}</span></div><h4>Market-regime reasons</h4>{regime_reasons}</div>
  <h2 id="recommendations">Today's Recommendations</h2>
  <div class="notice"><strong>Advisory only.</strong> Nothing here has been bought or sold, and model context
  (<span class="mono">predict</span> / <span class="mono">predict-v5</span>) is informational only — never a
  gate on any recommendation. <span class="muted mono">python main.py recommend</span></div>
  {recommendations_html}
  <h2 id="candidates">Candidate Ranking</h2>
  {candidate_validation_html}
  <div class="card">{candidate_html}</div>
  <h2 id="highest-risk">Highest-Risk Ranking</h2>
  {risk_validation_html}
  <div class="card">{risk_html}</div>
  <h2 id="changes">Changes From Previous Stored Analysis</h2>
  <div class="card">{changes_html}</div>
  {quality_section_html}
  <h2 id="symbols">Symbol Analysis</h2>
  {quickjump_html}
  {detail_html or '<p>No symbol results were available.</p>'}
  <details class="run-footer" id="run-details">
  <summary><span class="chevron">&#9662;</span> Run details</summary>
  <div class="run-footer-body">
  <h5>Methodology</h5>
  <p>This report presents deterministic, explainable Phase 2 classifications using data available through the stated as-of date. Opportunity, measured risk, and confidence are separate 0–100 scales. Missing inputs remain unavailable rather than being silently treated as zero.</p>
  <p>Charts use adjusted closing prices supplied to the report and trailing, non-centered 20-, 50-, and 200-session simple moving averages. Rows later than the report/as-of date are excluded from charts. Candidate ranking uses higher opportunity, then higher confidence, lower risk, and symbol as a deterministic tie-breaker. Highest-risk ranking is descending by measured risk.</p>
  <table class="metadata"><tbody>{footer_html}</tbody></table>
  </div>
  </details>
  </div>
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
        classification_status = _CLASSIFICATION_STATUS.get(str(result.get("classification") or ""), "neutral")
        risk_status = _RISK_LEVEL_STATUS.get(str(result.get("risk_level") or ""), "neutral")
        opportunity_cell = (
            f'<span class="gauge-row">{_score(result.get("opportunity_score"))}'
            f'{_gauge_html(result.get("opportunity_score"), classification_status)}</span>'
        )
        risk_cell = (
            f'<span class="gauge-row">{_score(result.get("risk_score"))}'
            f'{_gauge_html(result.get("risk_score"), risk_status)}</span>'
        )
        rows.append(
            f'<tr data-status="{classification_status}">'
            f'<td class="num">{ranks.get(symbol, "")}</td>'
            f'<td><a class="mono" href="#{_symbol_anchor_id(symbol)}">{_escape(symbol)}</a></td>'
            f'<td class="status-{classification_status}">{_display(result.get("classification"))}</td>'
            f'<td class="num">{opportunity_cell}</td>'
            f'<td class="num">{risk_cell}</td>'
            f'<td class="num">{_score(result.get("confidence_score"))}</td></tr>'
        )
    return (
        '<table><thead><tr><th>Rank</th><th>Symbol</th><th>Classification</th>'
        '<th class="num">Opportunity</th><th class="num">Risk</th><th class="num">Confidence</th></tr></thead>'
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def _grouped_lists_html(groups: Sequence[tuple[str, Sequence[tuple[str, Any]]]]) -> str:
    """Renders each group as one card containing only its non-empty sub-items —
    a symbol with nothing to say under a given sub-item shows no placeholder at
    all, and a group with nothing under any of its sub-items renders nothing."""
    sections = []
    for group_title, sub_items in groups:
        parts = [
            f"<h5>{_escape(sub_title)}</h5>" + _render_list(items, "")
            for sub_title, values in sub_items
            if (items := _as_list(values))
        ]
        if parts:
            sections.append(f"<section><h4>{_escape(group_title)}</h4>{''.join(parts)}</section>")
    return "".join(sections)


def _symbol_quickjump_html(entries: list[dict[str, Any]]) -> str:
    """A same-page jump strip above the symbol cards: one status-colored chip per
    entry, linking to that card's existing anchor id. Order matches ``entries``
    as given -- this doesn't re-sort or re-rank anything.
    """
    if not entries:
        return ""
    chips = "".join(
        f'<a class="symbol-chip symbol-chip-{_CLASSIFICATION_STATUS.get(str(entry["result"].get("classification") or ""), "neutral")}" '
        f'href="#{_symbol_anchor_id(str(entry["result"].get("symbol", "")))}">{_escape(entry["result"].get("symbol", ""))}</a>'
        for entry in entries
    )
    return (
        '<div class="card">'
        '<div class="quickjump-label">Jump to symbol</div>'
        f'<div class="quickjump-row">{chips}</div>'
        "</div>"
    )


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
    groups = (
        ("Why", (
            ("Positive factors", result.get("positive_factors")),
            ("Market-regime effects", result.get("market_regime_effects")),
        )),
        ("Watch for", (
            ("Risk factors", result.get("risk_factors")),
            ("Weakening conditions", result.get("weakening_conditions")),
            ("Blocking reasons", result.get("blocking_reasons")),
        )),
        ("Would change this", (
            ("Improvement conditions", result.get("improvement_conditions")),
            ("Confidence limitations", result.get("confidence_limitations")),
            ("Data-quality concerns", entry.get("quality_concerns")),
        )),
    )
    list_html = _grouped_lists_html(groups)
    classification_status = _CLASSIFICATION_STATUS.get(str(result.get("classification") or ""), "neutral")
    risk_status = _RISK_LEVEL_STATUS.get(str(result.get("risk_level") or ""), "neutral")
    classification_badge = _badge(result.get("classification"), _CLASSIFICATION_STATUS)
    risk_level_badge = _badge(result.get("risk_level"), _RISK_LEVEL_STATUS)
    flags_html = "".join(f'<span class="chip">{_escape(flag)}</span>' for flag in _as_list(result.get("flags")))
    return f"""<article class="stock" id="{_symbol_anchor_id(symbol)}">
<div class="stock-head"><h3>{_escape(symbol)}</h3>{classification_badge}{flags_html}</div>
<p class="stock-rank-line muted">rank {_display(entry.get('candidate_rank'))} of candidates &middot; risk rank {_display(entry.get('risk_rank'))}</p>
<p><strong>Primary reason:</strong> {_display(result.get('primary_reason'))}<br />
<strong>Data through:</strong> {_display(result.get('data_through_date'))} &nbsp; <strong>Trend state:</strong> {_display(result.get('trend_state'))}</p>
<div class="scores">
<div class="stat stat-{classification_status}"><div class="stat-label">Opportunity</div><div class="stat-value">{_score(result.get('opportunity_score'))}</div>{_gauge_html(result.get('opportunity_score'), classification_status, stat=True)}</div>
<div class="stat stat-{risk_status}"><div class="stat-label">Measured risk</div><div class="stat-value">{_score(result.get('risk_score'))}</div>{_gauge_html(result.get('risk_score'), risk_status, stat=True)}<div class="stat-sub">{risk_level_badge}</div></div>
<div class="stat"><div class="stat-label">Confidence</div><div class="stat-value">{_score(result.get('confidence_score'))}</div></div>
</div>
<h4>Adjusted Price and Moving Averages</h4>{chart}
<div class="lists">{list_html}</div>
<h4>Changes from previous stored analysis</h4><p>{_display(entry['change'].get('summary'))}</p>
<details class="raw"><summary>Raw scoring data (components &amp; indicators)</summary><div class="raw-body">
{component_html}
</div></details>
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
        f'<td class="mono">{_display(entry["result"].get("symbol"))}</td>'
        f'<td>{_badge((entry.get("previous") or {}).get("classification"), _CLASSIFICATION_STATUS)}</td>'
        f'<td>{_badge(entry["result"].get("classification"), _CLASSIFICATION_STATUS)}</td>'
        f'<td class="num">{_delta_html(entry["change"].get("opportunity_score_change"))}</td>'
        # Risk Δ is inverted: a falling risk score is the favorable direction,
        # so a negative delta reads green here even though it's the same sign
        # that would read red for Opportunity/Confidence.
        f'<td class="num">{_delta_html(entry["change"].get("risk_score_change"), invert=True)}</td>'
        f'<td class="num">{_delta_html(entry["change"].get("confidence_score_change"))}</td>'
        f"<td>{_display(entry['change'].get('summary'))}</td></tr>"
        for entry in entries
    )
    return (
        '<table><thead><tr><th>Symbol</th><th>Previous</th><th>Current</th>'
        '<th class="num">Opportunity Δ</th><th class="num">Risk Δ</th><th class="num">Confidence Δ</th>'
        "<th>Summary</th></tr></thead><tbody>" + rows + "</tbody></table>"
    )


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

    width, height = 900.0, 340.0
    # Right margin is widened (18 -> 90) to fit the end-of-line price chip
    # without crowding the plot.
    left, right, top, bottom = 58.0, 90.0, 24.0, 62.0
    plot_width, plot_height = width - left - right, height - top - bottom
    plot_bottom = top + plot_height
    minimum, maximum = min(available_values), max(available_values)
    padding = (maximum - minimum) * 0.05 if maximum != minimum else max(abs(maximum) * 0.05, 1.0)
    minimum -= padding
    maximum += padding

    def x_position(index: int) -> float:
        return left if len(points) == 1 else left + index * plot_width / (len(points) - 1)

    def y_position(value: float) -> float:
        return top + (maximum - value) * plot_height / (maximum - minimum)

    colors = {
        "adjusted-price": "var(--chart-blue)",
        "sma-20": "var(--chart-orange)",
        "sma-50": "var(--chart-aqua)",
        "sma-200": "var(--chart-yellow)",
    }
    labels = {"adjusted-price": "Adjusted price", "sma-20": "SMA20", "sma-50": "SMA50", "sma-200": "SMA200"}

    # Gridlines live inside the rounded clip; their value labels sit in the
    # left margin, outside it (a clip-path would silently delete anything
    # positioned outside its own shape, including text that's meant to sit
    # to the left of the plot rect).
    grid_lines, grid_labels = [], []
    for tick in range(5):
        fraction = tick / 4
        y = top + fraction * plot_height
        value = maximum - fraction * (maximum - minimum)
        grid_lines.append(f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{width-right:.1f}" y2="{y:.1f}" stroke="var(--line)"/>')
        grid_labels.append(f'<text x="{left-7:.1f}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="var(--ink-2)">{value:.2f}</text>')

    safe_id = _safe_html_id(symbol)
    fill_gradient_id = f"price-fill-{safe_id}"
    clip_id = f"plot-clip-{safe_id}"
    area_fills = []
    for segment in _contiguous_segments(series["adjusted-price"]):
        if len(segment) < 2:
            continue
        coordinates = " ".join(f"{x_position(index):.2f},{y_position(value):.2f}" for index, value in segment)
        first_x, last_x = x_position(segment[0][0]), x_position(segment[-1][0])
        area_fills.append(
            f'<polygon points="{first_x:.2f},{plot_bottom:.2f} {coordinates} {last_x:.2f},{plot_bottom:.2f}" '
            f'fill="url(#{fill_gradient_id})" stroke="none"/>'
        )

    # 52-week-high reference: a dashed, muted line — not a fifth series color,
    # since it isn't data, just a benchmark drawn from data already in hand.
    # Skipped outright for very short histories where "52-week" isn't meaningful.
    valid_prices = [value for value in prices if value is not None]
    fifty_two_week_high = None
    if len(valid_prices) >= 20:
        trailing_valid = [value for value in prices[-252:] if value is not None]
        if trailing_valid:
            fifty_two_week_high = max(trailing_valid)
    reference_line, reference_label = "", ""
    if fifty_two_week_high is not None:
        ref_y = y_position(fifty_two_week_high)
        reference_line = (
            f'<line x1="{left:.1f}" y1="{ref_y:.2f}" x2="{width-right:.1f}" y2="{ref_y:.2f}" '
            f'stroke="var(--ink-2)" stroke-width="1" stroke-dasharray="4 4" opacity="0.7"/>'
        )
        reference_label = (
            f'<text x="{left+8:.1f}" y="{ref_y-6:.2f}" font-size="10.5" fill="var(--muted)">'
            f"52w high &#183; {fifty_two_week_high:.2f}</text>"
        )

    paths = []
    for name, values in series.items():
        for segment in _contiguous_segments(values):
            coordinates = " ".join(f"{x_position(index):.2f},{y_position(value):.2f}" for index, value in segment)
            if len(segment) == 1:
                index, value = segment[0]
                paths.append(f'<circle data-series="{name}" cx="{x_position(index):.2f}" cy="{y_position(value):.2f}" r="2" fill="{colors[name]}"/>')
            else:
                paths.append(f'<polyline data-series="{name}" points="{coordinates}" fill="none" stroke="{colors[name]}" stroke-width="{2.2 if name == "adjusted-price" else 1.7}"/>')

    # End-of-line marker + current-price chip, adjusted-price only. Drawn
    # outside the clip path so neither is cut off if the last point lands
    # near the rounded corner.
    marker, chip = "", ""
    adjusted_segments = _contiguous_segments(series["adjusted-price"])
    if adjusted_segments:
        last_index, last_value = adjusted_segments[-1][-1]
        mx, my = x_position(last_index), y_position(last_value)
        marker = (
            f'<circle cx="{mx:.2f}" cy="{my:.2f}" r="7" fill="var(--inset-surface)"/>'
            f'<circle cx="{mx:.2f}" cy="{my:.2f}" r="4" fill="var(--chart-blue)"/>'
        )
        chip_w, chip_h = 74.0, 22.0
        chip_x, chip_y = mx + 12, my - chip_h / 2
        chip = (
            f'<rect x="{chip_x:.2f}" y="{chip_y:.2f}" width="{chip_w:.1f}" height="{chip_h:.1f}" rx="11" '
            f'fill="var(--chip-bg)" stroke="var(--border)"/>'
            f'<text x="{chip_x+chip_w/2:.2f}" y="{chip_y+chip_h/2+4:.2f}" text-anchor="middle" '
            f'font-family="var(--mono)" font-size="12" fill="var(--ink)">{last_value:.2f}</text>'
        )

    legend = []
    for index, name in enumerate(series):
        x = left + index * 150
        legend.append(f'<line x1="{x:.1f}" y1="{height-12:.1f}" x2="{x+22:.1f}" y2="{height-12:.1f}" stroke="{colors[name]}" stroke-width="3"/><text class="legend" x="{x+28:.1f}" y="{height-8:.1f}" fill="var(--ink-2)">{labels[name]}</text>')

    # Two intermediate date ticks (roughly the 1/3 and 2/3 marks) beyond the
    # existing first/last labels — short marks off the bottom axis, not
    # full-height gridlines, so they read as axis ticks rather than data.
    tick_marks = []
    if len(points) >= 6:
        for fraction in (1 / 3, 2 / 3):
            index = round(fraction * (len(points) - 1))
            tx = x_position(index)
            tick_marks.append(
                f'<line x1="{tx:.2f}" y1="{plot_bottom:.1f}" x2="{tx:.2f}" y2="{plot_bottom+5:.1f}" stroke="var(--line)"/>'
                f'<text x="{tx:.2f}" y="{height-bottom+18:.1f}" text-anchor="middle" font-size="11" fill="var(--ink-2)">{_escape(points[index][0])}</text>'
            )

    first_date, last_date = points[0][0], points[-1][0]
    defs = (
        f'<defs><linearGradient id="{fill_gradient_id}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="var(--chart-blue)" stop-opacity="0.22"/>'
        f'<stop offset="100%" stop-color="var(--chart-blue)" stop-opacity="0"/></linearGradient>'
        f'<clipPath id="{clip_id}"><rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" rx="10"/></clipPath>'
        f'</defs>'
    )
    clipped = (
        f'<g clip-path="url(#{clip_id})">'
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="var(--inset-surface)" stroke="var(--border)"/>'
        f'{"".join(grid_lines)}{"".join(area_fills)}{reference_line}{"".join(paths)}'
        f'</g>'
    )
    return f'''<div class="chart-wrap"><svg class="price-chart" viewBox="0 0 {int(width)} {int(height)}" role="img" aria-labelledby="chart-{_escape(safe_id)}-title"><title id="chart-{_escape(safe_id)}-title">{_escape(symbol)} adjusted price with 20-, 50-, and 200-session moving averages</title>{defs}{clipped}{"".join(grid_labels)}{reference_label}{marker}{chip}<text x="{left:.1f}" y="{height-bottom+18:.1f}" font-size="11" fill="var(--ink-2)">{_escape(first_date)}</text>{"".join(tick_marks)}<text x="{width-right:.1f}" y="{height-bottom+18:.1f}" text-anchor="end" font-size="11" fill="var(--ink-2)">{_escape(last_date)}</text>{''.join(legend)}</svg></div>'''


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


def _badge(value: Any, status_map: Mapping[str, str]) -> str:
    """A status pill: text label plus a color, never color alone (icon+label rule)."""
    text = "" if value is None else str(value)
    status = status_map.get(text, "neutral")
    label = text or "Unavailable"
    return f'<span class="badge badge-{status}">{_escape(label)}</span>'


def _gauge_html(value: Any, status: str, stat: bool = False) -> str:
    """A thin 0-100 bar so relative strength reads without comparing two numbers by eye.

    ``stat=True`` renders the full-width compact variant meant to sit inside a
    ``.stat`` box, directly under the number it illustrates.
    """
    number = _finite_number(value)
    if number is None:
        return ""
    pct = max(0.0, min(100.0, number))
    classes = f"gauge gauge-{status}" + (" gauge-stat" if stat else "")
    return f'<span class="{classes}"><span class="gauge-fill" style="width:{pct:.1f}%"></span></span>'


def _delta_html(value: Any, invert: bool = False) -> str:
    """Signed delta with a caret and color. ``invert`` flips the good/bad read —
    used for Risk Δ, where a drop in risk is the favorable direction."""
    number = _finite_number(value)
    if number is None:
        return '<span class="muted">Unavailable</span>'
    if number == 0:
        status, caret = "neutral", "•"
    else:
        favorable = (number < 0) if invert else (number > 0)
        status = "good" if favorable else "critical"
        caret = "▲" if number > 0 else "▼"
    return f'<span class="delta delta-{status}">{caret} {number:+.2f}</span>'


def _safe_html_id(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", text).strip("-") or "id"


def _symbol_anchor_id(symbol: str) -> str:
    return f"symbol-{_safe_html_id(symbol)}"


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
