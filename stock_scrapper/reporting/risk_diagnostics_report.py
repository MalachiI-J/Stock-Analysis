"""Self-contained offline HTML report for the risk-inversion diagnostics study.

Reuses this project's existing report theme (colors, dark/light toggle, fonts) from
``report_builder.py`` rather than introducing a second visual language — only the
diverging quintile bar chart markup/CSS is new, since no existing report needed one.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Mapping

from stock_scrapper.analysis.risk_diagnostics import DriverFeatureStudy, RiskDiagnosticsResult
from stock_scrapper.reporting.report_builder import _REPORT_STYLES, _THEME_SCRIPT

_EXTRA_STYLES = """\
    .diverging-chart { margin:10px 0 16px; }
    .div-row { display:flex; align-items:center; gap:10px; margin:6px 0; }
    .div-label { width:190px; flex:0 0 auto; font-size:12.5px; color:var(--ink-2); }
    .div-track { position:relative; flex:1 1 auto; height:16px; background:var(--track-bg); border-radius:4px; }
    .div-zero { position:absolute; top:-3px; bottom:-3px; left:50%; width:1px; background:var(--border); }
    .div-bar { position:absolute; top:0; bottom:0; border-radius:4px; }
    .div-bar-good { background:var(--good-fg); }
    .div-bar-critical { background:var(--critical-fg); }
    .div-value { width:70px; flex:0 0 auto; text-align:right; font-size:12.5px; }
    .driver-card h4 { margin:0 0 4px; }
    .driver-card .muted { color:var(--ink-2); font-size:12.5px; }
    [data-tooltip] { position:relative; cursor:default; }
    [data-tooltip]:hover::after {
      content:attr(data-tooltip); position:absolute; bottom:calc(100% + 6px); left:50%; transform:translateX(-50%);
      background:var(--surface); color:var(--ink); border:1px solid var(--border); border-radius:6px;
      padding:4px 8px; font-size:11.5px; white-space:nowrap; box-shadow:var(--card-shadow); z-index:5;
    }
"""


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _driver_card_html(study: DriverFeatureStudy) -> str:
    r_text = f"{study.pearson_r:+.3f}" if study.pearson_r is not None else "unavailable"
    if not study.quintiles:
        return (
            f'<div class="card driver-card"><h4>{_escape(study.feature)}</h4>'
            f'<p class="muted">Not enough non-missing rows to bucket.</p></div>'
        )
    max_abs = max(abs(bucket.mean_excess_return) for bucket in study.quintiles) or 1.0
    rows = []
    for bucket in study.quintiles:
        half_width = min(50.0, abs(bucket.mean_excess_return) / max_abs * 50.0)
        is_positive = bucket.mean_excess_return >= 0
        left = 50.0 if is_positive else 50.0 - half_width
        color_class = "div-bar-good" if is_positive else "div-bar-critical"
        tooltip = f"Q{bucket.quintile}: {bucket.mean_excess_return:+.2%} (n={bucket.n})"
        rows.append(
            '<div class="div-row">'
            f'<span class="div-label">Q{bucket.quintile} <span class="mono">[{bucket.feature_low:+.3f}, '
            f'{bucket.feature_high:+.3f}]</span></span>'
            '<div class="div-track"><div class="div-zero"></div>'
            f'<div class="div-bar {color_class}" data-tooltip="{_escape(tooltip)}" '
            f'style="left:{left:.2f}%;width:{half_width:.2f}%"></div></div>'
            f'<span class="div-value mono">{bucket.mean_excess_return:+.2%}</span>'
            "</div>"
        )
    table_rows = "".join(
        f"<tr><td>Q{bucket.quintile}</td><td>{bucket.n}</td>"
        f"<td>[{bucket.feature_low:+.3f}, {bucket.feature_high:+.3f}]</td>"
        f"<td>{bucket.mean_excess_return:+.2%}</td></tr>"
        for bucket in study.quintiles
    )
    return (
        '<div class="card driver-card">'
        f"<h4>{_escape(study.feature)} <span class=\"muted\">(n={study.n}, Pearson r={r_text})</span></h4>"
        f'<div class="diverging-chart" role="img" '
        f'aria-label="Mean forward excess return by {_escape(study.feature)} quintile">'
        f"{''.join(rows)}</div>"
        "<table><thead><tr><th>Quintile</th><th class=\"num\">n</th><th>Feature range</th>"
        '<th class="num">Mean excess return</th></tr></thead>'
        f"<tbody>{table_rows}</tbody></table>"
        "</div>"
    )


def _classification_section_html(result: RiskDiagnosticsResult) -> str:
    anchor = html.escape(result.classification.lower().replace(" ", "-"))
    if not result.driver_studies:
        body = '<p class="muted">No feature had enough non-missing rows to study.</p>'
    else:
        body = "".join(_driver_card_html(study) for study in result.driver_studies)
    return (
        f'<h2 id="{anchor}">"{_escape(result.classification)}" driver-feature study</h2>'
        f'<p class="subtitle">{result.horizon_days}-session forward excess return vs. benchmark &nbsp;·&nbsp; '
        f"sample size {result.sample_size} &nbsp;·&nbsp; distinct symbols {result.distinct_symbols}</p>"
        + body
    )


def render_risk_diagnostics_html(backtest_run_id: str, studies: Mapping[str, RiskDiagnosticsResult]) -> str:
    nav_links = "".join(
        f'<a href="#{html.escape(name.lower().replace(" ", "-"))}">{_escape(name)}</a>'
        for name in studies
    )
    sections = "".join(_classification_section_html(result) for result in studies.values())
    horizon_days = next((result.horizon_days for result in studies.values()), None)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Risk-Inversion Diagnostics — {_escape(backtest_run_id)}</title>
  <style>
{_REPORT_STYLES}
{_EXTRA_STYLES}
  </style>
{_THEME_SCRIPT}
</head>
<body>
  <div class="page-glow" aria-hidden="true"></div>
  <div class="page-fade" aria-hidden="true"></div>
  <nav class="term-nav" aria-label="Report sections">
    <div class="term-nav-links">{nav_links}</div>
    <div class="theme-toggle" role="group" aria-label="Theme">
      <button type="button" class="theme-btn" data-set-theme="light" aria-pressed="false" title="Light mode" aria-label="Light mode">&#9728;</button>
      <button type="button" class="theme-btn is-active" data-set-theme="dark" aria-pressed="true" title="Dark mode" aria-label="Dark mode">&#9789;</button>
    </div>
  </nav>
  <div class="page">
  <h1>Risk-Inversion Diagnostics</h1>
  <p class="subtitle">backtest run <span class="mono">{_escape(backtest_run_id)}</span></p>
  <div class="notice"><strong>Exploratory, not validated:</strong> this studies whether score_v1's forward
  excess return, within one classification bucket, concentrates in a range of an already-computed technical
  indicator. Quintile splits and correlations here share the same overlapping-window, symbol-clustered
  non-independence as <span class="mono">validate-signals</span> — read as descriptive hypothesis
  generation, not a statistically validated result. Nothing here changes score_v1's scoring, thresholds,
  or classification logic.</div>
  {sections}
  <h2>Methodology</h2>
  <p>Each classified (symbol, signal date) instance's already-computed technical indicators are paired with
  its forward excess return over the benchmark ({_escape(horizon_days)} sessions ahead, matching
  <span class="mono">validate-signals</span>/<span class="mono">predict</span>). Rows are split into 5
  equal-sized quintiles by one indicator at a time, ordered from lowest to highest value, and each
  quintile's mean forward excess return is reported alongside a Pearson correlation across the whole
  bucket. A pattern concentrated in the extreme quintiles (with a signed correlation matching) suggests
  that indicator is a driver; a flat pattern across quintiles suggests it is not.</p>
  </div>
</body>
</html>
"""


def write_risk_diagnostics_html_report(
    reports_dir: Path,
    backtest_run_id: str,
    studies: Mapping[str, RiskDiagnosticsResult],
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"risk_inversion_study_{backtest_run_id}.html"
    path.write_text(render_risk_diagnostics_html(backtest_run_id, studies), encoding="utf-8")
    return path
