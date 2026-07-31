"""Self-contained offline HTML dashboard combining today's digest, sized trade
recommendations, and real portfolio holdings into one page.

Reuses this project's existing report theme (``report_builder.py``) rather than
introducing a second visual language. Unlike the canonical Phase 2 report (every
symbol, every indicator), this is deliberately narrow: the handful of things a user
actually needs to act on today, with a link out to the full report for anyone who
wants the underlying detail.

Recommendations are read from the ``recommend`` command's own saved
``recommendations_<date>.summary.json`` rather than recomputed here, since that
computation trains predict/predict-v5 from scratch and is comparatively slow;
re-running it a second time just to build this page would roughly double the
daily pipeline's cost for no new information.
"""

from __future__ import annotations

import html
from typing import Any, Mapping, Sequence

from stock_scrapper.portfolio import HoldingAssessment
from stock_scrapper.reporting.report_builder import (
    _CLASSIFICATION_STATUS,
    _REGIME_STATUS,
    _REPORT_STYLES,
    _THEME_SCRIPT,
    _badge,
    _score,
)
from stock_scrapper.trading.recommendations import RecommendationRunResult, TradeRecommendation


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _money(value: float | None) -> str:
    return '<span class="muted">Unavailable</span>' if value is None else f"${value:,.2f}"


def _pct(value: float | None) -> str:
    return '<span class="muted">Unavailable</span>' if value is None else f"{value:+.1%}"


def _recommend_rows(recs: Sequence[TradeRecommendation]) -> str:
    if not recs:
        return "<p>None today.</p>"
    rows = []
    for rec in recs:
        context: list[str] = []
        if rec.model_probability is not None:
            context.append(f"model {rec.model_probability:.0%} beats benchmark")
        if rec.predict_v5_excess_return is not None:
            flag = ' <span class="badge badge-warning">LOW CONFIDENCE</span>' if rec.predict_v5_low_confidence else ""
            context.append(f"predict-v5 {rec.predict_v5_excess_return:+.1%}{flag}")
        context_html = " &middot; ".join(context)
        rows.append(
            "<tr>"
            f'<td class="mono">{_escape(rec.symbol)}</td>'
            f'<td class="num">{rec.shares:g}</td>'
            f'<td class="num">{_money(rec.estimated_dollars)}</td>'
            f"<td>{_escape(rec.reason)}</td>"
            f"<td>{context_html}</td>"
            "</tr>"
        )
    return (
        '<table><thead><tr><th>Symbol</th><th class="num">Shares</th><th class="num">Est. $</th>'
        "<th>Reason</th><th>Model context</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _digest_rows(entries: Sequence[Any]) -> str:
    if not entries:
        return "<p>None today.</p>"
    rows = []
    for entry in entries:
        change = (
            f' <span class="muted">(was {_escape(entry.previous_classification)})</span>'
            if entry.changed else ""
        )
        rows.append(
            "<tr>"
            f'<td class="mono">{_escape(entry.symbol)}</td>'
            f"<td>{_badge(entry.classification, _CLASSIFICATION_STATUS)}{change}</td>"
            f'<td class="num">{_score(entry.opportunity_score)}</td>'
            f'<td class="num">{_score(entry.risk_score)}</td>'
            f'<td class="num">{_score(entry.confidence_score)}</td>'
            f"<td>{_escape(entry.primary_reason)}</td>"
            "</tr>"
        )
    return (
        '<table><thead><tr><th>Symbol</th><th>Classification</th><th class="num">Opp</th>'
        '<th class="num">Risk</th><th class="num">Conf</th><th>Reason</th></tr></thead>'
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def _holdings_rows(holdings: Sequence[HoldingAssessment]) -> str:
    if not holdings:
        return "<p>No portfolio positions recorded.</p>"
    rows = []
    for holding in holdings:
        reasons = [reason for reason in (holding.rule_based_exit_reason, holding.price_stop_reason) if reason]
        reason_text = "; ".join(reasons) if reasons else (holding.primary_reason or "")
        recommendation_status = "serious" if holding.recommendation == "SELL" else "good"
        rows.append(
            "<tr>"
            f'<td class="mono">{_escape(holding.symbol)}</td>'
            f'<td class="num">{holding.shares:g}</td>'
            f'<td class="num">{_money(holding.average_cost_basis)}</td>'
            f'<td class="num">{_money(holding.latest_price)}</td>'
            f'<td class="num">{_money(holding.unrealized_pnl)} ({_pct(holding.unrealized_pnl_pct)})</td>'
            f'<td><span class="badge badge-{recommendation_status}">{_escape(holding.recommendation)}</span></td>'
            f"<td>{_escape(reason_text)}</td>"
            "</tr>"
        )
    return (
        '<table><thead><tr><th>Symbol</th><th class="num">Shares</th><th class="num">Avg cost</th>'
        '<th class="num">Latest</th><th class="num">Unrealized</th><th>Signal</th><th>Detail</th></tr></thead>'
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def render_dashboard_html(
    *,
    as_of_date: str,
    market_regime: str,
    market_regime_confidence: float | None,
    digest: Mapping[str, Any],
    recommend: RecommendationRunResult | None,
    phase2_report_href: str | None,
) -> str:
    """Combine today's digest, sized recommendations, and portfolio state into one page.

    ``recommend`` is ``None`` when no ``recommendations_<date>.summary.json`` exists yet
    for this date (e.g. ``recommend`` was never run) — shown as a note, not an error.
    """
    regime_badge = _badge(market_regime, _REGIME_STATUS)
    confidence_text = (
        "" if market_regime_confidence is None else f" &nbsp;&middot;&nbsp; confidence {market_regime_confidence:.1f}"
    )
    report_link = (
        f'<p><a href="{_escape(phase2_report_href)}">Open today&#8217;s full candidate/risk report &rarr;</a></p>'
        if phase2_report_href else ""
    )

    if recommend is None:
        recommend_section = (
            '<p class="muted">No recommendations_&lt;date&gt;.summary.json found for this date yet — '
            "run <span class=\"mono\">python main.py recommend</span> first.</p>"
        )
    else:
        buys = [rec for rec in recommend.recommendations if rec.action == "BUY"]
        sells = [rec for rec in recommend.recommendations if rec.action == "SELL"]
        skipped_html = (
            "<h3>Considered but not recommended</h3><ul>"
            + "".join(f"<li>{_escape(entry)}</li>" for entry in recommend.skipped)
            + "</ul>"
            if recommend.skipped else ""
        )
        recommend_section = (
            f'<p class="subtitle">Account value {_money(recommend.account_value)} &nbsp;&middot;&nbsp; '
            f"Available cash {_money(recommend.available_cash)} &nbsp;&middot;&nbsp; "
            f"Open positions {recommend.open_position_count}</p>"
            f"<h3>Buy — {len(buys)}</h3><div class=\"card\">{_recommend_rows(buys)}</div>"
            f"<h3>Sell — {len(sells)}</h3><div class=\"card\">{_recommend_rows(sells)}</div>"
            f"{skipped_html}"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stock Scrapper Dashboard — {_escape(as_of_date)}</title>
  <style>
{_REPORT_STYLES}
  </style>
{_THEME_SCRIPT}
</head>
<body>
  <div class="page-glow" aria-hidden="true"></div>
  <div class="page-fade" aria-hidden="true"></div>
  <nav class="term-nav" aria-label="Report sections">
    <div class="term-nav-links">
    <a href="#recommend">Recommendations</a><a href="#digest">Digest</a><a href="#holdings">Holdings</a>
    </div>
    <div class="theme-toggle" role="group" aria-label="Theme">
      <button type="button" class="theme-btn" data-set-theme="light" aria-pressed="false" title="Light mode" aria-label="Light mode">&#9728;</button>
      <button type="button" class="theme-btn is-active" data-set-theme="dark" aria-pressed="true" title="Dark mode" aria-label="Dark mode">&#9789;</button>
    </div>
  </nav>
  <div class="page">
  <h1>Stock Scrapper Dashboard</h1>
  <p class="subtitle">As of {_escape(as_of_date)} &nbsp;&middot;&nbsp; regime {regime_badge}{confidence_text}</p>
  {report_link}
  <div class="notice"><strong>Educational research only, not investment advice.</strong> Recommendations are
  sized suggestions only — nothing here has been bought or sold. Model context (<span class="mono">predict</span> /
  <span class="mono">predict-v5</span>) is displayed information only, never a gate on any recommendation.</div>

  <h2 id="recommend">Today's Recommendations</h2>
  {recommend_section}

  <h2 id="digest">Today's Digest</h2>
  <h3>Buy / Strong — {len(digest['buy'])}</h3>
  <div class="card">{_digest_rows(digest['buy'])}</div>
  <h3>Sell / Avoid (if held) — {len(digest['sell'])}</h3>
  <div class="card">{_digest_rows(digest['sell'])}</div>
  <h3>Watch — {len(digest['watch'])}</h3>
  <div class="card">{_digest_rows(digest['watch'])}</div>

  <h2 id="holdings">Your Holdings</h2>
  <div class="card">{_holdings_rows(digest.get('holdings', []))}</div>

  <h2>Research Disclaimer</h2>
  <p>This software is for educational and research use only. It does not provide personalized financial advice
  or recommend trades. Historical analysis does not guarantee future performance. Free market data may be
  delayed, revised, incomplete, or affected by survivorship and static-watchlist bias.</p>
  </div>
</body>
</html>
"""
