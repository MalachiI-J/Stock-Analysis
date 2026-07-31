"""SEC EDGAR XBRL fundamentals collector.

Fills the one lever left untried after four negative attempts at a price/volume-only
trading edge (see README's "Evaluation honesty" section): a genuinely new data source.
yfinance's free fundamentals were checked first and ruled out — ``Ticker.quarterly_
income_stmt`` only returns ~5 quarters of history, far short of the ~20-year window this
project's walk-forward validation needs. SEC EDGAR's XBRL ``companyfacts`` API is free,
needs no API key, and its history for a mega-cap like AAPL already spans 2009-2026 —
matching this project's existing backtest window almost exactly.

Every fact SEC returns carries an exact ``filed`` date: the day the value actually
became public. This is a *better* lookahead-safety primitive than price history has ever
needed, and it structurally prevents the single easiest way to accidentally cheat with
fundamentals data (using a number before anyone could have known it). Point-in-time
selection against that ``filed`` date happens downstream, in
``stock_scrapper/processing/fundamentals_features.py`` — this module only collects and
normalizes the raw facts.

This is a plain function-based module (like ``corporate_actions.py``), not a
``BaseCollector`` subclass: EDGAR's ``companyfacts`` endpoint always returns a company's
entire reporting history in one call, so there is no meaningful incremental start/end
window to request — forcing ``BaseCollector``'s ``collect(start_date, end_date,
full_refresh)`` interface here would be a fake fit for a data type that simply doesn't
work that way.

SEC's fair-use policy (not a hard rate limit) asks every request to carry an identifying
``User-Agent`` — configured in ``config/settings.yaml``'s ``edgar.user_agent``, never
hardcoded here.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Mapping

import requests

TICKER_CIK_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_FACTS_URL_TEMPLATE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Canonical concept name -> ordered US-GAAP XBRL tags accepted for it. Companies
# sometimes switch tags over time (most visibly, "Revenues" was largely replaced by
# "RevenueFromContractWithCustomerExcludingAssessedTax" after ASC 606 adoption around
# 2018) — every alias's facts are collected and merged under the one canonical name
# rather than only trusting whichever tag happens to be in use today, so a symbol's
# older history isn't silently dropped.
CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
    "net_income": ("NetIncomeLoss",),
    # "SalesRevenueNet" was the common pre-2018 tag before "Revenues" (itself later
    # largely superseded by "RevenueFromContractWithCustomerExcludingAssessedTax"
    # under ASC 606) -- live-checked against AAPL: without it, revenue coverage
    # only starts in 2018 instead of 2009, cutting nearly a decade of history for
    # every revenue-derived feature (revenue_growth_yoy).
    "revenue": ("SalesRevenueNet", "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"),
    "assets": ("Assets",),
    "liabilities": ("Liabilities",),
    # Large companies with noncontrolling (minority) interests commonly tag only the
    # NCI-inclusive total, not plain "StockholdersEquity" -- live-checked against T,
    # VZ, PG: all three have zero "StockholdersEquity" facts and rely entirely on
    # this alias. Preferring the parent-only tag first keeps "book value per share"
    # meaning what an investor would expect when a company reports both.
    "stockholders_equity": ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "eps_diluted": ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"),
    "shares_outstanding": ("CommonStockSharesOutstanding", "WeightedAverageNumberOfDilutedSharesOutstanding"),
}

# The one XBRL unit each canonical concept is collected in — everything else under that
# tag (e.g. a rare non-USD-denominated duplicate) is ignored rather than mixed in.
CONCEPT_UNITS: dict[str, str] = {
    "net_income": "USD",
    "revenue": "USD",
    "assets": "USD",
    "liabilities": "USD",
    "stockholders_equity": "USD",
    "eps_diluted": "USD/shares",
    "shares_outstanding": "shares",
}

# Only annual/quarterly reports carry the audited-or-reviewed figures this project wants
# to treat as "known fundamentals" — 8-Ks and other filing types are excluded even when
# EDGAR happens to attach a fact to one, and amendments (10-Q/A, 10-K/A) are kept since a
# restatement filed after the original is exactly the kind of later-vintage fact the
# filed-date-bounded lookup in fundamentals_features.py is designed to pick up correctly.
_ACCEPTED_FORM_PREFIXES = ("10-Q", "10-K")


def _request_json(url: str, *, user_agent: str, timeout_seconds: int, max_retries: int, retry_delay_seconds: float) -> Any:
    if not user_agent or not user_agent.strip():
        raise ValueError("A non-empty SEC EDGAR user_agent is required (see config/settings.yaml's edgar.user_agent)")
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout_seconds)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # pragma: no cover - network failure path
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(retry_delay_seconds)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


# SEC's own ticker->CIK file (fetched by fetch_ticker_cik_map below) is occasionally
# wrong for a ticker -- live-checked: it maps "XOM" to CIK 0002115436, "ExxonMobil
# Holdings Corp" (a subsidiary with zero XBRL facts on file), not CIK 0000034088,
# "Exxon Mobil Corporation" (the actual NYSE-listed parent, 438 concepts on file).
# Applied after the fetched mapping so an override always wins over SEC's own data.
_CIK_OVERRIDES: dict[str, str] = {
    "XOM": "0000034088",
}


def fetch_ticker_cik_map(
    *, user_agent: str, timeout_seconds: int = 20, max_retries: int = 3, retry_delay_seconds: float = 2.0,
) -> dict[str, str]:
    """Fetch SEC's ticker->CIK mapping once, covering every symbol in one call.

    Returns ``{TICKER: zero_padded_10_digit_cik}``. Deliberately not cached to disk:
    this is called once per ``collect-fundamentals`` invocation (a manual/periodic
    command, not a per-symbol or daily hot path), so an in-memory single fetch is
    simpler than a disk cache with its own staleness question.
    """
    payload = _request_json(
        TICKER_CIK_MAP_URL, user_agent=user_agent, timeout_seconds=timeout_seconds,
        max_retries=max_retries, retry_delay_seconds=retry_delay_seconds,
    )
    rows = payload.values() if isinstance(payload, Mapping) else payload
    mapping: dict[str, str] = {}
    for row in rows:
        ticker = str(row.get("ticker", "")).upper().strip()
        cik = row.get("cik_str")
        if ticker and cik is not None:
            mapping[ticker] = str(int(cik)).zfill(10)
    mapping.update(_CIK_OVERRIDES)
    return mapping


def fetch_company_facts(
    cik: str, *, user_agent: str, timeout_seconds: int = 20, max_retries: int = 3, retry_delay_seconds: float = 2.0,
) -> dict[str, Any]:
    """Fetch one company's entire XBRL reporting history in a single call."""
    url = COMPANY_FACTS_URL_TEMPLATE.format(cik=str(cik).zfill(10))
    return _request_json(
        url, user_agent=user_agent, timeout_seconds=timeout_seconds,
        max_retries=max_retries, retry_delay_seconds=retry_delay_seconds,
    )


def normalize_company_facts(symbol: str, facts_json: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten EDGAR's nested ``facts.us-gaap.<tag>.units.<unit>[]`` payload into flat
    records for the concepts this project actually uses (see ``CONCEPT_ALIASES``).

    Each record carries its own ``filed_date`` untouched from SEC's data — the raw
    lookahead-safety signal every downstream point-in-time lookup depends on.
    """
    symbol = symbol.upper().strip()
    gaap = (facts_json.get("facts") or {}).get("us-gaap") or {}
    collected_at = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []

    for concept, aliases in CONCEPT_ALIASES.items():
        unit = CONCEPT_UNITS[concept]
        for tag in aliases:
            tag_data = gaap.get(tag)
            if not tag_data:
                continue
            unit_rows = (tag_data.get("units") or {}).get(unit) or []
            for row in unit_rows:
                form = str(row.get("form") or "")
                if not form.startswith(_ACCEPTED_FORM_PREFIXES):
                    continue
                period_end = row.get("end")
                filed_date = row.get("filed")
                value = row.get("val")
                if period_end is None or filed_date is None or value is None:
                    continue
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                records.append(
                    {
                        "symbol": symbol,
                        "concept": concept,
                        "source_tag": tag,
                        "fiscal_year": row.get("fy"),
                        "fiscal_period": row.get("fp"),
                        "form": form,
                        "period_start": (str(row.get("start"))[:10] if row.get("start") else None),
                        "period_end": str(period_end)[:10],
                        "filed_date": str(filed_date)[:10],
                        "value": numeric_value,
                        "unit": unit,
                        "frame": row.get("frame"),
                        "collected_at": collected_at,
                    }
                )
    return records


def upsert_fundamentals(conn: Any, records: list[dict[str, Any]]) -> int:
    """Persist normalized fundamentals records, keyed on the same uniqueness
    constraint as the ``fundamentals`` table (see migration v11). Mirrors
    ``corporate_actions.upsert_actions``'s simpler ``ON CONFLICT DO UPDATE`` style —
    fundamentals don't need ``upsert_price_history``'s revision-audit trail, since a
    later-filed correction lands as its own distinct ``(concept, period_end, form,
    filed_date)`` row rather than silently overwriting what an earlier date's
    backtest actually saw.
    """
    changed = 0
    for row in records:
        cursor = conn.execute(
            """INSERT INTO fundamentals(
                 symbol, concept, source_tag, fiscal_year, fiscal_period, form,
                 period_start, period_end, filed_date, value, unit, frame, collected_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(symbol, concept, period_end, form, filed_date) DO UPDATE SET
                 source_tag=excluded.source_tag, fiscal_year=excluded.fiscal_year,
                 fiscal_period=excluded.fiscal_period, period_start=excluded.period_start,
                 value=excluded.value, unit=excluded.unit, frame=excluded.frame,
                 collected_at=excluded.collected_at""",
            (
                row["symbol"], row["concept"], row["source_tag"], row.get("fiscal_year"),
                row.get("fiscal_period"), row["form"], row.get("period_start"),
                row["period_end"], row["filed_date"], row["value"], row["unit"],
                row.get("frame"), row["collected_at"],
            ),
        )
        changed += cursor.rowcount
    return changed
