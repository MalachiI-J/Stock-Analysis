"""Point-in-time fundamental features derived from SEC EDGAR facts.

This is the module that actually enforces lookahead safety for fundamentals data: a
raw fact from ``stock_scrapper/collectors/sec_edgar_fundamentals.py`` is only usable
``as_of`` a date on or after its own ``filed_date``, mirroring the same "bound the
history before anything touches it" discipline price history already gets (see
``historical_features.py``'s ``history_as_of``). Every lookup here filters on
``filed_date <= as_of`` before anything else happens.

Two kinds of XBRL concepts need different point-in-time treatment:

- **Instant concepts** (``assets``, ``liabilities``, ``stockholders_equity``,
  ``shares_outstanding``) are balance-sheet snapshots — the most recently filed value
  known as of a date is simply the current balance.
- **Flow concepts** (``net_income``, ``revenue``, ``eps_diluted``) are period totals.
  Ratios like P/E conventionally use a trailing-twelve-month (TTM) figure, not a
  single quarter, so these are summed over the four most recent single-quarter
  records known as of a date (see ``trailing_four_quarter_sum``).

A known simplifying limitation: TTM summation only counts records whose reporting
duration is roughly one quarter (``_QUARTER_DURATION_DAYS``). Some companies also
tag cumulative year-to-date durations (e.g. a "9 months ended" figure) under the
same concept; those are deliberately excluded rather than double-counted or
misread as a single quarter. When fewer than four qualifying quarters are on file,
the TTM figure — and everything derived from it — comes back ``None`` rather than
an understated partial-year sum. Missing stays unavailable; it is never silently
treated as zero, matching this project's general practice (see e.g.
``report_builder.py``'s methodology paragraph).
"""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Sequence

# Balance-sheet snapshot concepts -- no meaningful "sum the last 4" for these.
_INSTANT_CONCEPTS: tuple[str, ...] = ("assets", "liabilities", "stockholders_equity", "shares_outstanding")

# Period-duration concepts -- trailing-four-quarter-summed for TTM ratios.
_FLOW_CONCEPTS: tuple[str, ...] = ("net_income", "revenue", "eps_diluted")

# A single fiscal quarter is ~90 days; this band is wide enough for real calendar
# variance (28-31 day months) but excludes half-year/9-month cumulative and
# full-year (~365-day) duration facts filed under the same concept name.
_QUARTER_DURATION_DAYS: tuple[int, int] = (80, 105)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None  # excludes NaN/inf


def _quarter_duration_days(record: Mapping[str, Any]) -> int | None:
    start, end = record.get("period_start"), record.get("period_end")
    if not start or not end:
        return None
    try:
        return (date.fromisoformat(str(end)[:10]) - date.fromisoformat(str(start)[:10])).days
    except ValueError:
        return None


def _known_as_of(records: Sequence[Mapping[str, Any]], concept: str, as_of: date) -> list[Mapping[str, Any]]:
    as_of_text = as_of.isoformat()
    return [
        record for record in records
        if record.get("concept") == concept and str(record.get("filed_date", ""))[:10] <= as_of_text
    ]


def _latest_instant_value(records: Sequence[Mapping[str, Any]], concept: str, as_of: date) -> float | None:
    known = _known_as_of(records, concept, as_of)
    if not known:
        return None
    latest = max(known, key=lambda r: (str(r.get("period_end") or ""), str(r.get("filed_date") or "")))
    return _finite(latest.get("value"))


def trailing_four_quarter_sum(records: Sequence[Mapping[str, Any]], concept: str, as_of: date) -> float | None:
    """Sum the four most recent single-quarter records known as of ``as_of``,
    deduplicated by ``period_end`` (a later-filed correction for the same period
    wins). Returns ``None`` if fewer than four qualifying quarters are on file.
    """
    known = _known_as_of(records, concept, as_of)
    quarterly = [record for record in known if _within_quarter_band(record)]
    by_period_end: dict[str, Mapping[str, Any]] = {}
    for record in quarterly:
        period_end = str(record.get("period_end"))
        existing = by_period_end.get(period_end)
        if existing is None or str(record.get("filed_date", "")) > str(existing.get("filed_date", "")):
            by_period_end[period_end] = record
    ordered = sorted(by_period_end.values(), key=lambda r: str(r.get("period_end") or ""))
    if len(ordered) < 4:
        return None
    values = [_finite(record.get("value")) for record in ordered[-4:]]
    if any(value is None for value in values):
        return None
    return sum(values)


def _within_quarter_band(record: Mapping[str, Any]) -> bool:
    days = _quarter_duration_days(record)
    return days is not None and _QUARTER_DURATION_DAYS[0] <= days <= _QUARTER_DURATION_DAYS[1]


def fundamentals_as_of(records: Sequence[Mapping[str, Any]], as_of: date) -> dict[str, float | None]:
    """Point-in-time raw concept values: latest known balance for instant
    concepts, trailing-twelve-month sum for flow concepts. Every value already
    respects ``filed_date <= as_of`` -- nothing filed after ``as_of`` can appear.
    """
    result: dict[str, float | None] = {}
    for concept in _INSTANT_CONCEPTS:
        result[concept] = _latest_instant_value(records, concept, as_of)
    for concept in _FLOW_CONCEPTS:
        result[concept] = trailing_four_quarter_sum(records, concept, as_of)
    return result


def _one_year_before(as_of: date) -> date:
    try:
        return as_of.replace(year=as_of.year - 1)
    except ValueError:  # as_of is Feb 29 and last year wasn't a leap year
        return as_of.replace(month=2, day=28, year=as_of.year - 1)


def trailing_pe(price: float | None, eps_ttm: float | None) -> float | None:
    if price is None or eps_ttm is None or eps_ttm == 0:
        return None
    return price / eps_ttm


def price_to_book(price: float | None, stockholders_equity: float | None, shares_outstanding: float | None) -> float | None:
    if price is None or stockholders_equity is None or not shares_outstanding:
        return None
    book_value_per_share = stockholders_equity / shares_outstanding
    if book_value_per_share == 0:
        return None
    return price / book_value_per_share


def debt_to_equity(liabilities: float | None, stockholders_equity: float | None) -> float | None:
    if liabilities is None or not stockholders_equity:
        return None
    return liabilities / stockholders_equity


def return_on_equity(net_income_ttm: float | None, stockholders_equity: float | None) -> float | None:
    if net_income_ttm is None or not stockholders_equity:
        return None
    return net_income_ttm / stockholders_equity


def _yoy_growth(records: Sequence[Mapping[str, Any]], concept: str, as_of: date) -> float | None:
    current = trailing_four_quarter_sum(records, concept, as_of)
    prior = trailing_four_quarter_sum(records, concept, _one_year_before(as_of))
    if current is None or not prior:
        return None
    return (current - prior) / prior


def revenue_growth_yoy(records: Sequence[Mapping[str, Any]], as_of: date) -> float | None:
    return _yoy_growth(records, "revenue", as_of)


def earnings_growth_yoy(records: Sequence[Mapping[str, Any]], as_of: date) -> float | None:
    return _yoy_growth(records, "net_income", as_of)


def fundamentals_features_as_of(
    records: Sequence[Mapping[str, Any]], as_of: date, *, price: float | None,
) -> dict[str, float | None]:
    """The final, model-ready fundamental feature dict for one (symbol, date) —
    keys match ``predict_v5``'s fundamental ``feature_keys`` exactly, so
    ``dataset.py`` can look values up directly by name.
    """
    raw = fundamentals_as_of(records, as_of)
    return {
        "trailing_pe": trailing_pe(price, raw["eps_diluted"]),
        "price_to_book": price_to_book(price, raw["stockholders_equity"], raw["shares_outstanding"]),
        "debt_to_equity": debt_to_equity(raw["liabilities"], raw["stockholders_equity"]),
        "return_on_equity": return_on_equity(raw["net_income"], raw["stockholders_equity"]),
        "revenue_growth_yoy": revenue_growth_yoy(records, as_of),
        "earnings_growth_yoy": earnings_growth_yoy(records, as_of),
    }
