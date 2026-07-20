"""Data validation helpers for price history rows."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping


def validate_price_record(record: Mapping[str, Any], previous_close: float | None = None, now_date: date | None = None) -> list[dict[str, Any]]:
    """Validate a single price row and return any quality issues found."""
    issues: list[dict[str, Any]] = []
    now = now_date or date.today()

    symbol = str(record.get("symbol", "")).strip()
    trade_date_raw = record.get("trade_date")
    trade_date = None
    if not symbol:
        issues.append(_build_issue(record, "missing_symbol", "critical", "Symbol is missing"))
    if not trade_date_raw:
        issues.append(_build_issue(record, "missing_trade_date", "critical", "Trading date is missing"))
    else:
        try:
            trade_date = datetime.strptime(str(trade_date_raw), "%Y-%m-%d").date()
        except ValueError:
            issues.append(_build_issue(record, "invalid_trade_date", "warning", "Trading date is not in YYYY-MM-DD format"))

    if trade_date is not None:
        if trade_date > now:
            issues.append(_build_issue(record, "future_trade_date", "warning", "Trading date is in the future"))
        if trade_date < now - timedelta(days=3650):
            issues.append(_build_issue(record, "stale_record", "warning", "Trading date is unexpectedly stale"))

    # Basic sanity checks protect the database from obviously bad or inconsistent rows.
    for name in ["open", "high", "low", "close"]:
        value = record.get(name)
        if value is None:
            issues.append(_build_issue(record, name + "_missing", "critical", f"{name} value is missing"))

    if record.get("open") is not None and float(record.get("open")) < 0:
        issues.append(_build_issue(record, "negative_price", "warning", "Open price is negative"))
    if record.get("high") is not None and float(record.get("high")) < 0:
        issues.append(_build_issue(record, "negative_price", "warning", "High price is negative"))
    if record.get("low") is not None and float(record.get("low")) < 0:
        issues.append(_build_issue(record, "negative_price", "warning", "Low price is negative"))
    if record.get("close") is not None and float(record.get("close")) < 0:
        issues.append(_build_issue(record, "negative_price", "warning", "Close price is negative"))

    if record.get("volume") is not None and float(record.get("volume")) < 0:
        issues.append(_build_issue(record, "negative_volume", "warning", "Volume is negative"))

    try:
        open_price = float(record.get("open")) if record.get("open") is not None else None
        high_price = float(record.get("high")) if record.get("high") is not None else None
        low_price = float(record.get("low")) if record.get("low") is not None else None
        close_price = float(record.get("close")) if record.get("close") is not None else None
    except (TypeError, ValueError):
        open_price = high_price = low_price = close_price = None

    if high_price is not None and low_price is not None and high_price < low_price:
        issues.append(_build_issue(record, "high_low_inversion", "warning", "High price is lower than low price"))
    if open_price is not None and high_price is not None and low_price is not None and (open_price < low_price or open_price > high_price):
        issues.append(_build_issue(record, "open_outside_range", "warning", "Open price is outside the reported high-low range"))
    if close_price is not None and high_price is not None and low_price is not None and (close_price < low_price or close_price > high_price):
        issues.append(_build_issue(record, "close_outside_range", "warning", "Close price is outside the reported high-low range"))
    if close_price is not None and close_price == 0:
        issues.append(_build_issue(record, "zero_close_price", "warning", "Close price is zero"))

    if previous_close is not None and previous_close != 0 and close_price is not None:
        change = (close_price - previous_close) / previous_close
        if abs(change) > 0.5:
            issues.append(_build_issue(record, "extreme_price_change", "warning", "One-day price change is unusually large"))

    return issues


def validate_price_records(records: list[Mapping[str, Any]], symbol: str | None = None, now_date: date | None = None) -> list[dict[str, Any]]:
    """Validate a list of records for one symbol."""
    issues: list[dict[str, Any]] = []
    if symbol is None:
        symbol = ""
    sorted_records = sorted(records, key=lambda item: str(item.get("trade_date", "")))

    previous_close: float | None = None
    for record in sorted_records:
        record_with_symbol = dict(record)
        if not record_with_symbol.get("symbol"):
            record_with_symbol["symbol"] = symbol
        issues.extend(validate_price_record(record_with_symbol, previous_close=previous_close, now_date=now_date))
        close_price = record_with_symbol.get("close")
        if close_price is not None:
            try:
                previous_close = float(close_price)
            except (TypeError, ValueError):
                previous_close = None
    return issues


def _build_issue(record: Mapping[str, Any], issue_type: str, severity: str, description: str) -> dict[str, Any]:
    return {
        "symbol": record.get("symbol"),
        "trade_date": record.get("trade_date"),
        "issue_type": issue_type,
        "severity": severity,
        "description": description,
        "detected_time": datetime.now(timezone.utc).isoformat(),
        "resolved_status": 0,
    }
