"""Eligibility checks for Phase 2 analysis."""

from __future__ import annotations

from datetime import date, datetime
from math import isfinite
from typing import Any


def evaluate_eligibility(
    symbol: str,
    history: list[dict[str, Any]],
    quality_issues: list[dict[str, Any]],
    as_of_date: date,
    minimum_history_days: int,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Return whether a symbol is eligible for scoring and the blocking reasons."""
    blocking_reasons: list[str] = []
    metadata: dict[str, Any] = {
        "history_length": len(history),
        "critical_data_issue": False,
        "insufficient_history": False,
        "invalid_data": False,
    }
    if not history:
        blocking_reasons.append("No usable price history")
        metadata["insufficient_history"] = True
        return False, blocking_reasons, metadata

    latest_row = history[-1]
    latest_date = latest_row.get("trade_date")
    if not latest_date:
        blocking_reasons.append("Missing trade date")
    else:
        try:
            latest_dt = datetime.strptime(str(latest_date), "%Y-%m-%d").date()
            if latest_dt > as_of_date:
                blocking_reasons.append("Latest data is beyond the as-of date")
                metadata["invalid_data"] = True
        except ValueError:
            blocking_reasons.append("Invalid trade date")
            metadata["invalid_data"] = True

    if len(history) < minimum_history_days:
        blocking_reasons.append("Insufficient price history")
        metadata["insufficient_history"] = True

    latest_close = latest_row.get("adjusted_close")
    try:
        numeric_latest_close = float(latest_close)
    except (TypeError, ValueError):
        numeric_latest_close = float("nan")
    if not isfinite(numeric_latest_close) or numeric_latest_close <= 0:
        blocking_reasons.append("Missing latest adjusted close")
        metadata["invalid_data"] = True

    critical_quality = [
        issue
        for issue in quality_issues
        if str(issue.get("severity", "")).lower() == "critical"
        and str(issue.get("symbol", "")).upper() == symbol.upper()
    ]
    if critical_quality:
        blocking_reasons.append("Unresolved critical quality issue")
        metadata["critical_data_issue"] = True

    metadata["latest_close"] = (
        numeric_latest_close if isfinite(numeric_latest_close) and numeric_latest_close > 0 else None
    )
    return not blocking_reasons, blocking_reasons, metadata
