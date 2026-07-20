"""Eligibility checks for Phase 2 analysis."""

from __future__ import annotations

from datetime import date, datetime
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
    if not history:
        blocking_reasons.append("No usable price history")
        return False, blocking_reasons, {"history_length": 0}

    latest_row = history[-1]
    latest_date = latest_row.get("trade_date")
    if not latest_date:
        blocking_reasons.append("Missing trade date")
    else:
        try:
            latest_dt = datetime.strptime(str(latest_date), "%Y-%m-%d").date()
            if latest_dt > as_of_date:
                blocking_reasons.append("Latest data is beyond the as-of date")
        except ValueError:
            blocking_reasons.append("Invalid trade date")

    if len(history) < minimum_history_days:
        blocking_reasons.append("Insufficient price history")

    latest_close = latest_row.get("close")
    if latest_close is None:
        blocking_reasons.append("Missing latest close")

    critical_quality = [issue for issue in quality_issues if issue.get("severity") == "critical" and issue.get("symbol") == symbol]
    if critical_quality:
        blocking_reasons.append("Unresolved critical quality issue")

    return not blocking_reasons, blocking_reasons, {"history_length": len(history), "latest_close": latest_close}
