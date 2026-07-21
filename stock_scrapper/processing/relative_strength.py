"""Trading-date-aligned benchmark and relative-strength calculations."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd


def _rows_by_date(history: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Normalize dates and retain the last supplied row for each trading date."""
    normalized: dict[str, dict[str, Any]] = {}
    for row in history:
        raw_date = row.get("trade_date")
        try:
            if isinstance(raw_date, datetime):
                trade_date = raw_date.date().isoformat()
            elif isinstance(raw_date, date):
                trade_date = raw_date.isoformat()
            else:
                trade_date = date.fromisoformat(str(raw_date)[:10]).isoformat()
        except (TypeError, ValueError):
            continue
        copied = dict(row)
        copied["trade_date"] = trade_date
        normalized[trade_date] = copied
    return normalized


def align_series(
    stock_history: list[dict[str, Any]],
    benchmark_history: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return stock and benchmark rows on their common trading dates.

    Rows are copied and date-sorted.  Values are not filled, interpolated, or
    substituted, so a missing adjusted close remains missing in the result.
    """
    stock_by_date = _rows_by_date(stock_history)
    benchmark_by_date = _rows_by_date(benchmark_history)
    common_dates = sorted(set(stock_by_date) & set(benchmark_by_date))
    return (
        [dict(stock_by_date[trade_date]) for trade_date in common_dates],
        [dict(benchmark_by_date[trade_date]) for trade_date in common_dates],
    )


def calculate_relative_strength_metrics(
    stock_history: list[dict[str, Any]],
    benchmark_history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate aligned excess returns, beta, correlation, and RS trend.

    Horizon returns are stock total price return minus benchmark total price
    return over 21, 63, 126, and 252 aligned trading intervals.  The 252-day
    beta is sample covariance divided by sample benchmark variance.  Relative-
    strength trend is the 63-session change in the stock/benchmark price ratio.
    """
    result: dict[str, Any] = {
        "benchmark_relative_return_21": None,
        "benchmark_relative_return_63": None,
        "benchmark_relative_return_126": None,
        "benchmark_relative_return_252": None,
        "beta_252": None,
        "beta": None,
        "correlation_252": None,
        "correlation": None,
        "relative_strength_trend": None,
        "benchmark_aligned_days": 0,
        "benchmark_alignment_ratio": 0.0,
        "benchmark_available": False,
        "benchmark_data_through_date": None,
    }

    stock_rows, benchmark_rows = align_series(stock_history, benchmark_history)
    if not stock_rows:
        return result

    aligned = pd.DataFrame(
        {
            "trade_date": [row["trade_date"] for row in stock_rows],
            "stock_price": pd.to_numeric(
                pd.Series([row.get("adjusted_close") for row in stock_rows]),
                errors="coerce",
            ),
            "benchmark_price": pd.to_numeric(
                pd.Series([row.get("adjusted_close") for row in benchmark_rows]),
                errors="coerce",
            ),
        }
    )
    aligned = aligned.dropna(subset=["stock_price", "benchmark_price"])
    aligned = aligned[
        (aligned["stock_price"] > 0) & (aligned["benchmark_price"] > 0)
    ].reset_index(drop=True)

    stock_valid_dates = {
        trade_date
        for trade_date, row in _rows_by_date(stock_history).items()
        if pd.notna(pd.to_numeric(row.get("adjusted_close"), errors="coerce"))
    }
    aligned_days = int(len(aligned))
    result["benchmark_aligned_days"] = aligned_days
    result["benchmark_alignment_ratio"] = (
        float(aligned_days / len(stock_valid_dates)) if stock_valid_dates else 0.0
    )
    result["benchmark_available"] = aligned_days >= 2
    if aligned_days:
        result["benchmark_data_through_date"] = str(aligned.iloc[-1]["trade_date"])
    if aligned_days < 2:
        return result

    def excess_return(window: int) -> float | None:
        if len(aligned) < window + 1:
            return None
        segment = aligned.tail(window + 1)
        stock_return = float(
            segment.iloc[-1]["stock_price"] / segment.iloc[0]["stock_price"] - 1.0
        )
        benchmark_return = float(
            segment.iloc[-1]["benchmark_price"]
            / segment.iloc[0]["benchmark_price"]
            - 1.0
        )
        return stock_return - benchmark_return

    for window in (21, 63, 126, 252):
        result[f"benchmark_relative_return_{window}"] = excess_return(window)

    ratio = aligned["stock_price"] / aligned["benchmark_price"]
    if len(ratio) >= 64 and ratio.iloc[-64] != 0:
        result["relative_strength_trend"] = float(ratio.iloc[-1] / ratio.iloc[-64] - 1.0)

    stock_returns = aligned["stock_price"].pct_change(fill_method=None)
    benchmark_returns = aligned["benchmark_price"].pct_change(fill_method=None)
    paired_returns = pd.DataFrame(
        {"stock": stock_returns, "benchmark": benchmark_returns}
    ).dropna()
    if len(paired_returns) >= 252:
        recent = paired_returns.tail(252)
        benchmark_variance = float(recent["benchmark"].var(ddof=1))
        if benchmark_variance > 0:
            beta = float(recent["stock"].cov(recent["benchmark"]) / benchmark_variance)
            result["beta_252"] = beta
            result["beta"] = beta
        stock_std = float(recent["stock"].std(ddof=1))
        benchmark_std = float(recent["benchmark"].std(ddof=1))
        if stock_std > 0 and benchmark_std > 0:
            correlation = float(recent["stock"].corr(recent["benchmark"]))
            result["correlation_252"] = correlation
            result["correlation"] = correlation

    return result


# Descriptive alias for callers that treat all outputs as benchmark metrics.
calculate_benchmark_metrics = calculate_relative_strength_metrics
