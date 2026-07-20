"""Relative-strength and benchmark-alignment helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


def align_series(stock_history: list[dict[str, Any]], benchmark_history: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Align two histories by a common trading date and return the filtered rows."""
    stock_frame = pd.DataFrame(stock_history)
    benchmark_frame = pd.DataFrame(benchmark_history)
    if stock_frame.empty or benchmark_frame.empty:
        return stock_history, benchmark_history

    stock_frame["trade_date"] = pd.to_datetime(stock_frame["trade_date"], errors="coerce")
    benchmark_frame["trade_date"] = pd.to_datetime(benchmark_frame["trade_date"], errors="coerce")
    stock_frame = stock_frame.dropna(subset=["trade_date"]).sort_values("trade_date")
    benchmark_frame = benchmark_frame.dropna(subset=["trade_date"]).sort_values("trade_date")

    merged = pd.merge(stock_frame, benchmark_frame, on="trade_date", suffixes=("_stock", "_benchmark"))
    if merged.empty:
        return [], []
    return merged[["trade_date", "close_stock", "adjusted_close_stock", "volume_stock"]].to_dict(orient="records"), merged[["trade_date", "close_benchmark", "adjusted_close_benchmark", "volume_benchmark"]].to_dict(orient="records")
