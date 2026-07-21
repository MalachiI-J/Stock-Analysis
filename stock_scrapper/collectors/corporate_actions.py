"""Explicit yfinance corporate-action normalization and persistence helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any


def action_records(symbol: str, frame: Any, source: str = "yfinance") -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if frame is None or frame.empty: return records
    collected = datetime.now(timezone.utc).isoformat()
    for index, row in frame.iterrows():
        explicit_day = row.get("trade_date")
        day = str(explicit_day)[:10] if explicit_day is not None else (index.date().isoformat() if hasattr(index, "date") else str(index)[:10])
        dividend = row.get("Dividends", row.get("dividends"))
        split = row.get("Stock Splits", row.get("stock_splits"))
        if dividend is not None and float(dividend) != 0:
            records.append({"symbol": symbol.upper(), "action_date": day, "action_type": "dividend", "dividend_amount": float(dividend), "split_ratio": None, "source": source, "collected_at": collected})
        if split is not None and float(split) != 0:
            records.append({"symbol": symbol.upper(), "action_date": day, "action_type": "split", "dividend_amount": None, "split_ratio": float(split), "source": source, "collected_at": collected})
    return records


def upsert_actions(conn: Any, records: list[dict[str, Any]]) -> int:
    changed = 0
    for row in records:
        cursor = conn.execute("""INSERT INTO corporate_actions(symbol,action_date,action_type,dividend_amount,split_ratio,source,collected_at,provider_reference)
          VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(symbol,action_date,action_type,source) DO UPDATE SET
          dividend_amount=excluded.dividend_amount, split_ratio=excluded.split_ratio, collected_at=excluded.collected_at""",
          (row["symbol"],row["action_date"],row["action_type"],row.get("dividend_amount"),row.get("split_ratio"),row["source"],row["collected_at"],row.get("provider_reference")))
        changed += cursor.rowcount
    return changed


def record_action_coverage(conn: Any, symbol: str, source: str, start: str, end: str, records: list[dict[str, Any]], *, status: str = "complete", error: str | None = None) -> None:
    now=datetime.now(timezone.utc).isoformat()
    dates=sorted(str(row["action_date"]) for row in records)
    response_hash=hashlib.sha256(json.dumps(records,sort_keys=True,separators=(",", ":"),default=str).encode()).hexdigest()
    confidence="complete" if status=="complete" else ("partial" if records else "unknown")
    conn.execute("""INSERT INTO corporate_action_coverage(symbol,data_source,requested_start_date,requested_end_date,
      earliest_action_date,latest_action_date,last_successful_collection_time,last_attempted_collection_time,
      collection_status,error_summary,coverage_confidence,response_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
      ON CONFLICT(symbol,data_source) DO UPDATE SET requested_start_date=excluded.requested_start_date,
      requested_end_date=excluded.requested_end_date,earliest_action_date=excluded.earliest_action_date,
      latest_action_date=excluded.latest_action_date,last_successful_collection_time=excluded.last_successful_collection_time,
      last_attempted_collection_time=excluded.last_attempted_collection_time,collection_status=excluded.collection_status,
      error_summary=excluded.error_summary,coverage_confidence=excluded.coverage_confidence,response_hash=excluded.response_hash""",
      (symbol.upper(),source,start,end,dates[0] if dates else None,dates[-1] if dates else None,
       now if status=="complete" else None,now,status,error,confidence,response_hash))
