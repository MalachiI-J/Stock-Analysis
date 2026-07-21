"""Offline market-data integrity summary."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

from stock_scrapper.market_calendar import SessionResolver


def assess_data_health(conn: Any, symbols: Sequence[str], provider_delay_minutes: int = 30) -> dict[str, Any]:
    resolver = SessionResolver(provider_delay_minutes)
    expected = resolver.previous_completed_session()
    details = []
    overall = "Healthy"
    for symbol in symbols:
        row = conn.execute("""SELECT COUNT(*) rows, MAX(trade_date) latest,
          SUM(CASE WHEN is_complete=0 THEN 1 ELSE 0 END) incomplete,
          SUM(CASE WHEN open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL OR volume IS NULL THEN 1 ELSE 0 END) null_ohlcv,
          SUM(CASE WHEN adjusted_close IS NULL THEN 1 ELSE 0 END) null_adjusted,
          SUM(CASE WHEN high < MAX(open,close,low) OR low > MIN(open,close,high) THEN 1 ELSE 0 END) invalid_ohlc
          FROM price_history WHERE symbol=?""", (symbol,)).fetchone()
        revisions = conn.execute("""SELECT COUNT(*) total,
          SUM(CASE WHEN revision_class='precision_noise' THEN 1 ELSE 0 END) precision_noise,
          SUM(CASE WHEN is_material=1 THEN 1 ELSE 0 END) material,
          SUM(CASE WHEN revision_class='material_price_revision' THEN 1 ELSE 0 END) unexplained_material,
          SUM(CASE WHEN revision_class='corporate_action_revision' THEN 1 ELSE 0 END) corporate,
          SUM(CASE WHEN review_status='unreviewed' THEN 1 ELSE 0 END) unreviewed
          FROM price_history_revisions WHERE symbol=?""", (symbol,)).fetchone()
        actions = conn.execute("SELECT COUNT(*) FROM corporate_actions WHERE symbol=?", (symbol,)).fetchone()[0]
        dates=[str(r[0]) for r in conn.execute("SELECT trade_date FROM price_history WHERE symbol=? AND is_complete=1 ORDER BY trade_date",(symbol,))]
        expected_dates={d.isoformat() for d in resolver.sessions_between(dates[0],expected)} if dates else set()
        actual_dates=set(dates); missing=sorted(expected_dates-actual_dates); extra=sorted(actual_dates-expected_dates)
        coverage=conn.execute("SELECT * FROM corporate_action_coverage WHERE symbol=? AND data_source='yfinance'",(symbol,)).fetchone()
        coverage_status=(coverage["collection_status"] if coverage else "unknown")
        unresolved=conn.execute("SELECT COUNT(*) FROM data_quality_issues WHERE symbol=? AND resolved_status=0",(symbol,)).fetchone()[0]
        factor_anomalies=conn.execute("""SELECT COUNT(*) FROM price_history WHERE symbol=? AND
          (adjusted_close IS NULL OR close IS NULL OR close<=0 OR adjusted_close<=0 OR adjusted_close/close<0.01 OR adjusted_close/close>100)""",(symbol,)).fetchone()[0]
        last_refresh=conn.execute("SELECT MAX(last_collected_at) FROM price_history WHERE symbol=?",(symbol,)).fetchone()[0]
        latest = row["latest"]
        stale = latest is None or latest < expected.isoformat()
        recent_missing=[d for d in missing if d >= resolver.overlap_start(expected,min(5,len(expected_dates) or 1)).isoformat()]
        status = "Critical" if not row["rows"] or row["null_ohlcv"] or row["invalid_ohlc"] or len(recent_missing)>1 else ("Warning" if stale or recent_missing or coverage_status != "complete" or factor_anomalies or revisions["unexplained_material"] else "Healthy")
        if status == "Critical": overall = "Critical"
        elif status == "Warning" and overall == "Healthy": overall = "Warning"
        details.append({"symbol": symbol, "status": status, "rows": row["rows"], "latest_stored_session": latest,
                        "last_completed_session": expected.isoformat(), "incomplete_rows": row["incomplete"],
                        "null_ohlcv": row["null_ohlcv"], "invalid_ohlc": row["invalid_ohlc"],
                        "null_adjusted_close": row["null_adjusted"], "revision_differences": revisions["total"],
                        "precision_noise_revisions": revisions["precision_noise"] or 0, "material_revisions": revisions["material"] or 0,
                        "unexplained_material_revisions": revisions["unexplained_material"] or 0,
                        "corporate_action_revisions": revisions["corporate"] or 0, "unreviewed_revisions": revisions["unreviewed"] or 0,
                        "corporate_actions": actions, "corporate_action_coverage": coverage_status,
                        "missing_expected_sessions": missing, "extra_non_session_dates": extra, "duplicate_rows": 0,
                        "adjustment_factor_anomalies": factor_anomalies, "unresolved_quality_issues": unresolved,
                        "earliest_valid_date": dates[0] if dates else None, "latest_valid_date": dates[-1] if dates else None,
                        "complete_bars": len(dates), "last_provider_refresh": last_refresh, "stale": stale})
    return {"status": overall, "checked_at": datetime.now(timezone.utc).isoformat(), "last_completed_session": expected.isoformat(), "symbols": details}
