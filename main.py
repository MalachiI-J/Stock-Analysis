"""Command-line entry point for Stock Scrapper."""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from stock_scrapper.collectors.yahoo_prices import YahooPriceCollector
from stock_scrapper.config import load_config, load_watchlist
from stock_scrapper.database import (
    create_connection,
    fetch_price_history,
    get_latest_trade_date,
    initialize_database,
    insert_collection_run,
    record_quality_issue,
    upsert_price_history,
)
from stock_scrapper.processing.indicators import calculate_indicators, classify_status
from stock_scrapper.processing.validation import validate_price_records
from stock_scrapper.reporting.report_builder import write_csv_report, write_html_report
from stock_scrapper.utilities.logging_setup import setup_logging


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Stock Scrapper Phase 1")
    subparsers = parser.add_subparsers(dest="command", required=True)

    update_parser = subparsers.add_parser("update", help="Collect and store missing market data")
    update_parser.add_argument("--symbols", nargs="+", help="Optional symbol list")
    update_parser.add_argument("--full-refresh", action="store_true", help="Refresh from the full historical lookback period")

    report_parser = subparsers.add_parser("report", help="Generate reports using stored data")
    report_parser.add_argument("--symbols", nargs="+", help="Optional symbol list")
    report_parser.add_argument("--date", help="Optional report date (YYYY-MM-DD)")

    run_parser = subparsers.add_parser("run", help="Run an end-to-end update and report cycle")
    run_parser.add_argument("--symbols", nargs="+", help="Optional symbol list")
    run_parser.add_argument("--full-refresh", action="store_true", help="Refresh from the full historical lookback period")

    subparsers.add_parser("validate", help="Run database-wide validation checks")
    subparsers.add_parser("status", help="Show project and database status")
    return parser


def ensure_directories(config: dict[str, Any]) -> None:
    """Create all required directories from the configuration."""
    for directory_name in [
        config["raw_data_dir"],
        config["processed_data_dir"],
        config["reports_dir"],
        config["logs_dir"],
    ]:
        Path(directory_name).mkdir(parents=True, exist_ok=True)


def update_symbols(config: dict[str, Any], logger: Any, symbols: list[str], full_refresh: bool = False) -> tuple[list[str], list[str], int, int]:
    """Collect missing data and write it to SQLite."""
    # This loop intentionally isolates one symbol from the rest so a single network issue does not stop the whole run.
    initialize_database(config["database_path"])
    conn = create_connection(config["database_path"])
    collector = YahooPriceCollector(max_retries=int(config.get("retry_count", 3)), retry_delay_seconds=float(config.get("retry_delay_seconds", 2)))

    successful_symbols: list[str] = []
    failed_symbols: list[str] = []
    inserted_count = 0
    updated_count = 0
    run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"
    start_time = datetime.now(timezone.utc).isoformat()

    try:
        for symbol in symbols:
            logger.info("Starting collection for %s", symbol)
            try:
                start_date: date | None = None
                if not full_refresh:
                    latest_date = get_latest_trade_date(conn, symbol)
                    if latest_date:
                        start_date = datetime.strptime(latest_date, "%Y-%m-%d").date() + timedelta(days=1)

                frame = collector.collect(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=date.today(),
                    full_refresh=full_refresh,
                )
                if frame.empty:
                    logger.info("No new rows returned for %s", symbol)
                    successful_symbols.append(symbol)
                    continue

                rows = frame.to_dict(orient="records")
                issues = validate_price_records(rows, symbol=symbol, now_date=date.today())
                conn.execute("BEGIN")
                try:
                    for row in rows:
                        inserted, updated = upsert_price_history(conn, row)
                        inserted_count += inserted
                        updated_count += updated
                    for issue in issues:
                        record_quality_issue(conn, issue)
                    conn.commit()
                except Exception:  # pragma: no cover - rollback path
                    conn.rollback()
                    raise

                successful_symbols.append(symbol)
                logger.info("Stored %s rows for %s", len(rows), symbol)
            except Exception as exc:  # pragma: no cover - error handling path
                failed_symbols.append(symbol)
                logger.exception("Failed to collect %s: %s", symbol, exc)
    except Exception as exc:  # pragma: no cover - fatal path
        logger.exception("Collection run failed: %s", exc)
        insert_collection_run(
            conn,
            {
                "run_id": run_id,
                "start_time": start_time,
                "end_time": datetime.now(timezone.utc).isoformat(),
                "status": "failed",
                "symbols_requested": ",".join(symbols),
                "symbols_updated": ",".join(successful_symbols),
                "symbols_failed": ",".join(failed_symbols),
                "records_inserted": inserted_count,
                "records_updated": updated_count,
                "error_summary": str(exc),
            },
        )
        conn.commit()
    else:
        insert_collection_run(
            conn,
            {
                "run_id": run_id,
                "start_time": start_time,
                "end_time": datetime.now(timezone.utc).isoformat(),
                "status": "completed_with_errors" if failed_symbols else "completed",
                "symbols_requested": ",".join(symbols),
                "symbols_updated": ",".join(successful_symbols),
                "symbols_failed": ",".join(failed_symbols),
                "records_inserted": inserted_count,
                "records_updated": updated_count,
                "error_summary": "",
            },
        )
        conn.commit()
    finally:
        conn.close()

    return successful_symbols, failed_symbols, inserted_count, updated_count


def validate_database(config: dict[str, Any], logger: Any) -> list[dict[str, Any]]:
    """Validate every stored record and store issues in the database."""
    initialize_database(config["database_path"])
    conn = create_connection(config["database_path"])
    issues: list[dict[str, Any]] = []
    try:
        symbols = load_watchlist(config["watchlist_path"])
        for symbol in symbols:
            history = fetch_price_history(conn, symbol)
            if not history:
                continue
            batch_issues = validate_price_records(history, symbol=symbol, now_date=date.today())
            for issue in batch_issues:
                record_quality_issue(conn, issue)
                issues.append(issue)
        conn.commit()
    finally:
        conn.close()
    return issues


def build_reports(
    config: dict[str, Any],
    logger: Any,
    symbols: list[str],
    report_date: str | None = None,
    successful_symbols: list[str] | None = None,
    failed_symbols: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Generate CSV and HTML reports from the stored history."""
    initialize_database(config["database_path"])
    conn = create_connection(config["database_path"])
    report_rows: list[dict[str, Any]] = []
    quality_issues: list[dict[str, Any]] = []
    try:
        cursor = conn.execute("SELECT symbol, trade_date, issue_type, severity, description FROM data_quality_issues ORDER BY detected_time DESC")
        quality_issues = [dict(row) for row in cursor.fetchall()]
        for symbol in symbols:
            history = fetch_price_history(conn, symbol)
            history_rows = [
                {
                    "trade_date": row.get("trade_date"),
                    "close": row.get("close"),
                    "adjusted_close": row.get("adjusted_close"),
                    "volume": row.get("volume"),
                }
                for row in history
            ]
            metrics = calculate_indicators(history_rows, symbol)
            status, flags = classify_status(metrics, has_quality_warning=any(issue.get("symbol") == symbol for issue in quality_issues))
            metrics["symbol"] = symbol
            metrics["status"] = status
            metrics["flags"] = flags
            metrics["history"] = history_rows
            report_rows.append(metrics)
    finally:
        conn.close()

    report_name = f"stock_summary_{report_date or date.today().strftime('%Y-%m-%d')}.csv"
    html_name = f"stock_summary_{report_date or date.today().strftime('%Y-%m-%d')}.html"
    csv_path = Path(config["reports_dir"]) / report_name
    html_path = Path(config["reports_dir"]) / html_name
    write_csv_report(csv_path, report_rows)
    write_html_report(
        html_path,
        report_rows,
        config,
        config.get("data_source", "yfinance"),
        successful_symbols or [row["symbol"] for row in report_rows],
        failed_symbols or [],
        quality_issues,
    )
    logger.info("Wrote reports to %s and %s", csv_path, html_path)
    return report_rows


def show_status(config: dict[str, Any], logger: Any) -> None:
    """Display basic database and run information."""
    initialize_database(config["database_path"])
    conn = create_connection(config["database_path"])
    try:
        count = conn.execute("SELECT COUNT(*) FROM price_history").fetchone()[0]
        issue_count = conn.execute("SELECT COUNT(*) FROM data_quality_issues").fetchone()[0]
        run_count = conn.execute("SELECT COUNT(*) FROM collection_runs").fetchone()[0]
        print(f"Database path: {config['database_path']}")
        print(f"Stored price rows: {count}")
        print(f"Stored quality issues: {issue_count}")
        print(f"Collection runs: {run_count}")
    finally:
        conn.close()


def main() -> int:
    """Run the requested CLI command."""
    parser = build_parser()
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    config = load_config(base_dir)
    ensure_directories(config)
    logger = setup_logging(config, run_id=datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"))
    logger.info("Starting Stock Scrapper")
    logger.info("Loaded configuration from %s", base_dir / "config" / "settings.yaml")

    symbols = load_watchlist(config["watchlist_path"]) if args.command not in {"status"} else load_watchlist(config["watchlist_path"])
    if getattr(args, "symbols", None):
        symbols = [symbol.upper() for symbol in args.symbols]

    if args.command == "update":
        successful, failed, inserted, updated = update_symbols(config, logger, symbols, full_refresh=getattr(args, "full_refresh", False))
        logger.info("Update complete: inserted=%s updated=%s successful=%s failed=%s", inserted, updated, successful, failed)
        return 0

    if args.command == "validate":
        issues = validate_database(config, logger)
        logger.info("Validation complete: %s issues found", len(issues))
        return 0

    if args.command == "report":
        build_reports(config, logger, symbols, report_date=getattr(args, "date", None))
        return 0

    if args.command == "run":
        # The run command chains the update, validation, and reporting steps into one workflow.
        successful, failed, _, _ = update_symbols(config, logger, symbols, full_refresh=getattr(args, "full_refresh", False))
        validate_database(config, logger)
        build_reports(config, logger, symbols, successful_symbols=successful, failed_symbols=failed)
        return 0

    if args.command == "status":
        show_status(config, logger)
        return 0

    parser.error("Unsupported command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
