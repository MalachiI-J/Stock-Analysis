"""Command-line entry point for the local Stock Scrapper research system."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4
from zoneinfo import ZoneInfo

import yaml

from stock_scrapper.analysis.repository import results_from_saved_run
from stock_scrapper.analysis.scoring_config import validate_scoring_config
from stock_scrapper.analysis.service import AnalysisBatch, AnalysisService
from stock_scrapper.backtesting.config import BacktestConfig, load_backtesting_config
from stock_scrapper.backtesting.engine import PortfolioBacktestResult, run_portfolio_backtest
from stock_scrapper.backtesting.persistence import (
    list_backtest_runs,
    load_backtest,
    persist_walk_forward,
)
from stock_scrapper.backtesting.reporting import write_backtest_reports
from stock_scrapper.backtesting.walk_forward import (
    InsufficientWalkForwardDataError,
    WalkForwardExecutionResult,
    run_walk_forward,
)
from stock_scrapper.collectors.yahoo_prices import YahooPriceCollector
from stock_scrapper.collectors.corporate_actions import action_records, upsert_actions, record_action_coverage
from stock_scrapper.market_calendar import SessionResolver
from stock_scrapper.config import load_config, load_watchlist, load_universes, validate_universes
from stock_scrapper.data_health import assess_data_health
from stock_scrapper.database import (
    create_connection,
    fetch_price_history,
    fetch_quality_issues,
    get_analysis_run,
    get_latest_analysis_run,
    get_latest_trade_date,
    initialize_database,
    insert_collection_run,
    list_analysis_runs,
    quality_issue_fingerprint,
    classify_price_revisions,
    record_quality_issue,
    resolve_quality_issues_after_validation,
    upsert_price_history,
)
from stock_scrapper.exceptions import (
    ExitCode,
    InvalidConfigurationError,
    InvalidDateError,
    MissingDataError,
    OperationFailedError,
)
from stock_scrapper.processing.validation import validate_price_records
from stock_scrapper.reporting.report_builder import write_phase2_reports
from stock_scrapper.utilities.hashing import canonical_json
from stock_scrapper.utilities.logging_setup import setup_logging
from stock_scrapper.utilities.provenance import collect_provenance


def load_scoring_rules(base_dir: Path) -> dict[str, Any]:
    """Load and strictly validate the canonical Phase 2 rules."""
    rules_path = base_dir / "config" / "scoring_rules.yaml"
    if not rules_path.exists():
        raise InvalidConfigurationError(f"Scoring configuration is missing: {rules_path}")
    try:
        with rules_path.open("r", encoding="utf-8") as handle:
            rules = yaml.safe_load(handle)
        if not isinstance(rules, dict):
            raise ValueError("scoring_rules.yaml must contain a mapping")
        return validate_scoring_config(rules)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise InvalidConfigurationError(str(exc)) from exc


def _add_as_of_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--as-of-date",
        "--date",
        dest="as_of_date",
        help="Inclusive analysis date (YYYY-MM-DD)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the complete Phase 1-3 CLI."""
    parser = argparse.ArgumentParser(
        description="Stock Scrapper: free, local market research and realistic portfolio backtesting"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    update_parser = subparsers.add_parser("update", help="Collect and store missing daily data")
    update_parser.add_argument("--symbols", nargs="+", help="Optional symbol list")
    update_parser.add_argument("--full-refresh", action="store_true")

    report_parser = subparsers.add_parser("report", help="Generate an offline Phase 2 report")
    report_parser.add_argument("--symbols", nargs="+", help="Optional symbol list")
    report_parser.add_argument("--date", help="Inclusive report date (YYYY-MM-DD)")

    run_parser = subparsers.add_parser("run", help="Update, validate, analyze, and report")
    run_parser.add_argument("--symbols", nargs="+", help="Optional symbol list")
    run_parser.add_argument("--full-refresh", action="store_true")

    analyze_parser = subparsers.add_parser("analyze", help="Run and save canonical Phase 2 analysis")
    analyze_parser.add_argument("--symbols", nargs="+", help="Optional symbol list")
    analyze_parser.add_argument("--include-incomplete-bars", action="store_true", help="EXPERIMENTAL: include unfinished/untrusted daily bars")
    _add_as_of_argument(analyze_parser)

    scores_parser = subparsers.add_parser("scores", help="Read the latest saved scores")
    scores_parser.add_argument("--symbols", nargs="+", help="Optional saved symbols")
    scores_source = scores_parser.add_mutually_exclusive_group()
    scores_source.add_argument("--run-id", help="Read one exact saved run")
    scores_source.add_argument("--recalculate", action="store_true")
    _add_as_of_argument(scores_parser)

    explain_parser = subparsers.add_parser("explain", help="Explain saved symbol analysis")
    explain_parser.add_argument("symbol", nargs="*", help="Symbol, e.g. AAPL")
    explain_parser.add_argument("--symbols", nargs="+", help="Optional symbol list")
    explain_source = explain_parser.add_mutually_exclusive_group()
    explain_source.add_argument("--run-id", help="Read one exact saved run")
    explain_source.add_argument("--recalculate", action="store_true")
    _add_as_of_argument(explain_parser)

    subparsers.add_parser("analysis-list", help="List saved Phase 2 runs")
    analysis_show = subparsers.add_parser("analysis-show", help="Show one saved Phase 2 run")
    analysis_show.add_argument("--run-id", required=True)

    backtest = subparsers.add_parser("backtest", help="Run and save a score_v1 portfolio backtest")
    backtest.add_argument("--strategy", choices=("score_v1",), default="score_v1")
    backtest.add_argument("--symbols", nargs="+", help="Optional candidate universe")
    backtest.add_argument("--start", help="Inclusive start date (YYYY-MM-DD)")
    backtest.add_argument("--end", help="Inclusive end date (YYYY-MM-DD)")
    backtest.add_argument("--initial-cash", type=float)
    backtest.add_argument("--commission-bps", type=float)
    backtest.add_argument("--slippage-bps", type=float)
    backtest.add_argument("--update", action="store_true", help="Explicitly update data first")

    subparsers.add_parser("backtest-list", help="List saved portfolio backtests")
    for command, help_text in (
        ("backtest-show", "Show one saved portfolio backtest"),
        ("backtest-report", "Generate offline reports for a saved backtest"),
        ("backtest-compare", "Compare a saved backtest with SPY and cash"),
    ):
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument("--run-id", required=True)
        if command == "backtest-show":
            child.add_argument("--full",action="store_true"); child.add_argument("--metrics",action="store_true"); child.add_argument("--trades",action="store_true"); child.add_argument("--provenance",action="store_true")
    for command in ("validate-backtest","strategy-diagnostics","benchmark-diagnostics"):
        child=subparsers.add_parser(command); child.add_argument("--run-id",required=True)

    walk = subparsers.add_parser("walk-forward", help="Run fixed-window strategy validation")
    walk.add_argument("--strategy", choices=("score_v1",), default="score_v1")
    walk.add_argument("--symbols", nargs="+", help="Optional candidate universe")
    walk.add_argument("--start", help="First development date (YYYY-MM-DD)")
    walk.add_argument("--end", help="Final holdout end date (YYYY-MM-DD)")

    subparsers.add_parser("validate", help="Run complete database validation")
    subparsers.add_parser("status", help="Show local database status")
    subparsers.add_parser("market-session", help="Show official XNYS session state")
    health = subparsers.add_parser("data-health", help="Show offline market-data health")
    health.add_argument("--symbols", nargs="+")
    health_report = subparsers.add_parser("data-health-report", help="Write offline data-health JSON report")
    health_report.add_argument("--symbols", nargs="+")
    reconcile = subparsers.add_parser("reconcile-prices", help="Refresh and audit recent provider history")
    reconcile.add_argument("--symbols", nargs="+")
    reconcile.add_argument("--sessions", type=int)
    reconcile.add_argument("--full", action="store_true")
    revisions = subparsers.add_parser("revisions", help="Show recorded price revisions")
    revisions.add_argument("--symbol")
    revisions.add_argument("--material-only", action="store_true")
    revisions.add_argument("--class", dest="revision_class")
    subparsers.add_parser("revisions-classify", help="Classify retained revision audit rows")
    actions = subparsers.add_parser("corporate-actions", help="Show explicitly stored corporate actions")
    actions.add_argument("--symbol")
    action_refresh = subparsers.add_parser("corporate-actions-refresh", help="Refresh explicit corporate actions and checked coverage")
    action_refresh.add_argument("--symbols", nargs="+")
    action_refresh.add_argument("--full", action="store_true")
    subparsers.add_parser("universe-show", help="Show role-aware symbol universes")
    subparsers.add_parser("universe-validate", help="Validate symbol universe roles")
    subparsers.add_parser("provenance", help="Show current software provenance")
    return parser


def ensure_directories(config: dict[str, Any]) -> None:
    """Create local output directories configured inside the project."""
    for key in ("raw_data_dir", "processed_data_dir", "reports_dir", "logs_dir"):
        Path(config[key]).mkdir(parents=True, exist_ok=True)


def _validate_runtime_config(config: dict[str, Any]) -> None:
    """Fail early when operational settings cannot be used safely."""
    required_paths = (
        "watchlist_path",
        "database_path",
        "raw_data_dir",
        "processed_data_dir",
        "reports_dir",
        "logs_dir",
    )
    for key in required_paths:
        if not isinstance(config.get(key), str) or not str(config[key]).strip():
            raise ValueError(f"{key} must be a non-empty path")
    lookback = config.get("historical_lookback_years")
    retries = config.get("retry_count")
    retry_delay = config.get("retry_delay_seconds")
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback <= 0:
        raise ValueError("historical_lookback_years must be a positive integer")
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("retry_count must be a nonnegative integer")
    if (
        isinstance(retry_delay, bool)
        or not isinstance(retry_delay, (int, float))
        or float(retry_delay) < 0
    ):
        raise ValueError("retry_delay_seconds must be a nonnegative number")


def _parse_date(value: str | None, *, field: str, default: date | None = None) -> date:
    if value is None:
        if default is None:
            raise InvalidDateError(f"{field} is required")
        return default
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise InvalidDateError(f"{field} must use YYYY-MM-DD: {value}") from exc


def _symbols_from_args(args: argparse.Namespace, watchlist: Sequence[str]) -> list[str]:
    values: list[str] = []
    if getattr(args, "symbol", None):
        values.extend(args.symbol)
    if getattr(args, "symbols", None):
        values.extend(args.symbols)
    if not values:
        values.extend(watchlist)
    return list(dict.fromkeys(str(symbol).strip().upper() for symbol in values if str(symbol).strip()))


def update_symbols(
    config: dict[str, Any],
    logger: Any,
    symbols: list[str],
    full_refresh: bool = False,
) -> tuple[list[str], list[str], int, int]:
    """Collect daily bars and complete each symbol's quality-issue lifecycle."""
    initialize_database(config["database_path"])
    conn = create_connection(config["database_path"])
    collector = YahooPriceCollector(
        max_retries=int(config.get("retry_count", 3)),
        retry_delay_seconds=float(config.get("retry_delay_seconds", 2)),
        historical_lookback_years=int(config.get("historical_lookback_years", 5)),
    )
    successful: list[str] = []
    failed: list[str] = []
    inserted_count = 0
    updated_count = 0
    run_id = f"collection-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"
    started_at = datetime.now(timezone.utc).isoformat()
    market_settings = config.get("market_data", {})
    resolver = SessionResolver(int(market_settings.get("provider_delay_minutes", 30)))
    completed_end = resolver.previous_completed_session()
    try:
        for symbol in symbols:
            try:
                latest = None if full_refresh else get_latest_trade_date(conn, symbol)
                if latest:
                    overlap = int(market_settings.get("recent_overlap_sessions", 10))
                    anchor = min(date.fromisoformat(latest), completed_end)
                    start = resolver.overlap_start(anchor, overlap)
                else:
                    start = None
                frame = collector.collect(
                    symbol=symbol,
                    start_date=start,
                    end_date=completed_end,
                    full_refresh=full_refresh,
                )
                if frame.empty and (start is None or start <= completed_end):
                    raise MissingDataError(
                        f"Provider returned no rows for {symbol} through completed session {completed_end}"
                    )
                actions_frame = frame
                if not full_refresh:
                    action_sessions = int(market_settings.get("corporate_action_refresh_sessions", 90))
                    action_start = resolver.overlap_start(completed_end, action_sessions)
                    if start is not None and action_start < start:
                        actions_frame = collector.collect(symbol=symbol, start_date=action_start, end_date=completed_end)
                conn.execute("BEGIN")
                try:
                    symbol_inserted = 0
                    symbol_updated = 0
                    for row in frame.to_dict(orient="records"):
                        inserted, updated = upsert_price_history(conn, row, collection_run_id=run_id, revision_policy=config.get("revision_policy"))
                        symbol_inserted += inserted
                        symbol_updated += updated
                    upsert_actions(conn, action_records(symbol, actions_frame))
                    complete_history = fetch_price_history(conn, symbol)
                    if not complete_history:
                        raise MissingDataError(
                            f"No stored or newly collected price history exists for {symbol}"
                        )
                    issues = validate_price_records(complete_history, symbol=symbol, now_date=date.today())
                    fingerprints: list[str] = []
                    for issue in issues:
                        record_quality_issue(conn, issue)
                        fingerprints.append(quality_issue_fingerprint(issue))
                    resolve_quality_issues_after_validation(conn, symbol, fingerprints)
                    conn.commit()
                    inserted_count += symbol_inserted
                    updated_count += symbol_updated
                except Exception:
                    conn.rollback()
                    raise
                successful.append(symbol)
                logger.info("Collection complete for %s: %s rows returned", symbol, len(frame))
            except Exception as exc:  # network/provider isolation
                failed.append(symbol)
                logger.exception("Collection failed for %s: %s", symbol, exc)
        status = "completed_with_errors" if failed and successful else ("failed" if failed else "completed")
        insert_collection_run(
            conn,
            {
                "run_id": run_id,
                "start_time": started_at,
                "end_time": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "symbols_requested": ",".join(symbols),
                "symbols_updated": ",".join(successful),
                "symbols_failed": ",".join(failed),
                "records_inserted": inserted_count,
                "records_updated": updated_count,
                "error_summary": "Collection failed for: " + ",".join(failed) if failed else "",
            },
        )
        conn.commit()
    finally:
        conn.close()
    return successful, failed, inserted_count, updated_count


def validate_database(config: dict[str, Any], logger: Any) -> list[dict[str, Any]]:
    """Completely validate stored histories, resolving issues no longer detected."""
    initialize_database(config["database_path"])
    conn = create_connection(config["database_path"])
    all_issues: list[dict[str, Any]] = []
    try:
        for symbol in load_watchlist(config["watchlist_path"]):
            history = fetch_price_history(conn, symbol)
            if history:
                detected = validate_price_records(history, symbol=symbol, now_date=date.today())
            else:
                detected = [
                    {
                        "symbol": symbol,
                        "trade_date": None,
                        "issue_type": "missing_history",
                        "severity": "critical",
                        "description": "No stored price history exists for the configured symbol",
                        "detected_time": datetime.now(timezone.utc).isoformat(),
                    }
                ]
            fingerprints: list[str] = []
            for issue in detected:
                record_quality_issue(conn, issue)
                fingerprints.append(quality_issue_fingerprint(issue))
                all_issues.append(issue)
            resolve_quality_issues_after_validation(conn, symbol, fingerprints)
        conn.commit()
        logger.info("Validation complete: %s active detections", len(all_issues))
    finally:
        conn.close()
    return all_issues


def _analysis_batch(
    config: dict[str, Any],
    base_dir: Path,
    symbols: Sequence[str],
    as_of_date: date,
    *,
    persist: bool,
    include_incomplete_bars: bool = False,
) -> AnalysisBatch:
    initialize_database(config["database_path"])
    conn = create_connection(config["database_path"])
    try:
        service = AnalysisService(
            conn,
            load_scoring_rules(base_dir),
            load_watchlist(config["watchlist_path"]),
            include_incomplete_bars=include_incomplete_bars,
        )
        return service.analyze_many_as_of(symbols, as_of_date, persist=persist)
    finally:
        conn.close()


def run_analysis(
    config: dict[str, Any],
    logger: Any,
    base_dir: Path,
    symbols: list[str],
    as_of_date: str | None = None,
) -> list[Any]:
    """Compatibility wrapper for a saved canonical analysis batch."""
    effective = _parse_date(as_of_date, field="as-of date", default=date.today())
    batch = _analysis_batch(config, base_dir, symbols, effective, persist=True)
    logger.info("Analysis %s complete for %s symbols", batch.analysis_run_id, len(batch.results))
    return batch.results


def _previous_analysis(conn: sqlite3.Connection, as_of_date: str) -> list[Any]:
    row = conn.execute(
        "SELECT analysis_run_id FROM analysis_runs WHERE status = 'completed' AND as_of_date < ? "
        "ORDER BY as_of_date DESC, COALESCE(completed_at, started_at) DESC LIMIT 1",
        (as_of_date,),
    ).fetchone()
    if row is None:
        return []
    saved = get_analysis_run(conn, str(row["analysis_run_id"]))
    return results_from_saved_run(saved) if saved else []


def build_reports(
    config: dict[str, Any],
    logger: Any,
    symbols: list[str],
    report_date: str | None = None,
    successful_symbols: list[str] | None = None,
    failed_symbols: list[str] | None = None,
    analysis_results: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate bounded, self-contained Phase 2 CSV and HTML reports."""
    del successful_symbols, failed_symbols
    effective = _parse_date(report_date, field="report date", default=date.today())
    project_root = Path(config.get("base_dir", Path(__file__).resolve().parent))
    if analysis_results is not None:
        for result in analysis_results:
            result_as_of = str(getattr(result, "as_of_date", ""))[:10]
            data_through = str(getattr(result, "data_through_date", "") or "")[:10]
            if result_as_of != effective.isoformat():
                raise InvalidDateError(
                    "Supplied analysis results do not match the requested report date"
                )
            if data_through and data_through > effective.isoformat():
                raise InvalidDateError(
                    "Supplied analysis results contain data after the requested report date"
                )
    batch = (
        None
        if analysis_results is not None
        else _analysis_batch(
            config,
            project_root,
            symbols,
            effective,
            persist=False,
        )
    )
    results = analysis_results if analysis_results is not None else batch.results  # type: ignore[union-attr]
    initialize_database(config["database_path"])
    conn = create_connection(config["database_path"])
    try:
        histories = {
            symbol: fetch_price_history(conn, symbol, end_date=effective) for symbol in symbols
        }
        issues = fetch_quality_issues(conn, unresolved_only=True, as_of_date=effective)
        previous = _previous_analysis(conn, effective.isoformat())
    finally:
        conn.close()
    if batch is None:
        regime = results[0].market_regime if results else "Insufficient Market Data"
        confidence = results[0].market_regime_confidence if results else None
        reasons = results[0].market_regime_effects if results else []
        rules = load_scoring_rules(project_root)
        metadata = {
            "as_of_date": effective.isoformat(),
            "data_through_date": max(
                (result.data_through_date for result in results if result.data_through_date),
                default=None,
            ),
            "scoring_version": rules.get("scoring_version"),
            "configuration_hash": AnalysisService(None, rules, symbols).configuration_hash,
            "benchmark_symbol": rules.get("benchmark_symbol", "SPY"),
            "market_regime": regime,
            "market_regime_confidence": confidence,
            "market_regime_reasons": reasons,
        }
    else:
        rules = load_scoring_rules(project_root)
        metadata = {
            "as_of_date": batch.as_of_date,
            "data_through_date": batch.data_through_date,
            "scoring_version": rules.get("scoring_version"),
            "configuration_hash": batch.configuration_hash,
            "benchmark_symbol": rules.get("benchmark_symbol", "SPY"),
            "market_regime": batch.market_context.regime,
            "market_regime_confidence": batch.market_context.confidence,
            "market_regime_reasons": batch.market_context.reasons,
        }
    paths = write_phase2_reports(
        config["reports_dir"],
        effective,
        metadata,
        results,
        histories,
        issues,
        previous,
    )
    logger.info("Wrote Phase 2 reports: %s", paths)
    return [result.__dict__ if hasattr(result, "__dict__") else dict(result) for result in results]
def _load_saved_results(
    config: dict[str, Any],
    run_id: str | None,
    required_symbols: Sequence[str] | None,
) -> tuple[dict[str, Any], list[Any]]:
    initialize_database(config["database_path"])
    conn = create_connection(config["database_path"])
    try:
        saved = (
            get_analysis_run(conn, run_id)
            if run_id
            else get_latest_analysis_run(conn, required_symbols)
        )
    finally:
        conn.close()
    if saved is None:
        raise MissingDataError("No matching saved analysis run exists; use --recalculate")
    results = results_from_saved_run(saved)
    if required_symbols:
        required = {symbol.upper() for symbol in required_symbols}
        available = {result.symbol for result in results}
        missing = sorted(required - available)
        if missing:
            raise MissingDataError(
                "The saved analysis does not contain: " + ", ".join(missing)
            )
        results = [result for result in results if result.symbol in required]
    if not results:
        raise MissingDataError("The saved analysis does not contain the requested symbols")
    return saved, results


def _backtest_config(
    base_dir: Path,
    args: argparse.Namespace,
) -> BacktestConfig:
    try:
        config = load_backtesting_config(base_dir / "config" / "backtesting_rules.yaml")
        overrides: dict[str, Any] = {"strategy_name": args.strategy}
        if getattr(args, "start", None):
            overrides["start_date"] = _parse_date(args.start, field="start date")
        if getattr(args, "end", None):
            overrides["end_date"] = _parse_date(args.end, field="end date")
        if getattr(args, "initial_cash", None) is not None:
            overrides["initial_cash"] = args.initial_cash
        if getattr(args, "commission_bps", None) is not None:
            overrides["commission_basis_points"] = args.commission_bps
        if getattr(args, "slippage_bps", None) is not None:
            overrides["slippage_basis_points"] = args.slippage_bps
        configured = config.with_overrides(**overrides)
        if configured.start_date and configured.end_date and configured.start_date > configured.end_date:
            raise InvalidDateError("Backtest start date is after end date")
        return configured
    except InvalidDateError:
        raise
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise InvalidConfigurationError(str(exc)) from exc


def _load_backtest_inputs(
    conn: sqlite3.Connection,
    config: dict[str, Any],
    rules: dict[str, Any],
    symbols: Sequence[str],
    backtest_config: BacktestConfig,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    end = backtest_config.end_date or date.today()
    watchlist = load_watchlist(config["watchlist_path"])
    context = [str(value).upper() for value in rules.get("market_context_symbols", [])]
    load_symbols = sorted(
        set(watchlist) | set(symbols) | set(context) | {backtest_config.benchmark.upper()}
    )
    histories = {
        symbol: fetch_price_history(conn, symbol, end_date=end) for symbol in load_symbols
    }
    benchmark = backtest_config.benchmark.upper()
    benchmark_dates = [
        str(row["trade_date"])[:10] for row in histories.get(benchmark, [])
    ]
    if not benchmark_dates:
        raise MissingDataError(f"No stored benchmark history exists for {benchmark}")
    start = backtest_config.start_date or date.fromisoformat(benchmark_dates[0])
    requested_end = backtest_config.end_date or date.fromisoformat(benchmark_dates[-1])
    if requested_end > date.fromisoformat(benchmark_dates[-1]):
        raise MissingDataError(
            f"Stored {benchmark} data ends on {benchmark_dates[-1]}, before the requested end date"
        )
    evaluation_dates = [
        value for value in benchmark_dates if start.isoformat() <= value <= requested_end.isoformat()
    ]
    if not evaluation_dates:
        raise MissingDataError("No benchmark sessions fall inside the requested backtest range")
    missing = [
        symbol
        for symbol in symbols
        if not any(
            start.isoformat() <= str(row.get("trade_date", ""))[:10] <= requested_end.isoformat()
            for row in histories.get(symbol, [])
        )
    ]
    if missing:
        raise MissingDataError(
            "No stored price history in the requested range for: " + ", ".join(missing)
        )
    issues = fetch_quality_issues(conn, unresolved_only=False, as_of_date=end)
    quality: dict[str, list[dict[str, Any]]] = {}
    for issue in issues:
        quality.setdefault(str(issue.get("symbol", "")).upper(), []).append(issue)
    return histories, quality


def run_backtests(
    config: dict[str, Any],
    logger: Any,
    symbols: list[str],
    *,
    base_dir: Path | None = None,
    backtest_config: BacktestConfig | None = None,
) -> PortfolioBacktestResult:
    """Run one persisted shared portfolio using only end-bounded local data."""
    root = base_dir or Path(__file__).resolve().parent
    rules = load_scoring_rules(root)
    typed = backtest_config or load_backtesting_config(root / "config" / "backtesting_rules.yaml")
    initialize_database(config["database_path"])
    conn = create_connection(config["database_path"])
    try:
        histories, quality = _load_backtest_inputs(conn, config, rules, symbols, typed)
        result = run_portfolio_backtest(
            symbols,
            histories,
            rules,
            typed,
            quality_by_symbol=quality,
            persist_conn=conn,
        )
        health=assess_data_health(conn,sorted(set(symbols)|{typed.benchmark}))
        coverage=[dict(row) for row in conn.execute("SELECT * FROM corporate_action_coverage WHERE symbol IN (%s)" % ",".join("?" for _ in set(symbols)),sorted(set(symbols))).fetchall()] if symbols else []
        conn.execute("UPDATE backtest_runs SET data_health_snapshot_json=?,corporate_action_coverage_json=?,revision_policy_version=?,universe_json=? WHERE run_id=?",
                     (canonical_json(health),canonical_json(coverage),str(config.get("revision_policy",{}).get("version","unknown")),canonical_json(load_universes(config)),result.run.run_id)); conn.commit()
        logger.info("Backtest %s persisted", result.run.run_id)
        return result
    finally:
        conn.close()


def show_status(config: dict[str, Any], logger: Any) -> None:
    """Display database coverage and saved-run counts."""
    initialize_database(config["database_path"])
    conn = create_connection(config["database_path"])
    try:
        values = {
            "price_rows": conn.execute("SELECT COUNT(*) FROM price_history").fetchone()[0],
            "first_date": conn.execute("SELECT MIN(trade_date) FROM price_history").fetchone()[0],
            "last_date": conn.execute("SELECT MAX(trade_date) FROM price_history").fetchone()[0],
            "unresolved_issues": conn.execute("SELECT COUNT(*) FROM data_quality_issues WHERE resolved_status=0").fetchone()[0],
            "collection_runs": conn.execute("SELECT COUNT(*) FROM collection_runs").fetchone()[0],
            "analysis_runs": conn.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0],
            "backtest_runs": conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0],
            "walk_forward_runs": conn.execute("SELECT COUNT(*) FROM walk_forward_runs").fetchone()[0],
        }
        print(f"Database path: {config['database_path']}")
        print(f"Stored price rows: {values['price_rows']}")
        print(f"Price coverage: {values['first_date']} through {values['last_date']}")
        print(f"Unresolved quality issues: {values['unresolved_issues']}")
        print(f"Collection runs: {values['collection_runs']}")
        print(f"Analysis runs: {values['analysis_runs']}")
        print(f"Backtest runs: {values['backtest_runs']}")
        print(f"Walk-forward runs: {values['walk_forward_runs']}")
    finally:
        conn.close()
    logger.info("Status complete")


def _print_scores(results: Sequence[Any]) -> None:
    for result in sorted(
        results,
        key=lambda item: (
            -(item.opportunity_score if item.opportunity_score is not None else -1),
            item.symbol,
        ),
    ):
        print(
            f"{result.symbol}: {result.classification} | risk={result.risk_score} "
            f"opportunity={result.opportunity_score} confidence={result.confidence_score} "
            f"regime={result.market_regime}"
        )


def _print_explanations(results: Sequence[Any]) -> None:
    for result in sorted(results, key=lambda item: item.symbol):
        print(f"{result.symbol}: {result.classification} — {result.primary_reason}")
        if result.positive_factors:
            print("  Positive factors: " + "; ".join(result.positive_factors))
        if result.risk_factors:
            print("  Risk factors: " + "; ".join(result.risk_factors))
        if result.confidence_limitations:
            print("  Confidence limitations: " + "; ".join(result.confidence_limitations))


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one command and return a documented process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if (
        args.command in {"scores", "explain"}
        and getattr(args, "as_of_date", None)
        and not args.recalculate
    ):
        parser.error("--as-of-date/--date requires --recalculate")
    base_dir = Path(__file__).resolve().parent
    logger: Any = None
    try:
        try:
            config = load_config(base_dir)
            config["base_dir"] = str(base_dir)
            _validate_runtime_config(config)
            ensure_directories(config)
            watchlist = load_watchlist(config["watchlist_path"])
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
            raise InvalidConfigurationError(str(exc)) from exc
        logger = setup_logging(config, run_id=datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"))
        logger.info("Starting Stock Scrapper")
        symbols = _symbols_from_args(args, watchlist)
        partial_update = False

        if args.command == "update":
            successful, failed, inserted, updated = update_symbols(
                config, logger, symbols, full_refresh=args.full_refresh
            )
            print(f"inserted={inserted} updated={updated} successful={len(successful)} failed={len(failed)}")
            if failed and successful:
                return int(ExitCode.PARTIAL_FAILURE)
            if failed:
                return int(ExitCode.OPERATION_FAILED)
            return int(ExitCode.SUCCESS)

        if args.command == "validate":
            issues = validate_database(config, logger)
            print(f"Active detections: {len(issues)}")
            return int(ExitCode.SUCCESS)

        if args.command == "market-session":
            resolver = SessionResolver(int(config.get("market_data", {}).get("provider_delay_minutes", 30)))
            now = datetime.now(timezone.utc)
            completed = resolver.previous_completed_session(now)
            today = now.astimezone(ZoneInfo("America/New_York")).date()
            payload = {"exchange": "XNYS", "now": now.isoformat(), "today_is_session": resolver.is_session(today), "last_completed_session": completed.isoformat()}
            if resolver.is_session(today): payload["today"] = resolver.session(today, now).__dict__ if hasattr(resolver.session(today, now), "__dict__") else {key: str(getattr(resolver.session(today, now), key)) for key in ("session_date","opens_at","closes_at","completion_at","is_early_close","is_complete")}
            print(json.dumps(payload, indent=2, default=str)); return int(ExitCode.SUCCESS)

        if args.command in {"universe-show", "universe-validate"}:
            universes = load_universes(config); warnings = validate_universes(universes)
            print(json.dumps({"universes": universes, "warnings": warnings}, indent=2))
            return int(ExitCode.PARTIAL_FAILURE if warnings else ExitCode.SUCCESS)

        if args.command == "provenance":
            print(json.dumps(collect_provenance(base_dir,scoring_version=str(load_scoring_rules(base_dir).get("scoring_version"))), indent=2, default=str)); return int(ExitCode.SUCCESS)

        if args.command in {"data-health", "data-health-report"}:
            initialize_database(config["database_path"]); conn = create_connection(config["database_path"])
            try: health = assess_data_health(conn, symbols, int(config.get("market_data", {}).get("provider_delay_minutes", 30)))
            finally: conn.close()
            print(json.dumps(health, indent=2, default=str))
            if args.command == "data-health-report":
                target = Path(config["reports_dir"]) / f"data_health_{date.today().isoformat()}.json"
                target.write_text(json.dumps(health, indent=2, default=str), encoding="utf-8"); print(f"Report: {target}")
                html_target=target.with_suffix(".html")
                rows="".join(f"<tr><td>{item['symbol']}</td><td>{item['status']}</td><td>{item['complete_bars']}</td><td>{len(item['missing_expected_sessions'])}</td><td>{item['material_revisions']}</td><td>{item['precision_noise_revisions']}</td><td>{item['corporate_action_coverage']}</td></tr>" for item in health["symbols"])
                html_target.write_text(f"<!doctype html><meta charset='utf-8'><title>Data health</title><style>body{{font-family:system-ui;margin:2rem}}table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:.4rem}}</style><h1>Data health: {health['status']}</h1><p>Last completed XNYS session: {health['last_completed_session']}</p><table><tr><th>Symbol</th><th>Status</th><th>Complete bars</th><th>Missing</th><th>Material revisions</th><th>Precision noise</th><th>Action coverage</th></tr>{rows}</table>",encoding="utf-8"); print(f"HTML: {html_target}")
            return int(ExitCode.OPERATION_FAILED if health["status"] == "Critical" else (ExitCode.PARTIAL_FAILURE if health["status"] == "Warning" else ExitCode.SUCCESS))

        if args.command == "reconcile-prices":
            if args.sessions is not None:
                if args.sessions < 1: parser.error("--sessions must be positive")
                config["market_data"]["recent_overlap_sessions"] = args.sessions
            successful, failed, inserted, updated = update_symbols(config, logger, symbols, full_refresh=args.full)
            print(f"inserted={inserted} revised={updated} unchanged rows are not counted successful={len(successful)} failed={len(failed)}")
            return int(ExitCode.PARTIAL_FAILURE if failed and successful else (ExitCode.OPERATION_FAILED if failed else ExitCode.SUCCESS))

        if args.command == "corporate-actions-refresh":
            initialize_database(config["database_path"]); conn=create_connection(config["database_path"])
            collector=YahooPriceCollector(max_retries=int(config.get("retry_count",3)),retry_delay_seconds=float(config.get("retry_delay_seconds",2)),historical_lookback_years=int(config.get("historical_lookback_years",5)))
            failures=[]; total=0
            try:
                for symbol in symbols:
                    coverage=conn.execute("SELECT MIN(trade_date),MAX(trade_date) FROM price_history WHERE symbol=?",(symbol,)).fetchone()
                    if not coverage or not coverage[0]: failures.append(symbol); continue
                    end=date.fromisoformat(coverage[1]); start=date.fromisoformat(coverage[0]) if args.full else SessionResolver().overlap_start(end,int(config.get("market_data",{}).get("corporate_action_refresh_sessions",90)))
                    try:
                        frame=collector.collect(symbol,start_date=start,end_date=end)
                        if frame.empty: raise MissingDataError(f"Provider returned no coverage response for {symbol}")
                        records=action_records(symbol,frame); total+=upsert_actions(conn,records)
                        record_action_coverage(conn,symbol,"yfinance",start.isoformat(),end.isoformat(),records)
                        conn.commit()
                    except Exception as exc:
                        conn.rollback(); failures.append(symbol)
                        record_action_coverage(conn,symbol,"yfinance",start.isoformat(),end.isoformat(),[],status="failed",error=str(exc)); conn.commit()
            finally: conn.close()
            print(f"actions_upserted={total} symbols={len(symbols)} failed={len(failures)}")
            return int(ExitCode.PARTIAL_FAILURE if failures else ExitCode.SUCCESS)

        if args.command == "revisions-classify":
            initialize_database(config["database_path"]); conn=create_connection(config["database_path"])
            try:
                counts=classify_price_revisions(conn,config.get("revision_policy")); conn.commit()
            finally: conn.close()
            print(json.dumps(counts,indent=2,sort_keys=True)); return int(ExitCode.SUCCESS)

        if args.command in {"revisions", "corporate-actions"}:
            initialize_database(config["database_path"]); conn = create_connection(config["database_path"])
            try:
                if args.command == "revisions":
                    clauses=[]; params=[]
                    if args.symbol: clauses.append("symbol=?"); params.append(args.symbol.upper())
                    if args.material_only: clauses.append("is_material=1")
                    if args.revision_class: clauses.append("revision_class=?"); params.append(args.revision_class)
                    rows = conn.execute("SELECT * FROM price_history_revisions" + ((" WHERE "+" AND ".join(clauses)) if clauses else "") + " ORDER BY detected_at DESC", params).fetchall()
                else:
                    rows = conn.execute("SELECT * FROM corporate_actions" + (" WHERE symbol=?" if args.symbol else "") + " ORDER BY action_date DESC", ((args.symbol.upper(),) if args.symbol else ())).fetchall()
            finally: conn.close()
            print(json.dumps([dict(row) for row in rows], indent=2, default=str)); return int(ExitCode.SUCCESS)

        if args.command == "analyze":
            effective = _parse_date(args.as_of_date, field="as-of date", default=date.today())
            if args.include_incomplete_bars:
                print("WARNING: EXPERIMENTAL analysis includes incomplete/untrusted daily bars", file=sys.stderr)
            else:
                conn = create_connection(config["database_path"])
                try: health = assess_data_health(conn, symbols, int(config.get("market_data", {}).get("provider_delay_minutes", 30)))
                finally: conn.close()
                if health["status"] == "Critical": raise MissingDataError("Critical market-data health blocks live classification")
            batch = _analysis_batch(config, base_dir, symbols, effective, persist=True, include_incomplete_bars=args.include_incomplete_bars)
            _print_scores(batch.results)
            if not any(result.eligible_for_scoring for result in batch.results):
                return int(ExitCode.MISSING_DATA)
            return int(ExitCode.SUCCESS)

        if args.command in {"scores", "explain"}:
            explicit = bool(getattr(args, "symbols", None) or getattr(args, "symbol", None))
            if args.recalculate:
                effective = _parse_date(args.as_of_date, field="as-of date", default=date.today())
                batch = _analysis_batch(config, base_dir, symbols, effective, persist=True)
                results = batch.results
            else:
                _, results = _load_saved_results(
                    config,
                    args.run_id,
                    symbols if explicit else None,
                )
            if args.command == "scores":
                _print_scores(results)
            else:
                _print_explanations(results)
            return int(ExitCode.SUCCESS)

        if args.command == "analysis-list":
            initialize_database(config["database_path"])
            conn = create_connection(config["database_path"])
            try:
                runs = list_analysis_runs(conn)
            finally:
                conn.close()
            for run in runs:
                print(
                    f"{run['analysis_run_id']} | {run['as_of_date']} | {run['status']} | "
                    f"regime={run['market_regime']} | through={run['data_through_date']}"
                )
            return int(ExitCode.SUCCESS)

        if args.command == "analysis-show":
            saved, results = _load_saved_results(config, args.run_id, None)
            print(canonical_json({key: value for key, value in saved.items() if key not in {"analyses", "regime"}}))
            _print_scores(results)
            return int(ExitCode.SUCCESS)

        if args.command == "report":
            effective = _parse_date(args.date, field="report date", default=date.today())
            batch = _analysis_batch(config, base_dir, symbols, effective, persist=False)
            scoring_rules = load_scoring_rules(base_dir)
            conn = create_connection(config["database_path"])
            try:
                histories = {symbol: fetch_price_history(conn, symbol, end_date=effective) for symbol in symbols}
                issues = fetch_quality_issues(conn, unresolved_only=True, as_of_date=effective)
                previous = _previous_analysis(conn, effective.isoformat())
            finally:
                conn.close()
            paths = write_phase2_reports(
                config["reports_dir"],
                effective,
                {
                    "as_of_date": batch.as_of_date,
                    "data_through_date": batch.data_through_date,
                    "scoring_version": scoring_rules.get("scoring_version"),
                    "configuration_hash": batch.configuration_hash,
                    "benchmark_symbol": scoring_rules.get("benchmark_symbol", "SPY"),
                    "market_regime": batch.market_context.regime,
                    "market_regime_confidence": batch.market_context.confidence,
                    "market_regime_reasons": batch.market_context.reasons,
                },
                batch.results,
                histories,
                issues,
                previous,
            )
            print(f"CSV: {paths['csv']}")
            print(f"HTML: {paths['html']}")
            return int(ExitCode.SUCCESS)

        if args.command == "backtest":
            typed = _backtest_config(base_dir, args)
            if args.update:
                update_universe = sorted(set(symbols) | set(watchlist))
                successful, failed, _, _ = update_symbols(config, logger, update_universe)
                if failed and not successful:
                    raise OperationFailedError("Explicit pre-backtest update failed for: " + ", ".join(failed))
                partial_update = bool(failed)
            result = run_backtests(
                config,
                logger,
                symbols,
                base_dir=base_dir,
                backtest_config=typed,
            )
            assert result.metrics is not None
            print(
                f"{result.run.run_id}: return={result.metrics.total_return:.2%} "
                f"ending_equity={result.metrics.ending_equity:.2f} "
                f"trades={result.metrics.number_of_trades} sharpe={result.metrics.sharpe_ratio}"
            )
            return int(ExitCode.PARTIAL_FAILURE if partial_update else ExitCode.SUCCESS)

        if args.command == "backtest-list":
            initialize_database(config["database_path"])
            conn = create_connection(config["database_path"])
            try:
                runs = list_backtest_runs(conn)
            finally:
                conn.close()
            for run in runs:
                print(
                    f"{run['run_id']} | {run['strategy_name']} {run['strategy_version']} | "
                    f"{run['start_date']}..{run['end_date']} | {run['status']} | equity={run['ending_equity']}"
                )
            return int(ExitCode.SUCCESS)

        if args.command in {"backtest-show", "backtest-report", "backtest-compare"}:
            initialize_database(config["database_path"])
            conn = create_connection(config["database_path"])
            try:
                saved = load_backtest(conn, args.run_id)
            finally:
                conn.close()
            if saved is None:
                raise MissingDataError(f"Backtest run does not exist: {args.run_id}")
            if args.command == "backtest-show":
                if args.full: payload=saved
                elif args.metrics: payload=saved.get("metrics",{})
                elif args.trades: payload=saved.get("trades",[])
                elif args.provenance: payload={k:saved.get(k) for k in ("application_version","strategy_name","strategy_version","scoring_version","schema_version","git_commit_hash","git_dirty","source_fingerprint","python_version","platform_info","configuration_hash","data_hash","deterministic_result_hash")}
                else:
                    metrics=saved.get("metrics",{}); payload={"run_id":saved.get("run_id"),"strategy":f"{saved.get('strategy_name')} {saved.get('strategy_version')}","requested_start":saved.get("requested_start_date"),"effective_start":saved.get("effective_start_date") or saved.get("start_date"),"effective_end":saved.get("effective_end_date") or saved.get("end_date"),"starting_equity":saved.get("initial_cash"),"ending_equity":saved.get("ending_equity"),"total_return":metrics.get("total_return"),"maximum_drawdown":metrics.get("maximum_drawdown"),"benchmark_return":metrics.get("benchmark_total_return"),"trades":metrics.get("number_of_trades"),"data_health":saved.get("data_health_snapshot_json"),"configuration_hash":saved.get("configuration_hash"),"result_hash":saved.get("deterministic_result_hash")}
                print(json.dumps(payload, sort_keys=True, indent=2, default=str))
            elif args.command == "backtest-report":
                paths = write_backtest_reports(config["reports_dir"], saved)
                for name, path in paths.items():
                    print(f"{name}: {path}")
            else:
                metrics = saved.get("metrics", {})
                print(f"Strategy total return: {metrics.get('total_return')}")
                print(f"SPY buy-and-hold return: {metrics.get('benchmark_total_return')}")
                print(f"Return versus SPY: {metrics.get('return_vs_benchmark')}")
                print("Cash return: 0.0")
                print(f"Drawdown versus SPY: {metrics.get('drawdown_vs_benchmark')}")
            return int(ExitCode.SUCCESS)

        if args.command in {"validate-backtest","strategy-diagnostics","benchmark-diagnostics"}:
            initialize_database(config["database_path"]); conn=create_connection(config["database_path"])
            try: saved=load_backtest(conn,args.run_id)
            finally: conn.close()
            if not saved: raise MissingDataError(f"Backtest run does not exist: {args.run_id}")
            if args.command=="strategy-diagnostics":
                print(json.dumps({"symbol_attribution":saved["symbol_attribution"],"signal_outcomes":saved["signal_outcomes"],"daily_diagnostics":saved["daily_diagnostics"]},indent=2,default=str))
            elif args.command=="benchmark-diagnostics": print(json.dumps({k:v for k,v in saved["metrics"].items() if "benchmark" in k or k in {"active_return","tracking_error","information_ratio","upside_capture","downside_capture","beta_to_benchmark","correlation_to_benchmark"}},indent=2))
            else:
                checks={"configuration_hash":bool(saved.get("configuration_hash")),"data_hash":bool(saved.get("data_hash")),"source_fingerprint":bool(saved.get("source_fingerprint")),"result_hash":bool(saved.get("deterministic_result_hash")),"warmup_metadata":saved.get("required_warmup_sessions") is not None,"benchmark_alignment":saved.get("effective_start_date") in (None,saved.get("start_date")),"linked_equity":bool(saved.get("equity_curve"))}
                print(json.dumps(checks,indent=2)); return int(ExitCode.SUCCESS if all(checks.values()) else ExitCode.OPERATION_FAILED)
            return int(ExitCode.SUCCESS)

        if args.command == "walk-forward":
            typed = _backtest_config(base_dir, args)
            rules = load_scoring_rules(base_dir)
            initialize_database(config["database_path"])
            conn = create_connection(config["database_path"])
            try:
                histories, quality = _load_backtest_inputs(conn, config, rules, symbols, typed)
                trading_dates = [
                    row["trade_date"] for row in histories.get(typed.benchmark.upper(), [])
                ]

                def executor(window: Any, immutable_config: BacktestConfig) -> WalkForwardExecutionResult:
                    window_config = immutable_config.with_overrides(
                        start_date=window.evaluation_start_date,
                        end_date=window.evaluation_end_date,
                        warm_up_days=immutable_config.walk_forward.warm_up_days,
                    )
                    outcome = run_portfolio_backtest(
                        symbols,
                        histories,
                        rules,
                        window_config,
                        quality_by_symbol=quality,
                        persist_conn=conn,
                        commit_persistence=False,
                        run_id=f"backtest-{window.window_id}",
                    )
                    return WalkForwardExecutionResult(
                        backtest_run_id=outcome.run.run_id,
                        metrics=outcome.metrics,
                    )

                walk_forward_run_id = (
                    "wf-"
                    + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
                    + "-"
                    + uuid4().hex[:8]
                )
                conn.execute("BEGIN")
                try:
                    walk_result = run_walk_forward(
                        typed,
                        trading_dates,
                        executor,
                        symbols=symbols,
                        walk_forward_run_id=walk_forward_run_id,
                    )
                    walk_result.benchmark_symbol = typed.benchmark
                    walk_result.symbols = list(symbols)
                    walk_result.configuration_snapshot = typed.to_dict()
                    persist_walk_forward(conn, walk_result)
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
            finally:
                conn.close()
            print(f"{walk_result.walk_forward_run_id}: {walk_result.status}")
            for window in walk_result.windows:
                print(
                    f"  {window.window_type}: {window.evaluation_start_date}.."
                    f"{window.evaluation_end_date} status={window.status} backtest={window.backtest_run_id}"
                )
            failed_windows = sum(window.status == "failed" for window in walk_result.windows)
            if failed_windows == len(walk_result.windows):
                return int(ExitCode.OPERATION_FAILED)
            if failed_windows:
                return int(ExitCode.PARTIAL_FAILURE)
            return int(ExitCode.SUCCESS)

        if args.command == "run":
            successful, failed, _, _ = update_symbols(
                config, logger, symbols, full_refresh=args.full_refresh
            )
            validate_database(config, logger)
            batch = _analysis_batch(config, base_dir, symbols, date.today(), persist=True)
            scoring_rules = load_scoring_rules(base_dir)
            conn = create_connection(config["database_path"])
            try:
                histories = {
                    symbol: fetch_price_history(conn, symbol, end_date=date.today())
                    for symbol in symbols
                }
                issues = fetch_quality_issues(
                    conn,
                    unresolved_only=True,
                    as_of_date=date.today(),
                )
                previous = _previous_analysis(conn, date.today().isoformat())
            finally:
                conn.close()
            paths = write_phase2_reports(
                config["reports_dir"],
                date.today(),
                {
                    "analysis_run_id": batch.analysis_run_id,
                    "as_of_date": batch.as_of_date,
                    "data_through_date": batch.data_through_date,
                    "scoring_version": scoring_rules.get("scoring_version"),
                    "configuration_hash": batch.configuration_hash,
                    "benchmark_symbol": scoring_rules.get("benchmark_symbol", "SPY"),
                    "market_regime": batch.market_context.regime,
                    "market_regime_confidence": batch.market_context.confidence,
                    "market_regime_reasons": batch.market_context.reasons,
                },
                batch.results,
                histories,
                issues,
                previous,
            )
            print(f"Reports: {paths}")
            if failed and successful:
                return int(ExitCode.PARTIAL_FAILURE)
            if failed:
                return int(ExitCode.OPERATION_FAILED)
            return int(ExitCode.SUCCESS)

        if args.command == "status":
            show_status(config, logger)
            return int(ExitCode.SUCCESS)
        parser.error("Unsupported command")
        return int(ExitCode.INVALID_ARGUMENTS)
    except InvalidDateError as exc:
        print(f"Invalid date: {exc}", file=sys.stderr)
        return int(ExitCode.INVALID_DATE)
    except InvalidConfigurationError as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return int(ExitCode.INVALID_CONFIGURATION)
    except MissingDataError as exc:
        print(f"Missing data: {exc}", file=sys.stderr)
        return int(ExitCode.MISSING_DATA)
    except InsufficientWalkForwardDataError as exc:
        print(f"Missing data: {exc}", file=sys.stderr)
        return int(ExitCode.MISSING_DATA)
    except sqlite3.Error as exc:
        print(f"Database failure: {exc}", file=sys.stderr)
        if logger:
            logger.exception("Database failure")
        return int(ExitCode.DATABASE_FAILURE)
    except OperationFailedError as exc:
        print(f"Operation failed: {exc}", file=sys.stderr)
        return int(ExitCode.OPERATION_FAILED)
    except Exception as exc:
        print(f"Operation failed: {exc}", file=sys.stderr)
        if logger:
            logger.exception("Complete operation failure")
        return int(ExitCode.OPERATION_FAILED)


if __name__ == "__main__":
    sys.exit(main())
