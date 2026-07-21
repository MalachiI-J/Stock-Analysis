"""Canonical live and historical as-of analysis service."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Mapping, Sequence
from uuid import uuid4

from stock_scrapper.analysis.engine import analyze_symbol, persist_analysis_results
from stock_scrapper.analysis.eligibility import evaluate_eligibility
from stock_scrapper.analysis.market_context import (
    MarketContext,
    calculate_market_context,
    calculate_watchlist_breadth,
)
from stock_scrapper.analysis.scoring_config import validate_scoring_config
from stock_scrapper.database import fetch_price_history, fetch_quality_issues
from stock_scrapper.models.analysis_models import AnalysisResult
from stock_scrapper.processing.historical_features import HistoricalFeatureCache
from stock_scrapper.utilities.hashing import stable_sha256
from stock_scrapper.utilities.provenance import collect_provenance
from stock_scrapper.data_health import assess_data_health
from pathlib import Path


@dataclass(frozen=True)
class AnalysisBatch:
    """All symbol results sharing one as-of market context."""

    results: list[AnalysisResult]
    market_context: MarketContext
    as_of_date: str
    data_through_date: str | None
    configuration_hash: str
    analysis_run_id: str | None = None


def _as_of(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _issue_active_as_of(issue: Mapping[str, Any], as_of: date) -> bool:
    """Reconstruct whether a persisted issue was active on ``as_of``."""
    cutoff = as_of.isoformat()
    first = str(issue.get("first_detected_at") or issue.get("detected_time") or "")[:10]
    if not first or first > cutoff:
        return False
    reopened = str(issue.get("reopened_at") or "")[:10]
    resolved = str(issue.get("resolved_at") or "")[:10]
    if reopened and reopened <= cutoff:
        return not resolved or resolved < reopened or resolved > cutoff
    return not resolved or resolved > cutoff


class AnalysisService:
    """Use identical indicators, scores, regime, and eligibility in every caller."""

    def __init__(
        self,
        conn: sqlite3.Connection | None,
        rules: dict[str, Any],
        watchlist: Sequence[str],
        include_incomplete_bars: bool = False,
    ) -> None:
        self.conn = conn
        self.rules = validate_scoring_config(rules)
        self.watchlist = list(dict.fromkeys(symbol.upper() for symbol in watchlist))
        self.include_incomplete_bars = include_incomplete_bars
        self.configuration_hash = stable_sha256(self.rules)
        self._historical_features: HistoricalFeatureCache | None = None

    def prime_historical_features(
        self,
        histories: Mapping[str, list[dict[str, Any]]],
        snapshot_dates: Sequence[str | date],
        feature_symbols: Sequence[str] | None = None,
    ) -> None:
        """Prepare causal rolling snapshots for repeated historical analysis."""
        benchmark = str(self.rules.get("benchmark_symbol", "SPY")).upper()
        self._historical_features = HistoricalFeatureCache(
            histories,
            benchmark,
            snapshot_dates,
            feature_symbols,
        )

    def analyze_as_of(
        self,
        symbol: str,
        as_of_date: str | date,
        persist: bool = False,
    ) -> AnalysisResult:
        """Analyze one symbol at a causal database snapshot."""
        return self.analyze_many_as_of([symbol], as_of_date, persist=persist).results[0]

    def analyze_many_as_of(
        self,
        symbols: Sequence[str],
        as_of_date: str | date,
        persist: bool = False,
        analysis_scope: str = "custom",
        universe_snapshot: Mapping[str, Any] | None = None,
        candidate_universe_hash: str | None = None,
    ) -> AnalysisBatch:
        """Load every input with ``trade_date <= as_of_date`` at SQL level."""
        if self.conn is None:
            raise RuntimeError("Database-backed analysis requires a SQLite connection")
        as_of = _as_of(as_of_date)
        requested = list(dict.fromkeys(symbol.upper() for symbol in symbols))
        benchmark = str(self.rules.get("benchmark_symbol", "SPY")).upper()
        context_symbols = {
            str(symbol).upper() for symbol in self.rules.get("market_context_symbols", [benchmark, "QQQ", "IWM"])
        }
        load_symbols = set(self.watchlist) | set(requested) | context_symbols | {benchmark}
        histories = {
            symbol: fetch_price_history(self.conn, symbol, end_date=as_of, include_incomplete=self.include_incomplete_bars)
            for symbol in sorted(load_symbols)
        }
        quality = fetch_quality_issues(
            self.conn, unresolved_only=True, as_of_date=as_of
        )
        quality_by_symbol: dict[str, list[dict[str, Any]]] = {}
        for issue in quality:
            quality_by_symbol.setdefault(str(issue.get("symbol", "")).upper(), []).append(issue)
        return self.analyze_loaded_many_as_of(
            requested,
            histories,
            as_of,
            quality_by_symbol=quality_by_symbol,
            persist=persist, analysis_scope=analysis_scope, universe_snapshot=universe_snapshot,
            candidate_universe_hash=candidate_universe_hash,
        )

    def analyze_loaded_many_as_of(
        self,
        symbols: Sequence[str],
        histories: Mapping[str, list[dict[str, Any]]],
        as_of_date: str | date,
        *,
        quality_by_symbol: Mapping[str, list[dict[str, Any]]] | None = None,
        persist: bool = False,
        analysis_scope: str = "custom",
        universe_snapshot: Mapping[str, Any] | None = None,
        candidate_universe_hash: str | None = None,
    ) -> AnalysisBatch:
        """Analyze preloaded, end-bounded histories for efficient backtests."""
        as_of = _as_of(as_of_date)
        requested = list(dict.fromkeys(symbol.upper() for symbol in symbols))
        feature_cache = self._historical_features
        cache_ready = feature_cache is not None and all(
            feature_cache.get(symbol, as_of) is not None for symbol in requested
        )
        if cache_ready:
            assert feature_cache is not None
            normalized_histories = {
                symbol.upper(): feature_cache.history_as_of(symbol, as_of)
                for symbol in histories
            }
        else:
            normalized_histories = {
                symbol.upper(): [
                    row
                    for row in rows
                    if row.get("trade_date") is not None
                    and str(row["trade_date"])[:10] <= as_of.isoformat()
                ]
                for symbol, rows in histories.items()
            }
        issue_map = {
            symbol.upper(): [issue for issue in issues if _issue_active_as_of(issue, as_of)]
            for symbol, issues in (quality_by_symbol or {}).items()
        }
        benchmark = str(self.rules.get("benchmark_symbol", "SPY")).upper()
        context_symbols = [
            str(symbol).upper()
            for symbol in self.rules.get("market_context_symbols", [benchmark, "QQQ", "IWM"])
        ]
        candidate_symbols: list[str] = []
        for symbol in self.watchlist:
            if symbol in set(context_symbols):
                continue
            eligible, _, _ = evaluate_eligibility(
                symbol=symbol,
                history=normalized_histories.get(symbol, []),
                quality_issues=issue_map.get(symbol, []),
                as_of_date=as_of,
                minimum_history_days=int(self.rules.get("minimum_history_days", 252)),
            )
            if eligible:
                candidate_symbols.append(symbol)
        breadth, breadth_above, breadth_eligible = calculate_watchlist_breadth(
            normalized_histories, candidate_symbols
        )
        context_histories = {
            symbol: normalized_histories.get(symbol, []) for symbol in context_symbols
        }
        market_context = calculate_market_context(
            normalized_histories.get(benchmark, []),
            context_histories,
            breadth,
            self.rules,
        )
        market_context.metrics.update(
            {
                "breadth_symbols_above_sma200": breadth_above,
                "breadth_symbols_eligible": breadth_eligible,
            }
        )
        results = [
            analyze_symbol(
                symbol,
                normalized_histories.get(symbol, []),
                normalized_histories.get(benchmark, []),
                issue_map.get(symbol, []),
                as_of,
                self.rules,
                int(self.rules.get("minimum_history_days", 252)),
                int(self.rules.get("minimum_recent_days", 20)),
                market_context=market_context,
                context_histories=context_histories,
                breadth_ratio=breadth,
                feature_snapshot=(
                    feature_cache.get(symbol, as_of)
                    if cache_ready and feature_cache is not None
                    else None
                ),
                history_is_normalized=cache_ready,
            )
            for symbol in requested
        ]
        data_dates = [result.data_through_date for result in results if result.data_through_date]
        data_through = max(data_dates) if data_dates else None
        run_id: str | None = None
        if persist:
            if self.conn is None:
                raise RuntimeError("Cannot persist analysis without a SQLite connection")
            run_id = (
                "analysis-"
                + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
                + "-"
                + uuid4().hex[:8]
            )
            persist_analysis_results(
                self.conn,
                run_id,
                results,
                as_of.isoformat(),
                data_through,
                benchmark,
                market_context.regime,
                market_context.confidence,
                requested,
                [result.symbol for result in results if result.eligible_for_scoring],
                [result.symbol for result in results if not result.eligible_for_scoring],
                "completed",
                str(self.rules.get("scoring_version", "phase2-v2")),
                self.configuration_hash,
                configuration_snapshot=self.rules,
                market_regime_metrics=market_context.metrics,
                market_regime_reasons=market_context.reasons,
                provenance=collect_provenance(Path(__file__).resolve().parents[2],scoring_version=str(self.rules.get("scoring_version","phase2-v2"))),
                data_health_status=assess_data_health(self.conn,sorted(histories))["status"],
                universe_snapshot=universe_snapshot or {"candidates":self.watchlist,"benchmark":benchmark,"market_context":sorted(context_symbols),"requested_analysis_symbols":requested,"analysis_scope":analysis_scope},
                data_hash=stable_sha256({symbol:histories.get(symbol,[]) for symbol in sorted(histories)}),
                analysis_scope=analysis_scope,candidate_universe_hash=candidate_universe_hash,
            )
            self.conn.commit()
        return AnalysisBatch(
            results=results,
            market_context=market_context,
            as_of_date=as_of.isoformat(),
            data_through_date=data_through,
            configuration_hash=self.configuration_hash,
            analysis_run_id=run_id,
        )
