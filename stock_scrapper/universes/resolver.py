"""Resolve configured roles independently from a command's requested symbols."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from stock_scrapper.universes.models import AnalysisScope, ResolvedUniverse
from stock_scrapper.utilities.hashing import stable_sha256


def _symbols(values: Sequence[object]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value).strip().upper() for value in values if str(value).strip()))


def resolve_universe(
    config: Mapping[str, Any], *, command: str, explicit_symbols: Sequence[str] | None = None,
    scope: str | None = None,
) -> ResolvedUniverse:
    raw = config.get("universes") or {}
    candidates = _symbols(raw.get("candidates") or ())
    benchmark_value = raw.get("benchmark") or "SPY"
    benchmark = str(benchmark_value.get("symbol") if isinstance(benchmark_value, Mapping) else benchmark_value).upper()
    market = _symbols(raw.get("market_context") or (benchmark, "QQQ", "IWM"))
    defensive = _symbols(raw.get("defensive_context") or ())
    data = _symbols((*candidates, benchmark, *market, *defensive))
    explicit = _symbols(explicit_symbols or ())
    warnings: list[str] = []
    if explicit:
        requested = explicit
        analysis_scope = AnalysisScope.CUSTOM
        if benchmark in requested:
            warnings.append(f"Benchmark {benchmark} is also an explicitly requested candidate")
    elif scope in {"all-data", "all_data_symbols"}:
        requested, analysis_scope = data, AnalysisScope.ALL_DATA_SYMBOLS
    elif command in {"update", "reconcile-prices", "corporate-actions-refresh", "validate", "data-health", "data-health-report"}:
        requested, analysis_scope = data, AnalysisScope.ALL_DATA_SYMBOLS
    else:
        requested, analysis_scope = candidates, AnalysisScope.CANDIDATE_UNIVERSE
    snapshot = {"candidates": candidates, "benchmark": benchmark, "market_context": market, "defensive_context": defensive}
    return ResolvedUniverse(candidates, benchmark, market, defensive, data, requested, analysis_scope, stable_sha256(snapshot), tuple(warnings))
