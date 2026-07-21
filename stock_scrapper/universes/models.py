"""Typed universe values shared by CLI workflows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AnalysisScope(StrEnum):
    CANDIDATE_UNIVERSE = "candidate_universe"
    ALL_DATA_SYMBOLS = "all_data_symbols"
    CUSTOM = "custom"


@dataclass(frozen=True)
class ResolvedUniverse:
    candidates: tuple[str, ...]
    benchmark: str
    market_context: tuple[str, ...]
    defensive_context: tuple[str, ...]
    data_symbols: tuple[str, ...]
    requested_symbols: tuple[str, ...]
    analysis_scope: AnalysisScope
    configuration_hash: str
    validation_warnings: tuple[str, ...] = ()

    def snapshot(self) -> dict[str, object]:
        return {
            "candidates": list(self.candidates),
            "benchmark": self.benchmark,
            "market_context": list(self.market_context),
            "defensive_context": list(self.defensive_context),
            "requested_analysis_symbols": list(self.requested_symbols),
            "analysis_scope": self.analysis_scope.value,
            "configuration_hash": self.configuration_hash,
            "warnings": list(self.validation_warnings),
        }
