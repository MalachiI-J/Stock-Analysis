"""Role-aware, command-specific universe resolution."""

from .models import AnalysisScope, ResolvedUniverse
from .resolver import resolve_universe

__all__ = ["AnalysisScope", "ResolvedUniverse", "resolve_universe"]
