"""Abstract base class for data collectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class BaseCollector(ABC):
    """Common interface for all market-data collectors."""

    @abstractmethod
    def collect(
        self,
        symbol: str,
        start_date: date | None = None,
        end_date: date | None = None,
        full_refresh: bool = False,
    ) -> pd.DataFrame:
        """Collect a symbol's daily price history."""
