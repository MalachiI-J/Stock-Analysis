"""Abstract base class for data collectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


PRICE_HISTORY_COLUMNS: tuple[str, ...] = (
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "dividends",
    "stock_splits",
    "data_source",
    "collected_at",
)


def empty_price_frame() -> pd.DataFrame:
    """Return an empty price-history frame with the canonical column order."""
    return pd.DataFrame(columns=list(PRICE_HISTORY_COLUMNS))


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
        """Collect daily history through the inclusive ``end_date``."""
