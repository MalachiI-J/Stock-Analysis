"""Universe validation helpers."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence


def missing_data_symbols(conn: sqlite3.Connection, symbols: Sequence[str]) -> list[str]:
    return [symbol for symbol in symbols if conn.execute("SELECT 1 FROM price_history WHERE symbol=? LIMIT 1", (symbol,)).fetchone() is None]
