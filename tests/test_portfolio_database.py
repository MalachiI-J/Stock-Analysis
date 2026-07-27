from __future__ import annotations

from pathlib import Path

import pytest

from stock_scrapper.database import (
    close_portfolio_lots_fifo,
    create_connection,
    initialize_database,
    insert_portfolio_lot,
    list_open_portfolio_symbols,
    list_portfolio_lots,
    list_portfolio_sales,
)
from stock_scrapper.exceptions import InsufficientHoldingsError


def _conn(tmp_path: Path):
    db_path = tmp_path / "market.db"
    initialize_database(db_path)
    return create_connection(db_path)


def test_insert_and_list_portfolio_lot(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    try:
        lot_id = insert_portfolio_lot(
            conn,
            {
                "symbol": "aapl",
                "shares": 10,
                "cost_basis_per_share": 150.0,
                "opened_date": "2026-01-05",
                "notes": "first buy",
            },
        )
        conn.commit()
        lots = list_portfolio_lots(conn, "AAPL")
        assert len(lots) == 1
        assert lots[0]["lot_id"] == lot_id
        assert lots[0]["symbol"] == "AAPL"
        assert lots[0]["shares"] == 10.0
        assert lots[0]["remaining_shares"] == 10.0
        assert lots[0]["status"] == "open"
        assert list_open_portfolio_symbols(conn) == ["AAPL"]
    finally:
        conn.close()


def test_close_portfolio_lots_fifo_across_multiple_lots(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    try:
        insert_portfolio_lot(
            conn,
            {"symbol": "AAPL", "shares": 5, "cost_basis_per_share": 100.0, "opened_date": "2026-01-01"},
        )
        insert_portfolio_lot(
            conn,
            {"symbol": "AAPL", "shares": 10, "cost_basis_per_share": 120.0, "opened_date": "2026-02-01"},
        )
        conn.commit()

        sales = close_portfolio_lots_fifo(
            conn, symbol="AAPL", shares=8, sale_price=130.0, sale_date="2026-03-01"
        )
        conn.commit()

        # FIFO: first the 5-share lot fully closes, then 3 shares from the second lot.
        assert [sale["shares"] for sale in sales] == [5.0, 3.0]
        assert sales[0]["realized_pnl"] == pytest.approx(5 * (130.0 - 100.0))
        assert sales[1]["realized_pnl"] == pytest.approx(3 * (130.0 - 120.0))

        lots = list_portfolio_lots(conn, "AAPL")
        assert lots[0]["status"] == "closed"
        assert lots[0]["remaining_shares"] == 0.0
        assert lots[1]["status"] == "open"
        assert lots[1]["remaining_shares"] == 7.0

        assert list_open_portfolio_symbols(conn) == ["AAPL"]
        recorded_sales = list_portfolio_sales(conn, "AAPL")
        assert len(recorded_sales) == 2
    finally:
        conn.close()


def test_close_portfolio_lots_fifo_fully_closes_symbol(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    try:
        insert_portfolio_lot(
            conn,
            {"symbol": "AAPL", "shares": 5, "cost_basis_per_share": 100.0, "opened_date": "2026-01-01"},
        )
        conn.commit()
        close_portfolio_lots_fifo(conn, symbol="AAPL", shares=5, sale_price=110.0, sale_date="2026-02-01")
        conn.commit()
        assert list_open_portfolio_symbols(conn) == []
    finally:
        conn.close()


def test_close_portfolio_lots_rejects_overselling(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    try:
        insert_portfolio_lot(
            conn,
            {"symbol": "AAPL", "shares": 5, "cost_basis_per_share": 100.0, "opened_date": "2026-01-01"},
        )
        conn.commit()
        with pytest.raises(InsufficientHoldingsError, match="only 5.0 shares"):
            close_portfolio_lots_fifo(conn, symbol="AAPL", shares=6, sale_price=110.0, sale_date="2026-02-01")
        # Nothing should have changed.
        lots = list_portfolio_lots(conn, "AAPL")
        assert lots[0]["remaining_shares"] == 5.0
        assert list_portfolio_sales(conn, "AAPL") == []
    finally:
        conn.close()


def test_close_portfolio_lots_rejects_symbol_with_no_holdings(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    try:
        with pytest.raises(InsufficientHoldingsError, match="only 0 shares"):
            close_portfolio_lots_fifo(conn, symbol="MSFT", shares=1, sale_price=100.0, sale_date="2026-02-01")
    finally:
        conn.close()
