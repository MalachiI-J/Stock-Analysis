from __future__ import annotations

from pathlib import Path

from stock_scrapper.database import create_connection, initialize_database, upsert_price_history
from stock_scrapper.revision_policy import compare_price_rows
from stock_scrapper.backtesting.diagnostics import calculate_diagnostics


def test_tiny_float_difference_is_precision_noise() -> None:
    result=compare_price_rows({"adjusted_close":504.296142578125},{"adjusted_close":504.2961730957031})
    assert result.revision_class == "precision_noise"
    assert result.material_differences == ()
    assert result.precision_only_differences == ("adjusted_close",)


def test_material_volume_dividend_and_split_changes() -> None:
    result=compare_price_rows({"close":100,"volume":10,"dividends":0,"stock_splits":0},
                              {"close":101,"volume":11,"dividends":0.2,"stock_splits":2})
    assert result.revision_class == "corporate_action_revision"
    assert set(result.material_differences) == {"close","volume","dividends","stock_splits"}


def test_precision_noise_is_not_stored_or_repeated(tmp_path: Path) -> None:
    db=initialize_database(tmp_path/"market.db"); conn=create_connection(db)
    row={"symbol":"AAPL","trade_date":"2024-01-02","open":500.0,"high":505.0,"low":499.0,
         "close":504.0,"adjusted_close":504.296142578125,"volume":10,"dividends":0.0,
         "stock_splits":0.0,"data_source":"fixture"}
    try:
        assert upsert_price_history(conn,row)==(1,0)
        tiny={**row,"adjusted_close":504.2961730957031}
        assert upsert_price_history(conn,tiny)==(0,0)
        assert upsert_price_history(conn,tiny)==(0,0)
        assert conn.execute("select count(*) from price_history_revisions").fetchone()[0] == 0
        assert conn.execute("select adjusted_close from price_history").fetchone()[0] == row["adjusted_close"]
    finally: conn.close()


def test_post_simulation_attribution_reconciles_without_mutating_inputs() -> None:
    trades=[{"trade_id":"t1","symbol":"AAA","realized_pnl":10.0,"total_commission":1.0,"total_slippage":0.5,"holding_period_days":5,"execution_date":"2024-01-02","exit_execution_date":"2024-01-03","fill_price":100.0,"exit_fill_price":110.0,"exit_reason":"test"},
            {"trade_id":"t2","symbol":"BBB","realized_pnl":-4.0,"total_commission":1.0,"total_slippage":0.5,"holding_period_days":2,"execution_date":"2024-01-02","exit_execution_date":"2024-01-03","fill_price":100.0,"exit_fill_price":96.0,"exit_reason":"test"}]
    signals=[{"signal_id":"s1","symbol":"AAA","signal_date":"2024-01-02","action":"entry"}]; original=[dict(v) for v in signals]
    histories={s:[{"trade_date":"2024-01-02","adjusted_close":100.0},{"trade_date":"2024-01-03","adjusted_close":p},{"trade_date":"2024-01-04","adjusted_close":p+1}] for s,p in (("AAA",110.0),("BBB",96.0))}
    result=calculate_diagnostics("run",trades,signals,[],[],histories,2)
    assert sum(row["net_pnl"] for row in result["attribution"]) == 6.0
    assert result["concentration"]["net_profit_excluding_best_trade"] == -4.0
    assert result["exit_diagnostics"]
    assert signals == original
