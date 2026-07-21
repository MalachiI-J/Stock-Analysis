from __future__ import annotations

import sqlite3
from pathlib import Path

from stock_scrapper.analysis.engine import persist_analysis_results
from stock_scrapper.database import create_connection, get_latest_canonical_analysis_run, initialize_database
from stock_scrapper.migrations.migration_manager import apply_migrations
from stock_scrapper.reporting.persistence import report_identity
from stock_scrapper.universes import AnalysisScope, resolve_universe


CONFIG={"universes":{"candidates":["AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","JPM","WMT","XOM"],"benchmark":{"symbol":"SPY"},"market_context":["SPY","QQQ","IWM"],"defensive_context":["TLT","GLD"]}}


def test_command_specific_universe_defaults_and_order() -> None:
    analysis=resolve_universe(CONFIG,command="analyze")
    data=resolve_universe(CONFIG,command="update")
    backtest=resolve_universe(CONFIG,command="backtest")
    assert analysis.analysis_scope is AnalysisScope.CANDIDATE_UNIVERSE
    assert analysis.requested_symbols==analysis.candidates
    assert backtest.requested_symbols==analysis.candidates
    assert data.requested_symbols==("AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","JPM","WMT","XOM","SPY","QQQ","IWM","TLT","GLD")


def test_explicit_symbols_are_custom_and_benchmark_overlap_warns() -> None:
    resolved=resolve_universe(CONFIG,command="backtest",explicit_symbols=["aapl","SPY","aapl"])
    assert resolved.requested_symbols==("AAPL","SPY")
    assert resolved.analysis_scope is AnalysisScope.CUSTOM
    assert "Benchmark SPY" in resolved.validation_warnings[0]


def test_report_identities_do_not_collide() -> None:
    candidate=report_identity("candidate_universe",["AAPL","MSFT"],"analysis-x-12345678")
    custom=report_identity("custom",["AAPL","MSFT"],"analysis-y-87654321")
    other=report_identity("custom",["AAPL"],"analysis-z-abcdef12")
    assert len({candidate,custom,other})==3
    assert candidate=="candidates_12345678"


def test_v6_legacy_scope_inference_is_idempotent(tmp_path: Path) -> None:
    path=tmp_path/"legacy.db"; initialize_database(path)
    conn=create_connection(path)
    conn.execute("INSERT INTO analysis_runs(analysis_run_id,started_at,status) VALUES('custom-run','2026-01-01','completed')")
    conn.execute("INSERT INTO stock_analysis(analysis_run_id,symbol,as_of_date,classification,primary_reason,eligible_for_scoring,created_at) VALUES('custom-run','AAPL','2026-01-01','Watch','x',1,'2026-01-01')")
    conn.execute("UPDATE analysis_runs SET analysis_scope=NULL,requested_symbols_json=NULL WHERE analysis_run_id='custom-run'"); conn.commit(); conn.close()
    apply_migrations(path); apply_migrations(path)
    conn=create_connection(path); row=conn.execute("SELECT analysis_scope,symbol_count,legacy_scope_inferred,is_canonical FROM analysis_runs WHERE analysis_run_id='custom-run'").fetchone()
    assert tuple(row)==("custom",1,1,0); conn.close()


def test_latest_canonical_never_falls_back_to_newer_custom(tmp_path: Path) -> None:
    path=tmp_path/"runs.db"; initialize_database(path); conn=create_connection(path)
    common="""INSERT INTO analysis_runs(analysis_run_id,started_at,completed_at,as_of_date,status,analysis_scope,is_canonical,symbol_count)
      VALUES(?,?,?,?,?,?,?,?)"""
    conn.execute(common,("canonical","2026-01-01","2026-01-01","2026-01-01","completed","candidate_universe",1,10))
    conn.execute(common,("custom","2026-01-02","2026-01-02","2026-01-02","completed","custom",0,2)); conn.commit()
    assert get_latest_canonical_analysis_run(conn)["analysis_run_id"]=="canonical"; conn.close()
