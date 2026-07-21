from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml

import stock_scrapper.analysis.engine as analysis_engine
from stock_scrapper.analysis.service import AnalysisService
from stock_scrapper.processing.historical_features import HistoricalFeatureCache


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
WATCHLIST = ["AAPL", *CONTEXT]


def _rules() -> dict[str, Any]:
    return yaml.safe_load(
        (PROJECT_ROOT / "config" / "scoring_rules.yaml").read_text(encoding="utf-8")
    )


def _dates(count: int) -> list[str]:
    result: list[str] = []
    current = date(2023, 1, 2)
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current.isoformat())
        current += timedelta(days=1)
    return result


def _histories() -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    sessions = _dates(340)
    histories: dict[str, list[dict[str, Any]]] = {}
    for symbol_index, symbol in enumerate(WATCHLIST):
        rows: list[dict[str, Any]] = []
        for index, trade_date in enumerate(sessions):
            if symbol == "AAPL" and index == 205:
                continue
            adjusted = (80.0 + symbol_index * 7.0) * (
                1.0 + 0.0008 * index + 0.004 * ((index % 9) - 4)
            )
            raw_multiplier = 2.0 if symbol == "AAPL" and index < 150 else 1.0
            raw_close = adjusted * raw_multiplier
            raw_open = raw_close * (1.0 + ((index % 5) - 2) * 0.0007)
            row: dict[str, Any] = {
                "symbol": symbol,
                "trade_date": trade_date,
                "open": raw_open,
                "high": max(raw_open, raw_close) * 1.006,
                "low": min(raw_open, raw_close) * 0.994,
                "close": raw_close,
                "adjusted_close": adjusted,
                "volume": 1_500_000 + index * 1_000,
                "dividends": 0.0,
                "stock_splits": 2.0 if symbol == "AAPL" and index == 150 else 0.0,
            }
            if symbol == "AAPL" and index == 110:
                row["volume"] = None
            if symbol == "AAPL" and index == 140:
                row["high"] = None
            if symbol == "AAPL" and index == 270:
                row["adjusted_close"] = None
            rows.append(row)
        histories[symbol] = rows
    return histories, sessions


def test_cached_results_exactly_match_complete_canonical_results() -> None:
    histories, sessions = _histories()
    snapshot_dates = [sessions[90], sessions[150], sessions[206], sessions[270], sessions[320]]
    rules = _rules()
    oracle = AnalysisService(None, rules, WATCHLIST)
    cached = AnalysisService(None, rules, WATCHLIST)
    cached.prime_historical_features(histories, snapshot_dates)

    for snapshot_date in snapshot_dates:
        expected = oracle.analyze_loaded_many_as_of(
            ["AAPL", "SPY"], histories, snapshot_date
        )
        actual = cached.analyze_loaded_many_as_of(
            ["AAPL", "SPY"], histories, snapshot_date
        )
        assert actual.market_context == expected.market_context
        assert [asdict(result) for result in actual.results] == [
            asdict(result) for result in expected.results
        ]


def test_cache_is_right_bounded_when_extreme_future_rows_are_present() -> None:
    histories, sessions = _histories()
    cutoff = sessions[250]
    truncated = {
        symbol: [row for row in rows if str(row["trade_date"]) <= cutoff]
        for symbol, rows in histories.items()
    }
    for symbol, rows in histories.items():
        rows[-1] = {
            **rows[-1],
            "open": 1_000_000.0,
            "high": 1_100_000.0,
            "low": 0.01,
            "close": 1_000_000.0,
            "adjusted_close": 1_000_000.0,
            "volume": 999_999_999,
        }
    rules = _rules()
    cache = AnalysisService(None, rules, WATCHLIST)
    cache.prime_historical_features(histories, [cutoff])
    cached = cache.analyze_loaded_many_as_of(["AAPL"], histories, cutoff)
    oracle = AnalysisService(None, rules, WATCHLIST).analyze_loaded_many_as_of(
        ["AAPL"], truncated, cutoff
    )

    assert cached.market_context == oracle.market_context
    assert asdict(cached.results[0]) == asdict(oracle.results[0])


def test_primed_dates_do_not_call_full_history_indicator_or_alignment_functions(
    monkeypatch,
) -> None:
    histories, sessions = _histories()
    cutoff = sessions[300]
    rules = _rules()
    service = AnalysisService(None, rules, WATCHLIST)
    service.prime_historical_features(histories, [cutoff])
    calls = {"indicators": 0, "relative": 0}
    original_indicators = analysis_engine.calculate_indicators
    original_relative = analysis_engine.calculate_relative_strength_metrics

    def counted_indicators(*args, **kwargs):
        calls["indicators"] += 1
        return original_indicators(*args, **kwargs)

    def counted_relative(*args, **kwargs):
        calls["relative"] += 1
        return original_relative(*args, **kwargs)

    monkeypatch.setattr(analysis_engine, "calculate_indicators", counted_indicators)
    monkeypatch.setattr(
        analysis_engine, "calculate_relative_strength_metrics", counted_relative
    )
    service.analyze_loaded_many_as_of(["AAPL", "SPY"], histories, cutoff)
    assert calls == {"indicators": 0, "relative": 0}

    unprimed = AnalysisService(None, rules, WATCHLIST)
    unprimed.analyze_loaded_many_as_of(["AAPL"], histories, cutoff)
    assert calls == {"indicators": 1, "relative": 1}


def test_history_prefix_lookup_does_not_backfill_missing_sessions() -> None:
    histories, sessions = _histories()
    cache = HistoricalFeatureCache(histories, "SPY", [sessions[206]])
    prefix = cache.history_as_of("AAPL", sessions[206])

    assert all(str(row["trade_date"]) <= sessions[206] for row in prefix)
    assert sessions[205] not in {str(row["trade_date"]) for row in prefix}
    assert prefix[-1]["trade_date"] == sessions[206]
