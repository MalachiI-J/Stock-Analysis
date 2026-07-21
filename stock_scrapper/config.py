"""Configuration loading helpers for the Stock Scrapper project."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml

# Default settings provide a working local configuration even before a user edits YAML values.
DEFAULT_SETTINGS: dict[str, Any] = {
    "app_name": "Stock Scrapper",
    "watchlist_path": "config/watchlist.csv",
    "database_path": "data/market.db",
    "raw_data_dir": "data/raw",
    "processed_data_dir": "data/processed",
    "reports_dir": "reports",
    "logs_dir": "logs",
    "historical_lookback_years": 5,
    "data_source": "yfinance",
    "retry_count": 3,
    "retry_delay_seconds": 2,
    "logging_level": "INFO",
    "archive_raw_downloads": False,
    "open_reports_automatically": False,
    "market_data": {
        "exchange": "XNYS", "timezone": "America/New_York",
        "provider_delay_minutes": 30, "incomplete_bar_policy": "exclude",
        "recent_overlap_sessions": 10, "corporate_action_refresh_sessions": 90,
        "scheduled_full_refresh_days": 7,
    },
    "universes": {
        "candidates": ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "WMT", "XOM"],
        "benchmark": {"symbol": "SPY"}, "market_context": ["SPY", "QQQ", "IWM"],
        "defensive_context": ["TLT", "GLD"],
    },
    "warmup": {"sessions": 252, "policy": "shift_start"},
    "revision_policy": {"version": "revision-v2", "price_absolute_tolerance": 0.0001,
        "price_relative_tolerance": 0.000001, "adjusted_price_absolute_tolerance": 0.0001,
        "adjusted_price_relative_tolerance": 0.000001, "dividend_absolute_tolerance": 0.00000001,
        "split_absolute_tolerance": 0.00000001, "volume_tolerance": 0, "store_precision_noise": False},
}


def _resolve_path(base_dir: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path.resolve())
    return str((base_dir / path).resolve())


def _ensure_default_files(base_dir: Path) -> tuple[Path, Path]:
    config_dir = base_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    settings_path = config_dir / "settings.yaml"
    watchlist_path = config_dir / "watchlist.csv"

    if not settings_path.exists():
        settings_payload = {
            "app_name": DEFAULT_SETTINGS["app_name"],
            "watchlist_path": "config/watchlist.csv",
            "database_path": "data/market.db",
            "raw_data_dir": "data/raw",
            "processed_data_dir": "data/processed",
            "reports_dir": "reports",
            "logs_dir": "logs",
            "historical_lookback_years": 5,
            "data_source": "yfinance",
            "retry_count": 3,
            "retry_delay_seconds": 2,
            "logging_level": "INFO",
            "archive_raw_downloads": False,
            "open_reports_automatically": False,
        }
        settings_path.write_text(yaml.safe_dump(settings_payload, sort_keys=False), encoding="utf-8")

    if not watchlist_path.exists():
        with watchlist_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["symbol"])
            for symbol in [
                "AAPL",
                "MSFT",
                "AMZN",
                "GOOGL",
                "META",
                "NVDA",
                "TSLA",
                "JPM",
                "WMT",
                "XOM",
                "SPY",
                "QQQ",
                "IWM",
                "TLT",
                "GLD",
            ]:
                writer.writerow([symbol])

    return settings_path, watchlist_path


def load_config(base_dir: str | Path | None = None) -> dict[str, Any]:
    """Load settings from YAML and resolve relative paths against the base directory."""
    base_path = Path(base_dir or Path(__file__).resolve().parent.parent)
    settings_path, _ = _ensure_default_files(base_path)

    with settings_path.open("r", encoding="utf-8") as handle:
        loaded_settings = yaml.safe_load(handle) or {}

    merged_settings = dict(DEFAULT_SETTINGS)
    merged_settings.update(loaded_settings)

    merged_settings["watchlist_path"] = _resolve_path(base_path, merged_settings.get("watchlist_path", DEFAULT_SETTINGS["watchlist_path"]))
    merged_settings["database_path"] = _resolve_path(base_path, merged_settings.get("database_path", DEFAULT_SETTINGS["database_path"]))
    merged_settings["raw_data_dir"] = _resolve_path(base_path, merged_settings.get("raw_data_dir", DEFAULT_SETTINGS["raw_data_dir"]))
    merged_settings["processed_data_dir"] = _resolve_path(base_path, merged_settings.get("processed_data_dir", DEFAULT_SETTINGS["processed_data_dir"]))
    merged_settings["reports_dir"] = _resolve_path(base_path, merged_settings.get("reports_dir", DEFAULT_SETTINGS["reports_dir"]))
    merged_settings["logs_dir"] = _resolve_path(base_path, merged_settings.get("logs_dir", DEFAULT_SETTINGS["logs_dir"]))

    return merged_settings


def load_watchlist(path: str | Path | None = None) -> list[str]:
    """Load the configured watchlist from a CSV file."""
    path_obj = Path(path) if path is not None else None
    if path_obj is None:
        config = load_config()
        path_obj = Path(config["watchlist_path"])

    if not path_obj.exists():
        raise FileNotFoundError(f"Watchlist file does not exist: {path_obj}")

    with path_obj.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    symbols = [row["symbol"].strip().upper() for row in rows if row.get("symbol", "").strip()]
    return symbols


def load_universes(config: dict[str, Any]) -> dict[str, Any]:
    """Return normalized role-aware universes, falling back to the legacy CSV."""
    raw = config.get("universes") or {}
    candidates = raw.get("candidates") or load_watchlist(config["watchlist_path"])
    benchmark = raw.get("benchmark", {"symbol": "SPY"})
    benchmark_symbol = benchmark.get("symbol", "SPY") if isinstance(benchmark, dict) else benchmark
    return {
        "candidates": list(dict.fromkeys(str(v).strip().upper() for v in candidates)),
        "benchmark": str(benchmark_symbol).strip().upper(),
        "market_context": list(dict.fromkeys(str(v).strip().upper() for v in raw.get("market_context", ["SPY", "QQQ", "IWM"]))),
        "defensive_context": list(dict.fromkeys(str(v).strip().upper() for v in raw.get("defensive_context", ["TLT", "GLD"]))),
    }


def validate_universes(universes: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if not universes.get("candidates"): warnings.append("Candidate universe is empty")
    if not universes.get("benchmark"): warnings.append("Benchmark symbol is missing")
    if universes.get("benchmark") in universes.get("candidates", []):
        warnings.append("Benchmark is also a candidate; comparisons may be ambiguous")
    return warnings
