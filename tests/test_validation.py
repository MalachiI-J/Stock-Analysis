from datetime import date

from stock_scrapper.processing.validation import validate_price_record


def test_validation_flags_stale_record_beyond_default_threshold() -> None:
    record = {"symbol": "AAPL", "trade_date": "2000-01-03", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}

    issues = validate_price_record(record, now_date=date(2026, 1, 1))

    assert "stale_record" in {issue["issue_type"] for issue in issues}


def test_validation_max_age_days_permits_older_history_without_a_false_positive() -> None:
    # 20 years old, well beyond the default 3650-day (10-year) threshold, but exactly
    # the kind of row a 20-year historical_lookback_years config legitimately collects.
    record = {"symbol": "AAPL", "trade_date": "2006-01-03", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}

    issues = validate_price_record(record, now_date=date(2026, 1, 1), max_age_days=20 * 366 + 30)

    assert "stale_record" not in {issue["issue_type"] for issue in issues}


def test_validation_reports_missing_and_invalid_values() -> None:
    record = {
        "symbol": "",
        "trade_date": "2024-01-02",
        "open": 100.0,
        "high": 101.0,
        "low": 102.0,
        "close": 0.0,
        "adjusted_close": 100.0,
        "volume": -5,
        "dividends": 0.0,
        "stock_splits": 0.0,
        "data_source": "test",
        "collected_at": "2024-01-02T00:00:00",
    }

    issues = validate_price_record(record, previous_close=100.0)
    issue_types = {issue["issue_type"] for issue in issues}

    assert "missing_symbol" in issue_types
    assert "negative_volume" in issue_types
    assert "zero_close_price" in issue_types
    assert "high_low_inversion" in issue_types
