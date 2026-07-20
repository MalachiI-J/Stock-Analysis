from stock_scrapper.processing.validation import validate_price_record


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
