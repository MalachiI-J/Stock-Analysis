from __future__ import annotations

from datetime import date

from stock_scrapper.processing.fundamentals_features import (
    debt_to_equity,
    earnings_growth_yoy,
    fundamentals_as_of,
    fundamentals_features_as_of,
    price_to_book,
    return_on_equity,
    revenue_growth_yoy,
    trailing_four_quarter_sum,
    trailing_pe,
)


def _flow(concept: str, start: str, end: str, filed: str, value: float) -> dict[str, object]:
    return {"concept": concept, "period_start": start, "period_end": end, "filed_date": filed, "value": value}


def _instant(concept: str, end: str, filed: str, value: float) -> dict[str, object]:
    return {"concept": concept, "period_start": None, "period_end": end, "filed_date": filed, "value": value}


def _net_income_quarters() -> list[dict[str, object]]:
    return [
        _flow("net_income", "2022-01-01", "2022-03-31", "2022-04-15", 100.0),
        _flow("net_income", "2022-04-01", "2022-06-30", "2022-07-15", 110.0),
        _flow("net_income", "2022-07-01", "2022-09-30", "2022-10-15", 120.0),
        _flow("net_income", "2022-10-01", "2022-12-31", "2023-01-15", 130.0),
        _flow("net_income", "2023-01-01", "2023-03-31", "2023-04-15", 140.0),
        _flow("net_income", "2023-04-01", "2023-06-30", "2023-07-15", 150.0),
        _flow("net_income", "2023-07-01", "2023-09-30", "2023-10-15", 160.0),
        _flow("net_income", "2023-10-01", "2023-12-31", "2024-01-15", 170.0),
        # Noise: a 9-month cumulative duration and a full fiscal-year duration
        # tagged under the same concept -- both must be excluded from the
        # quarter-band TTM sum, not double-counted alongside the discrete quarters.
        _flow("net_income", "2023-01-01", "2023-09-30", "2023-10-15", 450.0),
        _flow("net_income", "2023-01-01", "2023-12-31", "2024-01-15", 620.0),
    ]


def test_trailing_four_quarter_sum_excludes_future_and_non_quarterly_durations() -> None:
    records = _net_income_quarters()

    # As of 2023-11-01: Q3'23 (filed 2023-10-15) is known, Q4'23 (filed 2024-01-15) is not.
    assert trailing_four_quarter_sum(records, "net_income", date(2023, 11, 1)) == 130.0 + 140.0 + 150.0 + 160.0

    # As of 2024-02-01: Q4'23 is now known too, replacing Q4'22 in the trailing window.
    assert trailing_four_quarter_sum(records, "net_income", date(2024, 2, 1)) == 140.0 + 150.0 + 160.0 + 170.0


def test_trailing_four_quarter_sum_returns_none_with_fewer_than_four_quarters() -> None:
    records = _net_income_quarters()
    # Only Q1'22 has been filed by this date.
    assert trailing_four_quarter_sum(records, "net_income", date(2022, 5, 1)) is None


def test_trailing_four_quarter_sum_ignores_a_fact_filed_after_as_of() -> None:
    """The core lookahead-safety guarantee this module exists to provide."""
    records = [
        _flow("net_income", "2022-01-01", "2022-03-31", "2022-04-15", 100.0),
        _flow("net_income", "2022-04-01", "2022-06-30", "2022-07-15", 110.0),
        _flow("net_income", "2022-07-01", "2022-09-30", "2022-10-15", 120.0),
        # This quarter is filed one day AFTER the as_of date being tested below.
        _flow("net_income", "2022-10-01", "2022-12-31", "2023-01-16", 999999.0),
    ]
    assert trailing_four_quarter_sum(records, "net_income", date(2023, 1, 15)) is None
    assert trailing_four_quarter_sum(records, "net_income", date(2023, 1, 16)) == 100.0 + 110.0 + 120.0 + 999999.0


def test_fundamentals_as_of_returns_none_for_missing_concept() -> None:
    result = fundamentals_as_of([_flow("net_income", "2022-01-01", "2022-03-31", "2022-04-15", 100.0)], date(2023, 1, 1))
    assert result["revenue"] is None
    assert result["assets"] is None


def test_fundamentals_as_of_instant_concept_uses_latest_known_balance_and_respects_lookahead() -> None:
    records = [
        _instant("assets", "2022-09-30", "2022-10-15", 4800.0),
        _instant("assets", "2022-12-31", "2023-01-15", 5000.0),
    ]
    assert fundamentals_as_of(records, date(2023, 1, 10))["assets"] == 4800.0  # 2023-01-15 fact not yet filed
    assert fundamentals_as_of(records, date(2023, 1, 15))["assets"] == 5000.0


def test_revenue_and_earnings_growth_yoy_compare_two_ttm_windows_one_year_apart() -> None:
    net_income_records = _net_income_quarters()
    growth = earnings_growth_yoy(net_income_records, date(2024, 2, 1))
    # current TTM (as of 2024-02-01): Q1'23+Q2'23+Q3'23+Q4'23 = 140+150+160+170 = 620
    # prior TTM (as of 2023-02-01): Q1'22+Q2'22+Q3'22+Q4'22 = 100+110+120+130 = 460
    assert growth == (620.0 - 460.0) / 460.0

    revenue_records = [dict(record, concept="revenue") for record in _net_income_quarters()]
    assert revenue_growth_yoy(revenue_records, date(2024, 2, 1)) == (620.0 - 460.0) / 460.0


def test_growth_yoy_returns_none_when_either_window_is_unavailable() -> None:
    # Only two years' worth of quarters exist -- the "prior year" window one year
    # before the earliest as_of that has a full trailing four quarters is empty.
    records = _net_income_quarters()
    assert earnings_growth_yoy(records, date(2022, 12, 1)) is None  # no full prior-year TTM to compare against


def test_growth_yoy_handles_feb_29_as_of_date() -> None:
    records = _net_income_quarters()
    # Should not raise even though 2024-02-29 - 1 year (2023-02-29) doesn't exist.
    result = earnings_growth_yoy(records, date(2024, 2, 29))
    assert result == (620.0 - 460.0) / 460.0


def test_trailing_pe_price_to_book_debt_to_equity_return_on_equity_are_none_safe() -> None:
    assert trailing_pe(150.0, 5.0) == 30.0
    assert trailing_pe(None, 5.0) is None
    assert trailing_pe(150.0, None) is None
    assert trailing_pe(150.0, 0.0) is None

    assert price_to_book(150.0, 1000.0, 100.0) == 15.0  # book value/share = 10, 150/10 = 15
    assert price_to_book(150.0, 1000.0, None) is None
    assert price_to_book(150.0, 1000.0, 0.0) is None

    assert debt_to_equity(500.0, 1000.0) == 0.5
    assert debt_to_equity(500.0, None) is None
    assert debt_to_equity(500.0, 0.0) is None

    assert return_on_equity(200.0, 1000.0) == 0.2
    assert return_on_equity(200.0, None) is None
    assert return_on_equity(200.0, 0.0) is None


def test_fundamentals_features_as_of_assembles_the_full_predict_v5_feature_set() -> None:
    records = _net_income_quarters() + [
        dict(record, concept="revenue", value=record["value"] * 10) for record in _net_income_quarters()
        if record["concept"] == "net_income"
    ] + [
        _instant("stockholders_equity", "2023-12-31", "2024-01-15", 2000.0),
        _instant("liabilities", "2023-12-31", "2024-01-15", 1000.0),
        _instant("shares_outstanding", "2023-12-31", "2024-01-15", 100.0),
    ]

    features = fundamentals_features_as_of(records, date(2024, 2, 1), price=50.0)

    assert set(features) == {
        "trailing_pe", "price_to_book", "debt_to_equity",
        "return_on_equity", "revenue_growth_yoy", "earnings_growth_yoy",
    }
    # eps_diluted concept is absent from this fixture -> trailing_pe unavailable, not fabricated as 0.
    assert features["trailing_pe"] is None
    assert features["price_to_book"] == 50.0 / (2000.0 / 100.0)
    assert features["debt_to_equity"] == 1000.0 / 2000.0
    assert features["return_on_equity"] == 620.0 / 2000.0
    assert features["revenue_growth_yoy"] == (6200.0 - 4600.0) / 4600.0
    assert features["earnings_growth_yoy"] == (620.0 - 460.0) / 460.0
