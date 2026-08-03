from __future__ import annotations

import pytest

from stock_scrapper.trading.config import validate_trading_rules


def _rules(**overrides: object) -> dict[str, object]:
    base = dict(
        trading_rules_version="trade-v1",
        account_value=10000.0,
        available_cash=10000.0,
        max_position_weight=0.15,
        cash_reserve=0.05,
        max_open_positions=10,
        max_trade_dollar_amount=2000.0,
        min_trade_dollar_amount=100.0,
        max_new_buys_per_run=3,
        require_strong_candidate_for_buy=True,
        auto_execute=False,
    )
    base.update(overrides)
    return base


def test_validate_trading_rules_accepts_a_valid_config() -> None:
    validated = validate_trading_rules(_rules())
    assert validated["account_value"] == 10000.0
    assert validated["available_cash"] == 10000.0
    assert validated["auto_execute"] is False


def test_validate_trading_rules_rejects_available_cash_exceeding_account_value() -> None:
    with pytest.raises(ValueError, match="must not exceed account_value"):
        validate_trading_rules(_rules(account_value=1000.0, available_cash=2000.0))


def test_validate_trading_rules_rejects_negative_account_value() -> None:
    with pytest.raises(ValueError, match="account_value must be a nonnegative number"):
        validate_trading_rules(_rules(account_value=-1.0))


def test_validate_trading_rules_accepts_zero_available_cash() -> None:
    validated = validate_trading_rules(_rules(available_cash=0.0))
    assert validated["available_cash"] == 0.0


def test_validate_trading_rules_rejects_auto_execute_true() -> None:
    with pytest.raises(ValueError, match="not supported yet"):
        validate_trading_rules(_rules(auto_execute=True))


def test_validate_trading_rules_rejects_position_weight_exceeding_reserve_headroom() -> None:
    with pytest.raises(ValueError, match="leave room for cash_reserve"):
        validate_trading_rules(_rules(max_position_weight=0.98, cash_reserve=0.05))


def test_validate_trading_rules_rejects_min_above_max_trade_dollar_amount() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        validate_trading_rules(_rules(min_trade_dollar_amount=3000.0, max_trade_dollar_amount=2000.0))


def test_validate_trading_rules_rejects_non_bool_require_strong_candidate() -> None:
    with pytest.raises(ValueError, match="require_strong_candidate_for_buy"):
        validate_trading_rules(_rules(require_strong_candidate_for_buy="yes"))
