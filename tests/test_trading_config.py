from __future__ import annotations

import pytest

from stock_scrapper.trading.config import validate_trading_rules


def _rules(**overrides: object) -> dict[str, object]:
    base = dict(
        trading_rules_version="trade-v1",
        starting_capital=10000.0,
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
    assert validated["starting_capital"] == 10000.0
    assert validated["auto_execute"] is False


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
