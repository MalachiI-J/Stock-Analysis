"""Load and strictly validate config/trading_rules.yaml."""

from __future__ import annotations

from typing import Any

# Floor for account_value/available_cash (main.py account-set, and here as the
# backstop). Coupled to min_trade_dollar_amount's default only by coincidence --
# raising min_trade_dollar_amount later doesn't move this floor, but a $100
# account is the smallest this tool is designed to size a $100 minimum trade
# against at all (see build_recommendations()'s reserve/position-cap overrides).
MIN_ACCOUNT_DOLLARS = 100.0


def validate_trading_rules(rules: dict[str, Any]) -> dict[str, Any]:
    """Validate the trade-recommendation sizing/restriction configuration."""
    if not isinstance(rules, dict):
        raise ValueError("trading_rules.yaml must contain a mapping")

    def _positive_number(key: str) -> float:
        value = rules.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{key} must be a positive number")
        return float(value)

    def _nonnegative_number(key: str) -> float:
        value = rules.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{key} must be a nonnegative number")
        return float(value)

    def _positive_int(key: str) -> int:
        value = rules.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
        return value

    def _fraction(key: str) -> float:
        value = rules.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not (0.0 <= value < 1.0):
            raise ValueError(f"{key} must be a number in [0, 1)")
        return float(value)

    account_value = _nonnegative_number("account_value")
    available_cash = _nonnegative_number("available_cash")
    if account_value < MIN_ACCOUNT_DOLLARS:
        raise ValueError(f"account_value must be at least ${MIN_ACCOUNT_DOLLARS:,.2f}")
    # $0 stays legal (no cash to invest right now, e.g. fully deployed elsewhere) --
    # it's specifically a *nonzero but unusably small* amount that's rejected, since
    # nothing below the floor could ever clear min_trade_dollar_amount anyway.
    if 0.0 < available_cash < MIN_ACCOUNT_DOLLARS:
        raise ValueError(f"available_cash must be $0 or at least ${MIN_ACCOUNT_DOLLARS:,.2f}")
    if available_cash > account_value:
        raise ValueError("available_cash must not exceed account_value")
    max_position_weight = _fraction("max_position_weight")
    if max_position_weight <= 0.0:
        raise ValueError("max_position_weight must be greater than 0")
    cash_reserve = _fraction("cash_reserve")
    if max_position_weight > 1.0 - cash_reserve:
        raise ValueError("max_position_weight must leave room for cash_reserve (max_position_weight <= 1 - cash_reserve)")
    max_open_positions = _positive_int("max_open_positions")
    max_trade_dollar_amount = _positive_number("max_trade_dollar_amount")
    min_trade_dollar_amount = _positive_number("min_trade_dollar_amount")
    if min_trade_dollar_amount > max_trade_dollar_amount:
        raise ValueError("min_trade_dollar_amount must not exceed max_trade_dollar_amount")
    max_new_buys_per_run = _positive_int("max_new_buys_per_run")

    require_strong_candidate_for_buy = rules.get("require_strong_candidate_for_buy")
    if not isinstance(require_strong_candidate_for_buy, bool):
        raise ValueError("require_strong_candidate_for_buy must be true or false")

    auto_execute = rules.get("auto_execute")
    if not isinstance(auto_execute, bool):
        raise ValueError("auto_execute must be true or false")
    if auto_execute:
        raise ValueError(
            "auto_execute is not supported yet — automated order placement has not been built. "
            "Leave this false."
        )

    trading_rules_version = rules.get("trading_rules_version")
    if not isinstance(trading_rules_version, str) or not trading_rules_version.strip():
        raise ValueError("trading_rules_version must be a non-empty string")

    return {
        "trading_rules_version": trading_rules_version,
        "account_value": account_value,
        "available_cash": available_cash,
        "max_position_weight": max_position_weight,
        "cash_reserve": cash_reserve,
        "max_open_positions": max_open_positions,
        "max_trade_dollar_amount": max_trade_dollar_amount,
        "min_trade_dollar_amount": min_trade_dollar_amount,
        "max_new_buys_per_run": max_new_buys_per_run,
        "require_strong_candidate_for_buy": require_strong_candidate_for_buy,
        "auto_execute": auto_execute,
    }
