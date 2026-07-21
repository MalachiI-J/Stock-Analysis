"""Consistent adjusted-OHLC and corporate-action calculations.

Backtests use a total-return adjusted price basis. The reported adjusted close
is authoritative; open, high, and low are multiplied by the same per-session
factor. A missing or invalid factor remains unavailable, which prevents an
order from executing at a fabricated price.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class AdjustedOHLC:
    """A daily bar represented on one total-return-adjusted price basis."""

    symbol: str | None
    trade_date: str | None
    raw_open: float | None
    raw_high: float | None
    raw_low: float | None
    raw_close: float | None
    adjusted_open: float | None
    adjusted_high: float | None
    adjusted_low: float | None
    adjusted_close: float | None
    adjustment_factor: float | None
    volume: float | None
    dividends: float | None
    stock_splits: float | None
    adjustment_available: bool
    execution_price_available: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SplitAdjustment:
    """Quantity/cost-basis transformation for a raw-price position."""

    original_quantity: float
    original_average_cost: float
    split_ratio: float | None
    adjusted_quantity: float
    adjusted_average_cost: float
    applied: bool
    reason: str

    @property
    def original_cost_basis(self) -> float:
        return self.original_quantity * self.original_average_cost

    @property
    def adjusted_cost_basis(self) -> float:
        return self.adjusted_quantity * self.adjusted_average_cost


def _optional_number(value: Any, field_name: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not isfinite(result):
        return None
    return result


def calculate_adjustment_factor(raw_close: Any, adjusted_close: Any) -> float | None:
    """Return ``adjusted_close / raw_close`` or ``None`` when unavailable."""
    raw = _optional_number(raw_close, "raw_close")
    adjusted = _optional_number(adjusted_close, "adjusted_close")
    if raw is None or adjusted is None or raw <= 0 or adjusted <= 0:
        return None
    factor = adjusted / raw
    return factor if isfinite(factor) and factor > 0 else None


def adjust_price(raw_price: Any, adjustment_factor: float | None) -> float | None:
    """Adjust one raw OHLC value without inventing a missing factor or price."""
    price = _optional_number(raw_price, "raw_price")
    if price is None or adjustment_factor is None or adjustment_factor <= 0:
        return None
    adjusted = price * adjustment_factor
    return adjusted if isfinite(adjusted) and adjusted > 0 else None


def build_adjusted_ohlc(row: Mapping[str, Any]) -> AdjustedOHLC:
    """Build a consistently adjusted daily bar from a stored price row."""
    if not isinstance(row, Mapping):
        raise ValueError("price row must be a mapping")
    raw_open = _optional_number(row.get("open"), "open")
    raw_high = _optional_number(row.get("high"), "high")
    raw_low = _optional_number(row.get("low"), "low")
    raw_close = _optional_number(row.get("close"), "close")
    reported_adjusted_close = _optional_number(row.get("adjusted_close"), "adjusted_close")
    factor = calculate_adjustment_factor(raw_close, reported_adjusted_close)

    adjusted_open = adjust_price(raw_open, factor)
    adjusted_high = adjust_price(raw_high, factor)
    adjusted_low = adjust_price(raw_low, factor)
    adjusted_close = reported_adjusted_close if reported_adjusted_close is not None and reported_adjusted_close > 0 else None
    volume = _optional_number(row.get("volume"), "volume")
    dividends = _optional_number(row.get("dividends"), "dividends")
    splits = _optional_number(row.get("stock_splits"), "stock_splits")

    return AdjustedOHLC(
        symbol=str(row["symbol"]).upper() if row.get("symbol") is not None else None,
        trade_date=str(row["trade_date"]) if row.get("trade_date") is not None else None,
        raw_open=raw_open,
        raw_high=raw_high,
        raw_low=raw_low,
        raw_close=raw_close,
        adjusted_open=adjusted_open,
        adjusted_high=adjusted_high,
        adjusted_low=adjusted_low,
        adjusted_close=adjusted_close,
        adjustment_factor=factor,
        volume=volume,
        dividends=dividends,
        stock_splits=splits,
        adjustment_available=factor is not None,
        execution_price_available=adjusted_open is not None,
    )


def adjust_ohlc_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Dictionary-returning compatibility wrapper around :func:`build_adjusted_ohlc`."""
    return build_adjusted_ohlc(row).to_dict()


def adjust_position_for_split(quantity: Any, average_cost: Any, split_ratio: Any) -> SplitAdjustment:
    """Apply a split ratio to raw shares while preserving total cost basis.

    Yahoo-style zero means no split on the session. ``None`` likewise records
    that no split action was supplied. A positive ratio below one represents a
    reverse split; a ratio above one represents a conventional split.
    """
    quantity_value = _optional_number(quantity, "quantity")
    average_cost_value = _optional_number(average_cost, "average_cost")
    if quantity_value is None or quantity_value < 0:
        raise ValueError("quantity must be a finite nonnegative number")
    if average_cost_value is None or average_cost_value < 0:
        raise ValueError("average_cost must be a finite nonnegative number")
    ratio = _optional_number(split_ratio, "split_ratio")
    if ratio is None:
        return SplitAdjustment(
            quantity_value,
            average_cost_value,
            None,
            quantity_value,
            average_cost_value,
            False,
            "No split ratio reported",
        )
    if ratio < 0:
        raise ValueError("split_ratio cannot be negative")
    if ratio == 0 or ratio == 1:
        return SplitAdjustment(
            quantity_value,
            average_cost_value,
            ratio,
            quantity_value,
            average_cost_value,
            False,
            "No split action",
        )
    return SplitAdjustment(
        original_quantity=quantity_value,
        original_average_cost=average_cost_value,
        split_ratio=ratio,
        adjusted_quantity=quantity_value * ratio,
        adjusted_average_cost=average_cost_value / ratio,
        applied=True,
        reason="Forward split" if ratio > 1 else "Reverse split",
    )


def reconcile_position_for_price_basis(
    quantity: Any, average_cost: Any, split_ratio: Any, *, price_basis: str = "adjusted"
) -> SplitAdjustment:
    """Avoid double-counting a split when prices already use adjusted OHLC.

    On an adjusted basis, historical prices already encode splits and synthetic
    adjusted-share quantity remains constant. Raw-price simulations must apply
    the split to quantity and inverse-adjust average cost.
    """
    if price_basis not in {"adjusted", "raw"}:
        raise ValueError("price_basis must be 'adjusted' or 'raw'")
    if price_basis == "raw":
        return adjust_position_for_split(quantity, average_cost, split_ratio)
    quantity_value = _optional_number(quantity, "quantity")
    cost_value = _optional_number(average_cost, "average_cost")
    if quantity_value is None or quantity_value < 0 or cost_value is None or cost_value < 0:
        raise ValueError("quantity and average_cost must be finite nonnegative numbers")
    ratio = _optional_number(split_ratio, "split_ratio")
    if ratio is not None and ratio < 0:
        raise ValueError("split_ratio cannot be negative")
    return SplitAdjustment(
        original_quantity=quantity_value,
        original_average_cost=cost_value,
        split_ratio=ratio,
        adjusted_quantity=quantity_value,
        adjusted_average_cost=cost_value,
        applied=False,
        reason="Split already incorporated in adjusted prices",
    )


def position_market_value(quantity: Any, price: Any) -> float | None:
    """Return position market value, preserving an unavailable price as ``None``."""
    quantity_value = _optional_number(quantity, "quantity")
    price_value = _optional_number(price, "price")
    if quantity_value is None or price_value is None:
        return None
    if quantity_value < 0 or price_value < 0:
        raise ValueError("quantity and price must be nonnegative")
    return quantity_value * price_value


def cash_dividend_credit(
    quantity: Any, dividend_per_share: Any, *, price_basis: str = "adjusted"
) -> float | None:
    """Return dividend cash only for a raw-price simulation.

    Total-return adjusted closes already incorporate dividends, so their cash
    credit is intentionally zero to prevent double counting. Missing dividend
    information remains unavailable for a raw-price simulation.
    """
    if price_basis not in {"adjusted", "raw"}:
        raise ValueError("price_basis must be 'adjusted' or 'raw'")
    quantity_value = _optional_number(quantity, "quantity")
    if quantity_value is None or quantity_value < 0:
        raise ValueError("quantity must be a finite nonnegative number")
    if price_basis == "adjusted":
        return 0.0
    dividend = _optional_number(dividend_per_share, "dividend_per_share")
    if dividend is None:
        return None
    if dividend < 0:
        raise ValueError("dividend_per_share cannot be negative")
    return quantity_value * dividend


# Readable aliases for engine integration.
adjust_ohlc = build_adjusted_ohlc
apply_split = adjust_position_for_split

