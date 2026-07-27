"""Rule-based exit evaluation shared between the backtester and live portfolio tracking.

``evaluate_rule_based_exit`` is the single source of truth for the score_v1
classification/regime/score/SMA200/holding-period exit rules: the simulator's
engine and the live portfolio digest both call it so a real holding is judged
by exactly the same rules a backtest would have used to close it. It does not
cover the intraday stop-loss/trailing-stop/profit-target checks, which need
bar-level high/low data; see :mod:`stock_scrapper.portfolio` for the
close-price analog used outside the simulator.
"""

from __future__ import annotations

import math
from typing import Any

from stock_scrapper.backtesting.config import BacktestConfig
from stock_scrapper.models.analysis_models import AnalysisResult


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def evaluate_rule_based_exit(
    config: BacktestConfig,
    result: AnalysisResult,
    holding_period_days: int,
) -> str | None:
    """Return the first-triggered configured exit reason, or ``None`` if none apply."""
    thresholds = config.exit_thresholds
    if thresholds.exit_on_stress and result.market_regime == "Stress":
        return "Market entered Stress"
    if result.classification in set(thresholds.classifications):
        return f"Classification became {result.classification}"
    if result.risk_score is None or result.risk_score > thresholds.maximum_risk_score:
        return "Risk score exceeded the exit maximum or became unavailable"
    if result.opportunity_score is None or result.opportunity_score < thresholds.minimum_opportunity_score:
        return "Opportunity score fell below the exit threshold"
    if result.confidence_score is None or result.confidence_score < thresholds.minimum_confidence_score:
        return "Confidence score fell below the exit threshold"
    distance200 = _number(result.indicators.get("distance_from_sma200"))
    if thresholds.exit_below_sma200 and distance200 is not None and distance200 < 0:
        return "Price closed below the 200-day moving average"
    if config.maximum_holding_period is not None and holding_period_days >= config.maximum_holding_period:
        return "Maximum holding period reached"
    return None


def evaluate_price_stop(
    config: BacktestConfig,
    *,
    average_cost_basis: float,
    highest_price_since_entry: float,
    latest_price: float,
) -> str | None:
    """Return the triggered stop-loss/trailing-stop reason from closing prices.

    This is a close-price analog of the simulator's intraday
    ``Engine._stop_events``, which uses bar high/low. A live holding is only
    checked once per day against the latest stored close, so it can miss an
    intraday touch that reversed by the close; it exists to give a
    directionally correct daily signal, not to replay the simulator exactly.
    """
    stop_levels: list[tuple[str, float]] = []
    if config.stop_loss is not None:
        stop_levels.append(("Stop loss", average_cost_basis * (1.0 - config.stop_loss)))
    if config.trailing_stop is not None:
        stop_levels.append(("Trailing stop", highest_price_since_entry * (1.0 - config.trailing_stop)))
    if not stop_levels:
        return None
    reason, trigger = max(stop_levels, key=lambda item: item[1])
    return reason if latest_price <= trigger else None
