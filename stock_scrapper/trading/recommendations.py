"""Turn today's scores and real holdings into sized, restriction-bounded trade recommendations.

Advisory only: nothing here places an order. SELL signals are exactly the ones
:func:`stock_scrapper.portfolio.evaluate_holding` already computes (the same rules a
backtest would exit on), so a live sell recommendation and a backtest exit always agree.
BUY candidates come from score_v1's own classification (Strong Candidate / Candidate) —
the experimental prediction models' output is attached to each candidate purely as
displayed context, never as a gate, since neither's accuracy yet warrants driving a
real trade decision (see stock_scrapper/prediction/): predict's probability of beating
the benchmark, and predict-v5's predicted excess-return magnitude plus its
``low_confidence`` flag for statistically extreme, likely-extrapolated predictions.

Account value and available cash are two independent numbers you set directly
(``main.py account-set``, persisted in ``config/trading_rules.yaml``) — not derived
from the portfolio_lots/portfolio_sales tables. Sizing uses exactly those two
numbers regardless of what positions (if any) you've separately recorded via
``portfolio-buy``/``portfolio-sell``; those still drive SELL signals and the
already-held exclusion below, since that reflects real positions this tool tracks,
but they play no part in account_value/available_cash math.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping, Sequence

from stock_scrapper.models.analysis_models import AnalysisResult
from stock_scrapper.portfolio import HoldingAssessment, PortfolioPosition
from stock_scrapper.utilities.business_days import add_business_days

_BUY_ELIGIBLE_CLASSIFICATIONS_STRICT = ("Strong Candidate",)
_BUY_ELIGIBLE_CLASSIFICATIONS_RELAXED = ("Strong Candidate", "Candidate")

# Falls back to this when no signal_validation artifact is on file at all (main.py
# validate-signals hasn't been run yet) -- matches config/prediction_rules.yaml's own
# horizon_days default, so "Hold until next month" still resolves to a real date even
# with zero historical-outcome data behind it.
_DEFAULT_STANDARD_HORIZON_DAYS = 21


@dataclass(slots=True)
class TradeRecommendation:
    """One sized, actionable suggestion — the human still decides whether to place it."""

    symbol: str
    action: str  # "BUY" or "SELL"
    shares: float
    estimated_dollars: float
    reason: str
    model_probability: float | None = None
    predict_v5_excess_return: float | None = None
    predict_v5_low_confidence: bool = False
    # BUY-only historical-outcome context (see stock_scrapper.analysis.signal_validation);
    # SELL items leave all of these at their defaults. classification is the candidate's
    # score_v1 classification at the time of this recommendation, used to look up the
    # matching bucket/checkpoint data -- not necessarily anything about a held position.
    classification: str | None = None
    outcome_p10: float | None = None
    outcome_p90: float | None = None
    best_exit_sessions: int | None = None
    best_exit_date: str | None = None
    checkpoints_compared: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class RecommendationRunResult:
    """Complete outcome of one recommendation run: sizing context plus every suggestion.

    The four sizing-rule fields (``cash_reserve``, ``max_position_weight``,
    ``max_trade_dollar_amount``, ``min_trade_dollar_amount``) are carried through so a
    report can offer a client-side "what if my account/cash were different" resize of
    the already-chosen BUY list without a server round-trip. They default to ``None``
    so older persisted ``recommendations_<date>.summary.json`` files (written before
    this existed) still deserialize; a report should treat ``None`` as "no resize
    widget available for this file," not fail.
    """

    as_of_date: str
    account_value: float
    available_cash: float
    open_position_count: int
    recommendations: list[TradeRecommendation] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    cash_reserve: float | None = None
    max_position_weight: float | None = None
    max_trade_dollar_amount: float | None = None
    min_trade_dollar_amount: float | None = None


def _outcome_context(
    signal_validation: Mapping[str, Any] | None,
    classification: str,
    as_of: date,
    default_standard_horizon_days: int,
) -> dict[str, Any]:
    """Historical-outcome context for one BUY's classification, from the latest
    ``main.py validate-signals`` artifact (see ``_latest_signal_validation_summary``-
    style loading in ``report_builder.py`` / ``main.py``'s ``recommend`` handler).

    Degrades gracefully when there's no artifact yet, or this classification never
    got a ``best_exit_by_classification`` entry (e.g. zero backtested samples): the
    range/best-exit-sessions/checkpoints all come back empty, but ``best_exit_date``
    still resolves against ``default_standard_horizon_days`` -- "hold to the standard
    horizon" is always a real date, even with no richer data behind it yet.
    """
    entry = None
    if isinstance(signal_validation, Mapping):
        by_classification = signal_validation.get("best_exit_by_classification")
        if isinstance(by_classification, Mapping):
            candidate = by_classification.get(classification)
            if isinstance(candidate, Mapping):
                entry = candidate

    if entry is None:
        exit_date = add_business_days(as_of, default_standard_horizon_days).isoformat()
        return {
            "outcome_p10": None, "outcome_p90": None,
            "best_exit_sessions": None, "best_exit_date": exit_date,
            "checkpoints_compared": [],
        }

    standard_sessions = int(entry.get("standard_horizon_sessions") or default_standard_horizon_days)
    checkpoints = [c for c in (entry.get("checkpoints") or []) if isinstance(c, Mapping)]
    checkpoints_sorted = sorted(checkpoints, key=lambda c: c.get("sessions") or 0)
    standard_checkpoint = next((c for c in checkpoints_sorted if c.get("sessions") == standard_sessions), None)

    meaningfully_better = bool(entry.get("best_checkpoint_meaningfully_better"))
    best_sessions = entry.get("best_checkpoint_sessions") if meaningfully_better else None
    effective_sessions = int(best_sessions) if best_sessions is not None else standard_sessions

    checkpoints_compared = [
        {
            "sessions": checkpoint.get("sessions"),
            "date": add_business_days(as_of, int(checkpoint.get("sessions") or 0)).isoformat(),
            "p10_excess_return": checkpoint.get("p10_excess_return"),
            "p90_excess_return": checkpoint.get("p90_excess_return"),
            "outcome_sample_size": checkpoint.get("outcome_sample_size"),
        }
        for checkpoint in checkpoints_sorted
    ]

    return {
        "outcome_p10": standard_checkpoint.get("p10_excess_return") if standard_checkpoint else None,
        "outcome_p90": standard_checkpoint.get("p90_excess_return") if standard_checkpoint else None,
        "best_exit_sessions": int(best_sessions) if best_sessions is not None else None,
        "best_exit_date": add_business_days(as_of, effective_sessions).isoformat(),
        "checkpoints_compared": checkpoints_compared,
    }


def build_recommendations(
    *,
    as_of_date: str,
    results_by_symbol: Mapping[str, AnalysisResult],
    holdings: Sequence[HoldingAssessment],
    positions: Sequence[PortfolioPosition],
    latest_price_by_symbol: Mapping[str, float],
    rules: Mapping[str, Any],
    model_probability_by_symbol: Mapping[str, float] | None = None,
    predict_v5_excess_return_by_symbol: Mapping[str, float] | None = None,
    predict_v5_low_confidence_by_symbol: Mapping[str, bool] | None = None,
    signal_validation: Mapping[str, Any] | None = None,
    standard_horizon_days: int = _DEFAULT_STANDARD_HORIZON_DAYS,
) -> RecommendationRunResult:
    """Compute SELL signals for held positions and sized BUY candidates for the rest."""
    model_probability_by_symbol = model_probability_by_symbol or {}
    predict_v5_excess_return_by_symbol = predict_v5_excess_return_by_symbol or {}
    predict_v5_low_confidence_by_symbol = predict_v5_low_confidence_by_symbol or {}
    as_of = date.fromisoformat(as_of_date)
    held_symbols = {position.symbol for position in positions}
    account_value = float(rules["account_value"])
    available_cash = float(rules["available_cash"])

    recommendations: list[TradeRecommendation] = []
    skipped: list[str] = []

    for holding in holdings:
        reason = holding.rule_based_exit_reason or holding.price_stop_reason
        if reason is None:
            continue
        price = latest_price_by_symbol.get(holding.symbol)
        estimated_dollars = holding.shares * price if price is not None else 0.0
        recommendations.append(
            TradeRecommendation(
                symbol=holding.symbol, action="SELL", shares=holding.shares,
                estimated_dollars=estimated_dollars, reason=reason,
            )
        )

    open_position_count = len(positions)
    cash_reserve = float(rules["cash_reserve"])
    max_trade_dollar_amount = float(rules["max_trade_dollar_amount"])
    min_trade_dollar_amount = float(rules["min_trade_dollar_amount"])
    max_open_positions = int(rules["max_open_positions"])
    max_new_buys = int(rules["max_new_buys_per_run"])
    eligible_classifications = (
        _BUY_ELIGIBLE_CLASSIFICATIONS_STRICT
        if rules["require_strong_candidate_for_buy"]
        else _BUY_ELIGIBLE_CLASSIFICATIONS_RELAXED
    )
    # Position-weight and cash-reserve are both *soft* caps: they reduce what's
    # spendable, but neither can push the affordable size below min_trade_dollar_amount
    # for the first trade a run considers, provided available_cash covers at least
    # one min-size trade. Without this, a $100 account (account-set's own floor)
    # could never buy anything under the shipped defaults: 15% of $100 is $15, and a
    # 5% cash reserve already drops $100 of cash to $95 -- both under a $100 minimum
    # trade. The same floor applies uniformly regardless of account size (e.g. an
    # aggressive cash_reserve on a large account can hit the same wall) -- once that
    # first trade is taken, spendable is genuine and later candidates are sized (or
    # skipped) against the real remaining cash, with no further floor applied. These
    # floors are no-ops for any account comfortably above $667 (position weight) or
    # $105 (cash reserve) at the shipped defaults, where the normal caps already clear
    # min_trade_dollar_amount on their own.
    max_position_dollars = max(account_value * float(rules["max_position_weight"]), min_trade_dollar_amount)
    spendable = max(available_cash - account_value * cash_reserve, min(available_cash, min_trade_dollar_amount))

    candidates = sorted(
        (
            result for symbol, result in results_by_symbol.items()
            if symbol not in held_symbols and result.classification in eligible_classifications
        ),
        key=lambda result: (
            0 if result.classification == "Strong Candidate" else 1,
            -(result.opportunity_score if result.opportunity_score is not None else -1.0),
            result.symbol,
        ),
    )

    new_buys = 0
    for result in candidates:
        if new_buys >= max_new_buys:
            skipped.append(f"{result.symbol}: already reached max_new_buys_per_run ({max_new_buys}) for this run")
            continue
        if open_position_count >= max_open_positions:
            skipped.append(f"{result.symbol}: already at max_open_positions ({max_open_positions})")
            continue
        price = latest_price_by_symbol.get(result.symbol)
        if price is None or price <= 0:
            skipped.append(f"{result.symbol}: no current price available")
            continue
        target_dollars = min(max_trade_dollar_amount, max_position_dollars, spendable)
        if target_dollars < min_trade_dollar_amount:
            if target_dollars == max_position_dollars:
                bottleneck = (
                    f"{rules['max_position_weight']:.0%} position cap "
                    f"(${max_position_dollars:,.2f} of ${account_value:,.2f} account value)"
                )
            elif target_dollars == spendable:
                bottleneck = f"spendable cash (${spendable:,.2f})"
            else:
                bottleneck = f"max_trade_dollar_amount (${max_trade_dollar_amount:,.2f})"
            skipped.append(
                f"{result.symbol}: {bottleneck} is below min_trade_dollar_amount (${min_trade_dollar_amount:,.2f})"
            )
            if spendable < min_trade_dollar_amount:
                break  # no cash left for anyone further down the list either
            continue
        shares = math.floor(target_dollars / price)
        if shares < 1:
            skipped.append(f"{result.symbol}: price (${price:,.2f}) is too high for the affordable size")
            continue
        estimated_dollars = shares * price
        outcome = _outcome_context(signal_validation, result.classification, as_of, standard_horizon_days)
        recommendations.append(
            TradeRecommendation(
                symbol=result.symbol, action="BUY", shares=float(shares),
                estimated_dollars=estimated_dollars, reason=result.primary_reason,
                model_probability=model_probability_by_symbol.get(result.symbol),
                predict_v5_excess_return=predict_v5_excess_return_by_symbol.get(result.symbol),
                predict_v5_low_confidence=predict_v5_low_confidence_by_symbol.get(result.symbol, False),
                classification=result.classification,
                outcome_p10=outcome["outcome_p10"],
                outcome_p90=outcome["outcome_p90"],
                best_exit_sessions=outcome["best_exit_sessions"],
                best_exit_date=outcome["best_exit_date"],
                checkpoints_compared=outcome["checkpoints_compared"],
            )
        )
        spendable -= estimated_dollars
        open_position_count += 1
        new_buys += 1

    return RecommendationRunResult(
        as_of_date=as_of_date,
        account_value=account_value,
        available_cash=available_cash,
        open_position_count=len(positions),
        recommendations=recommendations,
        skipped=skipped,
        cash_reserve=cash_reserve,
        max_position_weight=float(rules["max_position_weight"]),
        max_trade_dollar_amount=max_trade_dollar_amount,
        min_trade_dollar_amount=min_trade_dollar_amount,
    )


def render_recommendations_text(result: RecommendationRunResult) -> str:
    """Render a recommendation run to a plain-text report, digest-style."""
    lines = [
        f"TRADE RECOMMENDATIONS — as of {result.as_of_date}",
        f"Account value: ${result.account_value:,.2f}  |  Available cash: ${result.available_cash:,.2f}  |  "
        f"Open positions: {result.open_position_count}",
        "",
    ]
    sells = [rec for rec in result.recommendations if rec.action == "SELL"]
    buys = [rec for rec in result.recommendations if rec.action == "BUY"]

    lines.append(f"SELL — {len(sells)}")
    if sells:
        for rec in sells:
            lines.append(f"  {rec.symbol:<6} {rec.shares:g} sh (~${rec.estimated_dollars:,.2f}) — {rec.reason}")
    else:
        lines.append("  None")

    lines.append("")
    lines.append(f"BUY — {len(buys)}")
    if buys:
        for rec in buys:
            probability_text = (
                "" if rec.model_probability is None else f" | model: {rec.model_probability:.0%} beats benchmark"
            )
            v5_text = ""
            if rec.predict_v5_excess_return is not None:
                confidence_suffix = " [LOW CONFIDENCE]" if rec.predict_v5_low_confidence else ""
                v5_text = f" | predict-v5: {rec.predict_v5_excess_return:+.1%} predicted excess return{confidence_suffix}"
            lines.append(
                f"  {rec.symbol:<6} {rec.shares:g} sh (~${rec.estimated_dollars:,.2f}) — "
                f"{rec.reason}{probability_text}{v5_text}"
            )
    else:
        lines.append("  None")

    if result.skipped:
        lines.append("")
        lines.append("Considered but not recommended:")
        for entry in result.skipped:
            lines.append(f"  {entry}")

    lines.append("")
    lines.append(
        "These are sized suggestions only — nothing here has been bought or sold. Review each "
        "before placing any order yourself. Educational research output, not investment advice."
    )
    return "\n".join(lines)
