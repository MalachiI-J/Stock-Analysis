"""Score past `recommend` output against what actually happened — accountability, not just theory.

The backtester and the prediction model's walk-forward folds both validate the
underlying rules against history. Neither one tells you whether a specific past
`recommend` run — with its specific sizing, specific candidates, and specific
day — actually worked out. This module closes that loop: given one saved
`recommendations_<date>.summary.json` payload, it looks up what each
recommended symbol's price actually did between the recommendation date and a
later review date, and reports it the same way the experimental predictor
defines success — excess return over the benchmark — so the two are directly
comparable.

Pure and DB-free like the rest of stock_scrapper/trading/: the caller supplies
a ``price_at(symbol, date)`` lookup, exactly like the callback pattern already
used in stock_scrapper/portfolio.py's benchmark-shadow comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


@dataclass(slots=True)
class RecommendationOutcome:
    """What actually happened to one past recommendation, in hindsight."""

    symbol: str
    action: str
    entry_price: float
    exit_price: float | None
    realized_return_pct: float | None
    benchmark_return_pct: float | None
    excess_return_pct: float | None
    outcome: str


@dataclass(slots=True)
class RecommendationReviewResult:
    """Complete hindsight review of one saved recommendation run."""

    recommendation_date: str
    review_date: str
    benchmark_symbol: str
    outcomes: list[RecommendationOutcome] = field(default_factory=list)
    buy_hit_rate: float | None = None
    average_buy_excess_return_pct: float | None = None
    sell_avoided_loss_rate: float | None = None
    unpriced_symbols: list[str] = field(default_factory=list)


def evaluate_recommendation_outcomes(
    recommendations: Sequence[Mapping[str, Any]],
    *,
    recommendation_date: str,
    review_date: str,
    benchmark_symbol: str,
    price_at: Callable[[str, str], float | None],
) -> RecommendationReviewResult:
    """Look up what each recommended symbol actually did, and grade it in hindsight.

    A BUY is graded exactly like the experimental predictor defines success:
    did the symbol beat the benchmark's own return over the same window. A
    SELL is graded in reverse: did the price actually fall after being sold
    (validating the exit) or would holding have paid off.
    """
    benchmark_entry = price_at(benchmark_symbol, recommendation_date)
    benchmark_exit = price_at(benchmark_symbol, review_date)
    benchmark_return = (
        (benchmark_exit - benchmark_entry) / benchmark_entry
        if benchmark_entry is not None and benchmark_exit is not None and benchmark_entry > 0
        else None
    )

    outcomes: list[RecommendationOutcome] = []
    unpriced_symbols: list[str] = []
    buy_hits: list[bool] = []
    buy_excess_returns: list[float] = []
    sell_avoided_losses: list[bool] = []

    for rec in recommendations:
        symbol = str(rec["symbol"])
        action = str(rec["action"])
        shares = float(rec.get("shares") or 0.0)
        estimated_dollars = float(rec.get("estimated_dollars") or 0.0)
        entry_price = estimated_dollars / shares if shares > 0 else None
        exit_price = price_at(symbol, review_date)

        if entry_price is None or entry_price <= 0 or exit_price is None:
            unpriced_symbols.append(symbol)
            outcomes.append(
                RecommendationOutcome(
                    symbol=symbol, action=action, entry_price=entry_price or 0.0, exit_price=exit_price,
                    realized_return_pct=None, benchmark_return_pct=benchmark_return,
                    excess_return_pct=None, outcome="no price data available for the review date",
                )
            )
            continue

        realized_return = (exit_price - entry_price) / entry_price
        excess_return = realized_return - benchmark_return if benchmark_return is not None else None

        if action == "BUY":
            if excess_return is None:
                outcome = "benchmark data unavailable"
            elif excess_return > 0:
                outcome = "beat the benchmark"
                buy_hits.append(True)
                buy_excess_returns.append(excess_return)
            else:
                outcome = "trailed the benchmark"
                buy_hits.append(False)
                buy_excess_returns.append(excess_return)
        else:  # SELL
            if realized_return < 0:
                outcome = "selling looks validated (price fell after)"
                sell_avoided_losses.append(True)
            else:
                outcome = "would have missed a gain by selling"
                sell_avoided_losses.append(False)

        outcomes.append(
            RecommendationOutcome(
                symbol=symbol, action=action, entry_price=entry_price, exit_price=exit_price,
                realized_return_pct=realized_return, benchmark_return_pct=benchmark_return,
                excess_return_pct=excess_return, outcome=outcome,
            )
        )

    return RecommendationReviewResult(
        recommendation_date=recommendation_date,
        review_date=review_date,
        benchmark_symbol=benchmark_symbol,
        outcomes=outcomes,
        buy_hit_rate=(sum(buy_hits) / len(buy_hits)) if buy_hits else None,
        average_buy_excess_return_pct=(sum(buy_excess_returns) / len(buy_excess_returns)) if buy_excess_returns else None,
        sell_avoided_loss_rate=(sum(sell_avoided_losses) / len(sell_avoided_losses)) if sell_avoided_losses else None,
        unpriced_symbols=unpriced_symbols,
    )


def render_review_text(result: RecommendationReviewResult) -> str:
    """Render a hindsight review to a plain-text report."""
    lines = [
        f"RECOMMENDATION REVIEW — {result.recommendation_date} recommendations, reviewed as of {result.review_date}",
        "",
    ]
    if not result.outcomes:
        lines.append("No recommendations were recorded for that date.")
        return "\n".join(lines)

    for item in result.outcomes:
        exit_text = "n/a" if item.exit_price is None else f"{item.exit_price:.2f}"
        realized_text = "n/a" if item.realized_return_pct is None else f"{item.realized_return_pct:+.1%}"
        excess_text = "n/a" if item.excess_return_pct is None else f"{item.excess_return_pct:+.1%}"
        lines.append(
            f"  {item.symbol:<6} {item.action:<4} entry {item.entry_price:.2f} -> exit {exit_text} "
            f"| return {realized_text} | vs {result.benchmark_symbol} {excess_text} | {item.outcome}"
        )

    lines.append("")
    hit_rate_text = "n/a" if result.buy_hit_rate is None else f"{result.buy_hit_rate:.0%}"
    avg_excess_text = (
        "n/a" if result.average_buy_excess_return_pct is None else f"{result.average_buy_excess_return_pct:+.1%}"
    )
    lines.append(f"BUY hit rate (beat {result.benchmark_symbol}): {hit_rate_text}  |  average excess return: {avg_excess_text}")
    avoided_text = "n/a" if result.sell_avoided_loss_rate is None else f"{result.sell_avoided_loss_rate:.0%}"
    lines.append(f"SELL avoided-loss rate: {avoided_text}")
    if result.unpriced_symbols:
        lines.append("Missing price data for: " + ", ".join(result.unpriced_symbols))

    lines.append("")
    lines.append(
        "One run is one data point, not a track record — look at this across many dated recommendation "
        "files before drawing conclusions. Educational research output, not investment advice."
    )
    return "\n".join(lines)
