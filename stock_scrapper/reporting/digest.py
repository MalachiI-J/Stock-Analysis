"""Build a short, plain-language daily buy/watch/sell digest from saved Phase 2 results.

This renders the same persisted classifications used elsewhere in the app
(``analyze``/``scores``/``report``) into a compact human-readable summary. It
does not compute anything new and does not know about any real portfolio —
"sell" here means "this symbol's own classification suggests exiting a
position, if you hold one," not a personalized instruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_BUY_CLASSIFICATIONS = ("Strong Candidate", "Candidate")
_SELL_CLASSIFICATIONS = ("Avoid", "High Risk")
_WATCH_CLASSIFICATIONS = ("Watch",)
_BLOCKED_CLASSIFICATIONS = ("Data Blocked", "Insufficient Data")


@dataclass
class DigestEntry:
    """One symbol's digest line: current state plus the change from last run."""

    symbol: str
    classification: str
    previous_classification: str | None
    primary_reason: str
    opportunity_score: float | None
    risk_score: float | None
    confidence_score: float | None

    @property
    def changed(self) -> bool:
        return self.previous_classification is not None and self.previous_classification != self.classification


def _entry(result: Any, previous_by_symbol: Mapping[str, Any]) -> DigestEntry:
    symbol = str(getattr(result, "symbol", "")).upper()
    previous = previous_by_symbol.get(symbol)
    return DigestEntry(
        symbol=symbol,
        classification=str(getattr(result, "classification", "Insufficient Data")),
        previous_classification=(str(getattr(previous, "classification", None)) if previous is not None else None),
        primary_reason=str(getattr(result, "primary_reason", "") or ""),
        opportunity_score=getattr(result, "opportunity_score", None),
        risk_score=getattr(result, "risk_score", None),
        confidence_score=getattr(result, "confidence_score", None),
    )


def build_digest(
    *,
    as_of_date: str,
    data_through_date: str | None,
    market_regime: str,
    market_regime_confidence: float | None,
    results: Sequence[Any],
    previous_results: Sequence[Any] = (),
) -> dict[str, Any]:
    """Bucket saved analysis results into buy/sell/watch/blocked groups plus a change log."""
    previous_by_symbol = {str(getattr(item, "symbol", "")).upper(): item for item in previous_results}
    entries = [_entry(result, previous_by_symbol) for result in results]
    by_symbol = {entry.symbol: entry for entry in entries}

    def bucket(classifications: Sequence[str]) -> list[DigestEntry]:
        return sorted(
            (entry for entry in entries if entry.classification in classifications),
            key=lambda entry: (
                -(entry.opportunity_score if entry.opportunity_score is not None else -1.0),
                entry.symbol,
            ),
        )

    changes = sorted(
        (entry for entry in entries if entry.changed),
        key=lambda entry: entry.symbol,
    )
    return {
        "as_of_date": as_of_date,
        "data_through_date": data_through_date,
        "market_regime": market_regime,
        "market_regime_confidence": market_regime_confidence,
        "buy": bucket(_BUY_CLASSIFICATIONS),
        "sell": bucket(_SELL_CLASSIFICATIONS),
        "watch": bucket(_WATCH_CLASSIFICATIONS),
        "blocked": bucket(_BLOCKED_CLASSIFICATIONS),
        "changes": changes,
        "symbols": by_symbol,
    }


def _format_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def _format_entry_line(entry: DigestEntry) -> str:
    change = f" (was {entry.previous_classification})" if entry.changed else ""
    return (
        f"  {entry.symbol:<6} {entry.classification}{change} — opp={_format_score(entry.opportunity_score)} "
        f"risk={_format_score(entry.risk_score)} conf={_format_score(entry.confidence_score)}\n"
        f"          {entry.primary_reason}"
    )


def render_digest_text(digest: Mapping[str, Any]) -> str:
    """Render a build_digest() payload as a plain-text daily summary."""
    lines: list[str] = []
    lines.append(f"Stock Scrapper daily digest — as of {digest['as_of_date']} (data through {digest.get('data_through_date') or 'unavailable'})")
    confidence = digest.get("market_regime_confidence")
    confidence_text = f" (confidence {confidence:.1f})" if isinstance(confidence, (int, float)) else ""
    lines.append(f"Market regime: {digest['market_regime']}{confidence_text}")
    lines.append("")

    lines.append(f"BUY / STRONG — {len(digest['buy'])} symbol(s)")
    if digest["buy"]:
        lines.extend(_format_entry_line(entry) for entry in digest["buy"])
    else:
        lines.append("  None today.")
    lines.append("")

    lines.append(f"SELL / AVOID (if held) — {len(digest['sell'])} symbol(s)")
    if digest["sell"]:
        lines.extend(_format_entry_line(entry) for entry in digest["sell"])
    else:
        lines.append("  None today.")
    lines.append("")

    lines.append(f"WATCH — {len(digest['watch'])} symbol(s)")
    if digest["watch"]:
        lines.extend(_format_entry_line(entry) for entry in digest["watch"])
    else:
        lines.append("  None today.")
    lines.append("")

    if digest["blocked"]:
        lines.append(f"DATA ISSUES — {len(digest['blocked'])} symbol(s) could not be scored")
        lines.extend(_format_entry_line(entry) for entry in digest["blocked"])
        lines.append("")

    lines.append(f"CHANGES SINCE LAST RUN — {len(digest['changes'])} symbol(s)")
    if digest["changes"]:
        for entry in digest["changes"]:
            lines.append(f"  {entry.symbol:<6} {entry.previous_classification} -> {entry.classification}")
    else:
        lines.append("  No classification changes since the previous saved run.")
    lines.append("")

    lines.append(
        "Educational research only, not personalized financial advice. Classifications reflect price/volume "
        "technicals as of the stated date and do not guarantee future performance."
    )
    return "\n".join(lines) + "\n"
