"""Why does score_v1's "High Risk" bucket beat the benchmark forward?

``validate-signals`` (see ``signal_validation.py``) established, on the full 20-year/
25-symbol history, that "High Risk" posts the *largest* forward excess return of any
classification bucket — a real, symbol-diverse inversion, not the old two-symbol
concentration artifact. That module deliberately stops at describing the inversion; it
does not explain it. This module asks a narrower follow-up question: within "High Risk"
(and, for contrast, "Strong Candidate"), does the forward excess return concentrate in a
specific range of any one already-computed technical indicator?

Two candidate explanations motivate the feature list below, both drawn from indicators
this project already computes (no new feature engineering):

- **Mean reversion**: "High Risk" stocks got there partly via recent negative
  momentum/drawdown (``risk_score.py``'s ``_drawdown_risk``/``_trend_deterioration``
  components read exactly these fields). If forward excess return is concentrated among
  the rows with the *most negative* trailing return, that is a reversion/bounce story.
- **Risk premium**: "High Risk" also partly reflects raw volatility/beta magnitude
  (``_realized_volatility``/``_beta_sensitivity``). If forward excess return instead
  scales with volatility/beta *magnitude* regardless of trailing return's sign, that is
  more consistent with compensated risk-taking than with reversion.

This is descriptive/exploratory, not a validated inferential result: the same
overlapping-window, symbol-clustered non-independence documented in
``signal_validation.py`` applies here too, and a quintile split on a handful of
correlated indicators can look suggestive by chance. Nothing here changes score_v1's
scoring, thresholds, or classification logic — it is read-only diagnostics over already-
classified signals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from stock_scrapper.analysis.signal_validation import _date_index, _finite, _get, forward_excess_return

# Every key here is already computed by this project (predict-v3's own feature_keys, or
# a risk_score.py component input) — chosen to cover both the mean-reversion story
# (trailing-return/drawdown fields) and the risk-premium story (volatility/beta
# magnitude), not to introduce a new indicator.
CANDIDATE_DRIVER_FEATURES: tuple[str, ...] = (
    "six_month_return",
    "three_month_return",
    "one_month_return",
    "distance_from_sma50",
    "distance_from_sma200",
    "one_year_max_drawdown",
    "atr_percentage",
    "twenty_day_volatility",
    "beta",
    "benchmark_relative_return_252",
    "benchmark_relative_return_63",
)

QUINTILE_COUNT = 5
MIN_ROWS_FOR_STUDY = QUINTILE_COUNT * 2


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Hand-rolled Pearson r — consistent with this project's no-scipy design (see
    ``signal_validation.py``'s hand-rolled z-test). ``None`` if fewer than 2 points or
    either series has zero variance (correlation is undefined, not zero, in that case).
    """
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    variance_y = sum((y - mean_y) ** 2 for y in ys)
    if variance_x <= 0 or variance_y <= 0:
        return None
    return covariance / math.sqrt(variance_x * variance_y)


@dataclass(slots=True)
class QuintileBucket:
    quintile: int  # 1 = lowest feature values, 5 = highest
    n: int
    feature_low: float
    feature_high: float
    mean_excess_return: float


@dataclass(slots=True)
class DriverFeatureStudy:
    feature: str
    n: int
    pearson_r: float | None
    quintiles: list[QuintileBucket] = field(default_factory=list)


@dataclass(slots=True)
class RiskDiagnosticsResult:
    classification: str
    horizon_days: int
    sample_size: int
    distinct_symbols: int
    driver_studies: list[DriverFeatureStudy] = field(default_factory=list)


def _quintile_buckets(pairs: Sequence[tuple[float, float]]) -> list[QuintileBucket]:
    """Split ``pairs`` (feature value, excess return) into up to 5 equal-sized buckets,
    ordered from lowest to highest feature value, and report each bucket's mean excess
    return. Ties in the boundary rows land in whichever bucket the sort order gives them
    — not adjusted for, since this is a descriptive split, not a rank test."""
    ordered = sorted(pairs, key=lambda pair: pair[0])
    n = len(ordered)
    boundaries = [round(n * index / QUINTILE_COUNT) for index in range(QUINTILE_COUNT + 1)]
    buckets: list[QuintileBucket] = []
    for quintile in range(QUINTILE_COUNT):
        chunk = ordered[boundaries[quintile] : boundaries[quintile + 1]]
        if not chunk:
            continue
        features = [value for value, _ in chunk]
        returns = [value for _, value in chunk]
        buckets.append(
            QuintileBucket(
                quintile=quintile + 1,
                n=len(chunk),
                feature_low=features[0],
                feature_high=features[-1],
                mean_excess_return=sum(returns) / len(returns),
            )
        )
    return buckets


def study_driver_features(
    rows: Sequence[Mapping[str, Any]],
    features: Sequence[str] = CANDIDATE_DRIVER_FEATURES,
) -> list[DriverFeatureStudy]:
    """One ``DriverFeatureStudy`` per feature with enough non-missing rows to bucket."""
    studies: list[DriverFeatureStudy] = []
    for feature in features:
        pairs = [
            (row[feature], row["excess_return"])
            for row in rows
            if row.get(feature) is not None and row.get("excess_return") is not None
        ]
        if len(pairs) < MIN_ROWS_FOR_STUDY:
            continue
        xs = [pair[0] for pair in pairs]
        ys = [pair[1] for pair in pairs]
        studies.append(
            DriverFeatureStudy(
                feature=feature,
                n=len(pairs),
                pearson_r=pearson_correlation(xs, ys),
                quintiles=_quintile_buckets(pairs),
            )
        )
    return studies


def build_classified_rows(
    signals: Sequence[Any],
    histories: Mapping[str, list[dict[str, Any]]],
    *,
    classification: str,
    benchmark_symbol: str,
    horizon_days: int,
    features: Sequence[str] = CANDIDATE_DRIVER_FEATURES,
) -> list[dict[str, Any]]:
    """One row per (symbol, signal_date) classified ``classification``, pairing its
    forward excess return with the already-computed indicator values carried on the
    ``Signal`` (in-memory only — ``backtest_signals`` does not persist ``indicators``,
    so this must run against a freshly produced ``result.signals`` list, not a
    previously persisted run)."""
    benchmark_history = histories.get(benchmark_symbol.upper(), [])
    benchmark_index = _date_index(benchmark_history)
    symbol_indices: dict[str, dict[str, int]] = {}

    rows: list[dict[str, Any]] = []
    for signal in signals:
        if _get(signal, "classification") != classification:
            continue
        symbol = str(_get(signal, "symbol") or "").upper()
        signal_date = str(_get(signal, "signal_date") or "")[:10]
        if not symbol or not signal_date:
            continue
        indicators = _get(signal, "indicators") or {}
        symbol_history = histories.get(symbol, [])
        if symbol not in symbol_indices:
            symbol_indices[symbol] = _date_index(symbol_history)
        excess = forward_excess_return(
            symbol_history,
            benchmark_history,
            signal_date,
            horizon_days,
            symbol_index=symbol_indices[symbol],
            benchmark_index=benchmark_index,
        )
        if excess is None:
            continue
        row: dict[str, Any] = {"symbol": symbol, "date": signal_date, "excess_return": excess}
        for feature in features:
            row[feature] = _finite(indicators.get(feature))
        rows.append(row)
    return rows


def investigate_classification(
    signals: Sequence[Any],
    histories: Mapping[str, list[dict[str, Any]]],
    *,
    classification: str,
    benchmark_symbol: str,
    horizon_days: int,
    features: Sequence[str] = CANDIDATE_DRIVER_FEATURES,
) -> RiskDiagnosticsResult:
    rows = build_classified_rows(
        signals,
        histories,
        classification=classification,
        benchmark_symbol=benchmark_symbol,
        horizon_days=horizon_days,
        features=features,
    )
    return RiskDiagnosticsResult(
        classification=classification,
        horizon_days=horizon_days,
        sample_size=len(rows),
        distinct_symbols=len({row["symbol"] for row in rows}),
        driver_studies=study_driver_features(rows, features),
    )


def render_risk_diagnostics_text(result: RiskDiagnosticsResult) -> str:
    """Plain-text report matching this project's existing CLI-output house style."""
    lines = [
        f'Risk-diagnostics study: "{result.classification}" driver-feature quintiles '
        f"({result.horizon_days}-session forward excess return)",
        f"  sample size: {result.sample_size}  distinct symbols: {result.distinct_symbols}",
        "  exploratory/descriptive only — same overlapping-window, symbol-clustered "
        "non-independence as validate-signals applies here too.",
    ]
    if not result.driver_studies:
        lines.append("  no feature had enough non-missing rows to study.")
        return "\n".join(lines)
    for study in result.driver_studies:
        r_text = f"{study.pearson_r:+.3f}" if study.pearson_r is not None else "--"
        lines.append("")
        lines.append(f"  {study.feature}  (n={study.n}, pearson r={r_text})")
        lines.append(f"    {'quintile':>9}{'n':>7}{'range':>24}{'mean xs ret':>14}")
        for bucket in study.quintiles:
            range_text = f"[{bucket.feature_low:+.3f}, {bucket.feature_high:+.3f}]"
            lines.append(
                f"    {bucket.quintile:>9}{bucket.n:>7}{range_text:>24}{bucket.mean_excess_return:>+14.2%}"
            )
    return "\n".join(lines)
