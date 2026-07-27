"""Orchestrate the experimental excess-return-vs-benchmark prediction: train, evaluate, predict.

The model predicts whether a symbol will beat the benchmark's own return over
the horizon, not merely whether its price rises — see
stock_scrapper/prediction/dataset.py for why raw direction is mostly just
market drift in disguise. Everything here is fit fresh from the
caller-supplied, as-of-date-bounded histories every time it runs — nothing is
persisted or reused across calls. Given the same stored price history and the
same config, the result is exactly reproducible (no randomness anywhere in
the pipeline).

Performance is estimated via expanding-window walk-forward cross-validation
(fit on everything before a chronological cut, test on the chunk right after
it, repeated over several cuts) rather than a single train/holdout split, so
the reported accuracy/Brier score isn't just the luck of one split. The model
actually used for today's predictions is then re-fit on the *entire* dataset,
since a deployed model should use all the history available to it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping, Sequence

from stock_scrapper.analysis.service import AnalysisService
from stock_scrapper.prediction.dataset import (
    build_training_dataset,
    feature_value,
    opportunity_percentiles,
    select_sample_dates,
)
from stock_scrapper.prediction.model import evaluate_holdout, fit_logistic_regression


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(slots=True)
class SymbolPrediction:
    """One symbol's predicted probability of beating the benchmark, or why it has none."""

    symbol: str
    probability_positive: float | None
    reason: str | None = None


@dataclass(slots=True)
class WalkForwardFold:
    """One expanding-window fold's train/test split size and holdout performance."""

    fold: int
    training_samples: int
    test_samples: int
    accuracy: float | None
    brier_score: float | None


@dataclass(slots=True)
class PredictionRunResult:
    """Complete outcome of one experimental prediction run."""

    status: str  # "ok" or "insufficient_data"
    message: str | None
    as_of_date: str
    horizon_days: int
    training_samples: int = 0
    holdout_samples: int = 0
    training_start_date: str | None = None
    training_end_date: str | None = None
    positive_label_rate: float | None = None
    holdout_accuracy: float | None = None
    holdout_brier_score: float | None = None
    walk_forward_folds: list[WalkForwardFold] = field(default_factory=list)
    coefficients: list[tuple[str, float]] = field(default_factory=list)
    predictions: list[SymbolPrediction] = field(default_factory=list)


def _fold_boundaries(sample_count: int, folds: int) -> list[int]:
    """Split ``sample_count`` chronological samples into ``folds + 1`` contiguous chunks.

    The first chunk is always training-only; each remaining chunk serves once as
    the held-out test set for the fold that follows it, with every earlier chunk
    (expanding window) as its training data.
    """
    segments = folds + 1
    base, remainder = divmod(sample_count, segments)
    boundaries = [0]
    cursor = 0
    for index in range(segments):
        cursor += base + (1 if index < remainder else 0)
        boundaries.append(cursor)
    return boundaries


def _run_walk_forward(
    feature_keys: list[str],
    features: Any,
    labels: Any,
    *,
    requested_folds: int,
    l2_lambda: float,
    learning_rate: float,
    iterations: int,
) -> list[WalkForwardFold]:
    sample_count = features.shape[0]
    folds = min(requested_folds, sample_count - 1)
    if folds < 1:
        return []
    boundaries = _fold_boundaries(sample_count, folds)
    results: list[WalkForwardFold] = []
    for fold_index in range(folds):
        train_end = boundaries[fold_index + 1]
        test_end = boundaries[fold_index + 2]
        if train_end == 0 or test_end == train_end:
            continue
        model = fit_logistic_regression(
            feature_keys, features[:train_end], labels[:train_end],
            l2_lambda=l2_lambda, learning_rate=learning_rate, iterations=iterations,
        )
        metrics = evaluate_holdout(model, features[train_end:test_end], labels[train_end:test_end])
        results.append(
            WalkForwardFold(
                fold=len(results) + 1,
                training_samples=train_end,
                test_samples=test_end - train_end,
                accuracy=metrics["accuracy"] if metrics["sample_count"] else None,
                brier_score=metrics["brier_score"] if metrics["sample_count"] else None,
            )
        )
    return results


def run_prediction(
    service: AnalysisService,
    target_symbols: Sequence[str],
    histories: Mapping[str, list[dict[str, Any]]],
    trading_dates: Sequence[str],
    *,
    as_of_date: str,
    rules: Mapping[str, Any],
    benchmark_symbol: str,
) -> PredictionRunResult:
    """Walk-forward evaluate, then fit a final model on all embargoed history, and predict today."""
    horizon_days = int(rules["horizon_days"])
    sample_dates = select_sample_dates(
        trading_dates,
        as_of_date=as_of_date,
        horizon_days=horizon_days,
        lookback_years=float(rules["lookback_years"]),
        stride_sessions=int(rules["sample_stride_sessions"]),
    )
    if not sample_dates:
        return PredictionRunResult(
            status="insufficient_data",
            message="Not enough historical trading sessions are stored to build any training sample.",
            as_of_date=as_of_date,
            horizon_days=horizon_days,
        )

    feature_keys = list(rules["feature_keys"])
    features, labels, meta = build_training_dataset(
        service, target_symbols, histories, sample_dates,
        horizon_days=horizon_days, feature_keys=feature_keys, benchmark_symbol=benchmark_symbol,
    )
    minimum_samples = int(rules["minimum_training_samples"])
    if features.shape[0] < minimum_samples:
        return PredictionRunResult(
            status="insufficient_data",
            message=(
                f"Only {features.shape[0]} training sample(s) were available; "
                f"{minimum_samples} are required. Collect more history or lower "
                "minimum_training_samples in config/prediction_rules.yaml."
            ),
            as_of_date=as_of_date,
            horizon_days=horizon_days,
            training_samples=int(features.shape[0]),
        )

    l2_lambda = float(rules["l2_lambda"])
    learning_rate = float(rules["learning_rate"])
    iterations = int(rules["iterations"])

    folds = _run_walk_forward(
        feature_keys, features, labels,
        requested_folds=int(rules["walk_forward_folds"]),
        l2_lambda=l2_lambda, learning_rate=learning_rate, iterations=iterations,
    )
    message = None
    scored_folds = [fold for fold in folds if fold.accuracy is not None]
    if scored_folds:
        holdout_accuracy = sum(fold.accuracy for fold in scored_folds) / len(scored_folds)
        holdout_brier_score = sum(fold.brier_score for fold in scored_folds) / len(scored_folds)
        holdout_samples = sum(fold.test_samples for fold in scored_folds)
    else:
        holdout_accuracy = None
        holdout_brier_score = None
        holdout_samples = 0
        message = (
            "Not enough samples to run any walk-forward evaluation folds; showing model "
            "coefficients with no holdout accuracy estimate. Collect more history or lower "
            "walk_forward_folds in config/prediction_rules.yaml."
        )

    # The deployed model is fit on the full embargoed dataset, not just one fold's
    # training slice, since a live prediction should use all the history available.
    final_model = fit_logistic_regression(
        feature_keys, features, labels,
        l2_lambda=l2_lambda, learning_rate=learning_rate, iterations=iterations,
    )

    today_batch = service.analyze_loaded_many_as_of(
        list(target_symbols), histories, date.fromisoformat(as_of_date), persist=False
    )
    percentiles = opportunity_percentiles(today_batch.results)
    predictions: list[SymbolPrediction] = []
    for result in today_batch.results:
        if not result.eligible_for_scoring:
            predictions.append(
                SymbolPrediction(result.symbol, None, "Not eligible for scoring (insufficient or blocked data)")
            )
            continue
        values = [feature_value(result, key, percentiles) for key in feature_keys]
        if any(value is None for value in values):
            predictions.append(
                SymbolPrediction(result.symbol, None, "One or more required indicators are unavailable")
            )
            continue
        predictions.append(SymbolPrediction(result.symbol, final_model.predict_proba(values), None))

    return PredictionRunResult(
        status="ok",
        message=message,
        as_of_date=as_of_date,
        horizon_days=horizon_days,
        training_samples=int(features.shape[0]),
        holdout_samples=holdout_samples,
        training_start_date=meta[0]["date"] if meta else None,
        training_end_date=meta[-1]["date"] if meta else None,
        positive_label_rate=float(labels.mean()) if labels.shape[0] else None,
        holdout_accuracy=holdout_accuracy,
        holdout_brier_score=holdout_brier_score,
        walk_forward_folds=folds,
        coefficients=final_model.coefficients(),
        predictions=predictions,
    )
