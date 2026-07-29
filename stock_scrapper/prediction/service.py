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
from stock_scrapper.utilities.hashing import stable_sha256


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
    """One expanding-window fold's train/test split, split honestly by calendar date
    (not row count — see _run_walk_forward), plus that fold's own baselines.

    ``baseline_accuracy``/``baseline_brier_score`` are computed from this fold's own
    test-period label rate, not a fixed 50/50 assumption: the majority-class baseline
    is ``max(rate, 1 - rate)`` and the constant-probability Brier baseline is
    ``rate * (1 - rate)`` (the true minimum achievable Brier for a predictor that
    always outputs the base rate — only 0.25 when that rate happens to be 50/50).
    """

    fold: int
    training_samples: int
    test_samples: int
    accuracy: float | None
    brier_score: float | None
    training_start_date: str | None = None
    training_end_date: str | None = None
    test_start_date: str | None = None
    test_end_date: str | None = None
    training_symbol_count: int = 0
    test_symbol_count: int = 0
    purged_samples: int = 0
    training_positive_rate: float | None = None
    test_positive_rate: float | None = None
    baseline_accuracy: float | None = None
    baseline_brier_score: float | None = None
    accuracy_vs_baseline: float | None = None
    # Positive means the model's Brier is *lower* (better) than the baseline's, since
    # lower Brier is better — this is baseline minus model, not model minus baseline.
    brier_improvement_vs_baseline: float | None = None


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
    # Dataset-wide positive rate, kept for context only — it is NOT a fair baseline for
    # holdout_accuracy, since each fold's own test-period rate (see WalkForwardFold) can
    # differ substantially from the whole dataset's rate. Compare holdout_accuracy against
    # baseline_accuracy below instead.
    positive_label_rate: float | None = None
    holdout_accuracy: float | None = None
    holdout_brier_score: float | None = None
    # Sample-weighted (by each fold's test_samples) aggregate of every fold's own
    # majority-class / constant-probability baseline — the honest comparison point for
    # holdout_accuracy/holdout_brier_score, built the same way those two are aggregated.
    baseline_accuracy: float | None = None
    baseline_brier_score: float | None = None
    walk_forward_folds: list[WalkForwardFold] = field(default_factory=list)
    coefficients: list[tuple[str, float]] = field(default_factory=list)
    predictions: list[SymbolPrediction] = field(default_factory=list)
    # Provenance: lets two runs over identical data/symbols/features be recognized as
    # reruns rather than independent evidence. None when no dataset was ever assembled
    # (the earliest "insufficient_data" case, before any symbols/features are known).
    dataset_fingerprint: str | None = None
    symbol_universe_hash: str | None = None
    feature_set_hash: str | None = None


def _fold_boundaries(count: int, folds: int) -> list[int]:
    """Split ``count`` chronological units into ``folds + 1`` contiguous chunks.

    The first chunk is always training-only; each remaining chunk serves once as
    the held-out test set for the fold that follows it, with every earlier chunk
    (expanding window) as its training data. ``count`` is a count of unique dates,
    not rows — see _run_walk_forward.
    """
    segments = folds + 1
    base, remainder = divmod(count, segments)
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
    meta: list[dict[str, str]],
    *,
    requested_folds: int,
    l2_lambda: float,
    learning_rate: float,
    iterations: int,
) -> list[WalkForwardFold]:
    """Expanding-window walk-forward, split by unique calendar date rather than row
    count. ``build_training_dataset`` emits rows date-major (every eligible symbol for
    one sample date, then the next), so a row-count boundary can and does land in the
    middle of one date's block — putting some of that date's symbols in training and
    others in testing for the same fold. Splitting by date first, then mapping back to
    rows, guarantees no two rows sharing a date ever end up on opposite sides of a fold
    boundary.

    Also purges any training row whose label resolves on or after the test period's
    first date: that row's *features* never see test-period data, but its *label*
    (a forward return ``horizon_days`` sessions out) does overlap the test window, so
    training on it would still leak information about test-period outcomes.
    """
    unique_dates = sorted({row["date"] for row in meta})
    date_count = len(unique_dates)
    folds = min(requested_folds, date_count - 1) if date_count > 0 else 0
    if folds < 1:
        return []
    boundaries = _fold_boundaries(date_count, folds)
    date_rank = {value: index for index, value in enumerate(unique_dates)}
    row_date_ranks = [date_rank[row["date"]] for row in meta]

    results: list[WalkForwardFold] = []
    for fold_index in range(folds):
        train_date_end = boundaries[fold_index + 1]
        test_date_end = boundaries[fold_index + 2]
        if train_date_end == 0 or test_date_end == train_date_end:
            continue
        test_start_date = unique_dates[train_date_end]

        train_indices = [index for index, rank in enumerate(row_date_ranks) if rank < train_date_end]
        test_indices = [
            index for index, rank in enumerate(row_date_ranks) if train_date_end <= rank < test_date_end
        ]
        purged_indices = [index for index in train_indices if meta[index]["label_end_date"] >= test_start_date]
        kept_train_indices = [
            index for index in train_indices if meta[index]["label_end_date"] < test_start_date
        ]
        if not kept_train_indices or not test_indices:
            continue

        train_features, train_labels = features[kept_train_indices], labels[kept_train_indices]
        test_features, test_labels = features[test_indices], labels[test_indices]

        model = fit_logistic_regression(
            feature_keys, train_features, train_labels,
            l2_lambda=l2_lambda, learning_rate=learning_rate, iterations=iterations,
        )
        metrics = evaluate_holdout(model, test_features, test_labels)
        accuracy = metrics["accuracy"] if metrics["sample_count"] else None
        brier_score = metrics["brier_score"] if metrics["sample_count"] else None

        test_positive_rate = float(test_labels.mean()) if test_labels.shape[0] else None
        training_positive_rate = float(train_labels.mean()) if train_labels.shape[0] else None
        if test_positive_rate is None:
            baseline_accuracy = baseline_brier_score = None
        else:
            baseline_accuracy = max(test_positive_rate, 1.0 - test_positive_rate)
            baseline_brier_score = test_positive_rate * (1.0 - test_positive_rate)

        train_dates = [meta[index]["date"] for index in kept_train_indices]
        test_dates = [meta[index]["date"] for index in test_indices]

        results.append(
            WalkForwardFold(
                fold=len(results) + 1,
                training_samples=len(kept_train_indices),
                test_samples=len(test_indices),
                accuracy=accuracy,
                brier_score=brier_score,
                training_start_date=min(train_dates),
                training_end_date=max(train_dates),
                test_start_date=min(test_dates),
                test_end_date=max(test_dates),
                training_symbol_count=len({meta[index]["symbol"] for index in kept_train_indices}),
                test_symbol_count=len({meta[index]["symbol"] for index in test_indices}),
                purged_samples=len(purged_indices),
                training_positive_rate=training_positive_rate,
                test_positive_rate=test_positive_rate,
                baseline_accuracy=baseline_accuracy,
                baseline_brier_score=baseline_brier_score,
                accuracy_vs_baseline=(
                    None if accuracy is None or baseline_accuracy is None else accuracy - baseline_accuracy
                ),
                brier_improvement_vs_baseline=(
                    None if brier_score is None or baseline_brier_score is None
                    else baseline_brier_score - brier_score
                ),
            )
        )
    return results


def _weighted_average(values: list[tuple[float, int]]) -> float | None:
    """Sample-size-weighted mean of ``(value, weight)`` pairs, or ``None`` if empty/zero-weight."""
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        return None
    return sum(value * weight for value, weight in values) / total_weight


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
    # Computed as soon as a dataset exists (even one too small to train on) so a rerun
    # that hits the same "not enough data" wall is still identifiable as a rerun.
    dataset_fingerprint = stable_sha256({"features": features.tolist(), "labels": labels.tolist()})
    symbol_universe_hash = stable_sha256(sorted(str(symbol).upper() for symbol in target_symbols))
    feature_set_hash = stable_sha256(list(feature_keys))

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
            dataset_fingerprint=dataset_fingerprint,
            symbol_universe_hash=symbol_universe_hash,
            feature_set_hash=feature_set_hash,
        )

    l2_lambda = float(rules["l2_lambda"])
    learning_rate = float(rules["learning_rate"])
    iterations = int(rules["iterations"])

    folds = _run_walk_forward(
        feature_keys, features, labels, meta,
        requested_folds=int(rules["walk_forward_folds"]),
        l2_lambda=l2_lambda, learning_rate=learning_rate, iterations=iterations,
    )
    message = None
    scored_folds = [fold for fold in folds if fold.accuracy is not None]
    if scored_folds:
        holdout_accuracy = _weighted_average([(fold.accuracy, fold.test_samples) for fold in scored_folds])
        holdout_brier_score = _weighted_average([(fold.brier_score, fold.test_samples) for fold in scored_folds])
        holdout_samples = sum(fold.test_samples for fold in scored_folds)
        baseline_folds = [fold for fold in scored_folds if fold.baseline_accuracy is not None]
        baseline_accuracy = _weighted_average(
            [(fold.baseline_accuracy, fold.test_samples) for fold in baseline_folds]
        )
        baseline_brier_score = _weighted_average(
            [(fold.baseline_brier_score, fold.test_samples) for fold in baseline_folds]
        )
    else:
        holdout_accuracy = None
        holdout_brier_score = None
        holdout_samples = 0
        baseline_accuracy = None
        baseline_brier_score = None
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
        baseline_accuracy=baseline_accuracy,
        baseline_brier_score=baseline_brier_score,
        walk_forward_folds=folds,
        coefficients=final_model.coefficients(),
        predictions=predictions,
        dataset_fingerprint=dataset_fingerprint,
        symbol_universe_hash=symbol_universe_hash,
        feature_set_hash=feature_set_hash,
    )
