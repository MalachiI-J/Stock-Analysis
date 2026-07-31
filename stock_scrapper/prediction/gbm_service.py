"""Orchestrate predict-v4: train, evaluate, predict — gradient-boosted regression
trees (``gbm.py``) on a continuous excess-return target (``dataset.py``'s
``build_regression_dataset``), instead of predict-v3's linear logistic regression on
a binarized "beat the benchmark" label.

Deliberately mirrors ``service.py``'s structure closely — the same date-grouped,
purged, expanding-window walk-forward (``_fold_boundaries`` is reused directly, not
reimplemented), the same fold-specific honest baselines, the same full-dataset
refit before predicting today — so predict-v3 and predict-v4 stay directly
comparable and neither model gets an evaluation methodology the other didn't. Only
the model, its target, and its evaluation metrics differ: mean squared error and an
information coefficient (Pearson correlation between predicted and actual excess
return) in place of accuracy/Brier score, since the target is now continuous.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping, Sequence

from stock_scrapper.analysis.risk_diagnostics import pearson_correlation
from stock_scrapper.analysis.service import AnalysisService
from stock_scrapper.prediction.dataset import (
    build_regression_dataset,
    feature_value,
    opportunity_percentiles,
    select_sample_dates,
)
from stock_scrapper.processing.fundamentals_features import fundamentals_features_as_of
from stock_scrapper.prediction.gbm import (
    GradientBoostedRegressor,
    evaluate_regression_holdout,
    fit_gradient_boosting,
)
from stock_scrapper.prediction.service import _fold_boundaries, _weighted_average
from stock_scrapper.utilities.hashing import stable_sha256


# A prediction more than this many training-target standard deviations from zero is
# flagged rather than trusted at face value — tree ensembles can extrapolate a rare,
# poorly-supported feature combination into an extreme value (see the real INTC case
# in README's "Evaluation honesty": +323.73%, then +266.11% even after a 4x/10x
# regularization increase, revealing this reflects genuine but sparse/noisy historical
# precedent, not a simple leaf-overfitting artifact regularization alone fixes).
OUTLIER_STD_MULTIPLIER = 3.0


@dataclass(slots=True)
class SymbolExcessReturnPrediction:
    """One symbol's predicted forward excess return over the benchmark, or why it
    has none. Unlike predict-v3's probability, this is a signed magnitude — e.g.
    ``+0.02`` means "predicted to beat the benchmark by about 2 percentage points".

    ``low_confidence`` is never used to suppress or alter ``predicted_excess_return``
    — it only flags that the raw value is statistically extreme (see
    ``OUTLIER_STD_MULTIPLIER``) relative to the training target's own spread, so a
    caller can warn rather than silently presenting an extrapolated outlier with the
    same apparent confidence as an ordinary prediction.
    """

    symbol: str
    predicted_excess_return: float | None
    reason: str | None = None
    low_confidence: bool = False


@dataclass(slots=True)
class GbmWalkForwardFold:
    """One expanding-window fold's train/test split and honest baseline — see
    ``WalkForwardFold`` in ``service.py`` for the shared date-grouped/purged
    splitting rationale; the fields here are the regression analogues of that
    dataclass's classification metrics.

    ``baseline_mse`` is this fold's own test-period target variance — the true
    minimum achievable MSE for a predictor that always outputs the test period's own
    mean excess return, mirroring how ``WalkForwardFold.baseline_accuracy`` is built
    from each fold's own label rate rather than a fixed assumption.
    """

    fold: int
    training_samples: int
    test_samples: int
    mse: float | None
    mean_absolute_error: float | None
    # Pearson correlation between predicted and actual excess return on the test
    # set — a standard quant "information coefficient"; unlike accuracy, this can be
    # computed even though the target is continuous rather than binary.
    information_coefficient: float | None
    training_start_date: str | None = None
    training_end_date: str | None = None
    test_start_date: str | None = None
    test_end_date: str | None = None
    training_symbol_count: int = 0
    test_symbol_count: int = 0
    purged_samples: int = 0
    baseline_mse: float | None = None
    # Positive means the model's MSE is *lower* (better) than the baseline's —
    # baseline minus model, matching WalkForwardFold.brier_improvement_vs_baseline's
    # sign convention.
    mse_improvement_vs_baseline: float | None = None


@dataclass(slots=True)
class GbmPredictionRunResult:
    """Complete outcome of one predict-v4 run."""

    status: str  # "ok" or "insufficient_data"
    message: str | None
    as_of_date: str
    horizon_days: int
    training_samples: int = 0
    holdout_samples: int = 0
    training_start_date: str | None = None
    training_end_date: str | None = None
    holdout_mse: float | None = None
    holdout_mean_absolute_error: float | None = None
    holdout_information_coefficient: float | None = None
    # Sample-weighted (by each fold's test_samples) aggregate of every fold's own
    # test-period target variance — the honest comparison point for holdout_mse,
    # built the same way holdout_mse itself is aggregated.
    baseline_mse: float | None = None
    walk_forward_folds: list[GbmWalkForwardFold] = field(default_factory=list)
    feature_importances: list[tuple[str, float]] = field(default_factory=list)
    predictions: list[SymbolExcessReturnPrediction] = field(default_factory=list)
    dataset_fingerprint: str | None = None
    symbol_universe_hash: str | None = None
    feature_set_hash: str | None = None


def _run_gbm_walk_forward(
    feature_keys: list[str],
    features: Any,
    targets: Any,
    meta: list[dict[str, str]],
    *,
    requested_folds: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    min_samples_leaf: int,
    min_samples_split: int,
    l2_lambda: float,
) -> list[GbmWalkForwardFold]:
    """Expanding-window walk-forward, split by unique calendar date and purged of
    label/test-period overlap — identical splitting logic to
    ``service.py::_run_walk_forward`` (see that function's docstring), just fitting
    a gradient-boosted regressor instead of a logistic regression each fold."""
    unique_dates = sorted({row["date"] for row in meta})
    date_count = len(unique_dates)
    folds = min(requested_folds, date_count - 1) if date_count > 0 else 0
    if folds < 1:
        return []
    boundaries = _fold_boundaries(date_count, folds)
    date_rank = {value: index for index, value in enumerate(unique_dates)}
    row_date_ranks = [date_rank[row["date"]] for row in meta]

    results: list[GbmWalkForwardFold] = []
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

        train_features, train_targets = features[kept_train_indices], targets[kept_train_indices]
        test_features, test_targets = features[test_indices], targets[test_indices]

        model = fit_gradient_boosting(
            feature_keys, train_features, train_targets,
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, l2_lambda=l2_lambda,
        )
        metrics = evaluate_regression_holdout(model, test_features, test_targets)
        mse = metrics["mse"] if metrics["sample_count"] else None
        mean_absolute_error = metrics["mean_absolute_error"] if metrics["sample_count"] else None

        if test_targets.shape[0]:
            predictions = model.predict_batch(test_features)
            information_coefficient = pearson_correlation(list(predictions), list(test_targets))
            baseline_mse = float(test_targets.var())
        else:
            information_coefficient = None
            baseline_mse = None

        train_dates = [meta[index]["date"] for index in kept_train_indices]
        test_dates = [meta[index]["date"] for index in test_indices]

        results.append(
            GbmWalkForwardFold(
                fold=len(results) + 1,
                training_samples=len(kept_train_indices),
                test_samples=len(test_indices),
                mse=mse,
                mean_absolute_error=mean_absolute_error,
                information_coefficient=information_coefficient,
                training_start_date=min(train_dates),
                training_end_date=max(train_dates),
                test_start_date=min(test_dates),
                test_end_date=max(test_dates),
                training_symbol_count=len({meta[index]["symbol"] for index in kept_train_indices}),
                test_symbol_count=len({meta[index]["symbol"] for index in test_indices}),
                purged_samples=len(purged_indices),
                baseline_mse=baseline_mse,
                mse_improvement_vs_baseline=(
                    None if mse is None or baseline_mse is None else baseline_mse - mse
                ),
            )
        )
    return results


def _latest_price_on_or_before(history: Sequence[Mapping[str, Any]], as_of_date: str) -> float | None:
    known = [row for row in history if str(row.get("trade_date", ""))[:10] <= as_of_date]
    if not known:
        return None
    latest = max(known, key=lambda row: str(row.get("trade_date"))[:10])
    try:
        return float(latest["adjusted_close"])
    except (TypeError, ValueError, KeyError):
        return None


def run_gbm_prediction(
    service: AnalysisService,
    target_symbols: Sequence[str],
    histories: Mapping[str, list[dict[str, Any]]],
    trading_dates: Sequence[str],
    *,
    as_of_date: str,
    rules: Mapping[str, Any],
    benchmark_symbol: str,
    feature_keys: Sequence[str] | None = None,
    gbm_config: Mapping[str, Any] | None = None,
    fundamentals_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> GbmPredictionRunResult:
    """Walk-forward evaluate, then fit a final model on all embargoed history, and
    predict today's excess return — the predict-v4 analogue of
    ``service.py::run_prediction``.

    ``feature_keys``/``gbm_config`` default to ``rules["feature_keys"]``/
    ``rules["gbm"]`` (predict-v4's exact behavior) but let a caller — namely
    ``predict_v5``, which widens the feature list with point-in-time fundamentals —
    pass its own, reusing this same walk-forward/fit/predict machinery rather than
    forking it. ``fundamentals_by_symbol`` (raw SEC EDGAR fact records per symbol) is
    threaded straight through to ``build_regression_dataset`` and to today's live
    prediction step; omitted, this function's behavior is unchanged from predict-v4.
    """
    horizon_days = int(rules["horizon_days"])
    sample_dates = select_sample_dates(
        trading_dates,
        as_of_date=as_of_date,
        horizon_days=horizon_days,
        lookback_years=float(rules["lookback_years"]),
        stride_sessions=int(rules["sample_stride_sessions"]),
    )
    if not sample_dates:
        return GbmPredictionRunResult(
            status="insufficient_data",
            message="Not enough historical trading sessions are stored to build any training sample.",
            as_of_date=as_of_date,
            horizon_days=horizon_days,
        )

    feature_keys = list(feature_keys) if feature_keys is not None else list(rules["feature_keys"])
    features, targets, meta = build_regression_dataset(
        service, target_symbols, histories, sample_dates,
        horizon_days=horizon_days, feature_keys=feature_keys, benchmark_symbol=benchmark_symbol,
        fundamentals_by_symbol=fundamentals_by_symbol,
    )
    dataset_fingerprint = stable_sha256({"features": features.tolist(), "targets": targets.tolist()})
    symbol_universe_hash = stable_sha256(sorted(str(symbol).upper() for symbol in target_symbols))
    feature_set_hash = stable_sha256(list(feature_keys))

    minimum_samples = int(rules["minimum_training_samples"])
    if features.shape[0] < minimum_samples:
        return GbmPredictionRunResult(
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

    gbm_rules = gbm_config if gbm_config is not None else rules["gbm"]
    n_estimators = int(gbm_rules["n_estimators"])
    max_depth = int(gbm_rules["max_depth"])
    gbm_learning_rate = float(gbm_rules["learning_rate"])
    min_samples_leaf = int(gbm_rules["min_samples_leaf"])
    min_samples_split = int(gbm_rules["min_samples_split"])
    gbm_l2_lambda = float(gbm_rules["l2_lambda"])

    folds = _run_gbm_walk_forward(
        feature_keys, features, targets, meta,
        requested_folds=int(rules["walk_forward_folds"]),
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=gbm_learning_rate,
        min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, l2_lambda=gbm_l2_lambda,
    )
    message = None
    scored_folds = [fold for fold in folds if fold.mse is not None]
    if scored_folds:
        holdout_mse = _weighted_average([(fold.mse, fold.test_samples) for fold in scored_folds])
        holdout_mean_absolute_error = _weighted_average(
            [(fold.mean_absolute_error, fold.test_samples) for fold in scored_folds]
        )
        holdout_samples = sum(fold.test_samples for fold in scored_folds)
        ic_folds = [fold for fold in scored_folds if fold.information_coefficient is not None]
        holdout_information_coefficient = _weighted_average(
            [(fold.information_coefficient, fold.test_samples) for fold in ic_folds]
        )
        baseline_folds = [fold for fold in scored_folds if fold.baseline_mse is not None]
        baseline_mse = _weighted_average([(fold.baseline_mse, fold.test_samples) for fold in baseline_folds])
    else:
        holdout_mse = None
        holdout_mean_absolute_error = None
        holdout_information_coefficient = None
        holdout_samples = 0
        baseline_mse = None
        message = (
            "Not enough samples to run any walk-forward evaluation folds; showing feature "
            "importances with no holdout estimate. Collect more history or lower "
            "walk_forward_folds in config/prediction_rules.yaml."
        )

    # The deployed model is fit on the full embargoed dataset, not just one fold's
    # training slice, since a live prediction should use all the history available.
    final_model = fit_gradient_boosting(
        feature_keys, features, targets,
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=gbm_learning_rate,
        min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, l2_lambda=gbm_l2_lambda,
    )

    # The threshold a live prediction is judged against: how spread out the actual
    # training targets were. A prediction many multiples of that spread away from zero
    # is extrapolating into territory the model saw little or nothing like in training.
    training_target_std = float(targets.std()) if targets.shape[0] else 0.0
    outlier_threshold = OUTLIER_STD_MULTIPLIER * training_target_std

    today_batch = service.analyze_loaded_many_as_of(
        list(target_symbols), histories, date.fromisoformat(as_of_date), persist=False
    )
    percentiles = opportunity_percentiles(today_batch.results)
    predictions: list[SymbolExcessReturnPrediction] = []
    for result in today_batch.results:
        if not result.eligible_for_scoring:
            predictions.append(
                SymbolExcessReturnPrediction(result.symbol, None, "Not eligible for scoring (insufficient or blocked data)")
            )
            continue
        fundamentals = (
            fundamentals_features_as_of(
                fundamentals_by_symbol.get(result.symbol, ()),
                date.fromisoformat(as_of_date),
                price=_latest_price_on_or_before(histories.get(result.symbol, []), as_of_date),
            )
            if fundamentals_by_symbol is not None else None
        )
        values = [feature_value(result, key, percentiles, fundamentals) for key in feature_keys]
        if any(value is None for value in values):
            predictions.append(
                SymbolExcessReturnPrediction(result.symbol, None, "One or more required indicators are unavailable")
            )
            continue
        predicted_value = final_model.predict(values)
        low_confidence = outlier_threshold > 0 and abs(predicted_value) > outlier_threshold
        predictions.append(SymbolExcessReturnPrediction(result.symbol, predicted_value, None, low_confidence))

    return GbmPredictionRunResult(
        status="ok",
        message=message,
        as_of_date=as_of_date,
        horizon_days=horizon_days,
        training_samples=int(features.shape[0]),
        holdout_samples=holdout_samples,
        training_start_date=meta[0]["date"] if meta else None,
        training_end_date=meta[-1]["date"] if meta else None,
        holdout_mse=holdout_mse,
        holdout_mean_absolute_error=holdout_mean_absolute_error,
        holdout_information_coefficient=holdout_information_coefficient,
        baseline_mse=baseline_mse,
        walk_forward_folds=folds,
        feature_importances=final_model.feature_importances(),
        predictions=predictions,
        dataset_fingerprint=dataset_fingerprint,
        symbol_universe_hash=symbol_universe_hash,
        feature_set_hash=feature_set_hash,
    )
