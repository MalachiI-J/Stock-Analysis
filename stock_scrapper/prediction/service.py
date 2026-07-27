"""Orchestrate the experimental forward-return prediction: train, evaluate, predict.

Everything here is fit fresh from the caller-supplied, as-of-date-bounded
histories every time it runs — nothing is persisted or reused across calls.
Given the same stored price history and the same config, the result is
exactly reproducible (no randomness anywhere in the pipeline).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping, Sequence

from stock_scrapper.analysis.service import AnalysisService
from stock_scrapper.prediction.dataset import build_training_dataset, select_sample_dates
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
    """One symbol's predicted probability of a positive forward return, or why it has none."""

    symbol: str
    probability_positive: float | None
    reason: str | None = None


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
    holdout_accuracy: float | None = None
    holdout_brier_score: float | None = None
    coefficients: list[tuple[str, float]] = field(default_factory=list)
    predictions: list[SymbolPrediction] = field(default_factory=list)


def run_prediction(
    service: AnalysisService,
    target_symbols: Sequence[str],
    histories: Mapping[str, list[dict[str, Any]]],
    trading_dates: Sequence[str],
    *,
    as_of_date: str,
    rules: Mapping[str, Any],
) -> PredictionRunResult:
    """Train a fresh logistic regression on embargoed history and predict for today."""
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

    features, labels, meta = build_training_dataset(
        service, target_symbols, histories, sample_dates,
        horizon_days=horizon_days, feature_keys=rules["feature_keys"],
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

    split_index = int(features.shape[0] * float(rules["train_fraction"]))
    split_index = max(1, min(split_index, features.shape[0] - 1))
    train_features, holdout_features = features[:split_index], features[split_index:]
    train_labels, holdout_labels = labels[:split_index], labels[split_index:]

    model = fit_logistic_regression(
        list(rules["feature_keys"]), train_features, train_labels,
        l2_lambda=float(rules["l2_lambda"]),
        learning_rate=float(rules["learning_rate"]),
        iterations=int(rules["iterations"]),
    )
    holdout_metrics = evaluate_holdout(model, holdout_features, holdout_labels)

    today_batch = service.analyze_loaded_many_as_of(
        list(target_symbols), histories, date.fromisoformat(as_of_date), persist=False
    )
    predictions: list[SymbolPrediction] = []
    for result in today_batch.results:
        if not result.eligible_for_scoring:
            predictions.append(
                SymbolPrediction(result.symbol, None, "Not eligible for scoring (insufficient or blocked data)")
            )
            continue
        values = [_finite(result.indicators.get(key)) for key in rules["feature_keys"]]
        if any(value is None for value in values):
            predictions.append(
                SymbolPrediction(result.symbol, None, "One or more required indicators are unavailable")
            )
            continue
        predictions.append(SymbolPrediction(result.symbol, model.predict_proba(values), None))

    return PredictionRunResult(
        status="ok",
        message=None,
        as_of_date=as_of_date,
        horizon_days=horizon_days,
        training_samples=int(train_features.shape[0]),
        holdout_samples=int(holdout_features.shape[0]),
        training_start_date=meta[0]["date"] if meta else None,
        training_end_date=meta[split_index - 1]["date"] if meta else None,
        holdout_accuracy=holdout_metrics["accuracy"] if holdout_features.shape[0] else None,
        holdout_brier_score=holdout_metrics["brier_score"] if holdout_features.shape[0] else None,
        coefficients=model.coefficients(),
        predictions=predictions,
    )
