"""Load and strictly validate config/prediction_rules.yaml."""

from __future__ import annotations

from typing import Any


def validate_prediction_config(rules: dict[str, Any]) -> dict[str, Any]:
    """Validate the experimental prediction configuration, rejecting unusable values."""
    if not isinstance(rules, dict):
        raise ValueError("prediction_rules.yaml must contain a mapping")

    def _positive_int(key: str) -> int:
        value = rules.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
        return value

    def _positive_number(key: str) -> float:
        value = rules.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{key} must be a positive number")
        return float(value)

    horizon_days = _positive_int("horizon_days")
    lookback_years = _positive_number("lookback_years")
    sample_stride_sessions = _positive_int("sample_stride_sessions")
    l2_lambda = rules.get("l2_lambda")
    if isinstance(l2_lambda, bool) or not isinstance(l2_lambda, (int, float)) or l2_lambda < 0:
        raise ValueError("l2_lambda must be a nonnegative number")
    learning_rate = _positive_number("learning_rate")
    iterations = _positive_int("iterations")
    minimum_training_samples = _positive_int("minimum_training_samples")

    walk_forward_folds = _positive_int("walk_forward_folds")

    feature_keys = rules.get("feature_keys")
    if not isinstance(feature_keys, list) or not feature_keys or not all(isinstance(key, str) and key.strip() for key in feature_keys):
        raise ValueError("feature_keys must be a non-empty list of indicator names")

    prediction_version = rules.get("prediction_version")
    if not isinstance(prediction_version, str) or not prediction_version.strip():
        raise ValueError("prediction_version must be a non-empty string")

    # predict-v4's gradient-boosted regressor (stock_scrapper/prediction/gbm.py) shares
    # horizon_days/lookback_years/sample_stride_sessions/walk_forward_folds/
    # minimum_training_samples/feature_keys above with predict-v3, so both models train
    # and evaluate over identical rows and stay directly comparable — only its own
    # model hyperparameters live in this nested section.
    gbm_raw = rules.get("gbm")
    if not isinstance(gbm_raw, dict):
        raise ValueError("gbm must be a mapping of predict-v4 hyperparameters")

    def _gbm_positive_int(key: str) -> int:
        value = gbm_raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"gbm.{key} must be a positive integer")
        return value

    def _gbm_positive_number(key: str) -> float:
        value = gbm_raw.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"gbm.{key} must be a positive number")
        return float(value)

    gbm_n_estimators = _gbm_positive_int("n_estimators")
    gbm_max_depth = _gbm_positive_int("max_depth")
    gbm_learning_rate = _gbm_positive_number("learning_rate")
    gbm_min_samples_leaf = _gbm_positive_int("min_samples_leaf")
    gbm_min_samples_split = _gbm_positive_int("min_samples_split")
    gbm_l2_lambda = gbm_raw.get("l2_lambda")
    if isinstance(gbm_l2_lambda, bool) or not isinstance(gbm_l2_lambda, (int, float)) or gbm_l2_lambda < 0:
        raise ValueError("gbm.l2_lambda must be a nonnegative number")

    return {
        "prediction_version": prediction_version,
        "horizon_days": horizon_days,
        "lookback_years": float(lookback_years),
        "sample_stride_sessions": sample_stride_sessions,
        "walk_forward_folds": walk_forward_folds,
        "l2_lambda": float(l2_lambda),
        "learning_rate": learning_rate,
        "iterations": iterations,
        "minimum_training_samples": minimum_training_samples,
        "feature_keys": [str(key) for key in feature_keys],
        "gbm": {
            "n_estimators": gbm_n_estimators,
            "max_depth": gbm_max_depth,
            "learning_rate": gbm_learning_rate,
            "min_samples_leaf": gbm_min_samples_leaf,
            "min_samples_split": gbm_min_samples_split,
            "l2_lambda": float(gbm_l2_lambda),
        },
    }
