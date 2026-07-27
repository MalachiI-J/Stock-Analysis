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
    }
