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

    def _validate_gbm_section(gbm_raw: Any, *, prefix: str) -> dict[str, Any]:
        if not isinstance(gbm_raw, dict):
            raise ValueError(f"{prefix} must be a mapping of gradient-boosting hyperparameters")

        def _section_positive_int(key: str) -> int:
            value = gbm_raw.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{prefix}.{key} must be a positive integer")
            return value

        def _section_positive_number(key: str) -> float:
            value = gbm_raw.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{prefix}.{key} must be a positive number")
            return float(value)

        n_estimators = _section_positive_int("n_estimators")
        max_depth = _section_positive_int("max_depth")
        section_learning_rate = _section_positive_number("learning_rate")
        min_samples_leaf = _section_positive_int("min_samples_leaf")
        min_samples_split = _section_positive_int("min_samples_split")
        section_l2_lambda = gbm_raw.get("l2_lambda")
        if isinstance(section_l2_lambda, bool) or not isinstance(section_l2_lambda, (int, float)) or section_l2_lambda < 0:
            raise ValueError(f"{prefix}.l2_lambda must be a nonnegative number")
        return {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": section_learning_rate,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_split": min_samples_split,
            "l2_lambda": float(section_l2_lambda),
        }

    # predict-v4's gradient-boosted regressor (stock_scrapper/prediction/gbm.py) shares
    # horizon_days/lookback_years/sample_stride_sessions/walk_forward_folds/
    # minimum_training_samples/feature_keys above with predict-v3, so both models train
    # and evaluate over identical rows and stay directly comparable — only its own
    # model hyperparameters live in this nested section.
    gbm_section = _validate_gbm_section(rules.get("gbm"), prefix="gbm")

    # predict-v5 widens predict-v4's feature list (base technical keys + fundamentals)
    # in its own nested section. Its "gbm" sub-section is optional -- absent, it falls
    # back to the shared "gbm" section above; present, it lets predict-v5 use its own
    # (typically heavier) regularization, since fundamentals only update quarterly and
    # can otherwise let boosting compound an extreme, poorly-supported leaf value (see
    # the prediction_rules.yaml comment for the concrete failure this fixed).
    predict_v5_raw = rules.get("predict_v5")
    if not isinstance(predict_v5_raw, dict):
        raise ValueError("predict_v5 must be a mapping")
    predict_v5_feature_keys = predict_v5_raw.get("feature_keys")
    if (
        not isinstance(predict_v5_feature_keys, list)
        or not predict_v5_feature_keys
        or not all(isinstance(key, str) and key.strip() for key in predict_v5_feature_keys)
    ):
        raise ValueError("predict_v5.feature_keys must be a non-empty list of indicator/fundamental names")
    predict_v5_gbm_raw = predict_v5_raw.get("gbm")
    predict_v5_gbm_section = (
        _validate_gbm_section(predict_v5_gbm_raw, prefix="predict_v5.gbm")
        if predict_v5_gbm_raw is not None else None
    )

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
        "gbm": gbm_section,
        "predict_v5": {
            "feature_keys": [str(key) for key in predict_v5_feature_keys],
            "gbm": predict_v5_gbm_section,
        },
    }
