from __future__ import annotations

import numpy as np
import pytest

from stock_scrapper.prediction.model import (
    evaluate_holdout,
    fit_logistic_regression,
)


def _separable_dataset(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    positive = rng.normal(loc=2.0, scale=0.5, size=(60, 2))
    negative = rng.normal(loc=-2.0, scale=0.5, size=(60, 2))
    features = np.vstack([positive, negative])
    labels = np.concatenate([np.ones(60), np.zeros(60)])
    return features, labels


def test_fit_logistic_regression_separates_linearly_separable_classes() -> None:
    features, labels = _separable_dataset()
    model = fit_logistic_regression(
        ["feature_a", "feature_b"], features, labels,
        l2_lambda=0.001, learning_rate=0.5, iterations=300,
    )
    metrics = evaluate_holdout(model, features, labels)
    assert metrics["accuracy"] > 0.95
    assert metrics["brier_score"] < 0.05


def test_fit_logistic_regression_is_deterministic() -> None:
    features, labels = _separable_dataset()
    model_a = fit_logistic_regression(
        ["a", "b"], features, labels, l2_lambda=0.01, learning_rate=0.3, iterations=200
    )
    model_b = fit_logistic_regression(
        ["a", "b"], features, labels, l2_lambda=0.01, learning_rate=0.3, iterations=200
    )
    assert model_a.weights == model_b.weights
    assert model_a.bias == model_b.bias


def test_fit_logistic_regression_rejects_empty_dataset() -> None:
    with pytest.raises(ValueError, match="zero training rows"):
        fit_logistic_regression(["a"], np.empty((0, 1)), np.empty((0,)), l2_lambda=0.0, learning_rate=0.1, iterations=10)


def test_predict_proba_matches_feature_direction() -> None:
    features, labels = _separable_dataset()
    model = fit_logistic_regression(
        ["feature_a", "feature_b"], features, labels,
        l2_lambda=0.001, learning_rate=0.5, iterations=300,
    )
    high_probability = model.predict_proba([3.0, 3.0])
    low_probability = model.predict_proba([-3.0, -3.0])
    assert high_probability > 0.9
    assert low_probability < 0.1


def test_coefficients_sorted_by_absolute_influence() -> None:
    features, labels = _separable_dataset()
    # Scale feature_b down so its learned weight should end up smaller in magnitude.
    features = features.copy()
    features[:, 1] *= 0.001
    model = fit_logistic_regression(
        ["feature_a", "feature_b"], features, labels,
        l2_lambda=0.001, learning_rate=0.5, iterations=300,
    )
    coefficients = model.coefficients()
    assert coefficients[0][0] == "feature_a"


def test_evaluate_holdout_handles_empty_input() -> None:
    features, labels = _separable_dataset()
    model = fit_logistic_regression(
        ["feature_a", "feature_b"], features, labels,
        l2_lambda=0.001, learning_rate=0.5, iterations=50,
    )
    metrics = evaluate_holdout(model, np.empty((0, 2)), np.empty((0,)))
    assert metrics["sample_count"] == 0
