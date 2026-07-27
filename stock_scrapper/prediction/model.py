"""A small, transparent, hand-rolled logistic regression.

Deliberately not scikit-learn: this project's whole design point is that
every score is explainable, so the fitted model is a plain list of
(feature, weight) pairs a user can read directly, not an opaque object. This
also avoids adding a heavy new dependency for one linear model.

Fitting is full-batch gradient descent with L2 regularization, zero
initialization, and a fixed iteration count — no randomness anywhere, so the
same training rows always produce the exact same weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _clip_for_exp(z: np.ndarray) -> np.ndarray:
    return np.clip(z, -35.0, 35.0)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-_clip_for_exp(z)))


@dataclass(slots=True)
class LogisticRegressionModel:
    """A fitted, standardized logistic regression with named features."""

    feature_keys: list[str]
    feature_means: list[float]
    feature_stds: list[float]
    weights: list[float]
    bias: float

    def predict_proba(self, feature_values: list[float]) -> float:
        """Return the predicted probability of the positive class for one row."""
        standardized = [
            (value - mean) / std if std > 0 else 0.0
            for value, mean, std in zip(feature_values, self.feature_means, self.feature_stds)
        ]
        z = self.bias + sum(weight * value for weight, value in zip(self.weights, standardized))
        return float(_sigmoid(np.array([z]))[0])

    def coefficients(self) -> list[tuple[str, float]]:
        """Feature/weight pairs sorted by absolute influence, most influential first."""
        return sorted(
            zip(self.feature_keys, self.weights),
            key=lambda item: -abs(item[1]),
        )


def fit_logistic_regression(
    feature_keys: list[str],
    features: np.ndarray,
    labels: np.ndarray,
    *,
    l2_lambda: float,
    learning_rate: float,
    iterations: int,
) -> LogisticRegressionModel:
    """Fit a deterministic, L2-regularized logistic regression via full-batch gradient descent."""
    if features.ndim != 2 or features.shape[0] != labels.shape[0]:
        raise ValueError("features must be a 2D array with one row per label")
    if features.shape[0] == 0:
        raise ValueError("Cannot fit a model with zero training rows")
    means = features.mean(axis=0)
    stds = features.std(axis=0)
    safe_stds = np.where(stds > 0, stds, 1.0)
    standardized = (features - means) / safe_stds

    sample_count, feature_count = standardized.shape
    weights = np.zeros(feature_count)
    bias = 0.0
    for _ in range(iterations):
        linear = standardized @ weights + bias
        predictions = _sigmoid(linear)
        error = predictions - labels
        gradient_weights = (standardized.T @ error) / sample_count + l2_lambda * weights
        gradient_bias = float(error.mean())
        weights = weights - learning_rate * gradient_weights
        bias = bias - learning_rate * gradient_bias

    return LogisticRegressionModel(
        feature_keys=list(feature_keys),
        feature_means=[float(value) for value in means],
        feature_stds=[float(value) for value in safe_stds],
        weights=[float(value) for value in weights],
        bias=float(bias),
    )


def evaluate_holdout(model: LogisticRegressionModel, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Accuracy and Brier score (mean squared probability error) on held-out rows."""
    if features.shape[0] == 0:
        return {"accuracy": float("nan"), "brier_score": float("nan"), "sample_count": 0}
    predictions = np.array([model.predict_proba(list(row)) for row in features])
    predicted_labels = (predictions >= 0.5).astype(float)
    accuracy = float((predicted_labels == labels).mean())
    brier_score = float(np.mean((predictions - labels) ** 2))
    return {"accuracy": accuracy, "brier_score": brier_score, "sample_count": int(features.shape[0])}
