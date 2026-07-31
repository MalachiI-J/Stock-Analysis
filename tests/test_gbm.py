from __future__ import annotations

import numpy as np
import pytest

from stock_scrapper.prediction.gbm import (
    GradientBoostedRegressor,
    _best_split,
    _build_tree,
    _TreeNode,
    evaluate_regression_holdout,
    fit_gradient_boosting,
)


def test_best_split_finds_the_obvious_threshold() -> None:
    # Values below 5 have residual -1, values >= 5 have residual +1: the only sane
    # split is between index 4 and 5 (feature values 4 and 5).
    feature = np.array([0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    residuals = np.array([-1.0, -1, -1, -1, -1, 1, 1, 1, 1, 1])

    result = _best_split(feature, residuals, min_samples_leaf=1)

    assert result is not None
    threshold, gain = result
    assert threshold == pytest.approx(4.5)
    assert gain > 0


def test_best_split_respects_min_samples_leaf() -> None:
    feature = np.array([0.0, 1, 2, 3])
    residuals = np.array([-1.0, -1, 1, 1])

    assert _best_split(feature, residuals, min_samples_leaf=3) is None


def test_best_split_returns_none_when_no_variance_to_explain() -> None:
    feature = np.array([0.0, 1, 2, 3, 4])
    residuals = np.array([1.0, 1, 1, 1, 1])

    assert _best_split(feature, residuals, min_samples_leaf=1) is None


def test_best_split_returns_none_for_constant_feature() -> None:
    feature = np.array([5.0, 5, 5, 5])
    residuals = np.array([-1.0, 1, -1, 1])

    assert _best_split(feature, residuals, min_samples_leaf=1) is None


def test_build_tree_reduces_sse_versus_a_single_leaf() -> None:
    rng_features = np.array([[float(i)] for i in range(20)])
    residuals = np.array([-2.0 if i < 10 else 2.0 for i in range(20)])
    single_leaf_sse = float(np.sum((residuals - residuals.mean()) ** 2))

    tree = _build_tree(
        rng_features, residuals, depth=0, max_depth=2,
        min_samples_leaf=1, min_samples_split=2, l2_lambda=0.0,
    )

    assert not tree.is_leaf
    predictions = np.array([
        _predict_row(tree, row) for row in rng_features
    ])
    tree_sse = float(np.sum((residuals - predictions) ** 2))
    assert tree_sse < single_leaf_sse


def _predict_row(node: _TreeNode, row: np.ndarray) -> float:
    while not node.is_leaf:
        node = node.left if row[node.feature_index] <= node.threshold else node.right
    return node.value


def test_fit_gradient_boosting_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        fit_gradient_boosting(
            ["a"], np.zeros((3, 1)), np.zeros(2),
            n_estimators=1, max_depth=1, learning_rate=0.1,
            min_samples_leaf=1, min_samples_split=2, l2_lambda=0.0,
        )


def test_fit_gradient_boosting_rejects_zero_rows() -> None:
    with pytest.raises(ValueError):
        fit_gradient_boosting(
            ["a"], np.zeros((0, 1)), np.zeros(0),
            n_estimators=1, max_depth=1, learning_rate=0.1,
            min_samples_leaf=1, min_samples_split=2, l2_lambda=0.0,
        )


def test_fit_gradient_boosting_reduces_training_mse_over_a_flat_baseline() -> None:
    # A clean nonlinear (U-shaped) signal: target is high at both extremes of x and
    # low in the middle — exactly the shape a single linear coefficient cannot
    # represent, but a shallow tree splits on directly.
    x = np.linspace(-1.0, 1.0, 200)
    y = x ** 2
    features = x.reshape(-1, 1)

    model = fit_gradient_boosting(
        ["x"], features, y,
        n_estimators=50, max_depth=3, learning_rate=0.3,
        min_samples_leaf=2, min_samples_split=4, l2_lambda=0.0,
    )

    baseline_mse = float(np.mean((y - y.mean()) ** 2))
    predictions = model.predict_batch(features)
    model_mse = float(np.mean((y - predictions) ** 2))
    assert model_mse < baseline_mse * 0.1


def test_predict_single_row_matches_predict_batch() -> None:
    x = np.linspace(-1.0, 1.0, 60)
    y = x ** 2
    features = x.reshape(-1, 1)
    model = fit_gradient_boosting(
        ["x"], features, y,
        n_estimators=10, max_depth=2, learning_rate=0.5,
        min_samples_leaf=2, min_samples_split=4, l2_lambda=0.0,
    )

    batch_predictions = model.predict_batch(features)
    single_predictions = [model.predict([float(value)]) for value in x]

    np.testing.assert_allclose(batch_predictions, single_predictions, atol=1e-9)


def test_fit_gradient_boosting_is_deterministic() -> None:
    x = np.linspace(-1.0, 1.0, 60)
    y = x ** 2 + 0.01 * np.sin(x * 10)
    features = x.reshape(-1, 1)

    model_a = fit_gradient_boosting(
        ["x"], features, y, n_estimators=15, max_depth=3, learning_rate=0.2,
        min_samples_leaf=2, min_samples_split=4, l2_lambda=0.5,
    )
    model_b = fit_gradient_boosting(
        ["x"], features, y, n_estimators=15, max_depth=3, learning_rate=0.2,
        min_samples_leaf=2, min_samples_split=4, l2_lambda=0.5,
    )

    np.testing.assert_allclose(model_a.predict_batch(features), model_b.predict_batch(features))


def test_feature_importances_favor_the_informative_feature() -> None:
    rng = np.random.default_rng(0)
    n = 300
    informative = rng.uniform(-1, 1, n)
    noise = rng.uniform(-1, 1, n)
    y = informative ** 2
    features = np.column_stack([informative, noise])

    model = fit_gradient_boosting(
        ["informative", "noise"], features, y,
        n_estimators=30, max_depth=3, learning_rate=0.3,
        min_samples_leaf=3, min_samples_split=6, l2_lambda=0.0,
    )

    importances = dict(model.feature_importances())
    assert importances["informative"] > importances["noise"]
    assert importances["informative"] == pytest.approx(sum(importances.values()) - importances["noise"])
    assert sum(importances.values()) == pytest.approx(1.0)


def test_feature_importances_all_zero_with_no_trees() -> None:
    model = GradientBoostedRegressor(feature_keys=["a", "b"], initial_value=0.0, learning_rate=0.1, trees=[])

    assert model.feature_importances() == [("a", 0.0), ("b", 0.0)]


def test_evaluate_regression_holdout_matches_hand_computed_metrics() -> None:
    model = GradientBoostedRegressor(feature_keys=["a"], initial_value=1.0, learning_rate=0.0, trees=[])
    features = np.array([[0.0], [1.0], [2.0]])
    targets = np.array([1.0, 3.0, -1.0])

    metrics = evaluate_regression_holdout(model, features, targets)

    # learning_rate=0.0 means every prediction is just initial_value=1.0
    assert metrics["mse"] == pytest.approx(np.mean((targets - 1.0) ** 2))
    assert metrics["mean_absolute_error"] == pytest.approx(np.mean(np.abs(targets - 1.0)))
    assert metrics["sample_count"] == 3


def test_evaluate_regression_holdout_empty_features() -> None:
    model = GradientBoostedRegressor(feature_keys=["a"], initial_value=0.0, learning_rate=0.1, trees=[])

    metrics = evaluate_regression_holdout(model, np.empty((0, 1)), np.empty(0))

    assert metrics["sample_count"] == 0
    assert metrics["mse"] != metrics["mse"]  # NaN
