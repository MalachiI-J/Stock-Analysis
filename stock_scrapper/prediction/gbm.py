"""A small, transparent, hand-rolled gradient-boosted regression tree ensemble.

predict-v3's logistic regression is linear, but the risk-inversion diagnostics
(``stock_scrapper/analysis/risk_diagnostics.py``, ``investigate-risk-inversion``)
found a genuinely *nonlinear* relationship between at least one existing feature
(trailing six-month return) and forward excess return — a U-shape where both the
most-negative and (to a lesser extent) most-positive quintiles outperform the
middle, which is exactly the kind of pattern a linear model's single coefficient
per feature cannot represent (that U-shape is why the linear correlation for that
feature came out near zero despite the effect being real). A shallow decision tree
can split on the same feature twice at different thresholds and capture exactly this
shape; boosting a sequence of them adds the ability to combine several such
nonlinear splits.

Deliberately not scikit-learn, for the same reason ``model.py``'s logistic
regression isn't: every split is a plain, readable (feature, threshold) decision a
user can trace by hand, not an opaque object, and this avoids a new heavy
dependency for one model. Fitting is fully deterministic — exhaustive best-split
search over every candidate threshold (no random feature/row subsampling), so the
same training rows always produce the exact same trees.

Predicts a *continuous* forward excess return (see ``dataset.py``'s
``build_regression_dataset``), not a binary "beats the benchmark" label — this
keeps the magnitude information predict-v3's binary target discards, which lines
up with the non-monotonic, magnitude-driven patterns the diagnostics already
found.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class _TreeNode:
    """One node of a regression tree. Leaves carry only ``value``; internal nodes
    carry the split and both children. ``gain`` (internal nodes only) is the SSE
    reduction this split achieved, used to attribute feature importance later."""

    is_leaf: bool
    value: float = 0.0
    feature_index: int = -1
    threshold: float = 0.0
    gain: float = 0.0
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None


def _sse(sum_: float, sum_sq: float, count: float) -> float:
    """Sum of squared errors of a group from its own mean, from sufficient statistics
    (sum, sum-of-squares, count) rather than the raw values — this is what lets the
    split search below be a handful of vectorized numpy operations instead of a
    per-candidate-threshold Python loop."""
    if count <= 0:
        return 0.0
    return sum_sq - (sum_ * sum_) / count


def _best_split(
    feature_column: np.ndarray, residuals: np.ndarray, min_samples_leaf: int
) -> tuple[float, float] | None:
    """The threshold and SSE-reduction gain of the best split on one feature column,
    or ``None`` if no split leaves both sides with at least ``min_samples_leaf`` rows
    and a strictly positive gain. Every candidate threshold is scored at once via
    cumulative sums (``prefix_sum``/``prefix_sq``) rather than a Python loop over
    rows — this is the difference between a model that fits in minutes and one that
    doesn't finish, at this project's dataset sizes (tens of thousands of rows).
    """
    order = np.argsort(feature_column, kind="mergesort")
    sorted_feature = feature_column[order]
    sorted_residuals = residuals[order]
    n = sorted_residuals.shape[0]
    if n < 2 * min_samples_leaf:
        return None

    prefix_sum = np.cumsum(sorted_residuals)
    prefix_sq = np.cumsum(sorted_residuals * sorted_residuals)
    total_sum = prefix_sum[-1]
    total_sq = prefix_sq[-1]
    parent_sse = _sse(total_sum, total_sq, n)

    low = min_samples_leaf - 1
    high = n - min_samples_leaf  # candidate split index i splits [0..i] | [i+1..n-1]
    if high <= low:
        return None
    candidate_index = np.arange(low, high)
    left_count = candidate_index + 1
    right_count = n - left_count
    left_sum = prefix_sum[candidate_index]
    right_sum = total_sum - left_sum
    left_sq = prefix_sq[candidate_index]
    right_sq = total_sq - left_sq
    left_sse = left_sq - (left_sum * left_sum) / left_count
    right_sse = right_sq - (right_sum * right_sum) / right_count
    gains = parent_sse - left_sse - right_sse

    # A split between two equal adjacent values doesn't actually separate them
    # (both land on the same side regardless of the threshold), so it isn't real.
    equal_neighbors = sorted_feature[candidate_index] == sorted_feature[candidate_index + 1]
    gains = np.where(equal_neighbors, -np.inf, gains)

    best_local = int(np.argmax(gains))
    best_gain = float(gains[best_local])
    if not np.isfinite(best_gain) or best_gain <= 0.0:
        return None
    best_index = int(candidate_index[best_local])
    threshold = float((sorted_feature[best_index] + sorted_feature[best_index + 1]) / 2.0)
    return threshold, best_gain


def _leaf_value(residuals: np.ndarray, l2_lambda: float) -> float:
    """Mean residual, ridge-shrunk toward zero by ``l2_lambda`` — a small leaf backed
    by few rows gets pulled toward "predict nothing extra" rather than fitting its
    handful of residuals exactly, the standard gradient-boosting regularization for
    squared-error loss (leaf weight = sum(residuals) / (count + l2_lambda))."""
    count = residuals.shape[0]
    denominator = count + l2_lambda
    return float(residuals.sum() / denominator) if denominator > 0 else 0.0


def _build_tree(
    features: np.ndarray,
    residuals: np.ndarray,
    *,
    depth: int,
    max_depth: int,
    min_samples_leaf: int,
    min_samples_split: int,
    l2_lambda: float,
) -> _TreeNode:
    count = residuals.shape[0]
    if depth >= max_depth or count < min_samples_split:
        return _TreeNode(is_leaf=True, value=_leaf_value(residuals, l2_lambda))

    best_feature_index = -1
    best_threshold = 0.0
    best_gain = 0.0
    for feature_index in range(features.shape[1]):
        result = _best_split(features[:, feature_index], residuals, min_samples_leaf)
        if result is None:
            continue
        threshold, gain = result
        if gain > best_gain:
            best_gain = gain
            best_feature_index = feature_index
            best_threshold = threshold

    if best_feature_index < 0:
        return _TreeNode(is_leaf=True, value=_leaf_value(residuals, l2_lambda))

    left_mask = features[:, best_feature_index] <= best_threshold
    right_mask = ~left_mask
    left = _build_tree(
        features[left_mask], residuals[left_mask],
        depth=depth + 1, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split, l2_lambda=l2_lambda,
    )
    right = _build_tree(
        features[right_mask], residuals[right_mask],
        depth=depth + 1, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split, l2_lambda=l2_lambda,
    )
    return _TreeNode(
        is_leaf=False, feature_index=best_feature_index, threshold=best_threshold,
        gain=best_gain, left=left, right=right,
    )


def _predict_tree_row(node: _TreeNode, row: np.ndarray) -> float:
    while not node.is_leaf:
        node = node.left if row[node.feature_index] <= node.threshold else node.right  # type: ignore[assignment]
    return node.value


def _predict_tree_batch(node: _TreeNode, features: np.ndarray) -> np.ndarray:
    """Vectorized tree prediction: recurse on boolean masks instead of one Python
    function call per row, so scoring an entire training set stays fast across many
    boosting rounds."""
    predictions = np.empty(features.shape[0], dtype=float)
    if node.is_leaf:
        predictions.fill(node.value)
        return predictions
    left_mask = features[:, node.feature_index] <= node.threshold
    right_mask = ~left_mask
    if left_mask.any():
        predictions[left_mask] = _predict_tree_batch(node.left, features[left_mask])  # type: ignore[arg-type]
    if right_mask.any():
        predictions[right_mask] = _predict_tree_batch(node.right, features[right_mask])  # type: ignore[arg-type]
    return predictions


def _sum_gain_by_feature(node: _TreeNode, totals: list[float]) -> None:
    if node.is_leaf:
        return
    totals[node.feature_index] += node.gain
    _sum_gain_by_feature(node.left, totals)  # type: ignore[arg-type]
    _sum_gain_by_feature(node.right, totals)  # type: ignore[arg-type]


@dataclass(slots=True)
class GradientBoostedRegressor:
    """A fitted, deterministic gradient-boosted regression tree ensemble predicting a
    continuous target (forward excess return, not a binary label)."""

    feature_keys: list[str]
    initial_value: float
    learning_rate: float
    trees: list[_TreeNode] = field(default_factory=list)

    def predict(self, feature_values: list[float]) -> float:
        """Predict one row — used for today's live predictions, where scoring a
        single symbol doesn't justify the batch machinery below."""
        row = np.array(feature_values, dtype=float)
        prediction = self.initial_value
        for tree in self.trees:
            prediction += self.learning_rate * _predict_tree_row(tree, row)
        return float(prediction)

    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        """Predict every row of a 2D array at once — used for walk-forward holdout
        evaluation, where a Python loop calling ``predict`` per row, per tree, would
        dominate runtime at this project's dataset sizes."""
        if features.shape[0] == 0:
            return np.empty(0, dtype=float)
        predictions = np.full(features.shape[0], self.initial_value, dtype=float)
        for tree in self.trees:
            predictions = predictions + self.learning_rate * _predict_tree_batch(tree, features)
        return predictions

    def feature_importances(self) -> list[tuple[str, float]]:
        """Each feature's share of total SSE-reduction gain across every split in
        every tree, sorted most-influential first — the tree-ensemble analogue of
        ``LogisticRegressionModel.coefficients()``. Sums to 1.0 (or is all-zero if
        the ensemble has no trees/splits)."""
        totals = [0.0] * len(self.feature_keys)
        for tree in self.trees:
            _sum_gain_by_feature(tree, totals)
        total_gain = sum(totals)
        if total_gain <= 0:
            return [(key, 0.0) for key in self.feature_keys]
        shares = [(key, value / total_gain) for key, value in zip(self.feature_keys, totals)]
        return sorted(shares, key=lambda item: -item[1])


def fit_gradient_boosting(
    feature_keys: list[str],
    features: np.ndarray,
    targets: np.ndarray,
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    min_samples_leaf: int,
    min_samples_split: int,
    l2_lambda: float,
) -> GradientBoostedRegressor:
    """Fit a deterministic gradient-boosted regression ensemble via repeated
    residual-fitting (the squared-error-loss special case of gradient boosting,
    where the functional gradient is just the residual itself)."""
    if features.ndim != 2 or features.shape[0] != targets.shape[0]:
        raise ValueError("features must be a 2D array with one row per target")
    if features.shape[0] == 0:
        raise ValueError("Cannot fit a model with zero training rows")

    initial_value = float(targets.mean())
    predictions = np.full(targets.shape[0], initial_value, dtype=float)
    trees: list[_TreeNode] = []
    for _ in range(n_estimators):
        residuals = targets - predictions
        tree = _build_tree(
            features, residuals, depth=0, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
            l2_lambda=l2_lambda,
        )
        trees.append(tree)
        predictions = predictions + learning_rate * _predict_tree_batch(tree, features)

    return GradientBoostedRegressor(
        feature_keys=list(feature_keys),
        initial_value=initial_value,
        learning_rate=learning_rate,
        trees=trees,
    )


def evaluate_regression_holdout(
    model: GradientBoostedRegressor, features: np.ndarray, targets: np.ndarray
) -> dict[str, float | int]:
    """Mean squared error and mean absolute error on held-out rows."""
    if features.shape[0] == 0:
        return {"mse": float("nan"), "mean_absolute_error": float("nan"), "sample_count": 0}
    predictions = model.predict_batch(features)
    mse = float(np.mean((predictions - targets) ** 2))
    mean_absolute_error = float(np.mean(np.abs(predictions - targets)))
    return {"mse": mse, "mean_absolute_error": mean_absolute_error, "sample_count": int(features.shape[0])}
