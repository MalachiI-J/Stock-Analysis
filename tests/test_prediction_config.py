from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from stock_scrapper.prediction.config import validate_prediction_config


def _valid_rules(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "prediction_version": "predict-v3",
        "horizon_days": 21,
        "lookback_years": 5,
        "sample_stride_sessions": 2,
        "walk_forward_folds": 5,
        "l2_lambda": 0.01,
        "learning_rate": 0.3,
        "iterations": 500,
        "minimum_training_samples": 200,
        "feature_keys": ["rsi_14"],
        "gbm": {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
            "min_samples_leaf": 50,
            "min_samples_split": 100,
            "l2_lambda": 1.0,
        },
    }
    base.update(overrides)
    return base


def test_validate_prediction_config_accepts_valid_rules() -> None:
    result = validate_prediction_config(_valid_rules())

    assert result["prediction_version"] == "predict-v3"
    assert result["horizon_days"] == 21
    assert result["gbm"] == {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "min_samples_leaf": 50,
        "min_samples_split": 100,
        "l2_lambda": 1.0,
    }


def test_validate_prediction_config_rejects_missing_gbm_section() -> None:
    rules = _valid_rules()
    del rules["gbm"]

    with pytest.raises(ValueError, match="gbm must be a mapping"):
        validate_prediction_config(rules)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("n_estimators", 0),
        ("n_estimators", -1),
        ("n_estimators", 1.5),
        ("max_depth", 0),
        ("learning_rate", 0.0),
        ("learning_rate", -0.1),
        ("min_samples_leaf", 0),
        ("min_samples_split", 0),
    ],
)
def test_validate_prediction_config_rejects_invalid_gbm_values(key: str, value: object) -> None:
    rules = _valid_rules()
    rules["gbm"] = {**rules["gbm"], key: value}  # type: ignore[dict-item]

    with pytest.raises(ValueError, match=f"gbm.{key}"):
        validate_prediction_config(rules)


def test_validate_prediction_config_rejects_negative_gbm_l2_lambda() -> None:
    rules = _valid_rules()
    rules["gbm"] = {**rules["gbm"], "l2_lambda": -1.0}  # type: ignore[dict-item]

    with pytest.raises(ValueError, match="gbm.l2_lambda"):
        validate_prediction_config(rules)


def test_validate_prediction_config_accepts_zero_gbm_l2_lambda() -> None:
    rules = _valid_rules()
    rules["gbm"] = {**rules["gbm"], "l2_lambda": 0.0}  # type: ignore[dict-item]

    result = validate_prediction_config(rules)

    assert result["gbm"]["l2_lambda"] == 0.0


def test_validate_prediction_config_rejects_non_dict_input() -> None:
    with pytest.raises(ValueError, match="must contain a mapping"):
        validate_prediction_config([])  # type: ignore[arg-type]


def test_real_prediction_rules_yaml_passes_validation() -> None:
    """Loads and validates the actual config/prediction_rules.yaml shipped with the
    project — catches exactly the kind of schema mismatch (a required section missing
    from the real file) that a purely synthetic-fixture test would miss."""
    repo_root = Path(__file__).resolve().parent.parent
    with (repo_root / "config" / "prediction_rules.yaml").open("r", encoding="utf-8") as handle:
        rules = yaml.safe_load(handle)

    result = validate_prediction_config(rules)

    assert result["gbm"]["n_estimators"] > 0
    assert result["feature_keys"]
