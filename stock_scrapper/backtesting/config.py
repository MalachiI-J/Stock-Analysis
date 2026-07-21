"""Validated, reproducible configuration for Phase 3 backtests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

import yaml


ALLOWED_MARKET_REGIMES = frozenset(
    {"Risk-On", "Neutral", "Risk-Off", "Stress", "Insufficient Market Data"}
)
ALLOWED_CLASSIFICATIONS = frozenset(
    {"Data Blocked", "Insufficient Data", "High Risk", "Avoid", "Watch", "Candidate", "Strong Candidate"}
)
ALLOWED_FREQUENCIES = frozenset({"daily", "weekly", "monthly"})
ALLOWED_POSITION_SIZING = frozenset({"equal_weight", "volatility_adjusted"})
ALLOWED_EXECUTION_TIMINGS = frozenset({"next_open"})
ALLOWED_FINAL_LIQUIDATION_TIMINGS = frozenset({"final_close"})
ALLOWED_AMBIGUITY_POLICIES = frozenset({"adverse_first", "favorable_first", "skip_bar"})
ALLOWED_VOLATILITY_LOOKBACK_DAYS = frozenset({20, 60, 252})


@dataclass(frozen=True, slots=True)
class EntryThresholds:
    """Eligibility thresholds applied before a candidate can be ranked."""

    classifications: tuple[str, ...]
    minimum_opportunity_score: float
    minimum_average_dollar_volume: float


@dataclass(frozen=True, slots=True)
class ExitThresholds:
    """Score and state thresholds that can close an existing position."""

    classifications: tuple[str, ...]
    minimum_opportunity_score: float
    minimum_confidence_score: float
    maximum_risk_score: float
    exit_below_sma200: bool
    exit_on_stress: bool


@dataclass(frozen=True, slots=True)
class FinalLiquidationRules:
    """Rules for closing positions at the configured end of a simulation."""

    enabled: bool
    timing: str
    apply_costs: bool


@dataclass(frozen=True, slots=True)
class WalkForwardRules:
    """Fixed windows used to evaluate consistency without optimization."""

    warm_up_days: int
    development_days: int
    validation_days: int
    final_holdout_days: int
    step_days: int


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Complete immutable configuration for a deterministic backtest."""

    strategy_name: str
    strategy_version: str
    benchmark: str
    initial_cash: float
    warm_up_days: int
    start_date: date | None
    end_date: date | None
    signal_frequency: str
    rebalancing_frequency: str
    entry_thresholds: EntryThresholds
    exit_thresholds: ExitThresholds
    allowed_market_regimes: tuple[str, ...]
    minimum_confidence: float
    maximum_risk: float
    maximum_positions: int
    maximum_position_weight: float
    cash_reserve: float
    fractional_shares: bool
    position_sizing: str
    volatility_lookback_days: int
    commission_basis_points: float
    minimum_commission: float
    slippage_basis_points: float
    stop_loss: float | None
    trailing_stop: float | None
    profit_target: float | None
    maximum_holding_period: int | None
    execution_timing: str
    final_liquidation: FinalLiquidationRules
    risk_free_rate: float
    annualization_factor: int
    daily_bar_ambiguity_policy: str
    walk_forward: WalkForwardRules

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible configuration snapshot."""
        payload = asdict(self)
        payload["start_date"] = self.start_date.isoformat() if self.start_date else None
        payload["end_date"] = self.end_date.isoformat() if self.end_date else None
        return payload

    @property
    def configuration_hash(self) -> str:
        """Return a stable SHA-256 hash of the canonical configuration."""
        return configuration_hash(self)

    @property
    def config_hash(self) -> str:
        """Compatibility alias for :attr:`configuration_hash`."""
        return self.configuration_hash

    def with_overrides(self, **overrides: Any) -> BacktestConfig:
        """Return a revalidated copy with top-level CLI overrides applied."""
        unknown = set(overrides) - set(self.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown backtesting override(s): {', '.join(sorted(unknown))}")
        payload = self.to_dict()
        payload.update(overrides)
        return validate_backtesting_config(payload)

    def get(self, key: str, default: Any = None) -> Any:
        """Offer a small mapping-style convenience for integration code."""
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


_TOP_LEVEL_KEYS = frozenset(BacktestConfig.__dataclass_fields__)
_ENTRY_KEYS = frozenset(EntryThresholds.__dataclass_fields__)
_EXIT_KEYS = frozenset(ExitThresholds.__dataclass_fields__)
_FINAL_LIQUIDATION_KEYS = frozenset(FinalLiquidationRules.__dataclass_fields__)
_WALK_FORWARD_KEYS = frozenset(WalkForwardRules.__dataclass_fields__)


def _require_exact_keys(payload: Mapping[str, Any], expected: frozenset[str], location: str) -> None:
    missing = expected - set(payload)
    unknown = set(payload) - expected
    messages: list[str] = []
    if missing:
        messages.append(f"missing {', '.join(sorted(missing))}")
    if unknown:
        messages.append(f"unknown {', '.join(sorted(unknown))}")
    if messages:
        raise ValueError(f"Invalid {location}: {'; '.join(messages)}")


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _boolean(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _number(value: Any, field_name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field_name} must be at most {maximum}")
    return result


def _positive_int(value: Any, field_name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")
    return value


def _optional_fraction(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    result = _number(value, field_name)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{field_name} must be greater than 0 and less than 1")
    return result


def _optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field_name)


def _date(value: Any, field_name: str) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must use YYYY-MM-DD format") from exc
    raise ValueError(f"{field_name} must be a date, YYYY-MM-DD string, or null")


def _choice(value: Any, field_name: str, choices: frozenset[str]) -> str:
    text = _text(value, field_name)
    if text not in choices:
        raise ValueError(f"{field_name} must be one of: {', '.join(sorted(choices))}")
    return text


def _string_tuple(value: Any, field_name: str, allowed: frozenset[str]) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field_name} must be a non-empty list")
    values = tuple(_text(item, field_name) for item in value)
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"{field_name} contains unknown values: {', '.join(sorted(unknown))}")
    return values


def validate_backtesting_config(payload: Mapping[str, Any]) -> BacktestConfig:
    """Validate a raw configuration mapping and return its typed form."""
    if not isinstance(payload, Mapping):
        raise ValueError("Backtesting configuration must be a mapping")
    _require_exact_keys(payload, _TOP_LEVEL_KEYS, "backtesting configuration")

    entry_raw = _mapping(payload["entry_thresholds"], "entry_thresholds")
    _require_exact_keys(entry_raw, _ENTRY_KEYS, "entry_thresholds")
    entry = EntryThresholds(
        classifications=_string_tuple(entry_raw["classifications"], "entry_thresholds.classifications", ALLOWED_CLASSIFICATIONS),
        minimum_opportunity_score=_number(
            entry_raw["minimum_opportunity_score"], "entry_thresholds.minimum_opportunity_score", minimum=0, maximum=100
        ),
        minimum_average_dollar_volume=_number(
            entry_raw["minimum_average_dollar_volume"], "entry_thresholds.minimum_average_dollar_volume", minimum=0
        ),
    )

    exit_raw = _mapping(payload["exit_thresholds"], "exit_thresholds")
    _require_exact_keys(exit_raw, _EXIT_KEYS, "exit_thresholds")
    exit_rules = ExitThresholds(
        classifications=_string_tuple(exit_raw["classifications"], "exit_thresholds.classifications", ALLOWED_CLASSIFICATIONS),
        minimum_opportunity_score=_number(
            exit_raw["minimum_opportunity_score"], "exit_thresholds.minimum_opportunity_score", minimum=0, maximum=100
        ),
        minimum_confidence_score=_number(
            exit_raw["minimum_confidence_score"], "exit_thresholds.minimum_confidence_score", minimum=0, maximum=100
        ),
        maximum_risk_score=_number(
            exit_raw["maximum_risk_score"], "exit_thresholds.maximum_risk_score", minimum=0, maximum=100
        ),
        exit_below_sma200=_boolean(exit_raw["exit_below_sma200"], "exit_thresholds.exit_below_sma200"),
        exit_on_stress=_boolean(exit_raw["exit_on_stress"], "exit_thresholds.exit_on_stress"),
    )
    if entry.minimum_opportunity_score < exit_rules.minimum_opportunity_score:
        raise ValueError("Entry opportunity threshold must not be below the exit opportunity threshold")

    final_raw = _mapping(payload["final_liquidation"], "final_liquidation")
    _require_exact_keys(final_raw, _FINAL_LIQUIDATION_KEYS, "final_liquidation")
    final_liquidation = FinalLiquidationRules(
        enabled=_boolean(final_raw["enabled"], "final_liquidation.enabled"),
        timing=_choice(final_raw["timing"], "final_liquidation.timing", ALLOWED_FINAL_LIQUIDATION_TIMINGS),
        apply_costs=_boolean(final_raw["apply_costs"], "final_liquidation.apply_costs"),
    )

    walk_raw = _mapping(payload["walk_forward"], "walk_forward")
    _require_exact_keys(walk_raw, _WALK_FORWARD_KEYS, "walk_forward")
    walk_forward = WalkForwardRules(
        warm_up_days=_positive_int(walk_raw["warm_up_days"], "walk_forward.warm_up_days"),
        development_days=_positive_int(walk_raw["development_days"], "walk_forward.development_days"),
        validation_days=_positive_int(walk_raw["validation_days"], "walk_forward.validation_days"),
        final_holdout_days=_positive_int(walk_raw["final_holdout_days"], "walk_forward.final_holdout_days"),
        step_days=_positive_int(walk_raw["step_days"], "walk_forward.step_days"),
    )

    start = _date(payload["start_date"], "start_date")
    end = _date(payload["end_date"], "end_date")
    if start is not None and end is not None and start > end:
        raise ValueError("start_date must be on or before end_date")

    maximum_position_weight = _number(payload["maximum_position_weight"], "maximum_position_weight")
    if not 0.0 < maximum_position_weight <= 1.0:
        raise ValueError("maximum_position_weight must be greater than 0 and at most 1")
    cash_reserve = _number(payload["cash_reserve"], "cash_reserve", minimum=0)
    if cash_reserve >= 1.0:
        raise ValueError("cash_reserve must be less than 1")
    if maximum_position_weight > 1.0 - cash_reserve:
        raise ValueError("maximum_position_weight cannot exceed the investable portfolio fraction")

    risk_free_rate = _number(payload["risk_free_rate"], "risk_free_rate")
    if risk_free_rate <= -1.0:
        raise ValueError("risk_free_rate must be greater than -1")
    volatility_lookback_days = _positive_int(
        payload["volatility_lookback_days"], "volatility_lookback_days"
    )
    if volatility_lookback_days not in ALLOWED_VOLATILITY_LOOKBACK_DAYS:
        allowed = ", ".join(str(value) for value in sorted(ALLOWED_VOLATILITY_LOOKBACK_DAYS))
        raise ValueError(f"volatility_lookback_days must be one of: {allowed}")

    config = BacktestConfig(
        strategy_name=_text(payload["strategy_name"], "strategy_name"),
        strategy_version=_text(payload["strategy_version"], "strategy_version"),
        benchmark=_text(payload["benchmark"], "benchmark").upper(),
        initial_cash=_number(payload["initial_cash"], "initial_cash", minimum=0.01),
        warm_up_days=_positive_int(payload["warm_up_days"], "warm_up_days"),
        start_date=start,
        end_date=end,
        signal_frequency=_choice(payload["signal_frequency"], "signal_frequency", ALLOWED_FREQUENCIES),
        rebalancing_frequency=_choice(payload["rebalancing_frequency"], "rebalancing_frequency", ALLOWED_FREQUENCIES),
        entry_thresholds=entry,
        exit_thresholds=exit_rules,
        allowed_market_regimes=_string_tuple(payload["allowed_market_regimes"], "allowed_market_regimes", ALLOWED_MARKET_REGIMES),
        minimum_confidence=_number(payload["minimum_confidence"], "minimum_confidence", minimum=0, maximum=100),
        maximum_risk=_number(payload["maximum_risk"], "maximum_risk", minimum=0, maximum=100),
        maximum_positions=_positive_int(payload["maximum_positions"], "maximum_positions"),
        maximum_position_weight=maximum_position_weight,
        cash_reserve=cash_reserve,
        fractional_shares=_boolean(payload["fractional_shares"], "fractional_shares"),
        position_sizing=_choice(payload["position_sizing"], "position_sizing", ALLOWED_POSITION_SIZING),
        volatility_lookback_days=volatility_lookback_days,
        commission_basis_points=_number(payload["commission_basis_points"], "commission_basis_points", minimum=0),
        minimum_commission=_number(payload["minimum_commission"], "minimum_commission", minimum=0),
        slippage_basis_points=_number(payload["slippage_basis_points"], "slippage_basis_points", minimum=0),
        stop_loss=_optional_fraction(payload["stop_loss"], "stop_loss"),
        trailing_stop=_optional_fraction(payload["trailing_stop"], "trailing_stop"),
        profit_target=(
            None
            if payload["profit_target"] is None
            else _number(payload["profit_target"], "profit_target", minimum=0.000000000001)
        ),
        maximum_holding_period=_optional_positive_int(payload["maximum_holding_period"], "maximum_holding_period"),
        execution_timing=_choice(payload["execution_timing"], "execution_timing", ALLOWED_EXECUTION_TIMINGS),
        final_liquidation=final_liquidation,
        risk_free_rate=risk_free_rate,
        annualization_factor=_positive_int(payload["annualization_factor"], "annualization_factor"),
        daily_bar_ambiguity_policy=_choice(
            payload["daily_bar_ambiguity_policy"], "daily_bar_ambiguity_policy", ALLOWED_AMBIGUITY_POLICIES
        ),
        walk_forward=walk_forward,
    )
    return config


def canonical_config_json(config: BacktestConfig | Mapping[str, Any]) -> str:
    """Serialize validated configuration using stable JSON ordering."""
    typed = config if isinstance(config, BacktestConfig) else validate_backtesting_config(config)
    return json.dumps(typed.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def configuration_hash(config: BacktestConfig | Mapping[str, Any]) -> str:
    """Calculate the cross-process-stable SHA-256 configuration hash."""
    return hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()


def load_backtesting_config(path: str | Path | None = None) -> BacktestConfig:
    """Load and validate ``config/backtesting_rules.yaml``."""
    if path is None:
        config_path = Path(__file__).resolve().parents[2] / "config" / "backtesting_rules.yaml"
    else:
        config_path = Path(path)
        if config_path.is_dir():
            direct = config_path / "backtesting_rules.yaml"
            config_path = direct if direct.exists() else config_path / "config" / "backtesting_rules.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Backtesting configuration does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("Backtesting configuration must contain a YAML mapping")
    return validate_backtesting_config(payload)


# Concise aliases for integration code and CLI handlers.
load_backtest_config = load_backtesting_config
validate_backtest_config = validate_backtesting_config
