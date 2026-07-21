"""Semantic provider-row comparison that separates precision noise from revisions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

DEFAULT_REVISION_POLICY = {
    "version": "revision-v2", "price_absolute_tolerance": 0.0001,
    "price_relative_tolerance": 0.000001, "adjusted_price_absolute_tolerance": 0.0001,
    "adjusted_price_relative_tolerance": 0.000001, "dividend_absolute_tolerance": 1e-8,
    "split_absolute_tolerance": 1e-8, "volume_tolerance": 0, "store_precision_noise": False,
}
PRICE_FIELDS = ("open", "high", "low", "close")


@dataclass(frozen=True, slots=True)
class PriceRowComparison:
    exact_differences: tuple[str, ...]
    material_differences: tuple[str, ...]
    precision_only_differences: tuple[str, ...]
    absolute_deltas: dict[str, float | None]
    relative_deltas: dict[str, float | None]
    revision_class: str
    analysis_critical: bool


def _number(value: Any) -> float | None:
    try: result = float(value)
    except (TypeError, ValueError): return None
    return result if math.isfinite(result) else None


def compare_price_rows(previous: Mapping[str, Any], incoming: Mapping[str, Any], policy: Mapping[str, Any] | None = None) -> PriceRowComparison:
    rules = {**DEFAULT_REVISION_POLICY, **(policy or {})}
    fields = (*PRICE_FIELDS, "adjusted_close", "volume", "dividends", "stock_splits", "data_source")
    exact, material, precision = [], [], []
    absolute: dict[str, float | None] = {}; relative: dict[str, float | None] = {}
    for field in fields:
        old, new = previous.get(field), incoming.get(field)
        if old == new or (_number(old) is None and _number(new) is None and field != "data_source"): continue
        exact.append(field)
        if field == "data_source": material.append(field); absolute[field] = relative[field] = None; continue
        old_n, new_n = _number(old), _number(new)
        if old_n is None or new_n is None:
            material.append(field); absolute[field] = relative[field] = None; continue
        delta = abs(new_n-old_n); denominator=max(abs(old_n),abs(new_n),1e-300)
        absolute[field], relative[field] = delta, delta/denominator
        if field == "volume": is_material = delta > float(rules["volume_tolerance"])
        elif field == "adjusted_close": is_material = delta > float(rules["adjusted_price_absolute_tolerance"]) and delta/denominator > float(rules["adjusted_price_relative_tolerance"])
        elif field == "dividends": is_material = delta > float(rules["dividend_absolute_tolerance"])
        elif field == "stock_splits": is_material = delta > float(rules["split_absolute_tolerance"])
        else: is_material = delta > float(rules["price_absolute_tolerance"]) and delta/denominator > float(rules["price_relative_tolerance"])
        (material if is_material else precision).append(field)
    if not exact: classification = "status_repair"
    elif material and any(f in material for f in ("dividends","stock_splits")): classification = "corporate_action_revision"
    elif material: classification = "material_price_revision"
    else: classification = "precision_noise"
    return PriceRowComparison(tuple(exact),tuple(material),tuple(precision),absolute,relative,classification,bool(set(material)&set((*PRICE_FIELDS,"adjusted_close","volume","dividends","stock_splits"))))
