"""
Deterministic guardrails for LLM output.

Nothing here calls an LLM. This module treats the LLM's output as
completely untrusted, malformed-JSON-capable input and converts it into a
list of `DirectiveInterpretation`-shaped dicts that are guaranteed to:

  - contain exactly one entry per operator note, in note_index order,
  - use an allowed directive_type,
  - carry a structured_adjustment that exactly matches the required shape
    for that directive_type (or null for no_op),
  - never contain out-of-range hours, out-of-range numbers, or impossible
    values.

Any directive that fails a check is safely downgraded to no_op rather than
rejected outright -- a bad LLM response must never crash the service or
silently reach the optimizer.
"""
from typing import Any, Dict, List

from pydantic import ValidationError

from .models import DIRECTIVE_ADJUSTMENT_MODELS

ALLOWED_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}

# directive_type -> the raw-payload keys (besides "hours") that get pulled
# out of the untrusted LLM item and handed to that type's strict Pydantic
# model in models.DIRECTIVE_ADJUSTMENT_MODELS for validation.
_EXTRA_FIELDS_BY_TYPE = {
    "solar_reduction": ("factor",),
    "minimum_battery_reserve": ("minimum_energy_kwh",),
    "no_charge_window": (),
    "no_discharge_window": (),
    "max_grid_window": ("max_grid_kwh",),
}


def _safe_no_op(note_index: int, reason: str) -> Dict[str, Any]:
    return {
        "note_index": note_index,
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": reason,
    }


def _validate_single(item: Dict[str, Any], battery_capacity_kwh: float) -> Dict[str, Any]:
    idx = item["note_index"]
    dtype = item.get("directive_type")
    explanation = item.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        explanation = "No explanation was provided by the model."

    if dtype not in ALLOWED_TYPES:
        return _safe_no_op(
            idx, "Unsupported or missing directive type in model output; treated as no_op for safety."
        )

    if dtype == "no_op":
        return _safe_no_op(idx, explanation)

    # Every other directive type has a strict Pydantic model (manual
    # discriminated dispatch keyed by directive_type -- see models.py for
    # why this isn't a nested-discriminator union). Constructing that model
    # from the raw, untrusted LLM fields enforces every substantive
    # guarantee (integer/range/uniqueness on hours, numeric range on the
    # type-specific value, no booleans coerced into numbers) in one place;
    # any violation raises ValidationError and we safely downgrade to no_op.
    model_cls = DIRECTIVE_ADJUSTMENT_MODELS[dtype]
    raw_payload = {"hours": item.get("hours")}
    for field in _EXTRA_FIELDS_BY_TYPE[dtype]:
        raw_payload[field] = item.get(field)

    try:
        adjustment = model_cls.model_validate(raw_payload)
    except ValidationError:
        return _safe_no_op(
            idx,
            f"{dtype} adjustment was missing, malformed, or out of range; treated as no_op for safety.",
        )

    # battery_capacity_kwh is request-level context the per-type model has
    # no way to know, so this bound is checked here rather than in models.py.
    if dtype == "minimum_battery_reserve" and adjustment.minimum_energy_kwh > battery_capacity_kwh:
        return _safe_no_op(
            idx, "Minimum battery reserve value exceeds battery capacity; treated as no_op for safety."
        )

    return {
        "note_index": idx,
        "applies": True,
        "directive_type": dtype,
        "structured_adjustment": adjustment.model_dump(),
        "explanation": explanation,
    }


def validate_directives(
    raw_directives: Any, num_notes: int, battery_capacity_kwh: float
) -> List[Dict[str, Any]]:
    """
    Deterministically validate raw (untrusted) LLM output.

    Guarantees on the return value:
      - exactly `num_notes` entries,
      - entries are ordered by note_index 0..num_notes-1 with no duplicates
        or gaps (missing/duplicate/out-of-range mappings are filled with a
        safe no_op),
      - every entry matches the exact structured_adjustment shape required
        by its directive_type.

    Never raises: malformed input at any level degrades to no_op instead of
    propagating an exception, so a broken LLM response can never crash the
    request or reach the optimizer unchecked.
    """
    by_index: Dict[int, Dict[str, Any]] = {}

    if isinstance(raw_directives, list):
        for item in raw_directives:
            try:
                if not isinstance(item, dict):
                    continue
                idx = item.get("note_index")
                if isinstance(idx, bool) or not isinstance(idx, int):
                    continue
                if idx < 0 or idx >= num_notes:
                    continue
                if idx in by_index:
                    # Duplicate mapping for the same note: keep the first
                    # valid one, ignore the rest (never let a later
                    # duplicate silently overwrite an already-accepted
                    # directive).
                    continue
                by_index[idx] = _validate_single(item, battery_capacity_kwh)
            except Exception:  # noqa: BLE001 - guardrail must never crash
                continue

    result: List[Dict[str, Any]] = []
    for i in range(num_notes):
        if i in by_index:
            result.append(by_index[i])
        else:
            result.append(
                _safe_no_op(
                    i,
                    "No valid interpretation was returned for this note; treated as no_op for safety.",
                )
            )
    return result
