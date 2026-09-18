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
from typing import Any, Dict, List, Optional

ALLOWED_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}


def _safe_no_op(note_index: int, reason: str) -> Dict[str, Any]:
    return {
        "note_index": note_index,
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": reason,
    }


def _validate_hours(raw_hours: Any) -> Optional[List[int]]:
    if not isinstance(raw_hours, list) or len(raw_hours) == 0:
        return None
    hours: List[int] = []
    for h in raw_hours:
        if isinstance(h, bool):
            return None
        if isinstance(h, int):
            hours.append(h)
        elif isinstance(h, float) and h.is_integer():
            hours.append(int(h))
        else:
            return None
    if any(h < 0 or h > 23 for h in hours):
        return None
    if len(set(hours)) != len(hours):
        return None
    if hours != sorted(hours):
        return None
    return hours


def _validate_number(raw: Any) -> Optional[float]:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


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

    hours = _validate_hours(item.get("hours"))
    if hours is None:
        return _safe_no_op(
            idx, "Directive hours were missing, malformed, or out of range; treated as no_op for safety."
        )

    if dtype == "solar_reduction":
        factor = _validate_number(item.get("factor"))
        if factor is None or not (0.0 <= factor <= 1.0):
            return _safe_no_op(
                idx, "Solar reduction factor was missing or out of the 0-1 range; treated as no_op for safety."
            )
        return {
            "note_index": idx,
            "applies": True,
            "directive_type": dtype,
            "structured_adjustment": {"hours": hours, "factor": factor},
            "explanation": explanation,
        }

    if dtype == "minimum_battery_reserve":
        value = _validate_number(item.get("minimum_energy_kwh"))
        if value is None or value < 0 or value > battery_capacity_kwh:
            return _safe_no_op(
                idx,
                "Minimum battery reserve value was missing, negative, or above battery capacity; "
                "treated as no_op for safety.",
            )
        return {
            "note_index": idx,
            "applies": True,
            "directive_type": dtype,
            "structured_adjustment": {"hours": hours, "minimum_energy_kwh": value},
            "explanation": explanation,
        }

    if dtype in ("no_charge_window", "no_discharge_window"):
        return {
            "note_index": idx,
            "applies": True,
            "directive_type": dtype,
            "structured_adjustment": {"hours": hours},
            "explanation": explanation,
        }

    if dtype == "max_grid_window":
        value = _validate_number(item.get("max_grid_kwh"))
        if value is None or value < 0:
            return _safe_no_op(
                idx, "Max grid import value was missing or negative; treated as no_op for safety."
            )
        return {
            "note_index": idx,
            "applies": True,
            "directive_type": dtype,
            "structured_adjustment": {"hours": hours, "max_grid_kwh": value},
            "explanation": explanation,
        }

    # Unreachable given ALLOWED_TYPES, kept as a last-resort safe fallback.
    return _safe_no_op(idx, "Unhandled directive type; treated as no_op for safety.")


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
