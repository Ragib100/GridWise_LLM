"""
Pydantic request/response models for GridWise LLM.

Schema is taken verbatim from the official Problem Statement + the public
sample case pack (BUP CSE Fest 2026 GridWise LLM preliminary). Do not change
field names/shapes without re-checking those documents.
"""
from typing import Any, Dict, List, Literal, Optional, Type

from pydantic import BaseModel, Field, field_validator, model_validator

DirectiveType = Literal[
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]

BatteryAction = Literal["charge", "discharge", "idle"]


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class HourEntry(BaseModel):
    hour: int = Field(ge=0, le=23)
    demand_kwh: float = Field(ge=0)
    solar_kwh: float = Field(ge=0)
    tariff_bdt_per_kwh: float = Field(ge=0)


class Battery(BaseModel):
    capacity_kwh: float = Field(gt=0)
    initial_energy_kwh: float = Field(ge=0)
    minimum_energy_kwh: float = Field(ge=0)
    max_charge_kwh_per_hour: float = Field(ge=0)
    max_discharge_kwh_per_hour: float = Field(ge=0)

    @model_validator(mode="after")
    def _check_bounds(self):
        if self.minimum_energy_kwh > self.capacity_kwh:
            raise ValueError("minimum_energy_kwh cannot exceed capacity_kwh")
        if self.initial_energy_kwh > self.capacity_kwh:
            raise ValueError("initial_energy_kwh cannot exceed capacity_kwh")
        return self


class OptimizeRequest(BaseModel):
    scenario_id: str
    operator_notes: List[str] = Field(min_length=1, max_length=3)
    hours: List[HourEntry] = Field(min_length=24, max_length=24)
    battery: Battery

    @field_validator("operator_notes")
    @classmethod
    def _notes_nonempty(cls, v: List[str]) -> List[str]:
        for note in v:
            if not isinstance(note, str) or not note.strip():
                raise ValueError("operator_notes entries must be non-empty strings")
        return v

    @model_validator(mode="after")
    def _check_hours(self):
        hour_values = sorted(h.hour for h in self.hours)
        if hour_values != list(range(24)):
            raise ValueError(
                "hours must contain exactly one entry for each hour from 0 to 23"
            )
        # normalize ordering so index == hour everywhere downstream
        self.hours = sorted(self.hours, key=lambda h: h.hour)
        return self


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class DirectiveInterpretation(BaseModel):
    note_index: int
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: Optional[dict] = None
    explanation: str


class HourlyPlanEntry(BaseModel):
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: BatteryAction
    battery_kwh: float
    battery_energy_after_kwh: float


class OptimizeResponse(BaseModel):
    scenario_id: str
    directive_interpretation: List[DirectiveInterpretation]
    hourly_plan: List[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str


# ---------------------------------------------------------------------------
# Directive adjustment models -- strict per-type validation
# ---------------------------------------------------------------------------
#
# The wire schema (fixed by the official spec/sample cases) keeps
# `directive_type` as a sibling of `structured_adjustment`, not a field
# nested inside it -- so a real Pydantic `Discriminator`-based union isn't
# applicable to the wire shape itself. Instead we get the same guarantee
# ("exactly one strict, type-specific shape per directive_type, chosen by
# that type") via manual discriminated dispatch: `validator.py` looks up
# the directive_type in DIRECTIVE_ADJUSTMENT_MODELS and constructs *that*
# model, which is exactly as strict as a discriminated union would be.
#
# These models are never touched by `interpret_notes()` (the LLM layer) --
# only `validator.py`'s deterministic guardrail calls them, on raw/untrusted
# LLM output, so a bad value (wrong type, out of range, duplicate hour,
# a bool where a number is required) raises `pydantic.ValidationError`
# rather than silently reaching the optimizer.


def _reject_bool(v: Any) -> Any:
    if isinstance(v, bool):
        raise ValueError("must be a number, not a boolean")
    return v


def _normalize_hours(v: Any) -> List[int]:
    if not isinstance(v, list) or len(v) == 0:
        raise ValueError("hours must be a non-empty list")
    hours: List[int] = []
    for h in v:
        if isinstance(h, bool):
            raise ValueError("hours must be integers, not booleans")
        if isinstance(h, int):
            hours.append(h)
        elif isinstance(h, float) and h.is_integer():
            hours.append(int(h))
        else:
            raise ValueError("hours must be integers")
    if any(h < 0 or h > 23 for h in hours):
        raise ValueError("hours must be between 0 and 23")
    if len(set(hours)) != len(hours):
        raise ValueError("hours must not contain duplicate values")
    # Sorted, not rejected, if out of order: an overnight window like
    # "11 PM to 2 AM" may naturally come back as [23, 0, 1]; that is a
    # fully valid, unambiguous set of hours and the API's required output
    # order is ascending regardless of what order the model produced.
    return sorted(hours)


class _HoursAdjustmentBase(BaseModel):
    """Shared `hours` field/validation for every windowed directive type."""

    hours: List[int]

    @field_validator("hours", mode="before")
    @classmethod
    def _validate_hours(cls, v: Any) -> List[int]:
        return _normalize_hours(v)


class SolarReductionAdjustment(_HoursAdjustmentBase):
    factor: float = Field(ge=0.0, le=1.0)

    @field_validator("factor", mode="before")
    @classmethod
    def _factor_not_bool(cls, v: Any) -> Any:
        return _reject_bool(v)


class MinimumBatteryReserveAdjustment(_HoursAdjustmentBase):
    minimum_energy_kwh: float = Field(ge=0.0)

    @field_validator("minimum_energy_kwh", mode="before")
    @classmethod
    def _value_not_bool(cls, v: Any) -> Any:
        return _reject_bool(v)


class NoChargeWindowAdjustment(_HoursAdjustmentBase):
    pass


class NoDischargeWindowAdjustment(_HoursAdjustmentBase):
    pass


class MaxGridWindowAdjustment(_HoursAdjustmentBase):
    max_grid_kwh: float = Field(ge=0.0)

    @field_validator("max_grid_kwh", mode="before")
    @classmethod
    def _value_not_bool(cls, v: Any) -> Any:
        return _reject_bool(v)


# directive_type (string) -> the strict model that validates its
# structured_adjustment shape. "no_op" carries no adjustment and is
# handled separately in validator.py, so it has no entry here.
DIRECTIVE_ADJUSTMENT_MODELS: Dict[str, Type[BaseModel]] = {
    "solar_reduction": SolarReductionAdjustment,
    "minimum_battery_reserve": MinimumBatteryReserveAdjustment,
    "no_charge_window": NoChargeWindowAdjustment,
    "no_discharge_window": NoDischargeWindowAdjustment,
    "max_grid_window": MaxGridWindowAdjustment,
}
