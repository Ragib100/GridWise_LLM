"""
Pydantic request/response models for GridWise LLM.

Schema is taken verbatim from the official Problem Statement + the public
sample case pack (BUP CSE Fest 2026 GridWise LLM preliminary). Do not change
field names/shapes without re-checking those documents.
"""
from typing import List, Literal, Optional

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
