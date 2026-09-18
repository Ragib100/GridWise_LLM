"""
GridWise LLM API.

Pipeline:  operator notes -> LLM interpreter -> deterministic guardrails
           -> LP optimizer -> defensive final replay (logged) -> response.

Endpoints (names/shapes are fixed by the official Problem Statement):
    GET  /health
    POST /optimize-energy
"""
import logging
import os

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - optional, only used for local dev
    pass

from . import optimizer
from .llm import LLMError, interpret_notes
from .models import OptimizeRequest, OptimizeResponse
from .validator import validate_directives

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("gridwise")

app = FastAPI(title="GridWise LLM Optimizer", version="1.0.0")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # The Problem Statement calls for 400 on malformed/structurally invalid
    # requests; FastAPI's default for pydantic validation failures is 422,
    # so we normalize to 400 here. No secrets/stack traces are included.
    return JSONResponse(
        status_code=400,
        content={"detail": "Malformed or structurally invalid request.", "errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error while processing %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/health")
def health():
    return {"status": "ok"}


def _build_summary(directives, total_cost_bdt: float, total_grid_kwh: float) -> str:
    applied = [d for d in directives if d.get("applies")]
    if not applied:
        return (
            f"No operator directives applied; grid+solar+battery scheduled for "
            f"total cost {total_cost_bdt:.2f} BDT using {total_grid_kwh:.2f} kWh from the grid."
        )
    kinds = ", ".join(sorted({d["directive_type"] for d in applied}))
    return (
        f"Applied {len(applied)} operator directive(s) ({kinds}) and scheduled the battery to "
        f"minimize grid cost: total cost {total_cost_bdt:.2f} BDT using {total_grid_kwh:.2f} kWh from the grid."
    )


@app.post("/optimize-energy", response_model=OptimizeResponse)
def optimize_energy(req: OptimizeRequest):
    hours = req.hours  # already validated + sorted so index == hour
    battery = req.battery

    try:
        raw_directives = interpret_notes(
            operator_notes=req.operator_notes,
            hours=[h.model_dump() for h in hours],
            battery=battery.model_dump(),
        )
    except LLMError as exc:
        logger.warning("LLM interpretation unavailable, falling back to no_op for all notes: %s", exc)
        raw_directives = []  # validator fills in a safe no_op per note below

    directives = validate_directives(raw_directives, len(req.operator_notes), battery.capacity_kwh)

    plan = optimizer.solve(hours, battery, directives)
    if plan is None:
        logger.warning("Optimizer LP was infeasible or failed; using safe fallback plan.")
        plan = optimizer.fallback_plan(hours, battery)

    violations = optimizer.final_replay(plan, hours, battery, directives)
    if violations:
        logger.warning("Final replay found %d issue(s): %s", len(violations), violations)

    total_grid_kwh = round(sum(p["grid_kwh"] for p in plan), 6)
    total_cost_bdt = round(sum(p["grid_kwh"] * hours[p["hour"]].tariff_bdt_per_kwh for p in plan), 6)
    peak_grid_kwh = round(max(p["grid_kwh"] for p in plan), 6)

    return {
        "scenario_id": req.scenario_id,
        "directive_interpretation": directives,
        "hourly_plan": plan,
        "total_grid_kwh": total_grid_kwh,
        "total_cost_bdt": total_cost_bdt,
        "peak_grid_kwh": peak_grid_kwh,
        "plan_summary": _build_summary(directives, total_cost_bdt, total_grid_kwh),
    }
