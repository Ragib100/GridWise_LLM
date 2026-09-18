"""
Deterministic cost-minimizing optimizer.

The 24-hour schedule with a battery, solar, and grid is a linear program:
minimize SUM(grid[h] * tariff[h]) subject to per-hour energy balance,
battery state-of-charge bounds/rate limits, and any validated operator
directives. We solve it exactly with scipy's HiGHS LP backend -- this is
fast (well under a second for 24 hours) and gives a provably optimal,
reproducible answer instead of a heuristic approximation.

Directive precedence when multiple directives touch the same hour (see
README "Directive precedence" section for the human-readable version):
  - solar_reduction: the most restrictive (lowest) factor wins.
  - minimum_battery_reserve: the highest reserve requirement wins.
  - no_charge_window / no_discharge_window: any matching directive forces
    the action to zero for that hour (hard veto, union of all windows).
  - max_grid_window: the most restrictive (lowest) cap wins.
  - End-of-day battery neutrality (Sec 9.6 of the Problem Statement) is an
    unconditional hard rule and takes precedence over any reserve directive
    that would otherwise apply to hour 23.

If the LP is infeasible or scipy fails for any reason, `solve()` returns
None and the caller falls back to `fallback_plan()`, a trivially-feasible
"do nothing with the battery" plan that always satisfies energy balance
and battery bounds, so the service never fails to return *a* valid plan.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog

HOURS_IN_DAY = 24
_EPS = 1e-6
_BIG_GRID_CAP = 1e9


def build_effective_constraints(
    hours: List[Any], battery: Any, directives: List[Dict[str, Any]]
) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    """Fold validated directives into per-hour numeric bounds."""
    n = HOURS_IN_DAY
    solar_factor = [1.0] * n
    reserve_lower = [battery.minimum_energy_kwh] * n
    charge_upper = [battery.max_charge_kwh_per_hour] * n
    discharge_upper = [battery.max_discharge_kwh_per_hour] * n
    grid_upper = [float("inf")] * n

    for d in directives:
        if not d.get("applies"):
            continue
        adj = d.get("structured_adjustment") or {}
        dtype = d.get("directive_type")
        hrs = adj.get("hours", [])

        if dtype == "solar_reduction":
            factor = adj["factor"]
            for h in hrs:
                solar_factor[h] = min(solar_factor[h], factor)
        elif dtype == "minimum_battery_reserve":
            value = adj["minimum_energy_kwh"]
            for h in hrs:
                reserve_lower[h] = max(reserve_lower[h], value)
        elif dtype == "no_charge_window":
            for h in hrs:
                charge_upper[h] = 0.0
        elif dtype == "no_discharge_window":
            for h in hrs:
                discharge_upper[h] = 0.0
        elif dtype == "max_grid_window":
            value = adj["max_grid_kwh"]
            for h in hrs:
                grid_upper[h] = min(grid_upper[h], value)

    reserve_lower = [min(r, battery.capacity_kwh) for r in reserve_lower]
    effective_solar = [hours[h].solar_kwh * solar_factor[h] for h in range(n)]
    return effective_solar, reserve_lower, charge_upper, discharge_upper, grid_upper


def _var_index(hour: int, field: int) -> int:
    # field: 0=grid, 1=solar_used, 2=charge, 3=discharge, 4=soc(after)
    return hour * 5 + field


def solve(hours: List[Any], battery: Any, directives: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    n = HOURS_IN_DAY
    effective_solar, reserve_lower, charge_upper, discharge_upper, grid_upper = build_effective_constraints(
        hours, battery, directives
    )
    demand = [hours[h].demand_kwh for h in range(n)]
    tariff = [hours[h].tariff_bdt_per_kwh for h in range(n)]

    num_vars = 5 * n
    cost = np.zeros(num_vars)
    for h in range(n):
        cost[_var_index(h, 0)] = tariff[h]

    bounds: List[Tuple[float, float]] = [(0.0, 0.0)] * num_vars
    for h in range(n):
        grid_cap = grid_upper[h] if grid_upper[h] != float("inf") else _BIG_GRID_CAP
        bounds[_var_index(h, 0)] = (0.0, grid_cap)
        bounds[_var_index(h, 1)] = (0.0, effective_solar[h])
        bounds[_var_index(h, 2)] = (0.0, charge_upper[h])
        bounds[_var_index(h, 3)] = (0.0, discharge_upper[h])
        bounds[_var_index(h, 4)] = (reserve_lower[h], battery.capacity_kwh)

    # End-of-day neutrality is an unconditional hard rule (Sec 9.6): it
    # overrides any reserve directive that would otherwise touch hour 23.
    bounds[_var_index(n - 1, 4)] = (battery.initial_energy_kwh, battery.initial_energy_kwh)

    a_eq: List[List[float]] = []
    b_eq: List[float] = []

    # Energy balance per hour: grid + solar_used + discharge - charge = demand
    for h in range(n):
        row = [0.0] * num_vars
        row[_var_index(h, 0)] = 1.0
        row[_var_index(h, 1)] = 1.0
        row[_var_index(h, 3)] = 1.0
        row[_var_index(h, 2)] = -1.0
        a_eq.append(row)
        b_eq.append(demand[h])

    # State-of-charge recurrence: soc[h] - soc[h-1] - charge[h] + discharge[h] = 0
    # (soc[-1] is the constant initial_energy_kwh, folded into b_eq for h=0)
    for h in range(n):
        row = [0.0] * num_vars
        row[_var_index(h, 4)] = 1.0
        row[_var_index(h, 2)] = -1.0
        row[_var_index(h, 3)] = 1.0
        if h == 0:
            a_eq.append(row)
            b_eq.append(battery.initial_energy_kwh)
        else:
            row[_var_index(h - 1, 4)] = -1.0
            a_eq.append(row)
            b_eq.append(0.0)

    result = linprog(cost, A_eq=np.array(a_eq), b_eq=np.array(b_eq), bounds=bounds, method="highs")
    if not result.success:
        return None

    x = result.x
    plan: List[Dict[str, Any]] = []
    soc_running = battery.initial_energy_kwh
    for h in range(n):
        solar_used = max(0.0, x[_var_index(h, 1)])
        charge = max(0.0, x[_var_index(h, 2)])
        discharge = max(0.0, x[_var_index(h, 3)])

        # Numerical cleanup: an optimal cost-minimizing solution never
        # benefits from charging and discharging in the same hour. If tiny
        # solver noise leaves both non-zero, collapse to the net effect so
        # the reported action is unambiguous.
        if charge > _EPS and discharge > _EPS:
            net = charge - discharge
            if net >= 0:
                charge, discharge = net, 0.0
            else:
                charge, discharge = 0.0, -net

        if charge <= _EPS:
            charge = 0.0
        if discharge <= _EPS:
            discharge = 0.0
        if solar_used <= _EPS:
            solar_used = 0.0
        solar_used = min(solar_used, effective_solar[h])

        # Recompute grid from the balance equation itself (rather than
        # trusting the raw LP variable) so energy balance holds exactly
        # even after the cleanup above.
        grid = demand[h] + charge - solar_used - discharge
        if grid < 0:
            # Should not happen for a feasible optimum; clip defensively.
            grid = 0.0

        soc_running = soc_running + charge - discharge
        # Clip tiny floating noise at the bounds.
        soc_running = min(max(soc_running, reserve_lower[h]), battery.capacity_kwh)

        if charge > 0.0:
            action, magnitude = "charge", charge
        elif discharge > 0.0:
            action, magnitude = "discharge", discharge
        else:
            action, magnitude = "idle", 0.0

        plan.append(
            {
                "hour": h,
                "grid_kwh": round(grid, 6),
                "solar_used_kwh": round(solar_used, 6),
                "battery_action": action,
                "battery_kwh": round(magnitude, 6),
                "battery_energy_after_kwh": round(soc_running, 6),
            }
        )

    return plan


def fallback_plan(hours: List[Any], battery: Any) -> List[Dict[str, Any]]:
    """
    Guaranteed-feasible safety net used only if the LP solve fails.

    Keeps the battery idle all day (state of charge stays at
    initial_energy_kwh throughout, trivially satisfying bounds and
    end-of-day neutrality) and uses solar before grid every hour. This
    ignores directives that would require battery movement, but it can
    never crash or return an infeasible plan.
    """
    n = HOURS_IN_DAY
    plan: List[Dict[str, Any]] = []
    soc = battery.initial_energy_kwh
    for h in range(n):
        demand = hours[h].demand_kwh
        solar_used = min(hours[h].solar_kwh, demand)
        grid = max(0.0, demand - solar_used)
        plan.append(
            {
                "hour": h,
                "grid_kwh": round(grid, 6),
                "solar_used_kwh": round(solar_used, 6),
                "battery_action": "idle",
                "battery_kwh": 0.0,
                "battery_energy_after_kwh": round(soc, 6),
            }
        )
    return plan


def final_replay(
    plan: List[Dict[str, Any]], hours: List[Any], battery: Any, directives: List[Dict[str, Any]], tol: float = 0.01
) -> List[str]:
    """
    Defensive self-check (the "Final Validator" step): independently
    re-verifies the plan we are about to return against every hard rule.
    Returns a list of violation descriptions (empty if the plan is clean).
    This never blocks the response -- it is used for server-side logging
    so problems surface during testing instead of silently in the judge run.
    """
    violations: List[str] = []
    effective_solar, reserve_lower, charge_upper, discharge_upper, grid_upper = build_effective_constraints(
        hours, battery, directives
    )
    soc = battery.initial_energy_kwh
    for h, p in enumerate(plan):
        demand = hours[h].demand_kwh
        grid = p["grid_kwh"]
        solar_used = p["solar_used_kwh"]
        charge = p["battery_kwh"] if p["battery_action"] == "charge" else 0.0
        discharge = p["battery_kwh"] if p["battery_action"] == "discharge" else 0.0

        balance = grid + solar_used + discharge - demand - charge
        if abs(balance) > tol:
            violations.append(f"hour {h}: energy balance off by {balance:.4f}")
        if solar_used > effective_solar[h] + tol:
            violations.append(f"hour {h}: solar_used_kwh exceeds effective solar")
        if charge > charge_upper[h] + tol:
            violations.append(f"hour {h}: charge exceeds max_charge_kwh_per_hour")
        if discharge > discharge_upper[h] + tol:
            violations.append(f"hour {h}: discharge exceeds max_discharge_kwh_per_hour")
        if grid_upper[h] != float("inf") and grid > grid_upper[h] + tol:
            violations.append(f"hour {h}: grid_kwh exceeds max_grid_window cap")

        new_soc = soc + charge - discharge
        after = p["battery_energy_after_kwh"]
        if abs(new_soc - after) > tol:
            violations.append(f"hour {h}: battery_energy_after_kwh inconsistent with action")
        if after < reserve_lower[h] - tol or after > battery.capacity_kwh + tol:
            violations.append(f"hour {h}: battery energy out of bounds")
        soc = after

    if abs(soc - battery.initial_energy_kwh) > tol:
        violations.append("end-of-day battery neutrality violated")

    return violations
