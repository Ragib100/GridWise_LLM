"""
Local test suite for GridWise LLM.

Two independent modes:

  python tests/test_gridwise.py unit
      Pure logic tests for validator.py + optimizer.py. No network, no LLM
      key required. Covers: malformed/invalid LLM output, out-of-range
      values, duplicate/missing note mappings, each directive type, no_op,
      battery capacity boundaries, zero solar, high solar, cheap-vs-expensive
      tariff periods, and directive precedence when windows overlap.

  python tests/test_gridwise.py live [--base-url http://localhost:8000]
      Runs the 10 official Public Sample Cases against a *running* server
      (so /optimize-energy actually calls your configured LLM). For each
      case it checks: directive_interpretation applies/type/hours/values
      match the public ground truth, the returned hourly_plan independently
      replays as valid (energy balance, effective solar, battery bounds,
      rate limits, directive constraints, end-of-day neutrality), and the
      reported totals match values recalculated from hourly_plan. Equivalent
      (not byte-identical) optimal schedules are accepted, per the official
      "no byte-for-byte matching" rule.

  python tests/test_gridwise.py all
      Runs both.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import optimizer  # noqa: E402
from app.validator import validate_directives  # noqa: E402

TOL = 0.01
FAILURES = []


def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def _battery(**kwargs):
    defaults = dict(
        capacity_kwh=200.0,
        initial_energy_kwh=100.0,
        minimum_energy_kwh=20.0,
        max_charge_kwh_per_hour=50.0,
        max_discharge_kwh_per_hour=50.0,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _hours(demand, solar, tariff):
    assert len(demand) == len(solar) == len(tariff) == 24
    return [
        SimpleNamespace(hour=h, demand_kwh=demand[h], solar_kwh=solar[h], tariff_bdt_per_kwh=tariff[h])
        for h in range(24)
    ]


def _replay_ok(plan, hours, battery, directives):
    violations = optimizer.final_replay(plan, hours, battery, directives)
    return len(violations) == 0, violations


# ---------------------------------------------------------------------------
# Validator (guardrail) unit tests
# ---------------------------------------------------------------------------

def test_validator_no_notes_all_missing():
    """No LLM output at all -> every note safely becomes no_op."""
    result = validate_directives([], num_notes=3, battery_capacity_kwh=200)
    check(
        "validator: missing LLM output -> all no_op",
        len(result) == 3 and all(d["directive_type"] == "no_op" and not d["applies"] for d in result),
    )


def test_validator_valid_each_directive_type():
    raw = [
        {"note_index": 0, "directive_type": "solar_reduction", "hours": [13, 14], "factor": 0.2,
         "minimum_energy_kwh": None, "max_grid_kwh": None, "explanation": "x"},
        {"note_index": 1, "directive_type": "no_charge_window", "hours": [2, 3, 4], "factor": None,
         "minimum_energy_kwh": None, "max_grid_kwh": None, "explanation": "x"},
        {"note_index": 2, "directive_type": "no_discharge_window", "hours": [18, 19], "factor": None,
         "minimum_energy_kwh": None, "max_grid_kwh": None, "explanation": "x"},
        {"note_index": 3, "directive_type": "minimum_battery_reserve", "hours": [18, 19, 20], "factor": None,
         "minimum_energy_kwh": 100.0, "max_grid_kwh": None, "explanation": "x"},
        {"note_index": 4, "directive_type": "max_grid_window", "hours": [18, 19], "factor": None,
         "minimum_energy_kwh": None, "max_grid_kwh": 155.0, "explanation": "x"},
        {"note_index": 5, "directive_type": "no_op", "hours": None, "factor": None,
         "minimum_energy_kwh": None, "max_grid_kwh": None, "explanation": "irrelevant"},
    ]
    result = validate_directives(raw, num_notes=6, battery_capacity_kwh=200)
    check("validator: all 6 directive types parse", len(result) == 6)
    check("validator: solar_reduction shape", result[0]["structured_adjustment"] == {"hours": [13, 14], "factor": 0.2})
    check("validator: no_charge_window shape", result[1]["structured_adjustment"] == {"hours": [2, 3, 4]})
    check("validator: no_discharge_window shape", result[2]["structured_adjustment"] == {"hours": [18, 19]})
    check("validator: minimum_battery_reserve shape",
          result[3]["structured_adjustment"] == {"hours": [18, 19, 20], "minimum_energy_kwh": 100.0})
    check("validator: max_grid_window shape",
          result[4]["structured_adjustment"] == {"hours": [18, 19], "max_grid_kwh": 155.0})
    check("validator: no_op has null adjustment and applies=False",
          result[5]["structured_adjustment"] is None and result[5]["applies"] is False)
    check("validator: non-no_op entries have applies=True", all(d["applies"] for d in result[:5]))


def test_validator_invalid_llm_output_is_safe():
    raw = [
        {"note_index": 0, "directive_type": "solar_reduction", "hours": [13, 14], "factor": 1.7,
         "minimum_energy_kwh": None, "max_grid_kwh": None, "explanation": "out of range factor"},
        {"note_index": 1, "directive_type": "teleport_energy", "hours": [1], "factor": None,
         "minimum_energy_kwh": None, "max_grid_kwh": None, "explanation": "unsupported type"},
        {"note_index": 2, "directive_type": "no_charge_window", "hours": [5, 3], "factor": None,
         "minimum_energy_kwh": None, "max_grid_kwh": None, "explanation": "hours not ascending"},
        {"note_index": 3, "directive_type": "no_charge_window", "hours": [5, 5], "factor": None,
         "minimum_energy_kwh": None, "max_grid_kwh": None, "explanation": "duplicate hour"},
        {"note_index": 4, "directive_type": "no_charge_window", "hours": [30], "factor": None,
         "minimum_energy_kwh": None, "max_grid_kwh": None, "explanation": "hour out of 0-23 range"},
        "not a dict",
    ]
    result = validate_directives(raw, num_notes=5, battery_capacity_kwh=200)
    check("validator: never crashes on malformed input", len(result) == 5)
    check("validator: all malformed directives downgraded to no_op", all(d["directive_type"] == "no_op" for d in result))


def test_validator_impossible_reserve_value():
    raw = [
        {"note_index": 0, "directive_type": "minimum_battery_reserve", "hours": [1], "factor": None,
         "minimum_energy_kwh": 99999.0, "max_grid_kwh": None, "explanation": "exceeds capacity"},
    ]
    result = validate_directives(raw, num_notes=1, battery_capacity_kwh=200)
    check("validator: reserve above battery capacity -> no_op", result[0]["directive_type"] == "no_op")


def test_validator_duplicate_and_out_of_range_note_index():
    raw = [
        {"note_index": 0, "directive_type": "no_charge_window", "hours": [1], "factor": None,
         "minimum_energy_kwh": None, "max_grid_kwh": None, "explanation": "first"},
        {"note_index": 0, "directive_type": "no_charge_window", "hours": [2], "factor": None,
         "minimum_energy_kwh": None, "max_grid_kwh": None, "explanation": "duplicate note_index, ignored"},
        {"note_index": 7, "directive_type": "no_charge_window", "hours": [3], "factor": None,
         "minimum_energy_kwh": None, "max_grid_kwh": None, "explanation": "out of range, ignored"},
    ]
    result = validate_directives(raw, num_notes=2, battery_capacity_kwh=200)
    check("validator: exactly one entry per note, no dup/oob leakage", len(result) == 2)
    check("validator: first mapping wins on duplicate note_index",
          result[0]["structured_adjustment"] == {"hours": [1]})
    check("validator: missing note_index 1 becomes no_op", result[1]["directive_type"] == "no_op")


# ---------------------------------------------------------------------------
# Optimizer physics unit tests
# ---------------------------------------------------------------------------

def test_optimizer_no_directives_feasible():
    demand = [100.0] * 24
    solar = [0.0] * 12 + [50.0] * 12
    tariff = [5.0] * 24
    hours = _hours(demand, solar, tariff)
    battery = _battery()
    plan = optimizer.solve(hours, battery, directives=[])
    check("optimizer: LP finds a solution with no directives", plan is not None)
    ok, violations = _replay_ok(plan, hours, battery, [])
    check("optimizer: plan with no directives replays clean", ok, str(violations))


def test_optimizer_zero_solar_all_day():
    demand = [80.0] * 24
    solar = [0.0] * 24
    tariff = [5.0 if h < 12 else 15.0 for h in range(24)]
    hours = _hours(demand, solar, tariff)
    battery = _battery()
    plan = optimizer.solve(hours, battery, directives=[])
    check("optimizer: zero solar all day is feasible", plan is not None)
    ok, violations = _replay_ok(plan, hours, battery, [])
    check("optimizer: zero solar plan replays clean", ok, str(violations))
    check("optimizer: zero solar -> no solar ever used", all(p["solar_used_kwh"] == 0 for p in plan))


def test_optimizer_high_solar_covers_demand():
    demand = [50.0] * 24
    solar = [200.0] * 24  # far exceeds demand every hour
    tariff = [10.0] * 24
    hours = _hours(demand, solar, tariff)
    battery = _battery(initial_energy_kwh=20.0, minimum_energy_kwh=20.0)
    plan = optimizer.solve(hours, battery, directives=[])
    check("optimizer: abundant solar is feasible", plan is not None)
    ok, violations = _replay_ok(plan, hours, battery, [])
    check("optimizer: abundant solar plan replays clean", ok, str(violations))
    check("optimizer: solar never exceeds demand+charge need when abundant",
          all(p["solar_used_kwh"] <= demand[h] + p["battery_kwh"] + TOL for h, p in enumerate(plan)))


def test_optimizer_cheap_vs_expensive_tariff_uses_battery():
    demand = [50.0] * 24
    solar = [0.0] * 24
    tariff = [2.0] * 12 + [50.0] * 12  # very cheap morning, very expensive evening
    hours = _hours(demand, solar, tariff)
    battery = _battery(capacity_kwh=400, initial_energy_kwh=50, minimum_energy_kwh=0,
                        max_charge_kwh_per_hour=50, max_discharge_kwh_per_hour=50)
    plan = optimizer.solve(hours, battery, directives=[])
    check("optimizer: cheap/expensive tariff scenario is feasible", plan is not None)
    ok, violations = _replay_ok(plan, hours, battery, [])
    check("optimizer: cheap/expensive tariff plan replays clean", ok, str(violations))
    charged_cheap = any(p["battery_action"] == "charge" for p in plan[:12])
    discharged_expensive = any(p["battery_action"] == "discharge" for p in plan[12:])
    check("optimizer: charges during cheap hours and discharges during expensive hours",
          charged_cheap and discharged_expensive)
    # A no-battery baseline costs sum(demand*tariff); using the battery to shift
    # load to cheap hours must not cost more than doing nothing.
    baseline_cost = sum(demand[h] * tariff[h] for h in range(24))
    plan_cost = sum(p["grid_kwh"] * tariff[h] for h, p in enumerate(plan))
    check("optimizer: battery arbitrage is at least as cheap as no battery use",
          plan_cost <= baseline_cost + TOL, f"plan_cost={plan_cost} baseline={baseline_cost}")


def test_optimizer_no_charge_and_no_discharge_windows_are_respected():
    demand = [50.0] * 24
    solar = [0.0] * 24
    tariff = [10.0] * 24
    hours = _hours(demand, solar, tariff)
    battery = _battery(initial_energy_kwh=100, minimum_energy_kwh=20, capacity_kwh=200)
    directives = [
        {"note_index": 0, "applies": True, "directive_type": "no_charge_window",
         "structured_adjustment": {"hours": [2, 3, 4]}, "explanation": "x"},
        {"note_index": 1, "applies": True, "directive_type": "no_discharge_window",
         "structured_adjustment": {"hours": [18, 19]}, "explanation": "x"},
    ]
    plan = optimizer.solve(hours, battery, directives)
    check("optimizer: no_charge/no_discharge scenario is feasible", plan is not None)
    ok, violations = _replay_ok(plan, hours, battery, directives)
    check("optimizer: no_charge/no_discharge plan replays clean", ok, str(violations))
    check("optimizer: no charging happens in the no_charge_window",
          all(plan[h]["battery_action"] != "charge" for h in (2, 3, 4)))
    check("optimizer: no discharging happens in the no_discharge_window",
          all(plan[h]["battery_action"] != "discharge" for h in (18, 19)))


def test_optimizer_minimum_reserve_is_respected():
    demand = [50.0] * 24
    solar = [0.0] * 24
    tariff = [10.0] * 24
    hours = _hours(demand, solar, tariff)
    battery = _battery(initial_energy_kwh=120, minimum_energy_kwh=10, capacity_kwh=200,
                        max_discharge_kwh_per_hour=100)
    directives = [
        {"note_index": 0, "applies": True, "directive_type": "minimum_battery_reserve",
         "structured_adjustment": {"hours": [18, 19, 20], "minimum_energy_kwh": 100.0}, "explanation": "x"},
    ]
    plan = optimizer.solve(hours, battery, directives)
    check("optimizer: minimum reserve scenario is feasible", plan is not None)
    ok, violations = _replay_ok(plan, hours, battery, directives)
    check("optimizer: minimum reserve plan replays clean", ok, str(violations))
    check("optimizer: reserve honored during the required window",
          all(plan[h]["battery_energy_after_kwh"] >= 100.0 - TOL for h in (18, 19, 20)))


def test_optimizer_max_grid_window_is_respected():
    demand = [200.0] * 24
    solar = [0.0] * 24
    tariff = [10.0] * 24
    hours = _hours(demand, solar, tariff)
    battery = _battery(initial_energy_kwh=150, minimum_energy_kwh=20, capacity_kwh=300,
                        max_discharge_kwh_per_hour=100, max_charge_kwh_per_hour=100)
    directives = [
        {"note_index": 0, "applies": True, "directive_type": "max_grid_window",
         "structured_adjustment": {"hours": [18, 19, 20], "max_grid_kwh": 120.0}, "explanation": "x"},
    ]
    plan = optimizer.solve(hours, battery, directives)
    check("optimizer: max_grid_window scenario is feasible", plan is not None)
    ok, violations = _replay_ok(plan, hours, battery, directives)
    check("optimizer: max_grid_window plan replays clean", ok, str(violations))
    check("optimizer: grid import capped during the window",
          all(plan[h]["grid_kwh"] <= 120.0 + TOL for h in (18, 19, 20)))


def test_optimizer_solar_reduction_caps_usage():
    demand = [150.0] * 24
    solar = [0.0] * 12 + [200.0] * 12
    tariff = [10.0] * 24
    hours = _hours(demand, solar, tariff)
    battery = _battery()
    directives = [
        {"note_index": 0, "applies": True, "directive_type": "solar_reduction",
         "structured_adjustment": {"hours": [12, 13], "factor": 0.25}, "explanation": "x"},
    ]
    plan = optimizer.solve(hours, battery, directives)
    check("optimizer: solar_reduction scenario is feasible", plan is not None)
    ok, violations = _replay_ok(plan, hours, battery, directives)
    check("optimizer: solar_reduction plan replays clean", ok, str(violations))
    check("optimizer: solar usage capped to 25% during the reduction window",
          all(plan[h]["solar_used_kwh"] <= 200.0 * 0.25 + TOL for h in (12, 13)))


def test_optimizer_battery_capacity_boundaries():
    """Battery must never exceed capacity or drop below its minimum."""
    demand = [10.0] * 12 + [300.0] * 12
    solar = [500.0] * 12 + [0.0] * 12  # huge solar surplus in the morning
    tariff = [10.0] * 24
    hours = _hours(demand, solar, tariff)
    battery = _battery(capacity_kwh=150, initial_energy_kwh=50, minimum_energy_kwh=10,
                        max_charge_kwh_per_hour=200, max_discharge_kwh_per_hour=200)
    plan = optimizer.solve(hours, battery, directives=[])
    check("optimizer: capacity-boundary scenario is feasible", plan is not None)
    ok, violations = _replay_ok(plan, hours, battery, [])
    check("optimizer: capacity-boundary plan replays clean", ok, str(violations))
    check("optimizer: battery energy never exceeds capacity",
          all(p["battery_energy_after_kwh"] <= 150.0 + TOL for p in plan))
    check("optimizer: battery energy never drops below minimum",
          all(p["battery_energy_after_kwh"] >= 10.0 - TOL for p in plan))


def test_optimizer_end_of_day_neutrality():
    demand = [60.0] * 24
    solar = [0.0] * 12 + [100.0] * 12
    tariff = [8.0] * 12 + [20.0] * 12
    hours = _hours(demand, solar, tariff)
    battery = _battery(initial_energy_kwh=75)
    plan = optimizer.solve(hours, battery, directives=[])
    check("optimizer: end-of-day neutrality scenario is feasible", plan is not None)
    check("optimizer: battery returns to its initial energy at end of day",
          abs(plan[-1]["battery_energy_after_kwh"] - 75.0) <= TOL)


def test_optimizer_overlapping_directives_take_the_stricter_bound():
    """Directive precedence: most restrictive wins when windows overlap."""
    demand = [50.0] * 24
    solar = [100.0] * 24
    tariff = [10.0] * 24
    hours = _hours(demand, solar, tariff)
    battery = _battery()
    directives = [
        {"note_index": 0, "applies": True, "directive_type": "solar_reduction",
         "structured_adjustment": {"hours": [10, 11], "factor": 0.6}, "explanation": "x"},
        {"note_index": 1, "applies": True, "directive_type": "solar_reduction",
         "structured_adjustment": {"hours": [10, 11], "factor": 0.2}, "explanation": "x"},
    ]
    effective_solar, *_ = optimizer.build_effective_constraints(hours, battery, directives)
    check("optimizer: overlapping solar_reduction directives use the stricter (lower) factor",
          abs(effective_solar[10] - 100.0 * 0.2) < TOL and abs(effective_solar[11] - 100.0 * 0.2) < TOL)


def run_unit_tests():
    print("=== Unit tests (validator + optimizer, no network) ===")
    test_validator_no_notes_all_missing()
    test_validator_valid_each_directive_type()
    test_validator_invalid_llm_output_is_safe()
    test_validator_impossible_reserve_value()
    test_validator_duplicate_and_out_of_range_note_index()
    test_optimizer_no_directives_feasible()
    test_optimizer_zero_solar_all_day()
    test_optimizer_high_solar_covers_demand()
    test_optimizer_cheap_vs_expensive_tariff_uses_battery()
    test_optimizer_no_charge_and_no_discharge_windows_are_respected()
    test_optimizer_minimum_reserve_is_respected()
    test_optimizer_max_grid_window_is_respected()
    test_optimizer_solar_reduction_caps_usage()
    test_optimizer_battery_capacity_boundaries()
    test_optimizer_end_of_day_neutrality()
    test_optimizer_overlapping_directives_take_the_stricter_bound()


# ---------------------------------------------------------------------------
# Live sample-case runner (hits a running server, exercises the real LLM)
# ---------------------------------------------------------------------------

def _replay_hourly_plan_from_response(hours_input, battery_input, directive_interpretation, hourly_plan):
    """Re-check a live response's hourly_plan against the same physics rules,
    using the request's own hours/battery (not app internals)."""
    hours = _hours(
        [h["demand_kwh"] for h in hours_input],
        [h["solar_kwh"] for h in hours_input],
        [h["tariff_bdt_per_kwh"] for h in hours_input],
    )
    battery = _battery(**battery_input)
    directives = []
    for d in directive_interpretation:
        directives.append(
            {
                "note_index": d["note_index"],
                "applies": d["applies"],
                "directive_type": d["directive_type"],
                "structured_adjustment": d["structured_adjustment"],
                "explanation": d.get("explanation", ""),
            }
        )
    plan = [
        {
            "hour": p["hour"],
            "grid_kwh": p["grid_kwh"],
            "solar_used_kwh": p["solar_used_kwh"],
            "battery_action": p["battery_action"],
            "battery_kwh": p["battery_kwh"],
            "battery_energy_after_kwh": p["battery_energy_after_kwh"],
        }
        for p in hourly_plan
    ]
    return optimizer.final_replay(plan, hours, battery, directives)


def run_live_tests(base_url: str):
    import requests

    print(f"=== Live sample-case tests against {base_url} ===")
    health = requests.get(f"{base_url}/health", timeout=10)
    check("live: GET /health returns 200", health.status_code == 200)
    check("live: GET /health body", health.json() == {"status": "ok"})

    cases_path = Path(__file__).resolve().parent / "sample_cases.json"
    with open(cases_path) as f:
        cases = json.load(f)["cases"]

    for case in cases:
        case_id = case["id"]
        req_body = case["input"]
        expected = case["expected_output"]

        resp = requests.post(f"{base_url}/optimize-energy", json=req_body, timeout=30)
        if not check(f"live[{case_id}]: HTTP 200", resp.status_code == 200, f"got {resp.status_code}: {resp.text[:300]}"):
            continue
        body = resp.json()

        check(f"live[{case_id}]: scenario_id echoed", body.get("scenario_id") == req_body["scenario_id"])

        got_di = body.get("directive_interpretation", [])
        exp_di = expected["directive_interpretation"]
        check(f"live[{case_id}]: one directive_interpretation entry per note, in order",
              [d["note_index"] for d in got_di] == list(range(len(req_body["operator_notes"]))))

        for exp_entry in exp_di:
            idx = exp_entry["note_index"]
            got_entry = next((d for d in got_di if d["note_index"] == idx), None)
            if got_entry is None:
                check(f"live[{case_id}] note {idx}: present", False)
                continue
            check(
                f"live[{case_id}] note {idx}: applies/type match ground truth",
                got_entry["applies"] == exp_entry["applies"] and got_entry["directive_type"] == exp_entry["directive_type"],
                f"expected {exp_entry['applies']}/{exp_entry['directive_type']}, "
                f"got {got_entry['applies']}/{got_entry['directive_type']}",
            )
            if exp_entry["applies"]:
                exp_adj = exp_entry["structured_adjustment"]
                got_adj = got_entry.get("structured_adjustment") or {}
                hours_match = got_adj.get("hours") == exp_adj.get("hours")
                numeric_ok = True
                for key in ("factor", "minimum_energy_kwh", "max_grid_kwh"):
                    if key in exp_adj:
                        got_val = got_adj.get(key)
                        numeric_ok = numeric_ok and got_val is not None and abs(got_val - exp_adj[key]) <= TOL
                check(
                    f"live[{case_id}] note {idx}: structured_adjustment matches ground truth",
                    hours_match and numeric_ok,
                    f"expected {exp_adj}, got {got_adj}",
                )

        hourly_plan = body.get("hourly_plan", [])
        check(f"live[{case_id}]: hourly_plan has 24 entries", len(hourly_plan) == 24)

        violations = _replay_hourly_plan_from_response(req_body["hours"], req_body["battery"], got_di, hourly_plan)
        check(f"live[{case_id}]: hourly_plan independently replays as valid", len(violations) == 0, str(violations))

        recomputed_grid = sum(p["grid_kwh"] for p in hourly_plan)
        recomputed_cost = sum(p["grid_kwh"] * req_body["hours"][p["hour"]]["tariff_bdt_per_kwh"] for p in hourly_plan)
        recomputed_peak = max(p["grid_kwh"] for p in hourly_plan)
        check(f"live[{case_id}]: total_grid_kwh matches recalculated value",
              abs(body.get("total_grid_kwh", -1) - recomputed_grid) <= TOL)
        check(f"live[{case_id}]: total_cost_bdt matches recalculated value",
              abs(body.get("total_cost_bdt", -1) - recomputed_cost) <= TOL)
        check(f"live[{case_id}]: peak_grid_kwh matches recalculated value",
              abs(body.get("peak_grid_kwh", -1) - recomputed_peak) <= TOL)

        expected_cost = expected["total_cost_bdt"]
        if expected_cost > TOL:
            ratio = expected_cost / recomputed_cost if recomputed_cost > TOL else 0
            check(
                f"live[{case_id}]: cost within ~1% of the public reference optimal cost",
                ratio >= 0.99,
                f"reference={expected_cost}, got={recomputed_cost}",
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["unit", "live", "all"], default="unit", nargs="?")
    parser.add_argument("--base-url", default=os.getenv("GRIDWISE_BASE_URL", "http://localhost:8000"))
    args = parser.parse_args()

    if args.mode in ("unit", "all"):
        run_unit_tests()
    if args.mode in ("live", "all"):
        run_live_tests(args.base_url)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for name in FAILURES:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
