# GridWise LLM — BUP CSE Fest 2026 Preliminary

An HTTP API that takes a 24-hour campus energy scenario plus 1–3 natural-language operator
notes, uses an LLM to turn each note into a structured directive (or `no_op`), deterministically
validates that directive, and hands it to an exact cost-minimizing LP optimizer that returns a
valid, lowest-cost 24-hour grid/solar/battery schedule.

```
operator notes ──▶ LLM interpreter ──▶ deterministic guardrails ──▶ LP optimizer ──▶ final replay (logged) ──▶ response
                    (app/llm.py)         (app/validator.py)          (app/optimizer.py)
```

## Why an LLM, and how it's used

The challenge requires that operator notes are interpreted by a real language model, not
regex/keyword matching, because hidden test notes paraphrase the same directive in different
wording ("6 PM" / "18:00" / "six in the evening"). `app/llm.py` sends the notes plus read-only
hourly context (demand/solar/tariff, so the model can ground relative phrases like "after
sunset") to the configured model with a strict JSON-schema / tool-use call, and the model's
**only** job is to return one structured directive per note — it never computes cost or the
schedule itself. Its output is treated as completely untrusted until it passes `app/validator.py`.

## Guardrail flow (`app/validator.py`)

Deterministic Python, no second LLM call, never raises. Each non-`no_op` directive type has a
strict Pydantic model in `app/models.py` (`SolarReductionAdjustment`,
`MinimumBatteryReserveAdjustment`, `NoChargeWindowAdjustment`, `NoDischargeWindowAdjustment`,
`MaxGridWindowAdjustment`), and the validator dispatches to the right one by `directive_type`
(`DIRECTIVE_ADJUSTMENT_MODELS`) — a manual discriminated union, since the fixed wire schema keeps
`directive_type` as a sibling of `structured_adjustment` rather than nested inside it. For every
note it:
1. Rejects any `directive_type` not in the six allowed values → downgrades to `no_op`.
2. Constructs the matching adjustment model from the raw fields; any `pydantic.ValidationError`
   (non-integer/out-of-range/duplicate hours, `factor` outside 0–1, negative grid cap, a boolean
   where a number is required, etc.) → `no_op`. Hours are normalized to ascending order rather
   than rejected for being out of order, so a model-emitted overnight window like `[23, 0, 1]`
   ("11 PM to 2 AM") is still accepted as `[0, 1, 23]`.
3. Separately rejects a reserve value above battery capacity → `no_op` (needs request-level
   context the per-type model doesn't have).
4. Fixes note-index bookkeeping: duplicate mappings keep the first valid one, out-of-range or
   missing indices are filled with a safe `no_op`, so the response always has exactly one entry
   per note, in order.
5. Forces `applies=False` + `structured_adjustment=null` for `no_op`, and `applies=True` with
   the exact required shape for everything else.

Malformed JSON, an unsupported directive type, or a missing API key never crashes the service —
worst case, every note becomes `no_op` and the schedule is still computed and returned.

## Optimization approach (`app/optimizer.py`)

The 24-hour problem (minimize `Σ grid[h] * tariff[h]` subject to energy balance, battery SOC
bounds/rate limits, and directive constraints) is a linear program. It's solved **exactly** with
`scipy.optimize.linprog` (HiGHS) — no heuristics, no approximation, sub-second for 24 hours. This
was checked against all 10 official public sample cases (ground-truth directives fed directly to
the optimizer): it reproduces the reference optimal cost **exactly** on every case.

Variables per hour: `grid`, `solar_used`, `charge`, `discharge`, `battery_energy_after` (SOC).
Hard constraints: energy balance, `0 ≤ solar_used ≤ effective_solar`, rate limits, SOC bounds,
and `SOC[23] = initial_energy_kwh` (end-of-day neutrality, fixed as an equality bound). If the LP
is ever infeasible (should not happen for valid, feasible scenarios), `fallback_plan()` returns a
trivially-feasible "battery idle, solar-first" plan so the service always answers with *something*
valid rather than erroring out.

A defensive `final_replay()` independently re-checks the returned plan against every hard rule
after the LP solves, and logs any violation server-side (it never blocks the response — the goal
is to surface bugs during development, not to fail the request).

### Directive precedence (when windows/directives overlap)

| Directive | Rule when multiple apply to the same hour |
|---|---|
| `solar_reduction` | most restrictive (lowest) factor wins |
| `minimum_battery_reserve` | highest reserve requirement wins |
| `no_charge_window` | any matching directive forces charge to 0 (hard veto, union of windows) |
| `no_discharge_window` | any matching directive forces discharge to 0 (hard veto, union of windows) |
| `max_grid_window` | most restrictive (lowest) cap wins |
| End-of-day neutrality | unconditional; overrides a `minimum_battery_reserve` that would otherwise touch hour 23 |

## API

### `GET /health`
Returns `200 {"status": "ok"}`.

### `POST /optimize-energy`
Request/response schema is taken verbatim from the official Problem Statement — see
`app/models.py`. Example:

```bash
curl -s -X POST http://localhost:8000/optimize-energy \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_id": "GRID-101",
    "operator_notes": [
      "Solar output will drop to about 20% from 1 PM to 3 PM.",
      "Do not charge the battery between 2 PM and 4 PM.",
      "The cafeteria menu changes tomorrow."
    ],
    "hours": [ {"hour": 0, "demand_kwh": 180, "solar_kwh": 0, "tariff_bdt_per_kwh": 7}, ... 24 total ... ],
    "battery": {
      "capacity_kwh": 500,
      "initial_energy_kwh": 200,
      "minimum_energy_kwh": 50,
      "max_charge_kwh_per_hour": 100,
      "max_discharge_kwh_per_hour": 100
    }
  }'
```

Response:
```json
{
  "scenario_id": "GRID-101",
  "directive_interpretation": [
    {"note_index": 0, "applies": true, "directive_type": "solar_reduction",
     "structured_adjustment": {"hours": [13, 14], "factor": 0.2}, "explanation": "..."},
    {"note_index": 1, "applies": true, "directive_type": "no_charge_window",
     "structured_adjustment": {"hours": [14, 15]}, "explanation": "..."},
    {"note_index": 2, "applies": false, "directive_type": "no_op",
     "structured_adjustment": null, "explanation": "..."}
  ],
  "hourly_plan": [ {"hour": 0, "grid_kwh": ..., "solar_used_kwh": ..., "battery_action": "idle", "battery_kwh": 0, "battery_energy_after_kwh": ...}, ... 24 total ... ],
  "total_grid_kwh": 0,
  "total_cost_bdt": 0,
  "peak_grid_kwh": 0,
  "plan_summary": "..."
}
```

HTTP codes: `200` success, `400` malformed/structurally invalid JSON body, `500` unexpected
internal error (no stack traces or secrets are ever included in a response).

## Environment variables

See `.env.example`. Copy it to `.env` for local runs (never commit `.env`).

| Variable | Meaning |
|---|---|
| `LLM_PROVIDER` | `gemini` (default — Google Gemini, native structured output), `openai` (any OpenAI-compatible Chat Completions API), or `anthropic` |
| `LLM_MODEL` | model name/id for the selected provider (default `gemini-2.5-flash`) |
| `LLM_API_KEY` | secret key, read only from the environment — never hard-coded |
| `LLM_BASE_URL` | only for `LLM_PROVIDER=openai`; point at a non-OpenAI OpenAI-compatible host (Groq, OpenRouter, a local server, etc.) |
| `LLM_FALLBACK_MODELS` | comma-separated extra model ids on the same provider, tried in order when the primary returns 429/5xx/404 or times out. **Recommended on free-tier keys** — Gemini free-tier quotas are small and per-model |
| `LLM_FALLBACK_PROVIDER` / `LLM_FALLBACK_MODEL` / `LLM_FALLBACK_API_KEY` / `LLM_FALLBACK_BASE_URL` | optional secondary provider (e.g. Groq via its OpenAI-compatible API) tried last, after every primary-provider model fails |
| `LLM_TIMEOUT_SECONDS` | per-call LLM timeout, default 12s |
| `LLM_TOTAL_BUDGET_SECONDS` | wall-clock budget for the whole interpretation step across all fallback attempts, default 24s (endpoint budget is 30s) |
| `LLM_RETRY_PAUSE_SECONDS` | pause before re-trying the whole chain once more after a fully failed pass, default 2s |
| `PORT` | port to listen on (hosting platforms usually inject this themselves) |

**Provider/model is swappable purely through these env vars** — `app/llm.py` dispatches on
`LLM_PROVIDER`; no code change needed to switch models or providers.

## Local setup & running

```bash
python3 -m venv .venv && source .venv/bin/activate      # optional but recommended
pip install -r requirements.txt
cp .env.example .env      # then fill in LLM_PROVIDER / LLM_MODEL / LLM_API_KEY
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Check readiness:
```bash
curl http://localhost:8000/health
```

### LLM resilience (fallback chain)

`interpret_notes()` tries the primary `LLM_MODEL` first, then each `LLM_FALLBACK_MODELS` entry,
then the optional secondary provider; a rate-limit (429), server error (5xx), unknown model (404)
or timeout moves on to the next attempt, and after one fully failed pass the chain is retried
once more after `LLM_RETRY_PAUSE_SECONDS`. All attempts share `LLM_TOTAL_BUDGET_SECONDS` so the
endpoint always answers inside its 30s budget. Only if **every** attempt fails (or
`LLM_API_KEY` is missing) does every note safely become `no_op` (logged as a warning) instead of
the service crashing — the schedule is still computed and returned.

## Testing

`tests/test_gridwise.py` has two independent modes:

```bash
# Pure logic tests for validator.py + optimizer.py. No network, no LLM key needed.
python3 tests/test_gridwise.py unit

# Runs the 10 official Public Sample Cases against a RUNNING server (exercises
# the real configured LLM end to end). Start the server first, then:
python3 tests/test_gridwise.py live --base-url http://localhost:8000

# Both:
python3 tests/test_gridwise.py all
```

`unit` covers: every directive type, `no_op`, malformed/invalid LLM output, out-of-range values,
duplicate/missing note mappings, battery capacity boundaries, zero solar, high solar,
cheap-vs-expensive tariff periods, and directive precedence on overlapping windows.

`live` POSTs each of the 10 cases in `tests/sample_cases.json`, checks `directive_interpretation`
against the public ground truth, independently replays the returned `hourly_plan` against every
hard energy/battery rule, checks the reported totals against values recalculated from
`hourly_plan`, and checks cost is within ~1% of the public reference optimal cost. (Public cases
are not the hidden judge set — hidden notes will be paraphrased differently.)

The optimizer itself was separately verified offline against all 10 public cases' ground-truth
directives (bypassing the LLM): it reproduces the reference optimal cost exactly on every case.

## Docker

### Pull the prebuilt image (fallback deployment)

A public image is published on Docker Hub — no build step needed:

| | |
|---|---|
| Image | `ragib100/gridwise-llm` |
| Tags | `v1`, `latest` |
| Digest | `sha256:b33a9fed51e87f99ad0852aedc6cf26fbcb41496d69566b1841e8fad9ed4314f` |
| Port | `8000` (bound on `0.0.0.0`; honors a platform-provided `$PORT`) |

```bash
docker pull ragib100/gridwise-llm:v1
# or pin the exact digest:
docker pull ragib100/gridwise-llm@sha256:b33a9fed51e87f99ad0852aedc6cf26fbcb41496d69566b1841e8fad9ed4314f

# Supply the LLM config at run time (copy .env.example to .env and fill in LLM_API_KEY).
docker run --rm -p 8000:8000 --env-file .env ragib100/gridwise-llm:v1

# Or pass the variables inline:
docker run --rm -p 8000:8000 \
  -e LLM_PROVIDER=gemini -e LLM_MODEL=gemini-3.5-flash-lite \
  -e LLM_FALLBACK_MODELS=gemini-flash-lite-latest,gemini-3.1-flash-lite \
  -e LLM_API_KEY=<your-key> \
  ragib100/gridwise-llm:v1

curl http://localhost:8000/health        # -> {"status":"ok"}
```

Verified: a clean `docker pull` by digest followed by the `docker run` above answers `/health`
within a few seconds and returns correct interpretations for the public sample cases.

### Build it yourself

```bash
docker build -t gridwise-llm .
docker run -p 8000:8000 --env-file .env gridwise-llm
curl http://localhost:8000/health
```

The image contains no secrets — `LLM_API_KEY` etc. must be supplied at `docker run` time via
`--env-file` or your platform's environment-variable settings. It respects a platform-provided
`$PORT` (Render/Railway/etc.), defaulting to 8000 otherwise.

## Deployment

Any platform that can run a Docker container / a Python web service works (Render, Railway, Fly.io,
etc.) — behavior is what's judged, not the provider. Steps for a typical platform:

1. Push this repository to GitHub (private during the event, public after the submission
   deadline, per the Participant Guide).
2. Create a new **Web Service** from the repo (Render: "New +" → "Web Service"; Railway:
   "New Project" → "Deploy from GitHub repo").
3. Either let the platform build the `Dockerfile` directly, or set:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Set the environment variables from `.env.example` in the platform's dashboard
   (`LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, optionally `LLM_BASE_URL`,
   `LLM_TIMEOUT_SECONDS`). Do **not** put the real key in any committed file.
5. Deploy, then verify from an external network (not your dev machine):
   ```bash
   curl https://<your-app>.onrender.com/health
   ```
6. Confirm no login/VPN/manual-approval gate is in front of either endpoint.

## Security / no-secret policy

- `LLM_API_KEY` is read only from the environment (`os.getenv`) — never hard-coded, never logged.
- `.env` is git-ignored; only `.env.example` (with empty values) is committed.
- The Docker image installs dependencies and copies `app/` only — no `.env`, no secrets baked in.
- Unhandled exceptions return a generic `500 {"detail": "Internal server error."}` — no stack
  traces or internal state are ever exposed in a response or log line beyond a short message.

## Known limitations

- The optimizer assumes no round-trip battery efficiency loss and no grid export, matching the
  official Problem Statement's energy-balance equation (there is no efficiency or export field in
  the request schema).
- The interpretation step is bounded by `LLM_TOTAL_BUDGET_SECONDS` (default 24s) to stay safely
  inside the 30s per-request budget; if every model/provider in the fallback chain is
  unavailable within that budget, that request degrades to `no_op` for all notes rather than
  timing out.
- Gemini free-tier keys have very small per-model daily quotas (e.g. 20 requests/day on
  `gemini-2.5-flash` at the time of writing). Configure `LLM_FALLBACK_MODELS` and/or a billed key
  or secondary provider for any sustained load.
- The LP optimizer requires `scipy`; if it's ever unavailable/infeasible, `fallback_plan()`
  provides a always-feasible (but not directive-aware) safety net rather than failing the request.

## Dependencies

FastAPI, Uvicorn, Pydantic — HTTP API and request/response validation.
`requests` — plain HTTP calls to the LLM provider (no vendor SDK dependency, keeps the provider
swap in `app/llm.py` a one-file change).
NumPy / SciPy (`linprog`, HiGHS) — exact LP solve for the optimizer.
`python-dotenv` — optional convenience for loading `.env` in local dev only.

## Tooling disclosure

This solution was developed with the help of an AI coding assistant (Claude), used interactively
throughout design, implementation, and testing. All architecture decisions, the LLM-interpretation
strategy, the guardrail/validation design, and the LP optimization model were specified and
reviewed by the author; the assistant was not used to bypass the "no hard-coded phrase matching"
requirement — `app/llm.py` calls a real LLM for every note, and `app/validator.py` only ever
validates that model's output deterministically.
