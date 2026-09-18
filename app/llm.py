"""
LLM operator-note interpretation.

The LLM's ONLY job is: natural-language note -> structured directive JSON.
It never sees or performs optimization, and its output is NEVER trusted
directly -- validator.py deterministically re-checks everything before the
optimizer touches it.

Provider is selected purely through environment variables so it can be
swapped without touching this file's call sites:

    LLM_PROVIDER          "gemini" (Google Gemini, native structured output
                           via responseSchema -- default), "openai" (any
                           OpenAI-compatible Chat Completions API: OpenAI,
                           Groq, OpenRouter, Together, Fireworks, local
                           vLLM/Ollama, etc.), or "anthropic" (Claude
                           Messages API, tool-use).
    LLM_MODEL              model name/id for the selected provider, e.g.
                           "gemini-3.5-flash-lite".
    LLM_API_KEY            secret key, read only from the environment.
    LLM_BASE_URL           only for provider=openai; override the base URL
                           to point at a non-OpenAI OpenAI-compatible host.
    LLM_TIMEOUT_SECONDS    per-call timeout, default 12s.
    LLM_TOTAL_BUDGET_SECONDS
                           wall-clock budget for the whole interpretation
                           step across all fallback attempts, default 24s
                           (endpoint budget is 30s total, so this leaves
                           headroom for the optimizer + serialization).
    LLM_FALLBACK_MODELS    optional comma-separated list of extra model ids
                           on the SAME provider, tried in order if the
                           primary model returns a rate-limit (429), a
                           server error (5xx), an unknown-model 404, or
                           times out. Free-tier quotas are small and
                           per-model, so this multiplies the effective
                           request budget under judge load.
    LLM_FALLBACK_PROVIDER / LLM_FALLBACK_MODEL / LLM_FALLBACK_API_KEY /
    LLM_FALLBACK_BASE_URL  optional secondary provider (e.g. Groq via the
                           openai-compatible API) tried last, after every
                           model on the primary provider has failed.

Resilience matters here because the interpretation step is the ONLY place
the LLM is used: if every attempt fails, main.py degrades every note to
no_op, which is safe but scores zero on directive interpretation.
"""
import json
import logging
import os
import time

import requests

logger = logging.getLogger("gridwise.llm")

ALLOWED_DIRECTIVE_TYPES = [
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]


class LLMError(Exception):
    """Raised whenever the LLM call could not produce usable output.

    Callers (main.py) catch this and fall back to treating every note as
    no_op -- the service must never crash or hang because of a flaky/missing
    LLM provider.
    """


SYSTEM_PROMPT = """You are the operator-note interpreter for GridWise, a campus energy \
scheduling system. Your ONLY job is to convert short natural-language notes from campus \
operators into structured energy-management directives. You do NOT calculate a schedule, \
you do NOT compute costs, and you do NOT produce a final plan -- a separate deterministic \
optimizer does that after your output has been validated.

You will receive:
- A list of 1-3 operator notes, each with its note_index.
- Read-only context: the same 24-hour hour-by-hour demand_kwh/solar_kwh/tariff_bdt_per_kwh \
values and the battery parameters the operator can see. Use this ONLY to resolve relative \
time expressions (e.g. "after sunset" -> the hour solar_kwh drops to 0 and stays 0 through \
the evening; "before sunrise" -> the hour solar_kwh first becomes > 0; "evening peak" -> the \
hours of highest demand_kwh or tariff_bdt_per_kwh in the evening). NEVER invent or change \
any demand, solar, tariff, or battery number from this context -- it is for grounding time \
references only.

For every note, decide whether it describes one of these six directive types, or whether it \
is irrelevant/unsupported/ambiguous (no_op):

1. solar_reduction - usable solar is reduced during some hours.
   Fields: hours (list), factor (0-1, the fraction of solar that REMAINS usable; an 80% \
reduction means factor = 0.2; "roughly a fifth of normal output" also means factor = 0.2).
2. minimum_battery_reserve - the battery must be kept at or above some energy level during \
some hours.
   Fields: hours (list), minimum_energy_kwh (an absolute kWh value; if the note gives a \
percentage of battery capacity, convert it to kWh using the battery capacity_kwh given in \
context).
3. no_charge_window - the battery must not be charged during some hours.
   Fields: hours (list).
4. no_discharge_window - the battery must not be discharged during some hours.
   Fields: hours (list).
5. max_grid_window - grid import must not exceed some kWh limit during some hours.
   Fields: hours (list), max_grid_kwh.
6. no_op - the note does not affect today's 24-hour energy schedule (irrelevant, a \
distractor, purely informational, or too ambiguous/unsupported to map safely).

Rules you MUST follow:
- Return exactly one directive per note, referencing its note_index.
- Time windows are start-inclusive, end-exclusive, in whole hours: "1 PM to 3 PM", \
"13:00-15:00", and "from one until three in the afternoon" all mean hours [13, 14]. \
"6 PM" alone as a single instant is normally the start of a window -- read the whole \
sentence to find both endpoints.
- The "hours" list must contain only unique integers from 0 to 23, in ascending order.
- Understand paraphrases, percentages ("80% reduction" = factor 0.2), and different ways of \
naming the same clock time (6 PM = 18:00 = "six in the evening").
- If a relative time expression (e.g. "after sunset", "overnight", "before sunrise") cannot \
be safely and confidently mapped to specific hours using the provided context, do NOT guess \
an hour -- return no_op for that note instead of hallucinating a time window.
- Never invent a directive type that is not in the list above.
- Never invent demand, solar, tariff, or battery numbers that were not given to you.
- If a note is irrelevant, purely administrative, or does not describe an energy-operations \
constraint, mark it no_op. Do not stretch unrelated notes into an energy rule.
- Preserve the operator's intended meaning; do not add restrictions the note does not state.
- Give a short, one-sentence, human-readable explanation for each note.

You must call the provided function/tool to return your answer as structured data. Do not \
reply with plain text."""


def _build_json_schema(num_notes: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "directives": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "note_index": {
                            "type": "integer",
                            "description": "Zero-based index of the operator note this directive belongs to.",
                        },
                        "directive_type": {
                            "type": "string",
                            "enum": ALLOWED_DIRECTIVE_TYPES,
                        },
                        "hours": {
                            "type": ["array", "null"],
                            "items": {"type": "integer"},
                            "description": "Unique ascending hour integers 0-23, or null for no_op.",
                        },
                        "factor": {
                            "type": ["number", "null"],
                            "description": "Usable solar fraction remaining (0-1). Only for solar_reduction, else null.",
                        },
                        "minimum_energy_kwh": {
                            "type": ["number", "null"],
                            "description": "Required minimum battery energy in kWh. Only for minimum_battery_reserve, else null.",
                        },
                        "max_grid_kwh": {
                            "type": ["number", "null"],
                            "description": "Maximum grid import in kWh per hour. Only for max_grid_window, else null.",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "One short sentence explaining the interpretation.",
                        },
                    },
                    "required": [
                        "note_index",
                        "directive_type",
                        "hours",
                        "factor",
                        "minimum_energy_kwh",
                        "max_grid_kwh",
                        "explanation",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["directives"],
        "additionalProperties": False,
    }


def _build_gemini_schema() -> dict:
    # Gemini's native responseSchema is a restricted subset of JSON Schema:
    # upper-case type names and a "nullable" flag instead of type: [x, "null"].
    # Fields are intentionally NOT all marked required (unlike the OpenAI
    # strict-mode schema below) -- Gemini is lenient about omitting fields
    # that don't apply to a given directive_type, and validator.py treats a
    # missing field exactly the same as an explicit null.
    return {
        "type": "OBJECT",
        "properties": {
            "directives": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "note_index": {
                            "type": "INTEGER",
                            "description": "Zero-based index of the operator note this directive belongs to.",
                        },
                        "directive_type": {
                            "type": "STRING",
                            "enum": ALLOWED_DIRECTIVE_TYPES,
                        },
                        "hours": {
                            "type": "ARRAY",
                            "items": {"type": "INTEGER"},
                            "nullable": True,
                            "description": "Unique ascending hour integers 0-23, or null for no_op.",
                        },
                        "factor": {
                            "type": "NUMBER",
                            "nullable": True,
                            "description": "Usable solar fraction remaining (0-1). Only for solar_reduction, else null.",
                        },
                        "minimum_energy_kwh": {
                            "type": "NUMBER",
                            "nullable": True,
                            "description": "Required minimum battery energy in kWh. Only for minimum_battery_reserve, else null.",
                        },
                        "max_grid_kwh": {
                            "type": "NUMBER",
                            "nullable": True,
                            "description": "Maximum grid import in kWh per hour. Only for max_grid_window, else null.",
                        },
                        "explanation": {
                            "type": "STRING",
                            "description": "One short sentence explaining the interpretation.",
                        },
                    },
                    "required": ["note_index", "directive_type", "explanation"],
                },
            }
        },
        "required": ["directives"],
    }


def _call_gemini(payload: dict, model: str, api_key: str, timeout: float) -> list:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": json.dumps(payload)}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": _build_gemini_schema(),
        },
    }
    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    parsed = json.loads(text)
    return parsed.get("directives", [])


def _call_openai_compatible(payload: dict, model: str, api_key: str, base_url: str, timeout: float) -> list:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "directive_interpretation",
                "strict": True,
                "schema": _build_json_schema(len(payload["operator_notes"])),
            },
        },
    }
    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    return parsed.get("directives", [])


def _call_anthropic(payload: dict, model: str, api_key: str, timeout: float) -> list:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    tool_name = "emit_directive_interpretation"
    body = {
        "model": model,
        "max_tokens": 2000,
        "temperature": 0,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": json.dumps(payload)}],
        "tools": [
            {
                "name": tool_name,
                "description": "Return the structured directive interpretation for every operator note.",
                "input_schema": _build_json_schema(len(payload["operator_notes"])),
            }
        ],
        "tool_choice": {"type": "tool", "name": tool_name},
    }
    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    for block in data.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == tool_name:
            return block.get("input", {}).get("directives", [])
    raise LLMError("Anthropic response did not include the expected tool_use block.")


def _call_provider(provider: str, payload: dict, model: str, api_key: str, base_url: str, timeout: float) -> list:
    if provider == "gemini":
        return _call_gemini(payload, model, api_key, timeout)
    if provider == "anthropic":
        return _call_anthropic(payload, model, api_key, timeout)
    return _call_openai_compatible(payload, model, api_key, base_url, timeout)


def _is_retryable(exc: Exception) -> bool:
    """Rate limits, server errors, unknown-model 404s and timeouts are worth
    trying the next model/provider for; anything else (400 bad request,
    401/403 bad key) would fail identically on retry with the same key."""
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        code = exc.response.status_code
        return code == 429 or code == 404 or code >= 500
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    # Malformed/empty response body from the provider (rare, transient).
    return isinstance(exc, (json.JSONDecodeError, KeyError, IndexError, TypeError))


def _build_attempts() -> list:
    """Ordered list of (provider, model, api_key, base_url) attempts from env."""
    attempts = []
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if api_key:
        provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
        base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").strip()
        models = [os.getenv("LLM_MODEL", "gemini-3.5-flash-lite").strip()]
        models += [m.strip() for m in os.getenv("LLM_FALLBACK_MODELS", "").split(",") if m.strip()]
        seen = set()
        for m in models:
            if m and m not in seen:
                seen.add(m)
                attempts.append((provider, m, api_key, base_url))

    fb_key = os.getenv("LLM_FALLBACK_API_KEY", "").strip()
    fb_model = os.getenv("LLM_FALLBACK_MODEL", "").strip()
    if fb_key and fb_model:
        fb_provider = os.getenv("LLM_FALLBACK_PROVIDER", "openai").strip().lower()
        fb_base = os.getenv("LLM_FALLBACK_BASE_URL", "https://api.openai.com/v1").strip()
        attempts.append((fb_provider, fb_model, fb_key, fb_base))
    return attempts


def interpret_notes(operator_notes: list, hours: list, battery: dict) -> list:
    """
    Call the configured LLM provider(s) to interpret operator notes.

    Tries the primary model first, then each LLM_FALLBACK_MODELS entry, then
    the optional secondary provider, stopping at the first usable response
    or when LLM_TOTAL_BUDGET_SECONDS is exhausted.

    Returns the RAW, UNTRUSTED list of directive dicts exactly as produced by
    the model. Callers must pass this straight into validator.validate_directives
    before it ever reaches the optimizer.

    Raises LLMError only when every attempt failed (missing key, quota,
    network/timeout, malformed response) so main.py can apply the safe
    no_op fallback.
    """
    attempts = _build_attempts()
    if not attempts:
        raise LLMError("LLM_API_KEY is not configured.")

    per_call_timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "12"))
    total_budget = float(os.getenv("LLM_TOTAL_BUDGET_SECONDS", "24"))
    started = time.monotonic()

    payload = {
        "operator_notes": [{"note_index": i, "text": n} for i, n in enumerate(operator_notes)],
        "context": {"hours": hours, "battery": battery},
    }

    # Free-tier rate limits are per-minute bursts as well as per-day caps, so
    # after one full pass over the chain fails we pause briefly and try the
    # whole chain once more, as long as the time budget allows.
    retry_pause = float(os.getenv("LLM_RETRY_PAUSE_SECONDS", "2"))
    errors = []
    for round_no in range(2):
        if round_no == 1:
            if total_budget - (time.monotonic() - started) <= retry_pause + 2.0:
                break
            logger.warning("All LLM attempts failed once; pausing %.1fs and retrying the chain.", retry_pause)
            time.sleep(retry_pause)
        for provider, model, api_key, base_url in attempts:
            remaining = total_budget - (time.monotonic() - started)
            if remaining <= 1.0:
                errors.append("time budget exhausted before trying %s/%s" % (provider, model))
                break
            timeout = min(per_call_timeout, remaining)
            try:
                raw = _call_provider(provider, payload, model, api_key, base_url, timeout)
            except LLMError as exc:
                errors.append(f"{provider}/{model}: {exc}")
                continue
            except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
                errors.append(f"{provider}/{model}: {exc}")
                if not _is_retryable(exc):
                    logger.warning("LLM attempt %s/%s failed with a non-retryable error: %s", provider, model, exc)
                    continue
                logger.warning("LLM attempt %s/%s failed (%s); trying next fallback.", provider, model, exc)
                continue

            if not isinstance(raw, list):
                errors.append(f"{provider}/{model}: response did not contain a directives list")
                continue
            if model != attempts[0][1] or provider != attempts[0][0]:
                logger.info("LLM interpretation served by fallback %s/%s.", provider, model)
            return raw

    raise LLMError("All LLM attempts failed: " + " | ".join(errors))
