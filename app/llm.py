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
                           "gemini-2.5-flash".
    LLM_API_KEY            secret key, read only from the environment.
    LLM_BASE_URL           only for provider=openai; override the base URL
                           to point at a non-OpenAI OpenAI-compatible host.
    LLM_TIMEOUT_SECONDS    per-call timeout, default 20s (endpoint budget is
                           30s total, so this leaves headroom for the
                           optimizer + serialization).
"""
import json
import os

import requests

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


def interpret_notes(operator_notes: list, hours: list, battery: dict) -> list:
    """
    Call the configured LLM provider to interpret operator notes.

    Returns the RAW, UNTRUSTED list of directive dicts exactly as produced by
    the model. Callers must pass this straight into validator.validate_directives
    before it ever reaches the optimizer.

    Raises LLMError on any failure (missing key, network/timeout, malformed
    response) so main.py can apply the safe no_op fallback.
    """
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key:
        raise LLMError("LLM_API_KEY is not configured.")

    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    model = os.getenv("LLM_MODEL", "gemini-2.5-flash").strip()
    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").strip()

    payload = {
        "operator_notes": [{"note_index": i, "text": n} for i, n in enumerate(operator_notes)],
        "context": {"hours": hours, "battery": battery},
    }

    try:
        if provider == "gemini":
            raw = _call_gemini(payload, model, api_key, timeout)
        elif provider == "anthropic":
            raw = _call_anthropic(payload, model, api_key, timeout)
        else:
            raw = _call_openai_compatible(payload, model, api_key, base_url, timeout)
    except LLMError:
        raise
    except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"LLM call failed: {exc}") from exc

    if not isinstance(raw, list):
        raise LLMError("LLM response did not contain a directives list.")
    return raw
