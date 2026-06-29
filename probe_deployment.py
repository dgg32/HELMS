#!/usr/bin/env python3
"""Probe Azure OpenAI deployment capabilities.

Strategy:
  1. /models/{id}  → boolean feature flags (fine_tune, chat_completion, …)
  2. Name-based heuristics → reasoning model? supports reasoning_effort?
  3. Live probe     → send max_completion_tokens=999999, read back actual cap
                      from usage.completion_tokens_details or error message

Reads LLM_ENDPOINT, LLM_API_KEY, LLM_MODEL, LLM_API_VERSION from .env.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import os

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# Models known to support reasoning_effort (low/medium/high)
_REASONING_EFFORT_MODELS = re.compile(
    r"^(o1|o3|o4|o3-mini|o4-mini|o1-mini|o1-pro|o3-pro|o3-deep-research|gpt-5\.5)", re.I
)
# Models that use hidden reasoning tokens (need high max_completion_tokens)
_REASONING_TOKEN_MODELS = re.compile(
    r"^(o1|o3|o4|gpt-5\.5|gpt-5\.4(?!-mini)|gpt-5\.3(?!-mini)|gpt-5\.2(?!-mini)|gpt-5\.1(?![-]))",
    re.I,
)


def _base_endpoint(raw: str) -> str:
    return re.sub(r"/openai/deployments/.*$", "", raw.rstrip("/"))


def _deployment_name(raw_model: str) -> str:
    return raw_model.split("/")[-1]


def _get(url: str, api_key: str, api_version: str) -> dict:
    resp = requests.get(
        url,
        params={"api-version": api_version},
        headers={"api-key": api_key},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def _probe_token_cap(base: str, deployment: str, api_key: str, api_version: str) -> dict:
    """Send a minimal chat request with huge max_completion_tokens.

    Returns dict with keys:
      actual_max  – int if we can infer it, else None
      finish_reason
      completion_tokens
      reasoning_tokens
      raw_usage
    """
    url = f"{base}/openai/deployments/{deployment}/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": "Reply with exactly the word: OK"}],
        "max_completion_tokens": 999_999,
    }
    try:
        resp = requests.post(
            url,
            params={"api-version": api_version},
            headers={"api-key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        if resp.status_code == 400:
            # Azure embeds the limit in the error message, e.g.:
            # "This model supports at most 128000 completion tokens"
            err = resp.json()
            msg = json.dumps(err)
            m = re.search(r"at most (\d+)", msg, re.I) or re.search(r"(\d{4,})\s+(?:completion\s+)?tokens", msg, re.I)
            return {
                "actual_max":        int(m.group(1)) if m else None,
                "finish_reason":     "error_400",
                "completion_tokens": None,
                "reasoning_tokens":  None,
                "raw_usage":         err,
            }
        resp.raise_for_status()
        data    = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return {
                "actual_max":        None,
                "finish_reason":     "no_choices",
                "completion_tokens": None,
                "reasoning_tokens":  None,
                "raw_usage":         data,
            }
        choice  = choices[0]
        usage   = data.get("usage", {})
        details = usage.get("completion_tokens_details", {})
        return {
            "actual_max":        None,  # request succeeded → cap > 999999 (unlikely) or not enforced
            "finish_reason":     choice.get("finish_reason"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens":  details.get("reasoning_tokens"),
            "raw_usage":         usage,
        }
    except requests.RequestException as e:
        return {"error": str(e)}


def main() -> None:
    raw_endpoint = os.environ.get("LLM_ENDPOINT", "")
    api_key      = os.environ.get("LLM_API_KEY", "")
    raw_model    = os.environ.get("LLM_MODEL", "")
    api_version  = os.environ.get("LLM_API_VERSION", "2025-04-01-preview")

    if not all([raw_endpoint, api_key, raw_model]):
        sys.exit("Missing LLM_ENDPOINT / LLM_API_KEY / LLM_MODEL in .env")

    base       = _base_endpoint(raw_endpoint)
    deployment = _deployment_name(raw_model)

    print(f"\n{'='*60}")
    print(f"Deployment : {deployment}")
    print(f"Base URL   : {base}")
    print(f"API version: {api_version}")

    # ── 1. Feature flags from /models ────────────────────────────────────────
    print(f"\n--- /models/{deployment} (feature flags) ---")
    try:
        data = _get(f"{base}/openai/models/{deployment}", api_key, api_version)
        caps = data.get("capabilities", {})
        for k, v in caps.items():
            print(f"  {k:30s}: {v}")
        model_id = data.get("id", deployment)
    except requests.HTTPError as e:
        print(f"  HTTP {e.response.status_code}: {e.response.text}")
        model_id = deployment

    # ── 2. Name-based heuristics ─────────────────────────────────────────────
    print(f"\n--- Name-based heuristics (model_id={model_id!r}) ---")
    is_reasoning_effort = bool(_REASONING_EFFORT_MODELS.match(model_id))
    is_reasoning_tokens = bool(_REASONING_TOKEN_MODELS.match(model_id))
    print(f"  supports reasoning_effort param : {is_reasoning_effort}")
    print(f"  uses hidden reasoning_tokens    : {is_reasoning_tokens}")
    if is_reasoning_tokens:
        print("  → set LLM_MAX_COMPLETION_TOKENS high (≥16384) to avoid finish_reason='length'")

    # ── 3. Live probe (actual token cap + reasoning token visibility) ─────────
    print("\n--- Live probe (max_completion_tokens=999999) ---")
    result = _probe_token_cap(base, deployment, api_key, api_version)
    if "error" in result:
        print(f"  probe failed: {result['error']}")
    else:
        if result.get("actual_max"):
            print(f"  max_completion_tokens cap    : {result['actual_max']:,}  ← set LLM_MAX_COMPLETION_TOKENS ≤ this")
        elif result.get("finish_reason") == "error_400":
            print("  Azure returned 400 but cap not parseable — check raw_usage below")
        else:
            print("  max_completion_tokens cap    : > 999,999 (no hard cap enforced by Azure)")
        print(f"  finish_reason                : {result.get('finish_reason')}")
        print(f"  completion_tokens used       : {result.get('completion_tokens')}")
        rt = result.get("reasoning_tokens")
        print(f"  reasoning_tokens (hidden)    : {rt if rt is not None else 'N/A or 0'}")
        print(f"  full usage                   : {json.dumps(result.get('raw_usage'), indent=4)}")

    print()


if __name__ == "__main__":
    main()
