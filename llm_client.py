#!/usr/bin/env python3
"""Shared LLM client factory for all pipeline modules.

All providers are routed through LiteLLM so structured output, retries, and
reasoning-effort handling are uniform across Azure, OpenAI, Anthropic, Gemini,
and Ollama.

Supported providers (set via LLM_PROVIDER in .env.yaml model entry):
  azure     — Azure OpenAI (default when LLM_ENDPOINT is set)
  openai    — OpenAI direct (OPENAI_API_KEY or LLM_API_KEY)
  anthropic — Anthropic Claude (LLM_API_KEY); requires litellm
  ollama    — Ollama local (LLM_ENDPOINT base URL, default http://localhost:11434); requires litellm
  gemini    — Google AI Studio (LLM_API_KEY); requires litellm

Example .env.yaml:
  models:
    gpt-4o:                              # Azure (default)
      LLM_ENDPOINT: https://...
      LLM_API_KEY: <azure-key>
      LLM_API_VERSION: 2024-12-01-preview
    claude-3-5-sonnet-20241022:          # Anthropic
      LLM_PROVIDER: anthropic
      LLM_API_KEY: sk-ant-...
    ollama/llama3.1:                     # Ollama local
      LLM_PROVIDER: ollama
      LLM_ENDPOINT: http://localhost:11434
    gpt-4o-direct:                       # OpenAI direct
      LLM_PROVIDER: openai
      LLM_MODEL: gpt-4o
      LLM_API_KEY: sk-...
    gemini-3.1-flash-lite:               # Google AI Studio
      LLM_PROVIDER: gemini
      LLM_API_KEY: <google-ai-studio-key>
"""
from __future__ import annotations

import os
import re
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

import yaml
from dotenv import load_dotenv

_ENV_YAML = Path(__file__).parent / ".env.yaml"
_models_config: dict[str, dict] = {}  # model_name -> {LLM_ENDPOINT, LLM_API_KEY, ...}


def load_config() -> dict:
    """Load .env.yaml (priority) then .env (dotenv fallback).

    Priority: shell env vars > .env.yaml > .env
    Model entries under ``models:`` are per-model credential dicts.
    ALL_CAPS scalar keys at the top level are loaded as env vars.
    Returns the raw config dict.
    """
    global _models_config
    root = Path(__file__).parent
    cfg: dict = {}

    if _ENV_YAML.exists():
        try:
            with open(_ENV_YAML, encoding="utf-8") as _f:
                cfg = yaml.safe_load(_f) or {}
        except (yaml.YAMLError, OSError, UnicodeDecodeError) as _e:
            raise RuntimeError(
                f"Failed to parse {_ENV_YAML}: {_e}. "
                "Fix the YAML syntax or remove the file to fall back to .env."
            ) from _e
        if not isinstance(cfg, dict):
            raise RuntimeError(
                f"{_ENV_YAML} must contain a top-level mapping, got {type(cfg).__name__}."
            )

        raw_models = cfg.get("models", {})
        _models_config = raw_models if isinstance(raw_models, dict) else {}

        # Load global ALL_CAPS scalar keys as env vars (setdefault = shell wins)
        for _k, _v in cfg.items():
            if _k == "models":
                continue
            if str(_k).upper() == str(_k) and isinstance(_v, (str, int, float)):
                os.environ.setdefault(str(_k), str(_v))

        # Bootstrap defaults from the first model entry
        if _models_config:
            _first_name, _first_cfg = next(iter(_models_config.items()))
            os.environ.setdefault("LLM_MODEL", _first_name.replace("azure/", ""))
            for _k, _v in (_first_cfg or {}).items():
                os.environ.setdefault(str(_k), str(_v))

    # .env fills anything still unset
    load_dotenv(root / ".env")
    return cfg


_config: dict = load_config()


def get_models() -> list[str]:
    """Model names from .env.yaml ``models:`` dict, or [current default]."""
    if _models_config:
        return list(_models_config.keys())
    return [os.environ.get("LLM_MODEL", "gpt-4o")]


def get_model_env(model_name: str) -> dict:
    """Env var overrides for a specific model (endpoint + key + version + name).

    Always emits LLM_PROVIDER so switching providers clears the previous value —
    without this, switching from Gemini to Azure leaves LLM_PROVIDER=gemini in
    os.environ and get_provider() returns the wrong backend.
    """
    env: dict = {"LLM_MODEL": model_name.replace("azure/", "")}
    for k, v in (_models_config.get(model_name) or {}).items():
        env[str(k)] = str(v)
    # Ensure LLM_PROVIDER is always present so runner's _prev_env captures and
    # restores it; default "" lets get_provider() fall through to LLM_ENDPOINT.
    env.setdefault("LLM_PROVIDER", "")
    return env


# Env vars that together identify a provider/model. A temporary model swap clears
# these before applying a new model's creds so nothing leaks across providers
# (e.g. a leftover azure LLM_ENDPOINT into an anthropic call).
_MODEL_ENV_KEYS = ("LLM_PROVIDER", "LLM_API_KEY", "LLM_ENDPOINT", "LLM_API_VERSION", "LLM_MODEL")


@contextmanager
def use_model_env(model_name: str | None):
    """Temporarily run under a different model's full environment, then restore.

    If ``model_name`` is a key in .env.yaml's ``models:`` dict, its COMPLETE
    credential set (provider, API key, endpoint, version, model) is applied, so
    the wrapped call can run on a DIFFERENT PROVIDER than the rest of the
    pipeline. If ``model_name`` is set but not in the config, only ``LLM_MODEL``
    is swapped (same provider, different model name). A falsy ``model_name`` is a
    no-op. The previous environment is always restored on exit, even on error.

    NOT thread-safe: it mutates global ``os.environ`` for the duration of the
    block. Do NOT use it where another thread reads the env concurrently (e.g. the
    semantic check runs in ``asyncio.to_thread`` alongside extraction). For that,
    use ``model_call_params()``, which computes call params without mutating
    global state. Kept for single-threaded/sequential callers.
    """
    if not model_name:
        yield os.environ.get("LLM_MODEL", "gpt-4o")
        return
    full_swap = model_name in _models_config
    overrides = get_model_env(model_name) if full_swap else {
        "LLM_MODEL": model_name.replace("azure/", "")
    }
    touched = set(_MODEL_ENV_KEYS) | set(overrides)
    prev = {k: os.environ.get(k) for k in touched}
    try:
        if full_swap:
            # Clear provider-identifying vars so a model entry that omits one
            # (e.g. no endpoint for anthropic) does not inherit the previous
            # provider's value.
            for k in _MODEL_ENV_KEYS:
                os.environ.pop(k, None)
        for k, v in overrides.items():
            os.environ[str(k)] = str(v)
        yield os.environ.get("LLM_MODEL", model_name)
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _env_num(name: str, default: str, cast):
    """Parse a numeric env var, raising a clear error instead of a bare ValueError.

    A malformed value (e.g. ``LLM_TIMEOUT=sixty``) would otherwise crash any
    module that imports llm_client with an opaque traceback.
    """
    raw = os.environ.get(name, default)
    try:
        return cast(raw)
    except (TypeError, ValueError) as _e:
        raise RuntimeError(
            f"Environment variable {name}={raw!r} is not a valid "
            f"{cast.__name__} (default {default!r})."
        ) from _e


# Per-call HTTP timeout in seconds.
LLM_TIMEOUT: float = _env_num("LLM_TIMEOUT", "120", float)

# Max completion tokens for structured-output calls.
LLM_MAX_COMPLETION_TOKENS: int = _env_num("LLM_MAX_COMPLETION_TOKENS", "8192", int)

# Reasoning effort for reasoning models (o1, o3, GPT-5.x).
LLM_REASONING_EFFORT: str = os.environ.get("LLM_REASONING_EFFORT", "low")

# Process-wide flag: cleared the first time any deployment rejects reasoning_effort.
_reasoning_effort_ok: bool = True


_litellm_logging_quieted: bool = False


def quiet_litellm_logging_worker() -> None:
    """Neutralise litellm's background async LoggingWorker (idempotent).

    On every completion, litellm calls ``GLOBAL_LOGGING_WORKER.ensure_initialized_
    and_enqueue(...)``, which spawns a ``_worker_loop`` task on the *current* event
    loop to run its async success/failure callbacks. We drive litellm from sync
    calls inside short-lived per-run event loops (e.g. the smart-retry worker
    thread), so that loop is closed while the worker task is still pending — Python
    then prints a noisy "Task was destroyed but it is pending! / RuntimeError:
    Event loop is closed" traceback at teardown, even though the run succeeded.

    We configure no async litellm callbacks, so the worker has no real work. Replace
    its enqueue method with one that simply closes the passed coroutine (so it is
    not leaked as "coroutine was never awaited") and never starts a background task.
    Guarded + idempotent so a litellm version that lacks the worker is a no-op.
    """
    global _litellm_logging_quieted
    if _litellm_logging_quieted:
        return
    try:
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

        def _drop_async_log(*args, **kwargs):
            import asyncio as _a
            for c in (*args, *kwargs.values()):
                if _a.iscoroutine(c):
                    try:
                        c.close()
                    except Exception:
                        pass

        GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue = _drop_async_log  # type: ignore[assignment]
        _litellm_logging_quieted = True
    except Exception:
        # Different litellm version / internal layout — nothing to quiet.
        _litellm_logging_quieted = True


def is_reasoning_effort_ok() -> bool:
    return _reasoning_effort_ok


def clear_reasoning_effort() -> None:
    global _reasoning_effort_ok
    _reasoning_effort_ok = False


def get_provider(env: Mapping | None = None) -> str:
    """LLM provider: azure | openai | openai_compatible | anthropic | ollama | gemini.

    Reads from ``env`` (defaults to ``os.environ``). ``openai_compatible`` is for
    OpenAI-shaped gateways (OpenCode Go, OpenRouter, vLLM, …): set
    ``LLM_PROVIDER: openai_compatible`` with a model's own ``LLM_ENDPOINT`` (base
    URL) and ``LLM_API_KEY``.
    """
    e = os.environ if env is None else env
    prov = e.get("LLM_PROVIDER", "")
    if prov:
        return prov.lower()
    if e.get("LLM_ENDPOINT", ""):
        return "azure"
    return "openai"


def _base_url(env: Mapping | None = None) -> str:
    e = os.environ if env is None else env
    endpoint = e.get("LLM_ENDPOINT", "")
    return re.sub(r"/openai(?:/deployments/[^/]+)?/?$", "", endpoint.rstrip("/"))


def _deployment(env: Mapping | None = None) -> str:
    e = os.environ if env is None else env
    return e.get("LLM_MODEL", "gpt-4o").replace("azure/", "")


def _get_litellm_model(model: str | None = None, env: Mapping | None = None) -> str:
    """Full model string for LiteLLM across all providers (reads ``env``)."""
    e = os.environ if env is None else env
    provider = get_provider(e)
    if model is None:
        model = e.get("LLM_MODEL", "gpt-4o")
    for prefix in ("azure/", "anthropic/", "ollama/", "gemini/", "openai/"):
        if model.startswith(prefix):
            model = model[len(prefix):]
            break
    if provider == "azure":
        return f"azure/{model}"
    if provider == "anthropic":
        return f"anthropic/{model}"
    if provider == "ollama":
        return f"ollama/{model}"
    if provider == "gemini":
        return f"gemini/{model}"
    if provider == "openai_compatible":
        # OpenAI-compatible gateway (OpenCode Go, OpenRouter, vLLM, any
        # /v1/chat/completions endpoint): keep the openai/ prefix so LiteLLM
        # honours the api_base supplied by _litellm_kwargs.
        return f"openai/{model}"
    return model  # openai: bare model name → api.openai.com


def _ensure_litellm_env() -> None:
    """Map HELMS env vars to provider-specific env vars expected by LiteLLM.

    Clears all known provider keys first so old values from a previous
    provider/model aren't leaked into the current call.
    """
    for _k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AZURE_API_KEY", "GEMINI_API_KEY", "OLLAMA_API_BASE"):
        os.environ.pop(_k, None)
    provider = get_provider()
    api_key = os.environ.get("LLM_API_KEY", "")
    if provider == "anthropic" and api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
    elif provider in ("openai", "openai_compatible") and api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    elif provider == "ollama":
        endpoint = os.environ.get("LLM_ENDPOINT", "http://localhost:11434")
        os.environ["OLLAMA_API_BASE"] = endpoint
    elif provider == "gemini" and api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    elif provider == "azure" and api_key:
        os.environ["AZURE_API_KEY"] = api_key


def _litellm_kwargs(env: Mapping | None = None) -> dict:
    """Provider-specific kwargs passed directly to litellm.(a)completion (reads ``env``)."""
    e = os.environ if env is None else env
    provider = get_provider(e)
    if provider == "azure":
        return {
            "api_base":    _base_url(e),
            "api_key":     e.get("LLM_API_KEY", ""),
            "api_version": e.get("LLM_API_VERSION", "2024-12-01-preview"),
        }
    if provider == "openai":
        return {"api_key": e.get("LLM_API_KEY") or e.get("OPENAI_API_KEY", "")}
    if provider == "openai_compatible":
        # Gateway behind an OpenAI-shaped /v1/chat/completions API. LiteLLM POSTs
        # to {api_base}/chat/completions with the supplied key.
        return {
            "api_key":  e.get("LLM_API_KEY", ""),
            "api_base": e.get("LLM_ENDPOINT", "").rstrip("/"),
        }
    if provider in ("anthropic", "gemini"):
        return {"api_key": e.get("LLM_API_KEY", "")}
    if provider == "ollama":
        return {"api_base": e.get("LLM_ENDPOINT", "http://localhost:11434")}
    return {}


def model_call_params(model_name: str | None) -> tuple[str, dict]:
    """``(litellm_model_string, call_kwargs)`` for ``model_name``, computed WITHOUT
    mutating ``os.environ`` — thread-safe. Pass the returned kwargs straight to
    ``litellm.completion``; concurrent calls routing other models cannot interfere.

    This is the thread-safe alternative to ``use_model_env``. The semantic check
    runs in ``asyncio.to_thread`` concurrently with extraction (which reads the
    same global env), so an env-mutating swap there would corrupt a concurrent
    extraction call. Computing params into a local dict avoids that entirely.

    - empty/None ⇒ the current process environment's model + kwargs.
    - a name in ``.env.yaml`` ``models:`` ⇒ that model's full provider/model/key/
      base; the model-identifying keys come ONLY from the entry, so a stale
      endpoint from another provider cannot leak in.
    - a bare name not in the config ⇒ current provider/creds, model id swapped.
    """
    if not model_name:
        return _get_litellm_model(), _litellm_kwargs()
    if model_name in _models_config:
        # Effective env = base with the model-identifying keys dropped, then the
        # entry's overrides layered on — same semantics as the old full env swap
        # but in a LOCAL dict that never touches os.environ.
        eff = {k: v for k, v in os.environ.items() if k not in _MODEL_ENV_KEYS}
        eff.update(get_model_env(model_name))
        return _get_litellm_model(env=eff), _litellm_kwargs(env=eff)
    return _get_litellm_model(model_name), _litellm_kwargs()


async def acreate_structured_output(
    text_input: str,
    system_prompt: str,
    response_model: type,
    model: str | None = None,
    max_completion_tokens: int = 8192,
    timeout: float = 120.0,
    retries: int = 3,
    base_delay: float = 1.0,
    log_prefix: str | None = None,
):
    """Shared async structured-output retry loop — all providers via LiteLLM.

    model      : deployment/model name (azure/ prefix stripped automatically)
    log_prefix : if set, prints attempt/retry messages; if None, silent
    """
    return await _acreate_litellm(
        text_input, system_prompt, response_model,
        model, max_completion_tokens, timeout, retries, base_delay, log_prefix,
    )


async def _acreate_litellm(
    text_input: str,
    system_prompt: str,
    response_model: type,
    model: str | None,
    max_completion_tokens: int,
    timeout: float,
    retries: int,
    base_delay: float,
    log_prefix: str | None,
):
    """LiteLLM-backed structured output for all providers.

    Passes response_format=response_model so litellm uses native schema enforcement
    where available (Azure, OpenAI, Gemini) and falls back to json_object mode for
    providers that don't support it (some Ollama models). Schema hint injected into
    the system prompt as a belt-and-suspenders fallback for the latter case.
    """
    import asyncio as _asyncio
    import json as _json
    try:
        import litellm as _litellm
        _litellm.suppress_debug_info = True
        quiet_litellm_logging_worker()
    except ImportError:
        raise ImportError(
            "litellm is required: pip install litellm\n"
            f"  Current provider: {get_provider()}"
        )
    _litellm_auth_errors: tuple = (
        _litellm.exceptions.AuthenticationError,
        _litellm.exceptions.BadRequestError,
    )
    _litellm_rate_errors: tuple = (_litellm.exceptions.RateLimitError,)

    _ensure_litellm_env()
    full_model = _get_litellm_model(model)
    provider_kwargs = _litellm_kwargs()

    # Schema hint in system prompt helps providers that fall back to json_object mode
    # (e.g. small Ollama models) produce the right structure.
    schema_hint = _json.dumps(response_model.model_json_schema(), indent=None)
    augmented_system = (
        f"{system_prompt}\n\n"
        f"Respond with valid JSON matching this schema exactly:\n{schema_hint}"
    )

    last_exc: Exception | None = None
    attempt = 0
    while attempt < retries:
        _reasoning_effort = LLM_REASONING_EFFORT if is_reasoning_effort_ok() else ""
        effort_kwargs = {"reasoning_effort": _reasoning_effort} if _reasoning_effort else {}
        try:
            resp = await _asyncio.wait_for(
                _litellm.acompletion(
                    model=full_model,
                    messages=[
                        {"role": "system", "content": augmented_system},
                        {"role": "user",   "content": text_input},
                    ],
                    response_format=response_model,
                    max_tokens=max_completion_tokens,
                    **provider_kwargs,
                    **effort_kwargs,
                ),
                timeout=timeout,
            )
            if not resp.choices:
                raise ValueError(f"litellm returned empty choices for model {full_model!r}")
            content = (resp.choices[0].message.content or "").strip()
            return response_model.model_validate_json(content)
        except _asyncio.TimeoutError:
            raise
        except Exception as e:
            # reasoning_effort recovery must run before fatal classification —
            # providers reject an unsupported reasoning_effort with BadRequestError,
            # which is also in _litellm_auth_errors; checking it first prevents that
            # fatal branch from shadowing the auto-disable retry.
            if _reasoning_effort and "reasoning_effort" in str(e).lower():
                if log_prefix:
                    print(
                        f"  {log_prefix} reasoning_effort={_reasoning_effort!r} rejected"
                        " — retrying without it.",
                        flush=True,
                    )
                clear_reasoning_effort()
                continue  # retry immediately without reasoning_effort
            if isinstance(e, _litellm_auth_errors):
                raise  # 401/403/400 — fatal, don't retry
            last_exc = e
            attempt += 1
            if attempt < retries:
                delay = base_delay * (2 ** (attempt - 1))
                if isinstance(e, _litellm_rate_errors):
                    try:
                        if hasattr(e, "response") and e.response is not None:
                            hdr = e.response.headers.get("retry-after")
                            if hdr is not None:
                                delay = float(hdr)
                    except Exception:
                        pass
                    if log_prefix:
                        print(
                            f"  {log_prefix} rate-limited (attempt {attempt}/{retries})."
                            f" Retrying in {delay:.1f}s…",
                            flush=True,
                        )
                elif log_prefix:
                    print(
                        f"  {log_prefix} attempt {attempt} failed: {e}."
                        f" Retrying in {delay:.1f}s…",
                        flush=True,
                    )
                await _asyncio.sleep(delay)
    if last_exc is None:
        raise RuntimeError(f"_acreate_litellm called with retries={retries}; must be >= 1")
    raise last_exc


def completion_structured(
    *,
    model: str,
    system_prompt: str,
    user_msg: str,
    response_model: type,
    max_completion_tokens: int,
    timeout: float = 120.0,
    call_kwargs: dict | None = None,
    retries: int = 3,
    base_delay: float = 2.0,
    log_prefix: str | None = None,
):
    """Synchronous structured-output completion with EXPLICIT model + provider creds.

    Sync sibling of ``_acreate_litellm`` for callers that have already resolved their
    model and provider credentials into explicit kwargs (e.g. the semantic-check
    grader via ``model_call_params``) and must NOT mutate ``os.environ``: this helper
    never touches ``os.environ`` or the env-reading resolvers, so it is safe on the
    concurrent grader path that runs in ``asyncio.to_thread`` alongside extraction
    (DESIGN_INVARIANTS #1). Pass the fully-qualified ``model`` (e.g. ``anthropic/x``)
    and any creds in ``call_kwargs`` (e.g. ``{"api_key": ..., "api_base": ...}``).

    Same retry semantics as ``_acreate_litellm``: reasoning_effort auto-recovery,
    rate-limit ``Retry-After`` parsing, and fatal auth/bad-request classification.
    Returns a validated ``response_model`` instance.
    """
    import json as _json
    import time as _time
    try:
        import litellm as _litellm
        _litellm.suppress_debug_info = True
        quiet_litellm_logging_worker()
    except ImportError:
        raise ImportError("litellm is required: pip install litellm")

    # Tolerate a stub/fake litellm without an `exceptions` namespace (test doubles):
    # an empty tuple makes every isinstance() check below False, so classification
    # degrades to "retry everything" rather than crashing on attribute access.
    _exc = getattr(_litellm, "exceptions", None)
    _auth_errors = tuple(
        e for e in (
            getattr(_exc, "AuthenticationError", None),
            getattr(_exc, "BadRequestError", None),
        ) if isinstance(e, type)
    )
    _rate_errors = tuple(
        e for e in (getattr(_exc, "RateLimitError", None),) if isinstance(e, type)
    )

    call_kwargs = call_kwargs or {}
    schema_hint = _json.dumps(response_model.model_json_schema(), indent=None)
    augmented_system = (
        f"{system_prompt}\n\n"
        f"Respond with valid JSON matching this schema exactly:\n{schema_hint}"
    )

    last_exc: Exception | None = None
    attempt = 0
    while attempt < retries:
        _reasoning_effort = LLM_REASONING_EFFORT if is_reasoning_effort_ok() else ""
        effort_kwargs = {"reasoning_effort": _reasoning_effort} if _reasoning_effort else {}
        try:
            completion = _litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": augmented_system},
                    {"role": "user",   "content": user_msg},
                ],
                response_format=response_model,
                max_tokens=max_completion_tokens,
                timeout=timeout,
                **call_kwargs,
                **effort_kwargs,
            )
            if not completion.choices:
                raise ValueError(f"litellm returned empty choices for model {model!r}")
            content = (completion.choices[0].message.content or "").strip()
            return response_model.model_validate_json(content)
        except Exception as e:
            # reasoning_effort recovery must precede fatal classification — an
            # unsupported reasoning_effort is rejected with BadRequestError, which is
            # also in _auth_errors; checking it first prevents the fatal branch from
            # shadowing the auto-disable retry (mirrors _acreate_litellm).
            if _reasoning_effort and "reasoning_effort" in str(e).lower():
                if log_prefix:
                    print(
                        f"  {log_prefix} reasoning_effort={_reasoning_effort!r} rejected"
                        " — retrying without it.",
                        flush=True,
                    )
                clear_reasoning_effort()
                continue  # retry immediately without reasoning_effort
            if _auth_errors and isinstance(e, _auth_errors):
                raise  # 401/403/400 — fatal, don't retry
            last_exc = e
            attempt += 1
            if attempt < retries:
                delay = base_delay * (2 ** (attempt - 1))
                if _rate_errors and isinstance(e, _rate_errors):
                    try:
                        if getattr(e, "response", None) is not None:
                            hdr = e.response.headers.get("retry-after")
                            if hdr is not None:
                                delay = float(hdr)
                    except Exception:
                        pass
                    if log_prefix:
                        print(
                            f"  {log_prefix} rate-limited (attempt {attempt}/{retries})."
                            f" Retrying in {delay:.1f}s…",
                            flush=True,
                        )
                elif log_prefix:
                    print(
                        f"  {log_prefix} attempt {attempt} failed: {e}."
                        f" Retrying in {delay:.1f}s…",
                        flush=True,
                    )
                _time.sleep(delay)
    if last_exc is None:
        raise RuntimeError(f"completion_structured called with retries={retries}; must be >= 1")
    raise last_exc
