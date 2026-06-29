"""UMLS node resolver with LLM-assisted candidate selection.

Resolves entity names against the UMLS REST API using a cascade of search
strategies, then calls an LLM to pick the best candidate when the API returns
more than one result.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).parent.parent / ".env")

_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import lookup_cache as _lc
from llm_client import (  # noqa: E402
    acreate_structured_output as _acreate_structured_output_shared,
)
from lookups import umls_search, sem_group_to_type_names, resolve_sem_type_filter, VERBOSE as _VERBOSE  # noqa: E402

from .base_resolver import NodeResolver, ResolveContext, _log_error  # noqa: F401

_SERVER_ERROR  = object()
_CONFIG_ERROR  = object()  # auth/key error — no retry, no sleep
_WARNED_CONFIG_ERROR: bool = False  # print UMLS config error once per process

# L1 in-process cache: (name_lower, sabs, sem_types_key) → resolved entity dict
_umls_pick_cache: dict[tuple, dict] = {}

_SEARCH_CASCADE = ("words", "normalizedWords", "normalizedString")

_PICK_SYSTEM = (
    "You are a UMLS entity disambiguation expert. "
    "Return the 1-based index of the best matching candidate, "
    "or 0 if no candidate is an acceptable match for the query term."
)

class _CandidatePick(BaseModel):
    index: int = Field(ge=0)  # 1-based; 0 = no acceptable match


async def _acreate_structured_output(
    text_input: str,
    system_prompt: str,
    response_model: type,
    _retries: int = 3,
    _base_delay: float = 1.0,
):
    import os as _os  # noqa: PLC0415
    # Read from os.environ at call time so pipeline_runner's env updates take effect.
    # Do NOT rely on module-level _LLM_MODEL / _LLM_MAX_COMPLETION_TOKENS / _LLM_TIMEOUT
    # which were captured at import time and are never patched by pipeline_runner.
    _model     = _os.environ.get("LLM_MODEL", "gpt-4o").replace("azure/", "")
    _max_tok   = int(_os.environ.get("LLM_MAX_COMPLETION_TOKENS", "8192"))
    _timeout   = float(_os.environ.get("LLM_TIMEOUT", "120"))
    return await _acreate_structured_output_shared(
        text_input, system_prompt, response_model,
        model=_model,
        max_completion_tokens=_max_tok,
        timeout=_timeout,
        retries=_retries,
        base_delay=_base_delay,
        log_prefix="[umls-resolver]",
    )


def _is_server_error(err: str) -> bool:
    low = err.lower()
    return any(k in low for k in ("500", "502", "503", "504", "timeout", "connection", "oserror", "network"))


def _build_candidate_props(candidate: dict) -> dict:
    return {
        "cui":            candidate["cui"],
        "name":           candidate["name"],
        "semantic_types": candidate.get("semantic_types", []),
        "root_source":    candidate.get("root_source", ""),
    }


def _get_all_candidates(
    name: str,
    sabs: str,
    sem_group: str,
    sem_types: list | None,
) -> list | object:
    """Synchronous cascade UMLS search. Returns candidate list or _SERVER_ERROR."""
    tuis_str, group_names = resolve_sem_type_filter(sem_types, sem_group)

    _config_error: str = ""
    for search_type in _SEARCH_CASCADE:
        raw    = umls_search(name, search_type, sabs=sabs, semantic_types=tuis_str, page_size=25)
        parsed = json.loads(raw)
        if "error" in parsed:
            err = parsed["error"]
            if _is_server_error(err):
                return _SERVER_ERROR
            if any(k in err.lower() for k in ("api_key", "api key", "401", "403", "unauthorized", "forbidden")):
                # Auth/config error — same for every search type; bail early and surface loudly
                _config_error = err
                break
            continue
        candidates = parsed.get("results", [])
        if group_names:
            candidates = [
                c for c in candidates
                if set(c.get("semantic_types", [])) & group_names
            ]
        if candidates:
            return candidates

    if _config_error:
        global _WARNED_CONFIG_ERROR
        if not _WARNED_CONFIG_ERROR:
            _WARNED_CONFIG_ERROR = True
            print(f"  [umls-resolver] FATAL: {_config_error} — set UMLS_API_KEY in .env or environment", flush=True)
        return _CONFIG_ERROR  # auth error: fail fast, no sleep, no retry
    return []


async def _llm_pick_best(name: str, candidates: list[dict]) -> dict | None:
    numbered = "\n".join(
        f"  {i + 1}. {c['cui']}: {c['name']} ({', '.join(c.get('semantic_types', []))})"
        for i, c in enumerate(candidates)
    )
    prompt = (
        f"Select the best UMLS match for \"{name}\".\n"
        f"Candidates are ordered by ElasticSearch relevance score (position 1 = highest BM25 score).\n"
        f"BM25 ranking reflects string overlap, not semantic equivalence — verify the selected concept\n"
        f"is semantically equivalent to or a recognised synonym of \"{name}\", not merely a string overlap.\n"
        f"Candidates:\n"
        f"{numbered}\n"
        f"Return the 1-based index of the best match, "
        f"or 0 if no candidate is semantically equivalent to or a recognised synonym of \"{name}\"."
    )
    pick = await _acreate_structured_output(prompt, _PICK_SYSTEM, _CandidatePick)
    if pick.index < 1 or pick.index > len(candidates):
        return None
    return _build_candidate_props(candidates[pick.index - 1])


# ── Candidate gathering + prefilter (no LLM) ──────────────────────────────────

async def _gather_candidates(name: str, node_def: dict, output_dir: Path, label: str) -> list | None:
    """Cascade-search UMLS for one term. Returns candidate list, or None on config/server error.

    No LLM pick happens here — shared pre-pick step for the per-entity and batched paths.
    """
    sabs      = ",".join(node_def.get("umls_vocabs", []))
    sem_group = node_def.get("sem_group", "")
    sem_types = node_def.get("semantic_types") or None

    candidates = await asyncio.to_thread(_get_all_candidates, name, sabs, sem_group, sem_types)
    if candidates is _CONFIG_ERROR:
        _log_error(output_dir, name, label, "UMLS not configured — set UMLS_API_KEY", print_prefix="umls-resolver")
        return None
    if candidates is _SERVER_ERROR:
        await asyncio.sleep(2)
        candidates = await asyncio.to_thread(_get_all_candidates, name, sabs, sem_group, sem_types)
        if candidates is _SERVER_ERROR:
            _log_error(output_dir, name, label, "server error after retry", print_prefix="umls-resolver")
            return None
    return candidates


def _prefilter(name: str, candidates: list, node_def: dict) -> tuple[str, dict | None]:
    """Resolve a term without an LLM where possible.

    Returns ("resolved", props), ("none", None), or ("pick", None).
    """
    if not candidates:
        return ("none", None)
    if len(candidates) == 1:
        return ("resolved", _build_candidate_props(candidates[0]))

    sem_group = node_def.get("sem_group", "")
    sem_types = node_def.get("semantic_types") or None
    _name_lower = name.lower()
    _exact = next((c for c in candidates if c.get("name", "").lower() == _name_lower), None)
    if _exact:
        if sem_group or sem_types:
            _expected = set(sem_types) if sem_types else sem_group_to_type_names(sem_group)
            _actual   = set(_exact.get("semantic_types", []))
            if not _expected or (_actual & _expected):
                return ("resolved", _build_candidate_props(_exact))
        else:
            return ("resolved", _build_candidate_props(_exact))
    return ("pick", None)


def _cache_store(pick_key: tuple, result: dict) -> None:
    """Write a resolved UMLS entity to both L1 and L2 caches.

    Stores a COPY in L1 so the caller's returned dict and the cached entry are
    independent — a caller that later mutates its result (e.g. build_props) cannot
    corrupt the cache for future resolutions. Matches the defensive-copy contract of
    lookup_cache.cached_async(copy=dict) used by the per-entity path.
    """
    _umls_pick_cache[pick_key] = dict(result)
    _lc.put("umls_pick", pick_key, json.dumps(result))


# ── Per-entity resolution (kept for the single-entity path + tests) ───────────

async def _do_resolve_one(
    name: str,
    label: str,
    node_def: dict,
    output_dir: Path,
) -> dict | None:
    candidates = await _gather_candidates(name, node_def, output_dir, label)
    if candidates is None:
        return None
    status, val = _prefilter(name, candidates, node_def)
    if status == "none":
        return None
    if status == "resolved":
        return val
    try:
        result = await _llm_pick_best(name, candidates)
        if result is None:
            _log_error(output_dir, name, label, "no acceptable UMLS match — dropped by LLM", print_prefix="umls-resolver")
        return result
    except Exception as e:
        _log_error(output_dir, name, label, f"LLM pick failed: {e} — dropped", print_prefix="umls-resolver")
        return None


def _umls_pick_key(name: str, node_def: dict) -> tuple:
    sabs      = ",".join(sorted(node_def.get("umls_vocabs", [])))
    sem_types = ",".join(sorted(node_def.get("semantic_types") or []))
    return (name.lower(), sabs, sem_types)


async def _resolve_one(
    name: str,
    label: str,
    node_def: dict,
    output_dir: Path,
) -> dict | None:
    """Cache wrapper around _do_resolve_one. L1 in-process + L2 SQLite."""
    pick_key = _umls_pick_key(name, node_def)

    async def _compute() -> tuple[dict | None, bool]:
        result = await _do_resolve_one(name, label, node_def, output_dir)
        return result, result is not None  # cache only a successful resolution

    return await _lc.cached_async(
        _umls_pick_cache, "umls_pick", pick_key, _compute,
        decode=json.loads, encode=json.dumps, copy=dict,
        verbose=_VERBOSE, label=f"umls_pick({name!r})",
    )


# ── Batched LLM pick ──────────────────────────────────────────────────────────

_BATCH_PICK_SIZE = 12  # entries per LLM pick call (bounds prompt token size)


class _BatchPickEntry(BaseModel):
    entry_id: int
    index: int = Field(ge=0)  # 1-based candidate index; 0 = no acceptable match


class _BatchPickResult(BaseModel):
    picks: list[_BatchPickEntry]


_BATCH_PICK_SYSTEM = (
    "You are a UMLS entity disambiguation expert. For EACH entry, return the 1-based "
    "index of the best matching candidate (or 0 if none is semantically equivalent to "
    "or a recognised synonym of the term), keyed by entry_id. Return exactly one pick "
    "per entry_id."
)


def _format_batch_entry(entry_id: int, name: str, candidates: list[dict]) -> str:
    numbered = "\n".join(
        f"  {i + 1}. {c['cui']}: {c['name']} ({', '.join(c.get('semantic_types', []))})"
        for i, c in enumerate(candidates)
    )
    return f"Entry {entry_id} — term \"{name}\":\n{numbered}"


async def _batch_llm_pick(
    entries: list[tuple[str, str, list[dict]]],
    output_dir: Path,
) -> dict[int, dict | None]:
    """Resolve many ambiguous UMLS terms in as few LLM calls as possible.

    ``entries`` is a list of (name, label, candidates). Returns a map from the
    entry's positional id → resolved props (or None).
    """
    results: dict[int, dict | None] = {}
    for chunk_start in range(0, len(entries), _BATCH_PICK_SIZE):
        chunk = entries[chunk_start : chunk_start + _BATCH_PICK_SIZE]
        blocks = [
            _format_batch_entry(chunk_start + i, name, cands)
            for i, (name, label, cands) in enumerate(chunk)
        ]
        prompt = (
            "Candidates are ordered by ElasticSearch BM25 relevance (position 1 = highest). "
            "BM25 reflects string overlap, not semantic equivalence — verify the selected "
            "concept is semantically equivalent to or a recognised synonym of the term, not "
            "merely a string overlap. Return index 0 if none qualifies.\n\n"
            "Select the best UMLS match for each entry below.\n\n" + "\n\n".join(blocks)
        )
        try:
            res = await _acreate_structured_output(prompt, _BATCH_PICK_SYSTEM, _BatchPickResult)
            picked_ids = {p.entry_id: p for p in res.picks}
        except Exception as e:
            for i, (name, label, cands) in enumerate(chunk):
                _log_error(output_dir, name, label, f"batch LLM pick failed: {e} — dropped", print_prefix="umls-resolver")
                results[chunk_start + i] = None
            continue
        for i, (name, label, cands) in enumerate(chunk):
            eid = chunk_start + i
            p = picked_ids.get(eid)
            if p is not None and 1 <= p.index <= len(cands):
                results[eid] = _build_candidate_props(cands[p.index - 1])
            else:
                _log_error(output_dir, name, label, "no acceptable UMLS match — dropped by LLM", print_prefix="umls-resolver")
                results[eid] = None
    return results


class UMLSResolver(NodeResolver):
    source = "umls"

    async def resolve_batch(
        self,
        entity_names: list[tuple[str, str]],
        nodes: dict,
        output_dir: Path,
        ctx: "ResolveContext",
    ) -> dict[tuple[str, str], dict | None]:
        # ctx accepted for ABC compatibility; UMLS resolution is driven by schema
        # sabs/sem_types, not domain_hint/abbr_map. Kept unused intentionally.
        if not entity_names:
            return {}
        output_dir.mkdir(parents=True, exist_ok=True)
        results: dict[tuple[str, str], dict | None] = {}

        # ── Phase A: per-entity cache check + candidate gather + no-LLM prefilter ──
        async def _phase_a(name: str, label: str):
            node_def = nodes.get(label, {})
            pick_key = _umls_pick_key(name, node_def)
            if pick_key in _umls_pick_cache:
                if _VERBOSE: print(f"  [L1-cache] umls_pick({name!r})")
                return ("done", dict(_umls_pick_cache[pick_key]), pick_key)
            db = _lc.get("umls_pick", pick_key)
            if db is not None:
                if _VERBOSE: print(f"  [L2-cache] umls_pick({name!r})")
                r = json.loads(db)
                _umls_pick_cache[pick_key] = r
                return ("done", dict(r), pick_key)
            candidates = await _gather_candidates(name, node_def, output_dir, label)
            if candidates is None:
                return ("done", None, pick_key)
            status, val = _prefilter(name, candidates, node_def)
            if status == "resolved":
                _cache_store(pick_key, val)
                return ("done", val, pick_key)
            if status == "none":
                return ("done", None, pick_key)
            return ("pick", candidates, pick_key)

        a_results = await asyncio.gather(
            *[_phase_a(name, label) for name, label in entity_names],
            return_exceptions=True,
        )

        needs_pick: list[tuple[str, str, list[dict]]] = []
        pick_meta: list[tuple[tuple[str, str], tuple]] = []
        for (name, label), res in zip(entity_names, a_results):
            if isinstance(res, Exception):
                results[(name, label)] = None
                continue
            status, payload, pick_key = res
            if status == "done":
                results[(name, label)] = payload
            else:
                needs_pick.append((name, label, payload))
                pick_meta.append(((name, label), pick_key))

        # ── Phase B: one (or few) batched LLM calls for the ambiguous remainder ──
        if needs_pick:
            picked = await _batch_llm_pick(needs_pick, output_dir)
            for local_id, ((name, label), pick_key) in enumerate(pick_meta):
                result = picked.get(local_id)
                results[(name, label)] = result
                if result is not None:
                    _cache_store(pick_key, result)

        return results

    def build_props(
        self,
        term: str,
        label: str,
        node_def: dict,
        llm_extras: dict,
        resolved_map: dict[tuple[str, str], dict | None],
    ) -> tuple[dict, dict] | None:
        umls_result = resolved_map.get((term, label))
        if umls_result is None:
            return None
        props: dict = {
            "cui":            umls_result["cui"],
            "name":           umls_result["name"],
            "semantic_types": umls_result.get("semantic_types", []),
        }
        meta: dict = {
            "types":       umls_result.get("semantic_types", []),
            "root_source": umls_result.get("root_source", ""),
        }
        for p in node_def.get("properties", []):
            if p.get("source") == "llm":
                val = llm_extras.get(p["name"])
                if val is None and not p.get("optional"):
                    return None
                if val is not None:
                    props[p["name"]] = val
        return (props, meta)
