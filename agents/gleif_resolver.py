"""GLEIF node resolver with LLM-assisted candidate selection.

Resolves legal entity names against the GLEIF REST API using a broad candidate
gather (exact + names + fuzzy + WHO OWNS parents), then calls an LLM to pick
the best match — mirroring the UMLS resolver pattern.

``gleif_get_candidates`` collects every plausible candidate including the direct
GLEIF parent of any subsidiary-looking result, so the LLM can choose Hon Hai
for "Foxconn", Taiwan Semiconductor for "TSMC", TDK Corporation for "TDK", etc.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).parent.parent / ".env")

_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json as _json

import lookup_cache as _lc
from llm_client import acreate_structured_output as _acreate_structured_output_shared  # noqa: E402
from lookups import gleif_get_candidates, VERBOSE as _VERBOSE  # noqa: E402

from .base_resolver import NodeResolver, ResolveContext, _log_error  # noqa: F401

# L1 in-process cache: (name_lower, domain_hint_prefix) → resolved entity dict
_gleif_pick_cache: dict[tuple, dict] = {}


async def _acreate_structured_output(text_input, system_prompt, response_model):
    import os as _os
    _model   = _os.environ.get("LLM_MODEL", "gpt-4o").replace("azure/", "")
    _max_tok = int(_os.environ.get("LLM_MAX_COMPLETION_TOKENS", "8192"))
    _timeout = float(_os.environ.get("LLM_TIMEOUT", "120"))
    return await _acreate_structured_output_shared(
        text_input, system_prompt, response_model,
        model=_model,
        max_completion_tokens=_max_tok,
        timeout=_timeout,
        retries=3,
        base_delay=1.0,
        log_prefix="[gleif-resolver]",
    )


# ── Abbreviation expansion ────────────────────────────────────────────────────

# Common unambiguous abbreviations — skip LLM expansion for these.
# Keys: all-caps, values: full official legal name for GLEIF search.
_ABBREV_TABLE: dict[str, str] = {
    "NVDA":  "NVIDIA Corporation",
    "QCOM":  "Qualcomm Incorporated",
    "INTC":  "Intel Corporation",
    "AVGO":  "Broadcom Inc.",
    "ASML":  "ASML Holding N.V.",
    "HPE":   "Hewlett Packard Enterprise Company",
    "IBM":   "International Business Machines Corporation",
    "AMD":   "Advanced Micro Devices, Inc.",
    "AAPL":  "Apple Inc.",
    "MSFT":  "Microsoft Corporation",
    "AMZN":  "Amazon.com, Inc.",
    "GOOGL": "Alphabet Inc.",
    "META":  "Meta Platforms, Inc.",
    "SAP":   "SAP SE",
    "TDK":   "TDK Corporation",
    "ABBV":  "AbbVie Inc.",
    "PFE":   "Pfizer Inc.",
    "JNJ":   "Johnson & Johnson",
    "MRK":   "Merck & Co., Inc.",
    "LLY":   "Eli Lilly and Company",
    "BMY":   "Bristol-Myers Squibb Company",
    "AMGN":  "Amgen Inc.",
    "AZN":   "AstraZeneca PLC",
}

_EXPAND_SYSTEM = (
    "You are a company name expert. Return the full official legal company name "
    "for the given abbreviation or short name, exactly as it would appear on legal documents. "
    "Return an empty string if the expansion is unknown or genuinely ambiguous."
)


class _AbbrevExpansion(BaseModel):
    full_name: str = Field(default="")  # empty string if unknown/ambiguous


async def _expand_abbreviation(abbrev: str, domain_hint: str = "") -> str | None:
    """Expand a short all-caps abbreviation to its full legal company name via LLM."""
    domain_block = f"Document domain context: {domain_hint[:200]}\n\n" if domain_hint else ""
    prompt = (
        f"{domain_block}"
        f"Full official legal company name for the abbreviation '{abbrev}':\n"
        f"(e.g. HPE → Hewlett Packard Enterprise Company, "
        f"IBM → International Business Machines Corporation, "
        f"TSMC → Taiwan Semiconductor Manufacturing Company Limited)"
    )
    try:
        result = await _acreate_structured_output(prompt, _EXPAND_SYSTEM, _AbbrevExpansion)
        name = (result.full_name or "").strip()
        return name if name and name.lower() != abbrev.lower() else None
    except Exception:
        return None


# ── LLM pick ─────────────────────────────────────────────────────────────────

_PICK_SYSTEM = (
    "You are a GLEIF legal entity disambiguation expert. "
    "Return the 1-based index of the best matching candidate, "
    "or 0 if no candidate is an acceptable match for the query term."
)


class _CandidatePick(BaseModel):
    index: int = Field(ge=0)  # 1-based; 0 = no acceptable match
    retry_search_term: str = Field(default="")  # when index=0, suggest a fuller name to re-search


def _strip_match_type(candidate: dict) -> dict:
    """Drop the internal ``match_type`` provenance key before returning a resolved node."""
    return {k: v for k, v in candidate.items() if k != "match_type"}


def _format_candidates(candidates: list[dict]) -> str:
    lines = []
    for i, c in enumerate(candidates, 1):
        lines.append(
            f"  {i}. LEI: {c.get('lei')} | Name: {c.get('name')} "
            f"| Category: {c.get('category')} | Status: {c.get('registration_status')} "
            f"| Jurisdiction: {c.get('jurisdiction')} | Found by: {c.get('match_type')}"
        )
    return "\n".join(lines)


async def _llm_pick_best(
    name: str,
    candidates: list[dict],
    node_def: dict,
    domain_hint: str = "",
) -> tuple[dict | None, str]:
    sem_types = node_def.get("semantic_types") or []
    type_hint = f"  The entity should be of type: {', '.join(sem_types)}.\n" if sem_types else ""
    domain_block = (
        f"Document context (use to prefer geographically/industry-relevant candidates):\n"
        f"{domain_hint[:500]}\n\n"
    ) if domain_hint else ""
    prompt = (
        f"Select the best GLEIF legal entity match for \"{name}\".\n"
        f"{domain_block}"
        f"{type_hint}"
        f"Rules:\n"
        f"  - Prefer ISSUED over LAPSED registration status.\n"
        f"  - Prefer the actual parent/ultimate company over subsidiaries.\n"
        f"  - Candidates with Found-by='who_owns' are the DIRECT PARENT of a matched "
        f"subsidiary — these are often the correct answer for trade names and abbreviations "
        f"(e.g. 'Foxconn' → Hon Hai, 'TSMC' → Taiwan Semiconductor).\n"
        f"  - A candidate with the search term as a strict prefix followed only by a "
        f"jurisdiction, product line, or subsidiary indicator is likely a subsidiary, not "
        f"the main company.\n"
        f"  - Use the document context to break ties: prefer candidates whose jurisdiction "
        f"and industry align with the document's domain (e.g. Taiwan for electronics/ODM, "
        f"Japan for components, USA for hyperscalers).\n"
        f"Candidates:\n"
        f"{_format_candidates(candidates)}\n"
        f"Return the 1-based index of the best match, or 0 if no candidate is acceptable.\n"
        f"When returning 0, also set retry_search_term to the full official company name "
        f"you believe is correct (e.g. 'Murata' → 'Murata Manufacturing Co., Ltd.', "
        f"'HPE' → 'Hewlett Packard Enterprise Company') so a second GLEIF search can be attempted. "
        f"Leave retry_search_term empty only if you have no confident expansion."
    )
    pick = await _acreate_structured_output(prompt, _PICK_SYSTEM, _CandidatePick)
    if pick.index < 1 or pick.index > len(candidates):
        return None, pick.retry_search_term or ""
    return _strip_match_type(candidates[pick.index - 1]), ""


# ── Candidate gathering + prefilter (no LLM) ──────────────────────────────────

async def _gather_candidates(
    name: str, node_def: dict, domain_hint: str = "", abbr_map: "dict[str, str] | None" = None
) -> list[dict]:
    """Collect GLEIF candidates for one term, expanding short all-caps abbreviations.

    No LLM pick happens here — this is the shared pre-pick step used by both the
    per-entity path (_do_resolve_one) and the batched path (resolve_batch).
    """
    candidates = await asyncio.to_thread(gleif_get_candidates, name)

    # Abbreviations (short all-caps) may not appear verbatim in GLEIF — expand and re-search.
    # Skip expansion when we already have a strong GENERAL+ISSUED candidate (no LLM call needed).
    if name.isupper() and len(name) <= 6:
        _has_strong = any(
            c.get("category") == "GENERAL" and c.get("registration_status") == "ISSUED"
            for c in candidates
        )
        if not _has_strong:
            # Expansion source order (cheapest + most faithful first):
            #   1. abbr_map — the document defined this abbreviation inline; deterministic, no LLM.
            #   2. _ABBREV_TABLE — static table of common well-known abbreviations; no LLM.
            #   3. _expand_abbreviation — LLM fallback when the doc/table don't know it.
            expanded = (
                (abbr_map or {}).get(name.upper())
                or _ABBREV_TABLE.get(name.upper())
                or await _expand_abbreviation(name, domain_hint)
            )
            if expanded:
                print(f"  [gleif-resolver] expanding abbreviation {name!r} → {expanded!r}", flush=True)
                extra = await asyncio.to_thread(gleif_get_candidates, expanded)
                seen_leis = {c.get("lei") for c in candidates}
                for c in extra:
                    if c.get("lei") not in seen_leis:
                        candidates.append(c)
                        seen_leis.add(c.get("lei"))
    return candidates


def _prefilter(name: str, candidates: list[dict]) -> tuple[str, dict | None]:
    """Resolve a term without an LLM where possible.

    Returns ("resolved", props) on single-candidate or exact-name match,
    ("none", None) when there are no candidates, or ("pick", None) when an LLM
    disambiguation is required.
    """
    if not candidates:
        return ("none", None)
    if len(candidates) == 1:
        return ("resolved", _strip_match_type(candidates[0]))
    name_lower = name.lower()
    exact = next(
        (c for c in candidates
         if c.get("name", "").lower() == name_lower
         and c.get("registration_status") == "ISSUED"),
        None,
    ) or next(
        (c for c in candidates if c.get("name", "").lower() == name_lower),
        None,
    )
    if exact:
        return ("resolved", _strip_match_type(exact))
    return ("pick", None)


def _pick_cache_key(
    name: str, domain_hint: str = "", abbr_map: "dict[str, str] | None" = None
) -> tuple:
    """Pick cache key. Includes this name's abbr_map expansion: abbr_map is
    per-document and decides the resolved entity (doc A: HPE=Health Plan East,
    doc B: HPE=Hewlett Packard), so omitting it poisons picks across docs that
    reuse the same token. Only THIS name's entry matters (expansion checks
    abbr_map.get(name.upper()) first), so fold just that in — not the whole map."""
    expansion = (abbr_map or {}).get(name.upper(), "")
    return (name.lower(), (domain_hint or "")[:100], expansion)


def _cache_store(pick_key: tuple, result: dict) -> None:
    """Write a resolved GLEIF entity to both L1 and L2 caches.

    Stores a COPY in L1 so the caller's returned dict and the cached entry are
    independent — a caller that later mutates its result (e.g. build_props) cannot
    corrupt the cache for future resolutions. Matches the defensive-copy contract of
    lookup_cache.cached_async(copy=dict) used by the per-entity path.
    """
    _gleif_pick_cache[pick_key] = dict(result)
    _lc.put("gleif_pick", pick_key, _json.dumps(result))


# ── Per-entity resolution (kept for the single-entity path + tests) ───────────

async def _do_resolve_one(
    name: str,
    label: str,
    node_def: dict,
    output_dir: Path,
    domain_hint: str = "",
    abbr_map: "dict[str, str] | None" = None,
) -> dict | None:
    candidates = await _gather_candidates(name, node_def, domain_hint, abbr_map)
    status, val = _prefilter(name, candidates)
    if status == "none":
        return None
    if status == "resolved":
        return val

    # LLM picks from full candidate set
    try:
        result, retry_term = await _llm_pick_best(name, candidates, node_def, domain_hint)
        if result is not None:
            return result
        # LLM found no match — retry with suggested full name if provided
        if retry_term and retry_term.lower() != name.lower():
            print(f"  [gleif-resolver] retrying {name!r} with suggested term {retry_term!r}", flush=True)
            retry_cands = await asyncio.to_thread(gleif_get_candidates, retry_term)
            seen_leis = {c.get("lei") for c in candidates}
            new_cands = [c for c in retry_cands if c.get("lei") not in seen_leis]
            if new_cands:
                result2, _ = await _llm_pick_best(name, new_cands, node_def, domain_hint)
                if result2 is not None:
                    return result2
        _log_error(
            output_dir, name, label,
            "no acceptable GLEIF match — dropped by LLM",
            print_prefix="gleif-resolver",
        )
        return None
    except Exception as e:
        _log_error(output_dir, name, label, f"LLM pick failed: {e} — dropped", print_prefix="gleif-resolver")
        return None


async def _resolve_one(
    name: str,
    label: str,
    node_def: dict,
    output_dir: Path,
    domain_hint: str = "",
    abbr_map: "dict[str, str] | None" = None,
) -> dict | None:
    """Cache wrapper around _do_resolve_one. L1 in-process + L2 SQLite."""
    pick_key = _pick_cache_key(name, domain_hint, abbr_map)

    async def _compute() -> tuple[dict | None, bool]:
        result = await _do_resolve_one(name, label, node_def, output_dir, domain_hint, abbr_map)
        return result, result is not None  # cache only a successful resolution

    return await _lc.cached_async(
        _gleif_pick_cache, "gleif_pick", pick_key, _compute,
        decode=_json.loads, encode=_json.dumps, copy=dict,
        verbose=_VERBOSE, label=f"gleif_pick({name!r})",
    )


# ── Batched LLM pick ──────────────────────────────────────────────────────────

_BATCH_PICK_SIZE = 12  # entries per LLM pick call (bounds prompt token size)


class _BatchPickEntry(BaseModel):
    entry_id: int
    index: int = Field(ge=0)               # 1-based candidate index; 0 = no acceptable match
    retry_search_term: str = Field(default="")


class _BatchPickResult(BaseModel):
    picks: list[_BatchPickEntry]


_BATCH_PICK_SYSTEM = (
    "You are a GLEIF legal entity disambiguation expert. For EACH entry, return the "
    "1-based index of the best matching candidate (or 0 if none is acceptable), keyed "
    "by entry_id. Return exactly one pick per entry_id."
)


def _format_batch_entry(entry_id: int, name: str, candidates: list[dict], node_def: dict) -> str:
    sem_types = node_def.get("semantic_types") or []
    type_hint = f" (expected type: {', '.join(sem_types)})" if sem_types else ""
    return (
        f"Entry {entry_id} — term \"{name}\"{type_hint}:\n"
        f"{_format_candidates(candidates)}"
    )


async def _batch_llm_pick(
    entries: list[tuple[str, str, dict, list[dict]]],
    domain_hint: str,
    output_dir: Path,
) -> dict[int, dict | None]:
    """Resolve many ambiguous terms in as few LLM calls as possible.

    ``entries`` is a list of (name, label, node_def, candidates). Returns a map
    from the entry's positional id → resolved props (or None). Handles the
    index=0 retry-term path with a single follow-up batched call.
    """
    domain_block = (
        f"Document context (prefer geographically/industry-relevant candidates):\n"
        f"{domain_hint[:500]}\n\n"
    ) if domain_hint else ""
    rules = (
        "Rules for every entry:\n"
        "  - Prefer ISSUED over LAPSED registration status.\n"
        "  - Prefer the actual parent/ultimate company over subsidiaries.\n"
        "  - Candidates with Found-by='who_owns' are the DIRECT PARENT of a matched "
        "subsidiary — often correct for trade names/abbreviations (e.g. 'Foxconn' → Hon Hai).\n"
        "  - A candidate that is the search term plus only a jurisdiction/product-line/"
        "subsidiary indicator is likely a subsidiary, not the main company.\n"
        "  - Use the document context to break ties.\n"
        "  - When returning index 0, set retry_search_term to the full official company "
        "name you believe is correct so a second search can be attempted.\n"
    )
    results: dict[int, dict | None] = {}
    needs_retry: list[tuple[int, str, str, dict, list[dict]]] = []

    for chunk_start in range(0, len(entries), _BATCH_PICK_SIZE):
        chunk = entries[chunk_start : chunk_start + _BATCH_PICK_SIZE]
        blocks = [
            _format_batch_entry(chunk_start + i, name, cands, node_def)
            for i, (name, label, node_def, cands) in enumerate(chunk)
        ]
        prompt = (
            f"{domain_block}{rules}\n"
            f"Select the best GLEIF match for each entry below.\n\n"
            + "\n\n".join(blocks)
        )
        try:
            res = await _acreate_structured_output(prompt, _BATCH_PICK_SYSTEM, _BatchPickResult)
            picked_ids = {p.entry_id: p for p in res.picks}
        except Exception as e:
            # Whole-chunk failure — log each and leave unresolved.
            for i, (name, label, node_def, cands) in enumerate(chunk):
                _log_error(output_dir, name, label, f"batch LLM pick failed: {e} — dropped", print_prefix="gleif-resolver")
                results[chunk_start + i] = None
            continue

        for i, (name, label, node_def, cands) in enumerate(chunk):
            eid = chunk_start + i
            p = picked_ids.get(eid)
            if p is not None and 1 <= p.index <= len(cands):
                results[eid] = _strip_match_type(cands[p.index - 1])
            elif p is not None and p.retry_search_term and p.retry_search_term.lower() != name.lower():
                needs_retry.append((eid, name, label, node_def, p.retry_search_term))
            else:
                _log_error(output_dir, name, label, "no acceptable GLEIF match — dropped by LLM", print_prefix="gleif-resolver")
                results[eid] = None

    # ── Retry round: gather candidates for suggested terms, re-pick in one batch ──
    if needs_retry:
        async def _retry_gather(eid, name, label, node_def, term):
            print(f"  [gleif-resolver] retrying {name!r} with suggested term {term!r}", flush=True)
            cands = await asyncio.to_thread(gleif_get_candidates, term)
            return (eid, name, label, node_def, cands)

        gathered = await asyncio.gather(*[
            _retry_gather(*r) for r in needs_retry
        ], return_exceptions=True)

        retry_entries: list[tuple[str, str, dict, list[dict]]] = []
        retry_eids: list[int] = []
        for g in gathered:
            if isinstance(g, Exception):
                continue
            eid, name, label, node_def, cands = g
            if cands:
                retry_eids.append(eid)
                retry_entries.append((name, label, node_def, cands))
            else:
                results[eid] = None

        if retry_entries:
            # Single follow-up batch; no further retry (avoid loops).
            sub = await _batch_llm_pick_once(retry_entries, domain_hint, output_dir)
            for local_id, eid in enumerate(retry_eids):
                results[eid] = sub.get(local_id)
        # Any retry entries that produced no candidates already set None above.
        for eid, name, label, *_ in needs_retry:
            results.setdefault(eid, None)

    return results


async def _batch_llm_pick_once(
    entries: list[tuple[str, str, dict, list[dict]]],
    domain_hint: str,
    output_dir: Path,
) -> dict[int, dict | None]:
    """One batched pick pass with NO retry handling (used for the retry round)."""
    domain_block = (
        f"Document context (prefer geographically/industry-relevant candidates):\n"
        f"{domain_hint[:500]}\n\n"
    ) if domain_hint else ""
    results: dict[int, dict | None] = {}
    for chunk_start in range(0, len(entries), _BATCH_PICK_SIZE):
        chunk = entries[chunk_start : chunk_start + _BATCH_PICK_SIZE]
        blocks = [
            _format_batch_entry(chunk_start + i, name, cands, node_def)
            for i, (name, label, node_def, cands) in enumerate(chunk)
        ]
        prompt = (
            f"{domain_block}Select the best GLEIF match for each entry below. "
            f"Return index 0 if no candidate is acceptable.\n\n" + "\n\n".join(blocks)
        )
        try:
            res = await _acreate_structured_output(prompt, _BATCH_PICK_SYSTEM, _BatchPickResult)
            picked_ids = {p.entry_id: p for p in res.picks}
        except Exception as e:
            for i, (name, label, node_def, cands) in enumerate(chunk):
                _log_error(output_dir, name, label, f"retry batch pick failed: {e} — dropped", print_prefix="gleif-resolver")
                results[chunk_start + i] = None
            continue
        for i, (name, label, node_def, cands) in enumerate(chunk):
            eid = chunk_start + i
            p = picked_ids.get(eid)
            if p is not None and 1 <= p.index <= len(cands):
                results[eid] = _strip_match_type(cands[p.index - 1])
            else:
                _log_error(output_dir, name, label, "no acceptable GLEIF match on retry — dropped", print_prefix="gleif-resolver")
                results[eid] = None
    return results


# ── Resolver class ────────────────────────────────────────────────────────────

class GLEIFResolver(NodeResolver):
    source = "gleif"

    async def resolve_batch(
        self,
        entity_names: list[tuple[str, str]],
        nodes: dict,
        output_dir: Path,
        ctx: "ResolveContext",
    ) -> dict[tuple[str, str], dict | None]:
        if not entity_names:
            return {}
        domain_hint = ctx.domain_hint
        abbr_map = ctx.abbr_map
        output_dir.mkdir(parents=True, exist_ok=True)
        results: dict[tuple[str, str], dict | None] = {}

        # ── Phase A: per-entity cache check + candidate gather + no-LLM prefilter ──
        async def _phase_a(name: str, label: str):
            node_def = nodes.get(label, {})
            pick_key = _pick_cache_key(name, domain_hint, abbr_map)
            if pick_key in _gleif_pick_cache:
                if _VERBOSE: print(f"  [L1-cache] gleif_pick({name!r})")
                return ("done", dict(_gleif_pick_cache[pick_key]))
            db = _lc.get("gleif_pick", pick_key)
            if db is not None:
                if _VERBOSE: print(f"  [L2-cache] gleif_pick({name!r})")
                r = _json.loads(db)
                _gleif_pick_cache[pick_key] = r
                return ("done", dict(r))
            candidates = await _gather_candidates(name, node_def, domain_hint, abbr_map)
            status, val = _prefilter(name, candidates)
            if status == "resolved":
                _cache_store(pick_key, val)
                return ("done", val)
            if status == "none":
                return ("done", None)
            return ("pick", candidates)

        a_results = await asyncio.gather(
            *[_phase_a(name, label) for name, label in entity_names],
            return_exceptions=True,
        )

        needs_pick: list[tuple[str, str, dict, list[dict]]] = []
        pick_keys: list[tuple[str, str]] = []
        for (name, label), res in zip(entity_names, a_results):
            if isinstance(res, Exception):
                results[(name, label)] = None
                continue
            status, payload = res
            if status == "done":
                results[(name, label)] = payload
            else:
                needs_pick.append((name, label, nodes.get(label, {}), payload))
                pick_keys.append((name, label))

        # ── Phase B: one (or few) batched LLM calls for the ambiguous remainder ──
        if needs_pick:
            picked = await _batch_llm_pick(needs_pick, domain_hint, output_dir)
            for local_id, (name, label) in enumerate(pick_keys):
                result = picked.get(local_id)
                results[(name, label)] = result
                if result is not None:
                    _cache_store(_pick_cache_key(name, domain_hint, abbr_map), result)

        return results

    def build_props(
        self,
        term: str,
        label: str,
        node_def: dict,
        llm_extras: dict,
        resolved_map: dict[tuple[str, str], dict | None],
    ) -> tuple[dict, dict] | None:
        gleif_result = resolved_map.get((term, label))
        if gleif_result is None:
            print(f"  [warn] gleif: lookup failed for {term!r} — dropping triple", flush=True)
            return None
        data = dict(gleif_result)
        sem_types = node_def.get("semantic_types") or None
        types = [x for x in [data.pop("entity_legal_form", None), data.pop("category", None)] if x]
        if sem_types and not any(t in types for t in sem_types):
            print(
                f"  [warn] gleif: type mismatch for {term!r} "
                f"(got {types}, want {sem_types}) — dropping triple",
                flush=True,
            )
            return None
        props: dict = dict(data)
        for p in node_def.get("properties", []):
            if p.get("source") == "llm":
                val = llm_extras.get(p["name"])
                if val is None and not p.get("optional"):
                    return None
                if val is not None:
                    props[p["name"]] = val
        return (props, {"types": types})
