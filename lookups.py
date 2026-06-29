#!/usr/bin/env python3
"""UMLS and GLEIF entity lookups with module-level caching.

Two API styles per service:
  *_search()  — returns a JSON string (agent tool interface)
  *_lookup()  — returns a dict or None (extract.py interface)
"""
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

import lookup_cache as _lc

load_dotenv(Path(__file__).parent / ".env")

# L1 hot-caches: every entry is keyed by a TUPLE so the same key drives both the
# L1 dict here and the L2 SQLite store (lookup_cache.cached). They are populated +
# capped exclusively through lookup_cache.cached / cached_async.
_umls_cache:             dict[tuple, str]        = {}  # (term_lower, search_type, sabs, stys, page) → JSON str
_umls_sabs_cache:        dict[tuple, frozenset]  = {}  # (cui,) → frozenset of SABs
_gleif_cache:            dict[tuple, str]        = {}  # (query_lower, search_type) → JSON str; (lei, "parent") → dict|None
_gleif_candidates_cache: dict[tuple, list[dict]] = {}  # (term_lower,) → candidate list
_MAX_CACHE_SIZE = 1_000  # L1 hot-cache cap; L2 SQLite handles persistence

VERBOSE: bool = True

# ── Semantic group name/abbr → TUI and type-name mappings ────────────────────
# Loaded once from SemGroups.txt (pipe-delimited: ABBR|Group Name|TUI|Type Name).
_sg_name_to_tuis:       dict[str, list[str]] = {}   # "Physiology" → ["T043", ...]
_sg_abbr_to_tuis:       dict[str, list[str]] = {}   # "PHYS" → ["T043", ...]
_sg_name_to_typenames:  dict[str, set[str]]  = {}   # "Physiology" → {"Molecular Function", ...}
_sg_abbr_to_typenames:  dict[str, set[str]]  = {}   # "PHYS" → {"Molecular Function", ...}
_type_name_to_tui:      dict[str, str]        = {}   # "Molecular Function" → "T044"
_tui_to_type_name:      dict[str, str]        = {}   # "T044" → "Molecular Function"


def _load_sem_groups() -> None:
    sg_file = Path(__file__).parent / "UMLS" / "SemGroups.txt"
    if not sg_file.exists():
        return
    with open(sg_file, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("|")
            if len(parts) < 4:
                continue
            abbr, name, tui, type_name = parts[0], parts[1], parts[2], parts[3]
            _sg_name_to_tuis.setdefault(name, []).append(tui)
            _sg_abbr_to_tuis.setdefault(abbr, []).append(tui)
            _sg_name_to_typenames.setdefault(name, set()).add(type_name)
            _sg_abbr_to_typenames.setdefault(abbr, set()).add(type_name)
            _type_name_to_tui[type_name] = tui
            _tui_to_type_name[tui] = type_name


_load_sem_groups()


def sem_group_to_tuis(group: str) -> list[str]:
    """Return TUI list for a semantic group name (e.g. 'Physiology') or abbr (e.g. 'PHYS')."""
    result = _sg_name_to_tuis.get(group)
    if result is not None:
        return result
    result = _sg_abbr_to_tuis.get(group.upper())
    return result if result is not None else []


def sem_group_to_type_names(group: str) -> set[str]:
    """Return semantic type name set for a group (e.g. 'Physiology' → {'Molecular Function', ...})."""
    result = _sg_name_to_typenames.get(group)
    if result is not None:
        return result
    result = _sg_abbr_to_typenames.get(group.upper())
    return result if result is not None else set()


def sem_type_names_to_tuis(names_or_tuis: list[str]) -> str:
    """Convert semantic type names/TUIs to a comma-separated TUI string for umls_search.

    Each item can be a type name ('Molecular Function') or TUI code ('T044').
    Unknown names are silently dropped.
    """
    resolved: list[str] = []
    for t in names_or_tuis:
        if t.startswith("T") and t[1:].isdigit():
            resolved.append(t)
        else:
            tui = _type_name_to_tui.get(t, "")
            if tui:
                resolved.append(tui)
    return ",".join(resolved)


def resolve_sem_type_filter(
    sem_types: list | None, sem_group: str = ""
) -> tuple[str, set[str]]:
    """Return (comma-joined TUI string, set of type-name strings) for UMLS filtering.

    The TUI string is passed to the API as ``semanticTypes=`` (server-side filter);
    the name set is used for client-side candidate verification.  ``sem_types``
    (specific type names or TUI codes) takes precedence over the coarse ``sem_group``.
    """
    if sem_types:
        resolved_tuis: list[str] = []
        resolved_names: set[str] = set()
        for t in sem_types:
            if t.startswith("T") and t[1:].isdigit():  # TUI code
                resolved_tuis.append(t)
                name = _tui_to_type_name.get(t)
                if name:
                    resolved_names.add(name)
            else:  # semantic type name
                tui = _type_name_to_tui.get(t)
                if tui:
                    resolved_tuis.append(tui)
                resolved_names.add(t)
        return ",".join(resolved_tuis), resolved_names
    if sem_group:
        return ",".join(sem_group_to_tuis(sem_group)), sem_group_to_type_names(sem_group)
    return "", set()


def _urlopen_json(url: str, retries: int = 3, base_delay: float = 1.0) -> dict:
    """Fetch a URL and return parsed JSON with exponential backoff.

    429 (rate limit) and 5xx errors are retried up to `retries` times.
    Other 4xx errors are raised immediately (permanent client errors).
    """
    if retries < 1:
        retries = 1
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                try:
                    retry_after = float(e.headers.get("Retry-After", base_delay * (2 ** attempt)))
                except ValueError:
                    retry_after = base_delay * (2 ** attempt)
                if attempt == retries - 1:
                    raise
                time.sleep(retry_after)
            elif e.code < 500:  # other 4xx: permanent client errors, don't retry
                raise
            else:
                if attempt == retries - 1:
                    raise
                time.sleep(base_delay * (2 ** attempt))
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))


# ── GLEIF ─────────────────────────────────────────────────────────────────────

# Legal suffixes stripped when building the short-form fallback search term and
# when detecting "subsidiary prefix" names (e.g. "TSMC Partners, Ltd.").
_GLEIF_STOP = frozenset({
    "the", "of", "and", "a", "an",
    "company", "limited", "incorporated", "corporation", "co", "ltd",
    "inc", "corp", "group", "holdings", "plc", "llc", "ag", "sa",
    "gmbh", "nv", "bv", "pty", "pte",
})


def _has_non_suffix_extra(search_term: str, entity_name: str) -> bool:
    """True when entity_name starts with search_term followed by non-suffix words.

    "TSMC Partners, Ltd." / "TSMC" → True  ("Partners" is not a legal suffix)
    "TDK Corporation"    / "TDK"  → False ("Corporation" is a suffix)
    Used to detect subsidiary names so the parent can be looked up.
    """
    if not entity_name.lower().startswith(search_term.lower()):
        return False
    remainder = entity_name[len(search_term):].strip(" ,.")
    if not remainder:
        return False
    extra = [w.lower().strip(".,") for w in remainder.split()]
    return any(w not in _GLEIF_STOP for w in extra)

def _gleif_attrs(rec: dict) -> dict:
    """Extract key disambiguation fields from a GLEIF lei-record object."""
    attrs  = rec.get("attributes", {})
    entity = attrs.get("entity", {})
    lf     = entity.get("legalForm", {})

    primary_name = entity.get("legalName", {}).get("name")

    # Asian companies have a non-Latin primary legalName; prefer the registered
    # English form so the graph stores readable ASCII names.  Priority:
    #   1. PREFERRED_ASCII_TRANSLITERATED_LEGAL_NAME in transliteratedOtherNames
    #      (e.g. TSMC, Powertech — Taiwanese/Chinese companies)
    #   2. ALTERNATIVE_LANGUAGE_LEGAL_NAME in otherNames
    #      (e.g. TDK — Japanese companies whose English name is filed as an alt)
    ascii_name = None
    for other in entity.get("transliteratedOtherNames", []):
        if other.get("type") == "PREFERRED_ASCII_TRANSLITERATED_LEGAL_NAME":
            ascii_name = other.get("name")
            break
    if not ascii_name:
        for other in entity.get("otherNames", []):
            if other.get("type") == "ALTERNATIVE_LANGUAGE_LEGAL_NAME":
                ascii_name = other.get("name")
                break

    return {
        "lei":                 attrs.get("lei"),
        "name":                ascii_name or primary_name,
        "status":              entity.get("status"),
        "jurisdiction":        entity.get("jurisdiction"),
        "category":            entity.get("category"),
        "registration_status": attrs.get("registration", {}).get("status"),
        "entity_legal_form":   lf.get("other") or lf.get("id") or None,
    }


def _fetch_gleif_parent(lei: str) -> dict | None:
    """Return the GENERAL entity that directly owns ``lei`` via filter[owns], or None.

    Result (and the GENERAL-only filter decision) is cached in both the L1 dict and
    the L2 SQLite store under the ``(lei, "parent")`` key.
    """
    pk = (lei.lower(), "parent")

    def _compute() -> tuple[dict | None, bool]:
        try:
            data = _urlopen_json(
                "https://api.gleif.org/api/v1/lei-records?"
                + urllib.parse.urlencode({"filter[owns]": lei, "page[size]": 1})
            )
        except Exception as e:
            print(f"  [warn] GLEIF parent lookup failed for {lei!r}: {e}", flush=True)
            return None, False  # transient network/timeout error — do not cache
        recs = data.get("data", [])
        parent = None
        if isinstance(recs, list) and recs:
            candidate = _gleif_attrs(recs[0])
            if candidate.get("category") == "GENERAL":
                parent = candidate
        # Deterministic outcome either way (a GENERAL parent, or confirmed none),
        # so it is cacheable — a no-parent LEI is never re-fetched from the live API.
        return parent, True

    return _lc.cached(
        _gleif_cache, "gleif", pk, _compute,
        decode=json.loads, encode=json.dumps, copy=dict, l1_max=_MAX_CACHE_SIZE,
    )


def gleif_search(query: str, search_type: str = "exact") -> str:
    """Search GLEIF for a legal entity.

    search_type: 'exact', 'names', or 'fuzzy'.
      'exact' — filter[entity.legalName] (primary legal name, case-insensitive exact)
      'names' — filter[entity.names] (searches primary + other + transliterated names;
                useful for Asian companies whose primary name is non-Latin)
      'fuzzy' — fuzzycompletions autocomplete on legalName
    Returns JSON {"results": [{"lei": ..., "name": ..., "status": ...,
                               "jurisdiction": ..., "category": ...,
                               "registration_status": ...}, ...],
                  "status": "success"}
    or {"error": ...}.  Up to 3 candidates so callers can pick the best match.
    """
    cache_key = (query.lower().strip(), search_type)

    def _compute() -> tuple[str, bool]:
        try:
            if search_type == "exact":
                params  = urllib.parse.urlencode({"filter[entity.legalName]": query, "page[size]": 3})
                data    = _urlopen_json(f"https://api.gleif.org/api/v1/lei-records?{params}")
                records = data.get("data", [])
                if not records:
                    return json.dumps({"error": f"No exact GLEIF match for '{query}'"}), True
                return json.dumps({
                    "results": [_gleif_attrs(r) for r in records],
                    "status":  "success",
                }), True  # deterministic API response (match or no-match)
            elif search_type == "names":
                # Searches primary legal name + other names + transliterated names.
                # Catches Asian companies whose primary GLEIF name is non-Latin.
                params  = urllib.parse.urlencode({"filter[entity.names]": query, "page[size]": 3})
                data    = _urlopen_json(f"https://api.gleif.org/api/v1/lei-records?{params}")
                records = data.get("data", [])
                if not records:
                    return json.dumps({"error": f"No names GLEIF match for '{query}'"}), True
                return json.dumps({
                    "results": [_gleif_attrs(r) for r in records],
                    "status":  "success",
                }), True
            else:  # fuzzy
                params  = urllib.parse.urlencode({"field": "entity.legalName", "q": query})
                data    = _urlopen_json(f"https://api.gleif.org/api/v1/fuzzycompletions?{params}")
                matches = data.get("data", [])
                if not matches:
                    return json.dumps({"error": f"No fuzzy GLEIF match for '{query}'"}), True
                # Extract LEIs from related-URL paths (.../lei-records/{LEI}) and
                # batch-fetch all records in one API call instead of N individual calls.
                leis: list[str] = []
                for m in matches[:10]:
                    related_url = (
                        m.get("relationships", {})
                        .get("lei-records", {}).get("links", {}).get("related")
                    )
                    if related_url:
                        lei = related_url.rstrip("/").rsplit("/", 1)[-1]
                        if lei:
                            leis.append(lei)
                candidates: list[dict] = []
                if leis:
                    try:
                        batch_params = urllib.parse.urlencode({
                            "filter[lei]": ",".join(leis),
                            "page[size]": len(leis),
                        })
                        batch_data = _urlopen_json(
                            f"https://api.gleif.org/api/v1/lei-records?{batch_params}"
                        )
                        recs = batch_data.get("data", [])
                        if isinstance(recs, dict):
                            recs = [recs]
                        candidates = [_gleif_attrs(r) for r in recs]
                    except Exception:
                        pass
                if not candidates:
                    # don't cache: malformed API response may be transient
                    return json.dumps({"error": "No GLEIF record at related URL"}), False
                return json.dumps({"results": candidates, "status": "success"}), True
        except Exception as e:
            return json.dumps({"error": str(e)}), False  # transient — do not cache

    return _lc.cached(
        _gleif_cache, "gleif", cache_key, _compute,
        l1_max=_MAX_CACHE_SIZE, verbose=VERBOSE, label=f"gleif({query!r}, {search_type})",
    )


def gleif_get_candidates(term: str) -> list[dict]:
    """Collect all GLEIF candidates for *term* from every search mode.

    Runs exact + names + fuzzy searches (+ short-form fallback) and, for every
    result whose name looks like a subsidiary (non-suffix words after the term),
    also fetches its direct GLEIF parent via ``filter[owns]``.  Results are
    deduplicated by LEI.  Each candidate dict carries a ``match_type`` key
    (``"exact"``, ``"names"``, ``"fuzzy"``, ``"fuzzy_short"``, ``"names_short"``,
    ``"who_owns"``) so callers / LLMs can reason about provenance.

    Designed for the GLEIF resolver's LLM-pick path.  Unlike ``gleif_lookup``
    (which applies heuristics to return a single answer), this function collects
    broadly and leaves the final choice to the caller.
    """
    _clean = re.sub(r'\s*\([^)]*\)', '', term).strip()
    if _clean:
        term = _clean

    _cand_key = (term.lower(),)

    def _compute() -> tuple[list[dict], bool]:
        seen_leis: set[str] = set()
        candidates: list[dict] = []

        def _add(rec: dict, match_type: str) -> None:
            lei = rec.get("lei")
            if lei and lei not in seen_leis:
                seen_leis.add(lei)
                candidates.append({**rec, "match_type": match_type})

        def _add_with_parent(rec: dict, match_type: str) -> None:
            _add(rec, match_type)
            lei  = rec.get("lei")
            name = rec.get("name") or ""
            if lei and _has_non_suffix_extra(term, name):
                parent = _fetch_gleif_parent(lei)
                if parent:
                    _add(parent, "who_owns")

        for _stype, _mtype in (("exact", "exact"), ("names", "names"), ("fuzzy", "fuzzy")):
            for rec in json.loads(gleif_search(term, _stype)).get("results", []):
                _add_with_parent(rec, _mtype)

        # Short-form fallback when main searches return nothing
        if not candidates:
            words = [w for w in re.split(r"[\s,\.]+", term) if w.lower() not in _GLEIF_STOP and w]
            if len(words) >= 1:
                short = " ".join(words[:3])
                if short.lower() != term.lower().strip() and len(short) >= 3:
                    for rec in json.loads(gleif_search(short, "fuzzy")).get("results", []):
                        _add_with_parent(rec, "fuzzy_short")
                    for rec in json.loads(gleif_search(short, "names")).get("results", []):
                        _add_with_parent(rec, "names_short")

        # Cache only a non-empty pool: an empty result is usually a transient miss
        # in one of the sub-searches, and re-running it cheaply self-heals.
        return candidates, bool(candidates)

    return _lc.cached(
        _gleif_candidates_cache, "gleif_candidates", _cand_key, _compute,
        decode=json.loads, encode=json.dumps, copy=list,
        verbose=VERBOSE, label=f"gleif_candidates({term!r})",
    )


def gleif_lookup(term: str) -> Optional[dict]:
    """Return {'lei': ..., 'name': ..., 'category': ..., 'entity_legal_form': ...} for the best GLEIF match, or None.

    Search cascade:
    1. Exact match on full name (`filter[entity.legalName]`)
    2. Names search on full name (`filter[entity.names]` — exact match across primary,
       other, and transliterated names; catches Asian companies and abbreviations like
       "TSMC" registered as an other name, without matching unrelated prefix hits)
    3. Fuzzy autocomplete on full name (broader prefix search, last resort for full name)
    4. Fuzzy autocomplete on suffix-stripped form
    5. Names search on suffix-stripped form
    6. "WHO OWNS" parent lookup: fires when (a) term is all-caps abbreviation (e.g. "TSMC")
       and top result looks like a named subsidiary ("TSMC Partners, Ltd."), OR (b) cascade
       returned only LAPSED subsidiaries and was upgraded to an ISSUED entity via fuzzy
       (e.g. "Foxconn" → "FOXCONN SINGAPORE PTE LTD" → parent Hon Hai).
       Parent is used only when its name does NOT contain the search term — guards against
       "Samsung" → "Samsung Holdings" false-positive substitution.
    """
    # Strip parenthetical abbreviations: "Powertech Technology (PTI)" → "Powertech Technology"
    _clean = re.sub(r'\s*\([^)]*\)', '', term).strip()
    if _clean:
        term = _clean

    result = json.loads(gleif_search(term, "exact"))
    if "error" in result:
        # names before fuzzy: filter[entity.names] does exact matching across all
        # registered name fields — finds abbreviations registered as other names
        # without returning unrelated prefix hits from autocomplete.
        result = json.loads(gleif_search(term, "names"))
    if "error" in result:
        result = json.loads(gleif_search(term, "fuzzy"))
    if "error" in result:
        words = [w for w in re.split(r"[\s,\.]+", term) if w.lower() not in _GLEIF_STOP and w]
        if len(words) >= 1:
            short = " ".join(words[:3])
            if short.lower() != term.lower().strip() and len(short) >= 3:
                result = json.loads(gleif_search(short, "fuzzy"))
                if "error" in result:
                    result = json.loads(gleif_search(short, "names"))
    if "error" in result:
        return None
    results = result.get("results", [])
    if not results:
        return None
    # GENERAL must be the primary key: gleif_resolver.build_props() enforces semantic_types=["GENERAL"]
    # and drops non-GENERAL entities. ISSUED is only a tiebreaker for mixed-case terms — all-caps
    # terms (e.g. "TSMC") trigger the WHO OWNS parent lookup below which replaces the starting entity,
    # so ISSUED preference on the subsidiary is irrelevant and can break the parent chain.
    _issued_tiebreak = not term.isupper()
    results = sorted(results, key=lambda r: (
        0 if r.get("category") == "GENERAL" else 1,
        0 if (_issued_tiebreak and r.get("registration_status") == "ISSUED") else 1,
    ))
    r = results[0]
    lei  = r.get("lei")
    name = r.get("name") or ""

    # If the cascade only returned LAPSED subsidiaries, try fuzzy for ISSUED alternatives.
    # Example: "Foxconn" exact-prefix search lands on "Foxconn Ventures Holdco" (LAPSED),
    # but fuzzy reveals "FOXCONN SINGAPORE PTE LTD" (ISSUED) whose parent IS Hon Hai.
    # Filter fuzzy candidates to only those whose name actually starts with the search term
    # to avoid picking fuzzy-similar but unrelated companies (e.g. "Foxconcept", "TSM").
    # "WHO OWNS" + LAPSED upgrade block
    # Strategy (in order):
    #   1. If top result is a named subsidiary (has non-suffix extras), try WHO OWNS.
    #      Use the parent only when its name does NOT contain the search term
    #      (rebrand check: "foxconn" not in "hon hai…" → use; "samsung" in "samsung holdings" → skip).
    #   2. If WHO OWNS returns no useful parent AND the entity is LAPSED, look for an
    #      ISSUED entity via names + fuzzy whose name starts with the search term
    #      (TDK Jansen LAPSED → TDK Corporation ISSUED; Foxconn Ventures LAPSED → FOXCONN SINGAPORE ISSUED).
    #      TSMC Partners is handled by step 1 (its WHO OWNS parent IS Taiwan Semiconductor).
    #   3. After an upgrade, if the new entity is itself a subsidiary, try WHO OWNS again
    #      (FOXCONN SINGAPORE → parent Hon Hai).
    def _rebrand_ok(parent: dict) -> bool:
        return term.lower() not in (parent.get("name") or "").lower()

    def _find_issued_upgrade() -> dict | None:
        """Search names + fuzzy for an ISSUED entity whose name starts with term."""
        _alts: list[dict] = []
        for _stype in ("names", "fuzzy"):
            for _x in json.loads(gleif_search(term, _stype)).get("results", []):
                if (_x.get("name") or "").lower().startswith(term.lower()):
                    _alts.append(_x)
        _alts.sort(key=lambda x: (
            0 if x.get("category") == "GENERAL" else 1,
            0 if x.get("registration_status") == "ISSUED" else 1,
        ))
        return _alts[0] if _alts and _alts[0].get("registration_status") == "ISSUED" else None

    if _has_non_suffix_extra(term, name):
        # Step 1 — WHO OWNS from current entity
        _par = _fetch_gleif_parent(lei)
        if _par and _rebrand_ok(_par):
            r, lei, name = _par, _par.get("lei"), _par.get("name") or ""
        elif r.get("registration_status") == "LAPSED":
            # Step 2 — upgrade to ISSUED alternative
            _upg = _find_issued_upgrade()
            if _upg:
                r, lei, name = _upg, _upg.get("lei"), _upg.get("name") or ""
                # Step 3 — WHO OWNS from upgraded entity (Foxconn SINGAPORE → Hon Hai)
                if _has_non_suffix_extra(term, name):
                    _par2 = _fetch_gleif_parent(lei)
                    if _par2 and _rebrand_ok(_par2):
                        r, lei, name = _par2, _par2.get("lei"), _par2.get("name") or ""

    if not lei or not name:
        return None
    return {
        "lei":               lei,
        "name":              name,
        "category":          r.get("category"),
        "entity_legal_form": r.get("entity_legal_form"),
    }


# ── UMLS ──────────────────────────────────────────────────────────────────────

def umls_search(term: str, search_type: str = "words", sabs: str = "", semantic_types: str = "", page_size: int = 5) -> str:
    """Search UMLS for a biomedical concept.

    search_type options: 'words' (default), 'exact', 'normalizedString',
    'normalizedWords', 'leftTruncation', 'rightTruncation'.
    sabs: comma-separated source vocabulary abbreviations to restrict results,
    e.g. 'MED-RT' or 'RXNORM,MSH,SNOMEDCT_US'. Empty string means all sources.
    semantic_types: comma-separated TUI codes for server-side filtering,
    e.g. 'T044' (Molecular Function). Passed as semanticTypes= to the UMLS API.
    page_size: number of candidates to retrieve (default 5, max 25). Node agent
    calls with 25 to get all candidates for LLM disambiguation.
    Returns JSON {"results": [{"cui": ..., "name": ..., "semantic_types": [...], "root_source": ...}, ...], "status": "success"}
    or {"error": ...}.
    semantic_types field in results is a list of UMLS semantic type name strings (e.g. ["Clinical Drug"]).
    root_source is the primary UMLS source vocabulary abbreviation for the result (e.g. "MED-RT", "GO", "RXNORM").
    """
    cache_key = (term.lower().strip(), search_type, sabs, semantic_types, page_size)

    def _compute() -> tuple[str, bool]:
        api_key = os.environ.get("UMLS_API_KEY")
        if not api_key:
            return json.dumps({"error": "UMLS_API_KEY not set"}), False  # config error — do not cache

        query: dict = {
            "string": term, "apiKey": api_key,
            "partialSearch": "true", "pageNumber": 1,
            "pageSize": page_size, "searchType": search_type,
        }
        if sabs:
            query["sabs"] = sabs
        if semantic_types:
            query["semanticTypes"] = semantic_types
        params = urllib.parse.urlencode(query)
        try:
            data    = _urlopen_json(f"https://uts-ws.nlm.nih.gov/rest/search/current?{params}")
            results = data.get("result", {}).get("results", [])
            valid   = [r for r in results if r.get("ui") not in (None, "NONE")]
            if not valid:
                return json.dumps({"error": f"No UMLS result for '{term}' (searchType={search_type})"}), True
            return json.dumps({
                "results": [
                    {
                        "cui":            r["ui"],
                        "name":           r["name"],
                        "semantic_types": list(r.get("semanticTypes", [])),
                        "root_source":    r.get("rootSource", ""),
                    }
                    for r in valid
                ],
                "status": "success",
            }), True  # deterministic API response (match or no-match)
        except Exception as e:
            err_str = str(e).replace(api_key, "***").replace(urllib.parse.quote(api_key, safe=""), "***")
            return json.dumps({"error": err_str}), False  # transient — do not cache

    return _lc.cached(
        _umls_cache, "umls", cache_key, _compute, l1_max=_MAX_CACHE_SIZE,
        verbose=VERBOSE, label=f"umls({term!r}, {search_type}, sabs={sabs!r}, stys={semantic_types!r})",
    )


def umls_get_sabs(cui: str) -> frozenset:
    """Return all source vocabulary abbreviations (SABs) for a UMLS CUI via the atoms endpoint.

    Uses the atoms endpoint: /rest/content/current/CUI/{cui}/atoms
    Returns a frozenset of SAB strings (e.g. frozenset({'RXNORM', 'MSH', 'MTH'})).
    Returns empty frozenset on error or missing API key — caller should treat that as
    'unknown' and skip any vocab check rather than blocking.
    """
    key = (cui.upper(),)

    def _compute() -> tuple[frozenset, bool]:
        api_key = os.environ.get("UMLS_API_KEY")
        if not api_key:
            return frozenset(), False  # config error — do not cache
        try:
            data  = _urlopen_json(
                f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{key[0]}/atoms"
                f"?apiKey={urllib.parse.quote(api_key, safe='')}&pageSize=100"
            )
            atoms = data.get("result", [])  # atoms endpoint returns result as a list directly
            return frozenset(r["rootSource"] for r in atoms if r.get("rootSource")), True
        except Exception:
            return frozenset(), False  # transient — do not cache

    return _lc.cached(
        _umls_sabs_cache, "umls_sabs", key, _compute,
        decode=lambda raw: frozenset(json.loads(raw)),
        encode=lambda s: json.dumps(sorted(s)),
        l1_max=_MAX_CACHE_SIZE,
    )


def umls_lookup(
    term: str,
    search_type: str = "words",
    sabs: str = "",
    sem_group: str = "",
    sem_types: Optional[list] = None,
) -> Optional[dict]:
    """Return {'cui': ..., 'name': ..., 'semantic_types': [...]} for the best UMLS match, or None.

    When search_type='words' (default), retries with 'normalizedWords' then 'normalizedString'
    before giving up. sabs restricts the source vocabularies searched (e.g. 'MED-RT').
    sem_group: coarse semantic group name (e.g. 'Physiology', 'Disorders') or abbreviation
      (e.g. 'PHYS', 'DISO'). All TUIs in the group are passed as semanticTypes= to the UMLS API.
    sem_types: list of specific semantic type names (e.g. ['Molecular Function']) or TUI codes
      (e.g. ['T044']). When provided, takes precedence over sem_group for TUI filtering.
      Client-side candidate iteration retained as a safety net. Returns None if no match found.
    """
    _types = ["words", "normalizedWords", "normalizedString"] if search_type == "words" else [search_type]

    tuis, group_names = resolve_sem_type_filter(sem_types, sem_group)

    for st in _types:
        result = json.loads(umls_search(term, st, sabs=sabs, semantic_types=tuis))
        if "error" in result:
            continue
        for r in result["results"]:
            stypes = r.get("semantic_types", [])
            if group_names and not (set(stypes) & group_names):
                continue
            return {"cui": r["cui"], "name": r["name"], "semantic_types": stypes}
    return None
