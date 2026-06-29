#!/usr/bin/env python3
"""UMLS agentic-extraction tools: LLM-facing tool specs + dispatch handlers.

Self-contained — handlers call lookups directly, so this toolset plugs into the
extraction agent purely by being listed in resolver_tools.RESOLVER_TOOLSETS.
"""
from __future__ import annotations

import json

from lookups import umls_search as _umls_search, sem_type_names_to_tuis as _sem_type_names_to_tuis

from .resolver_tools_types import ResolverToolset


def _tuis_from(semantic_types: str) -> str:
    if not semantic_types:
        return ""
    return _sem_type_names_to_tuis([t.strip() for t in semantic_types.split(",") if t.strip()])


def _umls_find(session, args: dict) -> str:
    """Run UMLS cascade (words→normalizedWords→normalizedString) in one call, return first hit."""
    term = args.get("term", "")
    sabs = args.get("sabs", "")
    tuis = _tuis_from(args.get("semantic_types", ""))
    last_error: str | None = None
    for search_type in ("words", "normalizedWords", "normalizedString"):
        raw = _umls_search(term, search_type, sabs=sabs, semantic_types=tuis, page_size=10)
        parsed = json.loads(raw)
        if "error" in parsed:
            last_error = parsed["error"]
            continue
        results = parsed.get("results", [])
        if results:
            return json.dumps({"results": results, "search_type_used": search_type, "count": len(results)})
    if last_error:
        return json.dumps({"results": [], "error": last_error, "message": f"UMLS API error for '{term}': {last_error}"})
    return json.dumps({"results": [], "message": f"No UMLS match for '{term}'"})


def _umls_search_tool(session, args: dict) -> str:
    """Fallback single-search-type UMLS lookup."""
    tuis = _tuis_from(args.get("semantic_types", ""))
    return _umls_search(
        args.get("term", ""), args.get("search_type", "words"),
        sabs=args.get("sabs", ""), semantic_types=tuis,
    )


_SABS_DESC = (
    "Comma-separated UMLS source vocabulary abbreviations to restrict results "
    "(e.g. 'MED-RT', 'RXNORM,MSH', 'SNOMEDCT_US,HPO'). "
    "Read from_umls_vocabs / to_umls_vocabs in the schema and pass them here. "
    "Leave empty only when schema has no umls_vocabs for this node."
)
_SEM_DESC = (
    "Comma-separated UMLS semantic type names or TUI codes to restrict results "
    "(e.g. 'Molecular Function', 'Pharmacologic Substance', 'T044'). "
    "Read from_semantic_types / to_semantic_types in the schema and pass them here. "
    "Leave empty only when schema has no semantic_types for this node."
)


_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "umls_find",
            "description": (
                "PREFERRED UMLS lookup. Runs words→normalizedWords→normalizedString cascade "
                "in a single call and returns candidates from the first successful search type. "
                "ALWAYS pass sabs when the schema specifies umls_vocabs for the target node "
                "(e.g. sabs='MED-RT' for mechanism nodes, sabs='RXNORM,MSH' for drug nodes). "
                "ALWAYS pass semantic_types when the schema specifies semantic_types for the target node "
                "(e.g. semantic_types='Molecular Function' for MOA nodes). "
                "Using both prevents wrong-vocabulary and wrong-type matches. "
                "Use umls_search only when umls_find returns no results and you need a "
                "specific search_type (e.g. rightTruncation, normalizedWords) or a "
                "reformulated query (shorter term, INN name)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {"type": "string", "description": "Biomedical term to look up"},
                    "sabs": {"type": "string", "description": _SABS_DESC, "default": ""},
                    "semantic_types": {"type": "string", "description": _SEM_DESC, "default": ""},
                },
                "required": ["term"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "umls_search",
            "description": (
                "Fallback UMLS search for a specific search type. Use umls_find first — "
                "it runs the full cascade in one call. Use umls_search only when umls_find "
                "returned no results and you need a specific search_type or reformulated query "
                "(shorter term, INN name, rightTruncation, normalizedWords). "
                "ALWAYS pass sabs and semantic_types as for umls_find."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {"type": "string", "description": "Biomedical term to look up"},
                    "search_type": {
                        "type": "string",
                        "enum": ["words", "normalizedWords", "exact", "normalizedString", "rightTruncation"],
                        "default": "words",
                    },
                    "sabs": {"type": "string", "description": _SABS_DESC, "default": ""},
                    "semantic_types": {"type": "string", "description": _SEM_DESC, "default": ""},
                },
                "required": ["term"],
            },
        },
    },
]


UMLS_TOOLSET = ResolverToolset(
    name="umls",
    specs=_SPECS,
    handlers={
        "umls_find":   _umls_find,
        "umls_search": _umls_search_tool,
    },
)
