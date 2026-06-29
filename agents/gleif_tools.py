#!/usr/bin/env python3
"""GLEIF agentic-extraction tools: LLM-facing tool specs + dispatch handlers.

Self-contained — handlers call lookups directly, so this toolset plugs into the
extraction agent purely by being listed in resolver_tools.RESOLVER_TOOLSETS.
"""
from __future__ import annotations

import json

from lookups import gleif_search as _gleif_search, gleif_get_candidates as _gleif_get_candidates

from .resolver_tools_types import ResolverToolset


def _gleif_find(session, args: dict) -> str:
    """Run all GLEIF searches (exact+names+fuzzy+parent detection) in one call."""
    term = args.get("term", "")
    candidates = _gleif_get_candidates(term)
    if not candidates:
        return json.dumps({"results": [], "message": f"No GLEIF match for '{term}'"})
    return json.dumps({"results": candidates, "count": len(candidates)})


def _gleif_search_tool(session, args: dict) -> str:
    """Fallback single-search-type GLEIF lookup."""
    return _gleif_search(args.get("query", ""), args.get("search_type", "exact"))


_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "gleif_find",
            "description": (
                "PREFERRED GLEIF lookup. Runs exact + names + fuzzy searches and parent detection "
                "in a single call and returns all candidates ranked by relevance. "
                "Each candidate includes: lei, name, status (ACTIVE/INACTIVE), category "
                "(GENERAL/BRANCH/FUND/SOLE_PROPRIETOR), jurisdiction, registration_status "
                "(ISSUED/LAPSED), and match_type ('exact', 'names', 'fuzzy', 'who_owns'). "
                "Candidates with match_type='who_owns' are the DIRECT PARENT of a matched "
                "subsidiary — these are often the correct answer for trade names (e.g. 'Foxconn' "
                "→ Hon Hai, 'TSMC' → Taiwan Semiconductor). "
                "Prefer GENERAL+ISSUED over BRANCH/FUND or LAPSED. "
                "Use gleif_search only when gleif_find returns no results and you need to "
                "try a reformulated query (expanded abbreviation, shorter term, etc.)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {"type": "string", "description": "Company name or trade name to look up"},
                },
                "required": ["term"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gleif_search",
            "description": (
                "Fallback GLEIF search for a specific search type. Use gleif_find first — "
                "it combines all search types in one call. Use gleif_search only when "
                "gleif_find returned no results and you want to retry with a reformulated "
                "query or a specific search_type. "
                "search_type 'exact': strict legal name match. "
                "search_type 'fuzzy': autocomplete prefix, broadest fallback."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":       {"type": "string", "description": "Company name to search"},
                    "search_type": {"type": "string", "enum": ["exact", "fuzzy"], "default": "exact"},
                },
                "required": ["query"],
            },
        },
    },
]


GLEIF_TOOLSET = ResolverToolset(
    name="gleif",
    specs=_SPECS,
    handlers={
        "gleif_find":   _gleif_find,
        "gleif_search": _gleif_search_tool,
    },
)
