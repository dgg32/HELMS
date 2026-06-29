#!/usr/bin/env python3
"""Pluggable agentic-extraction tool registry.

Each naming service (GLEIF, UMLS, Wikidata, …) ships a `ResolverToolset` bundling
its LLM-facing tool specs and the handlers that execute them. `extraction_agent`
assembles its `TOOLS` list and dispatch table from this registry, so adding a
naming service to the agentic-retry path is:

  1. write agents/<svc>_tools.py defining a ResolverToolset (specs + handlers)
  2. import it and append to RESOLVER_TOOLSETS below

No edits to extraction_agent.py, ExtractionSession, or _dispatch are required.
A handler has signature (session, args) -> str; it may read session state but the
GLEIF/UMLS handlers are self-contained (they call lookups directly).
"""
from __future__ import annotations

from .resolver_tools_types import ResolverToolset, ToolHandler  # noqa: F401 — re-exported for callers
from .gleif_tools import GLEIF_TOOLSET
from .umls_tools import UMLS_TOOLSET

# Registry — append a new naming service's toolset here. Order controls the order
# tools appear to the LLM (after the framework tools).
RESOLVER_TOOLSETS: list[ResolverToolset] = [
    GLEIF_TOOLSET,
    UMLS_TOOLSET,
]

# Flattened views consumed by extraction_agent.
RESOLVER_TOOL_SPECS: list[dict] = [spec for ts in RESOLVER_TOOLSETS for spec in ts.specs]
RESOLVER_HANDLERS: dict[str, ToolHandler] = {
    name: fn for ts in RESOLVER_TOOLSETS for name, fn in ts.handlers.items()
}
