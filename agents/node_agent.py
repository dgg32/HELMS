"""Node resolution orchestrator.

Dispatches entity resolution to the appropriate resolver for each
external source (UMLS, GLEIF, …). Future sources only need a new
NodeResolver subclass registered in ``_RESOLVERS`` below.

Public API
----------
resolver_for_node(node_def)   -> NodeResolver | None
await resolve_all_nodes(...)  -> dict[tuple, dict | None]
"""
from __future__ import annotations

from pathlib import Path

from .base_resolver import NodeResolver, ResolveContext  # ResolveContext re-exported for callers
from .gleif_resolver import GLEIFResolver
from .umls_resolver import UMLSResolver

# ── Registered resolvers (order matters: first match wins in resolver_for_node) ──

_RESOLVERS: list[NodeResolver] = [
    UMLSResolver(),
    GLEIFResolver(),
]


def resolver_for_node(node_def: dict) -> NodeResolver | None:
    """Return the resolver that handles node_def, or None for llm-only nodes."""
    for r in _RESOLVERS:
        if r.handles(node_def):
            return r
    return None


async def resolve_all_nodes(
    all_items: dict[str, list[dict]],
    rels: list[dict],
    nodes: dict,
    output_dir: Path,
    ctx: "ResolveContext | str | None" = None,
) -> dict[tuple[str, str], dict | None]:
    """Batch-resolve all external-source entities across all resolvers.

    Returns a unified (name, label) → props map consumed by extract.py
    when building triples.

    ``ctx`` is a ResolveContext. A bare string is accepted for backward
    compatibility and treated as ``domain_hint``.
    """
    if ctx is None:
        ctx = ResolveContext()
    elif isinstance(ctx, str):
        ctx = ResolveContext(domain_hint=ctx)
    unified: dict[tuple[str, str], dict | None] = {}
    for resolver in _RESOLVERS:
        unique = resolver.collect_unique_entities(all_items, rels, nodes)
        if not unique:
            continue
        batch = await resolver.resolve_batch(list(unique), nodes, output_dir, ctx)
        unified.update(batch)
    return unified
