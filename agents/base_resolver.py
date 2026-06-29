"""Abstract base class for node resolvers.

A NodeResolver encapsulates the logic for one external entity source
(e.g. UMLS, GLEIF). To add a new source:
  1. Subclass NodeResolver and set ``source`` to match the schema property
     ``source`` value (e.g. "my_db").
  2. Implement ``resolve_batch`` and ``build_props``.
  3. Register the instance in ``agents/node_agent.py:_RESOLVERS``.
"""
from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_log_lock = threading.Lock()


@dataclass
class ResolveContext:
    """Per-run inputs threaded from extract.py into every resolver's resolve_batch.

    Mirrors the ValidatorContext pattern on the semantic-check side: add a field
    here to pass new per-document context to resolvers WITHOUT changing every
    resolve_batch signature again.

    domain_hint: extraction instructions / document domain, used to disambiguate
                 candidates (e.g. Taiwan for ODM/electronics).
    abbr_map:    document-derived ABBR (upper-case) -> full name, from
                 extract._extract_abbreviations. Lets a resolver expand an
                 abbreviation the document already defined inline WITHOUT an LLM
                 call (deterministic, and more faithful than the model's guess).
    """
    domain_hint: str = ""
    abbr_map: "dict[str, str] | None" = None


def _log_error(output_dir: Path, name: str, label: str, error: str, *, print_prefix: str = "") -> None:
    entry = json.dumps({
        "entity": name,
        "label":  label,
        "error":  error,
        "ts":     datetime.now(timezone.utc).isoformat(),
    })
    log_path = output_dir / "_node_agent_errors.jsonl"
    with _log_lock:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(entry + "\n")
    if print_prefix:
        print(f"  [{print_prefix}] error logged: {name!r} ({label}) — {error}", flush=True)


class NodeResolver(ABC):
    source: str  # matches schema property source= value, e.g. "umls", "gleif"

    def handles(self, node_def: dict) -> bool:
        """Return True if this resolver owns at least one property of node_def."""
        return any(
            p.get("source") == self.source
            for p in node_def.get("properties", [])
        )

    def collect_unique_entities(
        self,
        all_items: dict[str, list[dict]],
        rels: list[dict],
        nodes: dict,
    ) -> set[tuple[str, str]]:
        """Return (name, label) pairs for all entities this resolver must resolve."""
        entities: set[tuple[str, str]] = set()
        for rel in rels:
            from_handles = self.handles(nodes[rel["from_node"]])
            to_handles   = self.handles(nodes[rel["to_node"]])
            for item in all_items[rel["rel_type"]]:
                if from_handles:
                    term = (item.get(rel["from_field"]) or "").strip()
                    if term:
                        entities.add((term, rel["from_node"]))
                if to_handles:
                    term = (item.get(rel["to_field"]) or "").strip()
                    if term:
                        entities.add((term, rel["to_node"]))
        return entities

    @abstractmethod
    async def resolve_batch(
        self,
        entity_names: list[tuple[str, str]],
        nodes: dict,
        output_dir: Path,
        ctx: "ResolveContext",
    ) -> dict[tuple[str, str], dict | None]:
        """Resolve all entities concurrently. Returns (name, label) → props or None."""
        ...

    @abstractmethod
    def build_props(
        self,
        term: str,
        label: str,
        node_def: dict,
        llm_extras: dict,
        resolved_map: dict[tuple[str, str], dict | None],
    ) -> tuple[dict, dict] | None:
        """Build (props, meta) for one node from the pre-resolved map.

        Returns None if the entity could not be resolved or fails type validation.
        meta is a dict with at least a "types" key for semantic/legal type info.
        Also applies any llm-sourced properties from llm_extras.
        """
        ...
