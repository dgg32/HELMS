from __future__ import annotations

import re
from abc import ABC, abstractmethod

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_ident(name: str) -> str:
    if not _IDENT_RE.match(name):
        raise ValueError(f"Unsafe identifier: {name!r}")
    return name


def _primary_key(node_def: dict) -> str:
    for p in node_def.get("properties", []):
        if p.get("primary_key"):
            return p["name"]
    raise ValueError(f"Node has no primary_key property: {node_def}")


class GraphBackend(ABC):
    @abstractmethod
    def setup(self, nodes: dict, rels: list[dict]) -> None: ...

    @abstractmethod
    def node_exists(self, label: str, pk_col: str, pk_val: str) -> bool: ...

    @abstractmethod
    def upsert_node(self, label: str, pk_col: str, props: dict) -> None: ...

    @abstractmethod
    def create_edge(
        self,
        from_label: str, from_pk: str, from_val: str,
        to_label: str, to_pk: str, to_val: str,
        rel_type: str,
        rel_props: dict | None = None,
    ) -> None: ...

    @abstractmethod
    def count_nodes(self, label: str) -> int: ...

    @abstractmethod
    def count_edges(self, rel_type: str) -> int: ...

    @abstractmethod
    def run_cypher(self, query: str, params: dict | None = None) -> list: ...

    @abstractmethod
    def delete_node(self, label: str, pk_col: str, pk_val: str) -> None: ...

    @abstractmethod
    def delete_edge(
        self,
        rel_type: str,
        from_label: str, from_pk: str, from_val: str,
        to_label: str,   to_pk: str,   to_val: str,
        source_doc: str | None = None,
        triple_id: str | None = None,
    ) -> None:
        """Delete an edge.

        When ``triple_id`` AND ``source_doc`` are both given, the match is
        endpoint-INDEPENDENT: the edge is identified by ``{triple_id, source_doc}``
        alone, so a re-run still removes the stale edge after a review correction
        changed the from/to primary-key values. Otherwise the match falls back to
        the from/to node pair (scoped to ``source_doc`` when present).
        """
        ...

    def close(self) -> None:
        pass
