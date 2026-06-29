"""Write-path coverage: apply_graph.py orchestration + backend edge identity.

apply_graph.py and the backend write methods had no prior tests. Two layers here:

  * `_FakeBackend` models the documented edge semantics (delete-then-create, scoped
    by triple_id+source_doc or by node pair). The apply_graph tests run the real
    `apply_review` against it end-to-end, so they exercise color filtering, the
    excluded-set deletion, and triple_id stamping — the orchestration logic.

  * `TestLadybugEdgeIdentity` pins the actual Cypher in LadybugBackend against a
    real on-disk Kuzu DB, so the modeled fake can't silently drift from reality.
"""
import asyncio
import json

import pytest

from backends.base import GraphBackend
from apply_graph import apply_review


# ── Modeled fake backend ──────────────────────────────────────────────────────

class _FakeBackend(GraphBackend):
    """In-memory backend mirroring the real delete-then-create edge semantics."""

    def __init__(self):
        self.nodes: dict[tuple, dict] = {}
        self.edges: list[dict] = []

    def setup(self, nodes, rels):
        pass

    def node_exists(self, label, pk_col, pk_val):
        return (label, pk_val) in self.nodes

    def upsert_node(self, label, pk_col, props):
        self.nodes.setdefault((label, props[pk_col]), dict(props))  # first-write-wins

    def create_edge(self, from_label, from_pk, from_val, to_label, to_pk, to_val, rel_type, rel_props=None):
        rp = rel_props or {}
        src = rp.get("source_doc")
        tid = rp.get("triple_id")
        # Delete-then-create, same preference order as the real backends.
        if tid is not None and src is not None:
            self.edges = [e for e in self.edges
                          if not (e["rel_type"] == rel_type and e["triple_id"] == tid and e["source_doc"] == src)]
        elif src is not None:
            self.edges = [e for e in self.edges
                          if not (e["rel_type"] == rel_type and e["from_val"] == from_val
                                  and e["to_val"] == to_val and e["source_doc"] == src)]
        else:
            self.edges = [e for e in self.edges
                          if not (e["rel_type"] == rel_type and e["from_val"] == from_val and e["to_val"] == to_val)]
        self.edges.append({
            "rel_type": rel_type, "from_label": from_label, "from_val": from_val,
            "to_label": to_label, "to_val": to_val,
            "source_doc": src, "triple_id": tid, "props": dict(rp),
        })

    def delete_edge(self, rel_type, from_label, from_pk, from_val, to_label, to_pk, to_val,
                    source_doc=None, triple_id=None):
        if triple_id is not None and source_doc is not None:
            self.edges = [e for e in self.edges
                          if not (e["rel_type"] == rel_type and e["triple_id"] == triple_id
                                  and e["source_doc"] == source_doc)]
            return
        def _match(e):
            if e["rel_type"] != rel_type or e["from_val"] != from_val or e["to_val"] != to_val:
                return False
            return source_doc is None or e["source_doc"] == source_doc
        self.edges = [e for e in self.edges if not _match(e)]

    def count_nodes(self, label):
        return sum(1 for (lbl, _) in self.nodes if lbl == label)

    def count_edges(self, rel_type):
        return sum(1 for e in self.edges if e["rel_type"] == rel_type)

    def run_cypher(self, query, params=None):
        return []

    def delete_node(self, label, pk_col, pk_val):
        self.nodes.pop((label, pk_val), None)


def _triple(_id, from_cui, to_cui, color="green"):
    return {
        "_id": _id,
        "rel_type": "TREATS",
        "from_label": "Substance", "from_pk": "cui",
        "from_props": {"cui": from_cui, "name": from_cui},
        "to_label": "Indication", "to_pk": "cui",
        "to_props": {"cui": to_cui, "name": to_cui},
        "rel_props": {}, "triple_color": color,
    }


def _write_raw(tmp_path, name, triples, events=None):
    raw_path = tmp_path / f"{name}_raw.json"
    raw_path.write_text(json.dumps({"doc": f"{name}.md", "triples": triples}))
    if events is not None:
        (tmp_path / f"{name}_review.json").write_text(
            json.dumps({"raw_hash": "", "events": events, "saved_at": ""})
        )
    return raw_path


def _apply(raw_path, backend, filter_level="moderate"):
    asyncio.run(apply_review(raw_path, backend, dry_run=False, filter_level=filter_level, run_id=""))


# ── apply_graph orchestration ─────────────────────────────────────────────────

def test_edge_is_stamped_with_triple_id(tmp_path):
    backend = _FakeBackend()
    _apply(_write_raw(tmp_path, "doc", [_triple("fid1", "C_A", "C_T")]), backend)
    assert backend.count_edges("TREATS") == 1
    assert backend.edges[0]["triple_id"] == "fid1"


def test_rerun_is_idempotent(tmp_path):
    backend = _FakeBackend()
    raw = _write_raw(tmp_path, "doc", [_triple("fid1", "C_A", "C_T")])
    _apply(raw, backend)
    _apply(raw, backend)
    assert backend.count_edges("TREATS") == 1  # delete-then-create, not duplicated


def test_override_entity_correction_leaves_no_orphan(tmp_path):
    """A review correcting the resolved entity must not orphan the old edge.

    First run writes the wrong entity; the correction re-run must delete that old
    edge (by triple_id) while creating the corrected one — exactly one edge, on C_RIGHT.
    """
    backend = _FakeBackend()
    # Run 1: extraction resolved the wrong entity, no review yet.
    raw1 = _write_raw(tmp_path, "doc", [_triple("fabc", "C_WRONG", "C_T")])
    _apply(raw1, backend)
    assert backend.edges[0]["from_val"] == "C_WRONG"

    # Run 2: reviewer overrides the from-entity to the correct one.
    events = {"fabc": {
        "action": "OVERRIDE",
        "from_props": {"cui": "C_RIGHT", "name": "C_RIGHT"},
        "to_props": {"cui": "C_T", "name": "C_T"},
        "rel_props": {},
    }}
    raw2 = _write_raw(tmp_path, "doc", [_triple("fabc", "C_WRONG", "C_T")], events=events)
    _apply(raw2, backend)

    assert backend.count_edges("TREATS") == 1
    assert backend.edges[0]["from_val"] == "C_RIGHT"  # no C_WRONG orphan remains


def test_rejected_triple_deletes_prior_edge(tmp_path):
    backend = _FakeBackend()
    raw1 = _write_raw(tmp_path, "doc", [_triple("fid1", "C_A", "C_T")])
    _apply(raw1, backend)
    assert backend.count_edges("TREATS") == 1

    events = {"fid1": {"action": "REJECT"}}
    raw2 = _write_raw(tmp_path, "doc", [_triple("fid1", "C_A", "C_T")], events=events)
    _apply(raw2, backend)
    assert backend.count_edges("TREATS") == 0  # rejection removed the prior edge


def test_color_filter_excludes_and_deletes_red_under_moderate(tmp_path):
    backend = _FakeBackend()
    # Run 1: triple is green and written.
    raw1 = _write_raw(tmp_path, "doc", [_triple("fid1", "C_A", "C_T", color="green")])
    _apply(raw1, backend)
    assert backend.count_edges("TREATS") == 1

    # Run 2: reviewer marks it red. moderate filter drops it AND the prior edge is removed.
    events = {"fid1": {
        "action": "OVERRIDE",
        "from_props": {"cui": "C_A", "name": "C_A"},
        "to_props": {"cui": "C_T", "name": "C_T"},
        "rel_props": {}, "triple_color": "red",
    }}
    raw2 = _write_raw(tmp_path, "doc", [_triple("fid1", "C_A", "C_T", color="green")], events=events)
    _apply(raw2, backend)
    assert backend.count_edges("TREATS") == 0


# ── Real LadybugDB backend: edge identity Cypher ──────────────────────────────

class TestLadybugEdgeIdentity:
    """Pin the real create_edge / delete_edge identity semantics against Kuzu."""

    @staticmethod
    def _make_backend(tmp_path):
        pytest.importorskip("ladybug")
        from backends.ladybug_backend import LadybugBackend
        nodes = {
            "Substance": {"properties": [
                {"name": "cui", "type": "STRING", "primary_key": True},
                {"name": "name", "type": "STRING"},
            ]},
            "Indication": {"properties": [
                {"name": "cui", "type": "STRING", "primary_key": True},
                {"name": "name", "type": "STRING"},
            ]},
        }
        rels = [{"rel_type": "TREATS", "from_node": "Substance", "to_node": "Indication", "properties": []}]
        backend = LadybugBackend(str(tmp_path / "kg.db"))
        backend.setup(nodes, rels)
        backend.upsert_node("Substance", "cui", {"cui": "C_WRONG", "name": "wrong"})
        backend.upsert_node("Substance", "cui", {"cui": "C_RIGHT", "name": "right"})
        backend.upsert_node("Indication", "cui", {"cui": "C_T", "name": "target"})
        return backend, nodes, rels

    def test_create_edge_by_triple_id_replaces_corrected_entity(self, tmp_path):
        backend, _, _ = self._make_backend(tmp_path)
        try:
            backend.create_edge("Substance", "cui", "C_WRONG", "Indication", "cui", "C_T",
                                 "TREATS", rel_props={"source_doc": "doc.md", "triple_id": "X"})
            assert backend.count_edges("TREATS") == 1
            # Same triple_id, corrected from-entity → old edge must be replaced, not duplicated.
            backend.create_edge("Substance", "cui", "C_RIGHT", "Indication", "cui", "C_T",
                                 "TREATS", rel_props={"source_doc": "doc.md", "triple_id": "X"})
            assert backend.count_edges("TREATS") == 1
            rows = backend.run_cypher("MATCH (a:Substance)-[r:TREATS]->(b:Indication) RETURN a.cui AS cui")
            assert rows[0]["cui"] == "C_RIGHT"  # no C_WRONG orphan
        finally:
            backend.close()

    def test_delete_edge_by_triple_id_is_endpoint_independent(self, tmp_path):
        backend, _, _ = self._make_backend(tmp_path)
        try:
            backend.create_edge("Substance", "cui", "C_WRONG", "Indication", "cui", "C_T",
                                 "TREATS", rel_props={"source_doc": "doc.md", "triple_id": "X"})
            # Delete by identity using DIFFERENT endpoints than the stored edge.
            backend.delete_edge("TREATS", "Substance", "cui", "C_RIGHT",
                                 "Indication", "cui", "C_T", source_doc="doc.md", triple_id="X")
            assert backend.count_edges("TREATS") == 0
        finally:
            backend.close()

    def test_other_source_doc_edge_survives(self, tmp_path):
        backend, _, _ = self._make_backend(tmp_path)
        try:
            backend.create_edge("Substance", "cui", "C_WRONG", "Indication", "cui", "C_T",
                                 "TREATS", rel_props={"source_doc": "docA.md", "triple_id": "X"})
            backend.create_edge("Substance", "cui", "C_WRONG", "Indication", "cui", "C_T",
                                 "TREATS", rel_props={"source_doc": "docB.md", "triple_id": "X"})
            # Deleting docA's edge by identity must leave docB's (same triple_id) intact.
            backend.delete_edge("TREATS", "Substance", "cui", "C_WRONG",
                                 "Indication", "cui", "C_T", source_doc="docA.md", triple_id="X")
            assert backend.count_edges("TREATS") == 1
        finally:
            backend.close()
