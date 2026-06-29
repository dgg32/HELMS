#!/usr/bin/env python3
"""
Apply a review JSON produced by extract.py to a graph database.

Edit the review file before running to correct, add, or remove triples:
  - Delete an object from "triples" to drop that triple entirely.
  - Edit a "lei", "cui", or "name" field to correct a resolved value.
  - Edit "rel_props" to change relationship properties.

Usage:
    python apply_graph.py --schema finance_schema.yaml --apply report_review.json --db finance_kg.db
    python apply_graph.py --schema drug_schema.yaml --apply drug_review.json --db drug.db --dry-run
    python apply_graph.py --schema drug_schema.yaml --apply drug_review.json --backend neo4j
"""
import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from backends import get_backend
from backends.base import GraphBackend
from extract import load_schema, colors_for_filter


# ── Apply ─────────────────────────────────────────────────────────────────────

async def apply_review(file_path: Path, backend: GraphBackend, dry_run: bool = False, filter_level: str = "moderate", run_id: str = "") -> None:
    """Read extraction output and write effective triples to the graph.

    Accepts both new-format _raw.json (with optional _review.json event log) and
    old-format _review.json (inline triples, no events — treated as fully accepted).
    """
    import review_layer as _rl
    print(f"\n{'='*60}")
    _excluded_triples: list[dict] = []  # triples to remove from DB (rejected or color-filtered)
    if file_path.name.endswith("_raw.json"):
        raw_data  = _rl.load_raw(file_path)
        rev_path  = _rl.review_path_for(file_path)
        events    = _rl.load_events(rev_path)
        all_active = _rl.materialize(raw_data, events)  # excludes REJECTs, includes all colors
        data      = raw_data
        _summary  = []
        if events:
            _n_rej = sum(1 for ev in events.values() if ev.get("action") == "REJECT")
            _n_ov  = sum(1 for ev in events.values() if ev.get("action") == "OVERRIDE")
            _n_add = sum(1 for ev in events.values() if ev.get("action") == "ADD")
            if _n_rej:  _summary.append(f"{_n_rej} rejected")
            if _n_ov:   _summary.append(f"{_n_ov} overridden")
            if _n_add:  _summary.append(f"{_n_add} added")
        _ev_note = f" — events: {', '.join(_summary)}" if _summary else ""
        print(f"Applying: {file_path.name}{_ev_note}")
    else:
        raw_data  = None
        events    = {}
        all_active = None
        data    = json.loads(file_path.read_text(encoding="utf-8"))
        triples_raw = data.get("triples", [])
        print(f"Applying review: {file_path.name}")

    _color_keep = colors_for_filter(filter_level)

    if all_active is not None:
        triples = [t for t in all_active if t.get("triple_color", "green") in _color_keep]
        # Build excluded set: color-filtered (active but below threshold) + rejected raw triples
        write_ids = {t.get("_id") for t in triples}
        _color_filtered = [t for t in all_active if t.get("_id") not in write_ids]
        _rejected = [
            t for t in raw_data.get("triples", [])
            if events.get(t.get("_id", ""), {}).get("action") == "REJECT"
        ]
        _excluded_triples = _color_filtered + _rejected
    else:
        triples = [t for t in triples_raw if t.get("triple_color", "green") in _color_keep]

    doc_name     = data.get("doc", "unknown")
    print(f"  {len(triples)} triple(s) to apply.")

    # Remove edges for rejected/color-filtered triples from prior runs of this source doc.
    # create_edge() handles delete-then-create for the write set; this covers the excluded set.
    # Pass triple_id (the stable raw _id) so the delete matches by edge identity rather
    # than the current node pair — this also clears the OLD edge of a triple whose entity
    # was corrected in review (the from/to PK changed), which a node-pair match would miss.
    if not dry_run and _excluded_triples:
        for _t in _excluded_triples:
            try:
                backend.delete_edge(
                    _t["rel_type"],
                    _t["from_label"], _t["from_pk"], _t["from_props"][_t["from_pk"]],
                    _t["to_label"],   _t["to_pk"],   _t["to_props"][_t["to_pk"]],
                    source_doc=doc_name,
                    triple_id=_t.get("_id"),
                )
            except Exception as _del_exc:
                _fp = _t.get("from_props") or {}
                _tp = _t.get("to_props")   or {}
                print(f"  [warn] delete_edge failed for {_t.get('rel_type')} ({_fp.get(_t.get('from_pk',''))} → {_tp.get(_t.get('to_pk',''))}): {_del_exc}")

    _REQUIRED = {"rel_type", "from_label", "from_pk", "from_props", "to_label", "to_pk", "to_props"}
    for i, triple in enumerate(triples):
        missing = _REQUIRED - set(triple.keys())
        if missing:
            raise SystemExit(
                f"Review file '{file_path.name}' triple #{i}: "
                f"missing required field(s): {', '.join(sorted(missing))}"
            )
        fpk, tpk = triple["from_pk"], triple["to_pk"]
        if fpk not in triple["from_props"]:
            raise SystemExit(
                f"Review file '{file_path.name}' triple #{i}: from_props missing PK '{fpk}'"
            )
        if tpk not in triple["to_props"]:
            raise SystemExit(
                f"Review file '{file_path.name}' triple #{i}: to_props missing PK '{tpk}'"
            )

    for triple in triples:
        from_props = dict(triple["from_props"])
        to_props   = dict(triple["to_props"])
        from_pk    = triple["from_pk"]
        to_pk      = triple["to_pk"]
        rp         = {**(triple.get("rel_props") or {}), "source_doc": doc_name}
        rp.setdefault("manually_added", False)
        # Stamp the stable raw _id onto the edge so a later re-run can delete THIS
        # edge by identity (create_edge / delete_edge), independent of the nodes it
        # connects — the link survives a review correction of the from/to entity.
        _tid = triple.get("_id")
        if _tid:
            rp["triple_id"] = _tid
        _tc = triple.get("triple_color", "green")
        if _tc:
            rp["triple_color"] = _tc
        _sq = (triple.get("supporting_quote") or "").strip()
        if _sq:
            rp["supporting_quote"] = _sq
        if run_id:
            rp["run"] = run_id
            from_props.setdefault("run", run_id)  # first-write-wins: skip if node already exists
            to_props.setdefault("run", run_id)
        list_props = {k: v for k, v in rp.items() if isinstance(v, list)}
        scalar_tag = " ".join(f"{k}={v}" for k, v in list_props.items()) if list_props else ""
        edge_str = (
            f"  {'[DRY-RUN] ' if dry_run else ''}✓ "
            f"({from_props.get('name', from_props[from_pk])} [{from_props[from_pk]}])"
            f" -[:{triple['rel_type']}]->"
            f" ({to_props.get('name', to_props[to_pk])} [{to_props[to_pk]}])"
            + (f"  {scalar_tag}" if scalar_tag else "")
        )
        print(edge_str)
        if not dry_run:
            backend.upsert_node(triple["from_label"], from_pk, from_props)
            backend.upsert_node(triple["to_label"],   to_pk,   to_props)
            backend.create_edge(
                triple["from_label"], from_pk, from_props[from_pk],
                triple["to_label"],   to_pk,   to_props[to_pk],
                triple["rel_type"],
                rel_props=rp,
            )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply a review JSON to a graph database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--apply", required=True, metavar="FILE_OR_DIR",
        help="Review JSON file produced by extract.py, or a directory of *_review.json files",
    )
    p.add_argument(
        "--schema", required=True, metavar="PATH",
        help="Schema YAML file (needed to initialise the graph backend tables)",
    )
    p.add_argument(
        "--db", default=None, metavar="PATH",
        help="Database path or URI (default: finance_kg.db for ladybug; NEO4J_URI env var for neo4j)",
    )
    p.add_argument(
        "--backend", default="ladybug", choices=("ladybug", "neo4j"), metavar="BACKEND",
        help="Graph backend. Available: ladybug, neo4j",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be written without committing to the graph.",
    )
    p.add_argument(
        "--filter", default="moderate", choices=["loose", "moderate", "strict"],
        help="Only apply triples at or above this color level "
             "(loose=all, moderate=green+yellow, strict=green only). Default: moderate.",
    )
    p.add_argument(
        "--run-id", default="", metavar="RUN_ID",
        help="Run identifier (e.g. '20260525_170300'). Written into node and edge 'run' field.",
    )
    return p.parse_args()


async def main(args=None) -> None:
    if args is None:
        args = parse_args()

    if args.db is None:
        if args.backend == "ladybug":
            args.db = Path(args.schema).stem.replace("_schema", "_kg") + ".db"
        elif args.backend == "neo4j":
            args.db = os.environ.get("NEO4J_URI")
            if not args.db:
                raise SystemExit("No Neo4j URI: pass --db or set NEO4J_URI in .env")

    if args.backend == "ladybug" and not args.dry_run:
        Path(args.db).parent.mkdir(parents=True, exist_ok=True)

    nodes, rels = load_schema(args.schema)
    backend = get_backend(args.backend, args.db, nodes, rels)

    if args.dry_run:
        print("\n[DRY-RUN] No graph writes will occur.")

    apply_path = Path(args.apply)
    if apply_path.is_dir():
        review_files = sorted(apply_path.glob("*_raw.json"))
        if not review_files:
            # Fallback: old-format run folder with only _review.json files
            review_files = sorted(apply_path.glob("*_review.json"))
        if not review_files:
            raise SystemExit(f"No *_raw.json or *_review.json files found in: {apply_path}")
    else:
        review_files = [apply_path]

    try:
        for rfile in review_files:
            await apply_review(rfile, backend, dry_run=args.dry_run, filter_level=args.filter, run_id=getattr(args, "run_id", ""))

        db_display = Path(args.db).resolve() if args.backend == "ladybug" else args.db
        print(f"\nDone. Graph written to: {db_display} (backend={args.backend})")

        print("\nGraph summary:")
        for label in nodes:
            print(f"  {label}: {backend.count_nodes(label)} node(s)")
        for rel in rels:
            print(f"  {rel['rel_type']}: {backend.count_edges(rel['rel_type'])} edge(s)")
    finally:
        backend.close()


if __name__ == "__main__":
    asyncio.run(main())
