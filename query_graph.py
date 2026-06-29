#!/usr/bin/env python3
"""
Query a Ladybug knowledge graph built by extract.py.

Run from the project root:
    python query_graph.py [--schema finance_schema.yaml] [--db drug_knowledge.db]
"""
import argparse
from pathlib import Path

import ladybug as kuzu
import yaml


def load_schema(path: str) -> tuple[dict, list[dict]]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["nodes"], data["relationships"]


def props_clause(node_def: dict, alias: str) -> str:
    """Build a RETURN sub-clause for all schema-defined properties of a node."""
    return ", ".join(f"{alias}.{p['name']}" for p in node_def.get("properties", []))


def primary_key(node_def: dict) -> str:
    for p in node_def.get("properties", []):
        if p.get("primary_key"):
            return p["name"]
    raise ValueError(f"Node has no primary_key property: {node_def}")


def run(conn: kuzu.Connection, title: str, cypher: str):
    print(f"\n── {title} {'─' * max(0, 55 - len(title))}")
    res = conn.execute(cypher)
    count = 0
    while res.has_next():
        print(" ", res.get_next())
        count += 1
    if count == 0:
        print("  (no results)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Query a Ladybug knowledge graph built by extract.py."
    )
    p.add_argument(
        "--schema", required=True, metavar="PATH",
        help="Schema YAML file",
    )
    p.add_argument(
        "--db", default="finance_kg.db", metavar="PATH",
        help="Ladybug database path (default: finance_kg.db)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.db.startswith("bolt://") or args.db.startswith("neo4j://"):
        raise SystemExit(
            "query_graph.py only supports LadybugDB. "
            "For Neo4j, use cypher-shell or Neo4j Browser."
        )
    if not Path(args.db).exists():
        raise SystemExit(f"Database not found: {args.db}")

    nodes, rels = load_schema(args.schema)

    db   = kuzu.Database(args.db)
    conn = kuzu.Connection(db)

    # ── All nodes by type ────────────────────────────────────────────────────
    for label, node_def in nodes.items():
        run(
            conn,
            f"All {label} nodes",
            f"MATCH (n:{label}) RETURN {props_clause(node_def, 'n')} ORDER BY n.{primary_key(node_def)}",
        )

    # ── All relationships ────────────────────────────────────────────────────
    for rel in rels:
        from_label = rel["from_node"]
        to_label   = rel["to_node"]
        rel_type   = rel["rel_type"]

        ret_parts  = [
            props_clause(nodes[from_label], "a"),
            props_clause(nodes[to_label],   "b"),
        ]
        # Append relationship-level properties (schema-defined + system fields)
        rel_props = rel.get("properties", [])
        if rel_props:
            ret_parts.append(", ".join(f"r.{p['name']}" for p in rel_props))
        ret_parts.append("r.source_doc")
        ret_parts.append("r.triple_color")
        ret_parts.append("r.supporting_quote")

        run(
            conn,
            f"{from_label} -[{rel_type}]-> {to_label}",
            f"MATCH (a:{from_label})-[r:{rel_type}]->(b:{to_label}) "
            f"RETURN {', '.join(ret_parts)} ORDER BY a.{primary_key(nodes[from_label])}",
        )

    # ── Counts ───────────────────────────────────────────────────────────────
    run(conn, "Node counts", "MATCH (n) RETURN label(n), count(*) ORDER BY label(n)")
    run(conn, "Edge counts", "MATCH ()-[r]->() RETURN label(r), count(*) ORDER BY label(r)")


if __name__ == "__main__":
    main()

