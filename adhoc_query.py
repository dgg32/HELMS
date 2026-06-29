#!/usr/bin/env python3
"""Ad-hoc Cypher queries against a LadybugDB database.

Edit the QUERIES list below, then run:
    python adhoc_query.py
    python adhoc_query.py --db supplychain_kg.db
"""
import argparse
from pathlib import Path

import ladybug as kuzu

# ── Edit your queries here ────────────────────────────────────────────────────

QUERIES = [
    ("All nodes",            "MATCH (n:Corporation) -[:PROVIDES]-> (m:Corporation) RETURN n.name, m.name"),
    #("All edges",            "MATCH ()-[r]->() RETURN label(r), count(*) ORDER BY label(r)"),
    # ("Drug list",          "MATCH (n:Substance) RETURN n.cui, n.name ORDER BY n.name"),
    # ("Treats edges",       "MATCH (a:Substance)-[r:MAY_TREAT]->(b:Indication) RETURN a.name, b.name, r.evidence_level"),
]

# ── End of queries ────────────────────────────────────────────────────────────


def run(conn: kuzu.Connection, title: str, cypher: str) -> None:
    print(f"\n── {title} {'─' * max(0, 60 - len(title))}")
    res = conn.execute(cypher)
    cols = res.get_column_names()
    print(f"  {cols}")
    count = 0
    while res.has_next():
        row = res.get_next()
        print(f"  {row}")
        count += 1
    if count == 0:
        print("  (no results)")
    else:
        print(f"  ({count} row{'s' if count != 1 else ''})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="drug_kg.db", help="LadybugDB database path (default: drug_kg.db)")
    args = p.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    db = kuzu.Database(str(db_path))
    conn = kuzu.Connection(db)

    for title, cypher in QUERIES:
        run(conn, title, cypher)


if __name__ == "__main__":
    main()
