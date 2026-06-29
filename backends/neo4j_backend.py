from __future__ import annotations

from neo4j import GraphDatabase

from .base import GraphBackend, _primary_key, _safe_ident


class Neo4jBackend(GraphBackend):
    def __init__(self, uri: str, username: str, password: str, read_only: bool = False):
        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self._read_only = read_only

    def _run(self, cypher: str, params: dict | None = None) -> list:
        kw = {"default_access_mode": "READ"} if self._read_only else {}
        with self._driver.session(**kw) as session:
            return session.run(cypher, params or {}).data()

    def setup(self, nodes: dict, rels: list[dict]) -> None:
        for label, node_def in nodes.items():
            _safe_ident(label)
            pk = _primary_key(node_def)
            _safe_ident(pk)
            self._run(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{pk} IS UNIQUE"
            )
            self._check_schema_drift(label, node_def.get("properties", []))

        for rel in rels:
            _safe_ident(rel["rel_type"])
            _safe_ident(rel["from_node"])
            _safe_ident(rel["to_node"])

    def _check_schema_drift(self, label: str, expected_props: list[dict]) -> None:
        try:
            result = self._run(f"MATCH (n:{label}) RETURN keys(n) AS k LIMIT 1")
            if not result:
                print(f"  [info] schema drift check skipped for '{label}': no nodes yet (Neo4j limitation).")
                return
            existing = set(result[0]["k"])
            missing = [p["name"] for p in expected_props if p["name"] not in existing]
            if missing:
                print(
                    f"  [WARN] schema drift on '{label}': "
                    f"property/ies {missing} defined in schema.yaml but absent from existing nodes. "
                    "Existing nodes will lack these until re-ingested."
                )
        except Exception as exc:
            print(f"  [WARN] schema drift check failed for '{label}': {exc}")

    def node_exists(self, label: str, pk_col: str, pk_val: str) -> bool:
        result = self._run(
            f"MATCH (n:{label} {{{pk_col}: $v}}) RETURN count(n) AS c",
            {"v": pk_val},
        )
        return result[0]["c"] > 0

    def upsert_node(self, label: str, pk_col: str, props: dict) -> None:
        # First-write-wins, atomic: MERGE matches or creates on PK only;
        # ON CREATE SET fires only for new nodes so existing props are untouched.
        _safe_ident(label)
        _safe_ident(pk_col)
        for k in props:
            _safe_ident(k)
        set_clause = ", ".join(f"n.{k} = ${k}" for k in props)
        self._run(
            f"MERGE (n:{label} {{{pk_col}: ${pk_col}}}) ON CREATE SET {set_clause}",
            props,
        )

    def create_edge(
        self,
        from_label: str, from_pk: str, from_val: str,
        to_label: str, to_pk: str, to_val: str,
        rel_type: str,
        rel_props: dict | None = None,
    ) -> None:
        # Source-aware last-write-wins: DELETE the existing edge from this source
        # document, then CREATE a fresh one.  This prevents stale properties from
        # persisting when a re-extraction produces a different prop set.
        # Falls back to CREATE-only when source_doc is absent.
        source_doc = (rel_props or {}).get("source_doc")
        triple_id  = (rel_props or {}).get("triple_id")
        params: dict = {"__neo4j_fv__": from_val, "__neo4j_tv__": to_val}

        if triple_id is not None and source_doc is not None:
            # Endpoint-independent delete by identity — survives a review correction
            # that changed the from/to node (see ladybug_backend / base for rationale).
            params["__neo4j_tid__"] = triple_id
            params["__neo4j_src__"] = source_doc
            self._run(
                f"MATCH ()-[r:{rel_type} "
                f"{{triple_id: $__neo4j_tid__, source_doc: $__neo4j_src__}}]->() DELETE r",
                params,
            )
        elif source_doc is not None:
            # Delete existing edge from this source (legacy edges without triple_id)
            params["__neo4j_src__"] = source_doc
            self._run(
                f"MATCH (a:{from_label} {{{from_pk}: $__neo4j_fv__}})"
                f"-[r:{rel_type} {{source_doc: $__neo4j_src__}}]->"
                f"(b:{to_label} {{{to_pk}: $__neo4j_tv__}}) DELETE r",
                params,
            )

        # Create new edge with all current properties
        prop_clause = ""
        if rel_props:
            for k in rel_props:
                _safe_ident(k)
            prop_clause = " {" + ", ".join(f"{k}: ${k}" for k in rel_props) + "}"
            params.update(rel_props)
        self._run(
            f"MATCH (a:{from_label} {{{from_pk}: $__neo4j_fv__}}), (b:{to_label} {{{to_pk}: $__neo4j_tv__}}) "
            f"CREATE (a)-[:{rel_type}{prop_clause}]->(b)",
            params,
        )

    def run_cypher(self, query: str, params: dict | None = None) -> list:
        return self._run(query, params)

    def count_nodes(self, label: str) -> int:
        result = self._run(f"MATCH (n:{label}) RETURN count(n) AS c")
        return result[0]["c"]

    def count_edges(self, rel_type: str) -> int:
        result = self._run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS c")
        return result[0]["c"]

    def delete_node(self, label: str, pk_col: str, pk_val: str) -> None:
        _safe_ident(label)
        _safe_ident(pk_col)
        self._run(
            f"MATCH (n:{label} {{{pk_col}: $v}}) DETACH DELETE n", {"v": pk_val}
        )

    def delete_edge(
        self,
        rel_type: str,
        from_label: str, from_pk: str, from_val: str,
        to_label: str,   to_pk: str,   to_val: str,
        source_doc: str | None = None,
        triple_id: str | None = None,
    ) -> None:
        _safe_ident(rel_type)
        _safe_ident(from_label); _safe_ident(from_pk)
        _safe_ident(to_label);   _safe_ident(to_pk)
        # Endpoint-independent delete by identity (see base.delete_edge docstring).
        if triple_id is not None and source_doc is not None:
            self._run(
                f"MATCH ()-[r:{rel_type} {{triple_id: $tid, source_doc: $sd}}]->() DELETE r",
                {"tid": triple_id, "sd": source_doc},
            )
            return
        src_filter = " {source_doc: $sd}" if source_doc is not None else ""
        params: dict = {"fv": from_val, "tv": to_val}
        if source_doc is not None:
            params["sd"] = source_doc
        self._run(
            f"MATCH (a:{from_label} {{{from_pk}: $fv}})"
            f"-[r:{rel_type}{src_filter}]->"
            f"(b:{to_label} {{{to_pk}: $tv}}) DELETE r",
            params,
        )

    def close(self) -> None:
        self._driver.close()
