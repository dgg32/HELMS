from __future__ import annotations

import threading

import ladybug as kuzu

from .base import GraphBackend, _primary_key, _safe_ident


class _MaterializedResult:
    """Row cursor backed by a pre-fetched list — lets the Ladybug DB/connection be freed immediately."""
    __slots__ = ("_col_names", "_rows", "_idx")

    def __init__(self, col_names: list, rows: list) -> None:
        self._col_names = list(col_names)
        self._rows = rows
        self._idx = 0

    def get_column_names(self) -> list:
        return self._col_names

    def has_next(self) -> bool:
        return self._idx < len(self._rows)

    def get_next(self):
        row = self._rows[self._idx]
        self._idx += 1
        return row




class LadybugBackend(GraphBackend):
    def __init__(self, db_path: str, read_only: bool = False):
        self._db_path  = db_path
        self._read_only = read_only
        self._db   = kuzu.Database(db_path, read_only=read_only)
        self._conn = kuzu.Connection(self._db)
        self._lock = threading.Lock()
        self._rel_allowed_props:  dict[str, set[str]] = {}
        self._node_allowed_props: dict[str, set[str]] = {}

    def close(self) -> None:
        with self._lock:
            try:
                del self._conn
            except AttributeError:
                pass
            try:
                del self._db
            except AttributeError:
                pass

    def _execute(self, query: str, params: dict | None = None):
        with self._lock:
            if self._read_only:
                # Reopen DB + connection so every query sees the latest checkpoint
                # written by the ingest server (or any other writer).
                _db   = kuzu.Database(self._db_path, read_only=True)
                _conn = kuzu.Connection(_db)
                try:
                    raw = _conn.execute(query, params) if params is not None else _conn.execute(query)
                    col_names = raw.get_column_names()
                    rows: list = []
                    while raw.has_next():
                        rows.append(raw.get_next())
                    return _MaterializedResult(col_names, rows)
                finally:
                    # Materializing before this point ensures no live reference to _conn/_db
                    # remains in the returned object, so the GC can free the file handles.
                    del _conn
                    del _db
            else:
                _conn = self._conn
                if params is not None:
                    return _conn.execute(query, params)
                return _conn.execute(query)

    def setup(self, nodes: dict, rels: list[dict]) -> None:
        for label, node_def in nodes.items():
            _safe_ident(label)
            props = node_def.get("properties", [])
            pk = _primary_key(node_def)
            for p in props:
                _safe_ident(p["name"])
            # Add system provenance field "run" (first-write-wins; not part of schema YAML)
            sys_props = list(props)
            if not any(p["name"] == "run" for p in sys_props):
                sys_props = sys_props + [{"name": "run", "type": "STRING"}]
            cols = ", ".join(f"{p['name']} {p['type']}" for p in sys_props)
            self._execute(
                f"CREATE NODE TABLE IF NOT EXISTS {label}({cols}, PRIMARY KEY ({pk}))"
            )
            self._node_allowed_props[label] = {p["name"] for p in sys_props}
            self._check_schema_drift(label, props)  # pass original props to avoid false drift warning

        for rel in rels:
            _safe_ident(rel["rel_type"])
            _safe_ident(rel["from_node"])
            _safe_ident(rel["to_node"])
            rel_prop_list = list(rel.get("properties", []))
            # Always include source_doc for idempotent re-run dedup
            if not any(p["name"] == "source_doc" for p in rel_prop_list):
                rel_prop_list.append({"name": "source_doc", "type": "STRING"})
            # Always include run for provenance
            if not any(p["name"] == "run" for p in rel_prop_list):
                rel_prop_list.append({"name": "run", "type": "STRING"})
            # Always include supporting_quote for evidence traceability
            if not any(p["name"] == "supporting_quote" for p in rel_prop_list):
                rel_prop_list.append({"name": "supporting_quote", "type": "STRING"})
            # Always include manually_added to distinguish human entries from extraction
            if not any(p["name"] == "manually_added" for p in rel_prop_list):
                rel_prop_list.append({"name": "manually_added", "type": "BOOLEAN"})
            # Always include triple_color so NVL / downstream tools can style by review status
            if not any(p["name"] == "triple_color" for p in rel_prop_list):
                rel_prop_list.append({"name": "triple_color", "type": "STRING"})
            # Always include triple_id (the raw triple's stable _id) so re-runs can
            # delete the prior version of an edge by identity, independent of which
            # nodes it currently connects (survives a review entity-correction).
            if not any(p["name"] == "triple_id" for p in rel_prop_list):
                rel_prop_list.append({"name": "triple_id", "type": "STRING"})
            for p in rel_prop_list:
                _safe_ident(p["name"])
            prop_cols = ", " + ", ".join(f"{p['name']} {p['type']}" for p in rel_prop_list)
            self._execute(
                f"CREATE REL TABLE IF NOT EXISTS {rel['rel_type']}"
                f"(FROM {rel['from_node']} TO {rel['to_node']}{prop_cols})"
            )
            self._rel_allowed_props[rel["rel_type"]] = {p["name"] for p in rel_prop_list}

    def _check_schema_drift(self, label: str, expected_props: list[dict]) -> None:
        try:
            res = self._execute(f"CALL table_info('{label}') RETURN *")
            existing: set[str] = set()
            while res.has_next():
                row = res.get_next()
                existing.add(row[1])
            missing = [p["name"] for p in expected_props if p["name"] not in existing]
            if missing:
                print(
                    f"  [WARN] schema drift on '{label}': "
                    f"column(s) {missing} defined in schema.yaml but absent from the DB. "
                    "Delete the database file and re-run to apply schema changes."
                )
        except Exception:
            pass

    def node_exists(self, label: str, pk_col: str, pk_val: str) -> bool:
        res = self._execute(
            f"MATCH (n:{label} {{{pk_col}: $v}}) RETURN count(n)", {"v": pk_val}
        )
        return res.get_next()[0] > 0

    def upsert_node(self, label: str, pk_col: str, props: dict) -> None:
        allowed = self._node_allowed_props.get(label)
        if allowed:
            props = {k: v for k, v in props.items() if k in allowed}
        for k in props:
            _safe_ident(k)
        col_str = ", ".join(f"{k}: ${k}" for k in props)
        try:
            self._execute(f"CREATE (:{label} {{{col_str}}})", props)
        except Exception as e:
            msg = str(e).lower()
            is_pk_violation = "already exists" in msg or (
                "duplicate" in msg and ("key" in msg or "primary" in msg)
            )
            if not is_pk_violation:
                raise
            # PK violation: node already exists — swallowed intentionally (upsert-by-ignore)

    def create_edge(
        self,
        from_label: str, from_pk: str, from_val: str,
        to_label: str, to_pk: str, to_val: str,
        rel_type: str,
        rel_props: dict | None = None,
    ) -> None:
        # CONTRACT: from_val / to_val MUST be the primary-key values of the from/to nodes
        # (e.g. CUI, LEI), NOT display names.  Passing a non-PK value silently matches
        # the first node whose PK happens to equal that string — B1.
        # Source-aware last-write-wins. The DELETE that clears the prior version of
        # THIS edge before re-create is chosen in preference order:
        #   1. {triple_id, source_doc} — endpoint-INDEPENDENT, so a review correction
        #      that changed the from/to PK still removes the stale old-identity edge.
        #   2. node-pair + source_doc — legacy edges with no triple_id.
        #   3. node-pair only — rel table has no source_doc property at all.
        # source_doc scoping is preserved in every case so other documents' edges coexist.
        _source_val = (rel_props or {}).get("source_doc")
        _triple_id  = (rel_props or {}).get("triple_id")
        if _triple_id is not None and _source_val is not None:
            del_query = (
                f"MATCH ()-[r:{rel_type} "
                f"{{triple_id: $__kuzu_tid__, source_doc: $__kuzu_src__}}]->() DELETE r"
            )
            _del_params: dict = {"__kuzu_tid__": _triple_id, "__kuzu_src__": _source_val}
        else:
            _del_params = {"__kuzu_fv__": from_val, "__kuzu_tv__": to_val}
            _src_filter = ""
            if _source_val is not None:
                _del_params["__kuzu_src__"] = _source_val
                _src_filter = " {source_doc: $__kuzu_src__}"
            del_query = (
                f"MATCH (a:{from_label} {{{from_pk}: $__kuzu_fv__}})"
                f"-[r:{rel_type}{_src_filter}]->"
                f"(b:{to_label} {{{to_pk}: $__kuzu_tv__}}) DELETE r"
            )

        params = {"__kuzu_fv__": from_val, "__kuzu_tv__": to_val}
        prop_clause = ""
        if rel_props:
            allowed = self._rel_allowed_props.get(rel_type)
            filtered = {k: v for k, v in rel_props.items() if allowed is None or k in allowed}
            dropped = set(rel_props) - set(filtered)
            if dropped:
                print(f"  [warn] dropping unknown rel props for {rel_type}: {sorted(dropped)}", flush=True)
            if filtered:
                prop_clause = " {" + ", ".join(f"{k}: ${k}" for k in filtered) + "}"
                params.update(filtered)
        create_query = (
            f"MATCH (a:{from_label} {{{from_pk}: $__kuzu_fv__}}), (b:{to_label} {{{to_pk}: $__kuzu_tv__}}) "
            f"CREATE (a)-[:{rel_type}{prop_clause}]->(b)"
        )

        with self._lock:
            self._conn.execute("BEGIN TRANSACTION")
            try:
                self._conn.execute(del_query, _del_params)
                self._conn.execute(create_query, params)
                self._conn.execute("COMMIT")
            except Exception:
                try:
                    self._conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise

    def count_nodes(self, label: str) -> int:
        res = self._execute(f"MATCH (n:{label}) RETURN count(n)")
        return res.get_next()[0]

    def count_edges(self, rel_type: str) -> int:
        res = self._execute(f"MATCH ()-[r:{rel_type}]->() RETURN count(r)")
        return res.get_next()[0]

    def run_cypher(self, query: str, params: dict | None = None) -> list:
        res = self._execute(query, params)
        col_names = res.get_column_names()
        rows = []
        while res.has_next():
            rows.append(dict(zip(col_names, res.get_next())))
        return rows

    def delete_node(self, label: str, pk_col: str, pk_val: str) -> None:
        _safe_ident(label)
        _safe_ident(pk_col)
        self._execute(
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
        # Endpoint-independent delete by identity (see base.delete_edge docstring):
        # removes the stale edge even after a review correction changed the endpoints.
        if triple_id is not None and source_doc is not None:
            self._execute(
                f"MATCH ()-[r:{rel_type} {{triple_id: $tid, source_doc: $sd}}]->() DELETE r",
                {"tid": triple_id, "sd": source_doc},
            )
            return
        src_filter = " {source_doc: $sd}" if source_doc is not None else ""
        params: dict = {"fv": from_val, "tv": to_val}
        if source_doc is not None:
            params["sd"] = source_doc
        self._execute(
            f"MATCH (a:{from_label} {{{from_pk}: $fv}})"
            f"-[r:{rel_type}{src_filter}]->"
            f"(b:{to_label} {{{to_pk}: $tv}}) DELETE r",
            params,
        )
