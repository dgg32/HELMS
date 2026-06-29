#!/usr/bin/env python3
"""
MCP server exposing a knowledge graph as tools.

Compatible with Claude Desktop, Kilo Code (VS Code), and Claude Code CLI.

Configure via environment variables:
  GRAPH_BACKEND   backend: ladybug (default) or neo4j
  GRAPH_SCHEMA    path to schema YAML (required, e.g. projects/drug/schema.yaml)

  Ladybug:
    LADYBUG_DB_PATH path to LadybugDB database directory (required,
                    e.g. projects/drug/runs/20260525_170300/drug_kg.db)

  Neo4j:
    NEO4J_URI       bolt URI (default: neo4j://127.0.0.1:7687)
    NEO4J_USERNAME  (default: neo4j)
    NEO4J_PASSWORD  (required)

  Just-in-time document context:
    DOCS_DIR        directory containing *_raw.json files produced by extract.py.
                    Defaults to the parent of LADYBUG_DB_PATH (the run folder).
                    Used by list_documents / read_document / search_document.
"""
import atexit
import concurrent.futures
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from mcp.server.fastmcp import FastMCP

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_WRITE_VERBS_RE = re.compile(
    r"\b(CREATE|DELETE|DETACH|SET|REMOVE|MERGE|DROP|FOREACH|LOAD)\b"
    r"|CALL\s*\{"
    r"|apoc\.[a-z.]*\b(create|delete|remove|merge|set|update|add)\b",
    re.IGNORECASE,
)
_STRING_LITERAL_RE = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")
_CYPHER_MAX_ROWS = int(os.environ.get("CYPHER_MAX_ROWS", "1000"))
_CYPHER_TIMEOUT_S = int(os.environ.get("CYPHER_TIMEOUT_S", "30"))

# ── Config ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
GRAPH_BACKEND = os.environ.get("GRAPH_BACKEND", "ladybug")
SCHEMA_PATH = os.environ.get("GRAPH_SCHEMA")

# ── Globals — populated by _server_init() before mcp.run() ────────────────────

_nodes: dict = {}
_rels: list = []
_PROMPTS: dict = {}
SCHEMA_TEXT: str = ""
_backend = None
_docs_dir: Path | None = None  # directory containing *_raw.json files

from extract import load_schema as _load_schema, _ALL_PROMPTS as _extract_prompts  # noqa: E402

def _build_schema_text() -> str:
    backend_label = "LadybugDB" if GRAPH_BACKEND == "ladybug" else "Neo4j"
    lines = [f"{backend_label} graph schema (openCypher):\n", "Node tables:"]
    for label, node_def in _nodes.items():
        props = ", ".join(
            f"{p['name']} {p['type']}" + (" [PK]" if p.get("primary_key") else "")
            for p in node_def.get("properties", [])
        )
        lines.append(f"  {label}({props})")
    lines.append("\nRelationship tables:")
    for rel in _rels:
        rel_props = rel.get("properties", [])
        prop_str = ""
        if rel_props:
            prop_str = " {" + ", ".join(f"{p['name']} {p['type']}" for p in rel_props) + "}"
        lines.append(
            f"  (:{rel['from_node']})-[:{rel['rel_type']}{prop_str}]->(:{rel['to_node']})"
        )
    return "\n".join(lines)


def _server_init() -> None:
    """Validate config, load schema, connect backend, index document directory."""
    global _nodes, _rels, SCHEMA_TEXT, _backend, _docs_dir  # _PROMPTS is mutated in place, not rebound
    if not SCHEMA_PATH:
        raise SystemExit(
            "GRAPH_SCHEMA environment variable not set.\n"
            "Add GRAPH_SCHEMA=/path/to/your_schema.yaml to your .env file."
        )
    _nodes, _rels = _load_schema(SCHEMA_PATH)
    _PROMPTS.update(_extract_prompts)
    SCHEMA_TEXT = _build_schema_text()
    mcp._mcp_server.instructions = _PROMPTS["mcp_server"]["instructions"] + "\n\n" + SCHEMA_TEXT

    if GRAPH_BACKEND == "ladybug":
        _db_path = os.environ.get("LADYBUG_DB_PATH")
        if not _db_path:
            raise SystemExit(
                "LADYBUG_DB_PATH not set. Add it to .env, e.g.:\n"
                "  LADYBUG_DB_PATH=projects/drug/runs/20260525_170300/drug_kg.db"
            )
        _db_uri = _db_path
    elif GRAPH_BACKEND == "neo4j":
        _db_uri = os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687")
    else:
        raise SystemExit(f"Unknown GRAPH_BACKEND: {GRAPH_BACKEND!r}. Available: ladybug, neo4j")
    from backends import get_backend as _get_backend
    _backend = _get_backend(GRAPH_BACKEND, _db_uri, _nodes, _rels, read_only=True, setup=False)
    atexit.register(_backend.close)

    # ── Document directory for JIT context ────────────────────────────────────
    _docs_dir_str = os.environ.get("DOCS_DIR") or (
        str(Path(os.environ.get("LADYBUG_DB_PATH", "")).parent)
        if os.environ.get("LADYBUG_DB_PATH") else None
    )
    if _docs_dir_str:
        _candidate = Path(_docs_dir_str)
        if _candidate.exists():
            _docs_dir = _candidate
            _n = len(list(_docs_dir.glob("*_raw.json")))
            print(f"[docs] JIT context: {_n} _raw.json file(s) in {_docs_dir}", file=sys.stderr)
        else:
            print(f"[docs] DOCS_DIR not found: {_docs_dir_str}", file=sys.stderr)


# ── MCP server ─────────────────────────────────────────────────────────────────

mcp = FastMCP("Knowledge Graph")


@mcp.resource("schema://graph")
@mcp.tool()
def get_schema() -> str:
    """Returns the full graph schema (node tables, relationship tables, property types)."""
    return SCHEMA_TEXT


@mcp.tool()
def run_cypher(query: str) -> str:
    """
    Execute an openCypher query against the knowledge graph.

    Returns a JSON array of result rows. On error, returns {"error": "..."}.

    Examples:
      MATCH (n) RETURN label(n), count(*) ORDER BY count(*) DESC
      MATCH (a)-[r]->(b) RETURN label(a), a.name, type(r), label(b), b.name LIMIT 20
      MATCH (n:Drug) RETURN n.name ORDER BY n.name
    """
    _stripped = _STRING_LITERAL_RE.sub("", query)
    if _WRITE_VERBS_RE.search(_stripped) or ";" in _stripped:
        return json.dumps({"error": "Write queries are not permitted. Use read-only MATCH queries."})
    _pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        _future = _pool.submit(_backend.run_cypher, query)
        try:
            rows = _future.result(timeout=_CYPHER_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            return json.dumps({"error": f"Query timed out after {_CYPHER_TIMEOUT_S}s."})
        truncated = len(rows) > _CYPHER_MAX_ROWS
        if truncated:
            rows = rows[:_CYPHER_MAX_ROWS]
        result = json.loads(json.dumps(rows, default=str))
        if truncated:
            result.append({"__truncated__": f"results capped at {_CYPHER_MAX_ROWS} rows"})
        return json.dumps(result)
    except Exception as exc:
        return json.dumps({"error": str(exc)})
    finally:
        _pool.shutdown(wait=False, cancel_futures=True)


@mcp.tool()
def get_node_count(label: str) -> str:
    """
    Return the number of nodes for a given label (e.g. 'Corporation').
    Convenience wrapper around run_cypher.
    """
    if not _IDENT_RE.match(label):
        return json.dumps({"error": f"Invalid label: {label!r}"})
    try:
        count = _backend.count_nodes(label)
        return json.dumps({"label": label, "count": count})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ── Just-in-time document context tools ───────────────────────────────────────

@mcp.tool()
def list_documents() -> str:
    """
    List all source documents available for just-in-time context retrieval.

    Returns each document's stem name and raw file name. Use the stem with
    read_document or search_document to load content on demand.
    Call this first to discover what documents are indexed before reading them.
    """
    if _docs_dir is None:
        return json.dumps({"error": "No document directory found. Set DOCS_DIR or LADYBUG_DB_PATH."})
    raw_files = sorted(_docs_dir.glob("*_raw.json"))
    if not raw_files:
        return json.dumps([])
    docs = []
    for rf in raw_files:
        stem = rf.name[: -len("_raw.json")]
        try:
            size_kb = round(rf.stat().st_size / 1024, 1)
        except OSError:
            size_kb = 0
        docs.append({"doc_stem": stem, "raw_file": rf.name, "file_kb": size_kb})
    return json.dumps(docs)


@mcp.tool()
def read_document(doc_stem: str, offset: int = 0, limit: int = 4000) -> str:
    """
    Read a portion of a source document's text for just-in-time context.

    doc_stem : document stem from list_documents (e.g. 'ozempic_label').
    offset   : character position to start reading from (default 0).
    limit    : max characters to return (default 4000, capped at 8000).

    Returns the text slice plus total_chars and has_more so you can paginate.
    Call repeatedly with increasing offset to read large documents in chunks.
    """
    if _docs_dir is None:
        return json.dumps({"error": "No document directory configured."})
    doc_stem = Path(doc_stem).name  # strip any directory traversal components
    raw_path = _docs_dir / f"{doc_stem}_raw.json"
    if not raw_path.exists():
        return json.dumps({"error": f"Document not found: {doc_stem!r}. Use list_documents() to see available stems."})
    try:
        meta = json.loads(raw_path.read_text(encoding="utf-8"))
        doc_text = meta.get("doc_text", "")
        if not doc_text:
            return json.dumps({"error": f"No text content in {doc_stem}_raw.json"})
        limit = min(max(1, limit), 8000)
        total = len(doc_text)
        offset = max(0, min(offset, total))
        slice_text = doc_text[offset: offset + limit]
        return json.dumps({
            "doc": meta.get("doc", f"{doc_stem}.md"),
            "offset": offset,
            "chars_returned": len(slice_text),
            "total_chars": total,
            "has_more": offset + limit < total,
            "text": slice_text,
        })
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def search_document(doc_stem: str, pattern: str, context_chars: int = 300) -> str:
    """
    Search for a pattern inside a source document and return matching passages.

    doc_stem      : document stem from list_documents.
    pattern       : text to search for (case-insensitive substring match).
    context_chars : characters of surrounding context per match (default 300, max 600).

    Returns up to 10 matching passages with their character offsets.
    Use this to pull targeted evidence before committing to a full read_document call.
    """
    if _docs_dir is None:
        return json.dumps({"error": "No document directory configured."})
    doc_stem = Path(doc_stem).name  # strip any directory traversal components
    raw_path = _docs_dir / f"{doc_stem}_raw.json"
    if not raw_path.exists():
        return json.dumps({"error": f"Document not found: {doc_stem!r}. Use list_documents() to see available stems."})
    try:
        meta = json.loads(raw_path.read_text(encoding="utf-8"))
        doc_text = meta.get("doc_text", "")
        if not doc_text:
            return json.dumps({"error": f"No text content in {doc_stem}_raw.json"})

        context_chars = max(50, min(context_chars, 600))
        pattern_lower = pattern.lower()
        text_lower = doc_text.lower()

        passages = []
        start = 0
        while len(passages) < 10:
            pos = text_lower.find(pattern_lower, start)
            if pos == -1:
                break
            ctx_start = max(0, pos - context_chars)
            ctx_end = min(len(doc_text), pos + len(pattern) + context_chars)
            passages.append({
                "offset": pos,
                "match_start_in_passage": pos - ctx_start,
                "passage": doc_text[ctx_start:ctx_end],
            })
            start = pos + 1

        return json.dumps({
            "doc": meta.get("doc", f"{doc_stem}.md"),
            "pattern": pattern,
            "match_count": len(passages),
            "passages": passages,
        })
    except Exception as exc:
        return json.dumps({"error": str(exc)})


if __name__ == "__main__":
    _server_init()
    try:
        mcp.run()
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
