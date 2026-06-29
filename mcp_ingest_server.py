#!/usr/bin/env python3
"""
MCP server for ingesting PDFs into the knowledge graph.

Companion to mcp_server.py (read-only query server). This server accepts
PDF file paths, runs the full pipeline (convert → extract → apply) in the
background, and tracks per-run status.

When a user attaches or references a PDF, call ingest_pdf with its path.
The pipeline runs in the background; poll poll_ingest until status is
"done" or "error", then report back.

Configure via environment variables:
  INGEST_PROJECT   path to the project folder (required)
                   must contain schema.yaml (or set INGEST_SCHEMA)
  INGEST_SCHEMA    path to schema YAML (default: <INGEST_PROJECT>/schema.yaml)
  INGEST_DB        shared LadybugDB that all ingests accumulate into
                   (default: <INGEST_PROJECT>/<schema_stem>_kg.db)
                   set LADYBUG_DB_PATH in mcp_server.py to the same path
  INGEST_CONVERTER default PDF converter: pymupdf4llm (default) | llamaparse
  INGEST_FILTER    default filter level: loose | moderate (default) | strict
"""
import asyncio
import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from mcp.server.fastmcp import FastMCP

# ── Config ────────────────────────────────────────────────────────────────────

_PROJECT_STR = os.environ.get("INGEST_PROJECT", "").strip()
_SCHEMA_STR   = os.environ.get("INGEST_SCHEMA",   "").strip()
_DB_STR       = os.environ.get("INGEST_DB",        "").strip()

_DEFAULT_CONVERTER = os.environ.get("INGEST_CONVERTER", "pymupdf4llm").strip()
_DEFAULT_FILTER    = os.environ.get("INGEST_FILTER",    "moderate").strip()

# Resolved in _server_init()
_PROJECT:    Path | None = None
_SCHEMA:     Path | None = None
_SHARED_DB:  Path | None = None

# ── Run state ─────────────────────────────────────────────────────────────────

_runs: dict[str, dict] = {}
_runs_lock = threading.Lock()


def _status_file(run_dir: Path) -> Path:
    return run_dir / "_ingest_status.json"


def _save_status(run_id: str, run_dir: Path, data: dict) -> None:
    with _runs_lock:
        _runs[run_id] = dict(data)
    try:
        _status_file(run_dir).write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def _load_status(run_id: str) -> dict | None:
    with _runs_lock:
        if run_id in _runs:
            return dict(_runs[run_id])
    # Fall back to disk (server restart case)
    if _PROJECT:
        sf = _status_file(_PROJECT / "runs" / run_id)
        if sf.exists():
            try:
                data = json.loads(sf.read_text(encoding="utf-8"))
                with _runs_lock:
                    _runs[run_id] = data
                return data
            except Exception:
                pass
    return None


# ── Background pipeline runner ────────────────────────────────────────────────

def _run_pipeline(
    run_id: str,
    run_dir: Path,
    pdf_path: Path,
    converter: str,
    filter_level: str,
    pages: str = "",
) -> None:
    """Background thread: convert → extract → apply for one PDF."""
    import apply_graph
    import convert_pdf
    import extract
    import pipeline_meta as _pm
    from pipeline_ns import build_apply_ns, build_convert_ns, build_extract_ns
    from pipeline_orchestrator import PipelineOrchestrator

    schema_str  = str(_SCHEMA)
    db_str      = str(_SHARED_DB)
    run_dir_str = str(run_dir)
    md_path     = run_dir / (pdf_path.stem + ".md")

    meta_dict = _pm.load_meta(str(_PROJECT / "meta.yaml"))
    page_list = [p.strip() for p in pages.split(",") if p.strip()]
    if page_list:
        meta_dict.setdefault("pages", {})[pdf_path.stem] = {"include": page_list}
    meta_str = None
    if meta_dict:
        tmp_meta_path = run_dir / "_meta.yaml"
        _pm.save_meta(meta_dict, tmp_meta_path)
        meta_str = str(tmp_meta_path)

    orch = PipelineOrchestrator([
        ("convert", convert_pdf.main, lambda: build_convert_ns(
            input=str(pdf_path),
            output=run_dir_str,
            converter=converter,
            force=True,
        )),
        ("extract", extract.main, lambda: build_extract_ns(
            schema=schema_str,
            input=str(md_path),
            output_dir=run_dir_str,
            filter=filter_level,
            force=True,
            meta=meta_str,
        )),
        ("apply", apply_graph.main, lambda: build_apply_ns(
            apply=run_dir_str,
            schema=schema_str,
            db=db_str,
            filter=filter_level,
            run_id=run_id,
        )),
    ])

    # Import pipeline_runner to install thread-local stdout routing (side-effect on first import).
    # This lets each ingest thread write to its own log file without touching the MCP stdio channel.
    from concurrent.futures import ThreadPoolExecutor as _TPE
    from pipeline_runner import _tls_err, _tls_out

    log_path = run_dir / "pipeline.log"
    try:
        with open(log_path, "w", encoding="utf-8", buffering=1) as _log:
            _tls_out.set(_log)
            _tls_err.set(_log)
            # Use a custom executor so asyncio.to_thread() worker threads (e.g. semantic_check,
            # GLEIF lookups) also route their print() calls to the log file rather than stdout.
            def _worker_init() -> None:
                _tls_out.set(_log)
                _tls_err.set(_log)

            loop = asyncio.new_event_loop()
            executor = _TPE(initializer=_worker_init)
            loop.set_default_executor(executor)
            try:
                loop.run_until_complete(orch.run_all(
                    on_step_start=lambda k: print(
                        f"[ingest:{run_id}] step={k}", flush=True
                    ),
                    on_step_error=lambda k, e: print(
                        f"[ingest:{run_id}] {k} error: {e}", flush=True
                    ),
                ))
            finally:
                executor.shutdown(wait=True)
                loop.close()
                _tls_out.clear()
                _tls_err.clear()

        # Read extraction stats from run.jsonl (created by the pipeline)
        stats: dict = {}
        run_log = run_dir / "run.jsonl"
        if run_log.exists():
            try:
                total_triples = 0
                color_dist: dict = {}
                for line in run_log.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue  # skip one corrupt line, keep the rest of the stats
                    if ev.get("msg") == "doc_done":
                        total_triples += ev.get("triple_count", 0)
                    if ev.get("msg") == "semantic_check_done":
                        for color, n in ev.get("color_dist", {}).items():
                            color_dist[color] = color_dist.get(color, 0) + n
                if total_triples or color_dist:
                    stats = {
                        "total_triples": total_triples,
                        **{f"{c}_count": n for c, n in color_dist.items()},
                    }
            except Exception:
                pass

        _save_status(run_id, run_dir, {
            "run_id":       run_id,
            "status":       "done",
            "pdf":          pdf_path.name,
            "pdf_path":     str(pdf_path),
            "run_dir":      run_dir_str,
            "db":           db_str,
            "converter":    converter,
            "filter_level": filter_level,
            "finished_at":  datetime.now(timezone.utc).isoformat(),
            **stats,
        })

    except Exception as exc:
        _save_status(run_id, run_dir, {
            "run_id":      run_id,
            "status":      "error",
            "pdf":         pdf_path.name,
            "pdf_path":    str(pdf_path),
            "run_dir":     run_dir_str,
            "error":       str(exc),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })


# ── MCP server ────────────────────────────────────────────────────────────────

mcp = FastMCP("Knowledge Graph Ingest")


@mcp.tool()
def ingest_pdf(
    pdf_path: str,
    converter: str = "",
    filter_level: str = "",
    pages: str = "",
) -> str:
    """
    Ingest a PDF into the knowledge graph.

    Runs convert → extract → apply in the background. Returns a run_id
    immediately. Call poll_ingest(run_id) every ~30 seconds until status
    is "done" or "error", then report the result to the user.

    pdf_path     : absolute (or ~-expanded) path to the PDF file.
    converter    : "pymupdf4llm" (default, fast, CPU-only) or "llamaparse"
                   (cloud API, best for complex layouts with tables/forms).
    filter_level : triples to apply — "loose" (all), "moderate" (default,
                   green+yellow), "strict" (green only).
    pages        : optional page filter, 1-based, e.g. "1, 2-6". Comma-separated
                   page numbers and/or ranges. Empty = whole document.
    """
    if _PROJECT is None or _SCHEMA is None or _SHARED_DB is None:
        return json.dumps({
            "error": "Server not initialised. Set INGEST_PROJECT in the environment."
        })

    pdf = Path(pdf_path).expanduser().resolve()
    if not pdf.exists():
        return json.dumps({"error": f"File not found: {pdf_path!r}"})
    if pdf.suffix.lower() != ".pdf":
        return json.dumps({"error": f"Not a PDF: {pdf_path!r}"})

    _conv = converter.strip() or _DEFAULT_CONVERTER
    _filt = filter_level.strip() or _DEFAULT_FILTER
    if _conv not in ("pymupdf4llm", "llamaparse"):
        return json.dumps({
            "error": f"Unknown converter {_conv!r}. Use 'pymupdf4llm' or 'llamaparse'."
        })
    if _filt not in ("loose", "moderate", "strict"):
        return json.dumps({
            "error": f"Unknown filter_level {_filt!r}. Use 'loose', 'moderate', or 'strict'."
        })

    # Millisecond suffix prevents collisions when called twice in the same second
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S_") + datetime.now().strftime("%f")[:3]
    run_dir = _PROJECT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _save_status(run_id, run_dir, {
        "run_id":       run_id,
        "status":       "running",
        "pdf":          pdf.name,
        "pdf_path":     str(pdf),
        "run_dir":      str(run_dir),
        "db":           str(_SHARED_DB),
        "converter":    _conv,
        "filter_level": _filt,
        "pages":        pages.strip(),
        "started_at":   datetime.now(timezone.utc).isoformat(),
    })

    threading.Thread(
        target=_run_pipeline,
        args=(run_id, run_dir, pdf, _conv, _filt, pages),
        daemon=True,
        name=f"ingest-{run_id}",
    ).start()

    return json.dumps({
        "run_id":  run_id,
        "status":  "started",
        "db":      str(_SHARED_DB),
        "message": (
            f"Pipeline started for {pdf.name!r}. "
            f"Call poll_ingest('{run_id}') to check progress (~1–5 min)."
        ),
    })


@mcp.tool()
def poll_ingest(run_id: str) -> str:
    """
    Check the status of a PDF ingest run.

    run_id : the run_id returned by ingest_pdf.

    Returns status ("running", "done", or "error").
    When done, also returns extraction stats: total_triples, green_count,
    yellow_count, red_count, entities_resolved, entity_resolution_rate.
    When error, returns the error message.
    """
    data = _load_status(run_id.strip())
    if data is None:
        return json.dumps({
            "error": (
                f"Run {run_id!r} not found. "
                "Use list_ingests() to see all runs."
            )
        })
    return json.dumps(data)


@mcp.tool()
def list_ingests() -> str:
    """
    List all past and current ingest runs, newest first.

    Shows run_id, status, pdf filename, started_at, and finished_at.
    Scans the runs folder so runs from previous server sessions appear too.
    """
    if _PROJECT is None:
        return json.dumps({
            "error": "Server not initialised. Set INGEST_PROJECT."
        })

    runs_dir = _PROJECT / "runs"
    if not runs_dir.exists():
        return json.dumps([])

    summaries: list[dict] = []
    seen: set[str] = set()

    for sf in sorted(runs_dir.glob("*/_ingest_status.json"), reverse=True):
        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
            rid  = data.get("run_id", sf.parent.name)
            seen.add(rid)
            summaries.append({
                "run_id":      rid,
                "status":      data.get("status"),
                "pdf":         data.get("pdf"),
                "started_at":  data.get("started_at"),
                "finished_at": data.get("finished_at"),
            })
        except Exception:
            pass

    with _runs_lock:
        for rid, data in _runs.items():
            if rid not in seen:
                summaries.append({
                    "run_id":      rid,
                    "status":      data.get("status"),
                    "pdf":         data.get("pdf"),
                    "started_at":  data.get("started_at"),
                    "finished_at": data.get("finished_at"),
                })

    summaries.sort(key=lambda r: r.get("run_id") or "", reverse=True)
    return json.dumps(summaries)


# ── Init ──────────────────────────────────────────────────────────────────────

def _server_init() -> None:
    global _PROJECT, _SCHEMA, _SHARED_DB

    if not _PROJECT_STR:
        raise SystemExit(
            "INGEST_PROJECT not set.\n"
            "Add INGEST_PROJECT=/path/to/project (e.g. projects/supplychain) to .env."
        )
    _PROJECT = Path(_PROJECT_STR).expanduser().resolve()
    if not _PROJECT.exists():
        raise SystemExit(f"INGEST_PROJECT directory not found: {_PROJECT}")

    _SCHEMA = (
        Path(_SCHEMA_STR).expanduser().resolve()
        if _SCHEMA_STR
        else _PROJECT / "schema.yaml"
    )
    if not _SCHEMA.exists():
        raise SystemExit(
            f"Schema not found: {_SCHEMA}\n"
            "Set INGEST_SCHEMA or add schema.yaml to INGEST_PROJECT."
        )

    if _DB_STR:
        _SHARED_DB = Path(_DB_STR).expanduser().resolve()
    else:
        schema_stem = _SCHEMA.stem.replace("_schema", "")
        _SHARED_DB = _PROJECT / f"{schema_stem}_kg.db"
    _SHARED_DB.parent.mkdir(parents=True, exist_ok=True)

    mcp._mcp_server.instructions = (
        "You are connected to the Knowledge Graph Ingest server. "
        "When a user shares a PDF file path, call ingest_pdf with that path. "
        "After calling ingest_pdf, poll poll_ingest every 30 seconds until "
        "status is 'done' or 'error', then report the result to the user "
        "(triples added, entity resolution rate, or error details). "
        "Use list_ingests to review all past ingests."
    )

    print(f"[ingest] project:   {_PROJECT}", file=sys.stderr)
    print(f"[ingest] schema:    {_SCHEMA}", file=sys.stderr)
    print(f"[ingest] shared DB: {_SHARED_DB}", file=sys.stderr)
    print(f"[ingest] converter: {_DEFAULT_CONVERTER}", file=sys.stderr)
    print(f"[ingest] filter:    {_DEFAULT_FILTER}", file=sys.stderr)


if __name__ == "__main__":
    _server_init()
    try:
        mcp.run()
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
