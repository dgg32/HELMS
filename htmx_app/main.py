#!/usr/bin/env python3
"""HELMS – FastAPI + HTMX + Alpine.js UI (experimental branch).

Run:
    cd cognee_poc
    pip install fastapi uvicorn jinja2 python-multipart
    python htmx_app/main.py

Then open http://localhost:8000
"""
from __future__ import annotations

import asyncio
import datetime
import json
import queue
import sys
import time
import uuid
from html import escape
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ── bootstrap: add repo root to sys.path ─────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import apply_graph as _apply_module
import convert_pdf as _convert_module
import extract as _extract_module
import pipeline_runner as _pr
import review_layer as _rl
import agents.extraction_agent as _agent_module
from llm_client import LLM_MAX_COMPLETION_TOKENS, _deployment as _llm_deployment, get_model_env, get_models as _get_llm_models
from pipeline_ns import build_apply_ns, build_convert_ns, build_extract_ns

# ── app setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="HELMS")
_TMPL_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TMPL_DIR))
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


# ── in-memory state (single-user tool) ───────────────────────────────────────

_STATE_FILE = Path(__file__).parent / ".helms_state.json"


class _AppState:
    def __init__(self) -> None:
        self.project_folder: str = str(_ROOT / "projects" / "drug")
        self.input_folder: str   = ""
        self.run_id: str         = ""
        self.schema_path: str    = ""
        self.filter_level: str   = "moderate"
        self.converter: str      = "pymupdf4llm"
        self.backend: str        = "ladybug"
        self.llm_model: str      = _llm_deployment()
        # Grader model for the semantic check. Empty = same as llm_model (no
        # separate model); a name from get_models() runs the check on that model
        # (and provider) via SEMANTIC_CHECK_MODEL so no model grades its own output.
        self.semantic_check_model: str = ""
        self.dry_run: bool       = False

        # runner registry: token → (PipelineRunner, queue.Queue, step_key, done, ok)
        self._runners: dict[str, dict] = {}

        self._load()

    def _load(self) -> None:
        if not _STATE_FILE.exists():
            return
        try:
            data = json.loads(_STATE_FILE.read_text())
            for key in ("project_folder", "run_id", "filter_level", "converter", "backend", "llm_model"):
                if data.get(key):
                    setattr(self, key, data[key])
            # semantic_check_model is loaded separately: "" is a valid (default)
            # value meaning "same as llm_model", so the truthy guard above skips it.
            if "semantic_check_model" in data:
                self.semantic_check_model = data["semantic_check_model"]
            if self.project_folder:
                p = self._proj()
                rd = p / "raw_documents"
                self.input_folder = str(rd) if rd.is_dir() else str(p)
        except Exception:
            pass

    def save(self) -> None:
        try:
            import tempfile, os as _os
            data = json.dumps({
                "project_folder": self.project_folder,
                "run_id":         self.run_id,
                "filter_level":   self.filter_level,
                "converter":      self.converter,
                "backend":        self.backend,
                "llm_model":      self.llm_model,
                "semantic_check_model": self.semantic_check_model,
            }, indent=2)
            with tempfile.NamedTemporaryFile(
                mode="w", dir=_STATE_FILE.parent, delete=False, suffix=".tmp"
            ) as f:
                f.write(data)
                tmp = f.name
            _os.replace(tmp, _STATE_FILE)
        except Exception as e:
            print(f"[warn] state save failed: {e}", flush=True)

    # ── project helpers ───────────────────────────────────────────────────────

    def _proj(self) -> Path:
        return Path(self.project_folder)

    def schema(self) -> str:
        if self.schema_path and Path(self.schema_path).exists():
            return self.schema_path
        return str(self._proj() / "schema.yaml")

    def meta(self) -> str | None:
        p = self._proj() / "meta.yaml"
        return str(p) if p.exists() else None

    def run_folder(self) -> Path | None:
        if not self.run_id or self.run_id == "new":
            return None
        return self._proj() / "runs" / self.run_id

    def ensure_run_folder(self) -> Path:
        if not self.run_id or self.run_id == "new":
            self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save()
        rf = self._proj() / "runs" / self.run_id
        rf.mkdir(parents=True, exist_ok=True)
        cfg = rf / "run_config.json"
        if not cfg.exists():
            cfg.write_text(json.dumps({
                "run_id": self.run_id,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "project_folder": str(self._proj()),
                "schema": self.schema(),
                "filter": self.filter_level,
                "converter": self.converter,
                "backend": self.backend,
                "input_folder": self.input_folder,
                "llm_model": self.llm_model,
                "llm_models": {
                    "extraction":      self.llm_model,
                    "node_resolution": self.llm_model,
                    "semantic_check":  self.semantic_check_model or self.llm_model,
                },
                "llm_max_completion_tokens": LLM_MAX_COMPLETION_TOKENS,
            }, indent=2))
        return rf

    def list_runs(self) -> list[str]:
        runs_dir = self._proj() / "runs"
        if not runs_dir.is_dir():
            return []
        return sorted(
            (d.name for d in runs_dir.iterdir() if d.is_dir()),
            reverse=True,
        )

    def list_runs_with_status(self) -> list[dict]:
        runs_dir = self._proj() / "runs"
        if not runs_dir.is_dir():
            return []
        result = []
        for d in sorted(runs_dir.iterdir(), key=lambda x: x.name, reverse=True):
            if not d.is_dir():
                continue
            if list(d.glob("*.db")):
                status = "write"
            elif list(d.glob("*_raw.json")):
                status = "extract"
            elif list(d.glob("*.md")):
                status = "convert"
            else:
                status = "new"
            result.append({"run_id": d.name, "status": status})
        return result

    def list_review_files(self) -> list[Path]:
        rf = self.run_folder()
        if rf is None or not rf.is_dir():
            return []
        return sorted(rf.glob("*_raw.json"))

    # ── runner helpers ────────────────────────────────────────────────────────

    def start_runner(self, fn, args, step_key: str, env: dict | None = None) -> str:
        now = time.monotonic()
        # Prune completed runners and those running for more than 30 minutes (hung threads)
        stale = [
            k for k, e in self._runners.items()
            if e.get("done") or (now - e.get("started_at", now)) > 1800
        ]
        for t in stale:
            del self._runners[t]
        token = uuid.uuid4().hex
        q: queue.Queue = queue.Queue()
        runner = _pr.PipelineRunner(fn, args, q, env=env)
        runner.start()
        self._runners[token] = {
            "runner": runner,
            "queue": q,
            "step": step_key,
            "done": False,
            "ok": None,
            "lines": [],
            "started_at": now,
        }
        return token

    def get_runner(self, token: str) -> dict | None:
        return self._runners.get(token)

    def db_path(self) -> str:
        schema_stem = Path(self.schema()).stem.replace("_schema", "")
        rf = self.run_folder()
        if rf:
            return str(rf / f"{schema_stem}_kg.db")
        return str(self._proj() / f"{schema_stem}_kg.db")


_state = _AppState()


# ── helpers ───────────────────────────────────────────────────────────────────

def _list_projects() -> list[str]:
    proj_dir = _ROOT / "projects"
    if not proj_dir.is_dir():
        return []
    # Skip dot/underscore-prefixed dirs (hidden, plus test/temp scratch like
    # `_qtest_*`) so they never appear as selectable projects.
    return sorted(
        d.name for d in proj_dir.iterdir()
        if d.is_dir() and not d.name.startswith((".", "_"))
    )


def _resolve_input_folder(proj: Path) -> str:
    raw_docs = proj / "raw_documents"
    return str(raw_docs) if raw_docs.is_dir() else str(proj)


def _scan_md(folder: str) -> list[str]:
    p = Path(folder)
    return sorted(str(f) for f in p.glob("*.md")) if p.is_dir() else []


def _triple_display_name(raw_path: Path) -> str:
    return raw_path.name.removesuffix("_raw.json")


# ── routes: pages ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    projects = _list_projects()
    if not _state.project_folder and projects:
        _state.project_folder = str(_ROOT / "projects" / projects[0])
    _state.input_folder = _resolve_input_folder(_state._proj())
    return templates.TemplateResponse(request, "index.html", {
        "projects": projects,
        "selected_project": _state._proj().name,
        "runs": _state.list_runs_with_status(),
        "run_id": _state.run_id,
        "filter_level": _state.filter_level,
        "converter": _state.converter,
        "backend": _state.backend,
        "llm_models": _get_llm_models(),
        "llm_model": _state.llm_model,
        "semantic_check_model": _state.semantic_check_model,
        "review_files": _state.list_review_files(),
        "triple_display_name": _triple_display_name,
    })


# ── routes: project / run selection ──────────────────────────────────────────

@app.post("/select-project", response_class=HTMLResponse)
async def select_project(request: Request, project: str = Form(...)):
    proj_path = _ROOT / "projects" / project
    if not proj_path.is_dir():
        return HTMLResponse(
            f"<p class='text-red-500 text-xs p-1'>Project '{escape(project)}' not found.</p>"
        )
    _state.project_folder = str(proj_path)
    _state.input_folder = _resolve_input_folder(_state._proj())
    _state.run_id = ""
    _state.schema_path = ""
    _state.save()
    resp = templates.TemplateResponse(request, "partials/sidebar_runs.html", {
        "runs": _state.list_runs_with_status(),
        "run_id": _state.run_id,
    })
    resp.headers["HX-Trigger"] = "helms-project-changed"
    return resp


@app.post("/select-run", response_class=HTMLResponse)
async def select_run(request: Request, run_id: str = Form(...)):
    _state.run_id = run_id
    _state.save()
    return templates.TemplateResponse(request, "partials/step2_files.html", {
        "review_files": _state.list_review_files(),
        "triple_display_name": _triple_display_name,
    })


@app.get("/sidebar/runs", response_class=HTMLResponse)
async def sidebar_runs(request: Request):
    """Refresh the run selector after a new run is created."""
    return templates.TemplateResponse(request, "partials/sidebar_runs.html", {
        "runs": _state.list_runs_with_status(),
        "run_id": _state.run_id,
    })


@app.get("/review/files", response_class=HTMLResponse)
async def list_review_files(request: Request):
    """Reload review file list (called after extract completes)."""
    return templates.TemplateResponse(request, "partials/step2_files.html", {
        "review_files": _state.list_review_files(),
        "triple_display_name": _triple_display_name,
    })


# ── routes: pipeline steps ────────────────────────────────────────────────────

@app.post("/step/convert")
async def run_convert(force: bool = Form(False)):
    rf = _state.ensure_run_folder()
    ns = build_convert_ns(
        input=_state.input_folder,
        output=str(rf),
        converter=_state.converter,
        force=force,
        meta=_state.meta(),
    )
    token = _state.start_runner(_convert_module.main, ns, "convert", env=get_model_env(_state.llm_model))
    return JSONResponse({"token": token, "step": "convert", "run_id": _state.run_id})


@app.post("/step/extract")
async def run_extract(
    force: bool = Form(False),
    input_file: str = Form(""),
):
    rf = _state.ensure_run_folder()
    # Retry: caller passes *_raw.json; derive the sibling .md file
    if input_file and input_file.endswith("_raw.json"):
        raw_p = Path(input_file)
        md_p  = raw_p.parent / (raw_p.name.removesuffix("_raw.json") + ".md")
        input_path = str(md_p) if md_p.exists() else str(rf)
    elif input_file:
        input_path = input_file
    else:
        # Default: run folder — Convert outputs MD files there
        input_path = str(rf)
    ns = build_extract_ns(
        schema=_state.schema(),
        input=input_path,
        output_dir=str(rf),
        filter=_state.filter_level,
        force=force,
        meta=_state.meta(),
    )
    token = _state.start_runner(_extract_module.main, ns, "extract", env={**get_model_env(_state.llm_model), "KG_CACHE_DIR": str(_state._proj() / ".cache"), "SEMANTIC_CHECK_MODEL": _state.semantic_check_model})
    return JSONResponse({"token": token, "step": "extract", "run_id": _state.run_id, "llm_model": _llm_deployment()})


@app.post("/step/write")
async def run_write(dry_run: bool = Form(False)):
    rf = _state.ensure_run_folder()
    db_path = _state.db_path()
    raw_files = sorted(rf.glob("*_raw.json"))
    if not raw_files:
        return JSONResponse({"error": "No raw files in run folder"}, status_code=400)
    # apply all raw files in run folder
    ns = build_apply_ns(
        apply=str(rf),
        schema=_state.schema(),
        db=db_path,
        backend=_state.backend,
        filter=_state.filter_level,
        run_id=_state.run_id,
        dry_run=dry_run,
    )
    token = _state.start_runner(_apply_module.main, ns, "apply", env=get_model_env(_state.llm_model))
    return JSONResponse({"token": token, "step": "apply", "run_id": _state.run_id})


# ── agent retry ──────────────────────────────────────────────────────────────

def _run_agent_semantic_check(raw_path: str, schema_path: str, filter_level: str, rel_type_filter: str) -> None:
    """Semantic-check the agent-retry triples in-place, writing ai_opinion + recolored results back."""
    import yaml
    from agents.semantic_check_agent import check_triples as _check_triples

    raw_p = Path(raw_path)
    data = json.loads(raw_p.read_text(encoding="utf-8"))
    all_triples = data.get("triples", [])
    # Only check triples not yet reviewed by semantic check — avoids re-coloring
    # batch triples that were already checked and whose color may have been manually overridden.
    targets = [t for t in all_triples if t.get("rel_type") == rel_type_filter and not t.get("_ai_reviewed")] if rel_type_filter else [t for t in all_triples if not t.get("_ai_reviewed")]
    if not targets:
        return

    doc_text = data.get("doc_text", "")
    if not doc_text and data.get("doc_source"):
        try:
            doc_text = Path(data["doc_source"]).read_text(encoding="utf-8")
        except Exception:
            pass

    schema_data = yaml.safe_load(Path(schema_path).read_text(encoding="utf-8"))
    schema_rels = schema_data.get("relationships", [])
    schema_nodes = schema_data.get("nodes", {})

    # Extraction instructions (meta.yaml) let the grader judge document-subject intent.
    _instructions = ""
    try:
        import pipeline_meta as _pm
        _meta_p = Path(schema_path).parent / "meta.yaml"
        if _meta_p.exists():
            _instructions = _pm.get_instructions(_pm.load_meta(str(_meta_p)))
    except Exception:
        pass

    label = f"[{rel_type_filter}]" if rel_type_filter else "all"
    print(f"\n  [semantic_check] {len(targets)} triple(s) for {label}…", flush=True)
    harvest_dir = Path(schema_path).parent / "harvest"
    checked = _check_triples(targets, doc_text, schema_rels=schema_rels, schema_nodes=schema_nodes, filter_level=filter_level, harvest_dir=harvest_dir if harvest_dir.exists() else None, doc_name=raw_p.name, instructions=_instructions)

    id_to_checked = {t["_id"]: t for t in checked if "_id" in t}
    data["triples"] = [id_to_checked.get(t.get("_id"), t) for t in all_triples]
    raw_p.write_text(json.dumps(data, indent=2, default=str, ensure_ascii=False))

    by_color = {c: sum(1 for t in checked if t.get("triple_color") == c) for c in ("green", "yellow", "red")}
    print(f"  [semantic_check] done — {by_color}", flush=True)


def _agent_retry_main(args) -> None:
    result_path = _agent_module.run_extraction(
        doc_path=args.doc_path,
        schema_path=args.schema_path,
        meta_path=getattr(args, "meta_path", ""),
        filter_level=args.filter_level,
        rel_type_filter=args.rel_type_filter,
        merge_raw_path=args.merge_raw_path,
        output_dir=args.output_dir,
    )
    if result_path and Path(result_path).exists():
        _run_agent_semantic_check(result_path, args.schema_path, args.filter_level, args.rel_type_filter)

    # Update harvest store — output_dir is the run folder; project_dir is two levels up
    try:
        from agents.harvest import harvest_project as _harvest_project
        _project_dir = Path(args.output_dir).parent.parent
        counts = _harvest_project(str(_project_dir))
        print(f"  [harvest] store updated: {counts}", flush=True)
    except Exception as _he:
        print(f"  [harvest] warning: {_he}", flush=True)


@app.post("/step/retry-agent")
async def run_retry_agent(
    raw_path: str = Form(...),
    rel_type: str = Form(...),
):
    rf = _state.run_folder()
    if rf is None:
        return JSONResponse({"error": "No run selected"}, status_code=400)
    raw_p = Path(raw_path).resolve()
    if not raw_p.is_relative_to(rf.resolve()):
        return JSONResponse({"error": "Invalid path"}, status_code=400)
    if not raw_p.exists():
        return JSONResponse({"error": f"Raw file not found: {raw_path}"}, status_code=400)
    md_p = raw_p.parent / (raw_p.name.removesuffix("_raw.json") + ".md")
    if not md_p.exists():
        return JSONResponse({"error": f"Markdown source not found: {md_p.name}"}, status_code=400)
    from argparse import Namespace as _NS
    ns = _NS(
        doc_path=str(md_p),
        schema_path=_state.schema(),
        meta_path=_state.meta() or "",
        filter_level=_state.filter_level,
        rel_type_filter=rel_type,
        merge_raw_path=str(raw_p),
        output_dir=str(rf),
    )
    token = _state.start_runner(_agent_retry_main, ns, "agent_retry", env={**get_model_env(_state.llm_model), "SEMANTIC_CHECK_MODEL": _state.semantic_check_model})
    return JSONResponse({"token": token, "step": "agent_retry"})


@app.get("/schema/rel-types")
def get_schema_rel_types():
    try:
        _, rels = _extract_module.load_schema(_state.schema())
        return JSONResponse({"rel_types": [r["rel_type"] for r in rels]})
    except Exception as e:
        return JSONResponse({"rel_types": [], "error": str(e)})


# ── routes: SSE log stream ────────────────────────────────────────────────────

@app.get("/stream/{token}")
async def stream_logs(token: str):
    entry = _state.get_runner(token)
    if entry is None:
        return HTMLResponse("Unknown token", status_code=404)

    async def event_gen():
        q = entry["queue"]
        loop = asyncio.get_running_loop()
        while True:
            try:
                line = await loop.run_in_executor(None, lambda: q.get(timeout=0.05))
                if line is None:
                    # runner finished; determine success
                    runner: _pr.PipelineRunner = entry["runner"]
                    ok = runner.returncode == 0
                    entry["done"] = True
                    entry["ok"] = ok
                    status_cls = "text-green-400" if ok else "text-red-400"
                    status_msg = "✓ Done" if ok else "✗ Failed"
                    yield (
                        f"data: <span class='{status_cls} font-bold'>{status_msg}</span>\n"
                        f"\n"
                    )
                    yield "event: done\ndata: \n\n"
                    # Rebuild harvest store after successful Step 3 write
                    if ok and entry.get("step") == "apply":
                        try:
                            from agents.harvest import harvest_project as _harvest_project
                            import threading as _threading
                            _project_dir = Path(_state.schema()).parent
                            def _harvest_bg(_pd: str = str(_project_dir)) -> None:
                                try:
                                    _counts = _harvest_project(_pd)
                                    print(f"[harvest] store updated after write: {_counts}", flush=True)
                                except Exception as _e:
                                    print(f"[harvest] post-write update error: {_e}", flush=True)
                            _threading.Thread(target=_harvest_bg, daemon=True).start()
                        except Exception as _he:
                            print(f"[harvest] post-write start error: {_he}", flush=True)
                    # Replace with lightweight tombstone to free runner/queue refs
                    _state._runners[token] = {
                        "done": True, "ok": ok, "step": entry["step"],
                        "started_at": entry.get("started_at", 0),
                    }
                    break
                safe = escape(line.rstrip("\n"))
                yield f"data: <span>{safe}</span><br>\n\n"
            except queue.Empty:
                # Empty queue — send keepalive comment
                yield ": keepalive\n\n"
                await asyncio.sleep(0.5)

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/runner-status/{token}")
async def runner_status(token: str):
    entry = _state.get_runner(token)
    if entry is None:
        return JSONResponse({"done": True, "ok": False})
    if entry.get("done"):
        return JSONResponse({"done": True, "ok": entry.get("ok", False), "step": entry.get("step", "")})
    runner: _pr.PipelineRunner = entry["runner"]
    done = runner.poll() is not None
    ok = runner.returncode == 0 if done else None
    return JSONResponse({"done": done, "ok": ok, "step": entry["step"]})


# ── routes: md preview ───────────────────────────────────────────────────────

@app.get("/md/list")
async def md_list():
    rf = _state.run_folder()
    files = _scan_md(str(rf)) if rf else []
    return JSONResponse({"files": [Path(f).name for f in files], "count": len(files)})


@app.get("/md/content")
async def md_content(filename: str):
    rf = _state.run_folder()
    if not rf:
        return JSONResponse({"error": "No run selected"}, status_code=400)
    p = rf / Path(filename).name
    if not p.exists() or p.suffix != ".md":
        return JSONResponse({"error": "File not found"}, status_code=404)
    content = p.read_text(errors="replace")
    truncated = len(content) > 8_000
    return JSONResponse({
        "content": content[:8_000],
        "truncated": truncated,
        "filename": p.name,
    })


# ── routes: quality summary ──────────────────────────────────────────────────

@app.get("/review/quality")
async def review_quality():
    files = _state.list_review_files()
    result = []
    for raw_path in files:
        try:
            raw_data = _rl.load_raw(raw_path)
            raw_triples = raw_data.get("triples", [])
            rev_path = _rl.review_path_for(raw_path)
            events   = _rl.load_events(rev_path)
            # materialize applies OVERRIDE (including color changes) and excludes REJECTs
            active_triples = _rl.materialize(raw_data, events)
            rejected = sum(1 for t in raw_triples if events.get(t.get("_id", ""), {}).get("action") == "REJECT")
            added    = sum(1 for ev in events.values() if ev.get("action") == "ADD")
            result.append({
                "name":     _triple_display_name(raw_path),
                "raw_path": str(raw_path),
                "rev_path": str(rev_path),
                "total":    len(raw_triples),
                "green":    sum(1 for t in active_triples if t.get("triple_color", "green") == "green"),
                "yellow":   sum(1 for t in active_triples if t.get("triple_color", "green") == "yellow"),
                "red":      sum(1 for t in active_triples if t.get("triple_color", "green") == "red"),
                "rejected": rejected,
                "added":    added,
                "active":   len(raw_triples) - rejected + added,
            })
        except Exception as _qe:
            print(f"[review_quality] skipped {raw_path.name}: {_qe}", flush=True)
    totals = {
        "total":    sum(f["total"]    for f in result),
        "green":    sum(f["green"]    for f in result),
        "yellow":   sum(f["yellow"]   for f in result),
        "red":      sum(f["red"]      for f in result),
        "rejected": sum(f["rejected"] for f in result),
        "added":    sum(f["added"]    for f in result),
        "active":   sum(f["active"]   for f in result),
    }
    return JSONResponse({"files": result, "total": totals})


# ── routes: review editor ─────────────────────────────────────────────────────

@app.get("/review/load", response_class=HTMLResponse)
async def load_review(request: Request, path: str = Query(...)):
    from extract import load_schema
    raw_path = Path(path).resolve()
    _proj_root = (_ROOT / "projects").resolve()
    if not raw_path.is_relative_to(_proj_root):
        return HTMLResponse("<p class='text-red-500'>Access denied</p>", status_code=403)
    if not raw_path.name.endswith("_raw.json"):
        return HTMLResponse("<p class='text-red-500'>Invalid file</p>", status_code=400)
    if not raw_path.exists():
        return HTMLResponse("<p class='text-red-500'>File not found</p>")
    import copy as _copy
    raw_data = _rl.load_raw(raw_path)
    rev_path = _rl.review_path_for(raw_path)
    events   = _rl.load_events(rev_path)
    # Review view: show ALL raw triples (rejected ones shown strikethrough).
    # Apply OVERRIDE fields (color/props) so saved edits appear after reload.
    # REJECT triples stay in the list — template uses events dict for strikethrough.
    triples = []
    for t in raw_data.get("triples", []):
        ev = events.get(t.get("_id", ""), {})
        if ev.get("action") == "OVERRIDE":
            t = _copy.deepcopy(t)
            for field in ("from_props", "to_props", "rel_props", "triple_color",
                          "supporting_quote", "evidence"):
                if field in ev:
                    t[field] = ev[field]
            # Sync entity dot colors to match the human-overridden triple_color
            if "triple_color" in ev:
                t["from_color"] = ev["triple_color"]
                t["to_color"]   = ev["triple_color"]
        triples.append(t)
    for tid, ev in events.items():
        if ev.get("action") == "ADD" and "triple" in ev:
            t = _copy.deepcopy(ev["triple"])
            t.setdefault("_id", tid)
            triples.append(t)
    try:
        nodes, rels = load_schema(_state.schema())
        node_labels = sorted(nodes.keys())
        rel_types   = sorted({r["rel_type"] for r in rels})
        rel_type_map = {}
        for _r in rels:
            _rt = _r.get("rel_type", "")
            if not _rt:
                continue
            _fn_def = nodes.get(_r.get("from_node", ""), {})
            _tn_def = nodes.get(_r.get("to_node", ""), {})
            rel_type_map[_rt] = {
                "from_label":    _r.get("from_node", ""),
                "from_pk":       _node_pk(_fn_def),
                "from_keys":     [p["name"] for p in _fn_def.get("properties", [])
                                  if p.get("source") not in ("pipeline",)],
                "to_label":      _r.get("to_node", ""),
                "to_pk":         _node_pk(_tn_def),
                "to_keys":       [p["name"] for p in _tn_def.get("properties", [])
                                  if p.get("source") not in ("pipeline",)],
                "rel_prop_keys": [p["name"] for p in _r.get("properties", [])
                                  if p.get("source") == "llm"],
            }
        node_pk_map = {
            label: _node_pk(node_def)
            for label, node_def in nodes.items()
        }
    except Exception:
        node_labels, rel_types, rel_type_map, node_pk_map = [], [], {}, {}
    return templates.TemplateResponse(request, "partials/triples_list.html", {
        "triples": triples,
        "events": events,
        "raw_path": str(raw_path),
        "rev_path": str(rev_path),
        "raw_hash": _rl.file_hash(raw_path),
        "node_labels": node_labels,
        "rel_types": rel_types,
        "rel_type_map": rel_type_map,
        "node_pk_map": node_pk_map,
    })


def _highlight_evidence(md: str, evidence: list[dict], from_term: str = "", to_term: str = "") -> str:
    """Wrap quote spans (yellow), subject occurrences (green), object occurrences (pink).

    Quote (evidence) spans are the base highlight: stored offsets are trusted only
    when md[start:end] still equals the span text (the .md can drift), otherwise the
    span is re-located via grounding. `from_term` (subject) and `to_term` (object) —
    the raw document wording — are highlighted at EVERY case-insensitive occurrence,
    including inside the quote, so a reviewer sees where each entity is mentioned.
    Precedence: subject (green) > object (pink) > quote (yellow), so a term inside the
    quote renders in its own color within the surrounding yellow. ev-first anchors the
    quote start (else the first term) for scroll. Pure + testable.
    """
    from grounding import locate as _locate
    qspans: list[tuple[int, int]] = []
    for ev in evidence or []:
        s, e, txt = ev.get("start"), ev.get("end"), ev.get("text", "")
        if isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(md) and md[s:e] == txt:
            qspans.append((s, e))
        elif txt:
            loc = _locate(txt, md)
            if loc:
                qspans.append(loc)
    qspans.sort()

    md_low = md.lower()

    def _occurrences(term: str) -> list[tuple[int, int]]:
        tl = (term or "").strip().lower()
        if len(tl) < 2:
            return []
        spans, start = [], 0
        while (i := md_low.find(tl, start)) >= 0:
            spans.append((i, i + len(tl)))
            start = i + len(tl)
        return spans

    sspans = _occurrences(from_term)  # subject -> green
    ospans = _occurrences(to_term)    # object  -> pink

    # Boundary sweep: cut at every span edge, color each elementary interval by
    # precedence subject > object > quote, so an entity term inside the quote shows
    # in its own color within the surrounding yellow rather than being hidden.
    cuts = {0, len(md)}
    for s, e in (*qspans, *sspans, *ospans):
        cuts.add(s)
        cuts.add(e)
    cuts = sorted(c for c in cuts if 0 <= c <= len(md))
    anchor_start = qspans[0][0] if qspans else min(
        (t[0] for t in (*sspans, *ospans)), default=None)

    out = []
    for a, b in zip(cuts, cuts[1:]):
        seg = escape(md[a:b])
        if any(s <= a < e for s, e in sspans):
            cls = "bg-green-300"
        elif any(s <= a < e for s, e in ospans):
            cls = "bg-pink-300"
        elif any(s <= a < e for s, e in qspans):
            cls = "bg-yellow-300"
        else:
            out.append(seg)
            continue
        anchor = ' id="ev-first"' if a == anchor_start else ""
        out.append(f'<mark{anchor} class="{cls} rounded px-0.5">{seg}</mark>')

    legend = (
        "<div class='text-xs text-gray-500 mb-2 flex gap-3'>"
        "<span><mark class='bg-yellow-300 rounded px-1'>quote</mark></span>"
        "<span><mark class='bg-green-300 rounded px-1'>subject</mark></span>"
        "<span><mark class='bg-pink-300 rounded px-1'>object</mark></span></div>"
    )
    banner = "" if qspans else (
        "<div class='bg-amber-50 border border-amber-200 text-amber-700 text-xs px-3 py-2 mb-2 rounded'>"
        "No evidence span located in this document. The supporting quote may be unlocatable or fabricated.</div>"
    )
    return (legend + banner +
            '<pre class="whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-gray-800">'
            + "".join(out) + "</pre>")


@app.get("/review/evidence", response_class=HTMLResponse)
async def review_evidence(raw_path: str = Query(...), triple_id: str = Query(...)):
    """Return the source Markdown with this triple's evidence spans <mark>ed."""
    rp = Path(raw_path).resolve()
    _proj_root = (_ROOT / "projects").resolve()
    if not rp.is_relative_to(_proj_root) or not rp.name.endswith("_raw.json"):
        return HTMLResponse("<p class='text-red-500 p-4'>Access denied</p>", status_code=403)
    md_p = rp.parent / (rp.name.removesuffix("_raw.json") + ".md")
    if not md_p.exists():
        return HTMLResponse("<p class='text-amber-600 p-4'>Source .md not found for this run.</p>")
    raw_data = _rl.load_raw(rp)
    triple = next((t for t in raw_data.get("triples", []) if t.get("_id") == triple_id), None)
    if not triple:
        return HTMLResponse("<p class='text-amber-600 p-4'>Triple not found.</p>")
    return HTMLResponse(_highlight_evidence(
        md_p.read_text(errors="replace"), triple.get("evidence") or [],
        from_term=triple.get("from_term") or "", to_term=triple.get("to_term") or ""))


@app.post("/review/save")
async def save_review(request: Request):
    body = await request.json()
    raw_path = Path(body["raw_path"]).resolve()
    rev_path = Path(body["rev_path"]).resolve()
    _proj_root = (_ROOT / "projects").resolve()
    if not raw_path.is_relative_to(_proj_root):
        return JSONResponse({"error": "invalid raw path"}, status_code=400)
    if not rev_path.is_relative_to(raw_path.parent):
        return JSONResponse({"error": "invalid review path"}, status_code=400)
    if not raw_path.exists():
        return JSONResponse({"error": "raw file not found"}, status_code=400)

    # Stale-review guard: reject save if raw was re-extracted since the client loaded it
    incoming_hash = body.get("raw_hash", "")
    if incoming_hash and _rl.file_hash(raw_path) != incoming_hash:
        return JSONResponse(
            {"error": "Raw file was re-extracted since you loaded this review. Reload to get the latest version."},
            status_code=409,
        )

    # For quote overrides: load the source .md + the original quotes so an edited
    # quote can be re-anchored (only a quote verbatim-in-document is stored).
    _md_p = raw_path.parent / (raw_path.name.removesuffix("_raw.json") + ".md")
    _doc_text = _md_p.read_text(errors="replace") if _md_p.exists() else ""
    _orig_quotes = {
        t.get("_id"): (t.get("supporting_quote") or "")
        for t in _rl.load_raw(raw_path).get("triples", [])
    }
    quote_warnings: list[dict] = []

    # Reconstruct events from submitted state
    events: dict[str, dict] = {}
    for item in (body.get("triples") or []):
        tid = item["_id"]
        action = item.get("action", "ACCEPT")
        if action == "REJECT":
            events[tid] = {"action": "REJECT"}
        elif action == "OVERRIDE":
            ev: dict = {
                "action": "OVERRIDE",
                "from_props": item.get("from_props") or {},
                "to_props":   item.get("to_props")   or {},
                "rel_props":  item.get("rel_props")  or {},
            }
            if item.get("triple_color"):
                ev["triple_color"] = item["triple_color"]
            # Quote override — only when the human actually changed the text. The
            # new quote is re-anchored via _build_evidence (grounding.locate): it is
            # stored ONLY if it is found verbatim in the .md, preserving the
            # "every quote is provably in the document" invariant. evidence is the
            # source of truth; supporting_quote is derived (/-joined span text).
            _new_q = (item.get("supporting_quote") or "").strip()
            if _new_q and _new_q != (_orig_quotes.get(tid, "") or "").strip():
                _spans = _extract_module._build_evidence([_new_q], _doc_text) if _doc_text else []
                if _spans and _spans[0].get("start") is not None:
                    ev["evidence"] = _spans
                    ev["supporting_quote"] = " / ".join(s["text"] for s in _spans)
                else:
                    quote_warnings.append({"id": tid, "quote": _new_q[:120]})
            events[tid] = ev
        # ACCEPT → no event needed

    for item in (body.get("added_triples") or []):
        # Skip entries missing required fields — harvest would silently drop them anyway
        if not item.get("rel_type") or not item.get("from_props") or not item.get("to_props"):
            continue
        tid = item.get("_id") or "add_" + uuid.uuid4().hex[:8]
        events[tid] = {
            "action": "ADD",
            "triple": {
                "_id":        tid,
                "from_label": item.get("from_label", ""),
                "from_pk":    item.get("from_pk", ""),
                "from_props": item.get("from_props") or {},
                "to_label":   item.get("to_label", ""),
                "to_pk":      item.get("to_pk", ""),
                "to_props":   item.get("to_props") or {},
                "rel_type":   item.get("rel_type", ""),
                "rel_props":  {**(item.get("rel_props") or {}), "manually_added": True},
                "triple_color": "green",
                "from_color":   "green",
                "to_color":     "green",
            },
        }

    _rl.save_events(rev_path, raw_path, events)

    # Update harvest store in background — project_dir is three levels up from raw file
    # (projects/<proj>/runs/<run_id>/<file>)
    _project_dir = raw_path.parent.parent.parent
    try:
        import threading as _threading
        from agents.harvest import harvest_project as _harvest_project
        def _harvest_bg(_pd: str = str(_project_dir)) -> None:
            try:
                _harvest_project(_pd)
            except Exception as _e:
                print(f"[harvest] background update error: {_e}", flush=True)
        _threading.Thread(target=_harvest_bg, daemon=True).start()
    except Exception as _he:
        print(f"[harvest] background update failed to start: {_he}", flush=True)

    return JSONResponse({"saved": len(events), "quote_warnings": quote_warnings})


# ── routes: graph summary ─────────────────────────────────────────────────────

def _node_display_field(node_def: dict) -> str:
    """Return the best human-readable property name for graph summary display queries."""
    props = node_def.get("properties", [])
    if any(p.get("name") == "name" for p in props):
        return "name"
    pk = next((p["name"] for p in props if p.get("primary_key")), None)
    return pk or (props[0]["name"] if props else "name")


def _node_pk(node_def: dict) -> str:
    """Return the primary key field name for graph write operations."""
    props = node_def.get("properties", [])
    pk = next((p["name"] for p in props if p.get("primary_key")), None)
    return pk or (props[0]["name"] if props else "name")


@app.get("/graph/summary", response_class=HTMLResponse)
async def graph_summary(request: Request):
    from extract import load_schema
    from backends import get_backend

    db_path = _state.db_path()
    if _state.backend == "ladybug" and not Path(db_path).exists():
        return HTMLResponse("<p class='text-gray-400 italic'>No graph database found for this run.</p>")
    backend = None
    try:
        nodes, rels = load_schema(_state.schema())
        backend = get_backend(_state.backend, db_path if _state.backend == "ladybug" else None, nodes, rels, read_only=True, setup=False)
        node_counts = {lbl: backend.count_nodes(lbl) for lbl in nodes}
        rel_counts  = {r["rel_type"]: backend.count_edges(r["rel_type"]) for r in rels}

        # Sample triples: up to 20 rows per relationship type
        try:
            from backends.base import _safe_ident as _si
        except ImportError:
            def _si(s: str) -> str: return s  # no validation if ladybug absent
        sample_triples: list[dict] = []
        for rel in rels:
            rt  = rel["rel_type"]
            fn  = rel.get("from_node", "")
            tn  = rel.get("to_node", "")
            fd  = _node_display_field(nodes.get(fn, {}))
            td  = _node_display_field(nodes.get(tn, {}))
            try:
                rows = backend.run_cypher(
                    f"MATCH (a:{_si(fn)})-[:{_si(rt)}]->(b:{_si(tn)}) "
                    f"RETURN a.`{fd}` AS from_name, b.`{td}` AS to_name LIMIT 20"
                )
                if rows:
                    sample_triples.append({
                        "label": f"{fn} —[{rt}]→ {tn}",
                        "rows": rows,
                    })
            except Exception:
                pass
    except Exception as exc:
        return HTMLResponse(f"<p class='text-red-500'>Error: {escape(str(exc))}</p>")
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass

    return templates.TemplateResponse(request, "partials/graph_summary.html", {
        "node_counts": node_counts,
        "rel_counts": rel_counts,
        "sample_triples": sample_triples,
    })


# ── routes: schema editor ────────────────────────────────────────────────────

@app.get("/schema/load")
async def schema_load():
    import yaml
    schema_path = _state.schema()
    if not Path(schema_path).exists():
        return JSONResponse({"error": "schema not found"}, status_code=404)
    raw = yaml.safe_load(Path(schema_path).read_text(encoding="utf-8")) or {}
    return JSONResponse({"nodes": raw.get("nodes", {}), "relationships": raw.get("relationships", [])})


@app.post("/schema/save")
async def schema_save(request: Request):
    import yaml
    body = await request.json()
    schema_path = Path(_state.schema())
    if not schema_path.exists():
        return JSONResponse({"error": "schema not found"}, status_code=404)
    raw = yaml.safe_load(schema_path.read_text(encoding="utf-8")) or {}
    # Merge only editable fields; structural fields (node keys, rel_type,
    # from_node, to_node, from_field, to_field, properties) are left unchanged.
    for node_key, edits in (body.get("nodes") or {}).items():
        if node_key not in raw.get("nodes", {}):
            continue
        node = raw["nodes"][node_key]
        for field in ("description", "sem_group"):
            if field in edits:
                node[field] = edits[field] or None
                if not node[field]:
                    node.pop(field, None)
        for list_field in ("umls_vocabs", "semantic_types"):
            if list_field in edits:
                vals = [v.strip() for v in edits[list_field] if str(v).strip()]
                if vals:
                    node[list_field] = vals
                else:
                    node.pop(list_field, None)
    for i, rel_edit in enumerate(body.get("relationships") or []):
        rel_type = rel_edit.get("rel_type")
        match = next((r for r in raw.get("relationships", []) if r.get("rel_type") == rel_type), None)
        if match is None:
            continue
        for field in ("from_hint", "to_hint", "extract_prompt"):
            if field in rel_edit:
                match[field] = rel_edit[field] or None
                if not match[field]:
                    match.pop(field, None)
        if "examples" in rel_edit:
            match["examples"] = rel_edit["examples"] or []
            if not match["examples"]:
                match.pop("examples", None)
        # editable llm-source rel props: only the hint field
        if "prop_hints" in rel_edit:
            for prop in match.get("properties", []):
                if prop.get("source") == "llm" and prop["name"] in rel_edit["prop_hints"]:
                    hint = rel_edit["prop_hints"][prop["name"]]
                    if hint:
                        prop["hint"] = hint
                    else:
                        prop.pop("hint", None)
    try:
        from extract import _SchemaDef
        _SchemaDef.model_validate(raw)
    except Exception as exc:
        return JSONResponse({"error": f"Schema validation failed: {exc}"}, status_code=400)
    schema_path.write_text(
        yaml.dump(raw, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return JSONResponse({"ok": True})


# ── routes: metadata ─────────────────────────────────────────────────────────

@app.get("/meta/load")
async def meta_load():
    import pipeline_meta as _pm
    meta_path = _state.meta()
    if meta_path is None:
        return JSONResponse({"instructions": "", "pages": {}})
    try:
        meta = _pm.load_meta(meta_path)
    except SystemExit as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    pages_out = {}
    for stem, entry in (meta.get("pages") or {}).items():
        entry = entry or {}
        pages_out[stem] = {
            "include": ", ".join(str(x) for x in (entry.get("include") or [])),
            "exclude": ", ".join(str(x) for x in (entry.get("exclude") or [])),
        }
    return JSONResponse({"instructions": meta.get("instructions", ""), "pages": pages_out})


@app.post("/meta/save")
async def meta_save(request: Request):
    import pipeline_meta as _pm
    body = await request.json()
    instructions = (body.get("instructions") or "").strip()
    pages_raw: dict = body.get("pages") or {}
    out_meta: dict = {}
    if instructions:
        out_meta["instructions"] = instructions
    pages_out: dict = {}
    for stem, entry in pages_raw.items():
        stem = stem.strip()
        if not stem:
            continue
        row: dict = {}
        inc = [x.strip() for x in str(entry.get("include", "")).split(",") if x.strip()]
        exc = [x.strip() for x in str(entry.get("exclude", "")).split(",") if x.strip()]
        if inc:
            row["include"] = inc
        if exc:
            row["exclude"] = exc
        pages_out[stem] = row
    if pages_out:
        out_meta["pages"] = pages_out
    meta_path = _state._proj() / "meta.yaml"
    _pm.save_meta(out_meta, meta_path)
    return JSONResponse({"ok": True})


@app.get("/review/unresolved")
async def review_unresolved():
    rf = _state.run_folder()
    if rf is None:
        return JSONResponse({"entries": []})
    err_log = rf / "_node_agent_errors.jsonl"
    if not err_log.exists():
        return JSONResponse({"entries": []})
    entries = []
    total = 0
    with err_log.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                if len(entries) < 50:
                    entries.append(json.loads(line))
            except Exception:
                pass
    return JSONResponse({"entries": entries, "total": total})


# ── routes: settings ──────────────────────────────────────────────────────────

@app.post("/settings")
async def save_settings(
    filter_level: str = Form("moderate"),
    converter: str = Form("pymupdf4llm"),
    backend: str = Form("ladybug"),
    llm_model: str = Form(""),
    semantic_check_model: str = Form(""),
):
    _state.filter_level = filter_level
    _state.converter    = converter
    _state.backend      = backend
    if llm_model:
        with _pr._env_lock:
            import os
            os.environ.update(get_model_env(llm_model))
        _state.llm_model = llm_model
    # "" is a valid selection: the grader reuses the extraction model. The grader
    # model's creds are applied per-run via SEMANTIC_CHECK_MODEL (see extract /
    # agent-retry runner env), not globally here.
    _state.semantic_check_model = semantic_check_model
    _state.save()
    return JSONResponse({"ok": True})


# ── routes: project wizard ───────────────────────────────────────────────────

@app.get("/project/templates")
async def project_templates():
    """List schemas that can be used as starter templates for a new project."""
    items: list[dict] = []
    proj_dir = _ROOT / "projects"
    if proj_dir.is_dir():
        for d in sorted(proj_dir.iterdir()):
            if d.is_dir() and (d / "schema.yaml").exists():
                items.append({"name": d.name, "label": f"Copy from '{d.name}'"})
    items.append({"name": "blank", "label": "Blank (no schema)"})
    return JSONResponse({"templates": items})


@app.post("/project/create")
async def project_create(request: Request):
    import re
    import shutil

    body = await request.json()
    name     = (body.get("name")     or "").strip()
    template = (body.get("template") or "blank").strip()

    if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9_-]*$', name):
        return JSONResponse(
            {"error": "Invalid name. Use letters, digits, hyphens, or underscores. Must start with a letter or digit."},
            status_code=400,
        )

    proj_path = (_ROOT / "projects" / name).resolve()
    if not proj_path.is_relative_to((_ROOT / "projects").resolve()):
        return JSONResponse({"error": "Invalid project name."}, status_code=400)
    if proj_path.exists():
        return JSONResponse({"error": f"Project '{escape(name)}' already exists."}, status_code=400)

    (proj_path / "raw_documents").mkdir(parents=True)

    if template != "blank":
        src_schema = (_ROOT / "projects" / template / "schema.yaml").resolve()
        if src_schema.is_relative_to((_ROOT / "projects").resolve()) and src_schema.exists():
            shutil.copy2(src_schema, proj_path / "schema.yaml")

    (proj_path / "meta.yaml").write_text("# Extraction metadata\n", encoding="utf-8")

    _state.project_folder = str(proj_path)
    _state.input_folder   = str(proj_path / "raw_documents")
    _state.run_id         = ""
    _state.schema_path    = ""
    _state.save()

    return JSONResponse({"ok": True, "name": name})


# ── routes: run folder utilities ─────────────────────────────────────────────

@app.get("/run/step-status")
async def run_step_status():
    rf = _state.run_folder()
    if rf is None or not rf.is_dir():
        return JSONResponse({"convert": "idle", "extract": "idle", "write": "idle"})
    return JSONResponse({
        "convert": "done" if any(rf.glob("*.md"))       else "idle",
        "extract": "done" if any(rf.glob("*_raw.json")) else "idle",
        "write":   "done" if any(rf.glob("*.db"))       else "idle",
    })


@app.get("/run/open-folder")
async def open_run_folder():
    import platform
    import subprocess

    rf = _state.run_folder()
    target = rf if (rf is not None and rf.is_dir()) else _state._proj()

    system = platform.system()
    if system == "Darwin":
        subprocess.run(["open", str(target)], check=False)
    elif system == "Windows":
        subprocess.run(["explorer", str(target)], check=False)
    else:
        subprocess.run(["xdg-open", str(target)], check=False)

    return JSONResponse({"ok": True})


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("htmx_app.main:app", host="0.0.0.0", port=8000, reload=True,
                reload_dirs=[str(_ROOT)])
