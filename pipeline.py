#!/usr/bin/env python3
"""
One-command pipeline: convert PDFs to Markdown, extract triples, apply to graph.

Step 1 — convert_pdf.py : PDFs → .md files (choice of converter)
Step 2 — extract.py     : .md files → <stem>_raw.json (LLM + UMLS/GLEIF)
Step 3 — apply_graph.py : raw JSONs → graph backend

Usage:
  python pipeline.py \\
    --schema finance_schema.yaml \\
    --input  finance_pdf/ \\
    --db     finance_kg.db \\
    --converter pymupdf4llm

"""
from __future__ import annotations

import argparse
import asyncio
import os
from argparse import Namespace
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import apply_graph
import convert_pdf
import extract
from convert_pdf import CONVERTERS
from pipeline_ns import build_apply_ns, build_convert_ns, build_extract_ns
from pipeline_orchestrator import PipelineOrchestrator


def parse_args() -> Namespace:
    p = argparse.ArgumentParser(
        description="Convert PDFs to Markdown then build a knowledge graph (one command).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Conversion flags ──────────────────────────────────────────────────────
    conv = p.add_argument_group("conversion")
    conv.add_argument(
        "--input", required=True, metavar="PATH",
        help="PDF file or directory of PDFs",
    )
    conv.add_argument(
        "--converter", default="pymupdf4llm", choices=list(CONVERTERS),
        help="PDF-to-Markdown converter",
    )
    conv.add_argument(
        "--force-convert", action="store_true",
        help="Re-convert even if the .md sidecar is already up to date",
    )

    # ── Extraction flags (passed to extract.py) ───────────────────────────────
    ext = p.add_argument_group("extraction")
    ext.add_argument(
        "--schema", required=True, metavar="PATH",
        help="Schema YAML file",
    )
    ext.add_argument(
        "--db", default=None, metavar="PATH",
        help="Database path or URI (default: finance_kg.db for ladybug; NEO4J_URI env var for neo4j)",
    )
    ext.add_argument(
        "--backend", default="ladybug", choices=("ladybug", "neo4j"),
        help="Graph backend",
    )
    ext.add_argument(
        "--skip-report", action="store_true",
        help="Print a summary of failed external lookups at the end",
    )
    ext.add_argument(
        "--force", action="store_true",
        help="Bypass extraction cache and re-run LLM even if a cached result exists",
    )
    ext.add_argument(
        "--meta", default=None, metavar="PATH",
        help="Pipeline metadata YAML (instructions + per-PDF page filters)",
    )
    ext.add_argument(
        "--filter", default="moderate",
        choices=["loose", "moderate", "strict"],
        help="Entity grounding filter level (default: moderate). "
             "loose=keep all | moderate=green+yellow | strict=green only.",
    )
    ext.add_argument(
        "--concurrency", default=2, type=int, metavar="N",
        help="Max documents processed in parallel (default: 2)",
    )
    ext.add_argument(
        "--verbose", action="store_true",
        help="Print cache hit messages from UMLS/GLEIF lookups.",
    )
    ext.add_argument(
        "--retries", type=int, default=2, metavar="N",
        help="Max retry attempts per document on transient failure (default: 2).",
    )

    return p.parse_args()


async def main():
    args = parse_args()
    input_path = Path(args.input)

    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise SystemExit(f"Input is not a PDF: {input_path}")
        output_dir = input_path.parent
    elif input_path.is_dir():
        output_dir = input_path
    else:
        raise SystemExit(f"Input path not found: {input_path.resolve()}")

    db_path = args.db
    if db_path is None:
        if args.backend == "ladybug":
            db_path = Path(args.schema).stem.replace("_schema", "_kg") + ".db"
        else:
            db_path = os.environ.get("NEO4J_URI")
            if not db_path:
                raise SystemExit("No Neo4j URI: pass --db or set NEO4J_URI in .env")

    if args.backend == "ladybug":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    _HEADERS = {
        "convert": f"Step 1: Convert  [converter={args.converter}]",
        "extract": "Step 2: Extract triples",
        "apply":   "Step 3: Apply to graph",
    }

    def _on_step_start(key: str) -> None:
        print(f"\n{'='*60}")
        print(_HEADERS[key])
        print(f"{'='*60}")

    orch = PipelineOrchestrator([
        ("convert", convert_pdf.main, lambda: build_convert_ns(
            input=str(input_path),
            output=str(output_dir),
            converter=args.converter,
            force=args.force_convert,
            meta=args.meta,
        )),
        ("extract", extract.main, lambda: build_extract_ns(
            schema=args.schema,
            input=str(output_dir),
            output_dir=str(output_dir),
            filter=args.filter,
            force=args.force,
            verbose=args.verbose,
            skip_report=args.skip_report,
            concurrency=args.concurrency,
            retries=args.retries,
            meta=args.meta,
        )),
        ("apply", apply_graph.main, lambda: build_apply_ns(
            apply=str(output_dir),
            schema=args.schema,
            db=db_path,
            backend=args.backend,
            filter=args.filter,
        )),
    ])

    await orch.run_all(
        on_step_start=_on_step_start,
        on_step_error=lambda k, e: print(f"\n[error] {k} failed: {e}", flush=True),
    )


if __name__ == "__main__":
    asyncio.run(main())
