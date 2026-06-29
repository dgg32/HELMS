#!/usr/bin/env python3
"""Shared Namespace builders for pipeline steps.

Both pipeline.py (CLI) and htmx_app/main.py (web UI) call these functions,
providing their own values from argparse / app state respectively.
Single source of truth for field names and defaults used by each step's main().
"""
from __future__ import annotations

from argparse import Namespace


def build_convert_ns(
    input: str,
    output: str,
    *,
    converter: str = "pymupdf4llm",
    force: bool = False,
    workers: int = 4,
    meta: str | None = None,
) -> Namespace:
    """Return the Namespace that convert_pdf.main() expects."""
    return Namespace(
        input=input,
        output=output,
        converter=converter,
        force=force,
        workers=workers,
        meta=meta,
    )


def build_extract_ns(
    schema: str,
    input: str,
    output_dir: str,
    *,
    filter: str = "moderate",
    force: bool = False,
    verbose: bool = False,
    skip_report: bool = False,
    concurrency: int = 2,
    retries: int = 2,
    meta: str | None = None,
) -> Namespace:
    """Return the Namespace that extract.main() expects."""
    return Namespace(
        schema=schema,
        input=input,
        output_dir=output_dir,
        filter=filter,
        force=force,
        verbose=verbose,
        skip_report=skip_report,
        concurrency=concurrency,
        retries=retries,
        meta=meta,
    )


def build_apply_ns(
    apply: str,
    schema: str,
    db: str,
    *,
    backend: str = "ladybug",
    filter: str = "moderate",
    run_id: str = "",
    dry_run: bool = False,
) -> Namespace:
    """Return the Namespace that apply_graph.main() expects."""
    return Namespace(
        apply=apply,
        schema=schema,
        db=db,
        backend=backend,
        filter=filter,
        run_id=run_id,
        dry_run=dry_run,
    )
