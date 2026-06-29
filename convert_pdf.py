#!/usr/bin/env python3
"""
Convert PDF files to Markdown.

Converters:
  pymupdf4llm  Fast, CPU-only, no OCR (default)
  llamaparse   Cloud API (llama-cloud>=2.1); requires LLAMA_CLOUD_API_KEY in .env

Usage:
  python convert_pdf.py --input finance_pdf/ --converter pymupdf4llm
  python convert_pdf.py --input finance_pdf/ --converter llamaparse
"""
from __future__ import annotations

import argparse
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from pipeline_meta import PageFilter

load_dotenv(Path(__file__).parent / ".env")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_range_str(pages: list[int]) -> str:
    """Compress a sorted list of page numbers into range notation.

    e.g. [1,2,3,7,8,9,10,16] → "1-3,7-10,16"
    """
    if not pages:
        return ""
    pages = sorted(set(pages))
    parts: list[str] = []
    start = end = pages[0]
    for p in pages[1:]:
        if p == end + 1:
            end = p
        else:
            parts.append(f"{start}-{end}" if end > start else str(start))
            start = end = p
    parts.append(f"{start}-{end}" if end > start else str(start))
    return ",".join(parts)


def _unmatched_meta_keys(meta: dict, pdf_stems: "set[str] | list[str]") -> list[str]:
    """Return meta.yaml `pages:` keys that match none of the given PDF stems.

    Catches a typo'd or stale stem (e.g. "dificil label" vs "dificid label"):
    `pipeline_meta.get_page_filter` silently no-ops on a miss, so the filter
    looks applied in meta.yaml but the PDF is converted in full with no warning.
    """
    pages_cfg = meta.get("pages") or {}
    stems = set(pdf_stems)
    return [k for k in pages_cfg if k not in stems]


# ── Converters ────────────────────────────────────────────────────────────────

def convert_pymupdf4llm(pdf_path: Path, pages: list[int] | None = None) -> str:
    import fitz
    import pymupdf4llm
    with fitz.open(str(pdf_path)) as doc:
        kw = {"pages": [p - 1 for p in pages]} if pages else {}
        return pymupdf4llm.to_markdown(doc, header=False, footer=False, **kw)



_llama_cloud_client = None
_llama_cloud_lock = threading.Lock()


def _get_llama_cloud_client():
    global _llama_cloud_client
    with _llama_cloud_lock:
        if _llama_cloud_client is None:
            try:
                from llama_cloud import LlamaCloud
            except ImportError:
                raise SystemExit(
                    "llama-cloud not installed.\n"
                    "  pip install 'llama-cloud>=2.1'\n"
                    "Requires LLAMA_CLOUD_API_KEY in .env"
                )
            import os
            if not os.environ.get("LLAMA_CLOUD_API_KEY"):
                raise SystemExit(
                    "LLAMA_CLOUD_API_KEY not set.\n"
                    "Add LLAMA_CLOUD_API_KEY=<your_key> to your .env file."
                )
            _llama_cloud_client = LlamaCloud()
        return _llama_cloud_client


def convert_llamaparse(pdf_path: Path, pages: list[int] | None = None) -> str:
    client = _get_llama_cloud_client()
    file_obj = client.files.create(file=str(pdf_path), purpose="parse")
    try:
        parse_kwargs: dict = {
            "file_id": file_obj.id,
            "tier": "agentic",
            "version": "latest",
            "expand": ["markdown"],
            "processing_options": {
                "ignore": {"ignore_diagonal_text": True},
                "cost_optimizer": {"enable": True},
            },
        }
        if pages:
            parse_kwargs["page_ranges"] = {"target_pages": _to_range_str(pages)}
        result = client.parsing.parse(**parse_kwargs)
        if not result or not getattr(result, "markdown", None) or not getattr(result.markdown, "pages", None):
            raise RuntimeError("LlamaParse returned empty or unexpected response shape")
        return "\n\n".join(page.markdown for page in result.markdown.pages)
    finally:
        try:
            client.files.delete(file_obj.id)
        except Exception:
            pass


CONVERTERS: dict[str, Callable[..., str]] = {
    "pymupdf4llm": convert_pymupdf4llm,
    "llamaparse":  convert_llamaparse,
}


# ── Core helper (importable by pipeline.py) ───────────────────────────────────

def convert_file(
    pdf_path: Path,
    output_dir: Path,
    converter_name: str,
    force: bool = False,
    page_filter: "PageFilter | None" = None,
    meta_mtime: float | None = None,
) -> Path:
    """Convert one PDF to Markdown; return the output .md path.

    Skips conversion if the .md file already exists and is newer than both the
    PDF and the meta file (meta_mtime), unless force=True.

    page_filter: if provided, only the listed pages are converted (1-indexed).
    Pages are passed natively to each converter — no temporary PDF is created.
    """
    out_path = output_dir / (pdf_path.stem + ".md")
    md_mtime = out_path.stat().st_mtime if out_path.exists() else 0.0
    if (
        not force
        and out_path.exists()
        and md_mtime >= pdf_path.stat().st_mtime
        and (meta_mtime is None or md_mtime >= meta_mtime)
    ):
        print(f"  SKIP  {pdf_path.name}  (cached → {out_path.name})")
        return out_path

    pages_note = f"  pages={page_filter.pages[:5]}{'…' if len(page_filter.pages) > 5 else ''}" if page_filter else ""
    print(f"  {pdf_path.name}  →  {out_path.name}  [{converter_name}]{pages_note}")

    pages = page_filter.pages if page_filter else None
    text = CONVERTERS[converter_name](pdf_path, pages=pages)
    out_path.write_text(text, encoding="utf-8")
    print(f"    {len(text):,} chars written")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert PDF files to Markdown using a pluggable converter."
    )
    p.add_argument(
        "--input", required=True, metavar="PATH",
        help="PDF file or directory of PDFs",
    )
    p.add_argument(
        "--output", default=None, metavar="DIR",
        help="Output directory for .md files (default: same directory as input)",
    )
    p.add_argument(
        "--converter", default="pymupdf4llm", choices=list(CONVERTERS),
        help="Converter to use (default: pymupdf4llm)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-convert even if the .md file is already up to date",
    )
    p.add_argument(
        "--workers", type=int, default=4, metavar="N",
        help="Parallel worker threads (default: 4)",
    )
    p.add_argument(
        "--meta", default=None, metavar="PATH",
        help="Pipeline metadata YAML (page filters per PDF stem)",
    )
    return p.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()
    input_path = Path(args.input)

    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise SystemExit(f"Input is not a PDF: {input_path}")
        pdf_files = [input_path]
        default_output = input_path.parent
    elif input_path.is_dir():
        pdf_files = sorted(input_path.glob("*.pdf"))
        default_output = input_path
    else:
        raise SystemExit(f"Input path not found: {input_path.resolve()}")

    if not pdf_files:
        raise SystemExit(f"No PDF files found at '{input_path.resolve()}'")

    output_dir = Path(args.output) if args.output else default_output
    output_dir.mkdir(parents=True, exist_ok=True)

    import pipeline_meta as _pm
    meta = _pm.load_meta(args.meta)
    meta_mtime: float | None = None
    if args.meta:
        _mp = Path(args.meta)
        if _mp.exists():
            meta_mtime = _mp.stat().st_mtime

    # Only meaningful in directory mode: pdf_files is then every PDF in the project,
    # so a meta key matching none of them is genuinely unused. In single-file mode
    # pdf_files has just one entry — other (legitimate) meta keys would falsely
    # flag as "unmatched" simply because they're out of scope for this run.
    if input_path.is_dir():
        for _bad_key in _unmatched_meta_keys(meta, {p.stem for p in pdf_files}):
            print(f"  [warn] meta.yaml page filter key {_bad_key!r} matches no PDF "
                  f"in {input_path} — that PDF will be converted IN FULL (filter silently "
                  f"not applied). Check for a typo against the actual filename.", flush=True)

    def _page_filter_for(pdf: Path):
        import fitz
        with fitz.open(str(pdf)) as doc:
            total = len(doc)
        return _pm.get_page_filter(meta, pdf.stem, total)

    workers = max(1, min(args.workers, len(pdf_files)))
    print(f"Converting {len(pdf_files)} PDF(s)  [converter={args.converter}  workers={workers}]  →  {output_dir.resolve()}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures: dict = {}
        for pdf in pdf_files:
            try:
                page_filter = _page_filter_for(pdf)
            except Exception as exc:
                print(f"  [warn] skipping {pdf.name}: failed to read page count: {exc}", flush=True)
                continue
            futures[pool.submit(convert_file, pdf, output_dir, args.converter, args.force, page_filter, meta_mtime)] = pdf
        _errors: list[str] = []
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                msg = f"  ERROR  {futures[future].name}: {exc}"
                print(msg, flush=True)
                _errors.append(msg)
    print("\nDone.")
    if _errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
