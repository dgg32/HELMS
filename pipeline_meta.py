#!/usr/bin/env python3
"""Pipeline metadata: global extraction instructions + per-PDF page filtering.

Usage:
    meta = load_meta("project/drug_instruction.yaml")
    instructions = get_instructions(meta)          # str or ""
    pf = get_page_filter(meta, "beyfortus label", total_pages=30)
    # pf.pages → [1,2,...,10,12,13,14,15]

YAML format:
    instructions: |
      Only extract information about Beyfortus (nirsevimab-alip).
      Do NOT extract comparator drugs (palivizumab/Synagis).

    pages:
      beyfortus label:        # PDF stem (filename without .pdf)
        include: [1-10, 12-15]
      some other report:
        exclude: [1]
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class PageFilter:
    pages: list[int]  # final INCLUDE list (exclude already inverted / subtracted)
    mode: str         # "include", "exclude", or "include+exclude" — informational only


def load_meta(meta_path: str | Path | None) -> dict:
    """Load metadata YAML. Returns {} if meta_path is None or file not found."""
    if meta_path is None:
        return {}
    p = Path(meta_path)
    if not p.exists():
        print(f"[warn] --meta path not found: {p}")
        return {}
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except OSError as e:
        raise SystemExit(f"Cannot read '{p}': {e}")
    except yaml.YAMLError as e:
        raise SystemExit(f"Malformed YAML in '{p}': {e}")


def get_instructions(meta: dict) -> str:
    """Return global instructions text, or '' if absent."""
    return (meta.get("instructions") or "").strip()


def get_page_filter(meta: dict, pdf_stem: str, total_pages: int) -> Optional[PageFilter]:
    """Return PageFilter with final include list, or None (= all pages).

    include only  → expand ranges, clamp to [1..total_pages].
    exclude only  → expand ranges, invert against [1..total_pages].
    both          → start with include set, subtract exclude set.
    """
    pages_cfg = meta.get("pages", {})
    entry = pages_cfg.get(pdf_stem)
    if not entry:
        return None

    has_include = "include" in entry
    has_exclude = "exclude" in entry

    if has_include and has_exclude:
        include_set = {p for p in _expand_page_ranges(entry["include"]) if 1 <= p <= total_pages}
        exclude_set = set(_expand_page_ranges(entry["exclude"]))
        pages = sorted(include_set - exclude_set)
        return PageFilter(pages=pages, mode="include+exclude") if pages else None

    if has_include:
        pages = [p for p in _expand_page_ranges(entry["include"]) if 1 <= p <= total_pages]
        return PageFilter(pages=pages, mode="include") if pages else None

    if has_exclude:
        excluded = set(_expand_page_ranges(entry["exclude"]))
        pages = [p for p in range(1, total_pages + 1) if p not in excluded]
        return PageFilter(pages=pages, mode="exclude") if pages else None

    return None


def _expand_page_ranges(raw: list) -> list[int]:
    """Expand mixed list of ints and 'a-b' strings into a sorted unique int list."""
    pages: list[int] = []
    for item in raw:
        s = str(item).strip()
        if not s:
            continue
        if "-" in s:
            parts = s.split("-", 1)
            try:
                a, b = int(parts[0].strip()), int(parts[1].strip())
                if a > b:
                    print(f"  [warn] reversed page range {s!r} — skipping", flush=True)
                    continue
                pages.extend(range(a, b + 1))
            except (ValueError, IndexError):
                print(f"  [warn] skipping malformed page range: {s!r}")
        else:
            try:
                pages.append(int(s))
            except ValueError:
                print(f"  [warn] skipping malformed page value: {s!r}")
    return sorted(set(pages))


def save_meta(meta: dict, meta_path: Path) -> None:
    """Write metadata dict back to YAML (preserves insertion order, unicode-safe)."""
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        yaml.dump(meta, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
