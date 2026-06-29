#!/usr/bin/env python3
"""Typed contract for the ``*_raw.json`` file — the spine of the pipeline.

Four stages write to / read from this file (batch extract, agent retry, review
layer, semantic check). Nothing used to enforce its shape, so drift between
writers was only caught later as a UI glitch or a wrong color, then patched with
another hand-written invariant. This module is the single machine-checked
definition.

The lever is ``extra="forbid"``: an unknown key (a typo, a writer inventing a
field) fails loudly at the boundary instead of silently flowing downstream. The
``evidence`` / ``supporting_quote`` / color fields are optional because they are
added by *later* stages (semantic check), so a freshly-extracted file is valid
before it has been colored.

CLI::

    python -m schema_raw                       # scan projects/*/runs/*/*_raw.json
    python -m schema_raw path/to/foo_raw.json  # validate specific file(s)

Exit code 1 if any file fails. Use in CI / pre-commit to lock the contract.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

Color = Literal["green", "yellow", "red"]


class Evidence(BaseModel):
    """One located grounding span: char offsets into ``doc_text`` + the text.

    Offsets are ``None`` when the quote could not be located (kept so the count
    of spans still matches the supporting_quote segments).
    """
    model_config = ConfigDict(extra="forbid")

    start: int | None
    end: int | None
    text: str


class RawTriple(BaseModel):
    """One extracted edge. ``evidence`` is the grounding source of truth;
    ``supporting_quote`` is a derived ``/``-joined projection of its text."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # Stable content-hash id assigned before write by both extract paths.
    id_: str = Field(alias="_id")

    # Structural identity — every writer always emits these.
    rel_type: str
    from_label: str
    to_label: str
    from_pk: str
    to_pk: str
    from_props: dict
    to_props: dict

    # Optional relationship / resolver metadata.
    rel_props: dict | None = None
    from_meta: dict | None = None
    to_meta: dict | None = None
    from_term: str | None = None
    to_term: str | None = None

    # Grounding (added at extraction time).
    evidence: list[Evidence] = []
    supporting_quote: str | None = None
    quote_unlocatable: bool | None = None

    # Color / verdict — written by the semantic-check agent (sole color authority),
    # so absent on a freshly-extracted, not-yet-checked triple.
    triple_color: Color | None = None
    from_color: Color | None = None
    to_color: Color | None = None
    constraint_violated: bool | None = None
    ai_opinion: str | None = None
    ai_reviewed: bool | None = Field(default=None, alias="_ai_reviewed")

    # Provenance: "agent_retry" for smart-retry triples; absent for batch.
    extraction_source: str | None = None


class RawFile(BaseModel):
    """Top-level ``*_raw.json`` document."""
    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    schema_version: str
    triples: list[RawTriple]

    # Optional metadata — emitted conditionally by the writers.
    doc: str | None = None
    doc_text: str | None = None
    doc_source: str | None = None
    schema_path: str | None = None
    filter_level: str | None = None
    grounding_warnings: list[dict] | None = None
    failed_chunks: list[int] | None = None
    ambiguous_pending: list[dict] | None = None


def validate_file(path: str | Path) -> list[str]:
    """Validate one ``*_raw.json``. Returns a list of human-readable errors
    (empty == valid). Does not raise."""
    p = Path(path)
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return [f"could not read/parse: {e}"]
    try:
        RawFile.model_validate(data)
    except ValidationError as e:
        return [f"{err['loc']}: {err['msg']}" for err in e.errors()]
    return []


def _main(argv: list[str]) -> int:
    targets = argv or sorted(glob.glob("projects/*/runs/*/*_raw.json"))
    if not targets:
        print("no *_raw.json files found")
        return 0
    failed = 0
    for t in targets:
        errs = validate_file(t)
        if errs:
            failed += 1
            print(f"FAIL {t}")
            for e in errs[:20]:
                print(f"     {e}")
        else:
            print(f"ok   {t}")
    print(f"\n{len(targets) - failed}/{len(targets)} valid")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
