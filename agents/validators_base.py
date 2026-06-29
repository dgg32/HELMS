#!/usr/bin/env python3
"""Pluggable triple-validator base classes.

A validator may contribute up to three things to the semantic check, all optional:

  annotate_item(triple, item, ctx)  — PRE-LLM: inject service-specific fields into the
                                       LLM batch item so the grounding prompt sees them
                                       (e.g. expected/actual UMLS semantic types,
                                       a GLEIF subsidiary-name warning).
  check(triples, ctx) -> {tid: Verdict}
                                     — POST-LLM (zero LLM cost): deterministic verdicts
                                       merged onto the LLM color by worst_color precedence.
  prompt_fragment: str               — this validator's slice of the system prompt,
                                       appended to the core prompt at the {service_fragments}
                                       marker.

To add a naming service (UMLS, GLEIF, Wikidata, …): write one module with a
TripleValidator subclass implementing whichever hooks it needs, then append an
instance to `_DETERMINISTIC_VALIDATORS` in semantic_check_agent.py. No edits to
the shared check engine are required.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Verdict:
    """One validator's opinion on one triple.

    color_floor:         raise triple_color to at least this (worst_color); None = no color opinion.
    constraint_violated: OR-ed into the triple's constraint_violated flag.
    note:                appended verbatim to ai_opinion (already bracket-formatted).
    from_color_anchor:   pin from_color to AT LEAST this greenness (best_color); None = no opinion.
    to_color_anchor:     pin to_color to AT LEAST this greenness (best_color); None = no opinion.

    Anchors are the INVERSE of color_floor: a floor raises a triple toward red (a
    safety net against false greens), an anchor raises an ENTITY toward green (a
    safety net against false reds, e.g. an entity the LLM wrongly reds despite it
    being verbatim in the quote). Anchors touch the per-entity from_color/to_color
    ONLY — never constraint_violated — so relation support/direction stays the LLM's.
    """
    color_floor: str | None = None
    constraint_violated: bool = False
    note: str = ""
    from_color_anchor: str | None = None
    to_color_anchor: str | None = None


@dataclass
class ValidatorContext:
    """Shared inputs every validator may read."""
    schema_nodes: dict | None = None
    schema_rels: list | None = None
    filter_level: str = "moderate"
    harvest_dir: "str | Path | None" = None
    doc_name: "str | None" = None


class TripleValidator:
    """Deterministic (no-LLM) triple checker with optional prompt + item annotation hooks.

    Subclasses override whichever of the three hooks they need; the defaults are
    inert, so a validator that only annotates items (no post-LLM verdict) simply
    leaves `check` returning {}.
    """
    name: str = "validator"
    prompt_fragment: str = ""

    def annotate_item(self, triple: dict, item: dict, ctx: ValidatorContext) -> None:
        """Inject service-specific fields into the LLM batch `item` dict in place."""
        return None

    def check(self, triples: list[dict], ctx: ValidatorContext) -> dict[str, Verdict]:
        """Return {triple_id: Verdict} for deterministic (no-LLM) findings."""
        return {}
