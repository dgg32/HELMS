#!/usr/bin/env python3
"""Domain plausibility checks for the semantic check agent.

Post-LLM, note-only: flags relationships whose target looks semantically wrong for
the relationship type. Currently ships the medical rule "a treatment-type relation
should point at a disease/condition, not a diagnostic procedure". Other domains can
add their own rule sets here (or in a sibling validator) — the registry slot is the
extension point, no engine edits needed.
"""
from __future__ import annotations

from .validators_base import TripleValidator, ValidatorContext, Verdict

# Treatment-type rels should point at diseases/conditions, not diagnostic procedures.
_TREAT_LIKE_RELS = frozenset({
    "MAY_TREAT", "TREATS", "INDICATED_FOR", "PREVENTS", "MAY_PREVENT",
    "USED_FOR", "HAS_INDICATION", "APPROVED_FOR",
})
_PROCEDURE_MARKERS = (
    "detection of", "detection and identification", "detection",
    "measurement of", "measurement and", "measurement",
    "assay", "assay for",
    "screening for", "screening of", "screening",
    "monitoring of", "monitoring",
    "assessment of", "assessment",
    "analysis of", "analysis",
    "testing for", "testing",
    "diagnosis of", "diagnosis",
    "diagnostic test",
    "multiplex amplified",
    "probe technique",
    "imaging",
)


class PlausibilityValidator(TripleValidator):
    """Domain semantic plausibility. Note-only (no recolor) — mirrors prior structural behavior."""
    name = "plausibility"

    def check(self, triples: list[dict], ctx: ValidatorContext) -> dict[str, Verdict]:
        out: dict[str, Verdict] = {}
        for t in triples:
            if t.get("_deleted"):
                continue
            rel_type = t.get("rel_type", "")
            if rel_type not in _TREAT_LIKE_RELS:
                continue
            to_name = (t.get("to_props") or {}).get("name", "")
            to_name_lc = to_name.lower()
            for marker in _PROCEDURE_MARKERS:
                if marker in to_name_lc:
                    out[t.get("_id", "")] = Verdict(
                        note=(
                            f"[Plausibility: '{rel_type}' target '{to_name}' "
                            f"looks like a diagnostic procedure ('{marker}'), not a disease/condition]"
                        )
                    )
                    break
        return out
