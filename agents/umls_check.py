#!/usr/bin/env python3
"""UMLS semantic-type / vocabulary checks for the semantic check agent.

Pre-LLM: injects expected/actual UMLS semantic types into the batch item.
Post-LLM: deterministic note-only flags for semantic-type and vocabulary mismatch.
Prompt:  the SEMANTIC TYPE RULE paragraph that teaches the LLM how to read the
         injected fields.
"""
from __future__ import annotations

from .validators_base import TripleValidator, ValidatorContext, Verdict

# UMLS synthetic / meta sources — not meaningful vocabulary indicators.
_METATHESAURUS_SOURCES = frozenset({"MTH", "SRC", ""})

_PROMPT_FRAGMENT = """\
   - 'from_expected_semantic_types' / 'to_expected_semantic_types': UMLS semantic types the schema \
requires for each node (e.g. ["Pharmacologic Substance", "Antibiotic"])
   - 'from_actual_semantic_types' / 'to_actual_semantic_types': UMLS semantic types the resolved \
entity actually has
   SEMANTIC TYPE RULE: Use ONLY the injected `expected_semantic_types` and `actual_semantic_types` \
fields for type checking. Do NOT derive expected types from node label names, node descriptions, \
or relationship names — these describe the schema concept, not the UMLS type. A type check PASSES \
when `actual_semantic_types` contains AT LEAST ONE type that appears in `expected_semantic_types`. \
Flag a type mismatch ONLY when the two lists have ZERO overlap. Example: a node labelled \
'AdverseEffect' with expected=["Disease or Syndrome", "Pathologic Function", "Sign or Symptom", ...] \
and actual=["Disease or Syndrome"] PASSES — "Disease or Syndrome" is a valid UMLS type for \
drug-induced conditions like anemia or neutropenia even though they are adverse effects. \
An entity whose actual semantic types have ZERO overlap with the expected types is a semantic \
type mismatch — set constraint_violated=true (a violated triple is colored red regardless of grounding)."""


class UMLSSemanticValidator(TripleValidator):
    """UMLS semantic-type + vocabulary validation.

    A semantic-TYPE mismatch (resolved entity's actual UMLS types have ZERO
    overlap with the schema's declared `semantic_types`) is provable and
    deterministic, so it is a HARD red floor + constraint_violated — it no longer
    depends on the LLM choosing to act on the injected warning. A VOCAB mismatch
    stays note-only (it is the softer, more false-positive-prone signal; see the
    MTH caveat below). The `prompt_fragment` is kept so the LLM still explains the
    mismatch in `ai_opinion`, but the red verdict no longer hinges on the LLM.
    """
    name = "umls_semantic"
    prompt_fragment = _PROMPT_FRAGMENT

    def annotate_item(self, triple: dict, item: dict, ctx: ValidatorContext) -> None:
        schema_nodes = ctx.schema_nodes
        if not schema_nodes:
            return
        for side, label_key, meta_key in [
            ("from", "from_label", "from_meta"),
            ("to",   "to_label",   "to_meta"),
        ]:
            node_label = triple.get(label_key, "")
            exp_types  = (schema_nodes.get(node_label) or {}).get("semantic_types") or []
            actual     = (triple.get(meta_key) or {}).get("types", [])
            if exp_types:
                item[f"{side}_expected_semantic_types"] = exp_types
            if actual:
                item[f"{side}_actual_semantic_types"] = actual

    def check(self, triples: list[dict], ctx: ValidatorContext) -> dict[str, Verdict]:
        schema_nodes = ctx.schema_nodes
        if not schema_nodes:
            return {}
        out: dict[str, Verdict] = {}
        for t in triples:
            if t.get("_deleted"):
                continue
            tid = t.get("_id", "")
            fp  = t.get("from_props") or {}
            tp  = t.get("to_props")   or {}
            issues: list[str] = []
            type_mismatch = False
            for node_label, node_props, meta_key in [
                (t.get("from_label", ""), fp, "from_meta"),
                (t.get("to_label",   ""), tp, "to_meta"),
            ]:
                node_schema = schema_nodes.get(node_label) or {}
                exp = node_schema.get("semantic_types") or []
                if exp:
                    actual = (t.get(meta_key) or {}).get("types", [])
                    if actual and not any(s in exp for s in actual):
                        _resolved = node_props.get("name") or node_props.get("cui") or "the resolved entity"
                        issues.append(
                            f"Type mismatch: '{node_label}' expects {exp}, "
                            f"but '{_resolved}' has {actual}"
                        )
                        type_mismatch = True
                allowed_vocabs = node_schema.get("umls_vocabs") or []
                if allowed_vocabs:
                    root_src = (t.get(meta_key) or {}).get("root_source", "")
                    # MTH = NLM Metathesaurus synthetic source; common for all UMLS concepts
                    # regardless of which vocabulary matched — not a reliable vocab indicator.
                    # Only flag when root_source is a specific domain vocabulary clearly outside
                    # the allowed set (e.g. GO, CHEBI, NCBI on a node expecting MED-RT).
                    if root_src and root_src not in _METATHESAURUS_SOURCES and root_src not in allowed_vocabs:
                        _resolved = node_props.get("name") or node_props.get("cui") or "the resolved entity"
                        issues.append(
                            f"Vocab mismatch: '{node_label}' requires vocabs {allowed_vocabs}, "
                            f"but '{_resolved}' has root_source='{root_src}'"
                        )
            if issues:
                # Type mismatch is a hard, provable red floor (+ constraint_violated);
                # a vocab-only flag stays note-only (color_floor=None).
                out[tid] = Verdict(
                    color_floor="red" if type_mismatch else None,
                    constraint_violated=type_mismatch,
                    note="[UMLS: " + "; ".join(issues) + "]",
                )
        return out
