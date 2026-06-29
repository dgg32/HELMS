#!/usr/bin/env python3
"""Semantic grounding + structural checking for extracted KG triples."""
from __future__ import annotations
import json
import os
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from llm_client import model_call_params, completion_structured  # noqa: E402
import extract as _extract_mod  # noqa: E402
import grounding as _grounding  # noqa: E402
# Validator base + per-naming-service validators. Add a new naming service by
# writing agents/<svc>_check.py and appending its validator to
# _DETERMINISTIC_VALIDATORS below — no edits to this engine are needed.
from .validators_base import TripleValidator, Verdict, ValidatorContext  # noqa: E402,F401
from .umls_check import UMLSSemanticValidator  # noqa: E402
from .gleif_check import GLEIFResolutionValidator, _gleif_name_suspicious  # noqa: E402  (raw-term suspicion gate, #2)
from .plausibility_check import PlausibilityValidator  # noqa: E402

_MAX_DOC_CHARS = 60_000
_BATCH_SIZE    = 25

# Core grounding/constraint prompt. Service-specific paragraphs (UMLS semantic
# types, GLEIF subsidiary warnings, …) are NOT here — each validator contributes
# its own slice via `prompt_fragment`, spliced in at {service_fragments} below.
_SYSTEM_PROMPT_CORE = """\
You are a knowledge graph quality checker. Given a document excerpt and a list of triples, \
perform two checks per triple.

The supporting quote has ALREADY been verified against the source document before you see it. \
Each triple carries a 'quote_verbatim' boolean:
   - quote_verbatim=true  : the quote was matched character-for-character to the source document. \
TRUST IT as authoritative document text. Do NOT doubt whether the quote (or its entities) "appears \
in the excerpt" — the excerpt above is only a truncated window and the quote may come from elsewhere \
in the document. Presence in the document is already established; do not re-litigate it.
   - quote_verbatim=false : the automatic locator could not match the quote as one contiguous span. \
This is OFTEN just a locator limitation (the quote was stitched from a comma-separated list or a table, \
which the matcher cannot find as a single run), NOT evidence of hallucination. If the entities are \
clearly visible in the excerpt or quote, judge on that and do NOT treat the quote as suspicious. \
quote_verbatim=false is grounds for caution (at most yellow) only when the entities ALSO cannot be \
found anywhere in the excerpt or quote; it is never by itself a reason to color red.
If a 'quote_context' field is present, it is verbatim document text surrounding the quote (provided \
because the quote falls outside the truncated excerpt). It is also authoritative.

1. ENTITY GROUNDING — judge whether each entity is actually referred to by the supporting quote \
(and its quote_context, if present).
   Color rules (per entity, from_color / to_color):
   - green  : the entity is clearly referred to in the quote, directly or via a synonym, abbreviation, \
or alternate name (e.g. "Nephrolithiasis" matches "kidney stone"; "NVIDIA CORPORATION" matches "NVIDIA"). \
With quote_verbatim=true this is the normal, expected case — do NOT downgrade to yellow merely because \
you cannot see the quote in the excerpt.
   - yellow : the entity reference is genuinely indirect or uncertain, or no quote was provided. \
(Do NOT drop to yellow merely because quote_verbatim=false when the entity is plainly present.)
   - red    : the entity is not referred to in the quote or quote_context at all.
   If supporting_quote has multiple ' / '-joined segments, judge each independently — one strong \
segment is sufficient to color an entity green.
   - DO NOT GROUND FROM THE WIDER EXCERPT: judge each triple's entities against THAT triple's own \
quote (and quote_context, if present) ONLY — never against the document excerpt shown above as a \
whole, even when the whole document is short enough to be fully visible there. The excerpt is shared \
context for the batch, not grounding for any single triple. An entity that appears elsewhere in the \
document (its title, an earlier paragraph, a different list item) but NOT in this triple's own quote \
is NOT grounded for this triple. Example (must be red, not green): a document about NVIDIA's GB200 \
supply chain explicitly states elsewhere that NVIDIA designs chips while OEMs assemble racks, and \
lists CoolIT among many cooling vendors "in between." A triple CoolIT --PROVIDES--> NVIDIA with quote \
"Cooling System: ... (Chillers/CDUs by CoolIT, Asetek, Submer)" must NOT get to_color=green for NVIDIA \
just because "NVIDIA" appears elsewhere in the document. The quote itself never names NVIDIA, so \
to_color=red and constraint_violated=true.
   - SINGLE-SUBJECT DOCUMENT OVERRIDE: a triple may carry 'from_is_document_subject': true (or \
'to_is_document_subject': true). That entity is the ONE subject the whole document is about — it is \
the only entity of its type in the document (e.g. the single drug a label describes, which the doc \
names in its title and throughout). Such an entity is grounded for EVERY relation it participates in: \
color it green even when its own quote does not repeat its name, because the entire document is about \
it. This OVERRIDES "DO NOT GROUND FROM THE WIDER EXCERPT" for the flagged subject entity ONLY — it does \
NOT license grounding any other entity from the excerpt. Example (must be green): in a DIFICID \
(fidaxomicin) drug label, the triple fidaxomicin --HAS_ADVERSE_EFFECT--> Nausea with quote "The most \
common adverse reactions ... are nausea, vomiting, ..." has from_is_document_subject=true; the quote \
grounds Nausea, and fidaxomicin is the document's sole subject, so from_color=green, to_color=green, \
constraint_violated=false. The CoolIT/NVIDIA case differs: that article names many companies, so NO \
entity is flagged as the subject and NVIDIA stays red.
   - INSTRUCTIONS MAY NAME THE SUBJECT: an 'Extraction instructions' block may appear before the \
document excerpt. Use it to decide document-subject intent. (a) If it names a single PRIMARY subject \
(e.g. "extract for the PRIMARY substance the label is about"), treat that one entity as the document \
subject and ground it document-wide EXACTLY as if from_is_document_subject were set — even if a second \
entity of the same type also appears (a stray comparator drug). Identify the primary entity from the \
title / what the document is chiefly about; ground ITS relations, but do NOT extend the override to the \
comparator's own relations. (b) If the instructions say to capture relationships REGARDLESS of any \
primary subject (i.e. the document is explicitly multi-subject, like a supply-chain article covering \
many companies), do NOT apply the single-subject override at all — judge every entity strictly against \
its own quote. When no instructions are given, rely only on the from_is_document_subject / \
to_is_document_subject flags above.

2. RELATION SUPPORT, DIRECTION & CONSTRAINTS — judge whether the quote actually asserts the \
relationship 'from_entity --[rel_type]--> to_entity' IN THIS DIRECTION. This is the most important \
check: a quote that names both entities but states a different or reversed relationship does NOT \
support the triple.
   - DIRECTION: the relationship is directional. A quote supporting the reverse is a violation. \
Example: for a supplier relationship, "A supplies components to B" supports 'A --PROVIDES--> B' but \
NOT 'B --PROVIDES--> A'. If the quote supports the opposite direction, set constraint_violated=true.
   - SUPPORT: the quote must assert THIS relationship, not merely co-mention the two entities. \
An unrelated sentence that happens to contain both names is not support. A bare section or column \
HEADER on its own (e.g. just "Adverse Reactions" or "Frequency") asserts no specific fact and is not \
support. \
   - LIST / TABLE ENTRIES ARE VALID SUPPORT: for membership-style relationships (e.g. \
HAS_ADVERSE_EFFECT, HAS_INDICATION), an entry within a labeled list or table IS legitimate support, \
because the list's heading supplies the relationship. Example: "Hyperglycemia" listed under an \
"Adverse Reactions" section supports a HAS_ADVERSE_EFFECT triple for that drug — do NOT redden it for \
being "only a list entry." This differs from a bare header (no specific entity) and from a co-mention \
in unrelated prose.
   - MULTI-SEGMENT QUOTES (best segment wins): if supporting_quote has multiple ' / '-joined \
segments, the relationship is SUPPORTED when ANY ONE segment asserts \
'from_entity --[rel_type]--> to_entity' in the correct direction. Do NOT set constraint_violated \
merely because another segment is weaker, off-topic, or only co-mentions the entities: one segment \
that genuinely asserts the relation in the right direction is sufficient. Only set \
constraint_violated when NO segment supports the relation, or when a segment asserts the REVERSED \
direction and none asserts the correct one.
   - QUOTE_CONTEXT ENTITY MENTIONS MUST BELONG TO THE SAME STATEMENT: an entity name appearing \
inside quote_context only grounds that entity, or supports the relationship, if it is part of the \
SAME sentence, list item, bullet, or table row that names the other entity / asserts the relation. \
An entity that merely appears nearby, in a different, unrelated list item, bullet, or sentence, \
does NOT ground it and does NOT support the relationship, even though the name is technically \
present in the context window. Example (NOT support): quote = "Cooling System: ... (Chillers/CDUs \
by CoolIT, Asetek, Submer)" with "NVIDIA" appearing only in the preceding, unrelated bullet about a \
different subsystem's vendors. This does not support CoolIT --PROVIDES--> NVIDIA: nothing in the \
statement about CoolIT's cooling products names or implies NVIDIA as the customer. Color the \
unsupported entity red (not referred to by the relevant statement) and set constraint_violated=true.
   - SUBJECT-GROUNDED SUPPORT: when from_is_document_subject (or to_is_document_subject) is true and \
the quote grounds the OTHER side as a valid member/attribute of the relationship (e.g. an entry in the \
document's adverse-reactions or indications list), the relationship IS supported — do NOT set \
constraint_violated merely because the single subject is not repeated inside that quote. The document's \
sole subject supplies that side. Direction and plausibility are still judged normally (a quote \
asserting the REVERSED relation is still a violation).
   Each triple may also include these schema-derived fields:
   - 'extraction_constraint': what is and is not allowed for the relationship (from extract_prompt)
   - 'from_hint' / 'to_hint': what kind of entity each side should be
   - 'from_node_description' / 'to_node_description': full definition of the expected node type
   - 'from_is_document_subject' / 'to_is_document_subject': true when this entity is the sole subject \
of the entire document (see the single-subject override in rule 1 — ground it green document-wide)
{service_fragments}
   Detect violations such as: the quote does not support the relationship; the direction is reversed; \
an adverse effect observed only in animals and not confirmed in humans; a mechanism described as a \
downstream effect rather than a receptor/target class; an entity on the wrong side of the relationship \
(e.g. brand name where INN is expected); or any service-specific mismatch described above. \
Set constraint_violated=true if ANY violation is found. This field drives filtering — a violated \
triple is colored red regardless of entity grounding.

Provide a brief 'opinion' (1-2 sentences) covering: color rationale, any reference/synonym concerns, \
and especially any relation-support or direction problem.

Return only the JSON — no prose.\
"""


class _TripleGrounding(BaseModel):
    triple_id:           str
    from_color:          Literal["green", "yellow", "red"]
    to_color:            Literal["green", "yellow", "red"]
    constraint_violated: bool
    opinion:             str


class _GroundingResult(BaseModel):
    results: list[_TripleGrounding]


def _build_items(
    triples: list[dict],
    rel_constraints: dict[str, str],
    rel_hints: dict[str, dict],
    node_descriptions: dict[str, str],
    schema_nodes: dict | None = None,
    doc_text: str = "",
    excerpt_len: int = 0,
    validators: "list[TripleValidator] | None" = None,
    ctx: "ValidatorContext | None" = None,
) -> list[dict]:
    # Document-subject detection (domain-agnostic). A label whose entities collapse
    # to a SINGLE distinct value across the whole document — and that recurs in ≥2
    # triples — is the document's subject (e.g. the one drug a label describes). Its
    # grounding is established document-wide, so a quote that grounds the OTHER side
    # supports the relation even when the subject is not repeated in that quote. A
    # multi-entity document (e.g. a supply-chain article naming many companies) has no
    # such sole label, so this never fires there — the CoolIT/NVIDIA rule still holds.
    _label_entities: dict[str, set] = {}
    _label_triple_count: dict[str, int] = {}
    for _t in triples:
        if _t.get("_deleted"):
            continue
        for _lbl_key, _pk_key, _props_key in (
            ("from_label", "from_pk", "from_props"),
            ("to_label",   "to_pk",   "to_props"),
        ):
            _lbl = _t.get(_lbl_key, "")
            _val = (_t.get(_props_key) or {}).get(_t.get(_pk_key, ""))
            if _lbl and _val is not None:
                _label_entities.setdefault(_lbl, set()).add(_val)
                _label_triple_count[_lbl] = _label_triple_count.get(_lbl, 0) + 1
    _subject_labels = {
        lbl for lbl, ents in _label_entities.items()
        if len(ents) == 1 and _label_triple_count.get(lbl, 0) >= 2
    }

    items = []
    for t in triples:
        if t.get("_deleted"):
            continue
        tid = t.get("_id", "")
        fp  = t.get("from_props") or {}
        tp  = t.get("to_props")   or {}
        fpk = t.get("from_pk", "")
        tpk = t.get("to_pk",   "")
        rel = t.get("rel_type", "")
        quote = (t.get("supporting_quote") or "").strip()
        evidence = t.get("evidence") or []
        item: dict = {
            "triple_id":   tid,
            "rel_type":    rel,
            "from_entity": fp.get("name") or fp.get(fpk, ""),
            "to_entity":   tp.get("name") or tp.get(tpk, ""),
            "quote":       quote,
        }
        # Deterministic presence. Evidence spans carry char offsets located at
        # extraction time (extract._build_evidence), so presence is already PROVEN —
        # no LLM, no re-locate here. We surface it as `quote_verbatim` so the LLM
        # stops re-judging presence (the old excerpt-visibility yellow) and spends
        # its call on relation support. quote_verbatim is true only when EVERY span
        # located (matching the extraction re-anchor contract); a partial multi-span
        # quote must not be advertised as fully grounded. When a span sits beyond the
        # truncated excerpt, attach its verbatim neighborhood (sliced directly from
        # the known offsets) so the LLM grounds direction/support in real context.
        if evidence:
            _all_located = True
            for sp in evidence:
                s = sp.get("start")
                if s is None:
                    _all_located = False
                elif excerpt_len and s >= excerpt_len and "quote_context" not in item:
                    e  = sp.get("end") or s
                    lo = max(0, s - 200)
                    hi = min(len(doc_text), e + 200)
                    item["quote_context"] = doc_text[lo:hi]
            item["quote_verbatim"] = _all_located
        elif quote and doc_text:
            # Legacy fallback for runs written before the evidence field existed:
            # split on " / " and locate each segment.
            _segs = [s.strip() for s in (quote.split(" / ") if " / " in quote else [quote]) if s.strip()]
            _all_located = bool(_segs)
            for _seg in _segs:
                span = _grounding.locate(_seg, doc_text)
                if span is None:
                    _all_located = False
                elif excerpt_len and span[0] >= excerpt_len and "quote_context" not in item:
                    item["quote_context"] = _grounding.context(_seg, doc_text, 200)
            item["quote_verbatim"] = _all_located
        if rel in rel_constraints:
            item["extraction_constraint"] = rel_constraints[rel]
        hints = rel_hints.get(rel, {})
        if hints.get("from_hint"):
            item["from_hint"] = hints["from_hint"]
        if hints.get("to_hint"):
            item["to_hint"] = hints["to_hint"]
        from_desc = node_descriptions.get(t.get("from_label", ""))
        to_desc   = node_descriptions.get(t.get("to_label",   ""))
        if from_desc:
            item["from_node_description"] = from_desc
        if to_desc:
            item["to_node_description"] = to_desc
        if t.get("from_label", "") in _subject_labels:
            item["from_is_document_subject"] = True
        if t.get("to_label", "") in _subject_labels:
            item["to_is_document_subject"] = True

        # Service-specific item annotations (UMLS semantic types, GLEIF subsidiary
        # warnings, …) are contributed by each validator's annotate_item hook.
        if validators and ctx is not None:
            for _v in validators:
                _v.annotate_item(t, item, ctx)

        items.append(item)
    return items


def _build_user_msg(items: list[dict], doc_excerpt: str, instructions: str = "") -> str:
    """Assemble the grader's user message: optional instructions block, excerpt, triples.

    The instructions block (meta.yaml) is shown only to judge document-subject intent
    (rule 1), and only when non-empty — variant A passes "" and gets no block.
    """
    instr_block = (
        f"Extraction instructions for this document (use ONLY to judge document-subject "
        f"intent per rule 1):\n{instructions.strip()}\n\n"
        if instructions and instructions.strip() else ""
    )
    return (
        f"{instr_block}"
        f"Document excerpt (first {len(doc_excerpt)} chars):\n{doc_excerpt}\n\n"
        f"Triples to check:\n{json.dumps(items, ensure_ascii=False, indent=2)}"
    )


def _call_llm_batch(
    items: list[dict], doc_excerpt: str, instructions: str = ""
) -> dict[str, _TripleGrounding]:
    """Semantic grounding via LiteLLM — all providers use native schema enforcement.

    ``instructions`` is the document's extraction instructions (meta.yaml). When
    present they are shown to the grader so it can judge document-subject intent
    (e.g. "the PRIMARY substance" vs "regardless of the primary subject").

    Retries up to 3 times with exponential backoff on transient failures via the
    shared `completion_structured` helper — which adds reasoning_effort auto-recovery,
    rate-limit Retry-After parsing, and fatal auth classification on top of the plain
    backoff (parity with the extraction path's `_acreate_litellm`).
    """
    # The grader can run on a separate model — and provider — from the extractor
    # via SEMANTIC_CHECK_MODEL, so no model grades its own extraction. Params are
    # resolved into explicit kwargs via model_call_params (NO os.environ mutation):
    # this runs in asyncio.to_thread concurrently with extraction, which reads the
    # same global env, so an env swap here would corrupt a concurrent extraction
    # call (it would send the extraction model to the grader's provider). Unset
    # SEMANTIC_CHECK_MODEL ⇒ params come from the pipeline's current env.
    # completion_structured honours this: it never touches os.environ, taking the
    # model + creds purely as explicit kwargs.
    _sc_model = os.environ.get("SEMANTIC_CHECK_MODEL", "").strip()
    full_model, _call_kwargs = model_call_params(_sc_model or None)
    user_msg = _build_user_msg(items, doc_excerpt, instructions)
    result = completion_structured(
        model=full_model,
        system_prompt=_SYSTEM_PROMPT,
        user_msg=user_msg,
        response_model=_GroundingResult,
        max_completion_tokens=getattr(_extract_mod, "_LLM_MAX_COMPLETION_TOKENS", 8192),
        timeout=float(os.environ.get("LLM_TIMEOUT", "120")),
        call_kwargs=_call_kwargs,
        retries=3,
        base_delay=2.0,
        log_prefix="[semantic_check]",
    )
    return {r.triple_id: r for r in result.results}


def _deterministic_flags(
    triples: list[dict],
    schema_nodes: dict | None = None,
    filter_level: str = "moderate",
) -> dict[str, dict[str, list[str]]]:
    """Structural checks that need no LLM: empty PKs, duplicates.

    Returns `tid -> {"empty": [...], "notes": [...]}`:
      - `empty`  = empty primary-key issues. A triple with no PK value is
        structurally invalid (it cannot be written as a node), so the caller
        red-floors it rather than leaving it note-only.
      - `notes`  = advisory issues (duplicates) — note-only, never recolor.

    Domain plausibility (treatment→procedure) and naming-service checks (UMLS
    semantic type / vocab, GLEIF subsidiary names) live in their own validators
    (agents/plausibility_check.py, agents/umls_check.py, agents/gleif_check.py).
    schema_nodes is retained for signature compatibility but no longer read here.
    """
    seen: dict[tuple, str] = {}
    flags: dict[str, dict[str, list[str]]] = {}

    for t in triples:
        if t.get("_deleted"):
            continue
        tid      = t.get("_id", "")
        fp       = t.get("from_props") or {}
        tp       = t.get("to_props")   or {}
        fpk      = t.get("from_pk", "")
        tpk      = t.get("to_pk",   "")
        fpv      = fp.get(fpk) or fp.get("name", "")
        tpv      = tp.get(tpk) or tp.get("name", "")
        rel_type = t.get("rel_type", "")
        empty: list[str] = []
        notes: list[str] = []

        if not fpv:
            empty.append(f"from_props.{fpk} is empty")
        if not tpv:
            empty.append(f"to_props.{tpk} is empty")

        key = (fpv, tpv, rel_type)
        if key in seen:
            notes.append(f"Duplicate of triple {seen[key]}")
        else:
            seen[key] = tid

        if empty or notes:
            flags[tid] = {"empty": empty, "notes": notes}
    return flags


# ── Composable deterministic validators ──────────────────────────────────────
#
# Each validator is pure-Python (NO LLM call). The single LLM grounding call
# (`_call_llm_batch`) produces the base color; deterministic verdicts are merged
# on top by `worst_color` precedence (red > yellow > green), so order does not
# affect correctness. To add a naming service (UMLS, GLEIF, Wikidata, …): write
# agents/<svc>_check.py with a TripleValidator subclass (annotate_item +/or check
# +/or prompt_fragment) and append an instance to `_DETERMINISTIC_VALIDATORS`
# below — zero LLM cost, no edits to this engine. An LLM-backed custom validator
# must join the existing batch, never spawn its own call.
#
# Verdict / ValidatorContext / TripleValidator are defined in validators_base and
# imported (and re-exported) at the top of this module.


def _load_rejected_logical(
    harvest_dir: "str | Path | None", doc_name: "str | None"
) -> set[tuple[str, str, str]]:
    """Build the (rel_type, from_display, to_display) set of human-rejected triples."""
    rejected: set[tuple[str, str, str]] = set()
    if not harvest_dir:
        return rejected
    _hd = Path(harvest_dir)
    for _jl in _hd.glob("*.jsonl"):
        try:
            for _line in _jl.read_text(encoding="utf-8").splitlines():
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _e = json.loads(_line)
                    if _e.get("source") == "rejected":
                        if doc_name is None or _e.get("doc_name") == doc_name:
                            rejected.add((_e["rel_type"], _e["from_display"], _e["to_display"]))
                except Exception:
                    pass
        except Exception:
            pass
    return rejected


def _node_source(node_def: dict | None) -> str:
    """Resolver source of a node ('umls' | 'gleif' | 'llm' | ''), read from its
    property `source` fields (the same field NodeResolver.handles matches on)."""
    for p in (node_def or {}).get("properties", []) or []:
        src = p.get("source")
        if src in ("umls", "gleif"):
            return src
    return "llm" if node_def else ""


class EntityPresenceValidator(TripleValidator):
    """Deterministic proof that an entity is named in its own supporting quote.

    `grounding.locate` proves token-level presence (markdown / curly-quote / citation
    tolerant, whole-word so no substring false positives). When ANY accepted name for
    the entity appears contiguously in ANY ' / '-joined quote segment, presence is PROVEN
    without the LLM — so we pin that entity's color green via from_color_anchor /
    to_color_anchor, a deterministic floor the LLM cannot lower. This removes the
    false-red entity class (the LLM reddening an entity that is literally in the quote).

    Accepted names per side (#2 — extend the anchor to the document's own surface form):
      1. the RESOLVED name (always);
      2. the RAW extracted term (`from_term`/`to_term`) — what the DOCUMENT actually
         called the entity, the most faithful synonym there is (doc says "kidney stone",
         resolved "Nephrolithiasis"; doc says "rash", resolved "Skin rash"). The raw term
         is GATED by resolver source: for a same-concept source (UMLS / LLM) it is always
         a valid synonym; for GLEIF (where a short term can resolve to a DIFFERENT
         corporate entity) it is dropped when `_gleif_name_suspicious` fires ("Quanta" ->
         "QUANTA LYON"), so a suspicious resolution is left for its validator / the LLM,
         not greened on the raw string. Unknown source is gated conservatively (over-
         blocking only costs an LLM judgement — the LLM still grounds it — while under-
         blocking would auto-green a possible mis-resolution).
      3. any resolver-supplied `synonyms` in `from_meta`/`to_meta` (always trusted —
         these are vetted: UMLS atoms, GLEIF other/trade names). The hook is in place; it
         is empty until a resolver persists it (a richer, API-backed follow-on).

    Scope stays tight: entity (grounding) axis ONLY — constraint_violated (relation
    support + direction) is untouched, so a present-but-reversed edge still reds, and a
    red color_floor (semantic-type mismatch, fabricated quote) still wins the EDGE color.

    annotate_item also surfaces the proof to the LLM (from_entity_in_quote /
    to_entity_in_quote) so it stops re-judging presence and spends its budget on
    relation support — the anchor is the post-LLM safety net behind that hint.
    """
    name = "entity_presence"
    prompt_fragment = (
        "   - PROVEN PRESENCE (deterministic): a triple may carry "
        "'from_entity_in_quote': true and/or 'to_entity_in_quote': true. This means a name "
        "for the entity (its resolved name, the raw extracted term, or a known synonym) was "
        "located VERBATIM in its supporting quote by the offline locator — its presence is "
        "already PROVEN. Color that entity green and do NOT re-judge whether it 'appears'; "
        "spend your judgement on relation support and direction instead. (Absence of the flag "
        "does NOT imply absence — the entity may still be present via a synonym the locator "
        "cannot match; judge those normally.)"
    )

    @staticmethod
    def _entity_name(props: dict | None, pk: str) -> str:
        p = props or {}
        return p.get("name") or p.get(pk) or ""

    @staticmethod
    def _segments(quote: str) -> list[str]:
        return [s.strip() for s in (quote.split(" / ") if " / " in quote else [quote]) if s.strip()]

    @staticmethod
    def _raw_term_ok(raw: str, resolved: str, src: str) -> bool:
        """Whether the raw extracted term is a trustworthy synonym for grounding.

        UMLS/LLM sources expand to the SAME concept, so the doc's surface form is always
        valid. GLEIF (and unknown source) can resolve a short term to a DIFFERENT corporate
        entity, so the raw term is dropped when the resolved name looks suspicious."""
        if not raw:
            return False
        if src in ("umls", "llm"):
            return True
        return not _gleif_name_suspicious(raw, resolved)

    def _accepted_names(self, t: dict, side: str, ctx: "ValidatorContext | None") -> list[str]:
        props = t.get(f"{side}_props")
        pk    = t.get(f"{side}_pk", "")
        resolved = self._entity_name(props, pk)
        names: list[str] = [resolved] if resolved else []
        schema_nodes = (ctx.schema_nodes if ctx else None) or {}
        src = _node_source(schema_nodes.get(t.get(f"{side}_label", "")))
        raw = (t.get(f"{side}_term") or "").strip()
        if raw and raw.lower() != resolved.lower() and self._raw_term_ok(raw, resolved, src):
            names.append(raw)
        for syn in (t.get(f"{side}_meta") or {}).get("synonyms", []) or []:
            if syn and isinstance(syn, str):
                names.append(syn)
        return names

    def _presence(self, t: dict, ctx: "ValidatorContext | None" = None) -> tuple[bool, bool]:
        quote = (t.get("supporting_quote") or "").strip()
        if not quote:
            return (False, False)
        segs = self._segments(quote)
        f_names = self._accepted_names(t, "from", ctx)
        t_names = self._accepted_names(t, "to",   ctx)
        f_present = any(_grounding.locate(n, seg) for n in f_names for seg in segs)
        t_present = any(_grounding.locate(n, seg) for n in t_names for seg in segs)
        return (f_present, t_present)

    def annotate_item(self, triple: dict, item: dict, ctx: ValidatorContext) -> None:
        f_present, t_present = self._presence(triple, ctx)
        if f_present:
            item["from_entity_in_quote"] = True
        if t_present:
            item["to_entity_in_quote"] = True

    def check(self, triples: list[dict], ctx: ValidatorContext) -> dict[str, Verdict]:
        out: dict[str, Verdict] = {}
        for t in triples:
            if t.get("_deleted"):
                continue
            f_present, t_present = self._presence(t, ctx)
            if f_present or t_present:
                out[t.get("_id", "")] = Verdict(
                    from_color_anchor="green" if f_present else None,
                    to_color_anchor="green"   if t_present else None,
                )
        return out


# Shared helper instance so the relation-support hints (#3) reuse the presence logic
# (accepted names + the source-gated raw term) without recomputing or duplicating it.
_PRESENCE_HELPER = EntityPresenceValidator()

# Negation / hedge cues that may flip a quote from ASSERTING a relation to denying or
# qualifying it (e.g. "not observed in humans", "ruled out", "in animals"). Word-bounded
# so "no" does not match inside "nodule" and "not" not inside "another". ADVISORY ONLY —
# the LLM still decides; this never sets a deterministic verdict.
_NEGATION_CUES = re.compile(
    r"\b(?:not|no|without|never|absence of|ruled out|denied|"
    r"in animals|not in humans|no evidence|did not|does not|were not|was not)\b",
    re.IGNORECASE,
)


class RelationSupportHintValidator(TripleValidator):
    """#3 — precompute deterministic SIGNALS for the LLM's relation-support judgment.

    Annotate-only (NO verdict): support and direction stay the LLM's job (the hard
    language-understanding part). These are precomputed facts the LLM would otherwise
    re-derive by re-reading the quote, handed to it so it spends its budget on the call,
    not the bookkeeping:

      * relation_endpoints_colocated: some ' / ' quote segment names BOTH endpoints.
        Co-presence in one statement is a NECESSARY (not sufficient) condition for a quote
        to assert from->to, so its ABSENCE is a useful 'likely not supported' signal — with
        ONE documented exception the LLM already handles: the single-subject override, where
        the subject legitimately need not appear in the quote. Hence ADVISORY, never a verdict.
      * possible_negation_in_quote: a negation/hedge cue appears in the quote. ADVISORY:
        the LLM must verify the quote ASSERTS the relation rather than denying/qualifying it;
        this is never a violation by itself (many negations are irrelevant to the relation).
    """
    name = "relation_support_hint"
    prompt_fragment = (
        "   - RELATION SUPPORT HINTS (deterministic, ADVISORY — they inform check #2, never "
        "decide it): 'relation_endpoints_colocated': true means some quote segment names BOTH "
        "entities, a necessary condition for the quote to assert the relation; if it is ABSENT "
        "and the entity is not a document subject, weigh whether the quote really supports "
        "from->to. 'possible_negation_in_quote': true means a negation/hedge word appears in the "
        "quote — verify the quote ASSERTS the relation rather than denying or qualifying it; do "
        "NOT treat this flag as proof of a violation on its own."
    )

    def annotate_item(self, triple: dict, item: dict, ctx: ValidatorContext) -> None:
        quote = (triple.get("supporting_quote") or "").strip()
        if not quote:
            return
        segs = _PRESENCE_HELPER._segments(quote)
        f_names = _PRESENCE_HELPER._accepted_names(triple, "from", ctx)
        t_names = _PRESENCE_HELPER._accepted_names(triple, "to", ctx)
        for seg in segs:
            if (any(_grounding.locate(n, seg) for n in f_names)
                    and any(_grounding.locate(n, seg) for n in t_names)):
                item["relation_endpoints_colocated"] = True
                break
        if _NEGATION_CUES.search(quote):
            item["possible_negation_in_quote"] = True


class StructuralValidator(TripleValidator):
    """Empty PKs -> hard red floor (structurally invalid, unwritable); duplicates -> note-only."""
    name = "structural"

    def check(self, triples: list[dict], ctx: ValidatorContext) -> dict[str, Verdict]:
        flags = _deterministic_flags(
            triples, schema_nodes=ctx.schema_nodes, filter_level=ctx.filter_level
        )
        out: dict[str, Verdict] = {}
        for tid, f in flags.items():
            msgs = f["empty"] + f["notes"]
            out[tid] = Verdict(
                # An empty primary key means the triple cannot be written at all —
                # red-floor it deterministically rather than relying on the LLM.
                color_floor="red" if f["empty"] else None,
                note="[Structural: " + "; ".join(msgs) + "]",
            )
        return out


class HarvestRejectionValidator(TripleValidator):
    """Force red any triple whose logical key matches a previously human-rejected harvest entry."""
    name = "harvest_rejection"

    def check(self, triples: list[dict], ctx: ValidatorContext) -> dict[str, Verdict]:
        rejected = _load_rejected_logical(ctx.harvest_dir, ctx.doc_name)
        if not rejected:
            return {}
        out: dict[str, Verdict] = {}
        for t in triples:
            fp  = t.get("from_props") or {}
            tp  = t.get("to_props")   or {}
            fpk = t.get("from_pk", "")
            tpk = t.get("to_pk",   "")
            from_d = fp.get("name") or fp.get(fpk) or "?"
            to_d   = tp.get("name") or tp.get(tpk) or "?"
            rel    = t.get("rel_type", "")
            if (rel, from_d, to_d) in rejected:
                out[t.get("_id", "")] = Verdict(
                    color_floor="red",
                    constraint_violated=True,
                    note="[Previously rejected by human reviewer]",
                )
        return out


class FabricatedQuoteValidator(TripleValidator):
    """Red-floor an agent-retry triple whose supporting quote could not be located
    verbatim ANYWHERE in the document.

    The agentic path supplies quotes from a long tool loop and can reconstruct or
    fabricate them (extraction_agent.add_triple already warns the agent on a failed
    reanchor). A quote with ZERO located evidence spans is suspect provenance even
    when the relation is true, so we red-floor it (NOT constraint_violated — the
    fact may be real) with a reason telling the reviewer to verify; the human can
    cycle it to green after reading the doc. The batch path is excluded on purpose:
    its grounding 'warn' is cosmetic (DESIGN_INVARIANTS §6) and `strict` already
    drops unlocatable quotes. A partially-verbatim quote (≥1 located span) is left
    to best-of-segment grounding.
    """
    name = "fabricated_quote"

    def check(self, triples: list[dict], ctx: ValidatorContext) -> dict[str, Verdict]:
        out: dict[str, Verdict] = {}
        for t in triples:
            src = t.get("extraction_source") or (t.get("rel_props") or {}).get("extraction_source")
            if src != "agent_retry":
                continue
            q  = (t.get("supporting_quote") or "").strip()
            ev = t.get("evidence") or []
            if q and ev and not any(e.get("start") is not None for e in ev):
                out[t.get("_id", "")] = Verdict(
                    color_floor="red",
                    note="[Supporting quote not found verbatim in the document — likely fabricated by the agent; read the doc to confirm before accepting]",
                )
        return out


class SchemaConformanceValidator(TripleValidator):
    """Red-floor a triple whose (rel_type, from_label, to_label) is not a declared
    schema edge.

    The batch extractor copies these three fields straight from the schema loop, so
    batch triples are conformant by construction. But human-added triples (Add Triple
    form), agent-retry triples, and hand-edited *_raw.json can carry an undeclared
    label, an undeclared rel_type, or a legal-parts-but-illegal-combination edge
    (e.g. MAY_TREAT: Substance -> AdverseEffect when the schema declares
    MAY_TREAT: Substance -> Indication). Such an edge cannot be written correctly,
    so it is a hard red floor + constraint_violated. No-ops when schema_rels is
    absent (nothing to check against).
    """
    name = "schema_conformance"

    def check(self, triples: list[dict], ctx: ValidatorContext) -> dict[str, Verdict]:
        if not ctx.schema_rels:
            return {}
        legal = {
            (r["rel_type"], r["from_node"], r["to_node"])
            for r in ctx.schema_rels
            if r.get("rel_type") and r.get("from_node") and r.get("to_node")
        }
        # No derivable edges (rels lack from_node/to_node) -> we have no reference
        # set to validate against. A hard red-floor must NOT fire on absence of
        # information, so no-op exactly as when schema_rels is missing entirely.
        if not legal:
            return {}
        out: dict[str, Verdict] = {}
        for t in triples:
            if t.get("_deleted"):
                continue
            rel = t.get("rel_type", "")
            fl  = t.get("from_label", "")
            tl  = t.get("to_label", "")
            if (rel, fl, tl) not in legal:
                out[t.get("_id", "")] = Verdict(
                    color_floor="red",
                    constraint_violated=True,
                    note=f"[Schema: '{fl} -{rel}-> {tl}' is not a declared edge in the schema]",
                )
        return out


# Registry — runs in order; verdicts merged by worst_color precedence (order-independent).
# Naming-service validators (UMLS, GLEIF, …) sit between the generic structural and
# harvest checks. Append a new naming service here; its prompt_fragment is spliced into
# the system prompt and its annotate_item/check hooks run automatically.
_DETERMINISTIC_VALIDATORS: list[TripleValidator] = [
    EntityPresenceValidator(),
    RelationSupportHintValidator(),
    StructuralValidator(),
    SchemaConformanceValidator(),
    PlausibilityValidator(),
    UMLSSemanticValidator(),
    GLEIFResolutionValidator(),
    FabricatedQuoteValidator(),
    HarvestRejectionValidator(),
]

# Assemble the full system prompt: core + each validator's prompt_fragment, spliced
# at the {service_fragments} marker in registry order.
_SYSTEM_PROMPT = _SYSTEM_PROMPT_CORE.replace(
    "{service_fragments}",
    "\n".join(v.prompt_fragment for v in _DETERMINISTIC_VALIDATORS if v.prompt_fragment),
)


def _best_anchor(verdicts: "list[Verdict]", attr: str) -> str | None:
    """Greenest entity-color anchor across a triple's deterministic verdicts, or None."""
    vals = [getattr(v, attr) for v in verdicts if getattr(v, attr)]
    return _extract_mod.best_color(*vals) if vals else None


def check_triples(
    triples: list[dict],
    doc_text: str,
    schema_rels: list | None = None,
    schema_nodes: dict | None = None,
    filter_level: str = "moderate",
    harvest_dir: "str | Path | None" = None,
    doc_name: "str | None" = None,
    instructions: str = "",
) -> list[dict]:
    """Re-color triples and annotate with ai_opinion.

    Runs:
      - LLM semantic grounding (green/yellow/red per entity)
      - LLM extraction-constraint check (extract_prompt from schema_rels)
      - Deterministic validators in `_DETERMINISTIC_VALIDATORS` (no LLM): structural
        (empty PKs, duplicates, treatment→procedure), naming-service checks (UMLS
        semantic type / vocab, GLEIF subsidiary names), and harvest rejection.
      - Harvest rejection: any triple whose logical key matches a previously rejected
        harvest entry is forced red regardless of LLM verdict.

    schema_rels:  relationship list from schema YAML — used to inject extract_prompt constraints.
    schema_nodes: node dict from schema YAML — passed to validators (UMLS type checks, …).
    harvest_dir:  project harvest/ directory — used to load rejection history.
    instructions: document extraction instructions (meta.yaml) — shown to the grader so it
                  can judge document-subject intent (e.g. "the PRIMARY substance" vs
                  "regardless of the primary subject"). Empty disables the intent signal.
    """
    doc_excerpt = doc_text[:_MAX_DOC_CHARS]

    rel_constraints: dict[str, str] = {}
    rel_hints: dict[str, dict] = {}
    if schema_rels:
        for rel in schema_rels:
            rt = rel["rel_type"]
            ep = (rel.get("extract_prompt") or "").strip()
            if ep:
                rel_constraints[rt] = ep
            fh = (rel.get("from_hint") or "").strip()
            th = (rel.get("to_hint")   or "").strip()
            if fh or th:
                rel_hints[rt] = {"from_hint": fh, "to_hint": th}

    node_descriptions: dict[str, str] = {}
    if schema_nodes:
        for label, node_def in schema_nodes.items():
            desc = (node_def.get("description") or "").strip()
            if desc:
                node_descriptions[label] = desc

    ctx = ValidatorContext(
        schema_nodes=schema_nodes,
        schema_rels=schema_rels,
        filter_level=filter_level,
        harvest_dir=harvest_dir,
        doc_name=doc_name,
    )

    items = _build_items(
        triples, rel_constraints, rel_hints, node_descriptions,
        schema_nodes=schema_nodes, doc_text=doc_text, excerpt_len=len(doc_excerpt),
        validators=_DETERMINISTIC_VALIDATORS, ctx=ctx,
    )
    if not items:
        return triples

    # Deterministic validators (no LLM) — collect verdicts per triple in registry order.
    det_verdicts: dict[str, list[Verdict]] = {}
    for _validator in _DETERMINISTIC_VALIDATORS:
        for _tid, _verdict in _validator.check(triples, ctx).items():
            det_verdicts.setdefault(_tid, []).append(_verdict)

    # LLM grounding — the single batched call that produces the base color.
    color_map: dict[str, _TripleGrounding] = {}
    for i in range(0, len(items), _BATCH_SIZE):
        batch = items[i : i + _BATCH_SIZE]
        try:
            color_map.update(_call_llm_batch(batch, doc_excerpt, instructions))
        except Exception as e:
            print(f"  [semantic_check] batch {i // _BATCH_SIZE + 1} failed: {e}", flush=True)

    # Merge: LLM verdict is the base color; deterministic floors raise it
    # (worst_color), constraint_violated is OR-ed, notes are appended in order.
    updated = []
    for t in triples:
        tid = t.get("_id", "")
        verdicts = det_verdicts.get(tid, [])
        notes = [v.note for v in verdicts if v.note]
        g = color_map.get(tid)

        if g is not None:
            # Deterministic green anchors (proven entity presence) pin the entity axis:
            # an entity located verbatim in its quote cannot be lowered below green by
            # the LLM. best_color picks the greenest of {LLM color, anchor}. The relation
            # axis (constraint_violated) is untouched, so a present-but-reversed edge
            # still reds, and a red color_floor (e.g. semantic-type mismatch, fabricated
            # quote) still wins the EDGE color below via worst_color.
            from_anchor = _best_anchor(verdicts, "from_color_anchor")
            to_anchor   = _best_anchor(verdicts, "to_color_anchor")
            from_color = _extract_mod.best_color(g.from_color, from_anchor) if from_anchor else g.from_color
            to_color   = _extract_mod.best_color(g.to_color,   to_anchor)   if to_anchor   else g.to_color
            t["from_color"] = from_color
            t["to_color"]   = to_color
            cv = g.constraint_violated
            color = "red" if cv else _extract_mod.worst_color(from_color, to_color)
            for v in verdicts:
                if v.constraint_violated:
                    cv = True
                if v.color_floor:
                    color = _extract_mod.worst_color(color, v.color_floor)
            if cv:
                color = "red"
            t["triple_color"]        = color
            t["constraint_violated"] = cv
            opinion = g.opinion.strip()
            if notes:
                opinion = (opinion + " " if opinion else "") + " ".join(notes)
            t["ai_opinion"]   = opinion
            t["_ai_reviewed"] = True
        else:
            t["_ai_reviewed"] = False
            floors = [v.color_floor for v in verdicts if v.color_floor]
            cv = any(v.constraint_violated for v in verdicts)
            color = _extract_mod.worst_color(*floors) if floors else None
            if cv:
                t["constraint_violated"] = True
                color = "red"
            if color is not None:
                t["triple_color"] = color
            if notes:
                t["ai_opinion"] = " ".join(notes)

        updated.append(t)
    return updated

