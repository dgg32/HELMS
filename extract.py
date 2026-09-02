#!/usr/bin/env python3
"""
Extract triples from documents and write <stem>_review.json for human review.

Reads PDFs/Markdown, calls the LLM with schema-driven structured extraction,
resolves external-source properties (UMLS, GLEIF), and writes a JSON review
file for each document.  No graph or vector-store writes occur here — use
apply_graph.py to commit the (optionally edited) review files to the database.

Usage:
    python extract.py --schema finance_schema.yaml --input finance_pdf/
    python extract.py --schema drug_schema.yaml --input drug.md --force
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Literal, Optional

from kg_logging import NULL_LOGGER as _NULL_LOGGER
from kg_logging import get_run_logger as _get_run_logger

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, create_model, model_validator

load_dotenv(Path(__file__).parent / ".env")

from llm_client import LLM_TIMEOUT as _LLM_TIMEOUT, LLM_MAX_COMPLETION_TOKENS as _LLM_MAX_COMPLETION_TOKENS, _deployment as _llm_deployment, acreate_structured_output as _acreate_structured_output_shared  # noqa: E402
from lookups import sem_group_to_tuis  # noqa: E402
import grounding as _grounding  # noqa: E402

_LLM_MODEL = _llm_deployment()


async def _acreate_structured_output(
    text_input: str,
    system_prompt: str,
    response_model: type,
    _retries: int = 3,
    _base_delay: float = 1.0,
):
    return await _acreate_structured_output_shared(
        text_input, system_prompt, response_model,
        model=_LLM_MODEL,
        max_completion_tokens=_LLM_MAX_COMPLETION_TOKENS,
        timeout=_LLM_TIMEOUT,
        retries=_retries,
        base_delay=_base_delay,
        log_prefix="[llm]",
    )


_CHUNK_SIZE    = int(os.environ.get("LLM_CHUNK_SIZE",    "20000"))
_CHUNK_OVERLAP = int(os.environ.get("LLM_CHUNK_OVERLAP",  "2000"))

_CACHE_DIR = Path(os.environ.get("KG_CACHE_DIR") or Path(__file__).parent / ".cache")
_REVIEW_MAX_DOC_CHARS = 500_000  # omit doc_text from review JSON if larger; store doc_source instead

DEFAULT_FILTER = "moderate"

# Color sets kept at each grounding filter level (shared by extract + apply_graph).
COLOR_KEEP: dict[str, set[str]] = {
    "loose":    {"green", "yellow", "red"},
    "moderate": {"green", "yellow"},
    "strict":   {"green"},
}


def colors_for_filter(filter_level: str) -> set[str]:
    """Return the set of triple colors retained at the given filter level."""
    return COLOR_KEEP.get(filter_level, {"green", "yellow", "red"})


def worst_color(*colors: str) -> str:
    """Combine entity colors into a triple color: red > yellow > green."""
    if "red" in colors:
        return "red"
    if "yellow" in colors:
        return "yellow"
    return "green"


def best_color(*colors: str) -> str:
    """Pick the GREENEST color: green > yellow > red (inverse of worst_color).

    Used to apply a deterministic green ANCHOR — proven entity presence pins an
    entity's color to at least green, so the LLM cannot lower a provably-present
    entity. None values are ignored (no opinion)."""
    cs = [c for c in colors if c]
    if "green" in cs:
        return "green"
    if "yellow" in cs:
        return "yellow"
    return "red"

def _load_prompts() -> dict:
    _p = Path(__file__).parent / "prompts.yaml"
    try:
        return yaml.safe_load(_p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"prompts.yaml not found at {_p}. Restore it from the repository.")

_ALL_PROMPTS:    dict         = _load_prompts()
_FILTER_PROMPTS: dict[str, str] = _ALL_PROMPTS["filter_prompts"]
_SYSTEM_EXTRACT: str = (_ALL_PROMPTS.get("extract") or {}).get("open_minded_system_prompt", "")


def _char_chunk_text(text: str) -> list[str]:
    """Split text into overlapping character-based chunks of _CHUNK_SIZE chars."""
    if _CHUNK_OVERLAP >= _CHUNK_SIZE:
        raise ValueError(f"LLM_CHUNK_OVERLAP ({_CHUNK_OVERLAP}) must be < LLM_CHUNK_SIZE ({_CHUNK_SIZE})")
    if len(text) <= _CHUNK_SIZE:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + _CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += _CHUNK_SIZE - _CHUNK_OVERLAP
    return chunks


def _semantic_chunk_text(text: str, max_chars: int | None = None) -> list[str]:
    """Split text on markdown section headers, falling back to char chunking for oversized sections.

    Sections starting with # / ## / ### are kept together. Consecutive sections are
    accumulated until the combined size exceeds max_chars, then the buffer is flushed.
    A single section exceeding max_chars is sub-chunked via _char_chunk_text.
    """
    if max_chars is None:
        max_chars = _CHUNK_SIZE
    if not text:
        return [""]
    sections = re.split(r"(?m)(?=^#{1,3}\s)", text)
    sections = [s for s in sections if s] or [""]
    result: list[str] = []
    buffer: list[str] = []
    buffer_chars: int = 0
    for section in sections:
        slen = len(section)
        if slen > max_chars:
            if buffer:
                result.append("".join(buffer))
                buffer, buffer_chars = [], 0
            result.extend(_char_chunk_text(section))
        elif buffer_chars + slen > max_chars:
            result.append("".join(buffer))
            buffer, buffer_chars = [section], slen
        else:
            buffer.append(section)
            buffer_chars += slen
    if buffer:
        result.append("".join(buffer))
    return result or [""]


_chunk_text = _semantic_chunk_text


_ABBR_FORWARD = re.compile(
    r'([A-Z][A-Za-z0-9\-]*(?:\s+[A-Za-z][A-Za-z0-9\-]*){0,4})\s*\(([A-Z][A-Z0-9]{1,8})\)'
)
_ABBR_REVERSE = re.compile(
    r'\b([A-Z][A-Z0-9]{1,8})\s*\(([A-Za-z][A-Za-z0-9\-]*(?:\s+[A-Za-z][A-Za-z0-9\-]*){0,4})\)'
)


def _extract_abbreviations(text: str) -> dict[str, str]:
    """Extract ABBR→full-name mappings from standard scientific notation.

    Handles both 'Full Name (ABBR)' and 'ABBR (Full Name)' forms.
    Forward matches take precedence over reverse on collision.
    """
    mapping: dict[str, str] = {}
    for full, abbr in _ABBR_FORWARD.findall(text):
        mapping[abbr.strip()] = full.strip()
    for abbr, full in _ABBR_REVERSE.findall(text):
        if abbr.strip() not in mapping:
            mapping[abbr.strip()] = full.strip()
    return mapping


def _cache_key(doc_text: str, schema_yaml: str, instructions: str = "", abbr_str: str = "", harvest_sig: str = "", model: str = "") -> str:
    # "|open" suffix invalidates old "|dual" cache entries from the two-pass era.
    # harvest_sig folds in the harvest store so a new rejection/example invalidates
    # cached chunks (the harvest few-shot + rejection reminders are part of the prompt).
    # `model` makes the cache model-AWARE: extraction output is model-dependent, so a
    # different extraction LLM must miss the cache and genuinely re-extract (otherwise
    # a run would silently reuse another model's triples while run_config claims the new
    # model — a correctness/provenance bug).
    payload = doc_text + schema_yaml + instructions + abbr_str + harvest_sig + "|m=" + model + "|open"
    return hashlib.sha256(payload.encode()).hexdigest()


def _harvest_signature(harvest_dir: "Path | None") -> str:
    """Content hash of the harvest store, for the chunk cache key.

    Any change to the injected few-shot positives or rejection reminders must
    invalidate cached chunks (otherwise a reject is served stale until --force).
    Hashes all *.jsonl contents; empty string when there is no store.
    """
    if not harvest_dir or not harvest_dir.exists():
        return ""
    h = hashlib.sha256()
    for p in sorted(harvest_dir.glob("*.jsonl")):
        try:
            h.update(p.name.encode())
            h.update(p.read_bytes())
        except Exception:
            pass
    return h.hexdigest()[:16]


def _cache_load(key: str, response_model: type):
    path = _CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        return response_model.model_validate_json(path.read_text())
    except Exception:
        return None


def _cache_save(key: str, result) -> None:
    _CACHE_DIR.mkdir(exist_ok=True)
    (_CACHE_DIR / f"{key}.json").write_text(result.model_dump_json(indent=2))




# Separators the LLM uses to stitch non-adjacent document fragments into one
# supporting_quote: " / " (merged quotes), and markdown list markers " * " / " - " /
# " • " (a section intro paired with a specific bullet item — e.g. "side effects
# include: * vomiting", where other bullets sit between them in the document).
# Splitting on these and re-anchoring each fragment lets such valid-but-non-
# contiguous quotes ground correctly.
_QUOTE_SEGMENT_SPLIT = re.compile(r"\s+/\s+|\s+[*•·]\s+|\s+-\s+")


def _verify_grounding(
    all_items: dict,
    rels: list[dict],
    doc_text: str,
    filter_level: str,
) -> tuple[int, int, list[dict]]:
    """Verify + re-anchor supporting_quote grounding. Modifies all_items in-place.

    For each kept item whose quote can be located in the document (token-based
    match, tolerant of markdown / quote-char swaps / citations / '...' elisions),
    ``supporting_quote`` is overwritten with the verbatim document substring it
    matched.  This guarantees downstream consumers never see LLM quote drift.

    A quote that cannot be located is treated as ungrounded: warned (moderate) or
    dropped (strict).  Returns (dropped, warned, warnings) where warnings is a list
    of structured warning dicts for moderate-level mismatches.
    """
    dropped = 0
    warned = 0
    warnings: list[dict] = []

    for rel in rels:
        kept = []
        for item in all_items.get(rel["rel_type"], []):
            quote = item.get("supporting_quote", "") or ""

            if filter_level == "loose":
                kept.append(item)
                continue

            if not quote.strip():
                if filter_level == "strict":
                    dropped += 1
                    print(f"    DROP (no quote): {rel['rel_type']} — {item.get(rel['from_field'], '?')}")
                    continue
                kept.append(item)
                continue

            anchored = _grounding.reanchor(quote, doc_text)
            if anchored is None:
                # Multi-part quote — the LLM stitched together separate document
                # fragments (merged quotes joined by " / ", or a list intro paired
                # with a non-adjacent bullet item). No single span covers it, but if
                # EVERY segment re-anchors we store the joined verbatim segments.
                # (Partial matches fall through to warn/drop.)
                _segs = [s.strip() for s in _QUOTE_SEGMENT_SPLIT.split(quote) if s.strip()]
                if len(_segs) > 1:
                    _re = [_grounding.reanchor(s, doc_text) for s in _segs]
                    if all(r is not None for r in _re):
                        anchored = " / ".join(_re)
            if anchored is not None:
                # Located: replace LLM quote with the verbatim document substring.
                item["supporting_quote"] = anchored
                kept.append(item)
                continue

            # Not locatable — genuine grounding failure.
            if filter_level == "moderate":
                warned += 1
                from_term = item.get(rel["from_field"], "?")
                to_field  = rel.get("to_field", "")
                to_term   = item.get(to_field, "?") if to_field else "?"
                print(f"    WARN (quote not in doc): {rel['rel_type']} — {from_term}")
                print(f"      quote: {quote[:120]}…")
                warnings.append({
                    "rel_type":     rel["rel_type"],
                    "from_term":    from_term,
                    "to_term":      to_term,
                    "message":      "supporting_quote not found verbatim in document",
                    "quote_prefix": quote[:120],
                })
                kept.append(item)

            elif filter_level == "strict":
                dropped += 1
                print(f"    DROP (quote not in doc): {rel['rel_type']} — {item.get(rel['from_field'], '?')}")
                continue

            else:
                kept.append(item)

        all_items[rel["rel_type"]] = kept

    return dropped, warned, warnings


_SCHEMA_TYPES: dict[str, type] = {
    "STRING": str,
    "INT64": int, "INT32": int, "INT16": int, "INT8": int,
    "UINT64": int, "UINT32": int, "UINT16": int, "UINT8": int,
    "DOUBLE": float, "FLOAT": float,
    "BOOLEAN": bool,
    "TIMESTAMP": str, "DATE": str, "INTERVAL": str,
    "BLOB": bytes,
}


def _schema_type_to_python(schema_type: str) -> type:
    base = schema_type[:-2].strip() if schema_type.endswith("[]") else schema_type
    t = _SCHEMA_TYPES.get(base)
    if t is None:
        raise ValueError(f"Unknown schema type: {base!r} (from {schema_type!r})")
    return list[t] if schema_type.endswith("[]") else t


# ── Schema Pydantic models ────────────────────────────────────────────────────

class _PropertyDef(BaseModel):
    model_config = {"extra": "forbid"}
    name: str
    type: str
    source: Literal["umls", "gleif", "llm", "pipeline"]
    primary_key: bool = False
    optional: bool = False
    pipeline_field: str | None = None
    hint: str = ""

    @model_validator(mode="after")
    def _pipeline_field_required(self) -> "_PropertyDef":
        if self.source == "pipeline" and not self.pipeline_field:
            raise ValueError("pipeline_field is required when source=pipeline")
        return self


class _NodeDef(BaseModel):
    model_config = {"extra": "forbid"}
    description: str = ""
    sem_group: str = ""
    semantic_types: list[str] = []
    umls_vocabs: list[str] = []
    properties: list[_PropertyDef]

    @model_validator(mode="after")
    def _one_primary_key(self) -> "_NodeDef":
        pks = [p for p in self.properties if p.primary_key]
        if len(pks) != 1:
            raise ValueError(f"exactly one primary_key required, found {len(pks)}")
        if pks[0].type.endswith("[]"):
            raise ValueError(f"primary_key must be a scalar type, not array: {pks[0].type!r}")
        return self


class _RelDef(BaseModel):
    model_config = {"extra": "forbid"}
    rel_type: str
    from_node: str
    from_field: str
    from_hint: str = ""
    to_node: str
    to_field: str
    to_hint: str = ""
    extract_prompt: str = ""
    examples: list[dict] = []
    properties: list[_PropertyDef] = []


class _SchemaDef(BaseModel):
    model_config = {"extra": "forbid"}
    nodes: dict[str, _NodeDef]
    relationships: list[_RelDef]

    @model_validator(mode="after")
    def _check_node_refs(self) -> "_SchemaDef":
        known = set(self.nodes)
        for rel in self.relationships:
            for side, node in (("from_node", rel.from_node), ("to_node", rel.to_node)):
                if node not in known:
                    raise ValueError(
                        f"relationship {rel.rel_type!r}: {side}={node!r} not defined in nodes"
                    )
        return self


# ── Schema loading ────────────────────────────────────────────────────────────

def load_schema(path: str = "schema.yaml") -> tuple[dict, list[dict]]:
    """Return (nodes_dict, rels_list) parsed from schema.yaml."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Schema error: {path} must be a YAML mapping, got {type(data).__name__}")
    try:
        _SchemaDef.model_validate(data)
    except ValidationError as e:
        raise SystemExit(f"Schema validation error in {path}:\n{e}")
    nodes = data["nodes"]
    for node_name, node_def in nodes.items():
        sg = node_def.get("sem_group", "")
        if sg and not sem_group_to_tuis(sg):
            print(
                f"[schema warn] {node_name}.sem_group: '{sg}' not found in "
                f"SemGroups.txt — check spelling",
                flush=True,
            )
    return nodes, data["relationships"]


def llm_props(node_def: dict) -> list[dict]:
    """Return LLM-sourced properties of a node definition."""
    return [p for p in node_def.get("properties", []) if p.get("source") == "llm"]


def llm_rel_props(rel: dict) -> list[dict]:
    """Return LLM-sourced properties defined on a relationship."""
    return [p for p in rel.get("properties", []) if p.get("source") == "llm"]


def pipeline_rel_props(rel: dict) -> list[dict]:
    """Return pipeline-sourced properties defined on a relationship."""
    return [p for p in rel.get("properties", []) if p.get("source") == "pipeline"]


def primary_key(node_def: dict) -> str:
    for p in node_def.get("properties", []):
        if p.get("primary_key"):
            return p["name"]
    raise ValueError(f"Node has no primary_key property: {node_def}")


# ── Dynamic Pydantic model ────────────────────────────────────────────────────

def build_extraction_model(rels: list[dict], nodes: dict) -> type[BaseModel]:
    """
    Build an Extraction Pydantic model from the schema.

    Each relationship produces one inner model with:
      - from_field / to_field: search terms for source and target nodes
      - any llm-sourced properties defined on either node
      - any llm-sourced properties defined on the relationship itself
    """
    top_fields: dict = {}
    for rel in rels:
        fields: dict = {
            rel["from_field"]: (str, ...),
            rel["to_field"]:   (str, ...),
        }
        # The primary key of an llm-only node is carried by from_field /
        # to_field, so it must NOT become a separate model field: one shared,
        # unexplained `name` column for both sides is what the model fills with
        # the relationship type.
        for side in ("from_node", "to_node"):
            node_def = nodes[rel[side]]
            node_pk = primary_key(node_def)
            for prop in llm_props(node_def):
                if prop["name"] == node_pk or prop["name"] in fields:
                    continue
                py_type = _schema_type_to_python(prop.get("type", "STRING"))
                ann = Optional[py_type] if prop.get("optional") else py_type
                fields[prop["name"]] = (ann, None if prop.get("optional") else ...)

        for prop in llm_rel_props(rel):
            if prop["name"] not in fields:
                py_type = _schema_type_to_python(prop.get("type", "STRING"))
                ann = Optional[py_type] if prop.get("optional") else py_type
                fields[prop["name"]] = (ann, None if prop.get("optional") else ...)

        # Grounding field: the LLM must cite a verbatim span from the document
        # that supports this relationship.  Triples whose quote cannot be found
        # in the source text are dropped as likely hallucinations.
        fields["supporting_quote"] = (str, ...)

        rel_model = create_model(f"{rel['rel_type']}_Relation", **fields)
        top_fields[rel["rel_type"]] = (list[rel_model], ...)

    return create_model("Extraction", **top_fields)


# ── External lookups ─────────────────────────────────────────────────────────


def _llm_only_props(
    node_def: dict,
    llm_extras: dict,
    term: str = "",
) -> Optional[tuple[dict, dict]]:
    """Build (props, {}) for a node whose properties are all llm-sourced.

    ``term`` is the entity name the LLM extracted for this side of the
    relationship (``from_field`` / ``to_field``). For a node with an external
    resolver the primary key comes from that resolver; for an llm-only node
    there is no other source, so the extracted term IS the primary key.

    Without this, the PK was read from ``llm_extras`` under its own property
    name — a field the extraction model exposes once for BOTH sides of the
    relationship, with no hint attached. The model has nothing sensible to put
    there and fills it with the relationship type, so every node ends up named
    ``USES`` or ``REQUIRES`` and the semantic check rejects the triple.
    """
    pk = primary_key(node_def)
    props: dict = {}
    for p in node_def.get("properties", []):
        if p.get("source") == "llm":
            val = term if (p["name"] == pk and term) else llm_extras.get(p["name"])
            if val is None and not p.get("optional"):
                return None
            if val is not None:
                props[p["name"]] = val
    # Ensure the primary key is present regardless of optional flag
    for p in node_def.get("properties", []):
        if p.get("primary_key") and p["name"] not in props:
            return None
    return (props, {})


_CORP_SUFFIXES = frozenset({
    "inc", "incorporated", "corp", "corporation", "ltd", "limited",
    "llc", "plc", "co", "company", "group", "holdings", "ag", "sa",
    "gmbh", "nv", "bv", "pty", "pte",
})


def _normalize_entity(entity: str) -> str:
    """Strip common corporate/legal suffixes and normalize to lowercase."""
    words = re.split(r'[\s,\.]+', entity.lower().strip())
    return " ".join(w for w in words if w and w not in _CORP_SUFFIXES)


def _entity_in_text(entity: str, text: str, fuzzy_threshold: int = 80, _text_lower: str | None = None) -> bool:
    """Multi-strategy entity mention detection. Lowercases both inputs internally."""
    entity_l = unicodedata.normalize("NFC", entity).lower().strip()
    text_l   = _text_lower if _text_lower is not None else unicodedata.normalize("NFC", text).lower()
    if not entity_l:
        return False

    # 1. Exact substring
    if entity_l in text_l:
        return True

    # 2. Suffix-stripped normalization
    norm = _normalize_entity(entity)
    if len(norm) >= 3 and norm in text_l:
        return True

    # 3. Word-boundary match on each hyphen/space/comma-split token (≥3 chars)
    for token in re.split(r'[\s,\.\-]+', entity):
        tok = token.rstrip('.').lower()
        if len(tok) >= 2 and re.search(r'\b' + re.escape(tok) + r'\b', text_l):
            return True

    return False


def _classify_entity(entity: str, doc_text: str, quote: str, _doc_lower: str | None = None) -> str:
    """Return 'green', 'yellow', or 'red'.

    green  = entity found in both document text and supporting quote
    yellow = entity found in document text only (quote absent or doesn't mention it)
    red    = entity not found in document text at all

    _doc_lower: pre-lowercased doc_text for performance (computed once per document
                in hot loops; derived from doc_text when omitted).
    """
    in_text  = _entity_in_text(entity, doc_text, _text_lower=_doc_lower)
    in_quote = _entity_in_text(entity, quote) if quote.strip() else False
    if in_text and in_quote:
        return "green"
    if in_text:
        return "yellow"
    return "red"


def _rescue_to_segment(to_term: str, doc_text: str, existing: list[str]) -> str | None:
    """Rescue a supporting-quote segment that actually names the `to` entity.

    The LLM sometimes emits a quote that does not mention the target (a table
    caption, a sibling list term, a bare header). The semantic check applies
    best-of-segment, so handing it one verbatim doc line that DOES name `to`
    lets the grounding + relation-support checks find real support instead of
    reddening a true triple. Returns None when `to` is already named in an
    existing segment (nothing to rescue) or cannot be located.

    ponytail: line-bounded, first match only. One good segment satisfies
    best-of; add multi-match / sentence-split only if a real doc needs it.
    """
    if not to_term:
        return None
    for seg in existing:
        if _grounding.locate(to_term, seg) is not None:
            return None  # target already present in an emitted segment
    span = _grounding.locate(to_term, doc_text)
    if span is None:
        return None
    start = doc_text.rfind("\n", 0, span[0]) + 1
    end   = doc_text.find("\n", span[1])
    if end == -1:
        end = len(doc_text)
    line = doc_text[start:end].strip()
    return line or None


def _build_evidence(quotes: list[str], doc_text: str) -> list[dict]:
    """Turn collected verbatim quote texts into evidence spans with char offsets.

    Offsets come from grounding.locate (deterministic), so the LLM is never asked
    for positions. `text` is the verbatim doc slice when located, else the original
    quote with null offsets. This is the single source of truth for grounding and
    provenance; `supporting_quote` is derived from it for display/back-compat.
    """
    ev: list[dict] = []
    seen: set[str] = set()
    for q in quotes:
        q = (q or "").strip()
        if not q or q in seen:
            continue
        seen.add(q)
        span = _grounding.locate(q, doc_text)
        if span:
            ev.append({"start": span[0], "end": span[1], "text": doc_text[span[0]:span[1]]})
        else:
            ev.append({"start": None, "end": None, "text": q})
    return ev


# ── PDF extraction with markdown cache ───────────────────────────────────────

def _extract_text(doc_path: Path) -> tuple[str, str]:
    """Load a document as markdown text.

    Accepts .md files directly. For .pdf input, a .md sidecar must already
    exist next to the PDF (produced by convert_pdf.py or pipeline.py).
    Raises SystemExit if a .pdf is given with no sidecar.
    """
    if doc_path.suffix.lower() == ".md":
        text = doc_path.read_text(encoding="utf-8")
        return text, f"{len(text):,} chars (pre-converted markdown)"

    sidecar = doc_path.with_suffix(".md")
    if sidecar.exists() and sidecar.stat().st_mtime >= doc_path.stat().st_mtime:
        text = sidecar.read_text(encoding="utf-8")
        return text, f"{len(text):,} chars (markdown sidecar)"

    raise SystemExit(
        f"No markdown sidecar found for '{doc_path.name}'.\n"
        f"  Convert first:  python convert_pdf.py --input \"{doc_path}\"\n"
        f"  Or use pipeline.py for the full convert → extract → apply flow."
    )


# ── Review file writing ───────────────────────────────────────────────────────

def _write_review_file(
    review_path: Path,
    doc_name: str,
    doc_text: str,
    dataset_name: str,
    schema_version: str,
    triples: list[dict],
    grounding_warnings: list[dict] | None = None,
    doc_source: Path | None = None,
    schema_path: str = "",
    filter_level: str = DEFAULT_FILTER,
    failed_chunks: list[int] | None = None,
) -> None:
    data: dict = {
        "doc": doc_name,
        "dataset_name": dataset_name,
        "schema_version": schema_version,
        "triples": triples,
    }
    if schema_path:
        data["schema_path"] = schema_path
    if filter_level:
        data["filter_level"] = filter_level
    if doc_source is not None:
        data["doc_source"] = str(doc_source.resolve())
    if len(doc_text) > _REVIEW_MAX_DOC_CHARS:
        print(
            f"  [info] doc_text ({len(doc_text):,} chars) exceeds {_REVIEW_MAX_DOC_CHARS:,} limit — "
            "omitting from review JSON; vector ingestion will read from doc_source.",
            flush=True,
        )
    else:
        data["doc_text"] = doc_text
    if grounding_warnings:
        data["grounding_warnings"] = grounding_warnings
    if failed_chunks:
        data["failed_chunks"] = failed_chunks
    review_path.write_text(json.dumps(data, indent=2, default=str))


# ── Core pipeline ─────────────────────────────────────────────────────────────

async def process_document(
    doc_path: Path,
    nodes: dict,
    rels: list,
    ExtractionModel: type,
    skip_log: list | None = None,
    schema_yaml: str = "",
    schema_path: str = "",
    instructions: str = "",
    filter_level: str = DEFAULT_FILTER,
    force_cache: bool = False,
    output_dir: Path | None = None,
    run_logger=None,
    chunk_retries: int = 1,
) -> Path | None:
    """Extract triples from one document and write <stem>_review.json. Returns the review path."""
    _log = run_logger if run_logger is not None else _NULL_LOGGER
    print(f"\n{'='*60}")
    print(f"Processing: {doc_path.name}")
    _log.info("doc_start", extra={"doc": doc_path.name, "stage": "start"})

    pipeline_context = {"doc_path": str(doc_path.resolve())}

    doc_text, extract_info = _extract_text(doc_path)
    print(f"  {extract_info}")
    if not doc_text.strip():
        print(f"  [skip] '{doc_path.name}' is empty — no LLM call made.", flush=True)
        return None

    abbr_map = _extract_abbreviations(doc_text)
    if abbr_map:
        preview = ", ".join(list(abbr_map.keys())[:6])
        ellipsis = "…" if len(abbr_map) > 6 else ""
        print(f"  Abbreviation map: {len(abbr_map)} entries ({preview}{ellipsis})")

    numbered_items = []
    for i, rel in enumerate(rels):
        from_node_def = nodes[rel["from_node"]]
        to_node_def   = nodes[rel["to_node"]]
        # Same reason as in build_extraction_model: the PK is the from/to term.
        _pks          = {primary_key(from_node_def), primary_key(to_node_def)}
        all_llm       = [p for p in llm_props(from_node_def) + llm_props(to_node_def)
                         if p["name"] not in _pks] + llm_rel_props(rel)
        extra_hints   = []
        for prop in all_llm:
            hint = prop.get("hint", prop["name"])
            opt  = " (optional)" if prop.get("optional") else ""
            extra_hints.append(f"'{prop['name']}': {hint}{opt}")
        extras_str = f" Also extract: {', '.join(extra_hints)}." if extra_hints else ""
        from_hint = rel.get("from_hint", "")
        to_hint   = rel.get("to_hint", "")
        from_label = f"{rel['from_field']} [{from_hint}]" if from_hint else rel["from_field"]
        to_label   = f"{rel['to_field']} [{to_hint}]"   if to_hint   else rel["to_field"]
        examples = rel.get("examples", [])
        if examples:
            ex_strs = [
                f"({ex.get(rel['from_field'], '?')} → {ex.get(rel['to_field'], '?')})"
                for ex in examples
            ]
            examples_str = " Examples: " + "; ".join(ex_strs) + "."
        else:
            examples_str = ""
        # Self-referential rel (from_node == to_node): the (from, to) tuple order
        # is the only structural direction signal, so spell out that it is ordered.
        self_loop = (
            f" (ORDERED pair: both sides are {rel['from_node']}; "
            f"do NOT swap {rel['from_field']} and {rel['to_field']}.)"
            if rel["from_node"] == rel["to_node"] else ""
        )
        numbered_items.append(
            f"{i+1}. ({from_label}, {to_label}) — "
            f"{rel['extract_prompt'].strip()}{extras_str}{examples_str}{self_loop}"
        )

    # Collect external-source property names so the LLM knows not to fill them in
    ext_prop_names = sorted({
        p["name"]
        for node_def in nodes.values()
        for p in node_def.get("properties", [])
        if p.get("source") not in ("llm", None)
    })
    ext_sources = sorted({
        p["source"].upper()
        for node_def in nodes.values()
        for p in node_def.get("properties", [])
        if p.get("source") not in ("llm", None)
    })
    lookup_note = (
        f"IMPORTANT: {', '.join(ext_prop_names)} for every node "
        f"are NOT extracted by you — they are resolved automatically from external "
        f"sources ({', '.join(ext_sources)}) using the field values you provide. "
        "Only supply the field values listed below."
    )

    node_desc_parts = []
    for label, node_def in nodes.items():
        desc = node_def.get("description")
        if desc:
            node_desc_parts.append(f"  {label}: {desc.strip()}")
    node_desc_block = (
        "\nNode type definitions:\n" + "\n".join(node_desc_parts) + "\n"
        if node_desc_parts else ""
    )

    instructions_block = f"\n\nExtraction scope:\n{instructions}" if instructions else ""
    # Build a shared base (schema metadata + node descriptions + scope).
    # The dual-agent personalities from prompts.yaml are appended after this block.
    _base_system = (
        f"You are an information extractor. {lookup_note}"
        f"{node_desc_block}{instructions_block}"
    )
    _system_prompt = _base_system + "\n\n" + _SYSTEM_EXTRACT

    # Inject harvest examples (quote → entities) for each rel_type
    _rejection_blocks: list[str] = []
    _harvest_dir: "Path | None" = None
    try:
        _harvest_dir = Path(schema_path).parent / "harvest" if schema_path else None
        if _harvest_dir and _harvest_dir.exists():
            from agents.harvest import load_examples as _load_harvest_examples, format_examples_block as _fmt_harvest_block, format_rejection_reminder as _fmt_rejection_reminder
            _harvest_blocks = []
            _injected_rels = []
            for _rel in rels:
                _exs = _load_harvest_examples(_harvest_dir, _rel["rel_type"])
                if _exs:
                    _harvest_blocks.append(_fmt_harvest_block(_exs, _rel["rel_type"]))
                    _injected_rels.append(_rel["rel_type"])
                    _rej = _fmt_rejection_reminder(_exs, _rel["rel_type"])
                    if _rej:
                        _rejection_blocks.append(_rej)
            if _harvest_blocks:
                _system_prompt += "\n\n" + "\n\n".join(_harvest_blocks)
                print(f"  [harvest] injected examples for: {_injected_rels}", flush=True)
    except Exception as _he:
        print(f"  [harvest] warning: {_he}", flush=True)

    _harvest_sig = _harvest_signature(_harvest_dir)
    # Extraction model folds into the chunk cache key (model-aware caching).
    _extract_model = os.environ.get("LLM_MODEL", "").replace("azure/", "")

    chunks = _semantic_chunk_text(doc_text)
    if len(chunks) > 1:
        print(f"  Splitting into {len(chunks)} chunk(s) (section-aware).")

    abbr_str = (
        "".join(f"{k}={v};" for k, v in sorted(abbr_map.items()))
        if abbr_map else ""
    )
    abbr_note = (
        "\n\nAbbreviations detected in this document (regex-extracted; values may include "
        "surrounding context words — use as a hint, not a guaranteed exact expansion):\n"
        + "\n".join(f"  {k}: {v}" for k, v in sorted(abbr_map.items()))
        + "\n"
    ) if abbr_map else ""

    # all_items: dict[rel_type → list[dict]] (model_dump output from open-minded extraction)
    all_items: dict[str, list[dict]] = {r["rel_type"]: [] for r in rels}

    async def _run_chunk(i: int):
        """Extract one chunk. Returns the parsed model, or None on LLM failure.

        Cache hits short-circuit the LLM call. asyncio.TimeoutError propagates so
        the doc-level retry can handle it; all other LLM errors return None so the
        caller can record the chunk for a selective retry pass.
        """
        chunk  = chunks[i]
        prefix = f"  [{i+1}/{len(chunks)}]" if len(chunks) > 1 else "  "
        chunk_key = _cache_key(chunk, schema_yaml, instructions, abbr_str, _harvest_sig, _extract_model) if schema_yaml else None

        if not force_cache and chunk_key:
            cached = _cache_load(chunk_key, ExtractionModel)
            if cached is not None:
                print(f"{prefix} Cache hit ({chunk_key[:12]}…) — skipping LLM.")
                return cached

        _rej_prefix = ("\n\n".join(_rejection_blocks) + "\n\n") if _rejection_blocks else ""
        text_input_str = (
            _rej_prefix
            + "Extract from this document text:\n"
            + "\n".join(numbered_items)
            + abbr_note
            + f"\n\nDocument text:\n{chunk}"
        )
        print(f"{prefix} Calling LLM (open-minded extraction)…", flush=True)
        try:
            result = await _acreate_structured_output(text_input_str, _system_prompt, ExtractionModel)
        except asyncio.TimeoutError:
            raise  # propagate so doc-level retry logic handles it
        except Exception as _chunk_exc:
            print(f"{prefix} [error] LLM call failed: {_chunk_exc}", flush=True)
            return None
        # Cache ONLY a non-empty extraction. An ExtractionModel with all-empty
        # rel-type lists is still truthy, so caching on `if result` alone would
        # freeze a transient/degraded empty response into the cache — every later
        # non-`--force` run would then serve 0 triples for this chunk forever.
        # Not caching lets the next run re-attempt the chunk (one LLM call); a
        # genuinely-empty chunk just costs that one call, a transient empty self-heals.
        if result and chunk_key:
            if any(getattr(result, r["rel_type"], None) for r in rels):
                _cache_save(chunk_key, result)
                print(f"{prefix} Cached ({chunk_key[:12]}…).")
            else:
                print(f"{prefix} Empty extraction — not cached (transient-safe).")
        return result

    results_by_idx: dict[int, object] = {}
    _failed_chunks: list[int] = []  # 1-based indices of chunks where the LLM call failed
    for i in range(len(chunks)):
        res = await _run_chunk(i)
        if res is None:
            _failed_chunks.append(i + 1)
        else:
            results_by_idx[i] = res

    # ── Selective retry: re-run only the failed chunks, leaving cached/successful
    #    chunks untouched (no wasted LLM calls on chunks that already succeeded). ──
    for _attempt in range(max(0, chunk_retries)):
        if not _failed_chunks:
            break
        retry_idx = _failed_chunks
        _failed_chunks = []
        print(
            f"  [chunk-retry {_attempt+1}/{chunk_retries}] re-running "
            f"{len(retry_idx)} failed chunk(s): {retry_idx}",
            flush=True,
        )
        for one_based in retry_idx:
            res = await _run_chunk(one_based - 1)
            if res is None:
                _failed_chunks.append(one_based)
            else:
                results_by_idx[one_based - 1] = res

    for _i in sorted(results_by_idx):
        for r in rels:
            all_items[r["rel_type"]].extend(
                item.model_dump() for item in (getattr(results_by_idx[_i], r["rel_type"]) or [])
            )

    counts = {r["rel_type"]: len(all_items[r["rel_type"]]) for r in rels}
    chunk_note = f" across {len(chunks)} chunks" if len(chunks) > 1 else ""
    print(f"  Extracted{chunk_note}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    # A whole-document zero is almost always a transient/degraded LLM response, not
    # a real "nothing to extract" — surface it loudly. The empty result is no longer
    # cached (see _run_chunk), so simply re-running the document should recover it.
    if sum(counts.values()) == 0 and not _failed_chunks:
        print(
            f"  [warn] {doc_path.name}: LLM returned ZERO extractions for the whole "
            "document. This is usually a transient LLM issue — re-run to recover "
            "(the empty result was NOT cached).",
            flush=True,
        )
    if _failed_chunks:
        print(f"  [warn] {len(_failed_chunks)}/{len(chunks)} chunk(s) failed — extraction may be partial: {_failed_chunks}", flush=True)
    _log.info("extraction_done", extra={
        "doc": doc_path.name, "stage": "extract",
        "counts": counts, "chunks_total": len(chunks),
        "chunks_failed": _failed_chunks,
    })

    # ── Quote grounding verification ──────────────────────────────────────────
    total_before = sum(len(v) for v in all_items.values())
    dropped, warned, grounding_warnings = _verify_grounding(all_items, rels, doc_text, filter_level)
    if dropped or warned:
        print(f"  Grounding: dropped {dropped}/{total_before} (no quote), {warned} warnings (quote mismatch).")
    _log.info("grounding_done", extra={
        "doc": doc_path.name, "stage": "grounding",
        "dropped": dropped, "total": total_before,
    })

    # ── Node agent: batch external-source resolution (UMLS, GLEIF, …) ──────────
    from agents.node_agent import resolve_all_nodes as _resolve_all_nodes, resolver_for_node as _resolver_for_node  # noqa: PLC0415
    from agents.base_resolver import ResolveContext as _ResolveContext  # noqa: PLC0415
    _node_err_dir = output_dir or doc_path.parent
    # abbr_map (doc-derived ABBR → full name) lets the GLEIF resolver expand an
    # abbreviation the document already defined, skipping the LLM expansion call.
    _resolve_ctx = _ResolveContext(domain_hint=instructions, abbr_map=abbr_map)
    resolved_map = await _resolve_all_nodes(all_items, rels, nodes, _node_err_dir, _resolve_ctx)
    if resolved_map:
        found = sum(1 for v in resolved_map.values() if v is not None)
        print(f"  Node agent: {found}/{len(resolved_map)} resolved.", flush=True)
        _log.info("node_resolve_done", extra={
            "doc": doc_path.name, "stage": "node_resolve",
            "found": found, "total": len(resolved_map),
        })

    src_label = "/".join(ext_sources) if ext_sources else "external sources"
    print(f"  Building triples ({src_label} via node agent)…")

    resolved_triples: list[dict] = []

    for rel in rels:
        from_node_def = nodes[rel["from_node"]]
        to_node_def   = nodes[rel["to_node"]]
        from_pk       = primary_key(from_node_def)
        to_pk         = primary_key(to_node_def)
        rp_defs          = llm_rel_props(rel)
        pipeline_rp_defs = pipeline_rel_props(rel)
        from_resolver = _resolver_for_node(from_node_def)
        to_resolver   = _resolver_for_node(to_node_def)

        _edges: dict[tuple, dict] = {}

        for item in all_items[rel["rel_type"]]:
            from_term   = item.get(rel["from_field"]) or ""
            to_term     = item.get(rel["to_field"])   or ""
            from_llm_extras = {p["name"]: item.get(p["name"]) for p in llm_props(from_node_def)}
            to_llm_extras   = {p["name"]: item.get(p["name"]) for p in llm_props(to_node_def)}

            # Resolve each side via the appropriate resolver, or fall back to llm-only.
            if from_resolver is not None:
                from_result = from_resolver.build_props(
                    from_term, rel["from_node"], from_node_def, from_llm_extras, resolved_map
                )
            else:
                from_result = _llm_only_props(from_node_def, from_llm_extras, from_term)

            if to_resolver is not None:
                to_result = to_resolver.build_props(
                    to_term, rel["to_node"], to_node_def, to_llm_extras, resolved_map
                )
            else:
                to_result = _llm_only_props(to_node_def, to_llm_extras, to_term)

            if from_result is None:
                print(f"    SKIP (lookup failed): {rel['from_node']} '{from_term}'")
                if skip_log is not None:
                    skip_log.append({"doc": doc_path.name, "rel": rel["rel_type"], "node": rel["from_node"], "term": from_term})
                continue
            if to_result is None:
                print(f"    SKIP (lookup failed): {rel['to_node']} '{to_term}'")
                if skip_log is not None:
                    skip_log.append({"doc": doc_path.name, "rel": rel["rel_type"], "node": rel["to_node"], "term": to_term})
                continue

            from_props, from_meta = from_result
            to_props,   to_meta   = to_result

            key = (from_props[from_pk], to_props[to_pk])
            if key not in _edges:
                _edges[key] = {
                    "from_props":       from_props,
                    "from_meta":        from_meta,
                    "to_props":         to_props,
                    "to_meta":          to_meta,
                    "from_term":        from_term,
                    "to_term":          to_term,
                    "rp":               {p["name"]: [] for p in rp_defs if p.get("type", "").endswith("[]")},
                    "supporting_quotes": [],
                }
            # Color is NOT computed here: the semantic-check agent is the sole
            # color authority (it overwrites any extraction color and always runs
            # before the color filter). Extraction only gathers evidence.

            sq = (item.get("supporting_quote", "") or "").strip()
            if sq:
                quotes_list = _edges[key]["supporting_quotes"]
                if sq not in quotes_list:
                    quotes_list.append(sq)

            for p in rp_defs:
                val = item.get(p["name"])
                if val is None:
                    continue
                if p.get("type", "").endswith("[]"):
                    lst = _edges[key]["rp"].setdefault(p["name"], [])
                    if isinstance(val, list):
                        lst.extend(v for v in val if v is not None)
                    else:
                        lst.append(val)
                else:
                    _edges[key]["rp"][p["name"]] = val

        for key, data in _edges.items():
            from_props = data["from_props"]
            from_meta  = data.get("from_meta", {})
            to_props   = data["to_props"]
            to_meta    = data.get("to_meta", {})
            # Rescue a segment naming the `to` entity when no emitted quote does,
            # so the semantic check's best-of-segment has real support to find.
            _rescued = _rescue_to_segment(
                data.get("to_term") or "", doc_text, data["supporting_quotes"]
            )
            if _rescued and _rescued not in data["supporting_quotes"]:
                data["supporting_quotes"].append(_rescued)
            rp = {k: v for k, v in data["rp"].items() if v is not None and v != []}
            if "evidence_level" in rp:
                _sq = (" / ".join(data["supporting_quotes"])).strip()
                if rp["evidence_level"] == "strong" and len(_sq) < 10:
                    rp["evidence_level"] = "moderate"
                elif rp["evidence_level"] == "moderate" and not _sq:
                    rp["evidence_level"] = "weak"
            for p in pipeline_rp_defs:
                pf = p.get("pipeline_field")
                if pf and pf in pipeline_context:
                    rp[p["name"]] = pipeline_context[pf]
            evidence = _build_evidence(data["supporting_quotes"], doc_text)
            resolved_triples.append({
                "rel_type":        rel["rel_type"],
                "from_label":      rel["from_node"],
                "from_pk":         from_pk,
                "from_props":      from_props,
                "from_meta":       from_meta,
                "from_term":       data.get("from_term"),
                "to_label":        rel["to_node"],
                "to_pk":           to_pk,
                "to_props":        to_props,
                "to_meta":         to_meta,
                "to_term":         data.get("to_term"),
                "rel_props":       rp or None,
                "evidence":        evidence,
                "supporting_quote": " / ".join(e["text"] for e in evidence) or None,
            })

    # ── Assign stable content-based IDs before semantic check ───────────────────
    # ID = sha256(rel_type NUL from_pk_value NUL to_pk_value)[:12] so the same
    # triple gets the same ID on re-extraction, keeping review events valid.
    for _t in resolved_triples:
        _fv = _t["from_props"].get(_t["from_pk"], "")
        _tv = _t["to_props"].get(_t["to_pk"], "")
        _key = f"{_t['rel_type']}\x00{_fv}\x00{_tv}"
        _t["_id"] = "f" + hashlib.sha256(_key.encode()).hexdigest()[:12]

    # ── Semantic check — always runs (no manual button needed) ────────────────
    if resolved_triples:
        from agents.semantic_check_agent import check_triples as _semantic_check  # noqa: PLC0415
        print(f"  Semantic check: {len(resolved_triples)} triple(s)…", flush=True)
        resolved_triples = await asyncio.to_thread(
            _semantic_check,
            resolved_triples,
            doc_text,
            schema_rels=rels,
            schema_nodes=nodes,
            filter_level=filter_level,
            harvest_dir=_harvest_dir,
            doc_name=f"{doc_path.stem}_raw.json",
            instructions=instructions,
        )

    # ── Color filter (applied after semantic check may have re-colored) ───────
    _color_keep = colors_for_filter(filter_level)
    before_color_filter = len(resolved_triples)
    resolved_triples = [t for t in resolved_triples if t.get("triple_color", "green") in _color_keep]
    color_dropped = before_color_filter - len(resolved_triples)
    if color_dropped:
        print(f"  Color filter ({filter_level}): dropped {color_dropped}/{before_color_filter} triple(s).")
    _color_dist = {
        c: sum(1 for t in resolved_triples if t.get("triple_color", "green") == c)
        for c in ("green", "yellow", "red")
    }
    _log.info("semantic_check_done", extra={
        "doc": doc_path.name, "stage": "semantic_check",
        "triple_count": len(resolved_triples),
        "color_dist": _color_dist,
        "dropped": color_dropped,
        "total": before_color_filter,
    })

    for triple in resolved_triples:
        from_props = triple["from_props"]
        to_props   = triple["to_props"]
        from_pk    = triple["from_pk"]
        to_pk      = triple["to_pk"]
        rp         = triple.get("rel_props") or {}
        list_props = {k: v for k, v in rp.items() if isinstance(v, list)}
        scalar_tag = " ".join(f"{k}={v}" for k, v in list_props.items()) if list_props else ""
        print(
            f"    ✓ "
            f"({from_props.get('name', from_props[from_pk])} [{from_props[from_pk]}])"
            f" -[:{triple['rel_type']}]->"
            f" ({to_props.get('name', to_props[to_pk])} [{to_props[to_pk]}])"
            + (f"  {scalar_tag}" if scalar_tag else "")
        )

    _path_hash = hashlib.sha256(str(doc_path.resolve()).encode()).hexdigest()[:8]
    dataset_name = f"{doc_path.stem.replace(' ', '_')}_{_path_hash}"
    sv = hashlib.sha256(schema_yaml.encode()).hexdigest()[:12] if schema_yaml else ""
    raw_path = (output_dir / f"{doc_path.stem}_raw.json") if output_dir else doc_path.with_name(f"{doc_path.stem}_raw.json")
    md_source = doc_path if doc_path.suffix.lower() == ".md" else doc_path.with_suffix(".md")
    _write_review_file(raw_path, doc_path.name, doc_text, dataset_name, sv, resolved_triples,
                       grounding_warnings=grounding_warnings or None,
                       doc_source=md_source if md_source.exists() else None,
                       schema_path=schema_path,
                       filter_level=filter_level,
                       failed_chunks=_failed_chunks or None)
    print(f"\n  Raw file: {raw_path.resolve()}")
    print(f"✓ Done: {doc_path.name}  ({len(resolved_triples)} triples)", flush=True)
    _log.info("doc_done", extra={
        "doc": doc_path.name, "stage": "done",
        "triple_count": len(resolved_triples),
        "run_path": str(raw_path),
    })
    return raw_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract triples from documents and write <stem>_review.json files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--schema", required=True, metavar="PATH",
        help="Schema YAML file",
    )
    p.add_argument(
        "--input", default=None, metavar="PATH",
        help="PDF/Markdown file or directory to process (default: pdf/)",
    )
    p.add_argument(
        "--skip-report", action="store_true",
        help="Print a summary of all failed external lookups (GLEIF / UMLS) at the end.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Bypass extraction cache and re-run LLM even if a cached result exists.",
    )
    p.add_argument(
        "--concurrency", type=int, default=2, metavar="N",
        help="Max documents processed concurrently (default: 2).",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print cache hit messages from UMLS/GLEIF lookups.",
    )
    p.add_argument(
        "--meta", default=None, metavar="PATH",
        help="Pipeline metadata YAML (instructions + per-PDF page filters). "
             "Required when page filtering is desired; omit for no instructions.",
    )
    p.add_argument(
        "--filter", default=DEFAULT_FILTER,
        choices=["loose", "moderate", "strict"],
        help="Entity grounding filter level (default: moderate). "
             "loose=keep all | moderate=green+yellow | strict=green only.",
    )
    p.add_argument("--constraint", dest="filter", choices=["loose", "moderate", "strict"], help=argparse.SUPPRESS)  # backward-compat alias
    p.add_argument(
        "--output-dir", default=None, metavar="DIR",
        help="Directory to write *_review.json files (default: alongside input files)",
    )
    p.add_argument(
        "--retries", type=int, default=2, metavar="N",
        help="Max retry attempts per document on transient failure (default: 2).",
    )
    p.add_argument(
        "--chunk-retries", type=int, default=1, metavar="N",
        help="Max selective retry passes over individual chunks whose LLM call failed, "
             "before falling back to a partial result (default: 1). Cached/successful "
             "chunks are never re-run.",
    )
    p.add_argument(
        "--estimate-only", action="store_true",
        help="Print a pre-flight chunk/token/cost estimate for the input and exit "
             "without making any LLM calls.",
    )
    return p.parse_args()


def _preflight_estimate(
    doc_files: list[Path],
    output_dir: Path | None,
    force: bool,
    schema_yaml: str,
    instructions: str,
) -> dict:
    """Estimate chunk count, input tokens, and cost before any LLM call is made.

    Reads each markdown (or .md sidecar) and runs the real section-aware chunker so
    the chunk count matches what extraction will produce. Documents whose *_raw.json
    already exists (and ``force`` is False) are skipped, mirroring main()'s skip logic.
    Token/cost figures are rough: ~4 chars/token, input only (completion size unknown).
    """
    # Static per-chunk prompt overhead (system prompt + schema + instructions), in chars.
    overhead_chars = len(schema_yaml) + len(instructions) + 2000

    n_docs = n_skipped = n_chunks = total_chars = 0
    for doc_path in doc_files:
        raw_path = (
            (output_dir / f"{doc_path.stem}_raw.json") if output_dir
            else doc_path.with_name(f"{doc_path.stem}_raw.json")
        )
        if not force and raw_path.exists():
            n_skipped += 1
            continue
        try:
            doc_text, _ = _extract_text(doc_path)
        except SystemExit:
            # Missing .md sidecar for a PDF — can't estimate; skip silently.
            continue
        if not doc_text.strip():
            continue
        chunks = _semantic_chunk_text(doc_text)
        n_docs += 1
        n_chunks += len(chunks)
        total_chars += sum(len(c) for c in chunks)

    input_chars  = total_chars + n_chunks * overhead_chars
    input_tokens = input_chars // 4  # ~4 chars/token heuristic

    est: dict = {
        "docs_to_extract": n_docs,
        "docs_skipped":    n_skipped,
        "chunks":          n_chunks,
        "input_tokens":    input_tokens,
    }

    print(
        f"[pre-flight] {n_docs} doc(s) to extract "
        f"({n_skipped} skipped — *_raw.json exists), "
        f"{n_chunks} chunk(s), ~{input_tokens:,} input tokens.",
        flush=True,
    )
    if n_chunks:
        try:
            from llm_client import _get_litellm_model  # noqa: PLC0415
            import litellm  # noqa: PLC0415
            litellm.suppress_debug_info = True
            model = _get_litellm_model()
            prompt_cost, _ = litellm.cost_per_token(
                model=model, prompt_tokens=input_tokens, completion_tokens=0
            )
            est["model"] = model
            est["input_cost_usd"] = round(prompt_cost, 4)
            print(
                f"[pre-flight] Est. input cost (model {model}): "
                f"${prompt_cost:,.4f}  (output tokens not estimated; one LLM call per chunk)",
                flush=True,
            )
        except Exception as _ce:
            print(f"[pre-flight] cost estimate unavailable: {_ce}", flush=True)
    return est


def _run_llm_models() -> dict:
    """Which LLM served each task this run, read from the current process env.

    Extraction and node-resolution (the resolver LLM picks) both use ``LLM_MODEL``;
    the semantic check uses ``SEMANTIC_CHECK_MODEL`` when set, else ``LLM_MODEL``.
    Recorded into run_config.json so a run is reproducible/auditable.
    """
    ext = os.environ.get("LLM_MODEL", "").replace("azure/", "")
    sc = os.environ.get("SEMANTIC_CHECK_MODEL", "").strip()
    return {"extraction": ext, "node_resolution": ext, "semantic_check": sc or ext}


def _compute_extraction_stats(raw_paths: list[Path], output_dir: Path) -> dict:
    """Aggregate quality metrics from a completed extraction run."""
    total = 0
    colors: dict[str, int] = {"green": 0, "yellow": 0, "red": 0}
    resolved_entities: set[tuple] = set()

    for path in raw_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for t in data.get("triples", []):
            total += 1
            c = t.get("triple_color", "green")
            if c in colors:
                colors[c] += 1
            for side in ("from", "to"):
                pk  = t.get(f"{side}_pk", "")
                lbl = t.get(f"{side}_label", "")
                val = t.get(f"{side}_props", {}).get(pk, "")
                if val and pk in ("cui", "lei"):
                    resolved_entities.add((lbl, val))

    err_log = output_dir / "_node_agent_errors.jsonl"
    unresolved = 0
    if err_log.exists():
        for line in err_log.read_text(encoding="utf-8").splitlines():
            try:
                json.loads(line)
                unresolved += 1
            except Exception:
                pass

    entity_resolved  = len(resolved_entities)
    entity_attempted = entity_resolved + unresolved
    return {
        "docs_processed":        len(raw_paths),
        "total_triples":         total,
        "colors":                colors,
        "green_pct":             round(colors["green"] / total, 3) if total else 0.0,
        "entity_resolved":       entity_resolved,
        "unresolved_entities":   unresolved,
        "entity_resolution_rate": round(entity_resolved / entity_attempted, 3) if entity_attempted else 0.0,
    }


async def main(args=None) -> list[Path]:
    """Run extraction and return a list of written review file paths."""
    if args is None:
        args = parse_args()

    import lookups as _lookups_mod
    _lookups_mod.VERBOSE = getattr(args, "verbose", False)

    import pipeline_meta as _pm
    _meta = _pm.load_meta(getattr(args, "meta", None))
    _instructions = _pm.get_instructions(_meta)
    if _instructions:
        print(f"[meta] Extraction instructions loaded ({len(_instructions)} chars).")

    nodes, rels = load_schema(args.schema)
    schema_yaml = Path(args.schema).read_text()

    needs_umls = any(
        p.get("source") == "umls"
        for node_def in nodes.values()
        for p in node_def.get("properties", [])
    )
    if needs_umls and not os.getenv("UMLS_API_KEY"):
        raise SystemExit("UMLS_API_KEY not found in .env — required by this schema.")
    ExtractionModel = build_extraction_model(rels, nodes)

    input_path = Path(args.input or "pdf")
    if input_path.is_file():
        suffix = input_path.suffix.lower()
        if suffix not in (".pdf", ".md"):
            raise SystemExit(f"Input file must be a .pdf or .md: {input_path}")
        doc_files = [input_path]
    elif input_path.is_dir():
        pdf_files = sorted(input_path.glob("*.pdf"))
        if pdf_files:
            doc_files = pdf_files
        else:
            doc_files = sorted(input_path.glob("*.md"))
    else:
        raise SystemExit(f"Input path not found: {input_path.resolve()}")

    if not doc_files:
        raise SystemExit(f"No PDF or Markdown files found at '{input_path.resolve()}'")

    _output_dir_arg = getattr(args, "output_dir", None)
    output_dir: Optional[Path] = None
    if _output_dir_arg:
        output_dir = Path(_output_dir_arg)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-flight chunk/token/cost estimate (no LLM calls). --estimate-only bails here.
    _preflight_estimate(doc_files, output_dir, args.force, schema_yaml, _instructions)
    if getattr(args, "estimate_only", False):
        print("[pre-flight] --estimate-only set — exiting before extraction.", flush=True)
        return []

    skip_log: list = [] if args.skip_report else None
    sem = asyncio.Semaphore(args.concurrency)
    max_retries = getattr(args, "retries", 2)
    _run_logger = _get_run_logger(output_dir) if output_dir else _NULL_LOGGER

    _total = len(doc_files)
    _started = 0
    print(f"[extract:progress] 0/{_total}", flush=True)

    async def _process(doc_path: Path) -> tuple:
        nonlocal _started
        raw_path = (
            (output_dir / f"{doc_path.stem}_raw.json") if output_dir
            else doc_path.with_name(f"{doc_path.stem}_raw.json")
        )
        if not args.force and raw_path.exists():
            _started += 1
            print(f"[extract:progress] {_started}/{_total}", flush=True)
            print(f"\nSKIP {doc_path.name}  (raw file exists; use --force to re-extract)")
            return raw_path, None
        async with sem:
            # Increment when doc starts processing so progress reflects docs
            # in-flight or done (not just fully completed ones).  This ensures
            # the "Grounding:" / "Verifying" phase shows a non-zero counter.
            _started += 1
            print(f"[extract:progress] {_started}/{_total}", flush=True)
            last_exc: Exception | None = None
            for attempt in range(1 + max_retries):
                try:
                    result = await process_document(
                        doc_path, nodes, rels, ExtractionModel,
                        skip_log,
                        schema_yaml=schema_yaml,
                        force_cache=args.force,
                        instructions=_instructions,
                        filter_level=getattr(args, "filter", DEFAULT_FILTER),
                        schema_path=str(Path(args.schema).resolve()),
                        output_dir=output_dir,
                        run_logger=_run_logger,
                        chunk_retries=getattr(args, "chunk_retries", 1),
                    )
                    return result, None
                except (SystemExit, KeyboardInterrupt):
                    raise
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        delay = 2 ** (attempt + 1)
                        print(
                            f"  [retry {attempt+1}/{max_retries}] {doc_path.name}: {exc} "
                            f"— retrying in {delay}s…",
                            flush=True,
                        )
                        await asyncio.sleep(delay)
            return None, (doc_path, last_exc)

    results = await asyncio.gather(*[_process(p) for p in doc_files])
    review_paths: list[Path] = [p for p, _ in results if p is not None]
    failed = [err for _, err in results if err is not None]

    # Write extraction quality stats into run_config.json
    if output_dir:
        _run_cfg = output_dir / "run_config.json"
        if _run_cfg.exists():
            try:
                _cfg = json.loads(_run_cfg.read_text(encoding="utf-8"))
                _all_raw = review_paths  # only files written in this run
                # Record which LLM actually served each task (from this run's env,
                # so it reflects the models used, not the scaffold's stale guess).
                _models = _run_llm_models()
                _cfg["llm_model"] = _models["extraction"]  # correct stale scaffold value
                _cfg["llm_models"] = _models
                _cfg["extraction_stats"] = _compute_extraction_stats(_all_raw, output_dir)
                _run_cfg.write_text(json.dumps(_cfg, indent=2))
            except Exception:
                pass

    if failed:
        print(f"\nFailed documents ({len(failed)}):")
        for doc_path, exc in failed:
            print(f"  {doc_path.name}: {exc}")

    if skip_log:
        print(f"\nDropped/skipped triples ({len(skip_log)} total):")
        print(f"  {'Document':<45} {'Rel':<20} {'Node':<15} {'Term':<30} Reason")
        print(f"  {'-'*45} {'-'*20} {'-'*15} {'-'*30} {'-'*22}")
        for s in skip_log:
            reason = s.get("reason", "lookup failed")
            print(f"  {s['doc']:<45} {s['rel']:<20} {s['node']:<15} {s['term']:<30} {reason}")
    elif args.skip_report:
        print("\nFailed lookups: none")

    return review_paths


if __name__ == "__main__":
    asyncio.run(main())
