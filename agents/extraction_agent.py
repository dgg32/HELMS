#!/usr/bin/env python3
"""
ExtractionAgent: intelligent triple extraction with adaptive entity resolution.

GLEIF and UMLS lookups are exposed as tools the LLM calls directly, enabling
iterative resolution with abbreviation expansion, fuzzy search, and query
reformulation. The agent self-selects reasoning strategies per document.

Unlike extract.py (hardcoded exact→fuzzy fallback, single-pass), this agent:
  - Retries failed lookups with alternate queries / search types
  - Expands abbreviations (e.g. "NVDA" → "NVIDIA Corporation")
  - Self-selects a reflection pass to catch missed pairs
  - Reports resolution failures with context rather than silently skipping

Usage:
    python agents/extraction_agent.py --schema schemas/drug_schema.yaml --input drug.md
    python agents/extraction_agent.py --schema supplychain_schema.yaml --input report.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from llm_client import _get_litellm_model, _litellm_kwargs, _ensure_litellm_env  # noqa: E402
from extract import _build_evidence, _entity_in_text, _ALL_PROMPTS as _PROMPTS  # noqa: E402
import grounding as _grounding  # noqa: E402
# Pluggable naming-service tools (GLEIF, UMLS, …). Add a service by writing
# agents/<svc>_tools.py and appending to resolver_tools.RESOLVER_TOOLSETS — no
# edits here. See agents/resolver_tools.py. Absolute import (not `from .`) because
# this module is also run directly as a script (python agents/extraction_agent.py).
from agents.resolver_tools import RESOLVER_TOOL_SPECS, RESOLVER_HANDLERS  # noqa: E402


_DOC_MAX_CHARS = 80_000

# ── Session state ──────────────────────────────────────────────────────────────

class ExtractionSession:
    def __init__(self):
        self.doc_text: str        = ""
        self.doc_path: str        = ""
        self.schema_nodes: dict   = {}
        self.schema_rels: list    = []
        self.triples: list[dict]  = []
        self.saved: bool          = False
        self.schema_version: str   = ""
        self.schema_path: str      = ""
        self.instructions: str     = ""  # loaded from pipeline_meta on load_document_and_schema
        self.filter_level: str = "moderate"
        self.umls_resolve:    str = "top"
        self.gleif_resolve:   str = "top"
        self.output_dir:      str = ""  # if set, write review JSON here instead of alongside input
        self._ambiguous_items: list[dict] = []
        self._triple_key_to_idx: dict[tuple, int] = {}  # (from_pk_val, to_pk_val, rel_type) → index
        self._rel_type_filter: str = ""   # if set, restrict extraction to this rel_type only
        self._output_filename: str = ""   # if set, override output filename in save_review()

    # ── Tool implementations ───────────────────────────────────────────────────

    def load_document_and_schema(self, doc_path: str, schema_path: str) -> str:
        """Load document text and schema, returning a structured description."""
        path = Path(doc_path)
        if path.suffix.lower() == ".md":
            if not path.exists():
                return json.dumps({"error": f"File not found: {doc_path}"})
            self.doc_text = path.read_text(encoding="utf-8")
        elif path.suffix.lower() == ".pdf":
            sidecar = path.with_suffix(".md")
            if not sidecar.exists():
                return json.dumps({"error": f"No .md sidecar for {path.name}. Run convert_pdf.py first."})
            self.doc_text = sidecar.read_text(encoding="utf-8")
        else:
            return json.dumps({"error": f"Unsupported file type: {path.suffix}"})
        if not self.doc_text:
            return json.dumps({"error": f"Document is empty: {doc_path}"})
        if len(self.doc_text) > _DOC_MAX_CHARS:
            print(f"  [warn] document truncated to {_DOC_MAX_CHARS:,} chars (was {len(self.doc_text):,})")
            self.doc_text = self.doc_text[:_DOC_MAX_CHARS]
        self.doc_path = str(path.resolve())

        try:
            schema_yaml = Path(schema_path).read_text(encoding="utf-8")
            schema_data = yaml.safe_load(schema_yaml)
            self.schema_nodes   = schema_data["nodes"]
            self.schema_rels    = schema_data["relationships"]
            if self._rel_type_filter:
                self.schema_rels = [r for r in self.schema_rels if r["rel_type"] == self._rel_type_filter]
                if not self.schema_rels:
                    return json.dumps({"error": f"rel_type {self._rel_type_filter!r} not found in schema"})
            self.schema_version = hashlib.sha256(schema_yaml.encode()).hexdigest()[:12]
        except Exception as e:
            return json.dumps({"error": f"Schema load failed: {e}"})

        self.schema_path = str(Path(schema_path).resolve())

        # Build description the agent uses to understand what to extract
        node_descs = {}
        for label, node_def in self.schema_nodes.items():
            sources = {p.get("source") for p in node_def.get("properties", [])}
            resolver = "gleif" if "gleif" in sources else ("umls" if "umls" in sources else "llm")
            pk_prop = next((p for p in node_def.get("properties", []) if p.get("primary_key")), None)
            pk_name = pk_prop["name"] if pk_prop else "unknown"
            llm_node_props = [p for p in node_def.get("properties", []) if p.get("source") == "llm"]
            node_descs[label] = {
                "description":    node_def.get("description", ""),
                "resolver":       resolver,
                "pk":             pk_name,
                "sem_group":      node_def.get("sem_group", ""),
                "semantic_types": node_def.get("semantic_types") or [],
                "umls_vocabs":    node_def.get("umls_vocabs") or [],
                "llm_props": [
                    {"name": p["name"], "hint": p.get("hint", p["name"]), "optional": p.get("optional", False)}
                    for p in llm_node_props
                ],
            }

        rel_descs = []
        for rel in self.schema_rels:
            fn, tn = rel["from_node"], rel["to_node"]
            if fn not in node_descs:
                return json.dumps({"error": f"Relationship '{rel['rel_type']}' references unknown node '{fn}'"})
            if tn not in node_descs:
                return json.dumps({"error": f"Relationship '{rel['rel_type']}' references unknown node '{tn}'"})
            from_node_desc = node_descs[fn]
            to_node_desc   = node_descs[tn]
            llm_rel_props  = [p for p in rel.get("properties", []) if p.get("source") == "llm"]
            examples = rel.get("examples", [])
            rel_descs.append({
                "rel_type":    rel["rel_type"],
                "extract_prompt": rel.get("extract_prompt", "").strip(),
                "from_node":   rel["from_node"],
                "from_field":  rel["from_field"],
                "from_hint":   rel.get("from_hint", rel["from_node"]),
                "from_pk":     from_node_desc["pk"],
                "from_resolver": from_node_desc["resolver"],
                "from_umls_vocabs": from_node_desc["umls_vocabs"],
                "from_sem_group":   from_node_desc["sem_group"],
                "from_semantic_types": from_node_desc["semantic_types"],
                "from_llm_props": from_node_desc["llm_props"],
                "to_node":     rel["to_node"],
                "to_field":    rel["to_field"],
                "to_hint":     rel.get("to_hint", rel["to_node"]),
                "to_pk":       to_node_desc["pk"],
                "to_resolver": to_node_desc["resolver"],
                "to_umls_vocabs": to_node_desc["umls_vocabs"],
                "to_sem_group":   to_node_desc["sem_group"],
                "to_semantic_types": to_node_desc["semantic_types"],
                "to_llm_props": to_node_desc["llm_props"],
                "llm_rel_props": [
                    {"name": p["name"], "hint": p.get("hint", p["name"]), "optional": p.get("optional", False)}
                    for p in llm_rel_props
                ],
                "examples": examples[:2] if examples else [],
            })

        result = {
            "doc":   path.name,
            "chars": len(self.doc_text),
            "nodes": node_descs,
            "relationships": rel_descs,
        }
        if self.instructions:
            result["instructions_loaded"] = True
        return json.dumps(result, indent=2)

    def search_document(self, query: str, max_results: int = 5) -> str:
        """Return verbatim document lines matching `query`.

        The agent works from a long tool loop where the document scrolls out of
        context, so it tends to reconstruct quotes from memory (hallucination).
        This lets it fetch the EXACT text to copy into add_triple's
        supporting_quote, the way the batch path copies from the doc-in-prompt.
        Normalised substring match over lines, with a grounding.locate fallback
        (enclosing line) for markdown-laden / multi-token queries.
        """
        q = (query or "").strip()
        if not q:
            return json.dumps({"results": [], "message": "Empty query."})
        if not self.doc_text:
            return json.dumps({"results": [], "message": "No document loaded."})
        nq = " ".join(q.lower().split())
        hits: list[str] = []
        seen: set[str] = set()
        for line in self.doc_text.splitlines():
            s = line.strip()
            if not s or s in seen:
                continue
            if nq in " ".join(s.lower().split()):
                hits.append(s)
                seen.add(s)
                if len(hits) >= max_results:
                    break
        if not hits:
            span = _grounding.locate(q, self.doc_text)
            if span:
                start = self.doc_text.rfind("\n", 0, span[0]) + 1
                end   = self.doc_text.find("\n", span[1])
                if end == -1:
                    end = len(self.doc_text)
                line = self.doc_text[start:end].strip()
                if line:
                    hits.append(line)
        msg = (
            "Copy ONE of these lines verbatim into add_triple's supporting_quote."
            if hits else
            f"No document line matched {q!r}. Try a shorter/different phrase; do NOT invent a quote."
        )
        return json.dumps({"results": hits, "count": len(hits), "message": msg})

    # GLEIF/UMLS lookup tools moved to agents/gleif_tools.py + agents/umls_tools.py
    # (registered via agents/resolver_tools.py). They are dispatched through
    # RESOLVER_HANDLERS in _dispatch, not as ExtractionSession methods.

    def add_triple(
        self,
        rel_type: str,
        from_label: str,
        from_pk: str,
        from_props_json: str,
        to_label: str,
        to_pk: str,
        to_props_json: str,
        rel_props_json: str = "{}",
        from_meta_json: str = "{}",
        to_meta_json: str = "{}",
        from_raw_term: str = "",
        to_raw_term: str = "",
        supporting_quote: str = "",
    ) -> str:
        """Add a resolved triple to the accumulation list.

        from_props_json / to_props_json: JSON object with all node properties.
        rel_props_json: JSON object with edge-level LLM properties (may be "{}").
        from_raw_term / to_raw_term: original LLM extraction terms (e.g. "Beyfortus"),
            used for color grounding instead of resolved canonical names so that quotes
            written from document context (which use the raw form) produce green correctly.
        """
        rel_def = next((r for r in self.schema_rels if r["rel_type"] == rel_type), None)
        if rel_def is None:
            return json.dumps({"error": f"Unknown rel_type: {rel_type!r}"})
        if from_label not in self.schema_nodes:
            return json.dumps({"error": f"Unknown from_label: {from_label!r}"})
        if to_label not in self.schema_nodes:
            return json.dumps({"error": f"Unknown to_label: {to_label!r}"})

        try:
            from_props = json.loads(from_props_json)
            to_props   = json.loads(to_props_json)
            rel_props  = json.loads(rel_props_json) if rel_props_json else {}
            from_meta  = json.loads(from_meta_json) if from_meta_json else {}
            to_meta    = json.loads(to_meta_json)   if to_meta_json   else {}
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"JSON parse error in props: {e}"})
        # Normalise semantic types key — LLM may pass either "semantic_types" or "types";
        # umls_resolver and semantic_check_agent use "types" as the canonical key.
        for _meta in (from_meta, to_meta):
            if "semantic_types" in _meta and "types" not in _meta:
                _meta["types"] = _meta.pop("semantic_types")

        # Vocab check — agent retry only.  Verify the resolved CUI actually has atoms in the
        # schema's allowed vocabularies.  Empty SAB set (API error/timeout) → skip check.
        from lookups import umls_get_sabs as _umls_get_sabs  # noqa: PLC0415
        for _side_label, _side_props in (
            (from_label, from_props),
            (to_label,   to_props),
        ):
            _allowed_voc = set((self.schema_nodes.get(_side_label) or {}).get("umls_vocabs") or [])
            if not _allowed_voc:
                continue
            _cui = _side_props.get("cui", "")
            if not _cui:
                continue
            _actual_sabs = _umls_get_sabs(_cui)
            if _actual_sabs and not (_actual_sabs & _allowed_voc):
                return json.dumps({
                    "error": (
                        f"Vocab mismatch for {_side_label}: CUI {_cui!r} belongs to "
                        f"vocabs {sorted(_actual_sabs)}, none in schema's allowed "
                        f"{sorted(_allowed_voc)}. Re-search with "
                        f"sabs='{','.join(sorted(_allowed_voc))}' to find the correct entry."
                    )
                })

        if from_pk not in from_props:
            return json.dumps({"error": f"from_props missing primary key '{from_pk}'"})
        if to_pk not in to_props:
            return json.dumps({"error": f"to_props missing primary key '{to_pk}'"})

        # Top-level param wins; fall back to rel_props embed for backward compat.
        quote = supporting_quote or rel_props.pop("supporting_quote", "") or ""

        # Re-anchor the quote to verbatim document text (same as the batch path's
        # grounding step) so the stored quote and color grounding never drift.
        # Verbatim guard: if reanchor fails the quote is NOT in the document (the
        # agent works from a long tool loop and can reconstruct/fabricate quotes,
        # unlike the batch path which copies from doc-in-prompt). We keep the triple
        # (the relation may still be true) but tell the agent so it can re-quote;
        # the semantic check forces such a triple red for human verification.
        quote_unlocatable = False
        if quote:
            _anchored = _grounding.reanchor(quote, self.doc_text)
            if _anchored is not None:
                quote = _anchored
            else:
                quote_unlocatable = True

        # Hard reject on ZERO grounding. The agent can invent entities outright
        # (even GPT-5.5 did, with adverse effects absent from the label). When the
        # quote is not verbatim in the document AND the RAW extracted term is not in
        # the document either, nothing grounds the triple — refuse to store it so the
        # agent moves on, rather than leaving a hallucination (red triples are still
        # written under --filter loose). We check the RAW term (from_raw_term /
        # to_raw_term), NOT the resolved props name: node normalization rewrites the
        # term entirely ("rash" → "Skin rash", a CUI's canonical label), so checking
        # the normalized name would false-reject real entities. A locatable quote
        # counts as grounding, so synonym extractions with a real quote are spared.
        if quote_unlocatable:
            for _side, _rt in (("from", from_raw_term.strip()), ("to", to_raw_term.strip())):
                if _rt and not _entity_in_text(_rt, self.doc_text):
                    return json.dumps({"error": (
                        f"Rejected: the {_side} term {_rt!r} does not appear in the document, "
                        f"and the supporting_quote is not verbatim either. Do not invent "
                        f"entities — only add a triple when its entities are actually present "
                        f"in the document text. Re-read the document and pick a real pair, or skip."
                    )})

        # Deduplicate by (from_pk_val, to_pk_val, rel_type); merge rel_props on match — O(1) via index
        key = (from_props[from_pk], to_props[to_pk], rel_type)
        if key in self._triple_key_to_idx:
            existing = self.triples[self._triple_key_to_idx[key]]
            if quote:
                existing_q = existing.get("supporting_quote") or ""
                if not quote_unlocatable and existing.get("quote_unlocatable"):
                    # Replace-on-retry: the agent re-quoted with verbatim text after a
                    # fabrication warning. Drop the unlocatable quote entirely (do not
                    # append) so the stored quote ends clean and the red flag clears.
                    existing["supporting_quote"] = quote
                    existing.pop("quote_unlocatable", None)
                elif not quote_unlocatable:
                    # Both quotes verbatim: append (dedup). Normalise trailing
                    # punctuation/space before the in-check so "RSV" and "RSV." don't
                    # both get appended — but store the original (un-stripped) quote (B10).
                    _cmp_q = quote.rstrip(". ")
                    if _cmp_q and _cmp_q not in existing_q:
                        existing["supporting_quote"] = (existing_q + " / " + quote).strip(" /") if existing_q else quote
                elif not existing_q.strip():
                    # New quote is unlocatable and there is no existing quote: keep it
                    # (flagged) so the triple still carries something for review.
                    existing["supporting_quote"] = quote
                    existing["quote_unlocatable"] = True
                # else: new quote unlocatable but a quote already exists — skip it, do
                # not pollute a good (or first-seen) quote with a fabrication.
            if rel_props:
                ep = existing.get("rel_props") or {}
                for k, v in rel_props.items():
                    if isinstance(v, list) and isinstance(ep.get(k), list):
                        ep[k] = ep[k] + [x for x in v if x not in ep[k]]
                    else:
                        ep[k] = v
                existing["rel_props"] = ep
            _dup_warn = (
                "  ⚠ supporting_quote still NOT found verbatim — re-call with the EXACT "
                "document text or this triple stays flagged red." if quote_unlocatable else ""
            )
            return f"⚠ Duplicate merged: ({from_props.get('name', from_props[from_pk])}) -[:{rel_type}]-> ({to_props.get('name', to_props[to_pk])}){_dup_warn}"

        entry: dict = {
            "rel_type":   rel_type,
            "from_label": from_label,
            "from_pk":    from_pk,
            "from_props": from_props,
            "to_label":   to_label,
            "to_pk":      to_pk,
            "to_props":   to_props,
            "rel_props":  rel_props,
        }
        if from_meta:
            entry["from_meta"] = from_meta
        if to_meta:
            entry["to_meta"] = to_meta

        # Color is NOT computed here: the semantic-check agent is the sole color
        # authority and runs after the merge (_run_agent_semantic_check /
        # _cli_semantic_check), writing colors back to *_raw.json.
        if from_raw_term.strip():
            entry["from_term"] = from_raw_term.strip()
        if to_raw_term.strip():
            entry["to_term"] = to_raw_term.strip()
        if quote:
            entry["supporting_quote"] = quote
            if quote_unlocatable:
                entry["quote_unlocatable"] = True
        self._triple_key_to_idx[key] = len(self.triples)
        self.triples.append(entry)

        from_id = from_props.get(from_pk, "?")
        to_id   = to_props.get(to_pk, "?")
        idx     = len(self.triples) - 1
        _warn = (
            "  ⚠ supporting_quote was NOT found verbatim in the document — it will be "
            "flagged red for human review. Re-call add_triple with the EXACT text copied "
            "from the document (do not paraphrase or add facts)." if quote_unlocatable else ""
        )
        return (
            f"✓ #{idx} added: "
            f"({from_props.get('name', from_id)} [{from_id}])"
            f" -[:{rel_type}]->"
            f" ({to_props.get('name', to_id)} [{to_id}]){_warn}"
        )

    def get_status(self) -> str:
        by_rel: dict[str, int] = {}
        for t in self.triples:
            by_rel[t["rel_type"]] = by_rel.get(t["rel_type"], 0) + 1
        return json.dumps({"total_triples": len(self.triples), "by_rel_type": by_rel})

    def save_review(self) -> str:
        if not self.doc_path:
            return json.dumps({"error": "No document loaded."})
        _REVIEW_MAX_DOC_CHARS = 500_000
        doc_path = Path(self.doc_path)
        if self._output_filename:
            _base = Path(self.output_dir) if self.output_dir else doc_path.parent
            raw_path = _base / self._output_filename
        elif self.output_dir:
            raw_path = Path(self.output_dir) / f"{doc_path.stem}_raw.json"
        else:
            raw_path = doc_path.with_name(f"{doc_path.stem}_raw.json")
        _path_hash = hashlib.sha256(str(doc_path.resolve()).encode()).hexdigest()[:8]
        dataset_name = f"{doc_path.stem.replace(' ', '_')}_{_path_hash}"

        # Assign stable content-hash IDs matching extract.py convention so review events survive re-extraction
        for t in self.triples:
            _fv = t["from_props"].get(t["from_pk"], "")
            _tv = t["to_props"].get(t["to_pk"], "")
            t["_id"] = "f" + hashlib.sha256(f"{t['rel_type']}\x00{_fv}\x00{_tv}".encode()).hexdigest()[:12]
            # Evidence spans (offsets) are the source of truth for grounding, same
            # contract as the batch path; supporting_quote is the joined display text.
            _q = t.get("supporting_quote") or ""
            t["evidence"] = _build_evidence(_q.split(" / ") if _q else [], self.doc_text)

        # Save all triples unfiltered; apply_graph.py applies filter_level at write time
        all_triples = [
            {k: v for k, v in t.items() if not k.startswith("_") or k == "_id"}
            for t in self.triples
        ]
        data: dict = {
            "doc":            doc_path.name,
            "dataset_name":   dataset_name,
            "schema_version": self.schema_version,
            "schema_path":    self.schema_path,
            "filter_level":   self.filter_level,
            "triples":        all_triples,
        }
        if self._ambiguous_items:
            data["ambiguous_pending"] = self._ambiguous_items
        if len(self.doc_text) <= _REVIEW_MAX_DOC_CHARS:
            data["doc_text"] = self.doc_text
        else:
            md_source = doc_path if doc_path.suffix.lower() == ".md" else doc_path.with_suffix(".md")
            if md_source.exists():
                data["doc_source"] = str(md_source.resolve())
        raw_path.write_text(json.dumps(data, indent=2, default=str, ensure_ascii=False))
        self.saved = True
        return json.dumps({"saved": len(self.triples), "file": str(raw_path)})


# ── Merge helper ──────────────────────────────────────────────────────────────

def _do_merge(merge_raw_path: str, agent_path: "Path") -> None:
    """Upsert agent triples into existing *_raw.json by _id; delete agent temp file."""
    existing = json.loads(Path(merge_raw_path).read_text(encoding="utf-8"))
    agent_data = json.loads(agent_path.read_text(encoding="utf-8"))
    agent_triples = agent_data.get("triples", [])

    merged = list(existing.get("triples", []))
    id_to_idx = {t["_id"]: i for i, t in enumerate(merged) if "_id" in t}

    upserted = appended = 0
    for at in agent_triples:
        aid = at.get("_id")
        if aid and aid in id_to_idx:
            prev = merged[id_to_idx[aid]]
            # Preserve fields the agent triple never carries (from save_review output):
            # extraction_source, from_term/to_term (raw LLM terms), from_meta/to_meta (resolver metadata)
            # rel_props: merge prev first so batch-set fields (source, publication_date,
            # manually_added) survive; agent props override individual keys on top (R7/B3).
            _prev_rp = prev.get("rel_props") or {}
            _at_rp   = at.get("rel_props") or {}
            merged[id_to_idx[aid]] = {
                **at,
                "extraction_source":  "agent_retry",
                "supporting_quote":   at.get("supporting_quote") or prev.get("supporting_quote"),
                "from_term": at.get("from_term") or prev.get("from_term"),
                "to_term":   at.get("to_term")   or prev.get("to_term"),
                "from_meta": at.get("from_meta") or prev.get("from_meta"),
                "to_meta":   at.get("to_meta")   or prev.get("to_meta"),
                "rel_props": {**_prev_rp, **_at_rp},
            }
            upserted += 1
        else:
            at["extraction_source"] = "agent_retry"  # genuinely new triple found by agent
            merged.append(at)
            appended += 1

    Path(merge_raw_path).write_text(
        json.dumps({**existing, "triples": merged}, indent=2, default=str, ensure_ascii=False)
    )
    agent_path.unlink(missing_ok=True)
    print(
        f"  [agent-retry] {upserted} updated + {appended} new "
        f"→ {len(merged)} total in {Path(merge_raw_path).name}",
        flush=True,
    )


# ── Tool definitions ───────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "load_document_and_schema",
            "description": (
                "Load a document and schema file. Call this FIRST before any other tool. "
                "Returns document length, node types with resolver/pk info, and relationships to extract."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_path":    {"type": "string", "description": "Path to .md or .pdf file"},
                    "schema_path": {"type": "string", "description": "Path to schema YAML file"},
                },
                "required": ["doc_path", "schema_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_document",
            "description": (
                "Search the loaded document for verbatim text. Returns up to 5 lines "
                "that contain your query, copied exactly from the document. ALWAYS call "
                "this to obtain the exact supporting_quote text BEFORE add_triple — copy "
                "a returned line verbatim. Never write a supporting_quote from memory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A short phrase or entity term you expect verbatim in the document "
                            "(e.g. an adverse-effect name, or part of the sentence that states "
                            "the relationship)."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    # GLEIF + UMLS (and any future naming service) tool specs come from the
    # pluggable registry. Append a ResolverToolset in agents/resolver_tools.py.
    *RESOLVER_TOOL_SPECS,
    {
        "type": "function",
        "function": {
            "name": "add_triple",
            "description": (
                "Add a resolved triple to the extraction output. "
                "Only call after both entities are successfully resolved. "
                "Duplicates are automatically skipped."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rel_type":       {"type": "string", "description": "Relationship type from schema"},
                    "from_label":     {"type": "string", "description": "Node label of the FROM entity"},
                    "from_pk":        {"type": "string", "description": "Primary key field name of FROM node (e.g. 'cui', 'lei')"},
                    "from_props_json": {"type": "string", "description": 'JSON object with all FROM node properties, e.g. {"cui":"C1234","name":"aspirin"}'},
                    "to_label":       {"type": "string", "description": "Node label of the TO entity"},
                    "to_pk":          {"type": "string", "description": "Primary key field name of TO node"},
                    "to_props_json":  {"type": "string", "description": "JSON object with all TO node properties"},
                    "rel_props_json": {"type": "string", "description": 'JSON object with edge-level LLM properties (excluding supporting_quote — use the dedicated field instead). Use "{}" if none.', "default": "{}"},
                    "from_meta_json": {"type": "string", "description": 'JSON object with resolver metadata for FROM node, e.g. {"types":["Pharmacologic Substance"],"root_source":"RXNORM"}. Copy types and root_source from the chosen umls_search result. Use "{}" if none.', "default": "{}"},
                    "to_meta_json":   {"type": "string", "description": 'JSON object with resolver metadata for TO node, e.g. {"types":["Molecular Function"],"root_source":"MED-RT"}. Copy types and root_source from the chosen umls_search result. Use "{}" if none.', "default": "{}"},
                    "from_raw_term":  {"type": "string", "description": "Original verbatim term for the FROM entity as it appears in the document (e.g. 'Beyfortus'). Used for color grounding.", "default": ""},
                    "to_raw_term":    {"type": "string", "description": "Original verbatim term for the TO entity as it appears in the document. Used for color grounding.", "default": ""},
                    "supporting_quote": {"type": "string", "description": "Verbatim sentence or phrase from the document that most directly supports this triple. Required — always extract a quote.", "default": ""},
                },
                "required": ["rel_type", "from_label", "from_pk", "from_props_json", "to_label", "to_pk", "to_props_json"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_status",
            "description": "Return current triple count by relationship type.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_review",
            "description": "Write all accumulated triples to <doc_stem>_raw.json. Call when extraction is complete.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


SYSTEM_PROMPT = _PROMPTS["extraction_agent"]["system_prompt"]


# ── Dispatch ───────────────────────────────────────────────────────────────────

def _dispatch(session: ExtractionSession, name: str, args: dict) -> str:
    if name == "load_document_and_schema":
        return session.load_document_and_schema(
            args.get("doc_path", ""), args.get("schema_path", "")
        )
    elif name == "search_document":
        return session.search_document(args.get("query", ""))
    elif name in RESOLVER_HANDLERS:
        # Naming-service lookup tools (gleif_find/gleif_search/umls_find/umls_search/…)
        # dispatched through the pluggable registry.
        return RESOLVER_HANDLERS[name](session, args)
    elif name == "add_triple":
        return session.add_triple(
            args.get("rel_type", ""),
            args.get("from_label", ""),
            args.get("from_pk", ""),
            args.get("from_props_json", "{}"),
            args.get("to_label", ""),
            args.get("to_pk", ""),
            args.get("to_props_json", "{}"),
            args.get("rel_props_json", "{}"),
            args.get("from_meta_json", "{}"),
            args.get("to_meta_json", "{}"),
            from_raw_term=args.get("from_raw_term", ""),
            to_raw_term=args.get("to_raw_term", ""),
            supporting_quote=args.get("supporting_quote", ""),
        )
    elif name == "get_status":
        return session.get_status()
    elif name == "save_review":
        return session.save_review()
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


# ── Main extraction loop ───────────────────────────────────────────────────────

def run_extraction(
    doc_path: str,
    schema_path: str,
    meta_path: str = "",
    filter_level: str = "moderate",
    umls_resolve: str = "top",
    gleif_resolve: str = "top",
    output_dir: str = "",
    rel_type_filter: str = "",
    merge_raw_path: str = "",
) -> str:
    import litellm as _litellm
    _litellm.suppress_debug_info = True
    # Silence litellm's background async LoggingWorker — its task is orphaned when
    # this run's event loop closes, printing a harmless but noisy "Event loop is
    # closed" traceback at teardown. See llm_client.quiet_litellm_logging_worker.
    try:
        from llm_client import quiet_litellm_logging_worker as _quiet_llm_log
        _quiet_llm_log()
    except Exception:
        pass
    session = ExtractionSession()

    try:
        import pipeline_meta as _pm
        instructions = _pm.get_instructions(_pm.load_meta(meta_path or None))
    except Exception:
        instructions = ""

    session.instructions = instructions
    session.filter_level = filter_level
    session.umls_resolve = umls_resolve
    session.gleif_resolve = gleif_resolve
    session.output_dir = output_dir
    session._rel_type_filter = rel_type_filter
    if rel_type_filter and merge_raw_path:
        session._output_filename = f"{Path(doc_path).stem}_agent_retry.json"
    active_system_prompt = SYSTEM_PROMPT
    if instructions:
        active_system_prompt += f"\n\nExtraction scope:\n{instructions}"

    # Inject harvest examples for the targeted rel_type(s)
    _rejection_blocks: list[str] = []
    try:
        import yaml as _yaml
        _schema_data = _yaml.safe_load(Path(schema_path).read_text(encoding="utf-8"))
        _schema_rels = _schema_data.get("relationships", [])
        if rel_type_filter:
            _schema_rels = [r for r in _schema_rels if r["rel_type"] == rel_type_filter]
        _harvest_dir = Path(schema_path).parent / "harvest"
        if _harvest_dir.exists() and _schema_rels:
            from agents.harvest import load_examples as _load_examples, format_examples_block as _fmt_examples, format_rejection_reminder as _fmt_rejection_reminder
            _harvest_blocks = []
            _injected_rels = []
            for _rel in _schema_rels:
                _exs = _load_examples(_harvest_dir, _rel["rel_type"])
                if _exs:
                    _harvest_blocks.append(_fmt_examples(_exs, _rel["rel_type"]))
                    _injected_rels.append(_rel["rel_type"])
                    _rej = _fmt_rejection_reminder(_exs, _rel["rel_type"])
                    if _rej:
                        _rejection_blocks.append(_rej)
            if _harvest_blocks:
                active_system_prompt += "\n\n" + "\n\n".join(_harvest_blocks)
                print(f"  [harvest] injected examples for: {_injected_rels}", flush=True)
    except Exception as _he:
        print(f"  [harvest] warning: {_he}", flush=True)

    _retry_context = ""
    if rel_type_filter and merge_raw_path and Path(merge_raw_path).exists():
        try:
            _existing_data = json.loads(Path(merge_raw_path).read_text(encoding="utf-8"))
            _prev = [t for t in _existing_data.get("triples", []) if t.get("rel_type") == rel_type_filter]
            if _prev:
                _prev_lines = []
                for t in _prev:
                    _ft = t.get("from_term") or t.get("from_props", {}).get(t.get("from_pk", ""), "?")
                    _tt = t.get("to_term") or t.get("to_props", {}).get(t.get("to_pk", ""), "?")
                    _color = t.get("triple_color", "?")
                    _prev_lines.append(f"  • {_ft} → {_tt}  [{_color}]")
                _retry_context = (
                    f"\n\nCONTEXT — USER-INITIATED RETRY: The user clicked 'Agent retry' for "
                    f"[{rel_type_filter}] because they believe the document contains more or better "
                    f"triples of this type than what was previously extracted.\n"
                    f"Previous extraction found {len(_prev)} triple(s):\n"
                    + "\n".join(_prev_lines)
                    + "\n\nPlease:\n"
                    "1. Read other parts of the document you may have overlooked — the correct information "
                    "may appear in a different section than where you first looked.\n"
                    "2. Try alternate search terms: synonyms, INN names, full expanded names for "
                    "abbreviations, and different UMLS search types (normalizedString, normalizedWords, "
                    "rightTruncation).\n"
                    "3. Do not simply re-add the same triples — find what is missing or incorrect."
                )
            else:
                _retry_context = (
                    f"\n\nCONTEXT — USER-INITIATED RETRY: The user clicked 'Agent retry' for "
                    f"[{rel_type_filter}] because they believe the document contains triples of this "
                    f"type that were completely missed in previous extraction.\n"
                    "No triples of this type were found before.\n\n"
                    "Please search the entire document carefully. Try multiple search strategies, "
                    "synonyms, abbreviation expansions, and different UMLS search types."
                )
        except Exception:
            pass

    _rej_prefix = ("\n\n".join(_rejection_blocks) + "\n\n") if _rejection_blocks else ""
    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                _rej_prefix
                + f"Extract all triples from this document using the given schema.\n"
                f"Document: {doc_path}\n"
                f"Schema:   {schema_path}"
                + (f"\nMeta:     {meta_path}" if meta_path else "")
                + _retry_context
            ),
        }
    ]

    print(f"\nExtractionAgent — {Path(doc_path).name}")
    print(f"Schema: {schema_path}\n")
    _t0 = time.monotonic()
    _ensure_litellm_env()
    _full_model = _get_litellm_model()
    _provider_kwargs = _litellm_kwargs()
    # Read live from the module so pipeline_runner env-override patches take effect (R3).
    import llm_client as _llm_client_mod
    _live_max_tokens = _llm_client_mod.LLM_MAX_COMPLETION_TOKENS
    _live_timeout    = _llm_client_mod.LLM_TIMEOUT

    for iteration in range(60):  # guard against runaway loops
        _elapsed = time.monotonic() - _t0
        print(f"── iter {iteration + 1}/60  ({_elapsed:.0f}s)  triples={len(session.triples)} ──", flush=True)
        tool_choice = "none" if session.saved else "auto"

        try:
            response = _litellm.completion(
                model=_full_model,
                messages=[{"role": "system", "content": active_system_prompt}] + messages,
                tools=TOOLS,
                tool_choice=tool_choice,
                max_tokens=_live_max_tokens,
                timeout=_live_timeout,
                **_provider_kwargs,
            )
        except Exception as api_exc:
            print(f"\n[error] LLM API call failed: {api_exc}", flush=True)
            if session.triples and not session.saved:
                print(f"  Auto-saving {len(session.triples)} partial triple(s)...", flush=True)
                session.save_review()
            raise
        msg = response.choices[0].message

        assistant_entry: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_entry)

        if msg.tool_calls:
            tool_results = []
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError as e:
                    print(f"  [warn] malformed tool args for {name}: {e}")
                    tool_results.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      json.dumps({"error": f"Malformed tool arguments: {e}"}),
                    })
                    continue
                arg_preview = ", ".join(
                    f"{k}={str(v)[:60]!r}" for k, v in args.items()
                    if k not in ("from_props_json", "to_props_json", "rel_props_json")
                )
                print(f"  [tool] {name}({arg_preview})")
                try:
                    result = _dispatch(session, name, args)
                except Exception as tool_exc:
                    # A lookup tool raised (vs. returning error JSON). Feed the error
                    # back as the tool result so the LLM can retry, instead of crashing
                    # the whole run and losing accumulated triples.
                    print(f"  [warn] tool {name} raised: {tool_exc}", flush=True)
                    result = json.dumps({"error": f"Tool {name} failed: {tool_exc}"})
                print(f"         → {result[:120]}")
                tool_results.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result,
                })
            messages.extend(tool_results)

        else:
            if msg.content:
                print(f"\nAgent: {msg.content}\n")
            if session.saved:
                break

    if not session.saved:
        if session.triples:
            print(f"  [warn] 60-iteration limit — auto-saving {len(session.triples)} triple(s)...", flush=True)
            session.save_review()
        else:
            print(f"  [warn] extraction ended without triples: {doc_path}", flush=True)

    if session.saved and rel_type_filter and merge_raw_path and Path(merge_raw_path).exists():
        if not session._output_filename:
            print(f"  [warn] _output_filename not set — skipping merge for {doc_path}", flush=True)
            return session.doc_path if session.saved else ""
        _base = Path(output_dir) if output_dir else Path(session.doc_path).parent
        _do_merge(merge_raw_path, _base / session._output_filename)
        return merge_raw_path

    return session.doc_path if session.saved else ""


# ── CLI helpers ────────────────────────────────────────────────────────────────

def _cli_semantic_check(raw_path: str, schema_path: str, filter_level: str) -> None:
    """Run semantic check on the extracted raw JSON (CLI equivalent of HTMX _run_agent_semantic_check).

    Loads the raw JSON, runs check_triples (LLM grounding + structural + harvest rejection),
    and writes the annotated triples back.  Failures are logged as warnings — the raw JSON
    is left intact so the user can still review the triples without ai_opinion.
    """
    import json as _json
    import yaml as _yaml
    from agents import semantic_check_agent as _sc

    _raw = Path(raw_path)
    if not _raw.exists():
        return

    try:
        data = _json.loads(_raw.read_text(encoding="utf-8"))
    except Exception as _e:
        print(f"  [semantic-check] could not read {_raw.name}: {_e}", flush=True)
        return

    triples = data.get("triples", [])
    if not triples:
        return

    try:
        schema_data = _yaml.safe_load(Path(schema_path).read_text(encoding="utf-8"))
    except Exception as _e:
        print(f"  [semantic-check] could not read schema: {_e}", flush=True)
        return

    schema_nodes = schema_data.get("nodes", {})
    schema_rels  = schema_data.get("relationships", [])
    doc_text     = data.get("doc_text", "")
    harvest_dir: Path | None = Path(schema_path).parent / "harvest"
    if not harvest_dir.exists():
        harvest_dir = None
    # harvest.py records doc_name as the *_raw.json filename (`raw_path.name`), not the
    # source .md name stored in data["doc"] — must match that convention or the harvest
    # rejection check's doc_name filter never matches (see htmx_app's _run_agent_semantic_check,
    # which already uses raw_p.name).
    doc_name = _raw.name
    # Extraction instructions (meta.yaml) let the grader judge document-subject intent.
    _instructions = ""
    try:
        import pipeline_meta as _pm  # noqa: PLC0415
        _meta_p = Path(schema_path).parent / "meta.yaml"
        if _meta_p.exists():
            _instructions = _pm.get_instructions(_pm.load_meta(str(_meta_p)))
    except Exception:
        pass

    try:
        _sc.check_triples(
            triples,
            doc_text=doc_text,
            schema_rels=schema_rels,
            schema_nodes=schema_nodes,
            filter_level=filter_level,
            harvest_dir=harvest_dir,
            doc_name=doc_name,
            instructions=_instructions,
        )
        data["triples"] = triples
        _raw.write_text(_json.dumps(data, indent=2, default=str, ensure_ascii=False))
        print(f"  [semantic-check] annotated {len(triples)} triple(s) in {_raw.name}", flush=True)
    except Exception as _e:
        print(f"  [warn] semantic check failed: {_e}", flush=True)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ExtractionAgent: adaptive triple extraction with iterative entity resolution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python agents/extraction_agent.py --schema schemas/drug_schema.yaml --input drug.md\n"
            "  python agents/extraction_agent.py --schema supplychain_schema.yaml --input reports/\n"
        ),
    )
    p.add_argument("--schema", required=True, metavar="PATH", help="Schema YAML file")
    p.add_argument("--input",  required=True, metavar="PATH",
                   help=".md file, .pdf file, or directory of .md files")
    p.add_argument("--force",  action="store_true",
                   help="Re-extract even if *_raw.json already exists")
    p.add_argument("--verbose", action="store_true",
                   help="Print cache hit messages from UMLS/GLEIF lookups.")
    p.add_argument("--meta", default=None, metavar="PATH",
                   help="Pipeline metadata YAML (instructions + per-PDF page filters).")
    p.add_argument(
        "--filter", default="moderate",
        choices=["loose", "moderate", "strict"],
        help="Entity grounding filter level (default: moderate). "
             "loose=keep all | moderate=green+yellow | strict=green only.",
    )
    p.add_argument("--constraint", dest="filter", help=argparse.SUPPRESS)  # backward-compat alias
    p.add_argument(
        "--rel-type", default="", metavar="REL_TYPE",
        help="Restrict extraction to a single relationship type (e.g. HAS_MECHANISM_OF_ACTION).",
    )
    p.add_argument(
        "--output-dir", default=None, metavar="DIR",
        help="Directory to write *_raw.json files (default: alongside input files)",
    )
    p.add_argument(
        "--umls-resolve", default="top",
        choices=["top", "agent"],
        help="UMLS multi-hit strategy: 'top' picks the first result (default); "
             "'agent' sends all candidates to the agent to pick the best one.",
    )
    p.add_argument(
        "--gleif-resolve", default="top",
        choices=["top", "agent"],
        help="GLEIF multi-hit strategy: 'top' picks the first result (default); "
             "'agent' sends all candidates to the agent to pick the best one.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import lookups as _lookups_mod
    _lookups_mod.VERBOSE = args.verbose

    input_path  = Path(args.input).resolve()
    schema_path = Path(args.schema).resolve()
    meta_path   = str(Path(args.meta).resolve()) if args.meta else ""
    output_dir  = args.output_dir or ""
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")
    if not schema_path.exists():
        raise SystemExit(f"Schema not found: {schema_path}")

    if input_path.is_dir():
        md_files = sorted(input_path.glob("*.md"))
        if not md_files:
            raise SystemExit(f"No .md files found in: {input_path}")
        print(f"ExtractionAgent — processing {len(md_files)} file(s) in {input_path}")
        for md_file in md_files:
            _rdir = Path(output_dir) if output_dir else md_file.parent
            raw_path = _rdir / (md_file.stem + "_raw.json")
            if not args.force and raw_path.exists():
                print(f"  SKIP  {md_file.name}  (cached → {raw_path.name})")
                continue
            _out = run_extraction(str(md_file), str(schema_path), meta_path, args.filter, args.umls_resolve, args.gleif_resolve, output_dir, rel_type_filter=args.rel_type)
            if _out:
                _cli_semantic_check(_out, str(schema_path), args.filter)
    else:
        _rdir = Path(output_dir) if output_dir else input_path.parent
        raw_path = _rdir / (input_path.stem + "_raw.json")
        if not args.force and raw_path.exists():
            print(f"  SKIP  {input_path.name}  (cached → {raw_path.name})")
        else:
            _out = run_extraction(str(input_path), str(schema_path), meta_path, args.filter, args.umls_resolve, args.gleif_resolve, output_dir, rel_type_filter=args.rel_type)
            if _out:
                _cli_semantic_check(_out, str(schema_path), args.filter)


if __name__ == "__main__":
    main()
