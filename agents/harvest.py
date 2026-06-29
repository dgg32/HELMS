#!/usr/bin/env python3
"""
Harvest reviewed triples from all runs into per-rel-type example stores.

Store location: <project_dir>/harvest/<rel_type>.jsonl
Each line is one JSON example entry.

Priority tiers (lower = more valuable):
  1 = ADD      — human wrote from scratch
  2 = OVERRIDE — human corrected an existing triple (stores original wrong values too)
  3 = agent_retry — batch missed it; agentic retry with tool calls got it right
  4 = batch    — easy first-pass green triple

Selection for prompt injection: sort by priority, then diversify by to_display to
avoid injecting the same target entity multiple times.
"""
from __future__ import annotations

import copy
import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

_harvest_lock = threading.Lock()

PRIORITY: dict[str, int] = {"add": 1, "override": 2, "agent_retry": 3, "batch": 4}
DEFAULT_CAP = 20              # max entries kept per rel_type in the store
DEFAULT_INJECT_K = 5          # max positive examples injected per rel_type
DEFAULT_INJECT_K_NEG = 3      # max negative (rejected) examples injected per rel_type

_SOURCE_LABEL = {
    "add":         "human-added",
    "override":    "human-corrected",
    "agent_retry": "agent-retry",
    "batch":       "auto-accepted",
    "rejected":    "human-rejected",
}


# ── Internal helpers ───────────────────────────────────────────────────────────

def _display(props: dict, pk: str) -> str:
    return props.get("name") or props.get(pk) or "?"


def _entry_id(rel_type: str, from_d: str, to_d: str, source: str) -> str:
    h = hashlib.sha256(f"{rel_type}|{from_d}|{to_d}|{source}".encode()).hexdigest()[:12]
    return f"h{h}"


def _make_entry(
    triple: dict,
    source: str,
    run_id: str,
    doc_name: str,
    *,
    original_from_props: dict | None = None,
    original_to_props: dict | None = None,
) -> dict | None:
    rel_type   = triple.get("rel_type", "")
    from_props = triple.get("from_props", {})
    to_props   = triple.get("to_props", {})
    from_pk    = triple.get("from_pk", "")
    to_pk      = triple.get("to_pk", "")
    if not rel_type or not from_props or not to_props:
        return None

    from_d = _display(from_props, from_pk)
    to_d   = _display(to_props,   to_pk)
    quote  = (
        (triple.get("rel_props") or {}).get("supporting_quote")
        or triple.get("supporting_quote")
        or ""
    )

    entry: dict = {
        "_id":              _entry_id(rel_type, from_d, to_d, source),
        "rel_type":         rel_type,
        "from_display":     from_d,
        "to_display":       to_d,
        "supporting_quote": quote,
        "source":           source,
        "priority":         PRIORITY.get(source, 99),
        "run_id":           run_id,
        "doc_name":         doc_name,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    }
    if original_from_props is not None:
        entry["original_from_display"] = _display(original_from_props, from_pk)
    if original_to_props is not None:
        entry["original_to_display"] = _display(original_to_props, to_pk)
    return entry


def _upsert(store: dict[str, dict], entry: dict) -> None:
    """Insert or upgrade an entry. Never downgrade to a lower-priority source."""
    eid = entry["_id"]
    if eid in store:
        if PRIORITY.get(entry["source"], 99) >= PRIORITY.get(store[eid]["source"], 99):
            return  # existing is same or better priority
    store[eid] = entry


# ── Public API ─────────────────────────────────────────────────────────────────

def harvest_project(project_dir: str | Path, *, cap: int = DEFAULT_CAP) -> dict[str, int]:
    """Scan all runs in project_dir, rebuild harvest store. Returns {rel_type: count}."""
    with _harvest_lock:
        return _harvest_project_locked(project_dir, cap=cap)


def _harvest_project_locked(project_dir: str | Path, *, cap: int = DEFAULT_CAP) -> dict[str, int]:
    project_dir = Path(project_dir)
    runs_dir    = project_dir / "runs"
    harvest_dir = project_dir / "harvest"
    harvest_dir.mkdir(exist_ok=True)

    if not runs_dir.exists():
        return {}

    # Rebuild store from scratch on every call so rejections are honoured immediately.
    # Loading the existing store and patching it would leave stale entries for rejected
    # triples — the scan already covers all runs so a full rebuild is equivalent.
    store: dict[str, dict[str, dict]] = {}

    # Logical triples suppressed across ALL runs: (rel_type, from_display, to_display).
    # A REJECT in any run or an OVERRIDE (original form) suppresses that logical triple
    # even if it appeared as a clean batch entry in an earlier run.
    rejected_logical: set[tuple[str, str, str]] = set()

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name

        for raw_path in sorted(run_dir.glob("*_raw.json")):
            review_path = raw_path.parent / raw_path.name.replace("_raw.json", "_review.json")
            try:
                raw_data = json.loads(raw_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            triples_by_id = {
                t["_id"]: t
                for t in raw_data.get("triples", [])
                if t.get("_id")
            }

            events: dict[str, dict] = {}
            if review_path.exists():
                try:
                    events = json.loads(review_path.read_text(encoding="utf-8")).get("events", {})
                except Exception:
                    pass

            rejected_ids: set[str] = {
                tid for tid, ev in events.items()
                if ev.get("action") == "REJECT"
            }

            # REJECT: suppress from positive store AND log as negative example
            for tid, ev in events.items():
                if ev.get("action") != "REJECT":
                    continue
                t = triples_by_id.get(tid)
                if not t:
                    continue
                from_d = _display(t.get("from_props", {}), t.get("from_pk", ""))
                to_d   = _display(t.get("to_props",   {}), t.get("to_pk",   ""))
                rejected_logical.add((t.get("rel_type", ""), from_d, to_d))
                # Store as negative example so LLM sees "this quote ≠ this mapping"
                entry = _make_entry(t, "rejected", run_id, raw_path.name)
                if entry:
                    rel = entry["rel_type"]
                    store.setdefault(rel, {})
                    _upsert(store[rel], entry)

            # Tier 1: ADD events
            for tid, ev in events.items():
                if ev.get("action") != "ADD":
                    continue
                t = ev.get("triple", {})
                entry = _make_entry(t, "add", run_id, raw_path.name)
                if entry:
                    rel = entry["rel_type"]
                    store.setdefault(rel, {})
                    _upsert(store[rel], entry)
                    # Explicit ADD overrides any prior rejection of the same logical triple.
                    rejected_logical.discard((entry["rel_type"], entry["from_display"], entry["to_display"]))

            # Tier 2: OVERRIDE events
            # Three cases based on what changed:
            #   entity corrected → suppress old logical triple, add new one if green
            #   color → red/yellow → suppress (mark bad/uncertain)
            #   color → green → add to store (explicit user validation), don't suppress
            for tid, ev in events.items():
                if ev.get("action") != "OVERRIDE":
                    continue
                t = triples_by_id.get(tid)
                if not t:
                    continue
                t = copy.deepcopy(t)
                orig_from = copy.deepcopy(t.get("from_props", {}))
                orig_to   = copy.deepcopy(t.get("to_props",   {}))
                for field in ("from_props", "to_props", "rel_props", "triple_color"):
                    if field in ev:
                        t[field] = ev[field]

                from_pk = t.get("from_pk", "")
                to_pk   = t.get("to_pk",   "")
                orig_from_d = _display(orig_from,           from_pk)
                orig_to_d   = _display(orig_to,             to_pk)
                new_from_d  = _display(t.get("from_props", {}), from_pk)
                new_to_d    = _display(t.get("to_props",   {}), to_pk)
                entity_changed = (orig_from_d != new_from_d) or (orig_to_d != new_to_d)
                result_color   = t.get("triple_color", "green")

                # Suppress original form when entity was corrected (old target wrong)
                # or triple was explicitly marked red/yellow (bad/uncertain)
                if entity_changed or result_color in ("red", "yellow"):
                    rejected_logical.add((t.get("rel_type", ""), orig_from_d, orig_to_d))

                # Add corrected/validated form to store only if green
                if result_color == "green":
                    entry = _make_entry(
                        t, "override", run_id, raw_path.name,
                        original_from_props=orig_from if entity_changed else None,
                        original_to_props=orig_to   if entity_changed else None,
                    )
                    if entry:
                        rel = entry["rel_type"]
                        store.setdefault(rel, {})
                        _upsert(store[rel], entry)

            # Tiers 3 & 4: raw triples (agent_retry or batch), green only, not rejected
            for t in raw_data.get("triples", []):
                tid = t.get("_id", "")
                if tid in rejected_ids:
                    continue
                if tid in events and events[tid].get("action") in ("ADD", "OVERRIDE"):
                    continue  # already handled above
                if t.get("triple_color") != "green":
                    continue
                src = "agent_retry" if t.get("extraction_source") == "agent_retry" else "batch"
                entry = _make_entry(t, src, run_id, raw_path.name)
                if entry:
                    rel = entry["rel_type"]
                    store.setdefault(rel, {})
                    _upsert(store[rel], entry)

    # Remove positive entries whose logical triple was rejected/overridden in any run.
    # Rejected entries (source="rejected") are exempt — they ARE the negative signal.
    for rel_type in list(store.keys()):
        store[rel_type] = {
            eid: e for eid, e in store[rel_type].items()
            if e["source"] == "rejected"
            or (e["rel_type"], e["from_display"], e["to_display"]) not in rejected_logical
        }

    # Delete all existing .jsonl files before writing so stale files for emptied or
    # removed rel_types are never read by load_examples on the next extraction run.
    for old_file in harvest_dir.glob("*.jsonl"):
        old_file.unlink(missing_ok=True)

    # Write back, capping positives and negatives separately so rejected examples
    # are never evicted when the positive store exceeds cap.
    counts: dict[str, int] = {}
    for rel_type, entries in store.items():
        pos = sorted(
            (e for e in entries.values() if e.get("source") != "rejected"),
            key=lambda e: (PRIORITY.get(e["source"], 99), e.get("timestamp", "")),
        )
        neg = sorted(
            (e for e in entries.values() if e.get("source") == "rejected"),
            key=lambda e: e.get("timestamp", ""),
            reverse=True,
        )
        capped = pos[:cap] + neg[:DEFAULT_INJECT_K_NEG]
        out_path = harvest_dir / f"{rel_type}.jsonl"
        out_path.write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in capped) + "\n",
            encoding="utf-8",
        )
        counts[rel_type] = len(capped)

    return counts


def load_examples(
    harvest_dir: str | Path,
    rel_type: str,
    k: int = DEFAULT_INJECT_K,
    k_negative: int = DEFAULT_INJECT_K_NEG,
) -> list[dict]:
    """Return top-k positive + up to k_negative negative examples for rel_type.

    Positives and negatives are returned in a single list; callers can distinguish
    them by checking entry["source"] == "rejected".
    """
    harvest_dir = Path(harvest_dir)
    path = harvest_dir / f"{rel_type}.jsonl"
    if not path.exists():
        return []

    positives: list[dict] = []
    negatives: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
            (negatives if e.get("source") == "rejected" else positives).append(e)
        except Exception:
            pass

    # Positives: sort by priority ASC, diversify by to_display, cap at k
    positives.sort(key=lambda e: (PRIORITY.get(e["source"], 99), e.get("timestamp", "")))
    seen_to: set[str] = set()
    diverse: list[dict] = []
    for e in positives:
        if e["to_display"] not in seen_to:
            diverse.append(e)
            seen_to.add(e["to_display"])
        if len(diverse) >= k:
            break

    # Negatives: most recent first, cap at k_negative
    negatives.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return diverse + negatives[:k_negative]


def _render_entry(e: dict) -> list[str]:
    """Return 1-2 prompt lines for one harvest entry."""
    label    = _SOURCE_LABEL.get(e["source"], e["source"])
    quote    = (e.get("supporting_quote") or "").strip()
    from_d   = e["from_display"]
    to_d     = e["to_display"]
    orig_from = e.get("original_from_display", "")
    orig_to   = e.get("original_to_display",   "")
    correction_parts = [
        f"from corrected from: {orig_from}" if orig_from and orig_from != from_d else "",
        f"to corrected from: {orig_to}"     if orig_to   and orig_to   != to_d   else "",
    ]
    correction = "  " + ", ".join(p for p in correction_parts if p) if any(correction_parts) else ""
    lines = [f'  • "{quote[:250]}"' if quote else "  •"]
    doc_tag = f", doc: {e.get('doc_name', '')}" if e.get("source") == "rejected" and e.get("doc_name") else ""
    lines.append(f'    → ({from_d}, {to_d})  [{label}{correction}{doc_tag}]')
    return lines


def format_examples_block(examples: list[dict], rel_type: str) -> str:
    """Format harvest examples as a prompt section with positive and negative subsections."""
    positives = [e for e in examples if e.get("source") != "rejected"]
    negatives = [e for e in examples if e.get("source") == "rejected"]
    if not positives and not negatives:
        return ""

    lines = [f"--- Past extraction examples for {rel_type} ---"]
    if positives:
        lines.append(
            "CORRECT extractions — these are STYLE PATTERNS, not an exhaustive list. "
            "Do NOT restrict yourself to entities shown here: extract EVERY pair in the "
            "document that fits this relationship, including ones absent from these examples."
        )
        for e in positives:
            lines.extend(_render_entry(e))
    if negatives:
        lines.append("PREVIOUSLY REJECTED extractions (rejection was document-specific — extract again only if the current document provides clearly stronger or different supporting evidence):")
        for e in negatives:
            lines.extend(_render_entry(e))
    return "\n".join(lines)


def format_rejection_reminder(examples: list[dict], rel_type: str) -> str:
    """Compact per-chunk rejection reminder injected into the user message (not system prompt).

    Placed immediately before the document text so the model sees it at highest recency.
    """
    negatives = [e for e in examples if e.get("source") == "rejected"]
    if not negatives:
        return ""
    lines = [
        f"⚠ REJECTION LIST for {rel_type} — these triples were explicitly rejected by a human reviewer.",
        "Do NOT extract any of these unless the document text below contains evidence that is MORE EXPLICIT and DIRECT than the originally rejected quote.",
    ]
    for e in negatives:
        from_d = e["from_display"]
        to_d   = e["to_display"]
        doc    = e.get("doc_name", "")
        quote  = (e.get("supporting_quote") or "").strip()
        lines.append(f"  • ({from_d}) -[{rel_type}]-> ({to_d})")
        if doc:
            lines.append(f"    Rejected from doc: {doc}")
        if quote:
            lines.append(f"    Rejected quote: \"{quote[:200]}\"")
    return "\n".join(lines)
