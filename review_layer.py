#!/usr/bin/env python3
"""
Immutable/mutable separation for extraction output.

_raw.json    — written by extract.py; never modified by human
_review.json — event log of human decisions (ACCEPT / REJECT / OVERRIDE / ADD)

Public API
----------
materialize(raw_data, events)               -> effective triple list
save_events(review_path, raw_path, events)  -> write _review.json
get_conflicts(raw_data, events, stored_hash, raw_path) -> list[dict]
"""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


# ── Path helpers ──────────────────────────────────────────────────────────────

def review_path_for(raw_path: Path) -> Path:
    """Return the _review.json path for a given _raw.json."""
    name = raw_path.name
    if name.endswith("_raw.json"):
        base = name[: -len("_raw.json")]
    else:
        base = raw_path.stem
    return raw_path.parent / f"{base}_review.json"


# ── IO helpers ────────────────────────────────────────────────────────────────

def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_events(review_path: Path) -> dict[str, dict]:
    """Load event dict from _review.json. Returns {} if file absent or unreadable."""
    if not review_path.exists():
        return {}
    try:
        data = json.loads(review_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] Could not parse review file {review_path.name}: {e} — review events ignored")
        return {}
    return data.get("events", {})


def load_review_meta(review_path: Path) -> dict:
    """Load full _review.json metadata (raw_hash, saved_at). Returns empty defaults if absent."""
    if not review_path.exists():
        return {"raw_hash": "", "events": {}, "saved_at": ""}
    try:
        return json.loads(review_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] Could not parse review meta {review_path.name}: {e} — returning empty defaults")
        return {"raw_hash": "", "events": {}, "saved_at": ""}


def save_events(
    review_path: Path,
    raw_path: Path,
    events: dict[str, dict],
) -> None:
    """Write the event dict to _review.json, recording the current raw hash."""
    out = {
        "raw_hash": file_hash(raw_path),
        "events": events,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    review_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))


# ── Core logic ────────────────────────────────────────────────────────────────

def materialize(raw_data: dict, events: dict[str, dict]) -> list[dict]:
    """Merge raw triples with human events to produce the effective triple list.

    Default (no event for a fact_id) = ACCEPT → include as-is.
    REJECT  → exclude.
    OVERRIDE → include with replaced from_props / to_props / rel_props.
    ADD     → append manually-created triple.
    """
    triples: list[dict] = []
    for t in raw_data.get("triples", []):
        tid = t.get("_id", "")
        ev  = events.get(tid, {})
        action = ev.get("action", "ACCEPT")
        if action == "REJECT":
            continue
        if action == "OVERRIDE":
            t = copy.deepcopy(t)
            for field in ("from_props", "to_props", "rel_props", "triple_color",
                          "supporting_quote", "evidence"):
                if field in ev:
                    t[field] = copy.deepcopy(ev[field])
            # Invariant #6: supporting_quote is DERIVED from evidence. The save path
            # always writes both together, but a hand-edited _review.json may set
            # evidence alone — derive the quote so the triple never carries a stale
            # supporting_quote inconsistent with its evidence spans. (The reverse —
            # quote without evidence — cannot be reconstructed here without the source
            # document; re-anchoring lives in the save path, not materialize.)
            if "evidence" in ev and "supporting_quote" not in ev:
                t["supporting_quote"] = " / ".join(
                    s.get("text", "") for s in (t.get("evidence") or []) if s.get("text")
                )
        triples.append(t)

    for tid, ev in events.items():
        if ev.get("action") == "ADD" and "triple" in ev:
            t = copy.deepcopy(ev["triple"])
            t.setdefault("_id", tid)
            triples.append(t)

    return triples


def get_conflicts(
    raw_data: dict,
    events: dict[str, dict],
    stored_hash: str,
    raw_path: Path,
) -> list[dict]:
    """Return conflict descriptions when raw was re-extracted after events were saved.

    A conflict arises when an OVERRIDE was recorded but the raw extraction has since
    changed the same field, making the stored 'original_*' values stale.
    """
    if not raw_path.exists() or not stored_hash:
        return []
    if file_hash(raw_path) == stored_hash:
        return []  # raw unchanged — no conflicts

    raw_by_id = {t.get("_id", ""): t for t in raw_data.get("triples", [])}
    conflicts: list[dict] = []

    for tid, ev in events.items():
        action = ev.get("action", "ACCEPT")
        if action != "OVERRIDE":
            continue
        if tid not in raw_by_id:
            conflicts.append({
                "fact_id": tid,
                "type":    "missing",
                "message": "Overridden fact no longer present in new extraction",
            })
        else:
            raw_t = raw_by_id[tid]
            for field in ("from_props", "to_props"):
                orig = ev.get(f"original_{field}")
                if orig and orig != raw_t.get(field, {}):
                    conflicts.append({
                        "fact_id": tid,
                        "type":    "changed",
                        "field":   field,
                        "message": f"Extraction changed {field} since override was saved",
                    })

    return conflicts
