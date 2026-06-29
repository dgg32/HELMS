#!/usr/bin/env python3
"""Persistent SQLite cache for UMLS and GLEIF lookup results.

Two-level lookup strategy used by lookups.py:
  L1: module-level dict  (fast, lost on restart)
  L2: this module        (persisted to lookup_cache.db across restarts)

DB location: LOOKUP_CACHE_DB env var, or <project_root>/lookup_cache.db.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path

_DB_PATH = Path(os.environ.get("LOOKUP_CACHE_DB", Path(__file__).parent / "lookup_cache.db"))

# Time-to-live. When > 0, entries older than this are treated as cache misses on
# read (and deleted), so stale GLEIF/UMLS data self-refreshes. Default 30 days;
# set LOOKUP_CACHE_TTL_DAYS=0 to disable expiry (entries live forever).
try:
    _TTL_SECONDS = float(os.environ.get("LOOKUP_CACHE_TTL_DAYS", "30")) * 86400.0
except ValueError:
    _TTL_SECONDS = 30.0 * 86400.0

_local = threading.local()


def _init_db() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS lookup_cache (
            service    TEXT NOT NULL,
            cache_key  TEXT NOT NULL,
            value      TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (service, cache_key)
        )
    """)
    conn.commit()
    conn.close()


_init_db()


def _conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn"):
        conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _local.conn = conn
    return _local.conn


def get(service: str, key: tuple) -> str | None:
    key_str = json.dumps(list(key), ensure_ascii=False)
    row = _conn().execute(
        "SELECT value, created_at FROM lookup_cache WHERE service = ? AND cache_key = ?",
        (service, key_str),
    ).fetchone()
    if not row:
        return None
    value, created_at = row
    if _TTL_SECONDS and (time.time() - created_at) > _TTL_SECONDS:
        # Expired — drop it and report a miss so the caller re-fetches fresh data.
        conn = _conn()
        conn.execute(
            "DELETE FROM lookup_cache WHERE service = ? AND cache_key = ?",
            (service, key_str),
        )
        conn.commit()
        return None
    return value


def put(service: str, key: tuple, value: str) -> None:
    key_str = json.dumps(list(key), ensure_ascii=False)
    _conn().execute(
        "INSERT OR REPLACE INTO lookup_cache (service, cache_key, value, created_at) VALUES (?, ?, ?, ?)",
        (service, key_str, value, time.time()),
    )
    _conn().commit()


# ── L1+L2 cache orchestration ────────────────────────────────────────────────
#
# Every lookup in this project used to hand-roll the same five-line dance:
#   check the L1 dict → check L2 (get) → promote to L1 → compute on miss →
#   store in L1 (put) only when the result is "cacheable".
#
# That last clause was the bug magnet: each call site decided "cacheable" with
# its own control flow, and one site (GLEIF parent lookup) silently forgot to
# store a deterministic no-match — so it re-hit the live API forever. The single
# `cached()` / `cached_async()` entry point below makes the contract structural:
# `compute()` returns `(value, cacheable)`, and the helper — not the caller — is
# the only place that writes the cache. A deterministic negative is cached by
# returning `cacheable=True`; only transient failures return `False`.

_MISS = object()  # sentinel distinct from a legitimately-cached None


def _l1_store(l1: dict, key, value, l1_max: int | None) -> None:
    """Insert into an L1 dict, evicting the oldest entry past l1_max (FIFO cap)."""
    if l1_max is not None and key not in l1 and len(l1) >= l1_max:
        l1.pop(next(iter(l1)))
    l1[key] = value


def _copy(value, copy):
    """Return a shallow copy so callers can't mutate the cached object in place."""
    if value is None:
        return None
    if copy is list:
        return list(value)
    if copy is dict:
        return dict(value)
    return value


def _l1l2_read(l1: dict, service: str, key, decode, l1_max: int | None):
    """Return the cached value, or _MISS. Promotes an L2 hit into L1 (decoded)."""
    if key in l1:
        return l1[key]
    raw = get(service, key)
    if raw is None:
        return _MISS
    value = decode(raw) if decode else raw
    _l1_store(l1, key, value, l1_max)
    return value


def cached(
    l1: dict,
    service: str,
    key: tuple,
    compute,
    *,
    decode=None,
    encode=None,
    copy=None,
    l1_max: int | None = None,
    verbose: bool = False,
    label: str = "",
):
    """L1→L2→compute cache. `compute()` returns `(value, cacheable: bool)`.

    decode/encode bridge the L2 string store to a richer in-memory value (e.g.
    json.loads/json.dumps for dict/list caches); omit them for string caches.
    `copy` (list or dict) makes returns defensive copies. `l1_max` caps the L1
    dict. Only `cacheable` results are written — to BOTH levels, in one place.
    """
    hit = _l1l2_read(l1, service, key, decode, l1_max)
    if hit is not _MISS:
        if verbose:
            print(f"  [cache] {label}", flush=True)
        return _copy(hit, copy)
    value, cacheable = compute()
    if cacheable:
        put(service, key, encode(value) if encode else value)
        _l1_store(l1, key, value, l1_max)
    return _copy(value, copy)


async def cached_async(
    l1: dict,
    service: str,
    key: tuple,
    compute,
    *,
    decode=None,
    encode=None,
    copy=None,
    l1_max: int | None = None,
    verbose: bool = False,
    label: str = "",
):
    """Async twin of `cached()`; `compute` is awaited. See `cached` for params."""
    hit = _l1l2_read(l1, service, key, decode, l1_max)
    if hit is not _MISS:
        if verbose:
            print(f"  [cache] {label}", flush=True)
        return _copy(hit, copy)
    value, cacheable = await compute()
    if cacheable:
        put(service, key, encode(value) if encode else value)
        _l1_store(l1, key, value, l1_max)
    return _copy(value, copy)


def stats() -> dict:
    """Return row counts and oldest/newest entry per service."""
    rows = _conn().execute("""
        SELECT service, COUNT(*) AS n,
               MIN(created_at) AS oldest,
               MAX(created_at) AS newest
        FROM lookup_cache GROUP BY service
    """).fetchall()
    return {
        r[0]: {"count": r[1], "oldest": r[2], "newest": r[3]}
        for r in rows
    }


def evict(pattern: str, service: str | None = None) -> int:
    """Delete cached rows whose cache_key contains `pattern` (case-insensitive substring).

    Use to drop a single poisoned entry (e.g. a wrong GLEIF/UMLS resolution) so the next
    run re-fetches it. Optionally scope to one service ('gleif', 'umls', 'gleif_pick', …).
    Returns the number of rows deleted.
    """
    conn = _conn()
    like = f"%{pattern.lower()}%"
    if service:
        cur = conn.execute(
            "DELETE FROM lookup_cache WHERE service = ? AND LOWER(cache_key) LIKE ?",
            (service, like),
        )
    else:
        cur = conn.execute(
            "DELETE FROM lookup_cache WHERE LOWER(cache_key) LIKE ?",
            (like,),
        )
    conn.commit()
    return cur.rowcount


def clear(service: str | None = None) -> int:
    """Delete every row, or all rows for one service. Returns the number deleted."""
    conn = _conn()
    if service:
        cur = conn.execute("DELETE FROM lookup_cache WHERE service = ?", (service,))
    else:
        cur = conn.execute("DELETE FROM lookup_cache")
    conn.commit()
    return cur.rowcount


def _format_ts(ts: float | None) -> str:
    if not ts:
        return "—"
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m lookup_cache",
        description="Inspect and maintain the UMLS/GLEIF lookup cache (lookup_cache.db).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stats", help="Show row counts and age per service.")

    pe = sub.add_parser(
        "evict",
        help="Delete entries whose cache key contains a substring (e.g. a company name).",
    )
    pe.add_argument("pattern", help="Case-insensitive substring matched against the cache key.")
    pe.add_argument("--service", default=None,
                    help="Limit to one service (gleif, umls, gleif_pick, umls_pick, …).")

    pc = sub.add_parser("clear", help="Delete all entries, or all of one service.")
    pc.add_argument("--service", default=None, help="Limit to one service.")
    pc.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")

    args = p.parse_args()

    if args.cmd == "stats":
        data = stats()
        if not data:
            print("Cache is empty.")
            return
        total = sum(v["count"] for v in data.values())
        print(f"DB: {_DB_PATH}")
        ttl = f"{_TTL_SECONDS / 86400:.1f} day(s)" if _TTL_SECONDS else "disabled"
        print(f"TTL: {ttl}")
        print(f"{'service':<20} {'rows':>7}  {'oldest':<20} {'newest':<20}")
        for svc in sorted(data):
            v = data[svc]
            print(f"{svc:<20} {v['count']:>7}  {_format_ts(v['oldest']):<20} {_format_ts(v['newest']):<20}")
        print(f"{'TOTAL':<20} {total:>7}")

    elif args.cmd == "evict":
        n = evict(args.pattern, service=args.service)
        scope = f" in service {args.service!r}" if args.service else ""
        print(f"Evicted {n} row(s) matching {args.pattern!r}{scope}.")

    elif args.cmd == "clear":
        scope = f"service {args.service!r}" if args.service else "ALL services"
        if not args.yes:
            reply = input(f"Delete every cached row for {scope}? [y/N] ").strip().lower()
            if reply not in ("y", "yes"):
                print("Aborted.")
                return
        n = clear(service=args.service)
        print(f"Cleared {n} row(s) ({scope}).")


if __name__ == "__main__":
    _main()
