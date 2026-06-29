#!/usr/bin/env python3
"""
Evaluate extraction quality against a gold-standard TSV.

Usage:
    python eval_extraction.py --gold projects/drug/eval/gold_template.tsv --run projects/drug/runs/20260610_120000
    python eval_extraction.py --gold projects/drug/eval/gold_template.tsv --run projects/drug/runs/20260610_120000 --rel-type MAY_TREAT --verbose
    python eval_extraction.py --gold projects/drug/eval/gold_template.tsv --run projects/drug/runs/20260610_120000 --filter strict

Gold TSV lives inside the project:
    projects/<name>/eval/gold_template.tsv  (fill in actual triples; delete placeholder rows)

Gold TSV columns (tab-separated):
    rel_type        e.g. HAS_INDICATION
    from_display    display name of source entity (case-insensitive match)
    to_display      display name of target entity (case-insensitive match)
    doc_name        stem of the source PDF/raw file, e.g. "beyfortus_label"
                    (omit column or leave blank for corpus-level match across all docs)
    notes           ignored; for human reference only

Matching modes:
    Doc-scoped  (doc_name present): extracted triples are filtered to that document.
                FP is also limited to triples from gold-referenced documents, so adding
                extra docs to the run folder doesn't inflate FP.
    Corpus-level (doc_name absent): gold triple counts as TP if found in any document.
                FP counts all extracted triples not in gold.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).parent))
import review_layer as _rl


class Triple(NamedTuple):
    rel_type: str
    from_display: str
    to_display: str


_COLOR_KEEP = {
    "loose":    {"green", "yellow", "red"},
    "moderate": {"green", "yellow"},
    "strict":   {"green"},
}


def load_gold(tsv_path: Path) -> tuple[dict[str, set[Triple]], set[Triple]]:
    """Return (doc_scoped, corpus).

    doc_scoped  : doc_stem -> set of gold triples for that document
    corpus      : gold triples with no doc_name (matched against all extracted)
    """
    doc_scoped: dict[str, set[Triple]] = defaultdict(set)
    corpus: set[Triple] = set()
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rel = row.get("rel_type", "").strip()
            frm = row.get("from_display", "").strip().lower()
            to  = row.get("to_display",  "").strip().lower()
            doc = row.get("doc_name",    "").strip()
            if not (rel and frm and to):
                continue
            t = Triple(rel, frm, to)
            if doc:
                doc_scoped[doc].add(t)
            else:
                corpus.add(t)
    return dict(doc_scoped), corpus


def _display(triple: dict, side: str) -> str:
    props = triple.get(f"{side}_props") or {}
    pk    = triple.get(f"{side}_pk", "")
    return (props.get("name") or props.get(pk) or "").strip().lower()


def load_extracted_by_doc(run_dir: Path, filter_level: str = "moderate") -> dict[str, set[Triple]]:
    """Return doc_stem -> set of extracted triples from that document."""
    keep = _COLOR_KEEP[filter_level]
    by_doc: dict[str, set[Triple]] = {}
    for raw_path in sorted(run_dir.glob("*_raw.json")):
        doc_stem = raw_path.name[: -len("_raw.json")]
        rev_path = _rl.review_path_for(raw_path)
        triples  = _rl.materialize(_rl.load_raw(raw_path), _rl.load_events(rev_path))
        doc_set: set[Triple] = set()
        for t in triples:
            if t.get("triple_color", "green") not in keep:
                continue
            rel = t.get("rel_type", "").strip()
            frm = _display(t, "from")
            to  = _display(t, "to")
            if rel and frm and to:
                doc_set.add(Triple(rel, frm, to))
        by_doc[doc_stem] = doc_set
    return by_doc


def compute_metrics(gold: set[Triple], extracted: set[Triple]) -> tuple:
    tp = gold & extracted
    fp = extracted - gold
    fn = gold - extracted
    precision = len(tp) / len(extracted) if extracted else 0.0
    recall    = len(tp) / len(gold)      if gold      else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return tp, fp, fn, precision, recall, f1


def _row(label: str, p: float, r: float, f: float, tp: int, fp: int, fn: int, gold_n: int, ext_n: int) -> str:
    return (
        f"  {label:<34} P={p:.3f}  R={r:.3f}  F1={f:.3f}"
        f"  (TP={tp} FP={fp} FN={fn} | gold={gold_n} extracted={ext_n})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate HELMS extraction against a gold-standard TSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--gold",     required=True, help="Gold-standard TSV file")
    ap.add_argument("--run",      required=True, help="Run folder with *_raw.json files")
    ap.add_argument("--rel-type", default=None,  help="Restrict to one rel_type")
    ap.add_argument("--filter",   default="moderate", choices=["loose", "moderate", "strict"],
                    help="Color filter matching Step 3 write filter")
    ap.add_argument("--verbose",  action="store_true", help="Print TP/FP/FN triple lists")
    args = ap.parse_args()

    gold_path = Path(args.gold)
    run_dir   = Path(args.run)
    if not gold_path.exists():
        sys.exit(f"Gold file not found: {gold_path}")
    if not run_dir.is_dir():
        sys.exit(f"Run directory not found: {run_dir}")

    doc_scoped, corpus = load_gold(gold_path)
    by_doc = load_extracted_by_doc(run_dir, filter_level=args.filter)
    extracted_all = set().union(*by_doc.values()) if by_doc else set()

    # Optional rel_type filter
    if args.rel_type:
        doc_scoped = {d: {t for t in s if t.rel_type == args.rel_type} for d, s in doc_scoped.items()}
        corpus     = {t for t in corpus if t.rel_type == args.rel_type}
        by_doc     = {d: {t for t in s if t.rel_type == args.rel_type} for d, s in by_doc.items()}
        extracted_all = {t for t in extracted_all if t.rel_type == args.rel_type}

    print(f"\n{'='*72}")
    print(f"Run:    {run_dir}")
    print(f"Gold:   {gold_path}")
    print(f"Filter: {args.filter}")
    print(f"{'='*72}")

    # ── Doc-scoped evaluation ──────────────────────────────────────────────────
    if doc_scoped:
        print(f"\nDoc-scoped evaluation ({sum(len(v) for v in doc_scoped.values())} gold triples across {len(doc_scoped)} document(s))")
        agg_tp: set[Triple] = set()
        agg_fp: set[Triple] = set()
        agg_fn: set[Triple] = set()
        agg_gold: set[Triple] = set()
        agg_ext:  set[Triple] = set()

        for doc_stem, gold_set in sorted(doc_scoped.items()):
            ext_set = by_doc.get(doc_stem, set())
            tp, fp, fn, p, r, f = compute_metrics(gold_set, ext_set)
            print(_row(doc_stem, p, r, f, len(tp), len(fp), len(fn), len(gold_set), len(ext_set)))
            agg_tp   |= tp
            agg_fp   |= fp
            agg_fn   |= fn
            agg_gold |= gold_set
            agg_ext  |= ext_set

        if len(doc_scoped) > 1:
            _, _, _, p, r, f = compute_metrics(agg_gold, agg_ext)
            print(_row("  TOTAL (doc-scoped)", p, r, f,
                       len(agg_tp), len(agg_fp), len(agg_fn), len(agg_gold), len(agg_ext)))

        if args.verbose:
            _print_detail(agg_tp, agg_fp, agg_fn)

        # Per rel_type breakdown
        rel_types = sorted({t.rel_type for t in agg_gold | agg_ext})
        if len(rel_types) > 1 and not args.rel_type:
            print("\n  Per rel_type (doc-scoped):")
            for rel in rel_types:
                g = {t for t in agg_gold if t.rel_type == rel}
                e = {t for t in agg_ext  if t.rel_type == rel}
                _, _, _, p, r, f = compute_metrics(g, e)
                tp2 = g & e; fp2 = e - g; fn2 = g - e
                print(_row(f"    {rel}", p, r, f, len(tp2), len(fp2), len(fn2), len(g), len(e)))

    # ── Corpus-level evaluation ────────────────────────────────────────────────
    if corpus:
        print(f"\nCorpus-level evaluation ({len(corpus)} gold triples, matched across all docs)")
        tp, fp, fn, p, r, f = compute_metrics(corpus, extracted_all)
        print(_row("ALL docs", p, r, f, len(tp), len(fp), len(fn), len(corpus), len(extracted_all)))

        if args.verbose:
            _print_detail(tp, fp, fn)

        rel_types = sorted({t.rel_type for t in corpus | extracted_all})
        if len(rel_types) > 1 and not args.rel_type:
            print("\n  Per rel_type (corpus):")
            for rel in rel_types:
                g = {t for t in corpus       if t.rel_type == rel}
                e = {t for t in extracted_all if t.rel_type == rel}
                _, _, _, p, r, f = compute_metrics(g, e)
                tp2 = g & e; fp2 = e - g; fn2 = g - e
                print(_row(f"    {rel}", p, r, f, len(tp2), len(fp2), len(fn2), len(g), len(e)))

    if not doc_scoped and not corpus:
        print("\n[warn] No valid gold triples found in TSV.")

    print()


def _print_detail(tp: set[Triple], fp: set[Triple], fn: set[Triple]) -> None:
    print(f"\n  True Positives ({len(tp)}):")
    for t in sorted(tp):
        print(f"    + {t.rel_type}: {t.from_display} → {t.to_display}")
    print(f"\n  False Positives ({len(fp)}) — extracted but not in gold:")
    for t in sorted(fp):
        print(f"    - {t.rel_type}: {t.from_display} → {t.to_display}")
    print(f"\n  False Negatives ({len(fn)}) — in gold but not extracted:")
    for t in sorted(fn):
        print(f"    ? {t.rel_type}: {t.from_display} → {t.to_display}")


if __name__ == "__main__":
    main()
