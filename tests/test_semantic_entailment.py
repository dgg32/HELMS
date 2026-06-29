#!/usr/bin/env python3
"""Step 3 — measuring the entailment/direction grader.

The semantic-check grader was repurposed from re-proving quote *presence* (which
`grounding.locate` already proves for free) to judging relation *support and
direction*. That upgrade can only be measured with the LLM in the loop, so this
is an INTEGRATION test: it is skipped automatically when no LLM credentials are
configured (CI / offline), and run on demand with real creds:

    LLM_API_KEY=... LLM_MODEL=... python -m pytest tests/test_semantic_entailment.py -v

It builds a tiny self-contained document whose ground truth we control, then
checks that the grader:
  * keeps a correctly-directed, quote-supported triple green/yellow (not red), and
  * reds a direction-swapped triple whose entities are both present but whose
    relation the quote does NOT support in that direction.

This is the regression net for the grader itself. For end-to-end extraction
precision/recall (the other half of Step 3) use eval_extraction.py against a
populated gold TSV — see this module's docstring footer.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load credentials the same way the app does: importing llm_client runs
# load_config(), which reads .env.yaml (the project's real LLM/UMLS creds) and
# setdefault()s them into os.environ. Doing it here means these LLM-in-the-loop
# tests run whenever .env.yaml has creds — standalone too — not only when some
# other test module happens to import llm_client first.
# SECURITY: .env.yaml is gitignored. Never commit it, and never echo its contents
# into code, logs, commits, PRs, or chat.
try:
    import llm_client  # noqa: F401  (import side effect: load_config populates os.environ)
except Exception:
    pass

# Skip the whole module unless an LLM is configured — the grader calls a model.
_HAS_LLM = bool(os.environ.get("LLM_API_KEY") or os.environ.get("LLM_ENDPOINT")
                or os.environ.get("OPENAI_API_KEY"))
pytestmark = pytest.mark.skipif(not _HAS_LLM, reason="no LLM credentials (set them in .env.yaml)")


# A controlled document: A supplies B, full stop. The reverse (B supplies A) is
# false and unsupported by any sentence here.
_DOC = (
    "# Supply Chain Note\n\n"
    "Acme Components Inc. supplies high-bandwidth memory modules to Globex Systems. "
    "Globex Systems integrates these modules into its flagship servers and sells the "
    "finished servers to enterprise customers. No other supplier relationship exists "
    "between the two companies.\n"
)
_QUOTE = "Acme Components Inc. supplies high-bandwidth memory modules to Globex Systems."

_SCHEMA_RELS = [{
    "rel_type": "PROVIDES",
    "from_node": "Corporation",
    "to_node": "Corporation",
    "from_hint": "the supplier",
    "to_hint": "the customer receiving goods or services",
    "extract_prompt": "from_node is the supplier; to_node is the customer. Direction matters.",
}]


def _triple(tid: str, frm: str, to: str) -> dict:
    return {
        "_id": tid, "rel_type": "PROVIDES",
        "from_label": "Corporation", "to_label": "Corporation",
        "from_pk": "name", "to_pk": "name",
        "from_props": {"name": frm}, "to_props": {"name": to},
        "supporting_quote": _QUOTE,
    }


def _grade():
    import agents.semantic_check_agent as sc
    correct = _triple("ok", "Acme Components Inc.", "Globex Systems")
    swapped = _triple("rev", "Globex Systems", "Acme Components Inc.")  # wrong direction
    out = sc.check_triples([correct, swapped], _DOC, schema_rels=_SCHEMA_RELS)
    return {t["_id"]: t for t in out}


def test_correct_direction_not_red():
    by_id = _grade()
    assert by_id["ok"]["triple_color"] != "red", by_id["ok"].get("ai_opinion")
    assert by_id["ok"].get("constraint_violated") is not True


def test_swapped_direction_is_red():
    by_id = _grade()
    # Both entities ARE present in the quote, so old presence-only grading would
    # have greened this. The direction check must catch it.
    assert by_id["rev"]["triple_color"] == "red", by_id["rev"].get("ai_opinion")
    assert by_id["rev"].get("constraint_violated") is True


# ── Single-subject document: subject is grounded even when absent from the quote ──
#
# A drug label is about ONE drug. An adverse-effect quote names the AE but not the
# drug, yet the relation IS supported because the whole document is about that drug.
# The subject is detected deterministically (sole entity of its label, recurring),
# and the grader must NOT red such triples for the subject being absent from the quote.

_LABEL_DOC = (
    "# DIFICID (fidaxomicin) tablets — Highlights of Prescribing Information\n\n"
    "DIFICID is a macrolide antibacterial drug indicated in adults for Clostridioides "
    "difficile-associated diarrhea.\n\n"
    "## 6 ADVERSE REACTIONS\n"
    "The most common adverse reactions in adults (incidence >=2%) are nausea, vomiting, "
    "abdominal pain, gastrointestinal hemorrhage, anemia, and neutropenia.\n"
)
_AE_QUOTE = (
    "The most common adverse reactions in adults (incidence >=2%) are nausea, vomiting, "
    "abdominal pain, gastrointestinal hemorrhage, anemia, and neutropenia."
)


def _ae_triple(tid: str, ae: str) -> dict:
    return {
        "_id": tid, "rel_type": "HAS_ADVERSE_EFFECT",
        "from_label": "Substance", "to_label": "AdverseEffect",
        "from_pk": "name", "to_pk": "name",
        "from_props": {"name": "fidaxomicin"}, "to_props": {"name": ae},
        "supporting_quote": _AE_QUOTE,
    }


def test_single_subject_adverse_effect_not_red():
    import agents.semantic_check_agent as sc
    rels = [{
        "rel_type": "HAS_ADVERSE_EFFECT",
        "from_node": "Substance", "to_node": "AdverseEffect",
        "from_hint": "the drug", "to_hint": "an adverse reaction the drug causes",
    }]
    triples = [_ae_triple("ae1", "Nausea"), _ae_triple("ae2", "Vomiting"),
               _ae_triple("ae3", "Anemia")]
    out = {t["_id"]: t for t in sc.check_triples(triples, _LABEL_DOC, schema_rels=rels)}
    # fidaxomicin is the sole Substance (document subject); the AEs are grounded in the
    # quote, so these must NOT be red just because the drug name is absent from the quote.
    for tid in ("ae1", "ae2", "ae3"):
        assert out[tid]["triple_color"] != "red", out[tid].get("ai_opinion")
        assert out[tid].get("constraint_violated") is not True


# ── Instructions resolve the stray-comparator case (variant B, the A/B winner) ──
#
# When a comparator drug slips into extraction the deterministic 'sole entity' flag
# can't fire (2 Substances), so the strict grader reds the PRIMARY drug's AEs. Passing
# the meta instructions ("the PRIMARY substance ... do NOT extract comparators") lets
# the grader green the primary's AEs while keeping the comparator's own AE red.

_COMPARATOR_DOC = (
    "# Zephyrol (zephyramine) Tablets — Highlights of Prescribing Information\n\n"
    "Zephyrol (zephyramine) is indicated for acute bacterial sinusitis in adults.\n\n"
    "## 6 ADVERSE REACTIONS\n"
    "The most common adverse reactions (incidence >=2%) were headache and nausea.\n\n"
    "## 14 CLINICAL STUDIES\n"
    "Zephyrol was compared with vancomycin in a randomized trial. "
    "Nephrotoxicity was more frequent in the comparator arm.\n"
)
_PRIMARY_INSTR = (
    "Extract triples for the PRIMARY substance described in the document (the drug the "
    "label is about). Do NOT extract triples for comparator drugs or substances "
    "mentioned only in passing."
)


def _hae(tid, drug, ae, quote):
    return {
        "_id": tid, "rel_type": "HAS_ADVERSE_EFFECT",
        "from_label": "Substance", "to_label": "AdverseEffect",
        "from_pk": "name", "to_pk": "name",
        "from_props": {"name": drug}, "to_props": {"name": ae},
        "supporting_quote": quote,
    }


def test_instructions_green_primary_keep_comparator_red():
    import agents.semantic_check_agent as sc
    rels = [{"rel_type": "HAS_ADVERSE_EFFECT", "from_node": "Substance",
             "to_node": "AdverseEffect", "from_hint": "the drug",
             "to_hint": "an adverse reaction the drug causes"}]
    ae_quote = "The most common adverse reactions (incidence >=2%) were headache and nausea."
    cmp_quote = "Nephrotoxicity was more frequent in the comparator arm."
    triples = [
        _hae("p1", "zephyramine", "Headache", ae_quote),
        _hae("p2", "zephyramine", "Nausea", ae_quote),
        _hae("c1", "vancomycin", "Nephrotoxicity", cmp_quote),  # comparator, drug absent from quote
    ]
    out = {t["_id"]: t for t in sc.check_triples(
        triples, _COMPARATOR_DOC, schema_rels=rels, instructions=_PRIMARY_INSTR)}
    # Primary drug's AEs: greened via the instruction-named subject (the 'sole entity'
    # flag cannot fire here — two Substances are present).
    for tid in ("p1", "p2"):
        assert out[tid]["triple_color"] != "red", out[tid].get("ai_opinion")
    # The comparator's own AE stays red: it is not the subject and the drug is not in its quote.
    assert out["c1"]["triple_color"] == "red", out["c1"].get("ai_opinion")


# ── End-to-end precision/recall (the other half of Step 3) ────────────────────
#
# The grader changes COLORS, and the Step-3 write gate filters by color
# (moderate keeps green+yellow, strict keeps green). So a stricter grader shows
# up as higher precision (fewer wrong triples survive the filter) at roughly
# equal recall. Measure it before/after the change with the existing harness:
#
#   1. Populate projects/<name>/eval/gold_template.tsv with real triples from a
#      reviewed run (rel_type, from_display, to_display, doc_name). Include a few
#      known-bad directional cases so the grader's catch is visible.
#   2. Run a fresh extraction, then:
#        python eval_extraction.py --gold projects/drug/eval/gold_template.tsv \
#            --run projects/drug/runs/<ts> --filter moderate --verbose
#        python eval_extraction.py ... --filter strict
#   3. Compare P/R/F1 before vs after. Precision up + recall flat = win;
#      recall drop = the grader is reding genuine triples (false positives) —
#      tighten the prompt before shipping.
