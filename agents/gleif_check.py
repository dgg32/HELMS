#!/usr/bin/env python3
"""GLEIF resolution-suspicion check for the semantic check agent.

Pre-LLM: when a GLEIF-resolved name looks like a regional subsidiary of the
extracted term (e.g. 'Quanta' → 'QUANTA LYON'), inject a warning into the batch
item so the LLM downgrades that side to yellow instead of accepting it as a synonym.
This validator has no deterministic post-LLM verdict — the warning is advisory and
the LLM makes the call, so `check` is left at the inert default.
"""
from __future__ import annotations

import re

from .validators_base import TripleValidator, ValidatorContext

# Generic words that legitimately appear in company names after the main term.
# If the resolved GLEIF name = term + only these words, it is NOT suspicious.
# City/country names are NOT here — "Quanta Lyon", "Foxconn Singapore" → suspicious.
_GLEIF_FORM_WORDS = frozenset({
    # Legal forms
    "corporation", "corp", "incorporated", "inc", "limited", "ltd", "co",
    "company", "group", "holdings", "llc", "plc", "gmbh", "ag", "sa",
    "bv", "nv", "srl", "oy", "ab", "as", "asa", "kk", "pte", "pvt",
    "llp", "lp", "lllp", "and", "the",
    # Generic industry/function descriptors
    "industries", "international", "global", "solutions", "services",
    "technologies", "technology", "systems", "system", "computer",
    "computers", "semiconductor", "semiconductors", "manufacturing",
    "precision", "electronics", "electronic", "electric", "electrical",
    "industrial", "industry", "enterprises", "enterprise",
})

_PROMPT_FRAGMENT = """\
   - 'from_gleif_resolution_warning' / 'to_gleif_resolution_warning': the GLEIF lookup for \
the affected side resolved to a name with geographic or other non-legal qualifiers that suggest \
it may be a regional subsidiary rather than the intended company (e.g. 'Quanta' resolved to \
'QUANTA LYON'). For the affected side, set its color to YELLOW — this overrides the synonym \
match in step 1 (do NOT treat 'Quanta' ≈ 'QUANTA LYON' as a valid synonym when this warning \
is present). Only override to GREEN if the document explicitly names the resolved entity verbatim. \
Do NOT set constraint_violated=true for this warning alone."""


def _words(text: str) -> list[str]:
    """Tokenize for suspicion comparison.

    Slashes inside abbreviated legal suffixes (Danish/Norwegian 'A/S', 'K/S', etc.)
    are removed rather than treated as a word separator, so 'A/S' normalizes to the
    single token 'as' (already whitelisted in _GLEIF_FORM_WORDS) instead of
    splitting into the two stray, unwhitelisted tokens 'a' and 's'.
    """
    return re.sub(r"[^\w]", " ", text.replace("/", "").lower()).split()


def _gleif_name_suspicious(orig_term: str, resolved_name: str) -> str | None:
    """Return a warning string when resolved_name looks like a subsidiary of orig_term.

    Fires when: resolved_name starts with orig_term (case-insensitive) AND the
    trailing words are not all generic legal-form / industry descriptors — e.g.
    'Quanta' → 'QUANTA LYON' triggers (city name trailer), but 'TDK' →
    'TDK CORPORATION' does not (legal form), and abbreviations ('TSMC', 'HPE')
    are skipped entirely since they routinely expand to full names.
    """
    if not orig_term or not resolved_name:
        return None
    # Short all-caps terms are likely abbreviations that expand to full names — skip
    if orig_term.isupper() and len(orig_term) <= 6:
        return None
    orig_words = _words(orig_term)
    res_words  = _words(resolved_name)
    if len(res_words) <= len(orig_words):
        return None
    if res_words[: len(orig_words)] != orig_words:
        return None  # resolved name doesn't start with the extracted term
    extra = res_words[len(orig_words) :]
    if all(w in _GLEIF_FORM_WORDS for w in extra):
        return None  # only legal suffixes / generic descriptors — fine
    return (
        f"Extracted term '{orig_term}' resolved to '{resolved_name}'. "
        f"The extra words {extra!r} may indicate a regional subsidiary or a different "
        f"entity rather than the intended company."
    )


class GLEIFResolutionValidator(TripleValidator):
    """Warn when a GLEIF-resolved name looks like a subsidiary. Advisory (LLM decides)."""
    name = "gleif_resolution"
    prompt_fragment = _PROMPT_FRAGMENT

    def annotate_item(self, triple: dict, item: dict, ctx: ValidatorContext) -> None:
        for side, term_key, props_key in [
            ("from", "from_term", "from_props"),
            ("to",   "to_term",   "to_props"),
        ]:
            orig  = triple.get(term_key) or ""
            props = triple.get(props_key) or {}
            if props.get("lei"):  # GLEIF-resolved node
                warn = _gleif_name_suspicious(orig, props.get("name", ""))
                if warn:
                    item[f"{side}_gleif_resolution_warning"] = warn
