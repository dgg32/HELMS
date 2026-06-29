#!/usr/bin/env python3
"""Shared quote-grounding utilities for the extraction pipeline.

Both the batch path (`extract.py`) and the agentic path
(`agents/extraction_agent.py`) ground LLM-produced ``supporting_quote`` strings
against the source document, and the semantic-check agent re-locates quotes to
show the LLM their real surrounding context.  This module is the single
implementation they share.

Why token-based instead of substring matching
----------------------------------------------
LLMs do not copy quotes verbatim.  They wrap them in quote characters, swap
ASCII/curly quotes, elide spans with ``...``, drop inline ``([[N]] url)``
citations the markdown converter injected, and occasionally reword.  A raw
``quote in doc_text`` test therefore fails on benign paraphrase.

We tokenise both sides into normalised words that carry their character offsets
in the *original* document, then match the quote's word sequence against the
document's.  This:

  * survives markdown markers, surrounding quotes, and curly/ASCII quote swaps
    (each word is stripped + normalised individually);
  * skips inline citations (blanked length-preservingly before tokenising);
  * supports ``...`` elisions (the quote is split into fragments matched in
    order); and
  * returns the exact ``(start, end)`` span in the original text, so callers can
    re-anchor the stored quote to guaranteed-verbatim document text.

Public API
----------
``locate(quote, doc_text)``    -> ``(start, end)`` char span in ``doc_text`` or ``None``
``reanchor(quote, doc_text)``  -> verbatim ``doc_text[start:end]`` or ``None``
``context(quote, doc_text, w)``-> a ±``w`` char window around the located span or ``""``
"""
from __future__ import annotations

import re
import unicodedata

# Inline citations the markdown converter injects, e.g. "([[32]] wccftech.com)".
# The LLM quotes the surrounding prose but drops these, so we blank them
# (length-preservingly, to keep offsets valid) before tokenising the document.
_CITATION_RE = re.compile(r"\(\[\[\d+\]\]\s+\S+\)")
# Inline HTML tags the converter emits for styling (e.g. <u>…</u>, <sub>2</sub>);
# blanked length-preservingly so they don't fuse into adjacent word tokens.
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_WORD_RE = re.compile(r"\S+")

# Characters stripped from the ends of every token: markdown markers, brackets,
# punctuation, bullet glyphs, and both ASCII and curly quote characters.
_STRIP_CHARS = "*_#>~`-.,;:!?\"'()[]{}“”‘’…®™©•◦‣·"


def _fold(w: str) -> str:
    """NFC-normalise, lowercase, fold curly quotes to ASCII (length-preserving for our text)."""
    w = unicodedata.normalize("NFC", w).lower()
    return (w.replace("“", '"').replace("”", '"')
             .replace("‘", "'").replace("’", "'"))


def _norm_word(w: str) -> str:
    """Folded word with edge markup/quotes stripped and remaining quotes unified."""
    return _fold(w).strip(_STRIP_CHARS).replace('"', "'")


def _core_offsets(raw: str, start: int) -> tuple[str, int, int] | None:
    """Return ``(norm_word, core_start, core_end)`` where the offsets bound just the
    word core (edge markup excluded) within the original document.

    ``_fold`` is length-preserving for the documents we handle (NFC + lowercase +
    curly→ASCII are 1:1 here), so lead-strip length maps cleanly back to offsets.
    """
    folded = _fold(raw)
    stripped = folded.strip(_STRIP_CHARS)
    if not stripped:
        return None
    if len(folded) != len(raw):  # rare length-changing fold — skip precise offsets
        return None
    lead = len(folded) - len(folded.lstrip(_STRIP_CHARS))
    core_start = start + lead
    core_end = core_start + len(stripped)
    return stripped.replace('"', "'"), core_start, core_end


def _blank_citations(text: str) -> str:
    """Blank ``([[N]] url)`` citations and inline HTML tags with equal-length spaces.

    Length-preserving so the document character offsets stay valid for re-anchoring.
    """
    text = _CITATION_RE.sub(lambda m: " " * len(m.group()), text)
    return _HTML_TAG_RE.sub(lambda m: " " * len(m.group()), text)


def _despace_tables(text: str) -> str:
    """Turn markdown table pipes into spaces so table cells tokenise as separate words.

    Rows are written with tight pipes (``|Dyspepsia|4.3|4.7|``) which otherwise
    captured as a single ``\\S+`` token that no quote could match. Replacing ``|``
    with a space is length-preserving, so document offsets stay valid for re-anchoring.
    """
    return text.replace("|", " ")


# A missing space after a sentence-ending period fuses two words into one token
# (a common PDF→Markdown artifact: "...to TSMC.On the other..." → token "tsmc.on"),
# so a quote that begins right after the period ("On the other...") cannot match.
# Replace the offending period with a space — length-preserving, so document offsets
# stay valid — but ONLY for a true run-on: a letter, a period, then a Capitalized
# word ([A-Z][a-z]). This spares dotted acronyms ("U.S."), decimals ("3.5"),
# "etc.)", and lowercase domains ("Amazon.com").
_RUNON_PERIOD_RE = re.compile(r"(?<=[A-Za-z])\.(?=[A-Z][a-z])")


def _split_runon_periods(text: str) -> str:
    """Length-preservingly turn a run-on sentence period ("TSMC.On") into a space."""
    return _RUNON_PERIOD_RE.sub(" ", text)


def _doc_tokens(doc_text: str) -> list[tuple[str, int, int]]:
    """Return ``(norm_word, start, end)`` for each non-empty normalised token.

    ``start``/``end`` are offsets into the ORIGINAL ``doc_text`` (citations and
    table pipes are blanked length-preservingly first, so offsets stay aligned).
    """
    cleaned = _split_runon_periods(_despace_tables(_blank_citations(doc_text)))
    toks: list[tuple[str, int, int]] = []
    for m in _WORD_RE.finditer(cleaned):
        core = _core_offsets(m.group(), m.start())
        if core is not None:
            toks.append(core)
        else:  # length-changing fold: fall back to raw token span
            nw = _norm_word(m.group())
            if nw:
                toks.append((nw, m.start(), m.end()))
    return toks


def _quote_fragments(quote: str) -> list[list[str]]:
    """Split a quote on ``...``/``…`` ellipses into lists of normalised words."""
    frags: list[list[str]] = []
    for part in re.split(r"\.\.\.|…", quote):
        words = [w for w in (_norm_word(x) for x in part.split()) if w]
        if words:
            frags.append(words)
    return frags


# Structural delimiters the LLM commonly elides ACROSS without a "..." marker:
# markdown emphasis runs that wrap section headers (``_Metabolic System:_``),
# colons that introduce a list, and explicit ellipses. Splitting on them recovers
# each contiguous run when the LLM quotes a list intro then jumps to a later item.
_STRUCT_SPLIT_RE = re.compile(r"\.\.\.|…|[_*]{1,2}|:")


def _struct_fragments(quote: str) -> list[list[str]]:
    """Split a quote on emphasis/colon/ellipsis boundaries into in-order word lists.

    Fallback for *silent* elisions: e.g. ``...include: _Metabolic ...:_ ...,
    hyperglycemia`` where the LLM dropped intervening list sections without ``...``.
    Each returned fragment must still match the document contiguously and in order.
    """
    frags: list[list[str]] = []
    for part in _STRUCT_SPLIT_RE.split(quote):
        words = [w for w in (_norm_word(x) for x in part.split()) if w]
        if words:
            frags.append(words)
    return frags


def _match_fragments(
    dtoks: list[tuple[str, int, int]], dwords: list[str], frags: list[list[str]]
) -> tuple[int, int] | None:
    """Match ``frags`` contiguously and in order against the doc tokens.

    Returns the ``(start, end)`` span from the first fragment's start to the last
    fragment's end, or ``None`` if any fragment fails to match at/after the cursor.
    """
    cursor = 0
    span_start: int | None = None
    span_end: int | None = None
    for frag in frags:
        idx = _find_contiguous(dwords, frag, cursor)
        if idx == -1:
            return None
        if span_start is None:
            span_start = dtoks[idx][1]
        span_end = dtoks[idx + len(frag) - 1][2]
        cursor = idx + len(frag)
    if span_start is None or span_end is None:
        return None
    return (span_start, span_end)


def _find_contiguous(dwords: list[str], frag: list[str], start: int) -> int:
    """Return the doc-token index where ``frag`` matches contiguously at/after ``start``, or -1."""
    n = len(frag)
    if n == 0:
        return -1
    for i in range(start, len(dwords) - n + 1):
        if dwords[i : i + n] == frag:
            return i
    return -1


def locate(quote: str, doc_text: str) -> tuple[int, int] | None:
    """Locate ``quote`` in ``doc_text``; return its ``(start, end)`` char span or ``None``.

    Matching is token-based and tolerant of markdown, quote-character swaps, and
    inline citations.  ``...`` elisions are honoured: each fragment must appear in
    order, and the returned span runs from the first fragment's start to the last
    fragment's end.
    """
    if not quote or not quote.strip():
        return None
    dtoks = _doc_tokens(doc_text)
    if not dtoks:
        return None
    dwords = [t[0] for t in dtoks]
    # Match the doc side: table pipes are spaces there, so drop them from the quote
    # too (offsets returned are doc offsets, so cleaning the quote is harmless).
    # Apply the SAME normalisations as _doc_tokens, in the same order, so both sides
    # tokenise alike. Blanking citations on the quote too is essential: the doc side
    # blanks "([[N]] url)" away, so a quote that includes a citation would otherwise
    # carry tokens ("wccftech.com") the doc no longer has, breaking the contiguous match.
    quote = _split_runon_periods(_despace_tables(_blank_citations(quote)))
    frags = _quote_fragments(quote)
    if not frags:
        return None

    span = _match_fragments(dtoks, dwords, frags)
    if span is not None:
        return span

    # Strict contiguous match failed. Retry with finer fragmentation on the
    # structural boundaries the LLM elides across without a "..." marker (list
    # intros, italic section headers). Only worth trying when those delimiters
    # actually produce more fragments than the ellipsis split did.
    sfrags = _struct_fragments(quote)
    if len(sfrags) > len(frags):
        return _match_fragments(dtoks, dwords, sfrags)
    return None


def reanchor(quote: str, doc_text: str) -> str | None:
    """Return the verbatim ``doc_text`` substring the quote matches, or ``None``.

    Use to overwrite an LLM-produced ``supporting_quote`` with guaranteed-verbatim
    document text, eliminating downstream "quote not in doc" drift.
    """
    span = locate(quote, doc_text)
    if span is None:
        return None
    return doc_text[span[0] : span[1]]


def context(quote: str, doc_text: str, window: int = 200) -> str:
    """Return a ±``window``-char slice of ``doc_text`` around the located quote, or ``""``.

    Lets the semantic-check agent show the LLM the quote's real neighbourhood even
    when the quote sits outside the truncated document excerpt.

    Clamped to the nearest blank-line boundary (paragraph / markdown list-item break)
    on each side, so the window never bleeds into an unrelated neighbouring bullet,
    paragraph, or table that happens to name a different entity. Without this, a
    blind char-count window can hand the semantic-check LLM a stray entity mention
    from an adjacent, unrelated list item and make it look like grounding.
    """
    span = locate(quote, doc_text)
    if span is None:
        return ""
    start = max(0, span[0] - window)
    end = min(len(doc_text), span[1] + window)
    left_break = doc_text.rfind("\n\n", start, span[0])
    if left_break != -1:
        start = left_break + 2
    right_break = doc_text.find("\n\n", span[1], end)
    if right_break != -1:
        end = right_break
    return doc_text[start:end]
