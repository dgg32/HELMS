"""Integration test: LLM agent picks the best (non-first) UMLS/GLEIF candidate.

Requires both UMLS_API_KEY and LLM_ENDPOINT + LLM_API_KEY in .env.

These tests verify that when umls_search() / gleif_search() return multiple candidates
with disambiguation metadata, the LLM agent selects the correct non-first hit —
the same decision the extraction agent makes during ambiguous resolution (step 3).
"""
import json
import os
import pytest

from lookups import gleif_search, umls_search


def _umls_available() -> bool:
    return bool(os.environ.get("UMLS_API_KEY"))


def _llm_available() -> bool:
    return bool(os.environ.get("LLM_ENDPOINT") and os.environ.get("LLM_API_KEY"))


def _both_available() -> bool:
    return _umls_available() and _llm_available()


_UMLS_SYSTEM = """\
You are a biomedical knowledge graph agent resolving ambiguous UMLS entity lookups.

Given a list of UMLS candidates (each has cui, name, semantic_types), pick the BEST
match for a Drug or Chemical node using these rules:
- Prefer the generic INN substance name over an abbreviation or code name.
- Prefer a single-substance entity over a combination product.
- Use semantic_types to distinguish:
    "Clinical Drug" = specific formulation (avoid when a generic exists)
    "Organic Chemical", "Pharmacologic Substance" = generic substance (prefer)
- If one hit is an abbreviation and another is the spelled-out INN name, prefer the INN.

Return ONLY the CUI string of the best candidate — no explanation, no punctuation."""

_GLEIF_SYSTEM = """\
You are a knowledge graph agent resolving ambiguous GLEIF legal entity lookups.

Given a list of GLEIF candidates (each has lei, name, status, category, jurisdiction,
registration_status), pick the BEST match for a Corporation node using these rules:
- Prefer category GENERAL over BRANCH, FUND, or SOLE_PROPRIETOR.
- Prefer status ACTIVE over INACTIVE.
- Prefer registration_status ISSUED over LAPSED.
- If names match exactly, prefer the GENERAL entity over any branch of the same name.

Return ONLY the LEI string of the best candidate — no explanation, no punctuation."""


def _llm_pick_umls(candidates: list[dict]) -> str:
    return _llm_pick(candidates, _UMLS_SYSTEM, "Return the CUI of the best candidate:")


def _llm_pick_gleif(candidates: list[dict]) -> str:
    return _llm_pick(candidates, _GLEIF_SYSTEM, "Return the LEI of the best candidate:")


def _llm_pick(candidates: list[dict], system: str, user_suffix: str) -> str:
    import litellm
    from llm_client import _get_litellm_model, _litellm_kwargs, _ensure_litellm_env
    _ensure_litellm_env()
    response = litellm.completion(
        model=_get_litellm_model(),
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Candidates:\n{json.dumps(candidates, indent=2)}\n\n{user_suffix}",
            },
        ],
        max_tokens=500,
        **_litellm_kwargs(),
    )
    return response.choices[0].message.content.strip()


@pytest.mark.skipif(not _both_available(), reason="UMLS_API_KEY or LLM_ENDPOINT/LLM_API_KEY not set")
class TestUmlsAgentSelection:
    """Verify the LLM picks the semantically best UMLS candidate, not the first-ranked one."""

    def _run(self, term: str, expected_cui: str, reason: str) -> None:
        r = json.loads(umls_search(term, "words"))
        assert "error" not in r, f"UMLS search failed: {r}"
        candidates = r["results"]

        print(f"\nQuery: {term!r}")
        print("Candidates:")
        for i, c in enumerate(candidates):
            marker = "← first (wrong)" if i == 0 and c["cui"] != expected_cui else ""
            print(f"  [{i}] {c['cui']}  {c['name']!r}  {c['semantic_types']}  {marker}")

        assert expected_cui in [c["cui"] for c in candidates], (
            f"Precondition: {expected_cui} not in candidates"
        )

        chosen = _llm_pick_umls(candidates)
        status = "✓ CORRECT" if chosen == expected_cui else "✗ WRONG"
        print(f"Agent chose: {chosen}  →  {status}  (expected {expected_cui} — {reason})")
        assert chosen == expected_cui, f"Agent chose {chosen!r}, expected {expected_cui}"

    def test_atropine_sulphate_picks_single_substance(self):
        """Agent must prefer C0596005 (atropine sulfate) over C0358790 (morphine+atropine combo)."""
        self._run("Atropine sulphate", "C0596005", "single substance over combination product")

    def test_abbv181_picks_budigalimab_inn(self):
        """Agent must prefer C4743556 (budigalimab INN) over C4527193 (ABBV-181 abbreviation)."""
        self._run("ABBV-181 (Budigalimab)", "C4743556", "INN generic name over abbreviation")

    def test_chlorhexidine_picks_generic_substance(self):
        """Agent must prefer C0055361 (chlorhexidine gluconate) over C5561554 (40 MG/ML formulation)."""
        self._run("Chlorhexidine Digluconate Solution", "C0055361", "generic substance over specific formulation")


@pytest.mark.skipif(not _llm_available(), reason="LLM_ENDPOINT/LLM_API_KEY not set")
class TestGleifAgentSelection:
    """Verify the LLM picks the best GLEIF candidate using status/category/jurisdiction."""

    def _run(self, query: str, expected_lei: str, reason: str) -> None:
        r = json.loads(gleif_search(query, "exact"))
        assert "error" not in r, f"GLEIF search failed: {r}"
        candidates = r["results"]

        print(f"\nQuery: {query!r}")
        print("Candidates:")
        for i, c in enumerate(candidates):
            marker = "← first (wrong)" if i == 0 and c["lei"] != expected_lei else ""
            print(f"  [{i}] {c['lei']}  {c['name']!r}  cat={c['category']}  "
                  f"status={c['status']}  reg={c['registration_status']}  {marker}")

        assert expected_lei in [c["lei"] for c in candidates], (
            f"Precondition: {expected_lei} not in candidates"
        )

        chosen = _llm_pick_gleif(candidates)
        status = "✓ CORRECT" if chosen == expected_lei else "✗ WRONG"
        print(f"Agent chose: {chosen}  →  {status}  (expected {expected_lei} — {reason})")
        assert chosen == expected_lei, f"Agent chose {chosen!r}, expected {expected_lei}"

    def test_barclays_picks_general_over_branch(self):
        """Agent must prefer G5GSEF7VJP5I7OUK5573 (GENERAL) over 2549003Q5J2G7ALNU316 (BRANCH)."""
        self._run(
            "Barclays Bank PLC",
            "G5GSEF7VJP5I7OUK5573",
            "GENERAL entity over BRANCH with same name",
        )
