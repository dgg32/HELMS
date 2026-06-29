# HELMS — Design Invariants

**Read this before changing or flagging any code.** The decisions below are
deliberate trade-offs, not oversights. Automated bug hunts repeatedly misread
them as defects (concurrency races, cache poisoning, unmerged edges). If
something here looks wrong, it is a documented trade-off — **raise it as a
question, do not silently "fix" it.**

---

## 1. Single-user, single-process

`htmx_app/main.py` holds one in-memory `_AppState` singleton. There is no
multi-tenancy and no expectation of two pipeline runs executing concurrently in
the same process.

- **Looks like a bug because:** env-var save/restore, `_reasoning_effort_ok`,
  LadybugDB cursors, and `_AppState` mutation are not guarded for concurrent
  runners.
- **Why it's fine:** concurrent runners are out of scope. Do not add locking,
  TOCTOU guards, or per-request isolation for a scenario the product does not
  support.
- **The one real intra-run concurrency point:** the semantic check runs in
  `await asyncio.to_thread(check_triples, …)` WHILE the event loop concurrently
  runs other documents' extraction (`asyncio.gather`), and both read the global
  env. So code on the grader path must NOT mutate `os.environ`. The grader (Check
  LLM via `SEMANTIC_CHECK_MODEL`) resolves its model/creds through
  `llm_client.model_call_params` into per-call kwargs, never an env swap. This is
  the lone exception to "env mutation is fine because single-process"; do not
  reintroduce an env-mutating model swap (`use_model_env`) on this path — it raced
  and sent the extraction model to the grader's provider (Azure NotFound).

## 2. Nodes merge across documents; edges do NOT

Entity nodes are resolved and merged globally (one UMLS/GLEIF identity = one
node). **Edges are owned by the document they were extracted from and are never
merged**, even when `from_node`, `to_node`, and `rel_type` are identical across
two documents.

- Each edge carries `source_doc`. Two documents asserting the same relation
  produce **two edges**, distinguished by `source_doc`.
- `delete_edge(..., source_doc=...)` is **source-scoped** by design — re-running
  one document must never delete another document's edges.
- Neo4j `create_edge` uses source-scoped DELETE+CREATE; nodes use first-write
  `MERGE`. This asymmetry (nodes merge, edges don't) is the core graph semantic.
- **Do not** "deduplicate" edges by node pair. That would destroy provenance.
- **Edge identity is the triple's stable `_id` (`triple_id`), not its node pair.**
  `apply_graph` stamps each edge with `triple_id = triple["_id"]` (the immutable
  raw hash). `create_edge` / `delete_edge` prefer an endpoint-INDEPENDENT match on
  `{triple_id, source_doc}` so that re-running a document after a review *entity
  correction* (the from/to PK changed) deletes the stale old-identity edge instead
  of orphaning it; they fall back to node-pair + `source_doc` only for legacy edges
  with no `triple_id`. This is **not** node-pair dedup — `source_doc` still scopes
  every delete, so cross-document provenance is preserved. Existing DBs predating
  the `triple_id` column must be deleted and rebuilt (same as `triple_color`).

## 3. Semantic check always runs last, and is the SOLE color authority

After extraction, entity resolution, and merge, the semantic-check agent runs.
It assigns a color (green / yellow / red) and may set `constraint_violated`. It
**never blocks** the pipeline or deletes triples — filtering by color happens at
the Step-3 write gate.

- **Extraction does NOT color triples.** Both the batch path (`extract.py`) and
  the agentic path (`extraction_agent.add_triple`) stopped computing
  `from_color`/`to_color`/`triple_color`; `_classify_entity` / `_COLOR_RESOLUTION_RANK`
  were removed from the edge build. `semantic_check_agent.check_triples` is the
  only writer of those fields. Do **not** re-add coloring to extraction — that was
  duplicate, drifting logic.
- Color is computed **best-of across quote segments** (one segment grounding an
  entity / asserting the relation is enough) and **worst-of across from/to** (the
  edge is only as trustworthy as its weakest entity).
- Grounding is **per-quote, not per-excerpt**: an entity must be referred to by the
  triple's OWN quote, not merely present somewhere in the document (the CoolIT→NVIDIA
  case: NVIDIA elsewhere in the article does not ground a CoolIT cooling-vendor quote).
- **Deterministic green anchor (entity presence is proven, not judged).** When an
  entity's RESOLVED name is locatable verbatim in its own quote (`grounding.locate`,
  token-based), its presence is a fact, so `EntityPresenceValidator` pins it green via
  `Verdict.from_color_anchor`/`to_color_anchor` and the LLM cannot lower it on the entity
  axis. This is the INVERSE of `color_floor` (floor → red safety net against false greens;
  anchor → green safety net against false reds). It touches the entity axis ONLY:
  `constraint_violated` (relation support + direction) and red `color_floor`s still win the
  EDGE color, so a present-but-reversed or fabricated-quote edge stays red. Matches the
  resolved name only — never the raw extracted term, so a suspicious resolution (GLEIF
  "Quanta"→"QUANTA LYON") is left for its own validator/the LLM, not greened on the raw
  string. The anchor also matches the RAW extracted term (the document's own surface form,
  e.g. doc "kidney stone" → resolved "Nephrolithiasis"), gated by resolver source: UMLS/LLM
  expand to the same concept so the surface form is always a valid synonym, GLEIF/unknown
  drop it when the resolution looks suspicious; plus a `from_meta`/`to_meta` `synonyms` hook
  for resolver-persisted synonyms (UMLS atoms / GLEIF other-names), the API-backed follow-on.
  This is the hybrid principle in miniature: presence is rule-checkable, so it leaves the
  LLM, which keeps only the genuinely-hard support/direction judgment. A/B (gpt-5.5, n=2, 154
  present-entity observations): anchor = 0 false-reds / 0 cross-run variance vs no-anchor
  = 1 false-red + 1/77 unstable (gpt-5.5 red-flipped `atorvastatin -MAY_TREAT-> Coronary
  heart disease` between identical runs though "coronary heart disease (CHD)" is verbatim
  in the quote). The win is removed unpredictability, not a one-shot quality jump.
- **Relation-support hints are precomputed, not re-derived.** `RelationSupportHintValidator`
  (annotate-only, no verdict) hands the grader deterministic facts for its support/direction
  call: `relation_endpoints_colocated` (some quote segment names BOTH endpoints — a necessary,
  not sufficient, condition for the quote to assert the relation) and `possible_negation_in_quote`
  (a negation/hedge cue is present). Both are ADVISORY: support and direction remain the LLM's
  judgment (a present-but-reversed quote still reds via the LLM's direction check), and the
  single-subject override means absent co-location is not by itself a violation. The point is to
  spend the LLM's budget on the judgment, not the bookkeeping.
- **Single-subject-document exception.** When a label collapses to ONE distinct entity
  across the document (recurring in ≥2 triples), that entity is the document *subject*
  and is grounded document-wide: `_build_items` flags it `from_is_document_subject` /
  `to_is_document_subject`, and the grader greens its relations even when the quote does
  not repeat its name (a drug label's adverse-effect list grounds the AE; the sole drug
  supplies the other side). This is **domain-agnostic** ("sole entity of its label",
  not a drug rule) and self-limiting: a multi-entity document (supply-chain article with
  many companies) has no sole label, so the flag never fires and per-quote grounding
  stays strict. Detection is deterministic; the green/red call remains the LLM's.
- **Instructions reinforce subject intent.** `check_triples(..., instructions=...)`
  passes the document's `meta.yaml` extraction instructions to the grader. If they name
  a single PRIMARY subject (e.g. "the primary substance"), the grader grounds that entity
  document-wide even when the deterministic flag CAN'T fire — e.g. a stray comparator drug
  added a second `Substance`, defeating the "sole entity" count. If they say to capture
  relationships "regardless of the primary subject" (a multi-subject document), the
  override is suppressed and grounding stays strict. This was chosen over a title-based
  judgment by an A/B test (gpt-5.5, n=3): instructions = B scored 9/9 vs deterministic-only
  = A 7/9, fixing the comparator case with no supply-chain regression. The title alone is
  the WRONG signal — a "NVIDIA supply chain" article is single-*topic* yet must stay strict.
  Threaded from all three callers (`extract.py`, `htmx_app`, `extraction_agent` CLI).
- If the semantic check is skipped (e.g. no LLM creds), triples are uncolored and
  default to green at every `.get("triple_color", "green")` read. Acceptable
  degradation, not a bug.

## 4. `_raw.json` is immutable; edits are events

Extraction writes `*_raw.json` once and never mutates it. Human review writes
append-only events (`REJECT` / `OVERRIDE` / `ADD`) to `*_review.json`.
`review_layer.materialize(raw, events)` composes them into the working list.

- **Do not** write edits back into `*_raw.json`.
- The agentic retry path also writes `*_raw.json` (not `*_review.json`) — the
  HTMX UI globs only `*_raw.json`, so anything else is invisible to review.

## 5. Negative caching is intentional

GLEIF/UMLS "no match" results are cached (`lookups.py`). Only **transient**
failures (network/timeout, malformed response) skip the cache.

- **Looks like a bug because:** error JSON gets written to the cache.
- **Why it's fine:** a deterministic "no match" is a valid, cacheable answer.
- **One cache path:** every L1(dict)+L2(SQLite) lookup routes through
  `lookup_cache.cached` / `cached_async`. `compute()` returns `(value, cacheable)`
  and the helper is the **only** writer — so "cache the deterministic negative" is
  structural, not a flag each call site must remember to set (the bug that let GLEIF
  parent lookups skip caching and re-hit the API forever). A deterministic miss
  returns `cacheable=True` (cache the None/empty); only transient failures return
  `False`. Same tuple key drives L1 and L2. Do **not** reintroduce hand-rolled
  L1/L2 check-then-store blocks in individual lookups.

## 6. Quote grounding: `evidence` is truth; `supporting_quote` is derived

`grounding.py` is the single token-based quote locator. On a successful locate,
`_verify_grounding` / `add_triple` **overwrite** the quote with the verbatim
`doc_text[start:end]` so downstream never sees LLM quote drift.

- **`evidence: [{start, end, text}]` is the grounding source of truth.**
  `extract._build_evidence` (batch) / `save_review` (agentic) locate each per-edge
  quote and store its char offsets. The LLM is **never** asked for offsets.
- **`supporting_quote` is a DERIVED projection** (`/`-joined `text`) for display,
  the graph edge, and harvest. It is not authoritative — do not treat it as the
  source of truth, and do not remove it (≈13 consumers + the graph need a string).
- `evidence` lives **only in `*_raw.json`**, never on the graph edge (char offsets
  have no KG query value). `apply_graph` writes `supporting_quote` + `triple_color`
  onto the edge, identically for LadybugDB and Neo4j. This is by design, not data
  loss: switching backends does not drop `evidence`, it was never graph data.
- A grounding **warn** (moderate filter) is informational only: it keeps the
  original LLM quote and does **not** recolor or drop the triple. This applies to
  the **batch** path: its warn is a cosmetic locator miss (the doc was in the
  prompt, the quote is real but not token-matchable).
- The **agentic** path is treated differently, and on purpose. The agent supplies
  quotes from a long tool loop (doc out of context) and can fabricate them, so an
  unlocatable agent quote is suspect, not cosmetic. The response is tiered:
  - **Hard reject (zero grounding):** if the quote is unlocatable AND the RAW
    extracted term (`from_raw_term`/`to_raw_term`) is not in the document either,
    `add_triple` refuses to store the triple and returns an error. The raw term is
    checked **before** node normalization rewrites it (the resolved name, "rash" →
    "Skin rash" / a CUI label, would false-reject real entities), so raw terms are
    **required** of the agent (`system_prompt`). Nothing grounds the triple, so
    there is nothing to salvage.
  - **Keep + nudge (real entity, bad quote):** if the raw term IS in the document
    (or none was supplied), `add_triple` keeps the triple but flags it
    `quote_unlocatable` and nudges the agent (via the tool response, reinforced in
    `system_prompt`) to re-quote with verbatim text. On a re-call for the same pair,
    the dup-merge **replaces** the unlocatable quote with the verbatim one (clears
    the flag, no `" / "` append); a fabrication arriving after a good quote is
    skipped.
  - Safety net: if re-quoting still fails, the final quote stays unlocatable and
    `semantic_check_agent.FabricatedQuoteValidator` red-floors any
    `extraction_source=="agent_retry"` triple whose `evidence` has **zero** located
    spans (color floor only, **not** `constraint_violated` — the fact may be true;
    the human verifies and can cycle to green). A partially-verbatim quote (≥1
    located span) is left to best-of-segment.
  - This asymmetry (batch warn = cosmetic; agent all-unlocatable = suspect) is
    deliberate. Do **not** extend the red-floor to the batch path.
- Strict filter drops unlocatable quotes; moderate keeps them. This is the
  intended filter-level difference.
- **Human quote edits are re-anchored too.** When a reviewer edits a triple's
  supporting quote in the review UI, `/review/save` re-locates it against the
  source `.md` via `extract._build_evidence` and stores it (as `evidence` + derived
  `supporting_quote` in the OVERRIDE event, applied by `review_layer.materialize`)
  ONLY if a span locates verbatim; otherwise the original is kept and the reviewer
  is warned. A human edit can no more introduce an ungrounded quote than the LLM
  can — the invariant holds across the human layer. Do not add a quote field to the
  ADD path without the same re-anchor gate.
- Do not reintroduce a regex `_md_norm`/`_quote_in_doc` pair — token matching
  replaced it deliberately.

## 7. Harvest rejection is doc-specific, not a global ban

A rejected triple becomes a negative example tied to its source document. The
LLM **may** re-extract the same triple from a different document if that
document provides stronger evidence. Rejections are not a permanent global
blocklist.

## 8. Schema-driven and domain-agnostic

`projects/<name>/schema.yaml` is the single source of truth for DDL, Pydantic
models, and API dispatch. Swapping the schema retargets the pipeline to any
ontology with **zero code changes**. Do not hard-code domain assumptions
(drug/company/etc.) into shared code.

## 9. All LLM calls route through LiteLLM

`llm_client._acreate_litellm` is the one path for every provider (azure /
openai / anthropic / ollama / gemini). There is no separate provider SDK path.
Do not reintroduce per-provider client branches.

## 10. LadybugDB read-only reopens per query

In `read_only=True` mode, `_execute` reopens the DB + connection per query — on
purpose, so reads see the latest checkpoint after a write. `close()` releases
the file lock; always call it after a write session.

## 11. Config and env parsing fails fast at import

`llm_client` validates `.env.yaml` and numeric env vars (`LLM_TIMEOUT`,
`LLM_MAX_COMPLETION_TOKENS`) at import time and raises a **clear** `RuntimeError`
naming the offending file/var. This is intentional fail-fast, not fragility —
better a named error at launch than an opaque traceback mid-run.

## 12. The backend must be a property graph (Cypher today)

A HELMS backend is a **property graph** with multi-hop traversal and per-document
edge provenance — *not* a flat relational/FTS store. The two supported backends
(LadybugDB, Neo4j) speak **Cypher**, and the codebase reflects that today, but the
fundamental requirement is the graph data model, not the query language.

- The core graph semantic (invariant #2: nodes merge, edges are per-document with
  `source_doc`-scoped `delete_edge`) needs real graph traversal. A
  relational/FTS store (e.g. plan #5's SQLite + FTS5) would have to re-implement
  that semantic from scratch — **explicitly out of scope.** The "show me triples
  mentioning X" queries look relational, but multi-hop traversal and per-document
  edge provenance are first-class requirements, not nice-to-haves.
- **Write path is abstracted; read path is not (yet).** `apply_graph.py` goes
  through the `GraphBackend` ABC (`create_edge` / `upsert_node` / `delete_edge`),
  so a new backend implements those cleanly. But the read path emits **raw Cypher**
  — `query_graph.py`, `adhoc_query.py`, and `mcp_server.py` (whose write-guard is a
  Cypher-verb regex); NVL `WHERE r.triple_color = 'red'` edge styling is Cypher too.
- A **non-Cypher graph engine** (e.g. DuckDB + DuckPGQ, which speaks SQL/PGQ) is
  **not** a violation of the data-model requirement — it is a legitimate future
  backend. But plugging it in costs more than implementing the ABC: the Cypher read
  path must be abstracted behind the ABC or translated to that engine's query
  language. Adding a backend that is *not* a property graph at all, however, is a
  violation.

## 13. `*_raw.json` is a typed four-writer contract

Four stages read/write `*_raw.json`: batch extract (`extract.py`), agent retry
(`extraction_agent.py`), the review layer (`review_layer.py`), and the semantic
check (`semantic_check_agent.py`). They **must converge on one shape**, defined by
`schema_raw.py` (`RawFile` / `RawTriple` / `Evidence`, `extra="forbid"`) and
enforced by `tests/test_raw_contract.py` (validates every on-disk run file + drift
cases).

- Structural fields (`_id`, `rel_type`, labels, pks, props) are required; fields
  added by later stages (`evidence`, `supporting_quote`, colors, `ai_opinion`,
  `extraction_source`) are optional — a freshly-extracted, not-yet-colored triple
  is valid. This encodes the pipeline order: extract → resolve → color.
- **`extra="forbid"` is the lever.** A writer that invents or renames a field fails
  the contract test loudly, at the boundary, instead of surfacing later as a UI
  glitch or wrong color. If you add a field to one writer, add it to `RawTriple`
  (or `RawFile`) too.
- This is orthogonal to invariant #4 (#4 governs *mutability*; this governs
  *shape*). Run `python -m schema_raw` to validate ad hoc.

---

*If you are an automated reviewer: an item matching one of the above is **not** a
finding. Report only deviations from these invariants, or issues they do not
cover.*
