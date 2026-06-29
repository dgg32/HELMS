# Schema-Driven Knowledge Graph from Documents

A pipeline that extracts a typed knowledge graph from PDF documents, validates
entities against external registries, persists the graph in LadybugDB or Neo4j,
and exposes it to AI clients via an MCP server with graph queries and
just-in-time document context retrieval. 

HELMS (Human, LLM, and Safeguards) weaves three threads into a single fabric: a
human-authored schema and review loop, an LLM that reads the documents and
proposes candidate facts, and a Safeguards layer (entity resolution against
authoritative registries plus a deterministic-rules/LLM hybrid semantic check)
that grades every triple before it reaches the graph. Pull any one thread and
the fabric unravels into a specific failure mode: no LLM means manual
extraction, no human means an ungrounded, self-invented schema, no Safeguards
means unverified hallucinations entering the graph.

For more description, please read
[https://dgg32.medium.com/helms-guided-and-grounded-knowledge-graphs-fb55daf0c955?postPublishedType=repub].

> **Contributors & AI agents:** read [DESIGN_INVARIANTS.md](DESIGN_INVARIANTS.md)
> before changing or reviewing code — it lists deliberate design choices that are
> commonly misread as bugs. [AGENTS.md](AGENTS.md) is the cross-tool agent entry point.

---

## Architecture

```
PDFs
 │
 ├──► convert_pdf.py  ── pymupdf4llm  (default, fast, CPU-only)
 │                        LlamaParse   (cloud API, best accuracy; requires LLAMA_CLOUD_API_KEY)
 │    outputs .md sidecars next to each PDF
 │
 ▼
extract.py  ──────────── single LLM call per chunk (open-minded, high recall)
 │  accepts .pdf or .md   LLM assigns evidence_level per triple
 │                        semantic chunking (header-aware)
 │
 ├──► grounding.py        — token-based quote locator; _verify_grounding re-anchors each
 │        quote to verbatim doc text (tolerant of markdown / quote-char swaps /
 │        citations / "..." elisions); _build_evidence stores [{start,end,text}] spans
 │        (supporting_quote derived from them); shared by batch + agentic paths
 │
 ├──► agents/node_agent.py  — batch UMLS + GLEIF resolvers (asyncio concurrent; ambiguous
 │        entities disambiguated in ONE batched LLM call, not one call per entity)
 │        logs failures → _node_agent_errors.jsonl; runs once per doc after extraction
 │
 ├──► agents/semantic_check_agent.py  — always-on after node agent; SOLE color authority
 │        (extraction no longer colors); best-of-segment grounding, worst-of from/to;
 │        attaches verbatim quote_context for quotes beyond the excerpt window
 │
 └──► <stem>_raw.json    — immutable LLM output (triples + full doc_text)
      <stem>_review.json — event log of human decisions (REJECT / OVERRIDE / ADD)

 ▼  (after human review / edit)

agents/harvest.py  ────── rebuilds <project>/harvest/<rel_type>.jsonl after each Step 3
 │  priority tiers: ADD(1) > OVERRIDE(2) > agent_retry(3) > batch(4)
 │  rejected triples stored as negative examples (doc-specific, not a global ban)
 │  next extraction run: positives → system prompt; rejections → per-chunk user message
 │  (highest-recency position — model sees them immediately before document text)

 ▼  (after human review / edit)

apply_graph.py  ─────── reads _raw.json + _review.json via review_layer.materialize()
 │
 └──► GraphBackend (backends/)  — pluggable graph persistence layer
       ├── LadybugBackend          — default; typed graph (openCypher)
       └── Neo4jBackend         — bolt, via --backend neo4j

pipeline.py — one-command orchestrator: convert_pdf.py → extract.py → apply_graph.py
htmx_app/main.py — web UI (FastAPI + HTMX): stepper interface for all three steps + review editor + Semantic Check

agents/extraction_agent.py       — on-demand agentic retry (HTMX UI Step 2 🥷🏻 Agent Retry, plus the CLI `--rel-type` path): LLM calls gleif_search/umls_search/add_triple tools iteratively for one rel_type in one document; adapts UMLS searchType (words→normalizedWords→normalizedString) per term; honors schema umls_vocabs (sabs), sem_group, and semantic_types constraints per node; auto-saves partial triples on 60-iteration limit; grounding guard (tiered): when `grounding.reanchor` fails (quote not in doc), `add_triple` **hard-rejects** the triple if the RAW extracted term (`from_raw_term`/`to_raw_term`, checked pre-normalization — raw terms are required of the agent) is also absent from the doc (zero grounding = invention); otherwise it keeps the triple, flags it, and nudges the agent to re-quote (reinforced in the system prompt), and a re-call replaces the unlocatable quote with the verbatim one (no append); if retry still fails the post-merge semantic check red-flags it (`FabricatedQuoteValidator`); `_cli_semantic_check()` (CLI) / `_run_agent_semantic_check()` (HTMX) auto-runs semantic check on the retried rel_type triples after merge — adds ai_opinion, recolors, flags constraint_violated in-place
agents/node_agent.py             — batch UMLS + GLEIF resolvers: `asyncio.gather` + `asyncio.to_thread`, page_size=25, cascade search (words→normalizedWords→normalizedString); exact-name shortcut skips LLM when a candidate matches the raw name exactly (case-insensitive) — guarded by semantic type validation when sem_group/sem_types active (UMLS server-side TUI filtering can be incomplete), falls through to LLM if type constraint not satisfied; **batched LLM pick**: each resolver's `resolve_batch` runs a Phase A (cache check + candidate gather + no-LLM prefilter, concurrent) then a Phase B that disambiguates ALL ambiguous entities in one batched structured LLM call (≤12 entries/call) instead of one call per entity — ~Nx fewer pick calls; LLM may return index=0 to reject all candidates (logged to _node_agent_errors.jsonl); retry-once on server error
agents/semantic_check_agent.py   — LLM semantic grounding: batches triples (25/batch), colors each entity green/yellow/red, adds `ai_opinion`; deterministic structural checks (empty PKs, duplicates, procedure-marker mismatch, UMLS semantic type mismatch); semantic type mismatch check runs at all filter levels when schema defines `semantic_types`; expected + actual types injected into LLM batch so LLM opinion mentions mismatches; **sole color authority**: extraction no longer colors triples (both batch and agentic paths stopped calling `_classify_entity`); this agent is the only writer of `from_color`/`to_color`/`triple_color`; **evidence-driven `quote_verbatim`**: each triple carries `evidence: [{start,end,text}]` spans (offsets located deterministically at extraction time, the LLM is never asked for positions); `_build_items` sets `quote_verbatim=true` only when EVERY span located, and the prompt tells the LLM to trust it and stop re-judging presence (no more false-yellow on quotes outside the 60k excerpt); a legacy fallback splits `supporting_quote` on `" / "` for runs predating the field; **relation support + direction check**: LLM judges whether the quote actually asserts `from --rel--> to` in this direction (a reversed or merely-co-mentioning quote sets `constraint_violated=true` → red); **best-of-segment**: for a `/`-joined multi-segment quote, one segment grounding an entity colors it green and one segment asserting the relation makes it supported, so a single bad segment no longer reds a true triple; **quote_context injection**: when a span falls outside the 60k excerpt window, the surrounding doc text (sliced straight from the stored offsets) is attached as `quote_context` so the LLM grounds the direction/support judgment against real context; **harvest rejection check**: after LLM verdict, any triple matching a rejected harvest entry is forced red regardless of LLM result — works on cache hits; always-on — runs automatically after node agent in `extract.py`, and after every agent retry (CLI and HTMX)

Graph DB
 │
 ├──► query_graph.py            — CLI dump of all nodes / edges
 └──► mcp_server.py             — MCP server (Claude Desktop, Kilo Code, VS Code)
                                   tools: run_cypher (graph) + list_documents /
                                          read_document / search_document (JIT doc context)
```

---

## Tech stack

| Component | Library / Service |
|-----------|------------------|
| Graph database | [LadybugDB](https://ladybugdb.com/) (embedded openCypher) |
| LLM | Azure OpenAI (default) · OpenAI · OpenAI-compatible gateways (OpenCode Go / OpenRouter / vLLM) · Anthropic · Ollama · Gemini — set `LLM_PROVIDER` env var |
| Structured output | All providers via LiteLLM (`litellm.acompletion` + `response_format` + Pydantic `model_validate_json`) — unified path, no provider-specific branching |
| MCP server | [FastMCP](https://github.com/jlowin/fastmcp) |
| PDF parsing | [pymupdf4llm](https://pypi.org/project/pymupdf4llm/) (default) · [LlamaParse](https://developers.llamaindex.ai/llamaparse/) (cloud API) |
| Entity validation | GLEIF API (LEI records), UMLS API |

---

## Setup

### 1. Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure credentials

**Option A — `.env.yaml` (recommended for multi-model setups)**

Create `.env.yaml` in the project root (git-ignored). This file drives the model-selector dropdown in the web UI. All-caps scalar keys are loaded as env vars; the `models:` block registers named deployments the user can switch between at runtime.

```yaml
# Global credentials (apply to all models unless overridden per model)
LLM_API_KEY: "<azure-openai-api-key>"
LLM_ENDPOINT: "https://<your-resource>.cognitiveservices.azure.com/openai/deployments/<deployment>"
LLM_API_VERSION: "2025-04-01-preview"
LLM_MAX_COMPLETION_TOKENS: 65536   # 65536 recommended for gpt-5.5 / o-series
LLM_REASONING_EFFORT: "low"        # low / medium / high — ignored by non-reasoning models
LLM_TIMEOUT: 120

# Models shown in the sidebar LLM Config dropdown (first entry is the default)
# Each key is a deployment name; the value dict overrides the global credentials for that model
models:
  gpt-5.5:
    LLM_ENDPOINT: "https://<resource>.cognitiveservices.azure.com/openai/deployments/gpt-5.5"
    LLM_API_KEY: "<key-for-gpt55>"
  gpt-4o:
    LLM_ENDPOINT: "https://<resource>.cognitiveservices.azure.com/openai/deployments/gpt-4o"
    LLM_API_KEY: "<key-for-4o>"
  # OpenAI-compatible gateway (OpenCode Go / OpenRouter / vLLM / …): set
  # LLM_PROVIDER: openai_compatible, the gateway base URL as LLM_ENDPOINT, and
  # the gateway key. LLM_MODEL is the model id the gateway's API expects.
  # (Pick a model that supports JSON-schema structured output — see table below.)
  mimo-v2.5-pro:
    LLM_PROVIDER: openai_compatible
    LLM_ENDPOINT: "https://opencode.ai/zen/go/v1"
    LLM_API_KEY: "<opencode-zen-key>"
    LLM_MODEL: "mimo-v2.5-pro"
```

Priority: shell env vars > `.env.yaml` > `.env`. The first model in `models:` is used as the default when no model is selected in the UI.

**OpenAI-compatible gateways (`openai_compatible`).** Any service exposing an OpenAI-shaped `/v1/chat/completions` endpoint (OpenCode Go, OpenRouter, a local vLLM server, …) is reachable by setting `LLM_PROVIDER: openai_compatible` with the gateway's base URL in `LLM_ENDPOINT` and its key in `LLM_API_KEY`. The call routes through LiteLLM to `{LLM_ENDPOINT}/chat/completions`. HELMS relies on **native structured output** (`response_format` with a JSON schema), so the gateway and the chosen model must honour it, otherwise extraction fails validation.

**OpenCode Go model compatibility (tested).** OpenCode Go exposes two wire formats. HELMS speaks only the OpenAI one (`/v1/chat/completions`), and within it the model must accept a `json_schema` `response_format`:

| OpenCode Go model id | Usable as Extraction / Check LLM? |
|---|---|
| `mimo-v2.5-pro`, `mimo-v2.5` | ✅ yes |
| `glm-5.2` (and `glm-5.1`) | ✅ yes |
| `kimi-k2.7-code` | ✅ yes — note the `-code` suffix; the bare `kimi-k2.7` is rejected as an unknown id |
| `deepseek-v4-pro`, `deepseek-v4-flash` | ❌ no — the gateway returns *"response_format type unavailable"*; DeepSeek supports only loose `json_object`, not `json_schema` |
| `minimax-m3`/`m2.7`/`m2.5`, `qwen3.7-max`/`qwen3.7-plus`/`qwen3.6-plus` | ❌ no — these are served on the **Anthropic** `/v1/messages` format, which the `openai_compatible` provider does not call |

Use a ✅ model. The id in `LLM_MODEL` is the bare gateway id (e.g. `mimo-v2.5-pro`), not the `opencode-go/…` form used by the OpenCode CLI.

**Independent grader model (`SEMANTIC_CHECK_MODEL`).** Set this env var to the name of a model defined in the `models:` block to run the semantic-check grader on a *different* model — and provider — than the one doing extraction, so no model grades its own output (a same-model judge shares the extractor's blind spots). The grader's model/credentials are resolved into an isolated, per-call configuration — `os.environ` is never mutated, so it is safe alongside the concurrent extraction that reads the same environment. Example: extract with an OpenCode Go model like `mimo-v2.5-pro`, grade with Azure `gpt-5.5`. A bare name not in `models:` swaps only the model id on the current provider. Unset means the grader reuses the pipeline's `LLM_MODEL`. The semantic check is one batched call per document, so a strong independent grader is cheap. These selectors appear in the Web UI as **Extraction LLM** and **Check LLM**.

> **Restart required after credential changes.** `llm_client.load_config()` reads `.env.yaml` once at import time via `os.environ.setdefault`. If you add or change keys (including `UMLS_API_KEY`) while the app is running, restart the app — the new values won't be picked up otherwise. A symptom of a missing `UMLS_API_KEY` is zero triples after extraction (all entity lookups silently fail); the HTMX SSE log will show a FATAL message and failures will be written to `_node_agent_errors.jsonl` in the run folder.

**Option B — `.env` (simple single-model setup)**

```env
# Azure OpenAI — used for LLM extraction (native structured output)
LLM_API_KEY=<azure-openai-api-key>
LLM_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/openai/deployments/<deployment>
LLM_MODEL=azure/<deployment-name>
LLM_API_VERSION=2025-04-01-preview

# LlamaParse — required only if using --converter llamaparse
LLAMA_CLOUD_API_KEY=<llama-cloud-api-key>

# UMLS — required only if your schema uses source: umls properties
UMLS_API_KEY=<umls-api-key>

# Neo4j — required only if using --backend neo4j
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<neo4j-password>

# Optional tuning
LLM_TIMEOUT=120                  # per-call HTTP timeout in seconds (default: 120)
LLM_MAX_COMPLETION_TOKENS=65536  # reasoning models (gpt-5.5, o-series) burn a hidden reasoning
                                 # budget before emitting output; 65536 recommended for gpt-5.5
LLM_REASONING_EFFORT=low         # low / medium / high — ignored by non-reasoning models
LOOKUP_CACHE_DB=lookup_cache.db  # path to SQLite cache for UMLS/GLEIF lookups

# LiteLLM multi-provider (alternative to Azure — pick ONE block, remove the others)
# Anthropic Claude
# LLM_PROVIDER=anthropic
# LLM_API_KEY=<anthropic-api-key>
# LLM_MODEL=claude-sonnet-4-6

# Ollama (local)
# LLM_PROVIDER=ollama
# LLM_ENDPOINT=http://localhost:11434
# LLM_MODEL=llama3

# OpenAI direct
# LLM_PROVIDER=openai
# LLM_API_KEY=<openai-api-key>
# LLM_MODEL=gpt-4o
```

`LLM_ENDPOINT` may include the full deployment path — the pipeline strips everything
after `/openai/` automatically. `LLM_MODEL` may carry an `azure/` prefix — also
stripped automatically.

---

## Schema design (`projects/<name>/schema.yaml`)

All extraction decisions live in a per-project YAML schema file — no code
changes needed to add new node types or relationships. The shipped `drug`
project (under `projects/drug/`) is the reference example; the historic
`schemas/supplychain_schema.yaml` (finance / `Corporation` → `PROVIDES`)
is still in git history but is not present in the working tree.

```yaml
nodes:
  Corporation:
    description: "Legal entity registered with a national authority. Examples: Apple Inc, NVIDIA Corporation, Samsung Electronics."
    properties:
      - name: lei          # GLEIF Legal Entity Identifier
        type: STRING
        source: gleif      # populated automatically via GLEIF API lookup
        primary_key: true
      - name: name
        type: STRING
        source: gleif

relationships:
  - rel_type: PROVIDES
    from_node: Corporation
    from_field: supplier_name   # LLM output field; used as GLEIF search term
    to_node: Corporation
    to_field: customer_name
    extract_prompt: >
      Every (supplier_name, customer_name) pair where one company supplies
      a product or service to another.
    examples:
      - supplier_name: "Micron Technology"
        customer_name: "NVIDIA Corporation"
      - supplier_name: "TSMC"
        customer_name: "Apple Inc"
    properties:
      - name: products
        type: STRING[]     # list — one edge per (supplier, customer) pair; all products merged in
        source: llm        # extracted by the LLM
        optional: true
      - name: source
        type: STRING
        source: pipeline   # auto-filled with the source document path
        pipeline_field: doc_path
```

### Optional schema fields

| Field | Where | Effect |
|-------|-------|--------|
| `description` | node definition | Injected into the LLM system prompt as "Node type definitions: …". Helps the LLM distinguish ambiguous types (e.g. Substance vs Tradename). |
| `examples` | relationship definition | Appended to the extraction instruction as "Examples: (X → Y); …". Few-shot anchors that improve recall on the target domain. |
| `sem_group` | node definition | UMLS semantic group name (e.g. `"Physiology"`) or abbreviation (e.g. `"PHYS"`). All TUIs in the group are passed as `semanticTypes=` for server-side filtering. Validated against `UMLS/SemGroups.txt` at schema load — typos fail fast. |
| `semantic_types` | node definition | List of specific UMLS semantic type names (e.g. `["Molecular Function"]`) or TUI codes (e.g. `["T044"]`). When present, takes precedence over `sem_group` — only these TUIs are sent to the API; `sem_group` is ignored at runtime. Use to tighten a broad group to a specific type within it. |
| `umls_vocabs` | node definition | List of UMLS source vocabulary abbreviations (e.g. `["MED-RT"]`). Restricts the UMLS search to those vocabularies via `sabs=`. Prevents general drug names from matching as mechanisms when MED-RT is specified. |

All fields are optional. Schemas without them behave identically to before.

### Property sources

| `source` | Populated by | Notes |
|----------|-------------|-------|
| `llm` | LLM extraction | Set `optional: true` if field may be absent. A `hint` string is appended to the extraction prompt for that field. Example: `publication_date` (ISO 8601 date extracted from the article header) or `products` (list of products supplied). |
| `gleif` | GLEIF REST API | LLM supplies the search term; pipeline resolves the LEI |
| `umls` | UMLS REST API | LLM supplies the search term; pipeline resolves the CUI + `semantic_types[]` (written to graph as a node property) |
| `pipeline` | Pipeline context | Use `pipeline_field: doc_path` for the source file path |

---

## Web UI

A web UI built with **FastAPI + HTMX + Alpine.js** (`htmx_app/`) wraps the full
pipeline — Convert → Extract → Write — with real-time SSE streaming of pipeline
output, an inline triple-review editor, a schema editor, a metadata editor, and a
project-creation wizard. It shares the same project/run folder layout as the CLI.

### Install extra deps

The UI needs a few packages on top of `requirements.txt`:

```bash
pip install -r htmx_app/requirements_extra.txt
```

### Start

```bash
python htmx_app/main.py
# then open http://localhost:8000
```

### Features

| Feature | Detail |
|---------|--------|
| Pipeline steps | Convert → Extract → Write, selectable checkboxes + **▶ Run selected steps** |
| Real-time output | Server-Sent Events stream pipeline stdout line-by-line; no polling |
| Triple review editor | Inline expandable cards; edit / delete triples; 💾 Save; conflict detection; expanded card shows **From / To side by side** (2-column grid), each with editable prop rows plus a read-only **raw term** row (click-to-select) so you can copy the exact extracted term and find it in the document when verifying a red triple. The **supporting quote is editable** too — an edited quote is re-anchored against the source `.md` on save and stored only if found verbatim (otherwise a warning toast, and the original is kept), so a stored quote is always provably in the document |
| 📄 Find in doc | Opens the source `.md` in a modal with the triple's evidence highlighted: **quote yellow**, **subject (`from_term`) green**, **object (`to_term`) pink** (color-coded by precedence so a term nested inside the quote keeps its own color), scrolled to the first match; banner if no span located (unlocatable / possibly fabricated quote) |
| ↺ Retry | Re-extract the selected document without leaving the review UI |
| 🥷🏻 Agent Retry | Per-rel-type targeted retry via `extraction_agent.py` — LLM calls UMLS/GLEIF tools iteratively with full schema constraints (`umls_vocabs`, `semantic_types`, `sem_group`); use when batch extraction missed a triple or resolved the wrong entity; **semantic check auto-runs on the retried rel_type triples after merge** (via `_run_agent_semantic_check` in this UI; the CLI uses `_cli_semantic_check` for parity) — adds `ai_opinion`, recolors, flags `constraint_violated`; button disables and shows "● agent (rel_type)" in the log header while running |
| Per-triple color override | Click the color dot on any triple card to cycle green → yellow → red; promotes a correctly-resolved red triple to green so it passes the `moderate` write filter in Step 3; color persisted as OVERRIDE event in `_review.json` |
| Harvest self-improvement | After Step 3, `agents/harvest.py` rebuilds `<project>/harvest/<rel_type>.jsonl` from all reviewed runs (background thread, triggered in SSE sentinel); also rebuilt after every Agent Retry save (same background-thread path). Positive examples (by priority: add→override→agent_retry→batch) + rejected triples (negative examples with original doc + quote) are injected into the next extraction; rejections are placed in every chunk's user message (highest LLM recency, immediately before document text). |
| 📊 Extraction quality | Triples, 🟢/🟡/🔴 counts, entity resolution rate, delta vs previous run |
| Schema editor | Edit node `description`, `sem_group`, `umls_vocabs`, `semantic_types`; relationship `extract_prompt`, `from_hint`, `to_hint`, `examples`, property `hint` — writes back to `schema.yaml` with Pydantic validation before save |
| Metadata editor | Edit `instructions` and per-PDF `pages` filters — writes back to `meta.yaml` |
| Extraction LLM / Check LLM | Two dropdowns: the model that **extracts** triples, and the (optional, independent) model that **grades** them in the semantic check. Defaults to "same as Extraction LLM"; pick a different model or provider so no model grades its own output. Switchable without restarting; both recorded in `run_config.json` |
| Graph summary | Node/edge counts + sample triples from the run DB |
| Runs dropdown | Select historic runs or create a new one; auto-resets on project change |
| Step badges | Circles show ✓ green on completion; restored automatically on run selection and page refresh by inspecting run folder contents (`*.md` / `*_raw.json` / `*.db`) |
| Project wizard | Create a new project folder (blank or from a template schema) without leaving the UI |
| Open in Finder | Open the current run folder in macOS Finder / Windows Explorer / Linux file manager from the sidebar |
| Persistent state | App state saved to `htmx_app/.helms_state.json` — survives restarts |
| Per-project cache | LLM extraction cache stored at `projects/<name>/.cache/` (isolated per project; schema **and the extraction model** are part of the cache key, so cross-project hits are impossible and switching the Extraction LLM forces a real re-extraction) |

The UI shares all backend modules with the CLI — same `pipeline_runner.py`, `review_layer.py`, `backends/`, and pipeline scripts.

### Runs system

Each pipeline execution writes into `projects/<name>/runs/<YYYYMMDD_HHMMSS>/`:

```
projects/drug/
├── schema.yaml
├── meta.yaml
├── raw_documents/          ← PDFs only
└── runs/
    ├── 20260525_170300/
    │   ├── run_config.json         ← params snapshot + extraction_stats after extract
    │   ├── doc_a_raw.json          ← immutable LLM output (triples + doc_text)
    │   ├── doc_a_review.json       ← event log (human decisions)
    │   ├── _node_agent_errors.jsonl ← UMLS failures (only present when errors occur)
    │   └── drug_kg.db
    └── 20260524_090000/
        └── ...
```

Select **New run** in the Runs dropdown to start a fresh run; select an existing
timestamp to view or re-run a historic run. Re-running a historic run with different
settings appends a `reruns` entry to its `run_config.json` rather than overwriting it.

`run_config.json` also stores `extraction_stats` after each extraction run:
docs processed, total triples, green/yellow/red counts, entity resolution count,
unresolved count, and entity resolution rate.

It also records **which LLM served each task** in an `llm_models` block, written
from the run's actual environment so it reflects the models used (not the configured
default):

```json
"llm_models": {
  "extraction":      "mimo-v2.5-pro",
  "node_resolution": "mimo-v2.5-pro",
  "semantic_check":  "gpt-5.5"
}
```

`extraction` and `node_resolution` use `LLM_MODEL`; `semantic_check` uses
`SEMANTIC_CHECK_MODEL` when set, else falls back to the extraction model.

Every node gets a `run` property (first-write-wins — earlier run's value preserved if
node already exists). Every edge gets a `run` property alongside `source_doc`.

### Semantic Check

`agents/semantic_check_agent.py` runs **automatically** at the end of every extraction — no button needed. It runs after `agents/node_agent.py` resolves UMLS entities, so triples carry real `semantic_types` when the check fires. The grader model is the **Check LLM** (`SEMANTIC_CHECK_MODEL`), which can be a different model and provider than the extractor so no model grades its own output (see [Configure credentials](#2-configure-credentials)). The agent:

1. Calls the LLM in batches of 25 triples, checking whether each entity is semantically present in the document text and supporting quote. The document is truncated to a 60k-char excerpt for the prompt; when a triple's (re-anchored, verbatim) quote falls *beyond* that window, the surrounding doc text is attached to the batch item as `quote_context` so the LLM grounds against the real neighbourhood instead of defaulting the entity to yellow.
2. Re-colors each triple's `from_color` / `to_color` (green / yellow / red) and derives `triple_color`.
3. Checks each triple against schema `extract_prompt` constraints (e.g. adverse effects only in rodents must not be attributed to humans). Sets `constraint_violated = true` and `triple_color = red` when a violation is found — regardless of entity grounding colors. Semantic type mismatch (actual UMLS types not overlapping schema `semantic_types`) is also treated as a constraint violation.
4. Writes a brief `ai_opinion` per triple (1–2 sentences covering color rationale and any constraint violations).
5. Runs deterministic structural checks (empty primary-key values, duplicates, procedure-marker mismatch), a **UMLS semantic type mismatch check** (fires whenever the schema node defines `semantic_types`, regardless of `--filter` level — now a **hard red floor + `constraint_violated`** since the mismatch is provable, no longer dependent on the LLM acting on the warning; a vocab-only mismatch stays note-only), and a **schema-conformance check** (`SchemaConformanceValidator`) that reds any triple whose `(rel_type, from_label, to_label)` is not a declared edge in the schema. Appends `[Structural: …]` / `[UMLS: …]` / `[Schema: …]` flags to `ai_opinion`.
6. Injects `{from,to}_expected_semantic_types` and `{from,to}_actual_semantic_types` into each LLM batch item, so the LLM opinion explicitly mentions type mismatches (e.g. schema expects `["Molecular Function"]`, resolved entity is `Pharmacologic Substance`).
7. **Harvest rejection check** (`harvest_dir` param): after the LLM and structural checks complete, scans `harvest_dir/*.jsonl` for `source="rejected"` entries and forces any matching triple to red + `constraint_violated=True` + `"[Previously rejected by human reviewer]"` in `ai_opinion`. Deterministic — no LLM call, runs even when the chunk was served from cache. Human can still override the color in the UI if the current document provides stronger evidence (doc-specific semantics preserved).
8. **Fabricated-quote check** (`FabricatedQuoteValidator`): red-floors any `extraction_source == "agent_retry"` triple whose `evidence` has **zero** located spans — i.e. the agent's supporting quote is nowhere in the document. Reason: `"[Supporting quote not found verbatim in the document — likely fabricated by the agent; read the doc to confirm before accepting]"`. Color floor only (not `constraint_violated`) — the relationship may be true, so the human verifies and can cycle to green. The batch path is excluded on purpose (its grounding "warn" is cosmetic; `strict` already drops unlocatable quotes), and a partially-verbatim quote (≥1 located span) is left to best-of-segment. The agentic `add_triple` also nudges the agent to re-quote verbatim when `grounding.reanchor` fails (reinforced in the agent system prompt); a re-call for the same pair **replaces** the unlocatable quote with the verbatim one (no append), so a successful retry ends clean and only a persistent failure lands red.
9. **Deterministic green anchor** (`EntityPresenceValidator`, runs first): proves an entity is named in its own supporting quote via `grounding.locate` — checking the resolved name, the raw extracted term (gated by resolver source: UMLS/LLM always trust it, GLEIF drops it on a suspicious resolution), and any resolver-supplied synonyms — and pins that entity's color to green. This is the inverse of a red floor: a floor raises a triple toward red against false greens, an anchor raises an entity toward green against false reds. It touches only the entity axis, so `constraint_violated` (relation support/direction) and red floors (steps 6–8) still win the edge color. Why: LLM grounding of a *provably present* entity is not just redundant, it's measurably inconsistent — the same input graded twice by gpt-5.5 flipped one entity's color between runs; the anchor makes that case deterministic. `annotate_item` also tells the LLM the entity's presence is already proven so it spends its judgment on relation support instead.
10. **Relation-support hints** (`RelationSupportHintValidator`, annotate-only, no verdict): precomputes two advisory facts for the LLM's relation-support/direction judgment — whether some quote segment names *both* endpoints together, and whether a negation/hedge cue (e.g. "not observed in humans") appears in the quote. Neither sets a color on its own; they hand the LLM facts it would otherwise have to re-derive by re-reading, so it spends its call on the actual judgment.

Color semantics:

| Color | Meaning |
|-------|---------|
| 🟢 green | Entity present in both document text and supporting quote; no constraint violation |
| 🟡 yellow | Entity present in document text but not clearly in the supporting quote |
| 🔴 red | Entity not found in document text (possible hallucination) **or** `constraint_violated = true` |

`triple_color` is `red` if `constraint_violated` is true or either entity is red; `yellow` if either entity is yellow; otherwise `green`. The per-entity color icons (🟢/🟡/🔴) are displayed next to the **From** and **To** headers inside each expanded triple card, and `ai_opinion` is shown below the relationship props.

`ai_opinion` is saved to the review JSON as an audit trail and **persists across file switches** — switching to another file and returning preserves the semantic check results. It is not written to the graph on Write — `apply_graph.py` ignores this field.

---

## Pipeline metadata

`pipeline_meta.py` is a shared module that loads a project-level YAML file
(`--meta PATH`) with two optional sections:

- **`instructions`** — a free-text block injected into the LLM system prompt for
  every document. Use it to restrict extraction scope (e.g. "only extract triples
  for Beyfortus; ignore comparator drugs").
- **`pages`** — per-PDF page filters. Each key is a PDF stem (filename without
  `.pdf`). Supports `include`, `exclude`, or both in the same entry. Reversed
  ranges (e.g. `10-5`) are warned and skipped.

```yaml
# project/drug_instruction.yaml
instructions: |
  Only extract triples for Beyfortus (nirsevimab, nirsevimab-alip).
  Do NOT extract any triples for comparator drugs mentioned in passing.

pages:
  beyfortus label:
    include: [1-14]           # pages 1–14 only
  some other report:
    exclude: [1, 15-16]       # all pages except cover and appendix
  another report:
    include: [1-40]
    exclude: [5, 10-12]       # include 1-40, then subtract 5, 10, 11, 12
```

Pass the metadata file with `--meta` to any of the pipeline scripts:

```bash
python pipeline.py --schema schemas/drug_schema.yaml --input drug_pdf/ \
  --meta project/drug_instruction.yaml

python convert_pdf.py --input drug_pdf/ --meta project/drug_instruction.yaml
python extract.py --schema schemas/drug_schema.yaml --input drug_pdf/ \
  --meta project/drug_instruction.yaml
python agents/extraction_agent.py --schema schemas/drug_schema.yaml \
  --input drug_pdf/beyfortus\ label.md --meta project/drug_instruction.yaml
```

`--meta` is optional everywhere. Omitting it preserves the original behaviour
(all pages converted, no instruction injection).

> **Cache note:** `instructions` is included in the extraction cache key. Changing
> instructions on an already-cached document will trigger a fresh LLM call.
> Changing a page filter in the YAML does **not** automatically invalidate an
> existing `.md` sidecar — re-run `convert_pdf.py --force` to regenerate.

---

## PDF conversion

`convert_pdf.py` converts PDFs to Markdown and saves `.md` sidecars next to each PDF.
`pipeline.py` is a one-command orchestrator that calls all three steps: convert → extract → write.

```bash
# One command: convert PDFs, extract triples, write to graph (LadybugDB)
python pipeline.py \
  --schema schemas/supplychain_schema.yaml \
  --input  finance_pdf/ \
  --db     supplychain_kg.db \
  --converter pymupdf4llm          # default; fast, CPU-only

# With metadata (page filtering + extraction instructions)
python pipeline.py \
  --schema schemas/drug_schema.yaml \
  --input  drug_pdf/ \
  --db     drug_kg.db \
  --meta   project/drug_instruction.yaml

# LlamaParse converter (cloud API; requires LLAMA_CLOUD_API_KEY in .env)
python pipeline.py \
  --schema schemas/supplychain_schema.yaml \
  --input  finance_pdf/ \
  --db     supplychain_kg.db \
  --converter llamaparse

# Convert only (no extraction)
python convert_pdf.py --input finance_pdf/ --converter pymupdf4llm
python convert_pdf.py --input finance_pdf/ --converter llamaparse
```

`extract.py` accepts `.md` files directly. For `.pdf` input a `.md` sidecar must
already exist next to the PDF (created by `convert_pdf.py` or `pipeline.py`). If no
sidecar is found, the run exits with a message pointing to `convert_pdf.py`.

```bash
# Pass a pre-converted .md directly
python extract.py --schema schemas/supplychain_schema.yaml --input finance_pdf/report.md

# Pass a .pdf only if convert_pdf.py has already produced report.md alongside it
python extract.py --schema schemas/supplychain_schema.yaml --input finance_pdf/report.pdf
```

### Converter comparison

| | pymupdf4llm | LlamaParse |
|---|---|---|
| Speed | Fast (seconds) | Cloud API (seconds, network-bound) |
| Accuracy | Good | Best; understands tables, forms |
| Images | Alt-text captured inline | Handled server-side |
| Page markers | Left in | Stripped |
| Heading hierarchy | Mostly `##` flat | Consistent |
| Multi-column boxes | Reads L→R across columns — content may interleave | Correctly parsed |
| Page footers / headers | Often dropped (e.g. "Revised: 9/2023" at bottom of a drug label page) | Captured |
| Install | Included in `requirements.txt` | `pip install 'llama-cloud>=2.1'` + API key |
| Cost | Free | API credits (free tier available) |

**Recommendation:** Use `pymupdf4llm` (default) for speed. Use `llamaparse` for
complex layouts (multi-column, tables, drug labels, forms) where extraction quality
matters most.

**Corrupted PDF resilience:** If one PDF in a batch fails to open (corrupted,
password-protected, etc.), `convert_pdf.py` warns and skips it — other PDFs in
the batch continue processing normally.

---

## Extraction pipeline

The pipeline is split into two explicit steps so you can inspect and edit
extracted triples before they are written to the graph.

### Step 1 — Extract (`extract.py`)

Runs the LLM, resolves entities against GLEIF/UMLS, and writes
`<stem>_raw.json` (immutable) and `<stem>_review.json` (event log). No graph writes occur.

```bash
python extract.py --schema schemas/supplychain_schema.yaml --input finance_pdf/report.md
# → produces report_raw.json + report_review.json
```

**`extract.py` flags**

| Flag | Default | Description |
|------|---------|-------------|
| `--schema` | *(required)* | Path to the per-project `schema.yaml` (e.g. `projects/drug/schema.yaml`) |
| `--input` | *(required)* | `.md` file or directory; `.pdf` accepted only if a `.md` sidecar exists alongside it |
| `--meta` | *(off)* | Path to pipeline metadata YAML; injects `instructions` into LLM system prompt |
| `--skip-report` | *(off)* | Print a table of skipped triples at the end (columns: Document, Rel, Node, Term, Reason); records entity-resolution failures (failed GLEIF/UMLS lookups). Quote-grounding drops (`DROP (no quote)` / `DROP (quote not in doc)`) are logged to the console, not this table |
| `--force` | *(off)* | Bypass extraction cache and re-run LLM even if a cached result exists |
| `--concurrency` | `2` | Max documents processed concurrently (async); UMLS/GLEIF calls run in threads to keep the event loop free |
| `--verbose` | *(off)* | Print cache-hit messages from UMLS/GLEIF lookups |
| `--filter` | `moderate` | Extraction strictness: `loose` (max recall, no quote required), `moderate` (warn if quote missing, keep triple), `strict` (drop triple if quote not found verbatim in document) |
| `--chunk-retries` | `1` | Max selective retry passes over the individual chunks whose LLM call failed, before falling back to a partial result. Cached/successful chunks are never re-run — only the failed chunk indices are retried |
| `--estimate-only` | *(off)* | Print a pre-flight chunk/token/cost estimate for the input and exit without making any LLM calls |

> **Pre-flight estimate.** Every `extract.py` run (CLI and both UIs) prints a `[pre-flight]` summary before any LLM call — docs to extract (and how many are skipped because `*_raw.json` already exists), total chunk count, estimated input tokens (~4 chars/token), and estimated input cost via LiteLLM's pricing table for the active model. Output tokens are not estimated. Pass `--estimate-only` to see the estimate and stop.

> **Selective failed-chunk retry.** When a chunk's LLM call fails mid-document (transient API error), the chunk index is recorded and re-run in a follow-up pass (`--chunk-retries`, default 1) instead of re-extracting the whole document. Successful and cache-hit chunks are untouched, so no LLM calls are wasted. `asyncio.TimeoutError` still propagates to the document-level retry (`--retries`).

### Step 2 — Review (human edit)

Open `<stem>_review.json` in the web UI or directly. Events are ACCEPT / REJECT / OVERRIDE / ADD actions keyed by stable triple `_id`. `review_layer.materialize()` overlays events onto the immutable `_raw.json` to produce the effective triple list for `apply_graph.py`.

```json
{
  "doc": "report.md",
  "triples": [
    {
      "rel_type": "PROVIDES",
      "from_label": "Corporation",
      "from_pk": "lei",
      "from_props": {"lei": "549300...", "name": "Acme Corp"},
      "to_label": "Corporation",
      "to_pk": "lei",
      "to_props": {"lei": "254900...", "name": "Globex Inc"},
      "rel_props": {"products": ["widget A"]},
      "_id": "t1a2b3c4d5e6"
    }
  ]
}
```

### Step 3 — Write (`apply_graph.py`)

Reads the (optionally edited) review JSON and writes to the graph.

```bash
# LadybugDB (default — embedded, no server needed)
python apply_graph.py \
  --schema schemas/supplychain_schema.yaml \
  --apply  report_review.json \
  --db     supplychain_kg.db

# Neo4j (requires NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD in .env)
python apply_graph.py \
  --schema  schemas/supplychain_schema.yaml \
  --apply   report_review.json \
  --backend neo4j

# Dry-run: show what would be written without touching the graph
python apply_graph.py \
  --schema  schemas/supplychain_schema.yaml \
  --apply   report_review.json \
  --dry-run
```

> **Note:** `query_graph.py` is Ladybug-specific. `mcp_server.py`
> supports both backends via `GRAPH_BACKEND` env var. For Neo4j CLI queries, use
> Neo4j Browser or any Cypher client directly.

**`apply_graph.py` flags**

| Flag | Default | Description |
|------|---------|-------------|
| `--apply` | *(required)* | Review JSON file produced by `extract.py` |
| `--schema` | `schemas/supplychain_schema.yaml` | Schema YAML file (needed to initialise backend tables) |
| `--db` | derived from schema stem (e.g. `supplychain_kg.db`) / `NEO4J_URI` env var (neo4j) | Database path or URI |
| `--backend` | `ladybug` | Graph backend: `ladybug` (embedded file) or `neo4j` (bolt URI) |
| `--filter` | `moderate` | Only apply triples at or above this color level: `loose` (all), `moderate` (green+yellow), `strict` (green only). Default is `moderate` (consistent with `extract.py` and `pipeline.py`). |
| `--dry-run` | *(off)* | Show what would be written without committing to graph |

### Extraction cache

LLM responses are cached as JSON files keyed by `sha256(chunk_text + schema + instructions + abbr_map + harvest_signature + extraction_model)` — one entry per chunk. The `harvest_signature` is a content hash of `harvest/*.jsonl`, so adding a rejection or example automatically invalidates affected chunks (no `--force` needed after rejecting a triple). The **extraction model** is part of the key too: switching the Extraction LLM (e.g. MiMo → GPT-5.5) misses the cache and genuinely re-extracts, instead of silently reusing another model's triples while `run_config.json` claims the new model. (The Check LLM is *not* in the key — the semantic check is not chunk-cached, so swapping it always re-grades.) Re-runs with identical inputs skip the LLM call entirely. Partial re-runs (e.g. schema change on a long document) only re-call the LLM for chunks whose key changed. Use `--force` to bypass the cache. Cache files are safe to delete — they will be rebuilt on the next run. One file is written per non-empty chunk, so a document long enough to split into multiple sections produces multiple cache files.

Cache location:
- **Web UI**: `projects/<name>/.cache/` (per-project, set via `KG_CACHE_DIR` env var)
- **CLI (`extract.py`, `pipeline.py`)**: `.cache/` in the project root (override with `KG_CACHE_DIR=<path>`)

Cross-project cache hits are impossible because `schema.yaml` content is part of the cache key.

### Lookup cache maintenance

UMLS/GLEIF API responses (and the resolver's LLM-pick decisions) are cached in
`lookup_cache.db` — separate from the per-chunk extraction cache above. This cache has
no expiry by default, so a single wrong resolution (e.g. a company name that resolved to
the wrong LEI) would otherwise persist across every future run. Two controls manage it:

**Maintenance CLI** (`python -m lookup_cache`):

```bash
python -m lookup_cache stats                  # row counts + oldest/newest per service
python -m lookup_cache evict "murata"         # drop entries whose key contains "murata"
python -m lookup_cache evict "murata" --service gleif_pick   # scope to one service
python -m lookup_cache clear --service umls    # wipe one service (prompts unless --yes)
python -m lookup_cache clear                   # wipe everything (prompts unless --yes)
```

Services: `gleif`, `gleif_candidates`, `gleif_pick`, `umls`, `umls_pick`, `umls_sabs`.

**TTL** (`LOOKUP_CACHE_TTL_DAYS`): default `30` days — `get()` treats rows older than the TTL
as cache misses, deleting them so the next lookup re-fetches fresh registry data. Set `0`
(e.g. `export LOOKUP_CACHE_TTL_DAYS=0`) to disable expiry and keep entries forever; raise/lower
the number to taste.

> **Caveats for the long-running web UI server:**
> - `lookup_cache.py` reads `LOOKUP_CACHE_TTL_DAYS` at import, before `.env` is loaded — set it
>   as a real shell env var **before launching** the server, not in `.env`.
> - The L1 in-process dicts in `lookups.py` have no TTL and no eviction hook. A CLI `evict`
>   deletes from the L2 SQLite only; a value already cached in the running server's L1 keeps
>   being served until you **restart the server**.

### Markdown sidecar requirement

`extract.py` does not convert PDFs. Conversion is the exclusive responsibility
of `convert_pdf.py` (or `pipeline.py`).

- Pass a `.md` file → read directly.
- Pass a `.pdf` file → looks for a `.md` sidecar with the same stem next to it (mtime ≥ PDF). Found → use it. Not found → `SystemExit` with a message pointing to `convert_pdf.py`.
- To force re-conversion, run `convert_pdf.py --force` (or `pipeline.py --force-convert`) then re-run `extract.py`.
- Empty `.md` files and sidecars produce a warning at extraction start but do not abort the run.

Documents are first split on Markdown section headers (`#` / `##` / `###`) to produce semantic chunks; sections exceeding 20 000 characters fall back to the char-based overlap splitter (`_CHUNK_SIZE=20_000`, `_CHUNK_OVERLAP=2_000`, both overridable via the `LLM_CHUNK_SIZE` and `LLM_CHUNK_OVERLAP` env vars). Each chunk is independently extracted by a single structured LLM call using the open-minded extractor prompt from `prompts.yaml`, which maximises recall. The LLM assigns an `evidence_level` to every extracted triple: `strong` for a primary claim repeated multiple times, `moderate` for a single explicit statement, and `weak` for inferred or briefly mentioned relationships. The `(from_pk, to_pk)` dedup collapses cross-chunk duplicates before node-agent UMLS resolution.

### Extraction filter levels (`--filter`)

`--filter` controls which triples get **written to the graph** after extraction. LLM extraction always runs at full depth regardless of this setting.

| Level | Triples written to DB |
|-------|----------------------|
| `strict` | `triple_color = green` only (clearly stated, high confidence) |
| `moderate` *(default)* | `green` + `yellow` (includes inferred or briefly mentioned) |
| `loose` | All colors (keeps even weak/speculative triples) |

Triples below the threshold are retained in the review JSON for inspection — they are not deleted, just not committed to the graph.

```bash
# High recall — keep all extracted pairs
python extract.py --schema schemas/drug_schema.yaml --input drug.md --filter loose

# Balanced (default)
python extract.py --schema schemas/drug_schema.yaml --input drug.md --filter moderate

# High precision — drop any pair whose supporting quote isn't verbatim in the document
python extract.py --schema schemas/drug_schema.yaml --input drug.md --filter strict
```

The `--filter` flag is also available on `agents/extraction_agent.py` and in the web UI's Step 2 settings.

### External lookup reliability

UMLS and GLEIF HTTP requests use exponential backoff (3 attempts, delays of
1 s → 2 s → 4 s). Transient rate-limit or network errors no longer cause
spurious entity skips.

`extract.py` UMLS resolution (orchestrated by `agents/node_agent.resolve_all_nodes`,
which dispatches to `agents/umls_resolver.UMLSResolver.resolve_batch`) falls back
through multiple search types before giving up: `words` (default) → `normalizedWords`
→ `normalizedString`. This significantly reduces entities lost due to abbreviations
or minor spelling variations that the `words` tokenizer cannot match. (All three
are valid UMLS REST `searchType` values; the earlier `approximate` step was not a
recognized type — it always errored, was never cached, and re-fired a wasted HTTP
request on every run.)

**GLEIF entity resolution pipeline.** `gleif_resolver.py` resolves company names in three stages: (1) `gleif_get_candidates` collects a broad candidate pool; (2) short all-caps abbreviations (≤6 chars, e.g. "HPE", "TSMC") with no strong direct candidate are expanded to a full official legal name and re-searched — source order is **`ctx.abbr_map` → static `_ABBREV_TABLE` → `_expand_abbreviation()` LLM**. `abbr_map` is the document-derived ABBR→full-name map from `extract._extract_abbreviations` (the `Full Name (ABBR)` notation the doc itself defines); when it or the static table answers, the LLM expansion call is skipped entirely (cheaper, and more faithful than the model's world-knowledge guess). The LLM fires only when both miss; (3) the winner is picked by an LLM, guided by a `domain_hint` string from the project's `meta.yaml` `instructions` (e.g. "supply chain of semiconductor manufacturers — Taiwan, Japan, US") so the LLM prefers Taiwan-jurisdiction companies over European ones for electronics context. When the LLM returns index=0 (no match), the `retry_search_term` field in its response carries a suggested full name; the resolver retries GLEIF search with that term and does a second pick. This handles cases like "Murata" (which GLEIF fuzzy-matches to Italian/Finnish/Indian companies) — the LLM suggests "Murata Manufacturing Co., Ltd." and the retry finds the correct Japanese entity.

> **Batched picks (cost).** Candidate gathering, abbreviation expansion, and the no-LLM prefilter (single-candidate / exact-name-match shortcuts) run per entity, concurrently. The remaining *ambiguous* entities are then disambiguated in a single batched structured LLM call (`resolve_batch` Phase B, ≤12 entries per call) rather than one call per entity. Index=0 retries are gathered and re-picked in one follow-up batch. For a document with ~32 entities, this collapses ~32 pick calls into ~3–4. Per-entity `_resolve_one`/`_do_resolve_one` are retained for the single-entity path and tests. The same Phase A / Phase B batching applies to `umls_resolver.py`.

**GLEIF resolution warning in semantic check.** `semantic_check_agent._gleif_name_suspicious(orig_term, resolved_name)` fires when the resolved name starts with the original search term AND has trailing words that are not legal-form suffixes (e.g. "Quanta" → "QUANTA LYON" — "LYON" is a city, not a legal suffix). When triggered, `from_gleif_resolution_warning`/`to_gleif_resolution_warning` fields are injected into the LLM batch item, and the system prompt instructs the LLM that this warning overrides synonym matching — so "Quanta" ≈ "QUANTA LYON" is NOT accepted as a valid synonym. Abbreviations (isupper + len≤6) are skipped — they expand via the abbreviation chain (`abbr_map` → static table → `_expand_abbreviation`) instead.

**GLEIF lookup cascade.** `gleif_lookup` in `lookups.py` tries six strategies in order before giving up:

1. Exact match: `filter[entity.legalName]` on the full name
2. Names search: `filter[entity.names]` on the full name — exact match across primary legal name **plus** other names and transliterated names. Placed before fuzzy so abbreviations like "TSMC" (registered as an other name) are found before prefix-match noise (e.g. "TSMC Partners, Ltd") is returned by autocomplete.
3. Fuzzy autocomplete: `fuzzycompletions` on the full name (broader prefix search, last resort for full name)
4. Fuzzy autocomplete on the suffix-stripped form (first 3 significant words, corporate suffixes removed) — catches LLM-expanded suffixes (e.g. "Company Limited" vs registered "Co., Ltd.")
5. Names search on the suffix-stripped form — final fallback
6. **"WHO OWNS" parent lookup:** if the search term is an all-caps abbreviation (e.g. `TSMC`) and the top result has non-suffix extra words after the abbreviation (e.g. "TSMC Partners, Ltd" — "Partners" is not a corporate suffix), the result is likely a subsidiary. In that case the pipeline calls `filter[owns]=<subsidiary-LEI>` one level up to retrieve the parent entity (e.g. Taiwan Semiconductor Manufacturing Company Limited, LEI 549300KB6NK5SBD14S87). Only fires for all-caps search terms to avoid false parent-following for full company names.

Parenthetical abbreviations are stripped before any search step (e.g. "Powertech Technology (PTI)" → "Powertech Technology"). When multiple candidates are returned, `GENERAL`-category entities are preferred over funds and branches.

**ASCII node names.** `_gleif_attrs` uses the registered English ASCII name as the node name for companies whose primary legal name is non-Latin. It checks two fields in priority order:

1. `transliteratedOtherNames` — looks for `type = PREFERRED_ASCII_TRANSLITERATED_LEGAL_NAME` (used by Taiwanese and Chinese companies, e.g. TSMC → "Taiwan Semiconductor Manufacturing Company Limited")
2. `otherNames` — falls back to `type = ALTERNATIVE_LANGUAGE_LEGAL_NAME` (used by Japanese companies, e.g. TDK Corporation)

If neither is present, the primary legal name is used as-is. **Cache note:** name values are baked into `lookup_cache.db` entries. If you added this fix after running the pipeline, delete `lookup_cache.db` to force fresh API calls and pick up the correct ASCII names.

**UMLS semantic type filtering.** Two schema fields control UMLS filtering. `sem_group` (e.g. `"Physiology"`) passes all TUIs in that group as `semanticTypes=`; TUI lists are resolved from `UMLS/SemGroups.txt` at import. `semantic_types` (e.g. `["Molecular Function"]`) specifies individual type names or TUI codes and takes precedence over `sem_group` when both are set — only the listed TUIs are sent to the API. For example, the MOA node with `semantic_types: ["Molecular Function"]` sends `&semanticTypes=T044`, preventing EPC drug-class terms ("Pharmacologic Substance") from matching as mechanisms of action. Client-side candidate iteration over all returned results acts as an additional safety net. Note: the older `sty=` parameter documented by UMLS does not filter results in practice; `semanticTypes=` is the working parameter.

**Post-extraction semantic type validation.** Even when the UMLS API returns a wrong-typed entity (e.g. `RNA Polymerase Inhibitor` resolved as `Pharmacologic Substance` instead of `Molecular Function`), `semantic_check_agent.py` catches it as a deterministic structural flag at all filter levels. The `_build_items` function injects the schema's expected semantic types and the resolved entity's actual types into every LLM batch item, so the LLM also reports the mismatch in `ai_opinion` and sets `constraint_violated=true`, which colors the triple red.

**UMLS LLM disambiguation and rejection.** When multiple candidates are returned, the resolver calls the LLM to pick the best match — batched across all ambiguous entities in one call (see *Batched picks* above). The LLM may return index 0 to reject all candidates when none is semantically equivalent to or a recognised synonym of the query term (e.g. a partial-string hit referring to a different concept entirely). Rejected entities are logged to `_node_agent_errors.jsonl` with reason "no acceptable UMLS match — dropped by LLM". UMLS API results for rejected terms are still cached in `lookup_cache.db` (deterministic API response); the LLM rejection is not cached and re-evaluated each run.

### Quote grounding + re-anchoring (`grounding.py`)

LLMs do not copy `supporting_quote` verbatim — they wrap it in quote characters,
swap ASCII/curly quotes, drop the inline `([[N]] url)` citations the markdown
converter injected, and elide spans with `...`. A naïve `quote in doc_text`
substring test therefore fails on benign paraphrase.

`grounding.py` is a shared, token-based locator. It tokenises both the quote and
the document into normalised words that carry their character offsets in the
*original* text, then returns the exact `(start, end)` span. The matcher absorbs
the formatting noise PDF→Markdown conversion and LLMs introduce:

- inline `([[N]] url)` citations and HTML tags (`<u>…</u>`) are blanked
  length-preservingly before tokenising (offsets stay valid) — on **both** the
  doc *and* the quote (a quote that includes a citation would otherwise carry a
  token the citation-blanked doc no longer has, and fail to match);
- a run-on sentence period that fused two words into one token (`"TSMC.On"`, a
  common PDF→Markdown artifact) is split back into a space so a quote starting
  right after the period can match — sparing `U.S.`, `3.5`, `etc.)`, `Amazon.com`;
- per-word edge normalisation folds curly→ASCII quotes and strips markdown
  markers, brackets, bullet glyphs (`• ◦ ‣ ·`), and trademark symbols (`® ™ ©`)
  — so `DIFICID[®]` ≡ `DIFICID®`, `**NVIDIA**` ≡ `NVIDIA`;
- `...`/`…` split the quote into fragments matched in order.

`extract.py::_verify_grounding` uses it to **re-anchor** every locatable quote:
the stored quote is overwritten with the verbatim `doc_text[start:end]` it matched.
The re-anchored per-edge quotes are then turned into `evidence: [{start, end, text}]`
by `extract._build_evidence` (offsets from `grounding.locate`), and **`supporting_quote`
becomes a derived projection** (`/`-joined `text`). So `evidence` is the grounding
source of truth; `supporting_quote` is the display/graph string. `evidence` lives
only in `*_raw.json` (not on the graph edge — offsets have no KG query value).
Consequences:

- Downstream consumers (triple building, semantic check, NVL display, harvest)
  never see LLM quote drift — the stored quote is guaranteed-in-document.
- The "quote not in doc" warning now fires **only** when a quote genuinely cannot
  be located (real paraphrase/hallucination), not on cosmetic formatting
  differences. Under `--filter moderate` such a triple is warned and kept (with
  its original LLM quote, color set normally by the semantic check — the warning
  does *not* recolor or empty it); under `strict` it is dropped.
- Span covers the matched **word cores**, so leading/trailing markup and the
  sentence-final period are excluded (e.g. `**ASE Group**.` → `ASE Group`);
  internal markup is preserved verbatim.

**Multi-segment rescue.** LLMs frequently stitch *non-adjacent* document fragments
into one quote — merged quotes joined with ` / `, or a list intro paired with a
specific bullet item (e.g. `"side effects include: * vomiting"`, where other
bullets sit between the intro and the item in the document). No single span covers
these, so when the whole-quote match fails, `_verify_grounding` splits the quote on
those separators (` / `, ` * `, ` - `, ` • `), re-anchors each segment, and — if
*every* segment is verbatim — stores the joined verbatim segments. This grounds
valid-but-non-contiguous quotes (on real drug-label runs it cut grounding warnings
~4× without admitting any non-verbatim text). Partial matches fall through to
warn/drop.

The agentic path (`agents/extraction_agent.py::add_triple`) re-anchors via the
same `grounding.reanchor`, so both extraction paths produce identical,
verbatim quotes.

> **Known limits (expected, not bugs).** A few markdown patterns still defeat
> token matching: an italic underscore that inserts a space splitting a hyphenated
> compound (`difficile-associated` in the quote vs `_C. difficile_ -associated` in
> the doc — one token vs two), and long comma-lists under italic section headers.
> Bridging these needs cross-token merging, which risks false-positive grounding,
> so they are left to warn. Under `moderate` the triples are still kept. For
> drug labels specifically, `--converter llamaparse` emits cleaner markdown (no
> `<u>` tags, no `[®]`, normalised hyphens) and avoids most of these.

### Quote grounding filter (batch path)

`_verify_grounding` runs over the extracted items **before** entity resolution
(so any dropped triple never costs a UMLS/GLEIF round-trip). It is purely
**quote-based** — there is no longer an entity-in-document drop or a quote-overlap
drop. Entity presence is now a *coloring* concern handled downstream by the
semantic-check agent (green/yellow/red, never a drop), and the agentic retry path
adds its own hard-reject (below).

Behavior by `--filter` level:

- **loose** — keep every triple; no grounding requirement.
- **moderate** — re-anchor each quote to the verbatim document span
  (`grounding.reanchor`); a located quote is overwritten with the exact text and
  kept. A quote that cannot be located is **kept with a warning**
  (`WARN (quote not in doc)`) — the warning is informational and does **not**
  recolor or drop the triple. A missing quote is kept.
- **strict** — drop a triple with no quote (`DROP (no quote)`) or whose quote
  cannot be located in the document (`DROP (quote not in doc)`).

Multi-segment quotes (LLM-stitched `" / "` fragments, or a list intro paired with
a non-adjacent bullet) are re-anchored per segment; the joined verbatim segments
are stored only if **every** segment locates, otherwise the whole quote falls to
warn/drop.

Entity-level hallucination is caught downstream, not here:

- the **semantic-check agent** colors a triple whose entity isn't supported by the
  quote yellow/red (it never drops);
- the **agentic retry** path hard-rejects a triple when the raw extracted term is
  absent from the document AND the quote is also unlocatable. The raw term is
  checked pre-normalization, so "kidney stones" passes even though UMLS resolves
  it to "Nephrolithiasis"; a real entity with only a bad quote is kept and
  red-flagged by `FabricatedQuoteValidator`, not dropped.

**Skip report (`--skip-report`)**

`skip_log` records **entity-resolution failures only** — a UMLS/GLEIF lookup that
returned no usable result (`SKIP (lookup failed)`). `--skip-report` prints a table
at the end with columns: Document, Rel, Node, Term, Reason (default
`reason = "lookup failed"`). Quote-grounding drops are logged to the console
(`DROP (no quote)` / `DROP (quote not in doc)`) but are not part of the skip
report.

### Schema evolution warning

If a property is added to `schema.yaml` and the pipeline runs against an
existing database, the backend detects the drift and prints:

```
  [WARN] schema drift on 'Label': column(s) ['new_col'] defined in schema.yaml but absent from the DB.
  Delete the database file and re-run to apply schema changes.
```

Drop the `.db` file and re-run to apply the updated schema.

---

## Deduplication policy

Understanding this policy is critical before changing any persistence code.

### Nodes — no duplicates (first-write-wins)

`upsert_node` checks whether a node with the same primary key already exists
before writing. If it does, the write is skipped entirely. Re-runs on the same
documents are idempotent.

### Edges — one per `(from, rel_type, to, source_doc)`, last-write-wins

Every edge automatically carries a `source_doc` property set to the originating
document path (the `doc` field from the review JSON). The uniqueness key is the
4-tuple `(from_pk_value, rel_type, to_pk_value, source_doc)`:

- **Different source documents, same node pair** → two separate edges, each with a
  different `source_doc`. Multi-document provenance is preserved.
- **Same source document re-run** → the existing edge for that `source_doc` is
  replaced with current properties (last-write-wins). No duplicates accumulate.

Ladybug: `DELETE` the matching edge by `source_doc`, then `CREATE` fresh — wrapped in an explicit transaction with rollback on failure, so a failed `CREATE` no longer leaves a silently deleted edge.
Neo4j: same source-scoped `DELETE` then `CREATE` — deletes the existing edge matching `(from, rel_type, to, source_doc)`, then creates a fresh one with current properties. Prevents stale property accumulation when a document is re-extracted.

`source_doc STRING`, `run STRING`, and `supporting_quote STRING` are injected automatically into every relationship table by `setup()` — no schema YAML change required.

Within a single document run, the pipeline's `_edges` dict deduplicates
`(from_pk, to_pk)` pairs before `apply_graph.py` runs, so no intra-document
duplicates reach the graph layer.

> **Existing Ladybug databases** built before this change lack the `source_doc` column.
> The backend detects this and falls back to unconditional `CREATE` (old behavior).
> Delete the `.db` file and re-run to pick up the new schema and idempotent re-runs.

---

## Incremental / snowball KG building

The graph is designed to grow across runs and years without re-extracting old
documents. The deduplication policy above makes this safe: duplicate node PKs are
silently skipped; edges from different source documents coexist; re-running the
same document replaces only that document's edges.

### Two patterns

**Pattern 1 — Copy DB forward (UI workflow)**

Copy a previous run's `*_kg.db` into the new run folder before clicking *Run Write*:

```bash
cp projects/drug/runs/20260524_090000/drug_kg.db \
   projects/drug/runs/20260525_170300/drug_kg.db
```

Then run Step 3 normally. New documents' nodes and edges are appended to the
copied database. No extraction steps need to be re-run for old documents.

**Pattern 2 — Shared DB (CLI workflow)**

Point `apply_graph.py` at one DB that lives outside the runs folder:

```bash
python apply_graph.py \
  --schema projects/drug/schema.yaml \
  --apply  projects/drug/runs/20260525_170300/drug_review.json \
  --db     projects/drug/drug_shared.db \
  --run-id 20260525_170300
```

All runs accumulate into `drug_shared.db`. Pass `--run-id` to stamp each node
and edge with the run that introduced it.

### What the `run` field tells you

Every node and edge carries a `run` column (auto-injected by `setup()`):

| Entity | Value |
|--------|-------|
| Node | `run_id` of the run that **first** introduced this entity (first-write-wins) |
| Edge | `run_id` of the run that **last wrote** this edge for this `source_doc` |

Query example:

```cypher
// Which run first introduced each substance?
MATCH (n:Substance) RETURN n.name, n.run ORDER BY n.run

// All edges written in a specific run
MATCH ()-[r:HAS_ADVERSE_EFFECT]->() WHERE r.run = '20260525_170300' RETURN r
```

### Caveats

- **Node identity is PK equality.** If the same real-world entity resolves to a
  different CUI/LEI across runs (OCR artefact, UMLS API variance), a second node
  is created rather than merged. External-authority resolution minimises this but
  does not eliminate it.
- **Schema changes are not applied to existing tables.** `CREATE NODE TABLE IF NOT
  EXISTS` skips creation when the table already exists — new columns won't appear.
  Drop and rebuild the DB if `schema.yaml` gains new properties.

---

## Querying the graph

### CLI dump

Prints all nodes and edges from the schema:

```bash
python query_graph.py --schema schemas/supplychain_schema.yaml --db supplychain_kg.db
```

### Interactive Cypher

Use LadybugDB's own explorer, or write openCypher queries directly:

```cypher
-- All corporations
MATCH (n:Corporation) RETURN n.lei, n.name ORDER BY n.name

-- Supply relationships with product list and review color
MATCH (a:Corporation)-[r:PROVIDES]->(b:Corporation)
RETURN a.name, r.products, b.name, r.triple_color

-- Filter by review status (green = high confidence, yellow = moderate, red = flagged)
MATCH (a:Substance)-[r:MAY_TREAT]->(b:Indication)
WHERE r.triple_color = 'green'
RETURN a.name, b.name, r.source_doc

-- Node / edge counts
MATCH (n) RETURN label(n), count(*) ORDER BY label(n)
```

---

## MCP server

`mcp_server.py` exposes the graph and source documents as an MCP server so any
MCP-compatible AI client can query them using tools. Supports both Ladybug and Neo4j
backends.

Graph queries (`run_cypher`) handle structured factual questions. Just-in-time document
tools (`list_documents`, `search_document`, `read_document`) give the AI client access
to the full source text stored in `*_raw.json` — no vector store or embeddings required.

**Retrieval priority:** the server instructs the AI client to call `run_cypher`
first for factual questions (who, what, how many, named relationships) and fall back
to `search_document` / `read_document` for narrative questions (dosage instructions,
warnings, contraindications, summaries) or when the graph returns no relevant results.

### Tools exposed

| Tool | Description |
|------|-------------|
| `run_cypher` | Execute any openCypher query against the graph; returns JSON rows |
| `get_node_count` | Count nodes by label |
| `get_schema` | Returns the full graph schema (node tables, relationship tables, property types) |
| `list_documents` | List all source documents available for JIT context retrieval |
| `search_document` | Keyword search within a document's full text; returns matching passages with context |
| `read_document` | Read paginated full text of a document (offset + limit) |

The schema is also embedded in the MCP server `instructions` so AI clients have it from the first message — no tool call needed to learn node/relationship names.

### Just-in-time document context

Source document text is stored in `doc_text` inside each `*_raw.json` file — the same
text fed to the LLM during extraction. `DOCS_DIR` defaults to the parent of `LADYBUG_DB_PATH`
(the run folder), so no extra configuration is needed if the db and raw files are in
the same run directory.

If you extracted only selected pages from a PDF (via `--meta` page filters), JIT sees
only those pages' text.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GRAPH_BACKEND` | `ladybug` | Backend: `ladybug` or `neo4j` |
| `GRAPH_SCHEMA` | *(required)* | Path to schema YAML |
| `LADYBUG_DB_PATH` | *(required for ladybug)* | LadybugDB database directory |
| `DOCS_DIR` | parent of `LADYBUG_DB_PATH` | Directory containing `*_raw.json` files for JIT context |
| `CYPHER_TIMEOUT_S` | `30` | Max seconds a `run_cypher` query may run before being cancelled |
| `CYPHER_MAX_ROWS` | `1000` | Max result rows returned by `run_cypher`; excess rows are truncated with a `__truncated__` notice |

For Neo4j, set `GRAPH_BACKEND=neo4j` and add `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`.

### Configure for Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` and add
an entry under `mcpServers`. The `env` block is **required** — without it the
server inherits the shell environment and may pick up a wrong `GRAPH_BACKEND`.

```json
{
  "mcpServers": {
    "HELMS-kg": {
      "command": "/path/to/cognee_poc/.venv/bin/python3",
      "args": ["/path/to/cognee_poc/mcp_server.py"],
      "env": {
        "GRAPH_BACKEND": "ladybug",
        "GRAPH_SCHEMA": "/path/to/cognee_poc/projects/drug/schema.yaml",
        "LADYBUG_DB_PATH": "/path/to/cognee_poc/projects/drug/runs/20260530_214907_930/drug_kg.db"
      }
    }
  }
}
```

- `DOCS_DIR` is omitted — defaults to the run folder (parent of `LADYBUG_DB_PATH`), which is correct when `*_raw.json` files are in the same directory as the db.
- Sensitive credentials (`UMLS_API_KEY`, `LLM_API_KEY`, etc.) are loaded from the project `.env` by `load_dotenv()` at startup — no need to repeat them here.
- Update `LADYBUG_DB_PATH` after each new extraction run to point at the latest run folder.
- **Fully quit and reopen Claude Desktop** after editing — the `env` block only takes effect on restart.
- To temporarily disable the server without losing the config, move the entry outside `mcpServers` (e.g. to `"_mcpServers_disabled"`).

### Configure for VS Code (Kilo Code / Claude Code extension)

`.vscode/mcp.json` in the project root:

```json
{
  "servers": {
    "HELMS-kg": {
      "type": "stdio",
      "command": "/path/to/cognee_poc/.venv/bin/python3",
      "args": ["/path/to/cognee_poc/mcp_server.py"]
    }
  }
}
```

VS Code extensions inherit the workspace environment, so `.env` is loaded
automatically by `mcp_server.py` via `python-dotenv`.

The MCP server opens LadybugDB in **read-only mode**, so multiple clients can
connect simultaneously without file-lock conflicts.

### Example queries

Once connected, ask the AI client questions that combine graph and document retrieval:

```
# Graph queries (run_cypher)
"What drugs are in the knowledge graph?"
"Which drugs target the same disease?"
"What side effects are associated with semaglutide?"

# JIT document context (search_document / read_document)
"Search the BEYFORTUS label for contraindications."
"What do the documents say about dosage for nirsevimab?"
"Find all mentions of adverse events across the documents."

# Cross-tool reasoning
"The graph shows drug A treats disease B — which document supports that, and what does it say exactly?"
```

---

## MCP ingest server

`mcp_ingest_server.py` is a companion MCP server that lets any MCP-compatible
AI client (Claude Desktop, VS Code) ingest a PDF into the knowledge graph by
tool call — no web UI or CLI required.

When the user shares a PDF file path, Claude calls `ingest_pdf`. The full
pipeline (convert → extract → apply) runs in a background thread. Claude
polls `poll_ingest` every ~30 seconds until the run finishes, then reports
the result (triples added, entity resolution rate, or error).

### Tools exposed

| Tool | Description |
|------|-------------|
| `ingest_pdf` | Start a background ingest run for a PDF file path. Returns `run_id` immediately. |
| `poll_ingest` | Check status of a run (`running` / `done` / `error`) plus extraction stats when done. |
| `list_ingests` | List all past and current ingest runs, newest first. |

### Run folder structure

Each ingest creates a timestamped run folder identical to a UI run:

```
projects/supplychain/
├── schema.yaml
├── supplychain_kg.db        ← shared DB; all ingests accumulate here
└── runs/
    └── 20260601_120000_123/  ← one folder per ingest_pdf call
        ├── report.md                ← converted Markdown
        ├── report_raw.json          ← immutable LLM output
        ├── report_review.json       ← event log
        ├── run.jsonl                ← structured pipeline events (doc_done, semantic_check_done, …)
        ├── run_config.json          ← params + extraction_stats (written by extract.py)
        ├── pipeline.log             ← full pipeline stdout captured for debugging
        └── _ingest_status.json      ← ingest server run status
```

The shared DB (`supplychain_kg.db`) is separate from the run folder so
`mcp_server.py` can point at a **stable path** that accumulates across ingests
without needing config updates after each run.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INGEST_PROJECT` | *(required)* | Path to the project folder (must contain `schema.yaml`) |
| `INGEST_SCHEMA` | `<INGEST_PROJECT>/schema.yaml` | Path to schema YAML |
| `INGEST_DB` | `<INGEST_PROJECT>/<schema_stem>_kg.db` | Shared LadybugDB that all ingests accumulate into. Point `mcp_server.py LADYBUG_DB_PATH` here. |
| `INGEST_CONVERTER` | `pymupdf4llm` | Default PDF converter (`pymupdf4llm` or `llamaparse`) |
| `INGEST_FILTER` | `moderate` | Default filter level (`loose` / `moderate` / `strict`) |

### Configure for Claude Desktop

Add a second entry alongside the existing `HELMS-kg` read server:

```json
{
  "mcpServers": {
    "HELMS-kg": {
      "command": "/path/to/cognee_poc/.venv/bin/python3",
      "args": ["/path/to/cognee_poc/mcp_server.py"],
      "env": {
        "GRAPH_BACKEND": "ladybug",
        "GRAPH_SCHEMA": "/path/to/cognee_poc/projects/supplychain/schema.yaml",
        "LADYBUG_DB_PATH": "/path/to/cognee_poc/projects/supplychain/supplychain_kg.db"
      }
    },
    "HELMS-kg-ingest": {
      "command": "/path/to/cognee_poc/.venv/bin/python3",
      "args": ["/path/to/cognee_poc/mcp_ingest_server.py"],
      "env": {
        "INGEST_PROJECT": "/path/to/cognee_poc/projects/supplychain"
      }
    }
  }
}
```

- Both servers point at the same DB (`supplychain_kg.db`): the read server opens it read-only; the ingest server writes to it.
- `INGEST_DB` is omitted — defaults to `supplychain_kg.db` inside `INGEST_PROJECT`.
- After the first ingest the shared DB exists and `mcp_server.py` serves it immediately — no config update needed for subsequent ingests.
- **Fully quit and reopen Claude Desktop** after editing.

### Example session

```
User:  Here is the new report: /Users/me/Downloads/tsmc_q1_2026.pdf
       Please add it to the supply chain graph.

Claude: [calls ingest_pdf("/Users/me/Downloads/tsmc_q1_2026.pdf")]
        → run_id = "20260601_120312_045", status = "started"

        [30 s later, calls poll_ingest("20260601_120312_045")]
        → status = "done", total_triples = 18, entities_resolved = 14,
          entity_resolution_rate = 0.78

        Done — 18 triples added (14 entities resolved, 4 unresolved).
        The graph now contains the new TSMC supply relationships.
        Ask me anything about it.
```

---

## Harvest self-improvement

After every Step 3 write, `agents/harvest.py` rebuilds the harvest store at `<project>/harvest/` — one JSONL file per relationship type. The rebuild runs in a background thread triggered by the SSE sentinel when the apply step succeeds (`step=="apply"` and `ok==True`). The HTMX UI also triggers a rebuild after every Agent Retry save, and the CLI's `agents/extraction_agent.py` re-builds the store inline at the end of `_agent_retry_main` so the CLI matches the UI's behavior. The store accumulates reviewed triples across all runs and is used to improve future extractions via few-shot prompt injection.

### Priority tiers

| Priority | Source | Meaning |
|----------|--------|---------|
| 1 | `add` | Human wrote this triple from scratch |
| 2 | `override` | Human corrected an existing triple |
| 3 | `agent_retry` | Batch missed it; agentic retry got it right |
| 4 | `batch` | Easy first-pass green triple |

### How injection works

On the next extraction run, `extract.py` loads the harvest store for each rel_type and injects:

- **Positive examples** → system prompt (few-shot anchors, diversified by target entity)
- **Rejected examples** → prepended to **every chunk's user message** (highest-recency position, immediately before the document text)

The rejection reminder format:

```
⚠ REJECTION LIST for MAY_TREAT — these triples were explicitly rejected by a human reviewer.
Do NOT extract any of these unless the document text below contains evidence that is MORE EXPLICIT and DIRECT than the originally rejected quote.
  • (nirsevimab) -[MAY_TREAT]-> (Acute lower respiratory infection caused by respiratory syncytial virus)
    Rejected from doc: beyfortus label_raw.json
    Rejected quote: "Children up to 24 months of age who remain vulnerable..."
```

### Doc-specific semantics

Rejections are **document-specific**, not global bans. The LLM is instructed it may re-extract a rejected triple if the current document provides clearly stronger or more direct evidence. This allows the same relationship to be correctly captured from a different source document.

### Cache interaction

The chunk cache key does not include harvest state. If a chunk was cached before a rejection, it will be served stale. Use `--force` (or the Force re-extract checkbox in the UI) after rejecting a triple to ensure the updated rejection signal takes effect.

---

## Evaluation harness

`eval_extraction.py` computes precision, recall, and F1 against a gold-standard TSV stored per project at `projects/<name>/eval/gold_template.tsv`.

### Gold TSV format

```tsv
rel_type	from_display	to_display	doc_name	notes
MAY_TREAT	nirsevimab	RSV lower respiratory tract infection	beyfortus_label	section 1
HAS_TRADENAME	nirsevimab	Beyfortus	beyfortus_label	package header
HAS_ADVERSE_EFFECT	nirsevimab	injection site redness		corpus-level (any doc)
```

- `doc_name` (optional): stem of the `*_raw.json` file, e.g. `beyfortus_label` for `beyfortus_label_raw.json`. When present, extracted triples are filtered to that document only — prevents cross-doc false positives.
- Omit `doc_name` to match a triple found in any document (corpus-level).
- `notes` column is ignored; use for human reference.
- Matching is **case-insensitive** on `from_display` / `to_display`.

### Running evaluation

```bash
# Doc-scoped + corpus-level, moderate filter (matches Step 3 default)
python eval_extraction.py \
  --gold projects/drug/eval/gold_template.tsv \
  --run  projects/drug/runs/20260610_225405

# Strict filter only
python eval_extraction.py \
  --gold projects/drug/eval/gold_template.tsv \
  --run  projects/drug/runs/20260610_225405 \
  --filter strict

# One rel_type with verbose TP/FP/FN output
python eval_extraction.py \
  --gold projects/drug/eval/gold_template.tsv \
  --run  projects/drug/runs/20260610_225405 \
  --rel-type MAY_TREAT --verbose
```

Output example:

```
========================================================================
Run:    projects/drug/runs/20260610_225405
Gold:   projects/drug/eval/gold_template.tsv
Filter: moderate
========================================================================

Doc-scoped evaluation (8 gold triples across 2 document(s))
  beyfortus_label           P=1.000  R=0.875  F1=0.933  (TP=7 FP=0 FN=1 | gold=8 extracted=7)

  Per rel_type (doc-scoped):
    MAY_TREAT               P=1.000  R=1.000  F1=1.000  (TP=3 FP=0 FN=0 | gold=3 extracted=3)
    HAS_ADVERSE_EFFECT      P=1.000  R=0.800  F1=0.889  (TP=4 FP=0 FN=1 | gold=5 extracted=4)
```

---

## Git branches

| Branch | Description |
|--------|-------------|
| `main` | Full pipeline: multi-provider LLM (Azure/OpenAI/Anthropic/Ollama via LiteLLM), LadybugDB/Neo4j graph backends, JIT document context MCP, harvest self-improvement, evaluation harness |

---

## File reference

| File | Purpose |
|------|---------|
| `htmx_app/main.py` | FastAPI + HTMX + Alpine.js UI; `python htmx_app/main.py` → http://localhost:8000; SSE streaming for real-time pipeline output; schema editor (`/schema/load`, `/schema/save`) — edits node `description`/`sem_group`/`umls_vocabs`/`semantic_types` and rel `extract_prompt`/`from_hint`/`to_hint`/`examples`/property `hint`, validated with Pydantic before save; metadata editor (`/meta/load`, `/meta/save`) — edits `instructions` and per-PDF `pages` filters; all pipeline steps (Convert/Extract/Write) via `pipeline_runner.py`; ▶ Run selected steps orchestrator; triple review editor; ↺ Retry; **📄 Find in doc**: `GET /review/evidence` (path-traversal-guarded) renders the source `.md` with the triple's evidence highlighted via `_highlight_evidence` — quote yellow, subject (`from_term`) green, object (`to_term`) pink, color-coded by precedence so a term nested in a quote keeps its own color, scrolled to the first match; 📊 quality panel; graph summary; LLM model selector; runs dropdown; project creation wizard; open-run-in-Finder; `/run/step-status` endpoint restores step badges on run selection and page refresh; state persisted to `htmx_app/.helms_state.json`; per-project LLM cache via `KG_CACHE_DIR`; **runner dict pruned** on each `start_runner()` call — entries older than 1800 s or already-done removed, preventing unbounded `_runners` growth in long-running server processes; **`htmx_app/templates/partials/triples_list.html`** `data-triple-*` attributes double-quote-escaped (`replace('"', '&quot;')`) — prevents HTML attribute injection from LLM-generated `supporting_quote` values |
| `agents/harvest.py` | Harvest self-improvement store: `harvest_project(project_dir)` rebuilds `<project>/harvest/<rel_type>.jsonl` from scratch by scanning all runs; `load_examples()` returns top-k positives + negatives; `format_examples_block()` for system prompt; `format_rejection_reminder()` for per-chunk user-message injection with original doc + quote |
| `agents/base_resolver.py` | `NodeResolver` ABC: `source` class attribute (matches schema `source=` value), `handles(node_def)`, `collect_unique_entities()`, abstract `resolve_batch(entity_names, nodes, output_dir, ctx)` and `build_props()`; **`ResolveContext` dataclass** — per-run inputs threaded into every `resolve_batch` (`domain_hint` + `abbr_map`); add a field here to pass new per-document context to resolvers without changing every signature (mirrors `ValidatorContext`); shared `_log_error(output_dir, name, label, error, *, print_prefix="")` — writes JSONL entry to `_node_agent_errors.jsonl` and optionally prints to stdout |
| `agents/node_agent.py` | Node resolution orchestrator: `resolve_all_nodes(all_items, rels, nodes, output_dir, ctx)` dispatches to all registered resolvers concurrently; `ctx` is a `ResolveContext` (re-exported here; a bare `str`/`None` is accepted for back-compat and coerced to `domain_hint`); `domain_hint` (from `meta.yaml` `instructions`) lets resolvers weight candidates by domain/geography, `abbr_map` (from `extract._extract_abbreviations`) lets GLEIF expand doc-defined abbreviations without an LLM; `resolver_for_node(node_def)` returns the resolver that owns a node definition. UMLS resolver cascade search (words→normalizedWords→normalizedString) with `page_size=25`; **exact-name shortcut**: if any candidate matches the raw entity name exactly (case-insensitive), returned immediately — no LLM call; **semantic type guard on exact match**: when `sem_group`/`sem_types` active, exact-name match verified against expected types — falls through to LLM if not satisfied; LLM `_CandidatePick` disambiguation invoked when no valid exact match; **LLM rejection (index=0)**: LLM may return 0 to reject all candidates — logged to `_node_agent_errors.jsonl` with reason "no acceptable UMLS match — dropped by LLM"; `asyncio.wait_for` timeout on LLM calls; retry-once on server error |
| `agents/semantic_check_agent.py` | LLM semantic grounding: batches triples in groups of 25, asks LLM to color each entity green/yellow/red based on presence in document text and supporting quote; LLM returns `constraint_violated: bool` — set true when schema `extract_prompt` constraint is violated (e.g. adverse effect observed only in rodents) **or** when actual UMLS semantic types don't match `semantic_types` from schema; `triple_color = red` when `constraint_violated` is true or either entity is red, regardless of entity grounding; adds `ai_opinion` (1–2 sentences covering color rationale + violations); **`_build_items` injects `{from,to}_expected_semantic_types` / `{from,to}_actual_semantic_types`** into each batch item — LLM opinion mentions type mismatches explicitly; deterministic structural checks (empty PKs, duplicate triples, procedure-marker mismatch, **UMLS semantic type mismatch**, **schema conformance**) appended to opinion; **type mismatch check runs at all filter levels** (not just `strict`) when schema defines `semantic_types`, and is now a **hard red floor + `constraint_violated`** (provable, no longer dependent on the LLM acting on the warning; vocab-only mismatch stays note-only); **`SchemaConformanceValidator`** reds any triple whose `(rel_type, from_label, to_label)` is not a declared edge in `schema_rels` (no-op when no derivable edges); deterministic validators are pluggable `TripleValidator` subclasses in `_DETERMINISTIC_VALIDATORS` whose `color_floor`s merge onto the LLM base color via `worst_color` (red>yellow>green); **`harvest_dir` param**: after all LLM + structural checks, scans `harvest_dir/*.jsonl` for `source="rejected"` entries — any triple matching a rejected logical key `(rel_type, from_display, to_display)` is forced red + `constraint_violated=True` + `"[Previously rejected by human reviewer]"` in `ai_opinion`; deterministic, no LLM, overrides even a green LLM verdict; works on cache-hit runs; both `extract.py` and `htmx_app/main.py` pass `harvest_dir` to `check_triples`; LLM client created once per `check_triples` call (not per batch); reads `_LLM_MAX_COMPLETION_TOKENS` **live from the `extract` module** via `getattr(_extract_mod, "_LLM_MAX_COMPLETION_TOKENS", 8192)` at call time — picks up env overrides patched by `pipeline_runner.py` without a stale module-level copy; always called automatically from `extract.py` (after node agent) — no manual trigger needed |
| `pipeline.py` | One-command orchestrator: convert PDFs → extract triples → apply to graph; `--filter` value correctly forwarded to `apply_review`; `output_dir` correctly passed to extraction step (was missing, causing review JSONs to land in wrong folder); `--meta` passes metadata to all steps; PyMuPDF (fitz) import wrapped with informative `SystemExit` when page filtering is requested but package is absent; per-document extraction failures logged with `[error]` prefix; failure count summary printed after `asyncio.gather`; `SystemExit` raised if all documents fail |
| `pipeline_meta.py` | Loads `--meta` YAML; provides `get_instructions()` and `get_page_filter()`; used by all pipeline scripts; malformed page-range inputs (`_expand_page_ranges`) skip with a warning instead of crashing; reversed ranges (e.g. "10-5") warn and skip instead of silently producing empty page lists |
| `convert_pdf.py` | PDF → Markdown converter (pymupdf4llm or LlamaParse); `--meta` enables page filtering; LlamaParse response validated for non-empty `result.markdown.pages` before accessing — raises `RuntimeError` with clear message instead of `AttributeError`; LlamaParse client singleton init is thread-safe via `threading.Lock()`; `_to_range_str` deduplicates via `sorted(set(pages))`; corrupted PDFs in batch are warned and skipped instead of aborting all conversions; LlamaParse uploaded file objects are cleaned up via `client.files.delete()` in `finally` block |
| `extract.py` | LLM extraction + UMLS/GLEIF enrichment → writes `<stem>_raw.json` (immutable) + `<stem>_review.json` (event log); **single-agent extraction**: one structured LLM call per chunk using the open-minded extractor prompt from `prompts.yaml` (maximises recall); LLM assigns `evidence_level` per triple (`strong`/`moderate`/`weak`); **semantic chunking**: splits on `#`/`##`/`###` headers first, falls back to char-overlap for oversized sections; post-chunk batch UMLS resolution via `agents/node_agent.py`; **always-on semantic check** via `agents/semantic_check_agent.py` after node agent; `filter_level` controls DB write threshold only (not injected into extraction prompts); concurrent multi-doc via `asyncio.gather` with `asyncio.Semaphore(args.concurrency)`; progress sentinel `[extract:progress] X/N` emitted when each doc starts processing (not completes) so in-flight progress is visible; `asyncio.wait_for` timeout on LLM calls; **Pydantic schema validation** at load time — rejects invalid YAML fields before any LLM calls; LLM structured-output call retries up to 3 attempts with exponential backoff (1 s, 2 s); `--meta` injects instructions into LLM prompt; NFC Unicode normalization in quote grounding; **quote grounding filter** (`_verify_grounding`, before resolution): quote-based only — re-anchors located quotes to verbatim text, `strict` drops no-quote / quote-not-in-doc, `moderate` warns + keeps, `loose` keeps all (entity-presence is a semantic-check coloring concern, not a drop); **empty extractions are not cached**: `_run_chunk` calls `_cache_save` only when the chunk produced ≥1 triple, so a transient empty LLM response no longer poisons every future non-`--force` run (the bug that produced 0 triples corpus-wide); a whole-doc zero with no failed chunks prints a `[warn]` to re-run; `--skip-report` table (Document / Rel / Node / Term / Reason) records resolution failures only; `_compute_extraction_stats()` aggregates after each run (docs, triples, colors, entity resolution) and writes to `run_config.json["extraction_stats"]`; **`.cache` directory** anchored to the script's own directory (`Path(__file__).parent / ".cache"`) — safe to run from any working directory, including when launched by Claude Desktop with `cwd=/`; **thread-local async client** (`_llm_client_local = threading.local()`) — each thread creates its own `AsyncAzureOpenAI` instance, preventing shared event-loop errors when multiple threads (UI runners, MCP ingest threads) each call `asyncio.run()` concurrently |
| `apply_graph.py` | Reads `_raw.json` + `_review.json` via `review_layer.materialize()` → writes to graph backend; validates PK value presence before any graph writes; `supporting_quote` from each triple written as an edge property alongside `rel_props` and `source_doc`; **stale-edge deletion**: color-filtered and rejected triples each get `delete_edge(..., source_doc=doc_name)` before the write loop — prevents stale edges after reject-then-rewrite; deletion is scoped to `source_doc` so edges from other documents are never removed |
| `review_layer.py` | Immutable/mutable separation: `_raw.json` written by `extract.py`, never modified; `_review.json` is an event log (ACCEPT / REJECT / OVERRIDE / ADD). Public API: `materialize(raw_data, events)` → effective triple list; `save_events(review_path, raw_path, events)`; `get_conflicts(raw_data, events, stored_hash, raw_path)` → detect when `_raw.json` changed since last review save. `raw_path_for()` / `review_path_for()` helpers derive sibling paths. **ADD triples get `_id` restored from their event key** on materialize — saves stripped `_` prefixed fields from the event log back onto the triple so the review editor can key on them without `KeyError`. **OVERRIDE events carry `triple_color`** — user-clicked color cycle (green→yellow→red) in the HTMX review UI is persisted; `materialize()` applies it so a red triple promoted to green by the user passes the `moderate` write filter in Step 3 |
| `llm_client.py` | Multi-provider LLM client. Every provider (azure / openai / anthropic / ollama / gemini) is routed through `litellm.acompletion` with `response_format=response_model` and Pydantic `model_validate_json()` — no per-provider SDK branches. `get_provider()` detects `LLM_PROVIDER` env var or infers `azure` from `LLM_ENDPOINT`; `load_config()` reads `.env.yaml` (priority: shell > `.env.yaml` > `.env`); `get_models()` returns model names from `models:` block (or `[current_env_model]`); `get_model_env(model_name)` returns env-var overrides for switching deployments at runtime; `LLM_TIMEOUT` (float, default 120 s); `LLM_MAX_COMPLETION_TOKENS` (int, default 8192 — override to 65536 in `.env.yaml` for gpt-5.x / o-series reasoning models); `LLM_REASONING_EFFORT` (str, default `"low"`) passed to reasoning models (o-series, gpt-5.x); `is_reasoning_effort_ok()` / `clear_reasoning_effort()` — process-wide flag cleared when a deployment rejects the parameter, so subsequent calls skip it automatically; `_reasoning_effort` recovery runs before the fatal auth-error branch so a single deployment's rejection doesn't shadow the auto-retry path. **`acreate_structured_output(text_input, system_prompt, response_model, model=None, max_completion_tokens=8192, timeout=120.0, retries=3, base_delay=1.0, log_prefix=None)`** — shared async LLM retry loop (exponential backoff: `base_delay * 2**(attempt-1)`) used by every caller that needs structured output. The `_get_litellm_model` helper prefixes the bare model name with the provider (`azure/`, `anthropic/`, `ollama/`, `gemini/`) and `_ensure_litellm_env` clears stale provider API-key env vars before each call so switching providers at runtime does not leak credentials. `make_client()` / `make_async_client()` are retained for callers that still want a raw SDK client. |
| `lookup_cache.py` | Persistent SQLite cache for UMLS and GLEIF lookups. Two-level strategy: L1 = module-level dict (fast, lost on restart); L2 = SQLite (persisted to `lookup_cache.db` across restarts). Thread-safe via per-thread `sqlite3.Connection`; WAL mode for concurrent reads. Caches successful API responses only — timeouts and network errors are not cached. DB path: `LOOKUP_CACHE_DB` env var or `<project_root>/lookup_cache.db`. **TTL** via `LOOKUP_CACHE_TTL_DAYS` (default `30` days; set `0` to disable); `get()` treats rows older than the TTL as misses and deletes them so stale registry data self-refreshes. Public API: `get(service, key)`, `put(service, key, value)`, `stats()`, `evict(pattern, service=None)`, `clear(service=None)`. **Maintenance CLI**: `python -m lookup_cache <stats\|evict <substring>\|clear> [--service X]` — inspect counts/age, drop a single poisoned entry, or wipe a service (see "Lookup cache maintenance" below) |
| `pipeline_runner.py` | In-process pipeline runner — replaces `subprocess.Popen` for extract/convert/apply steps in both UIs. Thread-local stdout/stderr router: main thread passes through to original streams; background runner threads write to a `queue.Queue` that the SSE handler reads line by line. Mimics the subprocess interface (`.poll()`, `.wait(timeout)`, `.terminate()`, `.kill()`, `.returncode`) without spawning a new process; `_env_lock` serialises `os.environ` save/update/restore so concurrent runners don't race; `.terminate()`/`.kill()` stop the asyncio event loop via `loop.call_soon_threadsafe(loop.stop)`; **`_QueueWriter.flush()` called before the `None` sentinel** — pushes any partial last line (output that didn't end in `\n`) into the queue before the consumer sees EOF, preventing the final log line from being silently dropped. **Module-level env-override patches** (for the duration of each run, with restore in `finally:`): `extract._CACHE_DIR`, `extract._LLM_MODEL`, `extract._LLM_MAX_COMPLETION_TOKENS`, and **also** `llm_client.LLM_MAX_COMPLETION_TOKENS` + `llm_client.LLM_TIMEOUT` — so callers that read live from the `llm_client` module (e.g. `extraction_agent.run_extraction`) see the same env override the runner received. Runners older than 30 minutes or already-done are pruned from the HTMX UI's `_runners` dict on each `start_runner()` call to prevent unbounded growth. |
| `mcp_server.py` | MCP server for AI clients; tools: `run_cypher` (graph), `get_schema`, `get_node_count`, `list_documents`, `read_document`, `search_document` (JIT doc context from `doc_text` in `*_raw.json`); `DOCS_DIR` defaults to parent of `LADYBUG_DB_PATH`; lazy init via `_server_init()`; `mcp._mcp_server.instructions` set at init time with schema appended; Neo4j backend opened with `read_only=True`; `run_cypher` write-verb check strips string literals before regex match — prevents bypass via `'CREATE'` in a string value; **write-verb regex extended** to cover `FOREACH`, `LOAD`, `CALL {}`, and bare `;` (multi-statement injection) — all checked on the string-literal-stripped query so `WHERE name = 'a;b'` is safe; module-level `ThreadPoolExecutor` reused across calls and shut down via `atexit`; per-query timeout cancels the future without killing the shared pool |
| `mcp_ingest_server.py` | MCP ingest server — companion to `mcp_server.py`; tools: `ingest_pdf(pdf_path, converter, filter_level)` starts full pipeline (convert → extract → apply) in a background daemon thread, returns `run_id` immediately; `poll_ingest(run_id)` returns status (`running`/`done`/`error`) + extraction stats parsed from `run.jsonl` (`doc_done` event → `triple_count`, `semantic_check_done` → `color_dist`); `list_ingests()` scans `<INGEST_PROJECT>/runs/` for past runs; all pipeline stdout redirected to per-run `pipeline.log` (never written to the MCP JSON-RPC channel); all ingests write to a single shared DB (`INGEST_DB`) so `mcp_server.py` can point at a stable path; per-run folder contains `<stem>_raw.json`, `<stem>_review.json`, `run.jsonl`, `run_config.json`, `pipeline.log`, `_ingest_status.json`; millisecond suffix in run_id prevents collision on rapid successive calls; env vars: `INGEST_PROJECT` (required), `INGEST_SCHEMA`, `INGEST_DB`, `INGEST_CONVERTER`, `INGEST_FILTER` |
| `backends/base.py` | `GraphBackend` abstract interface; `_safe_ident` + `_primary_key` shared helpers (deduped from ladybug + neo4j); `delete_edge()` abstract method accepts optional `source_doc: str | None` for scoped deletion; `get_backend(name, db_path, nodes, rels, read_only=False, setup=True)` factory |
| `backends/ladybug_backend.py` | LadybugDB implementation of `GraphBackend`; property keys validated with `_safe_ident()` (imported from `backends.base`) before Cypher interpolation — prevents Cypher injection via malicious key names; `setup()` auto-injects `source_doc STRING`, `run STRING`, `supporting_quote STRING`, `manually_added BOOLEAN`, and **`triple_color STRING`** into every rel table — `triple_color` enables NVL / Cypher edge styling by review verdict without schema YAML changes; **`_node_allowed_props`**: built in `setup()` as `label → set(column_names)`; `upsert_node()` filters `props` through this set — prevents LadybugDB column-not-found errors when a resolver returns extra fields not declared in schema.yaml; unknown rel props filtered against schema-declared columns before `CREATE`; same-doc re-run uses DELETE-then-CREATE wrapped in an explicit transaction with rollback on failure; `upsert_node` uses atomic `CREATE` with string-matched exception swallowing for duplicate PK; **real `close()`**: `del self._conn` + `del self._db` releases LadybugDB's background subprocess and file lock — required before the next Step 3 write; **read-only per-query reopen**: when constructed with `read_only=True`, `_execute` reopens `kuzu.Database` + `kuzu.Connection` on every call so queries always see the latest checkpoint — avoids stale reads and does not hold a write lock; **source-aware `delete_edge`**: when `source_doc` is provided, Cypher filter `{source_doc: $sd}` is added to the rel pattern so only that document's edges are deleted |
| `backends/neo4j_backend.py` | Neo4j implementation of `GraphBackend`; `upsert_node` uses atomic `MERGE ... ON CREATE SET` — first-write-wins for nodes; `create_edge` uses source-scoped `DELETE` then `CREATE` (matches LadybugDB behavior) — prevents stale property accumulation on re-runs while preserving edges from other documents; `read_only` constructor param enforces `READ` access mode at driver level |
| `backends/__init__.py` | `get_backend()` factory — selects backend at runtime |
| `query_graph.py` | CLI graph dump; returns `r.triple_color` for every relationship |
| `eval_extraction.py` | Precision/recall/F1 evaluation script; `--gold <tsv>` `--run <run_dir>` `--filter` `--rel-type` `--verbose`; doc-scoped + corpus-level matching |
| `projects/<name>/eval/gold_template.tsv` | Per-project gold standard (fill before running `eval_extraction.py`) |
| `skills/helms-schema/SKILL.md` | Claude Code `/helms-schema` skill; generates `schema.yaml` from Q&A; install to `~/.claude/skills/helms-schema/SKILL.md` |
| `agents/extraction_agent.py` | **Used by HTMX UI Step 2 🥷🏻 Agent Retry** and by the CLI (`python agents/extraction_agent.py --rel-type ...`). Triggered per rel_type for one document when batch extraction missed or returned wrong-entity triples; LLM calls gleif_search/umls_search/add_triple tools iteratively; adapts `searchType` per term (`words`→`normalizedString`→`rightTruncation`); honors schema `umls_vocabs` (sabs), `sem_group`, and `semantic_types` constraints injected into every `umls_search` call; `--filter loose\|moderate\|strict` sets grounding strictness; partial triples auto-saved on 60-iteration limit or LLM API failure; **`save_review()` writes `*_raw.json`** with unfiltered triples and stable content-hash `_id` values (`"f" + sha256(rel_type|from_pk_value|to_pk_value)[:12]`) — makes agentic output visible to the HTMX review UI which only globs `*_raw.json`; **`_do_merge()` upserts agent triples into the existing `*_raw.json` by `_id`**; **`rel_props` merge is per-key** (`{**prev_rp, **at_rp}`) so batch-set rel props (`source` / `publication_date` / `manually_added`) survive the agent re-extract and only the agent-supplied keys override; **near-duplicate quote guard** in `add_triple` strips trailing `.`/space before the in-check but stores the original (un-stripped) quote, so a 60-iteration retry does not accumulate identical segments; **`add_triple` stores `from_term`/`to_term`** in the entry dict when non-empty — enables `gleif_check.GLEIFResolutionValidator` to annotate agent-retry-added triples with GLEIF resolution warnings (matching batch path behavior). **Semantic check auto-runs after agent retry** via two paths: `_run_agent_semantic_check()` in `htmx_app/main.py` for the HTMX UI; `_cli_semantic_check()` in `agents/extraction_agent.py` for the CLI's `main()` (added so `python agents/extraction_agent.py --rel-type X` matches HTMX behavior). Both call `agents.semantic_check_agent.check_triples` and write the annotated triples (with `ai_opinion`, updated `triple_color`, and `constraint_violated`) back to the raw JSON. |
| `agents/resolver_tools_types.py` | Zero-dep shared types for the pluggable agentic tool registry. `ToolHandler = Callable[[session, args], str]`; `ResolverToolset(name, specs, handlers)` dataclass. Kept separate so `gleif_tools`/`umls_tools` can import `ResolverToolset` without creating a circular dependency with `resolver_tools.py` (which imports them). `resolver_tools` re-exports both for backward compat. |
| `schemas/supplychain_schema.yaml` | Finance domain schema (Corporation, PROVIDES) |
| `schemas/drug_schema.yaml` | Pharma domain schema |
| `project/drug_instruction.yaml` | Example pipeline metadata: extraction instructions + page filters |
| `prompts.yaml` | All editable LLM prompts: extraction agent system prompt, bulk extractor prompt (`extract.open_minded_system_prompt`, used by `extract.py` — single pass; the old dual open-minded/cautious scheme was dropped), filter_prompts (loose/moderate/strict), MCP server instructions (includes JIT retrieval strategy). Edit here and restart to apply. |
| `kg_logging.py` | Structured JSONL logging: `_JsonlHandler` writes one JSON object per event to `<run_dir>/run.jsonl`; `get_run_logger(output_dir)` returns idempotent logger; `NULL_LOGGER` drops events silently. `extract.py` emits 6 events per document: `doc_start`, `extraction_done`, `grounding_done`, `node_resolve_done`, `semantic_check_done`, `doc_done` |
| `probe_deployment.py` | Deployment probe: sends a minimal LLM call and reports latency, model name, token usage, and any auth errors. Useful for verifying `.env` credentials before a long extraction run |
| `UMLS/SemGroups.txt` | Pipe-delimited UMLS semantic group reference (`ABBR\|Group Name\|TUI\|Type Name`); loaded at import by `lookups.py` to build group→TUI and type-name→TUI mappings; source of truth for `sem_group` and `semantic_types` filtering |
| `UMLS/semantic_types.txt` | Pipe-delimited UMLS semantic type reference; reference file, not loaded by current code |
| `tests/test_schema.py` | Unit tests for schema parsing and dynamic Pydantic model building |
| `.env` | Credentials (git-ignored) — fallback when `.env.yaml` absent |
| `.env.yaml` | Structured credentials + model registry (git-ignored); all-caps scalar keys loaded as env vars; `models:` dict registers named deployments for the UI model selector; loaded by `llm_client.load_config()` at import |
| `.cache/` | LLM extraction cache (git-ignored; keyed by sha256(chunk+schema+strategy+instructions)) |
