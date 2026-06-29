# AGENTS.md — instructions for coding agents

This file is the cross-tool entry point for any AI coding agent (Codex, Cursor,
Aider, Kilo Code, OpenCode, Copilot, Gemini, Claude Code, …) working in this
repository.

## Read first, before editing or reviewing

1. **[DESIGN_INVARIANTS.md](DESIGN_INVARIANTS.md)** — deliberate design choices.
   Treat every item there as intentional. Do **not** flag an invariant as a bug
   and do **not** "fix" it. If one looks wrong, raise it as a question.
2. **[CLAUDE.md](CLAUDE.md)** — file map and detailed conventions (where code
   lives, naming, the resolver pattern, the review layer, etc.).
3. **[README.md](README.md)** — user-facing overview and setup.

## Ground rules

- **Honor the invariants.** The most common failure mode here is an agent
  reporting a documented trade-off (single-user concurrency, per-document edges,
  negative caching) as a critical bug. Verify against DESIGN_INVARIANTS.md first.
- **Verify before claiming.** This codebase has been mis-audited before. Read
  the actual source lines before asserting a bug exists; do not trust a prior
  report's line-logic claims.
- **Stay in scope.** Match surrounding style. No drive-by refactors, no renames,
  no new dependencies unless asked.
- **Tests.** Install the test deps once (`pip install -r requirements-dev.txt`),
  then `python -m pytest -q` must stay green (331 passed, 4 skipped at last
  check). Run it after changes.

## Credentials for tests (`.env.yaml`)

Real, working credentials live in **`.env.yaml`** at the repo root (Azure OpenAI /
Gemini LLM keys, `UMLS_API_KEY`, Neo4j, LlamaParse). You may use them to run tests.

- **They load automatically.** `llm_client.load_config()` runs at import and
  `setdefault`s the first model's keys (`LLM_ENDPOINT`, `LLM_API_KEY`, `LLM_MODEL`)
  plus the top-level `ALL_CAPS` keys into `os.environ`. So `python -m pytest -q`
  picks them up, and the **LLM-in-the-loop tests in
  `tests/test_semantic_entailment.py` actually run** (they self-skip only when no
  creds are found). To use them in a script/REPL, just `import llm_client` first.
- The `UMLS_API_KEY`-gated tests in `tests/test_agent_selection.py` likewise run
  when that key is present in `.env.yaml`.
- **SECURITY — do not leak these.** `.env.yaml` is gitignored and must STAY that
  way. Never commit it, never paste its contents (or any key from it) into source,
  test fixtures, logs, commit messages, PR descriptions, or chat. Reference creds
  only via `os.environ` / `llm_client`, never by hard-coding a value.

## Working practices (LLM tests, model choice, big changes)

- **Pick the right model for the job.** `.env.yaml` always carries at least one
  fast model (e.g. `gemini-3.1-flash-lite`) and one slow thinking model (e.g.
  `gpt-5.5`, the default first entry). Iterate and sanity-check on the fast model;
  reserve the slow model for the decisive run, the one where strict discrimination
  is the thing under test. A lenient fast model can hide a regression the strict
  slow model exposes (it was the slow model that surfaced the over-red AE bug), so
  the verdict run belongs there.
- **Important LLM test: no timeout, status every ~60s.** A grader/extraction test
  on the slow model can run for minutes. Do NOT cap it at the default 2-minute
  shell timeout (a kill is not a failure). Run it in the background (`nohup … &`)
  and poll, posting a short status line roughly every 60s until it finishes. Let
  it complete.
- **A/B test before any uncertain or large code change.** When a change is big or
  the right approach is genuinely in doubt (especially anything touching the
  semantic-check grader, prompts, or extraction behavior), do not just ship the
  hunch. Implement both arms behind one switch (e.g. an optional parameter so the
  same code runs A and B), define ground truth, and run them head-to-head on the
  discriminating model with n≥3 for stochastic stability. Ship the winner and
  record the result (the instructions-vs-title decision for single-subject
  detection is the worked example: B won 9/9 vs A 7/9 on `gpt-5.5`). A tie on the
  fast model is not a result; rerun on the slow model where the cases differ.

## Quick architecture

PDF → Markdown → LLM triple extraction → UMLS/GLEIF entity resolution →
semantic check (colors triples) → human review → graph write (LadybugDB / Neo4j) →
MCP query server. Schema-driven and domain-agnostic via `schema.yaml`.
