# HELMS Schema Boilerplate Skill

## Trigger
`/helms-schema`

## Purpose
Guide the user through defining a `schema.yaml` for the HELMS knowledge graph pipeline. Asks targeted questions about their domain, then generates a ready-to-use schema file.

## Instructions

When this skill is invoked:

1. **Greet and collect domain info** — ask these questions in one message:
   - What is the domain? (e.g. pharma, finance, biomedical, legal)
   - What are the main entity types (nodes)? List 3–8.
   - For each node: is its identity resolved via UMLS (biomedical), GLEIF (legal entities), or extracted directly by the LLM?
   - What relationships connect these entities? List 3–10 rel_types (use SCREAMING_SNAKE_CASE, e.g. `HAS_INDICATION`, `BRAND_OF`).
   - For each relationship: what properties should be stored on the edge? (e.g. dose, route, onset_date)

2. **Generate the schema.yaml** — produce a complete, valid HELMS `schema.yaml` using the rules below. Output it as a fenced YAML code block.

3. **Explain key decisions** — briefly note any assumptions made, especially about UMLS sem_group / semantic_types choices.

---

## Schema.yaml Rules

### Node definition rules
- Every node needs exactly **one** `primary_key: true` property.
- `source` must be one of: `umls`, `gleif`, `llm`, `pipeline`.
- UMLS nodes: primary key is `cui` (STRING), plus `name` (STRING). Set `sem_group` and optionally `semantic_types` and `umls_vocabs`.
- GLEIF nodes: primary key is `lei` (STRING), plus `name` (STRING). No sem_group needed.
- LLM nodes: primary key is a meaningful string field (e.g. `name`, `id`). All properties `source: llm`.
- Pipeline nodes: `source: pipeline` with `pipeline_field: doc_path` for the document path property.

### Relationship definition rules
- `rel_type`: SCREAMING_SNAKE_CASE string.
- `from_node` / `to_node`: must match a node key defined in `nodes:`.
- `from_field` / `to_field`: the LLM search term field name (what the LLM extracts to look up the entity). For UMLS nodes use the UMLS preferred name; for GLEIF use legal name; for LLM nodes use the PK field.
- `from_hint` / `to_hint`: plain-English description of what entity the LLM should fill in (shown in the extraction prompt).
- `extract_prompt`: one sentence instructing the LLM what to extract for this relationship type.
- `properties`: list of edge properties (each with `name`, `type`, `source: llm`, optional `hint`).

### UMLS sem_group values (common)
`Chemicals & Drugs`, `Disorders`, `Physiology`, `Anatomy`, `Genes & Molecular Sequences`, `Organizations`, `Procedures`, `Living Beings`

### UMLS umls_vocabs values (common)
`RXNORM`, `MSH`, `SNOMEDCT_US`, `HPO`, `DRUGBANK`, `ATC`, `MED-RT`, `NCI`, `GO`, `OMIM`

---

## Example Output

```yaml
nodes:
  Substance:
    description: "Generic or INN drug substance."
    sem_group: "Chemicals & Drugs"
    umls_vocabs: [RXNORM, MSH, DRUGBANK]
    properties:
      - name: cui
        type: STRING
        source: umls
        primary_key: true
      - name: name
        type: STRING
        source: umls

  Indication:
    description: "Disease or condition the drug treats or prevents."
    sem_group: "Disorders"
    umls_vocabs: [MSH, SNOMEDCT_US, HPO]
    properties:
      - name: cui
        type: STRING
        source: umls
        primary_key: true
      - name: name
        type: STRING
        source: umls

  Company:
    description: "Legal entity (manufacturer, sponsor)."
    properties:
      - name: lei
        type: STRING
        source: gleif
        primary_key: true
      - name: name
        type: STRING
        source: gleif

rels:
  - rel_type: HAS_INDICATION
    from_node: Substance
    to_node: Indication
    from_field: substance_name
    to_field: indication_name
    from_hint: "Generic drug substance name (INN)"
    to_hint: "Disease or medical condition being treated or prevented"
    extract_prompt: >
      Extract all approved indications (diseases or conditions) for each drug substance mentioned
      in this document. Include both primary and secondary indications.
    examples:
      - from: "nirsevimab"
        to: "RSV lower respiratory tract infection"
        quote: "Beyfortus is indicated for the prevention of RSV lower respiratory tract disease"
    properties:
      - name: approval_status
        type: STRING
        source: llm
        hint: "Approval status: approved, investigational, off-label"

  - rel_type: BRAND_OF
    from_node: Tradename
    to_node: Substance
    from_field: brand_name
    to_field: substance_name
    from_hint: "Commercial brand name"
    to_hint: "Active drug substance (INN)"
    extract_prompt: >
      Extract brand name to active substance mappings.
    properties: []
```

---

## Installation for new users

To use this skill in your own Claude Code environment:

1. Copy this file to `~/.claude/skills/helms-schema/SKILL.md`
2. Add to your `~/.claude/CLAUDE.md`:
   ```
   # helms-schema
   - **helms-schema** (`~/.claude/skills/helms-schema/SKILL.md`) — boilerplate a HELMS schema.yaml. Trigger: `/helms-schema`
   When the user types `/helms-schema`, invoke the Skill tool with `skill: "helms-schema"` before doing anything else.
   ```
3. Type `/helms-schema` in any HELMS project to start schema generation.
