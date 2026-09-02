"""Unit tests for schema parsing, dynamic Pydantic model building, and lookups."""
import json
import os
import textwrap
import pytest

from extract import load_schema, build_extraction_model, primary_key, llm_props
from lookups import umls_search, umls_lookup, gleif_search, gleif_lookup
from convert_pdf import _unmatched_meta_keys


# ── Fixtures ──────────────────────────────────────────────────────────────────

VALID_SCHEMA = textwrap.dedent("""\
    nodes:
      Drug:
        properties:
          - name: cui
            type: STRING
            source: umls
            primary_key: true
          - name: name
            type: STRING
            source: umls
          - name: form
            type: STRING
            source: llm
            optional: true
      Disease:
        properties:
          - name: cui
            type: STRING
            source: umls
            primary_key: true
          - name: name
            type: STRING
            source: umls
    relationships:
      - rel_type: TREATS
        from_node: Drug
        from_field: drug_name
        from_hint: "generic INN name"
        to_node: Disease
        to_field: disease_name
        to_hint: "standard disease name"
        extract_prompt: "Extract drug-disease treatment pairs."
        properties:
          - name: evidence
            type: STRING
            source: llm
            optional: true
""")

SCHEMA_MISSING_NODES = textwrap.dedent("""\
    relationships: []
""")

SCHEMA_MISSING_RELS = textwrap.dedent("""\
    nodes:
      Drug:
        properties: []
""")

SCHEMA_UNKNOWN_TYPE = textwrap.dedent("""\
    nodes:
      Drug:
        properties:
          - name: cui
            type: STRING
            source: umls
            primary_key: true
    relationships:
      - rel_type: TREATS
        from_node: Drug
        from_field: drug_name
        to_node: Drug
        to_field: disease_name
        extract_prompt: "x"
        properties:
          - name: score
            type: BADTYPE
            source: llm
""")

# Schema with GLEIF-resolved nodes for testing company lookups
GLEIF_SCHEMA = textwrap.dedent("""\
    nodes:
      Corporation:
        properties:
          - name: lei
            type: STRING
            source: gleif
            primary_key: true
          - name: name
            type: STRING
            source: gleif
      Product:
        properties:
          - name: name
            type: STRING
            source: llm
            primary_key: true
    relationships:
      - rel_type: MAKES
        from_node: Corporation
        from_field: company_name
        to_node: Product
        to_field: product_name
        extract_prompt: "Extract company-product pairs."
""")


def _write_schema(tmp_path, content: str) -> str:
    p = tmp_path / "schema.yaml"
    p.write_text(content)
    return str(p)


# ── load_schema ───────────────────────────────────────────────────────────────

def test_load_schema_valid(tmp_path):
    path = _write_schema(tmp_path, VALID_SCHEMA)
    nodes, rels = load_schema(path)
    assert "Drug" in nodes
    assert "Disease" in nodes
    assert len(rels) == 1
    assert rels[0]["rel_type"] == "TREATS"


def test_load_schema_missing_nodes_key(tmp_path):
    path = _write_schema(tmp_path, SCHEMA_MISSING_NODES)
    with pytest.raises(SystemExit):
        load_schema(path)


def test_load_schema_missing_relationships_key(tmp_path):
    path = _write_schema(tmp_path, SCHEMA_MISSING_RELS)
    with pytest.raises(SystemExit):
        load_schema(path)


def test_load_schema_file_not_found():
    with pytest.raises((FileNotFoundError, SystemExit, OSError)):
        load_schema("/nonexistent/path/schema.yaml")


# ── primary_key ───────────────────────────────────────────────────────────────

def test_primary_key_found(tmp_path):
    path = _write_schema(tmp_path, VALID_SCHEMA)
    nodes, _ = load_schema(path)
    assert primary_key(nodes["Drug"]) == "cui"


def test_primary_key_missing():
    node_def = {"properties": [{"name": "name", "type": "STRING", "source": "llm"}]}
    with pytest.raises(ValueError):
        primary_key(node_def)


# ── llm_props ─────────────────────────────────────────────────────────────────

def test_llm_props_filters_correctly(tmp_path):
    path = _write_schema(tmp_path, VALID_SCHEMA)
    nodes, _ = load_schema(path)
    props = llm_props(nodes["Drug"])
    names = [p["name"] for p in props]
    assert "form" in names
    assert "cui" not in names
    assert "name" not in names


# ── build_extraction_model ────────────────────────────────────────────────────

def test_build_extraction_model_has_rel_field(tmp_path):
    path = _write_schema(tmp_path, VALID_SCHEMA)
    nodes, rels = load_schema(path)
    Model = build_extraction_model(rels, nodes)
    assert "TREATS" in Model.model_fields


def test_build_extraction_model_rel_has_from_to_fields(tmp_path):
    path = _write_schema(tmp_path, VALID_SCHEMA)
    nodes, rels = load_schema(path)
    Model = build_extraction_model(rels, nodes)
    TreatsModel = Model.model_fields["TREATS"].annotation.__args__[0]
    assert "drug_name" in TreatsModel.model_fields
    assert "disease_name" in TreatsModel.model_fields


def test_build_extraction_model_optional_field(tmp_path):
    path = _write_schema(tmp_path, VALID_SCHEMA)
    nodes, rels = load_schema(path)
    Model = build_extraction_model(rels, nodes)
    TreatsModel = Model.model_fields["TREATS"].annotation.__args__[0]
    assert "evidence" in TreatsModel.model_fields
    field = TreatsModel.model_fields["evidence"]
    assert field.default is None


def test_build_extraction_model_unknown_type_raises(tmp_path):
    path = _write_schema(tmp_path, SCHEMA_UNKNOWN_TYPE)
    nodes, rels = load_schema(path)
    with pytest.raises(ValueError, match="Unknown schema type"):
        build_extraction_model(rels, nodes)


def test_build_extraction_model_instantiation(tmp_path):
    path = _write_schema(tmp_path, VALID_SCHEMA)
    nodes, rels = load_schema(path)
    Model = build_extraction_model(rels, nodes)
    instance = Model(TREATS=[{"drug_name": "aspirin", "disease_name": "headache", "supporting_quote": "Aspirin treats headache."}])
    assert len(instance.TREATS) == 1
    assert instance.TREATS[0].drug_name == "aspirin"


# ── build_extraction_model: supporting_quote field ────────────────────────────

def test_build_extraction_model_has_supporting_quote(tmp_path):
    """Every relationship model must have a supporting_quote field for grounding."""
    path = _write_schema(tmp_path, VALID_SCHEMA)
    nodes, rels = load_schema(path)
    Model = build_extraction_model(rels, nodes)
    TreatsModel = Model.model_fields["TREATS"].annotation.__args__[0]
    assert "supporting_quote" in TreatsModel.model_fields


def test_build_extraction_model_supporting_quote_required(tmp_path):
    """supporting_quote is required (no default), not optional."""
    path = _write_schema(tmp_path, VALID_SCHEMA)
    nodes, rels = load_schema(path)
    Model = build_extraction_model(rels, nodes)
    TreatsModel = Model.model_fields["TREATS"].annotation.__args__[0]
    field = TreatsModel.model_fields["supporting_quote"]
    # Required fields have no default (PydanticUndefined in v2, Ellipsis in v1)
    try:
        from pydantic_core import PydanticUndefined
        assert field.default is PydanticUndefined
    except ImportError:
        assert field.default is ...


# ── GLEIF schema model ────────────────────────────────────────────────────────

def test_build_extraction_model_gleif_schema(tmp_path):
    """GLEIF schema with Corporation → Product relationship builds correctly."""
    path = _write_schema(tmp_path, GLEIF_SCHEMA)
    nodes, rels = load_schema(path)
    Model = build_extraction_model(rels, nodes)
    assert "MAKES" in Model.model_fields
    MakesModel = Model.model_fields["MAKES"].annotation.__args__[0]
    assert "company_name" in MakesModel.model_fields
    assert "product_name" in MakesModel.model_fields
    assert "supporting_quote" in MakesModel.model_fields


# ── UMLS search integration tests ─────────────────────────────────────────────
# These require a valid UMLS_API_KEY in .env. Skipped if missing.

def _umls_key_available() -> bool:
    import os
    return bool(os.environ.get("UMLS_API_KEY"))


@pytest.mark.skipif(not _umls_key_available(), reason="UMLS_API_KEY not set")
class TestUmlsSearch:
    """Integration tests for UMLS API lookups."""

    def test_umls_search_diabetes_returns_multiple(self):
        """Searching 'Type 2 diabetes' should return up to 3 results."""
        result = json.loads(umls_search("Type 2 diabetes", "words"))
        assert "error" not in result, f"UMLS search failed: {result}"
        assert "results" in result
        assert len(result["results"]) >= 1
        assert len(result["results"]) <= 5

    def test_umls_search_diabetes_cui(self):
        """'Type 2 diabetes' should include C0011860 (Diabetes Mellitus, Non-Insulin-Dependent)."""
        result = json.loads(umls_search("Type 2 diabetes", "words"))
        assert "error" not in result
        cuis = [r["cui"] for r in result["results"]]
        assert "C0011860" in cuis, f"C0011860 not found in results. Got: {cuis}"

    def test_umls_search_diabetes_name(self):
        """C0011860 should have the canonical name 'Diabetes Mellitus, Non-Insulin-Dependent'."""
        result = json.loads(umls_search("Type 2 diabetes", "words"))
        assert "error" not in result
        for r in result["results"]:
            if r["cui"] == "C0011860":
                assert "Diabetes Mellitus" in r["name"]
                break
        else:
            pytest.fail("C0011860 not found in UMLS results")

    def test_umls_lookup_returns_dict(self):
        """umls_lookup should return a dict with 'cui' and 'name'."""
        result = umls_lookup("Type 2 diabetes")
        assert result is not None
        assert "cui" in result
        assert "name" in result
        assert isinstance(result["cui"], str)
        assert isinstance(result["name"], str)

    def test_umls_lookup_top_result(self):
        """umls_lookup picks the first (top-ranked) result."""
        search_result = json.loads(umls_search("Type 2 diabetes", "words"))
        lookup_result = umls_lookup("Type 2 diabetes")
        assert lookup_result is not None
        assert lookup_result["cui"] == search_result["results"][0]["cui"]

    def test_umls_search_normalizedwords_type(self):
        """search_type='normalizedWords' should return results for a known term."""
        result = json.loads(umls_search("aspirin", "normalizedWords"))
        assert "error" not in result
        assert len(result["results"]) >= 1

    def test_umls_search_no_match(self):
        """A nonsense term should return an error."""
        result = json.loads(umls_search("xyzzyplugh12345", "words"))
        assert "error" in result

    def test_umls_search_caching(self):
        """Repeated searches should hit the module-level cache."""
        import lookups
        lookups._umls_cache.clear()
        r1 = umls_search("aspirin", "words")
        r2 = umls_search("aspirin", "words")
        assert r1 == r2  # cached response is identical



# ── GLEIF search integration tests ────────────────────────────────────────────

class TestGleifSearch:
    """Integration tests for GLEIF API lookups."""

    def test_gleif_search_exact_success(self):
        """Searching a known company should return results list with lei, name, and meta fields."""
        result = json.loads(gleif_search("Apple Inc.", "exact"))
        if "error" not in result:
            assert result["status"] == "success"
            assert "results" in result
            assert len(result["results"]) >= 1
            r0 = result["results"][0]
            assert "lei" in r0
            assert "name" in r0
            assert "status" in r0
            assert "category" in r0
            assert "jurisdiction" in r0
            assert "registration_status" in r0

    def test_gleif_lookup_returns_dict(self):
        """gleif_lookup should return a dict with lei, name, category, entity_legal_form."""
        result = gleif_lookup("Apple Inc.")
        if result is not None:
            assert {"lei", "name", "category", "entity_legal_form"}.issubset(result.keys())

    def test_gleif_search_fuzzy_fallback(self):
        """Fuzzy search should return results list with lei and name."""
        result = json.loads(gleif_search("Micron Technology", "fuzzy"))
        if "error" not in result:
            assert result["status"] == "success"
            assert "results" in result
            assert len(result["results"]) >= 1
            assert "lei" in result["results"][0]

    def test_gleif_search_no_match(self):
        """A nonsense company name should return an error."""
        result = json.loads(gleif_search("xyzzyplugh12345corp", "exact"))
        assert "error" in result

    def test_gleif_search_caching(self):
        """Repeated searches should hit the module-level cache."""
        import lookups
        lookups._gleif_cache.clear()
        r1 = gleif_search("Apple Inc.", "exact")
        r2 = gleif_search("Apple Inc.", "exact")
        assert r1 == r2


# ── GLEIF multi-hit: first ≠ best ────────────────────────────────────────────
# Verify gleif_search returns multiple candidates with disambiguation fields
# so the agent can pick GENERAL over BRANCH/FUND and ISSUED over LAPSED.

class TestGleifMultiHit:
    """Verify cases where the best GLEIF hit is not the first-ranked result."""

    def test_disambiguation_fields_present(self):
        """Every GLEIF result must include status, category, jurisdiction, registration_status."""
        result = json.loads(gleif_search("Apple Inc.", "exact"))
        assert "error" not in result, f"GLEIF search failed: {result}"
        for r in result["results"]:
            for field in ("lei", "name", "status", "category", "jurisdiction", "registration_status"):
                assert field in r, f"Field {field!r} missing from result: {r}"

    def test_exact_returns_up_to_three(self):
        """Exact search for a common name should return multiple candidates."""
        result = json.loads(gleif_search("Barclays Bank PLC", "exact"))
        assert "error" not in result, f"GLEIF search failed: {result}"
        assert len(result["results"]) >= 2, "Expected ≥2 candidates for 'Barclays Bank PLC'"

    def test_barclays_first_is_branch_best_is_general(self):
        """'Barclays Bank PLC': first hit is BRANCH (wrong), second is GENERAL (correct).

        The agent should prefer the GENERAL entity over the BRANCH entity
        when multiple records have the same legal name.
        """
        result = json.loads(gleif_search("Barclays Bank PLC", "exact"))
        assert "error" not in result, f"GLEIF search failed: {result}"
        candidates = result["results"]
        cuis = [c["lei"] for c in candidates]

        # G5GSEF7VJP5I7OUK5573 = Barclays Bank PLC GENERAL entity
        assert "G5GSEF7VJP5I7OUK5573" in cuis, (
            f"Barclays GENERAL LEI G5GSEF7VJP5I7OUK5573 not in candidates: {cuis}"
        )
        assert candidates[0]["category"] == "BRANCH", (
            f"Expected first hit to be BRANCH, got {candidates[0]['category']}"
        )
        general_idx = next(
            i for i, c in enumerate(candidates) if c["lei"] == "G5GSEF7VJP5I7OUK5573"
        )
        assert general_idx > 0, "GENERAL entity should not be the first result"


# ── UMLS multi-hit: first ≠ best ─────────────────────────────────────────────
# These verify that umls_search returns enough candidates for an agent to pick
# a better match than the blind top-1 result, and that semantic_types are
# present to aid selection.

@pytest.mark.skipif(not _umls_key_available(), reason="UMLS_API_KEY not set")
class TestUmlsMultiHit:
    """Verify cases where the best UMLS hit is not the first-ranked result."""

    def test_semantic_types_present_in_results(self):
        """Every result from umls_search must include a non-empty semantic_types list."""
        result = json.loads(umls_search("aspirin", "words"))
        assert "error" not in result, f"Search failed: {result}"
        for r in result["results"]:
            assert "semantic_types" in r, f"semantic_types missing: {r}"
            assert isinstance(r["semantic_types"], list)
            assert len(r["semantic_types"]) >= 1, f"Expected ≥1 semantic type: {r}"

    def test_abbv181_budigalimab_best_not_first(self):
        """'ABBV-181 (Budigalimab)': first hit is ABBV-181 (C4527193), best is budigalimab (C4743556).

        A blind top-1 pick returns the abbreviation form. The agent should prefer
        the INN generic name (C4743556), which ranks lower.
        """
        result = json.loads(umls_search("ABBV-181 (Budigalimab)", "words"))
        assert "error" not in result, f"Search failed: {result}"
        cuis = [r["cui"] for r in result["results"]]
        assert "C4743556" in cuis, f"budigalimab C4743556 not in results: {cuis}"
        assert result["results"][0]["cui"] == "C4527193", (
            f"Expected first hit ABBV-181 C4527193, got {result['results'][0]['cui']}"
        )
        assert cuis.index("C4743556") > 0, "budigalimab C4743556 should not be the first result"

    def test_atropine_sulphate_best_not_first(self):
        """'Atropine sulphate': first hit is the combination product (C0358790), best is the single substance (C0596005).

        Blind top-1 returns 'Morphine sulphate+atropine sulphate'. The agent should
        prefer the specific single-substance CUI (C0596005).
        """
        result = json.loads(umls_search("Atropine sulphate", "words"))
        assert "error" not in result, f"Search failed: {result}"
        cuis = [r["cui"] for r in result["results"]]
        assert "C0596005" in cuis, f"atropine sulfate C0596005 not in results: {cuis}"
        assert result["results"][0]["cui"] == "C0358790", (
            f"Expected first hit C0358790 (combination product), got {result['results'][0]['cui']}"
        )
        assert cuis.index("C0596005") > 0, "atropine sulfate C0596005 should not be the first result"

    def test_chlorhexidine_semantic_types_distinguish_formulation_vs_substance(self):
        """'Chlorhexidine Digluconate Solution': semantic types expose formulation vs. substance distinction.

        First hit: chlorhexidine gluconate 40 MG/ML Topical Solution (C5561554) — 'Clinical Drug'
        Best hit:  chlorhexidine gluconate (C0055361) — 'Organic Chemical, Pharmacologic Substance'
        The agent can use semantic_types to prefer the generic substance over a specific formulation.
        """
        result = json.loads(umls_search("Chlorhexidine Digluconate Solution", "words"))
        assert "error" not in result, f"Search failed: {result}"
        cuis = [r["cui"] for r in result["results"]]
        assert "C0055361" in cuis, f"chlorhexidine gluconate C0055361 not in results: {cuis}"

        first = result["results"][0]
        assert first["cui"] == "C5561554", (
            f"Expected first hit formulation C5561554, got {first['cui']}"
        )
        assert any("Drug" in st for st in first["semantic_types"]), (
            f"Expected 'Clinical Drug' type for first hit, got: {first['semantic_types']}"
        )

        best = next(r for r in result["results"] if r["cui"] == "C0055361")
        assert any(
            kw in st for st in best["semantic_types"]
            for kw in ("Chemical", "Substance", "Pharmacologic")
        ), f"Expected substance/chemical type for best hit, got: {best['semantic_types']}"

        assert cuis.index("C0055361") > 0, "best hit C0055361 should not be the first result"


# ── Drug schema UMLS lookup integration tests ────────────────────────────────
# Verify that umls_lookup with sem_group + sabs returns the expected CUIs for
# the drug_schema.yaml node types (MOA = Physiology/MED-RT, Indication = Disorders).

@pytest.mark.skipif(not _umls_key_available(), reason="UMLS_API_KEY not set")
class TestDrugSchemaLookups:
    """Integration tests for drug schema UMLS resolution."""

    def test_moa_fusion_protein_inhibitors(self):
        """MOA lookup: 'respiratory syncytial virus F protein-directed fusion inhibitor'
        with sem_types=Molecular Function + sabs=MED-RT → C2267039 (Fusion Protein Inhibitors).
        Matches drug_schema.yaml MOA node: semantic_types=[Molecular Function], umls_vocabs=[MED-RT]."""
        result = umls_lookup(
            "respiratory syncytial virus F protein-directed fusion inhibitor",
            sem_types=["Molecular Function"],
            sabs="MED-RT",
        )
        assert result is not None, "MOA lookup returned None"
        assert result["cui"] == "C2267039", f"Expected C2267039, got {result['cui']} ({result['name']})"
        assert "Molecular Function" in result["semantic_types"]

    def test_ind_rsv_lrtd_returns_disease(self):
        """IND lookup: 'RSV lower respiratory tract disease' with sem_group=Disorders
        + sabs from drug_schema (MSH, SNOMEDCT_US, HPO) → C0035235 (RSV Infections)."""
        result = umls_lookup(
            "Respiratory Syncytial Virus (RSV) lower respiratory tract disease",
            sem_group="Disorders",
            sabs="MSH,SNOMEDCT_US,HPO",
        )
        assert result is not None, "IND lookup returned None"
        assert result["cui"] == "C0035235", f"Expected C0035235, got {result['cui']} ({result['name']})"
        assert "Disease or Syndrome" in result["semantic_types"]


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_build_extraction_model_multiple_rels(tmp_path):
    """Schema with multiple relationships produces multiple model fields."""
    schema = textwrap.dedent("""\
        nodes:
          Drug:
            properties:
              - name: cui
                type: STRING
                source: umls
                primary_key: true
          Disease:
            properties:
              - name: cui
                type: STRING
                source: umls
                primary_key: true
          Mechanism:
            properties:
              - name: name
                type: STRING
                source: llm
                primary_key: true
        relationships:
          - rel_type: TREATS
            from_node: Drug
            from_field: drug_name
            to_node: Disease
            to_field: disease_name
            extract_prompt: "x"
          - rel_type: HAS_MECHANISM
            from_node: Drug
            from_field: drug_name
            to_node: Mechanism
            to_field: mechanism_name
            extract_prompt: "y"
    """)
    path = _write_schema(tmp_path, schema)
    nodes, rels = load_schema(path)
    Model = build_extraction_model(rels, nodes)
    assert "TREATS" in Model.model_fields
    assert "HAS_MECHANISM" in Model.model_fields
    assert len(Model.model_fields) == 2


def test_build_extraction_model_rel_props_included(tmp_path):
    """Relationship-level LLM props are included in the extraction model."""
    path = _write_schema(tmp_path, VALID_SCHEMA)
    nodes, rels = load_schema(path)
    Model = build_extraction_model(rels, nodes)
    TreatsModel = Model.model_fields["TREATS"].annotation.__args__[0]
    # 'evidence' is defined on the relationship in VALID_SCHEMA
    assert "evidence" in TreatsModel.model_fields


# ══════════════════════════════════════════════════════════════════════════════
# STANDARDIZED UNIT TESTS — reusable patterns from past bug hunts
# ══════════════════════════════════════════════════════════════════════════════


# ── _schema_type_to_python ────────────────────────────────────────────────────

from extract import _schema_type_to_python, _SCHEMA_TYPES, _chunk_text, _verify_grounding, _write_review_file, _cache_key, llm_rel_props, pipeline_rel_props, _normalize_entity, _entity_in_text, _classify_entity


class TestSchemaTypeToPython:
    """Verify all schema type strings map correctly to Python types."""

    def test_string_type(self):
        assert _schema_type_to_python("STRING") is str

    def test_int_types(self):
        for t in ("INT64", "INT32", "INT16", "INT8", "UINT64", "UINT32", "UINT16", "UINT8"):
            assert _schema_type_to_python(t) is int, f"{t} should map to int"

    def test_float_types(self):
        assert _schema_type_to_python("DOUBLE") is float
        assert _schema_type_to_python("FLOAT") is float

    def test_boolean_type(self):
        assert _schema_type_to_python("BOOLEAN") is bool

    def test_temporal_types_are_string(self):
        for t in ("TIMESTAMP", "DATE", "INTERVAL"):
            assert _schema_type_to_python(t) is str, f"{t} should map to str"

    def test_blob_type(self):
        assert _schema_type_to_python("BLOB") is bytes

    def test_array_types(self):
        assert _schema_type_to_python("STRING[]") == list[str]
        assert _schema_type_to_python("INT64[]") == list[int]
        assert _schema_type_to_python("FLOAT[]") == list[float]

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown schema type"):
            _schema_type_to_python("FOOBAR")

    def test_unknown_array_type_raises(self):
        with pytest.raises(ValueError, match="Unknown schema type"):
            _schema_type_to_python("FOOBAR[]")

    def test_all_registered_types_roundtrip(self):
        """Every type in _SCHEMA_TYPES can be converted without error."""
        for name in _SCHEMA_TYPES:
            _schema_type_to_python(name)


# ── _chunk_text ───────────────────────────────────────────────────────────────

class TestChunkText:
    """Verify text chunking boundary behavior."""

    def test_short_text_single_chunk(self):
        text = "hello world"
        chunks = _chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_exact_chunk_size_single(self):
        from extract import _CHUNK_SIZE
        text = "x" * _CHUNK_SIZE
        chunks = _chunk_text(text)
        assert len(chunks) == 1

    def test_just_over_chunk_size_produces_two(self):
        from extract import _CHUNK_SIZE
        text = "x" * (_CHUNK_SIZE + 1)
        chunks = _chunk_text(text)
        assert len(chunks) == 2

    def test_chunk_overlap_is_applied(self):
        from extract import _CHUNK_SIZE, _CHUNK_OVERLAP
        text = "a" * _CHUNK_SIZE + "b" * _CHUNK_SIZE
        chunks = _chunk_text(text)
        # Advance per chunk = _CHUNK_SIZE - _CHUNK_OVERLAP = 38000
        # 80000 chars → starts at 0, 38000, 76000 → 3 chunks
        assert len(chunks) == 3
        # Second chunk starts at _CHUNK_SIZE - _CHUNK_OVERLAP
        expected_start = _CHUNK_SIZE - _CHUNK_OVERLAP
        assert chunks[1][:100] == text[expected_start:expected_start + 100]

    def test_no_content_lost(self):
        from extract import _CHUNK_SIZE
        text = "".join(chr(i % 26 + ord('a')) for i in range(_CHUNK_SIZE * 3))
        chunks = _chunk_text(text)
        # Every character must appear in at least one chunk
        for ch in set(text):
            assert any(ch in c for c in chunks), f"Character {ch!r} lost in chunking"

    def test_empty_string(self):
        chunks = _chunk_text("")
        assert chunks == [""]


# ── _verify_grounding ─────────────────────────────────────────────────────────

class TestVerifyGrounding:
    """Verify quote grounding verification under all constraint levels."""

    def _make_item(self, quote: str):
        """Create a mock item dict with supporting_quote and from_field."""
        return {"supporting_quote": quote, "drug_name": "aspirin", "disease_name": "headache"}

    def _make_rel(self):
        return {
            "rel_type": "TREATS",
            "from_field": "drug_name",
            "to_field": "disease_name",
        }

    def test_loose_keeps_all(self):
        items = {"TREATS": [self._make_item(""), self._make_item("no match")]}
        dropped, warned, warnings = _verify_grounding(items, [self._make_rel()], "some doc", "loose")
        assert dropped == 0
        assert warned == 0
        assert len(items["TREATS"]) == 2

    def test_moderate_keeps_but_warns_on_mismatch(self):
        doc = "Aspirin is a common painkiller used for headaches."
        items = {"TREATS": [self._make_item("This quote does not appear in the document at all.")]}
        rels = [self._make_rel()]
        dropped, warned, warnings = _verify_grounding(items, rels, doc, "moderate")
        assert dropped == 0
        assert warned == 1
        assert len(warnings) == 1
        assert len(items["TREATS"]) == 1  # kept

    def test_moderate_keeps_empty_quote(self):
        doc = "Some document text."
        items = {"TREATS": [self._make_item("")]}
        rels = [self._make_rel()]
        dropped, warned, warnings = _verify_grounding(items, rels, doc, "moderate")
        assert dropped == 0
        assert warned == 0
        assert len(items["TREATS"]) == 1  # kept: moderate mode does not drop no-quote items

    def test_strict_drops_mismatch(self):
        doc = "Aspirin is a common painkiller."
        items = {"TREATS": [self._make_item("This quote is not in the doc at all and is long enough.")]}
        rels = [self._make_rel()]
        dropped, warned, warnings = _verify_grounding(items, rels, doc, "strict")
        assert dropped == 1
        assert len(items["TREATS"]) == 0

    def test_strict_keeps_verbatim_match(self):
        doc = "Aspirin treats headaches effectively."
        quote = "Aspirin treats headaches effectively."
        items = {"TREATS": [self._make_item(quote)]}
        rels = [self._make_rel()]
        dropped, warned, warnings = _verify_grounding(items, rels, doc, "strict")
        assert dropped == 0
        assert len(items["TREATS"]) == 1

    def test_short_quote_dropped_if_absent_strict(self):
        """Short quotes not in document are dropped in strict mode."""
        doc = "Some long document text here."
        short = "short"  # 5 chars, not in doc
        items = {"TREATS": [self._make_item(short)]}
        rels = [self._make_rel()]
        dropped, warned, warnings = _verify_grounding(items, rels, doc, "strict")
        assert dropped == 1

    def test_short_quote_warned_if_absent_moderate(self):
        """Short quotes not in document generate a warning in moderate mode."""
        doc = "Some long document text here."
        short = "short"  # 5 chars, not in doc
        items = {"TREATS": [self._make_item(short)]}
        rels = [self._make_rel()]
        dropped, warned, warnings = _verify_grounding(items, rels, doc, "moderate")
        assert dropped == 0
        assert warned == 1

    def test_moderate_no_warning_for_matching_quote(self):
        doc = "Aspirin treats headaches. It is widely used."
        quote = "Aspirin treats headaches."
        items = {"TREATS": [self._make_item(quote)]}
        rels = [self._make_rel()]
        dropped, warned, warnings = _verify_grounding(items, rels, doc, "moderate")
        assert dropped == 0
        assert warned == 0
        assert warnings == []


# ── resolve_node (now via NodeResolver / build_props) ─────────────────────────

class TestResolveNode:
    """Verify property resolution via resolver build_props (replaces removed resolve_node)."""

    def _node_def(self, props):
        return {"properties": props}

    def test_llm_only_node_has_no_resolver(self):
        """Node with only LLM props: resolver_for_node returns None."""
        from agents.node_agent import resolver_for_node
        node_def = self._node_def([
            {"name": "form", "source": "llm"},
            {"name": "dosage", "source": "llm", "optional": True},
        ])
        assert resolver_for_node(node_def) is None

    def test_umls_node_has_resolver(self):
        """Node with UMLS props: resolver_for_node returns UMLSResolver."""
        from agents.node_agent import resolver_for_node
        from agents.umls_resolver import UMLSResolver
        node_def = self._node_def([
            {"name": "cui", "source": "umls", "primary_key": True},
            {"name": "name", "source": "umls"},
        ])
        assert isinstance(resolver_for_node(node_def), UMLSResolver)

    def test_optional_llm_prop_missing(self):
        """Optional LLM prop missing from extras: build_props succeeds without it."""
        from agents.umls_resolver import UMLSResolver
        node_def = self._node_def([
            {"name": "cui", "source": "umls", "primary_key": True},
            {"name": "name", "source": "umls"},
            {"name": "color", "source": "llm", "optional": True},
        ])
        resolved_map = {("aspirin", "Drug"): {"cui": "C0000000", "name": "Aspirin", "semantic_types": []}}
        result = UMLSResolver().build_props("aspirin", "Drug", node_def, {}, resolved_map)
        assert result is not None
        props, _ = result
        assert props["cui"] == "C0000000"
        assert "color" not in props

    def test_required_llm_prop_missing_returns_none(self):
        """Required LLM prop missing from extras: build_props returns None."""
        from agents.umls_resolver import UMLSResolver
        node_def = self._node_def([
            {"name": "cui", "source": "umls", "primary_key": True},
            {"name": "name", "source": "umls"},
            {"name": "form", "source": "llm"},
        ])
        resolved_map = {("aspirin", "Drug"): {"cui": "C0000000", "name": "Aspirin", "semantic_types": []}}
        result = UMLSResolver().build_props("aspirin", "Drug", node_def, {}, resolved_map)
        assert result is None

    def test_umls_lookup_failure_returns_none(self):
        """resolved_map has None for entity: build_props returns None."""
        from agents.umls_resolver import UMLSResolver
        node_def = self._node_def([
            {"name": "cui", "source": "umls", "primary_key": True},
            {"name": "name", "source": "umls"},
        ])
        resolved_map = {("unknown drug", "Drug"): None}
        result = UMLSResolver().build_props("unknown drug", "Drug", node_def, {}, resolved_map)
        assert result is None

    def test_umls_lookup_success(self):
        """resolved_map has UMLS data: build_props populates props correctly."""
        from agents.umls_resolver import UMLSResolver
        node_def = self._node_def([
            {"name": "cui", "source": "umls", "primary_key": True},
            {"name": "name", "source": "umls"},
        ])
        resolved_map = {("aspirin", "Drug"): {"cui": "C0012345", "name": "Test Drug", "semantic_types": ["Pharmacologic Substance"]}}
        result = UMLSResolver().build_props("aspirin", "Drug", node_def, {}, resolved_map)
        assert result is not None
        props, meta = result
        assert props["cui"] == "C0012345"
        assert props["name"] == "Test Drug"
        assert "types" in meta

    def test_gleif_lookup_failure_returns_none(self):
        """resolved_map has None for GLEIF entity: build_props returns None."""
        from agents.gleif_resolver import GLEIFResolver
        node_def = self._node_def([
            {"name": "lei", "source": "gleif", "primary_key": True},
            {"name": "name", "source": "gleif"},
        ])
        resolved_map = {("unknown corp", "Company"): None}
        result = GLEIFResolver().build_props("unknown corp", "Company", node_def, {}, resolved_map)
        assert result is None

    def test_gleif_lookup_success(self):
        """resolved_map has GLEIF data: build_props populates props + meta."""
        from agents.gleif_resolver import GLEIFResolver
        node_def = self._node_def([
            {"name": "lei", "source": "gleif", "primary_key": True},
            {"name": "name", "source": "gleif"},
        ])
        resolved_map = {("Test Corp", "Company"): {
            "lei": "LEI123", "name": "Test Corp",
            "entity_legal_form": "XYZ", "category": "GENERAL",
        }}
        result = GLEIFResolver().build_props("Test Corp", "Company", node_def, {}, resolved_map)
        assert result is not None
        props, meta = result
        assert props["lei"] == "LEI123"
        assert "types" in meta
        assert "XYZ" in meta["types"]
        assert "GENERAL" in meta["types"]

    def test_mixed_sources(self):
        """Node with UMLS + LLM props: build_props assembles both."""
        from agents.umls_resolver import UMLSResolver
        node_def = self._node_def([
            {"name": "cui", "source": "umls", "primary_key": True},
            {"name": "name", "source": "umls"},
            {"name": "body_system", "source": "llm"},
        ])
        resolved_map = {("test", "Entity"): {"cui": "C999", "name": "Test", "semantic_types": []}}
        result = UMLSResolver().build_props("test", "Entity", node_def, {"body_system": "respiratory"}, resolved_map)
        assert result is not None
        props, _ = result
        assert props["cui"] == "C999"
        assert props["body_system"] == "respiratory"


# ── llm_rel_props / pipeline_rel_props ────────────────────────────────────────

class TestRelProps:
    """Verify relationship property filters."""

    def test_llm_rel_props_filters(self):
        rel = {
            "properties": [
                {"name": "evidence", "source": "llm"},
                {"name": "source_doc", "source": "pipeline"},
                {"name": "cui", "source": "umls"},
            ]
        }
        result = llm_rel_props(rel)
        assert len(result) == 1
        assert result[0]["name"] == "evidence"

    def test_pipeline_rel_props_filters(self):
        rel = {
            "properties": [
                {"name": "evidence", "source": "llm"},
                {"name": "source_doc", "source": "pipeline"},
            ]
        }
        result = pipeline_rel_props(rel)
        assert len(result) == 1
        assert result[0]["name"] == "source_doc"

    def test_no_properties_key(self):
        rel = {}
        assert llm_rel_props(rel) == []
        assert pipeline_rel_props(rel) == []

    def test_empty_properties(self):
        rel = {"properties": []}
        assert llm_rel_props(rel) == []
        assert pipeline_rel_props(rel) == []


# ── _write_review_file ────────────────────────────────────────────────────────

class TestWriteReviewFile:
    """Verify review JSON structure written by _write_review_file."""

    def test_basic_structure(self, tmp_path):
        review_path = tmp_path / "test_review.json"
        _write_review_file(
            review_path=review_path,
            doc_name="test.pdf",
            doc_text="Some text.",
            dataset_name="test_abc123",
            schema_version="sv001",
            triples=[{"rel_type": "TREATS", "from": "aspirin"}],
        )
        data = json.loads(review_path.read_text())
        assert data["doc"] == "test.pdf"
        assert data["dataset_name"] == "test_abc123"
        assert data["schema_version"] == "sv001"
        assert data["doc_text"] == "Some text."
        assert len(data["triples"]) == 1
        assert "grounding_warnings" not in data

    def test_grounding_warnings_included(self, tmp_path):
        review_path = tmp_path / "test_review.json"
        warnings = [{"rel_type": "TREATS", "message": "not verbatim"}]
        _write_review_file(
            review_path=review_path,
            doc_name="test.pdf",
            doc_text="text",
            dataset_name="d",
            schema_version="",
            triples=[],
            grounding_warnings=warnings,
        )
        data = json.loads(review_path.read_text())
        assert data["grounding_warnings"] == warnings

    def test_no_grounding_warnings_when_none(self, tmp_path):
        review_path = tmp_path / "test_review.json"
        _write_review_file(
            review_path=review_path,
            doc_name="test.pdf",
            doc_text="text",
            dataset_name="d",
            schema_version="",
            triples=[],
        )
        data = json.loads(review_path.read_text())
        assert "grounding_warnings" not in data

    def test_large_doc_text_omitted(self, tmp_path):
        review_path = tmp_path / "test_review.json"
        from extract import _REVIEW_MAX_DOC_CHARS
        big_text = "x" * (_REVIEW_MAX_DOC_CHARS + 1)
        _write_review_file(
            review_path=review_path,
            doc_name="big.pdf",
            doc_text=big_text,
            dataset_name="d",
            schema_version="",
            triples=[],
            doc_source=tmp_path / "big.md",
        )
        data = json.loads(review_path.read_text())
        assert "doc_text" not in data
        assert "doc_source" in data

    def test_schema_path_included(self, tmp_path):
        review_path = tmp_path / "test_review.json"
        _write_review_file(
            review_path=review_path,
            doc_name="test.pdf",
            doc_text="text",
            dataset_name="d",
            schema_version="",
            triples=[],
            schema_path="/some/path/schema.yaml",
        )
        data = json.loads(review_path.read_text())
        assert data["schema_path"] == "/some/path/schema.yaml"

    def test_filter_level_included(self, tmp_path):
        review_path = tmp_path / "test_review.json"
        _write_review_file(
            review_path=review_path,
            doc_name="test.pdf",
            doc_text="text",
            dataset_name="d",
            schema_version="",
            triples=[],
            filter_level="strict",
        )
        data = json.loads(review_path.read_text())
        assert data["filter_level"] == "strict"


# ── _cache_key ────────────────────────────────────────────────────────────────

class TestCacheKey:
    """Verify cache key generation properties."""

    def test_deterministic(self):
        k1 = _cache_key("text", "schema")
        k2 = _cache_key("text", "schema")
        assert k1 == k2

    def test_different_inputs_different_keys(self):
        k1 = _cache_key("text1", "schema")
        k2 = _cache_key("text2", "schema")
        assert k1 != k2

    def test_instructions_affect_key(self):
        k1 = _cache_key("text", "schema", "")
        k2 = _cache_key("text", "schema", "do X")
        assert k1 != k2

    def test_filter_level_affects_key(self):
        k1 = _cache_key("text", "schema", "", "moderate")
        k2 = _cache_key("text", "schema", "", "strict")
        assert k1 != k2

    def test_returns_hex_string(self):
        k = _cache_key("a", "b", "c")
        assert len(k) == 64  # SHA-256 hex
        int(k, 16)  # should not raise


# ── load_schema edge cases ────────────────────────────────────────────────────

class TestLoadSchemaEdgeCases:
    """Additional load_schema error and edge cases."""

    def test_non_dict_yaml_raises(self, tmp_path):
        path = _write_schema(tmp_path, "- item1\n- item2\n")
        with pytest.raises(SystemExit, match="YAML mapping"):
            load_schema(path)



# ── build_extraction_model edge cases ─────────────────────────────────────────

class TestBuildExtractionModelEdgeCases:
    """Additional model building edge cases."""

    def test_node_llm_props_added_to_rel_model(self, tmp_path):
        """LLM props from from_node and to_node are included in the rel model."""
        schema = textwrap.dedent("""\
            nodes:
              Drug:
                properties:
                  - name: cui
                    type: STRING
                    source: umls
                    primary_key: true
                  - name: body_system
                    type: STRING
                    source: llm
                    hint: "anatomical system"
              Disease:
                properties:
                  - name: cui
                    type: STRING
                    source: umls
                    primary_key: true
            relationships:
              - rel_type: TREATS
                from_node: Drug
                from_field: drug_name
                to_node: Disease
                to_field: disease_name
                extract_prompt: "x"
        """)
        path = _write_schema(tmp_path, schema)
        nodes, rels = load_schema(path)
        Model = build_extraction_model(rels, nodes)
        TreatsModel = Model.model_fields["TREATS"].annotation.__args__[0]
        assert "body_system" in TreatsModel.model_fields
        assert "drug_name" in TreatsModel.model_fields
        assert "disease_name" in TreatsModel.model_fields

    def test_array_type_in_model(self, tmp_path):
        """Schema with ARRAY type produces list[...] field in model."""
        schema = textwrap.dedent("""\
            nodes:
              Drug:
                properties:
                  - name: name
                    type: STRING
                    source: llm
                    primary_key: true
              Disease:
                properties:
                  - name: name
                    type: STRING
                    source: llm
                    primary_key: true
            relationships:
              - rel_type: TREATS
                from_node: Drug
                from_field: drug_name
                to_node: Disease
                to_field: disease_name
                extract_prompt: "x"
                properties:
                  - name: tags
                    type: "STRING[]"
                    source: llm
                    optional: true
        """)
        path = _write_schema(tmp_path, schema)
        nodes, rels = load_schema(path)
        Model = build_extraction_model(rels, nodes)
        TreatsModel = Model.model_fields["TREATS"].annotation.__args__[0]
        assert "tags" in TreatsModel.model_fields
        field = TreatsModel.model_fields["tags"]
        assert field.default is None  # optional

    def test_primary_key_prop_not_a_model_field(self, tmp_path):
        """A node's PK is carried by from_field/to_field, not by its own field.

        Both nodes here declare an llm-sourced PK called 'name'. Exposing it as
        a model field gives the LLM one shared, unexplained column for both
        sides of the relationship — which it fills with the relationship type,
        so every node ends up named 'TREATS'. The PK must come from drug_name /
        disease_name instead.
        """
        schema = textwrap.dedent("""\
            nodes:
              Drug:
                properties:
                  - name: name
                    type: STRING
                    source: llm
                    primary_key: true
              Disease:
                properties:
                  - name: name
                    type: STRING
                    source: llm
                    primary_key: true
            relationships:
              - rel_type: TREATS
                from_node: Drug
                from_field: drug_name
                to_node: Disease
                to_field: disease_name
                extract_prompt: "x"
        """)
        path = _write_schema(tmp_path, schema)
        nodes, rels = load_schema(path)
        Model = build_extraction_model(rels, nodes)
        TreatsModel = Model.model_fields["TREATS"].annotation.__args__[0]
        assert "name" not in TreatsModel.model_fields
        assert "drug_name" in TreatsModel.model_fields
        assert "disease_name" in TreatsModel.model_fields

    def test_required_rel_prop_no_default(self, tmp_path):
        """Non-optional relationship LLM prop has no default (required)."""
        schema = textwrap.dedent("""\
            nodes:
              Drug:
                properties:
                  - name: name
                    type: STRING
                    source: llm
                    primary_key: true
              Disease:
                properties:
                  - name: name
                    type: STRING
                    source: llm
                    primary_key: true
            relationships:
              - rel_type: TREATS
                from_node: Drug
                from_field: drug_name
                to_node: Disease
                to_field: disease_name
                extract_prompt: "x"
                properties:
                  - name: evidence_level
                    type: STRING
                    source: llm
        """)
        path = _write_schema(tmp_path, schema)
        nodes, rels = load_schema(path)
        Model = build_extraction_model(rels, nodes)
        TreatsModel = Model.model_fields["TREATS"].annotation.__args__[0]
        field = TreatsModel.model_fields["evidence_level"]
        try:
            from pydantic_core import PydanticUndefined
            assert field.default is PydanticUndefined
        except ImportError:
            assert field.default is ...


# ── pipeline_meta ─────────────────────────────────────────────────────────────

from pipeline_meta import load_meta, get_instructions, get_page_filter, _expand_page_ranges, PageFilter


class TestPipelineMeta:
    """Verify pipeline metadata loading, instructions, and page filtering."""

    def test_load_meta_none_returns_empty(self):
        assert load_meta(None) == {}

    def test_load_meta_missing_file(self, tmp_path):
        result = load_meta(tmp_path / "nonexistent.yaml")
        assert result == {}

    def test_load_meta_valid(self, tmp_path):
        p = tmp_path / "meta.yaml"
        p.write_text("instructions: do stuff\n")
        meta = load_meta(str(p))
        assert meta["instructions"] == "do stuff"

    def test_get_instructions_present(self):
        meta = {"instructions": "Only extract X."}
        assert get_instructions(meta) == "Only extract X."

    def test_get_instructions_missing(self):
        assert get_instructions({}) == ""

    def test_get_instructions_none(self):
        assert get_instructions({"instructions": None}) == ""

    def test_get_instructions_whitespace_only(self):
        assert get_instructions({"instructions": "   \n  "}) == ""


class TestExpandPageRanges:
    """Verify _expand_page_ranges handles ints, ranges, and mixed lists."""

    def test_single_ints(self):
        assert _expand_page_ranges([1, 3, 5]) == [1, 3, 5]

    def test_range_string(self):
        assert _expand_page_ranges(["1-5"]) == [1, 2, 3, 4, 5]

    def test_mixed(self):
        assert _expand_page_ranges([1, "3-5", 7]) == [1, 3, 4, 5, 7]

    def test_overlapping_deduped(self):
        assert _expand_page_ranges(["1-3", "2-4"]) == [1, 2, 3, 4]

    def test_sorted_output(self):
        assert _expand_page_ranges([5, "1-3", 2]) == [1, 2, 3, 5]

    def test_empty_list(self):
        assert _expand_page_ranges([]) == []


class TestGetPageFilter:
    """Verify page filter include/exclude/both logic."""

    def test_no_pages_entry(self):
        meta = {"pages": {"other doc": {"include": [1, 2]}}}
        assert get_page_filter(meta, "my doc", total_pages=10) is None

    def test_include_only(self):
        meta = {"pages": {"my doc": {"include": ["1-5"]}}}
        pf = get_page_filter(meta, "my doc", total_pages=10)
        assert pf is not None
        assert pf.pages == [1, 2, 3, 4, 5]
        assert pf.mode == "include"

    def test_exclude_only(self):
        meta = {"pages": {"my doc": {"exclude": [1, 2]}}}
        pf = get_page_filter(meta, "my doc", total_pages=5)
        assert pf is not None
        assert pf.pages == [3, 4, 5]
        assert pf.mode == "exclude"

    def test_include_and_exclude(self):
        meta = {"pages": {"my doc": {"include": ["1-10"], "exclude": [5, 6]}}}
        pf = get_page_filter(meta, "my doc", total_pages=10)
        assert pf is not None
        assert 5 not in pf.pages
        assert 6 not in pf.pages
        assert 1 in pf.pages
        assert 10 in pf.pages
        assert pf.mode == "include+exclude"

    def test_include_beyond_total_pages_clamped(self):
        meta = {"pages": {"my doc": {"include": ["1-20"]}}}
        pf = get_page_filter(meta, "my doc", total_pages=5)
        assert pf is not None
        assert pf.pages == [1, 2, 3, 4, 5]

    def test_exclude_all_pages_returns_none(self):
        meta = {"pages": {"my doc": {"exclude": ["1-5"]}}}
        pf = get_page_filter(meta, "my doc", total_pages=5)
        assert pf is None

    def test_empty_include_returns_none(self):
        meta = {"pages": {"my doc": {"include": []}}}
        pf = get_page_filter(meta, "my doc", total_pages=5)
        assert pf is None


class TestUnmatchedMetaKeys:
    """A typo'd meta.yaml page-filter key silently no-ops (get_page_filter returns
    None on a miss) — the warning helper exists to catch that before it ships."""

    def test_typo_key_flagged(self):
        # Real repro: "dificil label" vs the actual file "dificid label.pdf".
        meta = {"pages": {"dificil label": {"include": [1]}}}
        assert _unmatched_meta_keys(meta, {"dificid label", "lipitor label"}) == ["dificil label"]

    def test_matching_key_not_flagged(self):
        meta = {"pages": {"dificid label": {"include": [1]}}}
        assert _unmatched_meta_keys(meta, {"dificid label", "lipitor label"}) == []

    def test_no_pages_section(self):
        assert _unmatched_meta_keys({}, {"dificid label"}) == []

    def test_multiple_unmatched(self):
        meta = {"pages": {"typo one": {}, "ok doc": {}, "typo two": {}}}
        assert set(_unmatched_meta_keys(meta, {"ok doc"})) == {"typo one", "typo two"}


# ── _extract_text ─────────────────────────────────────────────────────────────

from extract import _extract_text


class TestExtractText:
    """Verify _extract_text document loading."""

    def test_md_file_direct(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("# Hello\nContent here.")
        text, info = _extract_text(md)
        assert text == "# Hello\nContent here."
        assert "markdown" in info.lower()

    def test_pdf_without_sidecar_raises(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        with pytest.raises(SystemExit, match="markdown sidecar"):
            _extract_text(pdf)

    def test_pdf_with_stale_sidecar_raises(self, tmp_path):
        """Sidecar older than PDF should trigger conversion message."""
        import time
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        time.sleep(0.05)
        md = tmp_path / "test.md"
        md.write_text("converted text")
        # Make the md older than the pdf
        import os
        os.utime(str(md), (0, 0))
        with pytest.raises(SystemExit, match="markdown sidecar"):
            _extract_text(pdf)

    def test_pdf_with_fresh_sidecar(self, tmp_path):
        """Fresh sidecar (.md newer than .pdf) should be used."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        # Small delay so md gets a distinctly newer mtime
        import time
        time.sleep(0.05)
        md = tmp_path / "test.md"
        md.write_text("converted content")
        text, info = _extract_text(pdf)
        assert text == "converted content"
        assert "sidecar" in info.lower()


# ── mcp_server module safety ──────────────────────────────────────────────────

class TestMcpServerModule:
    """Verify mcp_server module can be imported and has expected structure."""

    def test_safe_import(self):
        """Importing mcp_server should not crash even without GRAPH_SCHEMA."""
        import importlib
        mod = importlib.import_module("mcp_server")
        assert hasattr(mod, "mcp")
        assert hasattr(mod, "get_schema")
        assert hasattr(mod, "run_cypher")
        assert hasattr(mod, "get_node_count")
        assert hasattr(mod, "_server_init")

    def test_mcp_instructions_property_is_settable(self):
        """FastMCP.instructions must be writable (via _mcp_server.instructions)."""
        import importlib
        mod = importlib.import_module("mcp_server")
        mcp_instance = mod.mcp
        # FastMCP.instructions is a read-only property; the writable path is _mcp_server.instructions
        assert hasattr(mcp_instance, "_mcp_server")
        assert hasattr(mcp_instance._mcp_server, "instructions")

    def test_get_schema_returns_string(self):
        """get_schema should return a string (empty if no schema loaded)."""
        import importlib
        mod = importlib.import_module("mcp_server")
        result = mod.get_schema()
        assert isinstance(result, str)

    def test_run_cypher_write_query_blocked(self):
        """run_cypher should reject write queries."""
        import importlib
        mod = importlib.import_module("mcp_server")
        result = mod.run_cypher("CREATE (n:Test {name: 'x'})")
        parsed = json.loads(result)
        assert "error" in parsed
        assert "not permitted" in parsed["error"].lower()

    def test_get_node_count_invalid_label(self):
        """get_node_count should reject invalid label names."""
        import importlib
        mod = importlib.import_module("mcp_server")
        result = mod.get_node_count("DROP TABLE; --")
        parsed = json.loads(result)
        assert "error" in parsed



# ── GraphBackend interface ────────────────────────────────────────────────────

class TestGraphBackendInterface:
    """Verify GraphBackend ABC contract."""

    def test_cannot_instantiate_abstract(self):
        from backends.base import GraphBackend
        with pytest.raises(TypeError):
            GraphBackend()

    def test_has_all_abstract_methods(self):
        from backends.base import GraphBackend
        expected = {"setup", "node_exists", "upsert_node", "create_edge",
                    "count_nodes", "count_edges", "run_cypher", "delete_node", "delete_edge"}
        actual = {name for name, val in vars(GraphBackend).items()
                  if getattr(val, "__isabstractmethod__", False)}
        assert expected == actual

    def test_close_is_not_abstract(self):
        from backends.base import GraphBackend
        assert not hasattr(GraphBackend.close, "__isabstractmethod__") or \
               not GraphBackend.close.__isabstractmethod__


# ── _normalize_entity ─────────────────────────────────────────────────────────

class TestNormalizeEntity:
    def test_strips_inc_suffix(self):
        assert _normalize_entity("Micron Technology, Inc.") == "micron technology"

    def test_strips_corp_suffix(self):
        assert _normalize_entity("3M Company") == "3m"

    def test_preserves_significant_words(self):
        result = _normalize_entity("Pfizer Biopharmaceuticals")
        assert result == "pfizer biopharmaceuticals"

    def test_empty_string(self):
        assert _normalize_entity("") == ""

    def test_all_suffixes(self):
        assert _normalize_entity("Acme Corp Ltd") == "acme"


# ── _entity_in_text ───────────────────────────────────────────────────────────

class TestEntityInText:
    def test_exact_match(self):
        assert _entity_in_text("aspirin", "patient took aspirin daily")

    def test_case_insensitive(self):
        assert _entity_in_text("Aspirin", "patient took aspirin daily")

    def test_suffix_strip_match(self):
        assert _entity_in_text("Micron Technology, Inc.", "micron announced a new chip")

    def test_hyphen_token_match(self):
        assert _entity_in_text("nirsevimab-alip", "dose of nirsevimab was administered")

    def test_no_match(self):
        assert not _entity_in_text("Pfizer", "novartis announced results")

    def test_empty_entity_returns_false(self):
        assert not _entity_in_text("", "some text here")

    def test_empty_text_returns_false(self):
        assert not _entity_in_text("aspirin", "")


# ── _classify_entity ─────────────────────────────────────────────────────────

class TestClassifyEntity:
    def test_green_both_text_and_quote(self):
        assert _classify_entity("aspirin", "aspirin reduces pain", "aspirin is used daily") == "green"

    def test_yellow_text_only(self):
        assert _classify_entity("aspirin", "aspirin reduces pain", "medication reduces pain") == "yellow"

    def test_yellow_empty_quote(self):
        assert _classify_entity("aspirin", "aspirin reduces pain", "") == "yellow"

    def test_red_not_in_text(self):
        assert _classify_entity("ibuprofen", "aspirin reduces pain", "aspirin helps") == "red"

    def test_red_quote_only_is_red(self):
        # entity in quote but NOT in doc text → red (not yellow)
        assert _classify_entity("ibuprofen", "aspirin reduces pain", "ibuprofen is used") == "red"

    def test_red_neither(self):
        assert _classify_entity("ibuprofen", "take this medicine", "take this medicine") == "red"


# ══════════════════════════════════════════════════════════════════════════════
# CACHE TESTS — L1 (in-process dict) and L2 (SQLite) cache behavior
# All tests use dummy data and mocked HTTP / LLM calls — no real API hits.
# ══════════════════════════════════════════════════════════════════════════════

import asyncio
import json as _json_mod
from unittest.mock import patch, AsyncMock


@pytest.fixture
def temp_cache(tmp_path, monkeypatch):
    """Redirect lookup_cache to a temporary SQLite DB for L2 isolation.

    Resets the thread-local connection so tests always talk to the temp DB,
    not the project's real lookup_cache.db.
    """
    import lookup_cache
    temp_db = tmp_path / "test_lookup_cache.db"
    monkeypatch.setattr(lookup_cache, "_DB_PATH", temp_db)
    if hasattr(lookup_cache._local, "conn"):
        lookup_cache._local.conn.close()
        del lookup_cache._local.conn
    lookup_cache._init_db()
    yield lookup_cache
    if hasattr(lookup_cache._local, "conn"):
        lookup_cache._local.conn.close()
        del lookup_cache._local.conn


# ── Generic L1+L2 cache helper (lookup_cache.cached / cached_async) ───────────

class TestCachedHelper:
    """The single cache entry point every lookup now routes through.

    The contract that prevents the recurring 'forgot to cache the deterministic
    negative' bug: compute() returns (value, cacheable); the helper is the ONLY
    writer, and it caches whenever cacheable is True — including a None/empty value.
    """

    def test_cacheable_negative_is_stored_and_served(self, temp_cache):
        import lookup_cache as lc
        l1: dict = {}
        calls = {"n": 0}

        def _compute():
            calls["n"] += 1
            return None, True  # deterministic no-match — MUST be cached

        first = lc.cached(l1, "svc_test", ("k1",), _compute, decode=_json_mod.loads, encode=_json_mod.dumps)
        second = lc.cached(l1, "svc_test", ("k1",), _compute, decode=_json_mod.loads, encode=_json_mod.dumps)
        assert first is None and second is None
        assert calls["n"] == 1                       # second call served from cache
        assert temp_cache.get("svc_test", ("k1",)) is not None  # L2 actually written

    def test_uncacheable_result_is_not_stored(self, temp_cache):
        import lookup_cache as lc
        l1: dict = {}
        calls = {"n": 0}

        def _compute():
            calls["n"] += 1
            return {"transient": True}, False  # e.g. network error — never cache

        lc.cached(l1, "svc_test2", ("k2",), _compute, decode=_json_mod.loads, encode=_json_mod.dumps)
        lc.cached(l1, "svc_test2", ("k2",), _compute, decode=_json_mod.loads, encode=_json_mod.dumps)
        assert calls["n"] == 2                        # recomputed each time
        assert ("k2",) not in l1
        assert temp_cache.get("svc_test2", ("k2",)) is None

    def test_l1_hit_returns_defensive_copy(self, temp_cache):
        import lookup_cache as lc
        l1: dict = {("k3",): {"a": 1}}
        out = lc.cached(l1, "svc_test3", ("k3",), lambda: ({}, True), copy=dict)
        out["mutated"] = True
        assert "mutated" not in l1[("k3",)]

    def test_l1_max_evicts_oldest(self, temp_cache):
        import lookup_cache as lc
        l1: dict = {}
        for i in range(5):
            lc.cached(l1, "svc_test4", (f"k{i}",), lambda i=i: (i, True), l1_max=3)
        assert len(l1) == 3
        assert ("k0",) not in l1 and ("k4",) in l1  # FIFO eviction

    def test_cached_async_caches_success_only(self, temp_cache):
        import lookup_cache as lc
        l1: dict = {}
        calls = {"n": 0}

        async def _compute():
            calls["n"] += 1
            return {"ok": True}, True

        async def _none():
            return None, False

        first = asyncio.run(lc.cached_async(l1, "svc_a", ("ka",), _compute,
                                            decode=_json_mod.loads, encode=_json_mod.dumps, copy=dict))
        second = asyncio.run(lc.cached_async(l1, "svc_a", ("ka",), _compute,
                                             decode=_json_mod.loads, encode=_json_mod.dumps, copy=dict))
        assert first == {"ok": True} and second == {"ok": True}
        assert calls["n"] == 1
        # An uncacheable async miss is never stored.
        assert asyncio.run(lc.cached_async(l1, "svc_a", ("kb",), _none, copy=dict)) is None
        assert ("kb",) not in l1


# ── GLEIF search: L1 cache ────────────────────────────────────────────────────

class TestGleifSearchL1Cache:
    """L1 in-process dict cache for gleif_search."""

    _KEY = ("test corp l1", "exact")
    _FAKE = _json_mod.dumps({
        "results": [{"lei": "L1LEI", "name": "Test Corp L1", "status": "PUBLISHED",
                     "category": "GENERAL", "jurisdiction": "US",
                     "registration_status": "ISSUED"}],
        "status": "success",
    })

    def setup_method(self):
        import lookups
        lookups._gleif_cache.pop(self._KEY, None)

    def teardown_method(self):
        import lookups
        lookups._gleif_cache.pop(self._KEY, None)

    def test_l1_hit_returns_cached_value(self):
        import lookups
        lookups._gleif_cache[self._KEY] = self._FAKE
        with patch("lookups._urlopen_json") as mock_http:
            result = lookups.gleif_search("Test Corp L1", "exact")
            mock_http.assert_not_called()
        assert result == self._FAKE

    def test_l1_hit_prevents_api_call(self):
        import lookups
        lookups._gleif_cache[self._KEY] = self._FAKE
        with patch("lookups._urlopen_json") as mock_http:
            lookups.gleif_search("Test Corp L1", "exact")
        mock_http.assert_not_called()


# ── GLEIF search: L2 cache ────────────────────────────────────────────────────

class TestGleifSearchL2Cache:
    """L2 SQLite cache for gleif_search."""

    _KEY = ("test corp l2", "exact")
    _FAKE = _json_mod.dumps({
        "results": [{"lei": "L2LEI", "name": "Test Corp L2", "status": "PUBLISHED",
                     "category": "GENERAL", "jurisdiction": "US",
                     "registration_status": "ISSUED"}],
        "status": "success",
    })

    def teardown_method(self):
        import lookups
        lookups._gleif_cache.pop(self._KEY, None)

    def test_l2_hit_no_api_call(self, temp_cache):
        import lookups
        temp_cache.put("gleif", self._KEY, self._FAKE)
        lookups._gleif_cache.pop(self._KEY, None)
        with patch("lookups._urlopen_json") as mock_http:
            result = lookups.gleif_search("Test Corp L2", "exact")
            mock_http.assert_not_called()
        assert result == self._FAKE

    def test_l2_hit_populates_l1(self, temp_cache):
        """After L2 hit the value is promoted into L1 (write-through)."""
        import lookups
        temp_cache.put("gleif", self._KEY, self._FAKE)
        lookups._gleif_cache.pop(self._KEY, None)
        with patch("lookups._urlopen_json"):
            lookups.gleif_search("Test Corp L2", "exact")
        assert self._KEY in lookups._gleif_cache


# ── GLEIF candidates: L1 cache ────────────────────────────────────────────────

class TestGleifCandidatesL1Cache:
    """L1 dict cache for gleif_get_candidates."""

    _TERM_RAW = "Candidates Corp L1"
    _TERM_KEY = ("candidates corp l1",)  # (term.lower(),) — same tuple key drives L1 and L2
    _CANDS = [{"lei": "CAND1", "name": "Candidates Corp L1", "match_type": "exact",
               "category": "GENERAL", "registration_status": "ISSUED"}]

    def setup_method(self):
        import lookups
        lookups._gleif_candidates_cache.pop(self._TERM_KEY, None)

    def teardown_method(self):
        import lookups
        lookups._gleif_candidates_cache.pop(self._TERM_KEY, None)

    def test_l1_hit_returns_candidates(self):
        import lookups
        lookups._gleif_candidates_cache[self._TERM_KEY] = self._CANDS
        with patch("lookups.gleif_search") as mock_search:
            result = lookups.gleif_get_candidates(self._TERM_RAW)
            mock_search.assert_not_called()
        assert result == self._CANDS

    def test_l1_hit_returns_copy_not_reference(self):
        """Mutating the returned list does not corrupt L1."""
        import lookups
        lookups._gleif_candidates_cache[self._TERM_KEY] = self._CANDS
        result = lookups.gleif_get_candidates(self._TERM_RAW)
        result.append({"lei": "EXTRA"})
        assert len(lookups._gleif_candidates_cache[self._TERM_KEY]) == 1


# ── GLEIF candidates: L2 cache ────────────────────────────────────────────────

class TestGleifCandidatesL2Cache:
    """L2 SQLite cache for gleif_get_candidates."""

    _TERM_RAW = "Candidates Corp L2"
    _L2_KEY = ("candidates corp l2",)   # same tuple key for L1 and L2
    _TERM_KEY = _L2_KEY
    _CANDS = [{"lei": "CAND2", "name": "Candidates Corp L2", "match_type": "names",
               "category": "GENERAL", "registration_status": "ISSUED"}]

    def teardown_method(self):
        import lookups
        lookups._gleif_candidates_cache.pop(self._TERM_KEY, None)

    def test_l2_hit_no_search_call(self, temp_cache):
        import lookups
        temp_cache.put("gleif_candidates", self._L2_KEY, _json_mod.dumps(self._CANDS))
        lookups._gleif_candidates_cache.pop(self._TERM_KEY, None)
        with patch("lookups.gleif_search") as mock_search:
            result = lookups.gleif_get_candidates(self._TERM_RAW)
            mock_search.assert_not_called()
        assert result == self._CANDS

    def test_l2_hit_populates_l1(self, temp_cache):
        import lookups
        temp_cache.put("gleif_candidates", self._L2_KEY, _json_mod.dumps(self._CANDS))
        lookups._gleif_candidates_cache.pop(self._TERM_KEY, None)
        with patch("lookups.gleif_search"):
            lookups.gleif_get_candidates(self._TERM_RAW)
        assert self._TERM_KEY in lookups._gleif_candidates_cache


# ── GLEIF parent lookup: negative-result caching ─────────────────────────────

class TestGleifParentNegativeCache:
    """_fetch_gleif_parent must cache a deterministic 'no GENERAL parent' result.

    Its compute() returns (value, cacheable=True) for a deterministic outcome, so
    lookup_cache.cached stores even a None (DESIGN_INVARIANTS.md #5: a deterministic
    no-match is a valid, cacheable answer). Without this, every subsidiary-looking
    candidate with no useful parent re-hits the live GLEIF API on every future
    resolution instead of being memoized like every other lookup.
    """

    def teardown_method(self):
        import lookups
        lookups._gleif_cache.pop(("nogenparent1", "parent"), None)
        lookups._gleif_cache.pop(("nogenparent2", "parent"), None)

    def test_no_records_found_is_cached(self, temp_cache):
        import lookups
        with patch("lookups._urlopen_json", return_value={"data": []}) as mock_http:
            first = lookups._fetch_gleif_parent("NoGenParent1")
            mock_http.assert_called_once()
        assert first is None
        with patch("lookups._urlopen_json") as mock_http2:
            second = lookups._fetch_gleif_parent("NoGenParent1")
            mock_http2.assert_not_called()
        assert second is None

    def test_non_general_parent_is_cached_as_none(self, temp_cache):
        import lookups
        _branch_record = {
            "data": [{
                "attributes": {
                    "lei": "BRANCHLEI",
                    "entity": {
                        "legalName": {"name": "Some Branch"},
                        "category": "BRANCH",
                    },
                    "registration": {"status": "ISSUED"},
                },
            }],
        }
        with patch("lookups._urlopen_json", return_value=_branch_record) as mock_http:
            first = lookups._fetch_gleif_parent("NoGenParent2")
            mock_http.assert_called_once()
        assert first is None
        with patch("lookups._urlopen_json") as mock_http2:
            second = lookups._fetch_gleif_parent("NoGenParent2")
            mock_http2.assert_not_called()
        assert second is None


# ── Defensive-copy contract: cached values cannot be corrupted by a caller ───

class TestResolverCacheDefensiveCopy:
    """A caller that mutates a resolved entity must NOT corrupt the L1 cache.

    The batched resolve_batch path stores the pick result in L1 via _cache_store
    and returns the same logical value to the caller. _cache_store must copy into
    L1 (F1) so the two are independent — matching cached_async(copy=dict) used by
    the per-entity path. Likewise _fetch_gleif_parent must hand back a copy (F2).
    """

    def test_gleif_cache_store_isolates_l1_from_caller(self):
        from agents import gleif_resolver as gr
        key = ("defensive gleif", "", "")
        gr._gleif_pick_cache.pop(key, None)
        result = {"lei": "LEI1", "name": "Corp"}
        gr._cache_store(key, result)
        result["name"] = "MUTATED"                       # caller mutates its own copy
        assert gr._gleif_pick_cache[key]["name"] == "Corp"
        gr._gleif_pick_cache.pop(key, None)

    def test_umls_cache_store_isolates_l1_from_caller(self):
        from agents import umls_resolver as ur
        key = ("defensive umls", "", "")
        ur._umls_pick_cache.pop(key, None)
        result = {"cui": "C1", "name": "Drug"}
        ur._cache_store(key, result)
        result["name"] = "MUTATED"
        assert ur._umls_pick_cache[key]["name"] == "Drug"
        ur._umls_pick_cache.pop(key, None)

    def test_fetch_gleif_parent_returns_copy_not_l1_ref(self, temp_cache):
        import lookups
        lookups._gleif_cache.pop(("lei_copy", "parent"), None)
        _parent_record = {
            "data": [{
                "attributes": {
                    "lei": "PARENTLEI",
                    "entity": {"legalName": {"name": "Parent Co"}, "category": "GENERAL"},
                    "registration": {"status": "ISSUED"},
                },
            }],
        }
        with patch("lookups._urlopen_json", return_value=_parent_record):
            first = lookups._fetch_gleif_parent("lei_copy")
        assert first and first["name"] == "Parent Co"
        first["name"] = "MUTATED"                        # caller mutates returned dict
        with patch("lookups._urlopen_json") as mock_http:
            second = lookups._fetch_gleif_parent("lei_copy")  # served from cache
            mock_http.assert_not_called()
        assert second["name"] == "Parent Co"             # cache untouched by the mutation
        lookups._gleif_cache.pop(("lei_copy", "parent"), None)


# ── UMLS search: L1 cache ─────────────────────────────────────────────────────

class TestUmlsSearchL1Cache:
    """L1 dict cache for umls_search."""

    # cache_key = (term_lower, search_type, sabs, semantic_types, page_size)
    _KEY = ("test drug l1", "words", "", "", 5)
    _FAKE = _json_mod.dumps({
        "results": [{"cui": "C0000001", "name": "Test Drug L1",
                     "semantic_types": ["Pharmacologic Substance"], "root_source": "RXNORM"}],
        "status": "success",
    })

    def setup_method(self):
        import lookups
        lookups._umls_cache.pop(self._KEY, None)

    def teardown_method(self):
        import lookups
        lookups._umls_cache.pop(self._KEY, None)

    def test_l1_hit_returns_cached_value(self):
        import lookups
        lookups._umls_cache[self._KEY] = self._FAKE
        with patch("lookups._urlopen_json") as mock_http:
            result = lookups.umls_search("Test Drug L1", "words")
            mock_http.assert_not_called()
        assert result == self._FAKE

    def test_l1_hit_prevents_api_call(self):
        import lookups
        lookups._umls_cache[self._KEY] = self._FAKE
        with patch("lookups._urlopen_json") as mock_http:
            lookups.umls_search("Test Drug L1")
        mock_http.assert_not_called()


# ── UMLS search: L2 cache ─────────────────────────────────────────────────────

class TestUmlsSearchL2Cache:
    """L2 SQLite cache for umls_search."""

    _KEY = ("test drug l2", "words", "", "", 5)
    _FAKE = _json_mod.dumps({
        "results": [{"cui": "C0000002", "name": "Test Drug L2",
                     "semantic_types": ["Pharmacologic Substance"], "root_source": "RXNORM"}],
        "status": "success",
    })

    def teardown_method(self):
        import lookups
        lookups._umls_cache.pop(self._KEY, None)

    def test_l2_hit_no_api_call(self, temp_cache):
        import lookups
        temp_cache.put("umls", self._KEY, self._FAKE)
        lookups._umls_cache.pop(self._KEY, None)
        with patch("lookups._urlopen_json") as mock_http:
            result = lookups.umls_search("Test Drug L2", "words")
            mock_http.assert_not_called()
        assert result == self._FAKE

    def test_l2_hit_populates_l1(self, temp_cache):
        import lookups
        temp_cache.put("umls", self._KEY, self._FAKE)
        lookups._umls_cache.pop(self._KEY, None)
        with patch("lookups._urlopen_json"):
            lookups.umls_search("Test Drug L2", "words")
        assert self._KEY in lookups._umls_cache


# ── GLEIF pick cache: L1 ─────────────────────────────────────────────────────

class TestGleifPickCacheL1:
    """L1 dict cache for the GLEIF resolver LLM pick step (_resolve_one)."""

    _NAME = "gleif pick l1 corp"
    from agents.gleif_resolver import _pick_cache_key as _pck
    _KEY = _pck(_NAME, "")   # (name.lower(), domain_hint[:100], abbr expansion)
    _RESULT = {"lei": "PICLEI1", "name": "GLEIF Pick L1 Corp", "category": "GENERAL"}

    def setup_method(self):
        from agents import gleif_resolver as gr
        gr._gleif_pick_cache.pop(self._KEY, None)

    def teardown_method(self):
        from agents import gleif_resolver as gr
        gr._gleif_pick_cache.pop(self._KEY, None)

    def test_l1_hit_skips_do_resolve(self, tmp_path):
        from agents import gleif_resolver as gr
        gr._gleif_pick_cache[self._KEY] = self._RESULT
        with patch.object(gr, "_do_resolve_one", new_callable=AsyncMock) as mock_do:
            result = asyncio.run(gr._resolve_one(self._NAME, "Corporation", {}, tmp_path))
            mock_do.assert_not_called()
        assert result == self._RESULT

    def test_l1_hit_returns_copy_not_reference(self, tmp_path):
        """Mutating the returned dict does not corrupt L1."""
        from agents import gleif_resolver as gr
        gr._gleif_pick_cache[self._KEY] = self._RESULT
        result = asyncio.run(gr._resolve_one(self._NAME, "Corporation", {}, tmp_path))
        result["injected"] = True
        assert "injected" not in gr._gleif_pick_cache[self._KEY]


# ── GLEIF pick cache: L2 ─────────────────────────────────────────────────────

class TestGleifPickCacheL2:
    """L2 SQLite cache for the GLEIF resolver LLM pick step."""

    _NAME = "gleif pick l2 corp"
    from agents.gleif_resolver import _pick_cache_key as _pck
    _KEY = _pck(_NAME, "")
    _RESULT = {"lei": "PICLEI2", "name": "GLEIF Pick L2 Corp", "category": "GENERAL"}

    def teardown_method(self):
        from agents import gleif_resolver as gr
        gr._gleif_pick_cache.pop(self._KEY, None)

    def test_l2_hit_no_do_resolve(self, tmp_path, temp_cache):
        from agents import gleif_resolver as gr
        temp_cache.put("gleif_pick", self._KEY, _json_mod.dumps(self._RESULT))
        gr._gleif_pick_cache.pop(self._KEY, None)
        with patch.object(gr, "_do_resolve_one", new_callable=AsyncMock) as mock_do:
            result = asyncio.run(gr._resolve_one(self._NAME, "Corp", {}, tmp_path))
            mock_do.assert_not_called()
        assert result == self._RESULT

    def test_l2_hit_populates_l1(self, tmp_path, temp_cache):
        from agents import gleif_resolver as gr
        temp_cache.put("gleif_pick", self._KEY, _json_mod.dumps(self._RESULT))
        gr._gleif_pick_cache.pop(self._KEY, None)
        with patch.object(gr, "_do_resolve_one", new_callable=AsyncMock):
            asyncio.run(gr._resolve_one(self._NAME, "Corp", {}, tmp_path))
        assert self._KEY in gr._gleif_pick_cache

    def test_write_through_on_fresh_resolve(self, tmp_path, temp_cache):
        """_do_resolve_one result is written to both L1 and L2."""
        from agents import gleif_resolver as gr
        name = "gleif pick fresh corp"
        key = gr._pick_cache_key(name, "")
        result_data = {"lei": "FRESHPIC", "name": "GLEIF Pick Fresh Corp"}
        gr._gleif_pick_cache.pop(key, None)

        async def _mock_do(*args, **kwargs):
            return result_data

        with patch.object(gr, "_do_resolve_one", side_effect=_mock_do):
            result = asyncio.run(gr._resolve_one(name, "Corp", {}, tmp_path))

        assert result == result_data
        assert key in gr._gleif_pick_cache
        assert temp_cache.get("gleif_pick", key) is not None
        gr._gleif_pick_cache.pop(key, None)


# ── UMLS pick cache: L1 ──────────────────────────────────────────────────────

class TestUmlsPickCacheL1:
    """L1 dict cache for the UMLS resolver LLM pick step (_resolve_one)."""

    _NAME = "umls pick l1 drug"
    _NODE_DEF: dict = {}   # → key suffix ("", "") via _umls_pick_key
    _RESULT = {"cui": "C0001001", "name": "UMLS Pick L1 Drug",
               "semantic_types": ["Pharmacologic Substance"]}

    @staticmethod
    def _key(name: str, node_def: dict):
        from agents.umls_resolver import _umls_pick_key
        return _umls_pick_key(name, node_def)

    def setup_method(self):
        from agents import umls_resolver as ur
        ur._umls_pick_cache.pop(self._key(self._NAME, self._NODE_DEF), None)

    def teardown_method(self):
        from agents import umls_resolver as ur
        ur._umls_pick_cache.pop(self._key(self._NAME, self._NODE_DEF), None)

    def test_l1_hit_skips_do_resolve(self, tmp_path):
        from agents import umls_resolver as ur
        key = self._key(self._NAME, self._NODE_DEF)
        ur._umls_pick_cache[key] = self._RESULT
        with patch.object(ur, "_do_resolve_one", new_callable=AsyncMock) as mock_do:
            result = asyncio.run(ur._resolve_one(self._NAME, "Drug", self._NODE_DEF, tmp_path))
            mock_do.assert_not_called()
        assert result == self._RESULT

    def test_l1_hit_returns_copy_not_reference(self, tmp_path):
        from agents import umls_resolver as ur
        key = self._key(self._NAME, self._NODE_DEF)
        ur._umls_pick_cache[key] = self._RESULT
        result = asyncio.run(ur._resolve_one(self._NAME, "Drug", self._NODE_DEF, tmp_path))
        result["injected"] = True
        assert "injected" not in ur._umls_pick_cache[key]


# ── UMLS pick cache: L2 ──────────────────────────────────────────────────────

class TestUmlsPickCacheL2:
    """L2 SQLite cache for the UMLS resolver LLM pick step."""

    _NAME = "umls pick l2 drug"
    _NODE_DEF: dict = {}
    _RESULT = {"cui": "C0001002", "name": "UMLS Pick L2 Drug",
               "semantic_types": ["Pharmacologic Substance"]}

    @staticmethod
    def _key(name: str, node_def: dict):
        from agents.umls_resolver import _umls_pick_key
        return _umls_pick_key(name, node_def)

    def teardown_method(self):
        from agents import umls_resolver as ur
        ur._umls_pick_cache.pop(self._key(self._NAME, self._NODE_DEF), None)

    def test_l2_hit_no_do_resolve(self, tmp_path, temp_cache):
        from agents import umls_resolver as ur
        key = self._key(self._NAME, self._NODE_DEF)
        temp_cache.put("umls_pick", key, _json_mod.dumps(self._RESULT))
        ur._umls_pick_cache.pop(key, None)
        with patch.object(ur, "_do_resolve_one", new_callable=AsyncMock) as mock_do:
            result = asyncio.run(ur._resolve_one(self._NAME, "Drug", self._NODE_DEF, tmp_path))
            mock_do.assert_not_called()
        assert result == self._RESULT

    def test_l2_hit_populates_l1(self, tmp_path, temp_cache):
        from agents import umls_resolver as ur
        key = self._key(self._NAME, self._NODE_DEF)
        temp_cache.put("umls_pick", key, _json_mod.dumps(self._RESULT))
        ur._umls_pick_cache.pop(key, None)
        with patch.object(ur, "_do_resolve_one", new_callable=AsyncMock):
            asyncio.run(ur._resolve_one(self._NAME, "Drug", self._NODE_DEF, tmp_path))
        assert key in ur._umls_pick_cache

    def test_write_through_on_fresh_resolve(self, tmp_path, temp_cache):
        """_do_resolve_one result is written to both L1 and L2."""
        from agents import umls_resolver as ur
        name = "umls pick fresh drug"
        key = self._key(name, self._NODE_DEF)
        result_data = {"cui": "C0001003", "name": "UMLS Pick Fresh Drug",
                       "semantic_types": ["Pharmacologic Substance"]}
        ur._umls_pick_cache.pop(key, None)

        async def _mock_do(*args, **kwargs):
            return result_data

        with patch.object(ur, "_do_resolve_one", side_effect=_mock_do):
            result = asyncio.run(ur._resolve_one(name, "Drug", self._NODE_DEF, tmp_path))

        assert result == result_data
        assert key in ur._umls_pick_cache
        assert temp_cache.get("umls_pick", key) is not None
        ur._umls_pick_cache.pop(key, None)


# ══════════════════════════════════════════════════════════════════════════
# Grounding re-anchor + batched resolver picks + semantic-check quote_context
# ══════════════════════════════════════════════════════════════════════════
import grounding
from extract import _verify_grounding as _verify_grounding_for_tests


# ── grounding: locate / reanchor / context ────────────────────────────────────

class TestGroundingLocate:
    DOC = ("NVIDIA designs the Blackwell GPU and Grace CPU, but **outsources "
           "fabrication to TSMC** ([[5]] anand.com). Memory chips from SK Hynix/Micron.")

    def test_plain_substring_located(self):
        assert grounding.locate("NVIDIA designs the Blackwell GPU", self.DOC) is not None

    def test_empty_quote_returns_none(self):
        assert grounding.locate("", self.DOC) is None
        assert grounding.locate("   ", self.DOC) is None

    def test_paraphrase_not_located(self):
        # "manufacturers" never appears — genuine drift must fail
        assert grounding.locate("connector and cable manufacturers exist", self.DOC) is None

    def test_span_offsets_valid(self):
        span = grounding.locate("Memory chips from SK Hynix/Micron", self.DOC)
        assert span is not None
        s, e = span
        assert self.DOC[s:e] == "Memory chips from SK Hynix/Micron"

    def test_runon_period_missing_space(self):
        # PDF artifact: no space after the sentence period ("TSMC.On"). The quote
        # begins right after it and must still locate verbatim.
        doc = "outsources fabrication to TSMC.On the other, OEMs assemble the racks."
        span = grounding.locate("On the other, OEMs assemble the racks", doc)
        assert span is not None
        assert doc[span[0]:span[1]] == "On the other, OEMs assemble the racks"

    def test_runon_split_spares_acronyms_and_decimals(self):
        # Dotted acronyms, decimals, "etc.)", and lowercase domains must NOT be split.
        for s in ("the U.S. Senate", "grew 3.5 percent", "(Foxconn, etc.) today", "via Amazon.com"):
            assert grounding._split_runon_periods(s) == s
        assert grounding._split_runon_periods("to TSMC.On the") == "to TSMC On the"

    def test_quote_with_inline_citation_locates(self):
        # The doc blanks "([[N]] url)" citations before tokenising; the quote must be
        # blanked the same way, or its citation tokens break the contiguous match.
        doc = ("Over 50 unique subcomponent categories must be sourced "
               "([[1]] newsletter.semianalysis.com), with multiple vendors each.")
        quote = ("Over 50 unique subcomponent categories must be sourced "
                 "([[1]] newsletter.semianalysis.com), with multiple vendors each")
        span = grounding.locate(quote, doc)
        assert span is not None
        # span covers the real prose, citation included in the located range
        assert "subcomponent categories" in doc[span[0]:span[1]]

    # Silent structural elision: the LLM quotes a list intro then jumps to a later
    # italic-header section, dropping the sections between WITHOUT a "..." marker.
    # The strict contiguous match fails; the structural-boundary fallback recovers it.
    SILENT_ELISION_DOC = (
        "Other adverse reactions reported include: _Body as a Whole:_ malaise, pyrexia "
        "_Digestive System:_ flatulence, hepatitis "
        "_Metabolic and Nutritional System:_ transaminases increase, hyperglycemia"
    )

    def test_silent_structural_elision_located(self):
        # Body/Digestive sections are dropped from the quote with no ellipsis.
        quote = ("Other adverse reactions reported include: "
                 "_Metabolic and Nutritional System:_ transaminases increase, hyperglycemia")
        span = grounding.locate(quote, self.SILENT_ELISION_DOC)
        assert span is not None
        assert self.SILENT_ELISION_DOC[span[0]:span[1]].endswith("hyperglycemia")

    def test_silent_elision_fallback_no_false_positive(self):
        # Same structural shape, but the content phrases are fabricated → must fail.
        quote = ("Other adverse reactions reported include: "
                 "_Imaginary System:_ teleportation sickness, spontaneous levitation")
        assert grounding.locate(quote, self.SILENT_ELISION_DOC) is None

    # Markdown table rows use tight pipes (|Dyspepsia|4.3|...|) which otherwise
    # tokenise as one blob no quote can match. Pipes are treated as separators.
    TABLE_DOC = ("|**Adverse Reaction**|**Placebo**|**Any dose**|\n"
                 "|Dyspepsia|4.3|4.7|\n|Nausea|3.5|4.0|")

    def test_table_row_located_tight_pipes(self):
        span = grounding.locate("Dyspepsia|4.3|4.7", self.TABLE_DOC)
        assert span is not None
        assert self.TABLE_DOC[span[0]:span[1]] == "Dyspepsia|4.3|4.7"

    def test_table_row_located_spaced_pipes(self):
        # LLM often re-spaces the pipes; must still match the tight-pipe doc row.
        assert grounding.locate("Dyspepsia | 4.3 | 4.7", self.TABLE_DOC) is not None

    def test_table_row_fabricated_not_located(self):
        assert grounding.locate("Teleportation | 9.9 | 8.8", self.TABLE_DOC) is None


class TestGroundingReanchor:
    def test_strips_surrounding_double_quotes(self):
        doc = "Acme Corp supplies widgets to Globex."
        q = '"Acme Corp supplies widgets to Globex."'
        assert grounding.reanchor(q, doc) == "Acme Corp supplies widgets to Globex"  # trailing . excluded (word-span)

    def test_curly_quote_swap_tolerated(self):
        doc = "Micron’s CEO said NVIDIA is “one of its primary customers” for HBM3e."
        q = "Micron's CEO said NVIDIA is 'one of its primary customers' for HBM3e"
        r = grounding.reanchor(q, doc)
        assert r is not None and r.startswith("Micron")

    def test_inline_citation_skipped(self):
        doc = "shipped Blackwell ([[32]] wccftech.com). Similarly, Foxconn delivered racks."
        q = "shipped Blackwell. Similarly, Foxconn delivered racks"
        assert grounding.reanchor(q, doc) is not None

    def test_ellipsis_elision(self):
        doc = "ASE Group (Taiwan) (with Powertech in affiliate) performs CoWoS packaging."
        q = "ASE Group (Taiwan) ... performs CoWoS packaging"
        assert grounding.locate(q, doc) is not None

    def test_leading_markdown_excluded_from_span(self):
        doc = "**ASE Group** is a top OSAT firm."
        r = grounding.reanchor("ASE Group is a top OSAT firm", doc)
        assert r is not None and r.startswith("ASE Group")  # leading ** not in span

    def test_returns_verbatim_doc_substring(self):
        doc = "Powertech Technology (PTI) supplies substrates."
        # LLM wraps + swaps quote style; re-anchor must yield real doc text
        q = "“Powertech Technology (PTI) supplies substrates.”"
        r = grounding.reanchor(q, doc)
        # span covers word cores: leading/trailing markup + sentence period excluded
        assert r == "Powertech Technology (PTI) supplies substrates"

    def test_unlocatable_returns_none(self):
        assert grounding.reanchor("totally invented sentence", "real document text here") is None


class TestGroundingContext:
    def test_window_around_span(self):
        doc = ("X " * 100) + "alpha beta gamma " + ("Y " * 100)
        ctx = grounding.context("alpha beta gamma", doc, window=10)
        assert "alpha beta gamma" in ctx
        assert len(ctx) < len(doc)

    def test_unlocatable_returns_empty(self):
        assert grounding.context("nope nope nope", "some other text", window=10) == ""


class TestGroundingMarkdownArtifacts:
    def test_trademark_bracket_tolerated(self):
        # pymupdf4llm renders ® as [®]; the LLM quote uses the raw ® symbol
        doc = "DIFICID[®] (fidaxomicin) tablets, for oral use."
        q = "DIFICID® (fidaxomicin) tablets, for oral use"
        assert grounding.reanchor(q, doc) is not None

    def test_trademark_symbols_stripped_both_sides(self):
        doc = "Acme™ Corp© and Beta® supply parts."
        assert grounding.locate("Acme Corp and Beta supply parts", doc) is not None

    def test_html_tags_blanked(self):
        # converter renders bullet headers with <u>…</u>; quote drops the tags
        doc = "• <u>Pancreatitis:</u> Has been reported in clinical trials."
        q = "Pancreatitis: Has been reported in clinical trials"
        assert grounding.reanchor(q, doc) is not None

    def test_bullet_glyph_tolerated(self):
        doc = "Side effects:\n• vomiting\n• stomach pain"
        assert grounding.locate("vomiting", doc) is not None

    def test_math_lt_not_treated_as_html(self):
        # "< 5" must NOT be blanked as an HTML tag
        doc = "incidence < 5 percent in adults"
        assert grounding.locate("incidence < 5 percent", doc) is not None


# ── Point 2: re-anchor inside _verify_grounding ───────────────────────────────

class TestVerifyGroundingReanchor:
    RELS = [{"rel_type": "R", "from_field": "f", "to_field": "t"}]

    def test_moderate_reanchors_located_quote(self):
        doc = "Acme Corp supplies widgets to Globex under a 2024 deal."
        items = {"R": [{"f": "Acme", "t": "Globex",
                        "supporting_quote": '"Acme Corp supplies widgets to Globex under a 2024 deal."'}]}
        dropped, warned, warns = _verify_grounding_for_tests(items, self.RELS, doc, "moderate")
        assert (dropped, warned) == (0, 0)
        # quote overwritten with verbatim doc substring (surrounding quotes gone)
        assert items["R"][0]["supporting_quote"] == "Acme Corp supplies widgets to Globex under a 2024 deal"

    def test_moderate_warns_and_keeps_paraphrase(self):
        doc = "connector and cable makers exist in the chain."
        items = {"R": [{"f": "x", "t": "y",
                        "supporting_quote": "connector and cable manufacturers exist"}]}
        dropped, warned, warns = _verify_grounding_for_tests(items, self.RELS, doc, "moderate")
        assert (dropped, warned) == (0, 1)
        assert len(warns) == 1
        # not overwritten — original suspect quote preserved for downstream review
        assert items["R"][0]["supporting_quote"] == "connector and cable manufacturers exist"

    def test_strict_drops_paraphrase(self):
        doc = "connector and cable makers exist in the chain."
        items = {"R": [{"f": "x", "t": "y",
                        "supporting_quote": "connector and cable manufacturers exist"}]}
        dropped, warned, warns = _verify_grounding_for_tests(items, self.RELS, doc, "strict")
        assert dropped == 1
        assert items["R"] == []  # dropped

    def test_loose_passes_through_untouched(self):
        doc = "anything"
        original = "completely different text"
        items = {"R": [{"f": "x", "t": "y", "supporting_quote": original}]}
        _verify_grounding_for_tests(items, self.RELS, doc, "loose")
        assert items["R"][0]["supporting_quote"] == original  # loose never re-anchors

    def test_tolerates_sparse_all_items_dict(self):
        # rels lists a rel_type with no extracted items — must not KeyError
        rels = [
            {"rel_type": "R", "from_field": "f", "to_field": "t"},
            {"rel_type": "EMPTY", "from_field": "f", "to_field": "t"},
        ]
        doc = "Acme Corp supplies widgets to Globex."
        items = {"R": [{"f": "Acme", "t": "Globex",
                        "supporting_quote": "Acme Corp supplies widgets to Globex"}]}
        # "EMPTY" absent from items — uses all_items.get(rel_type, [])
        dropped, warned, warns = _verify_grounding_for_tests(items, rels, doc, "moderate")
        assert (dropped, warned) == (0, 0)

    def test_slash_joined_quote_reanchored_per_segment(self):
        # LLM stitches two separate doc lines with " / "; no single span covers it,
        # but every segment locates → store joined verbatim segments, no warning.
        doc = ("DIFICID tablets, for oral use Initial U.S. Approval: 2011. "
               "Later: DIFICID is indicated for the treatment of CDAD.")
        quote = ("DIFICID tablets, for oral use / "
                 "DIFICID is indicated for the treatment of CDAD")
        items = {"R": [{"f": "x", "t": "y", "supporting_quote": quote}]}
        dropped, warned, warns = _verify_grounding_for_tests(items, self.RELS, doc, "moderate")
        assert (dropped, warned) == (0, 0)
        stored = items["R"][0]["supporting_quote"]
        assert " / " in stored
        for seg in stored.split(" / "):
            assert seg in doc  # each segment verbatim

    def test_slash_joined_partial_match_warns(self):
        # one segment is a genuine paraphrase → not fully grounded → warn, keep original
        doc = "DIFICID tablets, for oral use."
        quote = "DIFICID tablets, for oral use / totally invented clause here"
        items = {"R": [{"f": "x", "t": "y", "supporting_quote": quote}]}
        dropped, warned, warns = _verify_grounding_for_tests(items, self.RELS, doc, "moderate")
        assert (dropped, warned) == (0, 1)
        assert items["R"][0]["supporting_quote"] == quote  # untouched

    def test_bullet_intro_plus_item_reanchored(self):
        # LLM pairs a list intro with a non-adjacent bullet item; other bullets sit
        # between them in the doc, so it isn't one contiguous span — but both the
        # intro and the item are verbatim, so per-segment re-anchor grounds it.
        doc = ("The most common side effects in adults include:\n\n"
               "* nausea\n* vomiting\n* stomach pain")
        quote = "The most common side effects in adults include: * vomiting"
        items = {"R": [{"f": "x", "t": "y", "supporting_quote": quote}]}
        dropped, warned, warns = _verify_grounding_for_tests(items, self.RELS, doc, "moderate")
        assert (dropped, warned) == (0, 0)
        for seg in items["R"][0]["supporting_quote"].split(" / "):
            assert seg in doc  # each stored segment verbatim


# ── Point 1: batched GLEIF resolver picks ─────────────────────────────────────

class _FakeBatchPick:
    def __init__(self, entry_id, index, retry=""):
        self.entry_id = entry_id
        self.index = index
        self.retry_search_term = retry


class _FakeBatchRes:
    def __init__(self, picks):
        self.picks = picks


class TestGleifBatchedPick:
    def _cands(self):
        return {
            "alpha": [
                {"lei": "A1", "name": "Alpha Group Inc", "category": "GENERAL",
                 "registration_status": "ISSUED", "match_type": "exact"},
                {"lei": "A2", "name": "Alpha Subsidiary", "category": "BRANCH",
                 "registration_status": "LAPSED", "match_type": "fuzzy"},
            ],
            "beta": [
                {"lei": "B1", "name": "Beta Holdings Corp", "category": "GENERAL",
                 "registration_status": "ISSUED", "match_type": "exact"},
                {"lei": "B2", "name": "Beta Lyon", "category": "BRANCH",
                 "registration_status": "ISSUED", "match_type": "fuzzy"},
            ],
        }

    def test_two_ambiguous_resolved_in_one_llm_call(self, tmp_path, temp_cache):
        from agents import gleif_resolver as gr
        gr._gleif_pick_cache.clear()
        cands = self._cands()
        fake = _FakeBatchRes([_FakeBatchPick(0, 1), _FakeBatchPick(1, 1)])
        with patch.object(gr, "gleif_get_candidates", side_effect=lambda t: cands.get(t.lower(), [])), \
             patch.object(gr, "_acreate_structured_output", new=AsyncMock(return_value=fake)) as mock_llm:
            out = asyncio.run(gr.GLEIFResolver().resolve_batch(
                [("Alpha", "Corp"), ("Beta", "Corp")], {"Corp": {}}, tmp_path,
                gr.ResolveContext(domain_hint="chips")))
        assert mock_llm.await_count == 1  # ONE batched call for two entities
        assert out[("Alpha", "Corp")]["name"] == "Alpha Group Inc"
        assert out[("Beta", "Corp")]["name"] == "Beta Holdings Corp"
        assert "match_type" not in out[("Alpha", "Corp")]  # provenance stripped
        gr._gleif_pick_cache.clear()

    def test_single_candidate_skips_llm(self, tmp_path, temp_cache):
        from agents import gleif_resolver as gr
        gr._gleif_pick_cache.clear()
        cands = {"solo": [{"lei": "S1", "name": "Solo Inc", "category": "GENERAL",
                           "registration_status": "ISSUED", "match_type": "exact"}]}
        with patch.object(gr, "gleif_get_candidates", side_effect=lambda t: cands.get(t.lower(), [])), \
             patch.object(gr, "_acreate_structured_output", new_callable=AsyncMock) as mock_llm:
            out = asyncio.run(gr.GLEIFResolver().resolve_batch(
                [("Solo", "Corp")], {"Corp": {}}, tmp_path, gr.ResolveContext()))
        mock_llm.assert_not_called()
        assert out[("Solo", "Corp")]["name"] == "Solo Inc"
        gr._gleif_pick_cache.clear()

    def test_cache_hit_skips_gather_and_llm(self, tmp_path, temp_cache):
        from agents import gleif_resolver as gr
        key = gr._pick_cache_key("Cached Corp", "")
        temp_cache.put("gleif_pick", key, _json_mod.dumps({"lei": "C1", "name": "Cached Corp"}))
        gr._gleif_pick_cache.pop(key, None)
        with patch.object(gr, "gleif_get_candidates") as mock_g, \
             patch.object(gr, "_acreate_structured_output", new_callable=AsyncMock) as mock_llm:
            out = asyncio.run(gr.GLEIFResolver().resolve_batch(
                [("Cached Corp", "Corp")], {"Corp": {}}, tmp_path, gr.ResolveContext()))
        mock_g.assert_not_called()
        mock_llm.assert_not_called()
        assert out[("Cached Corp", "Corp")]["name"] == "Cached Corp"
        gr._gleif_pick_cache.pop(key, None)

    def test_picked_result_is_cached(self, tmp_path, temp_cache):
        from agents import gleif_resolver as gr
        gr._gleif_pick_cache.clear()
        cands = self._cands()
        fake = _FakeBatchRes([_FakeBatchPick(0, 1)])
        with patch.object(gr, "gleif_get_candidates", side_effect=lambda t: cands.get(t.lower(), [])), \
             patch.object(gr, "_acreate_structured_output", new=AsyncMock(return_value=fake)):
            asyncio.run(gr.GLEIFResolver().resolve_batch(
                [("Alpha", "Corp")], {"Corp": {}}, tmp_path, gr.ResolveContext(domain_hint="d")))
        key = gr._pick_cache_key("Alpha", "d")
        assert key in gr._gleif_pick_cache
        assert temp_cache.get("gleif_pick", key) is not None
        gr._gleif_pick_cache.clear()

    def test_abbr_map_differs_no_cross_doc_poison(self):
        """Same token, different per-doc abbr expansion → distinct pick keys.
        Regression: omitting abbr_map poisoned picks across docs (HPE = Health
        Plan East vs Hewlett Packard)."""
        from agents import gleif_resolver as gr
        k_a = gr._pick_cache_key("HPE", "d", {"HPE": "Health Plan East"})
        k_b = gr._pick_cache_key("HPE", "d", {"HPE": "Hewlett Packard Enterprise"})
        assert k_a != k_b
        # No abbr entry for this token → key unchanged (no over-invalidation).
        k_none = gr._pick_cache_key("HPE", "d", {"IBM": "International Business Machines"})
        assert k_none == gr._pick_cache_key("HPE", "d")


# ── Abbreviation expansion: doc-derived abbr_map displaces the LLM ─────────────

class TestGleifAbbrevExpansion:
    """A short all-caps term with no strong direct GLEIF hit triggers expansion.
    Source order must be: doc abbr_map → static _ABBREV_TABLE → LLM fallback.
    The first two are deterministic and must skip the _expand_abbreviation LLM call."""

    @staticmethod
    def _gc_for(expanded_name):
        """gleif_get_candidates stub: empty for the abbrev, one strong hit for its expansion."""
        def _gc(term):
            if term == expanded_name:
                return [{"lei": "L1", "name": expanded_name, "category": "GENERAL",
                         "registration_status": "ISSUED", "match_type": "exact"}]
            return []  # the bare abbreviation finds nothing strong
        return _gc

    def test_abbr_map_expands_without_llm(self):
        from agents import gleif_resolver as gr
        # "ZQX" is not in the static table, so only the doc map can avoid the LLM.
        with patch.object(gr, "gleif_get_candidates", side_effect=self._gc_for("Zeta Quantum X Corp")), \
             patch.object(gr, "_expand_abbreviation", new_callable=AsyncMock) as mock_llm:
            cands = asyncio.run(gr._gather_candidates(
                "ZQX", {}, "", {"ZQX": "Zeta Quantum X Corp"}))
        mock_llm.assert_not_called()                       # doc map used, no LLM
        assert any(c["name"] == "Zeta Quantum X Corp" for c in cands)

    def test_static_table_used_without_llm(self):
        from agents import gleif_resolver as gr
        # "HPE" is in _ABBREV_TABLE; even with an empty doc map, no LLM call.
        with patch.object(gr, "gleif_get_candidates",
                          side_effect=self._gc_for("Hewlett Packard Enterprise Company")), \
             patch.object(gr, "_expand_abbreviation", new_callable=AsyncMock) as mock_llm:
            cands = asyncio.run(gr._gather_candidates("HPE", {}, "", {}))
        mock_llm.assert_not_called()
        assert any("Hewlett Packard" in c["name"] for c in cands)

    def test_unknown_abbrev_falls_through_to_llm(self):
        from agents import gleif_resolver as gr
        # Not in doc map, not in static table → LLM fallback must still fire.
        with patch.object(gr, "gleif_get_candidates", side_effect=self._gc_for("Zeta Quantum X Corp")), \
             patch.object(gr, "_expand_abbreviation",
                          new=AsyncMock(return_value="Zeta Quantum X Corp")) as mock_llm:
            cands = asyncio.run(gr._gather_candidates("ZQX", {}, "", {}))   # empty doc map
        mock_llm.assert_awaited_once()
        assert any(c["name"] == "Zeta Quantum X Corp" for c in cands)

    def test_doc_map_preferred_over_static_table(self):
        from agents import gleif_resolver as gr
        # "HPE" is in the static table, but the doc defines it differently.
        # The doc's own definition must win (more faithful), and no LLM.
        with patch.object(gr, "gleif_get_candidates", side_effect=self._gc_for("Health Plan East")), \
             patch.object(gr, "_expand_abbreviation", new_callable=AsyncMock) as mock_llm:
            cands = asyncio.run(gr._gather_candidates(
                "HPE", {}, "", {"HPE": "Health Plan East"}))
        mock_llm.assert_not_called()
        assert any(c["name"] == "Health Plan East" for c in cands)   # doc map, not static "Hewlett Packard"

    def test_resolve_batch_threads_abbr_map_end_to_end(self, tmp_path, temp_cache):
        """The full path: ResolveContext.abbr_map reaches the expansion, no LLM at all."""
        from agents import gleif_resolver as gr
        gr._gleif_pick_cache.clear()
        with patch.object(gr, "gleif_get_candidates", side_effect=self._gc_for("Zeta Quantum X Corp")), \
             patch.object(gr, "_expand_abbreviation", new_callable=AsyncMock) as mock_expand, \
             patch.object(gr, "_acreate_structured_output", new_callable=AsyncMock) as mock_pick:
            out = asyncio.run(gr.GLEIFResolver().resolve_batch(
                [("ZQX", "Corp")], {"Corp": {}}, tmp_path,
                gr.ResolveContext(abbr_map={"ZQX": "Zeta Quantum X Corp"})))
        mock_expand.assert_not_called()    # doc map expansion, no LLM
        mock_pick.assert_not_called()      # single resolved candidate, no LLM pick
        assert out[("ZQX", "Corp")]["name"] == "Zeta Quantum X Corp"
        gr._gleif_pick_cache.clear()


# ── Point 1: batched UMLS resolver picks ──────────────────────────────────────

class TestUmlsBatchedPick:
    def _cands(self):
        return {
            "drugx": [
                {"cui": "C1", "name": "Drug X Compound", "semantic_types": ["Pharmacologic Substance"]},
                {"cui": "C2", "name": "Drug X Variant", "semantic_types": ["Clinical Drug"]},
            ],
            "drugy": [
                {"cui": "C3", "name": "Drug Y Compound", "semantic_types": ["Pharmacologic Substance"]},
                {"cui": "C4", "name": "Drug Y Variant", "semantic_types": ["Clinical Drug"]},
            ],
        }

    def test_two_ambiguous_resolved_in_one_llm_call(self, tmp_path, temp_cache):
        from agents import umls_resolver as ur
        ur._umls_pick_cache.clear()
        cands = self._cands()
        fake = _FakeBatchRes([_FakeBatchPick(0, 1), _FakeBatchPick(1, 1)])
        with patch.object(ur, "_get_all_candidates",
                          side_effect=lambda name, sabs, sg, st: cands.get(name.lower(), [])), \
             patch.object(ur, "_acreate_structured_output", new=AsyncMock(return_value=fake)) as mock_llm:
            out = asyncio.run(ur.UMLSResolver().resolve_batch(
                [("DrugX", "Drug"), ("DrugY", "Drug")], {"Drug": {}}, tmp_path, ur.ResolveContext()))
        assert mock_llm.await_count == 1
        assert out[("DrugX", "Drug")]["cui"] == "C1"
        assert out[("DrugY", "Drug")]["cui"] == "C3"
        ur._umls_pick_cache.clear()

    def test_exact_match_skips_llm(self, tmp_path, temp_cache):
        from agents import umls_resolver as ur
        ur._umls_pick_cache.clear()
        cands = {"aspirin": [
            {"cui": "C9", "name": "Aspirin", "semantic_types": ["Pharmacologic Substance"]},
            {"cui": "C8", "name": "Aspirin Low Dose", "semantic_types": ["Clinical Drug"]},
        ]}
        with patch.object(ur, "_get_all_candidates",
                          side_effect=lambda name, sabs, sg, st: cands.get(name.lower(), [])), \
             patch.object(ur, "_acreate_structured_output", new_callable=AsyncMock) as mock_llm:
            out = asyncio.run(ur.UMLSResolver().resolve_batch(
                [("Aspirin", "Drug")], {"Drug": {}}, tmp_path, ur.ResolveContext()))
        mock_llm.assert_not_called()  # exact name match resolves without LLM
        assert out[("Aspirin", "Drug")]["cui"] == "C9"
        ur._umls_pick_cache.clear()


# ── Point 4: semantic check quote_context ─────────────────────────────────────

class TestSemanticCheckQuoteContext:
    def _triple(self, quote):
        return {
            "_id": "t1", "rel_type": "R", "from_label": "O", "to_label": "O",
            "from_pk": "name", "to_pk": "name",
            "from_props": {"name": "Acme Corp"}, "to_props": {"name": "Globex"},
            "supporting_quote": quote,
        }

    def test_context_attached_when_quote_beyond_excerpt(self):
        import agents.semantic_check_agent as sc
        doc = ("HEADER. " * 50) + "Acme Corp supplies Globex worldwide."
        t = self._triple("Acme Corp supplies Globex worldwide.")
        items = sc._build_items([t], {}, {}, {}, schema_nodes={}, doc_text=doc, excerpt_len=80)
        assert "quote_context" in items[0]
        assert "Acme Corp supplies Globex" in items[0]["quote_context"]

    def test_no_context_when_quote_within_excerpt(self):
        import agents.semantic_check_agent as sc
        doc = "Acme Corp supplies Globex worldwide. " + ("X " * 100)
        t = self._triple("Acme Corp supplies Globex worldwide.")
        items = sc._build_items([t], {}, {}, {}, schema_nodes={}, doc_text=doc, excerpt_len=10000)
        assert "quote_context" not in items[0]

    def test_no_context_when_excerpt_len_zero(self):
        import agents.semantic_check_agent as sc
        doc = ("HEADER. " * 50) + "Acme Corp supplies Globex worldwide."
        t = self._triple("Acme Corp supplies Globex worldwide.")
        items = sc._build_items([t], {}, {}, {}, schema_nodes={}, doc_text=doc, excerpt_len=0)
        assert "quote_context" not in items[0]  # back-compat: disabled when 0

    def test_context_handles_slash_joined_quotes(self):
        # extract.py merges per-edge quotes with " / "; the whole join won't locate
        # as one span, but the first beyond-excerpt segment should still attach context.
        import agents.semantic_check_agent as sc
        doc = ("HEADER. " * 50) + "Acme Corp supplies Globex. Globex resells to Initech."
        joined = "Acme Corp supplies Globex. / Globex resells to Initech."
        t = self._triple(joined)
        items = sc._build_items([t], {}, {}, {}, schema_nodes={}, doc_text=doc, excerpt_len=80)
        assert "quote_context" in items[0]

    # ── Step 1: deterministic quote_verbatim flag ────────────────────────────
    # The quote is re-anchored to verbatim doc text before the check runs, so a
    # quote that locates IS in the document — we hand that proof to the LLM via
    # `quote_verbatim` instead of paying it to re-judge presence.

    def test_quote_verbatim_true_when_located_within_excerpt(self):
        import agents.semantic_check_agent as sc
        doc = "Acme Corp supplies Globex worldwide. " + ("X " * 100)
        t = self._triple("Acme Corp supplies Globex worldwide.")
        items = sc._build_items([t], {}, {}, {}, schema_nodes={}, doc_text=doc, excerpt_len=10000)
        assert items[0]["quote_verbatim"] is True
        assert "quote_context" not in items[0]  # visible in excerpt → no neighborhood needed

    def test_quote_verbatim_true_when_located_beyond_excerpt(self):
        import agents.semantic_check_agent as sc
        doc = ("HEADER. " * 50) + "Acme Corp supplies Globex worldwide."
        t = self._triple("Acme Corp supplies Globex worldwide.")
        items = sc._build_items([t], {}, {}, {}, schema_nodes={}, doc_text=doc, excerpt_len=80)
        assert items[0]["quote_verbatim"] is True
        assert "quote_context" in items[0]  # beyond excerpt → context attached for entailment

    def test_quote_verbatim_false_when_not_in_doc(self):
        import agents.semantic_check_agent as sc
        doc = "Acme Corp supplies Globex worldwide. " + ("X " * 100)
        t = self._triple("Initech manufactures rocket engines in orbit.")  # not in doc
        items = sc._build_items([t], {}, {}, {}, schema_nodes={}, doc_text=doc, excerpt_len=10000)
        assert items[0]["quote_verbatim"] is False

    def test_quote_verbatim_false_when_multi_segment_partial(self):
        # All-segment contract: a " / "-joined quote with one un-located segment
        # is NOT verbatim, must match extract._verify_grounding (re-anchors only
        # when EVERY segment locates). Regression for debug-260620-1532 finding 4.
        import agents.semantic_check_agent as sc
        doc = "Acme Corp supplies Globex worldwide. " + ("X " * 100)
        t = self._triple("Acme Corp supplies Globex worldwide. / Initech builds rockets.")
        items = sc._build_items([t], {}, {}, {}, schema_nodes={}, doc_text=doc, excerpt_len=10000)
        assert items[0]["quote_verbatim"] is False

    def test_quote_verbatim_true_when_all_segments_located(self):
        import agents.semantic_check_agent as sc
        doc = "Acme Corp supplies Globex worldwide. Beta provides Gamma chips. " + ("X " * 100)
        t = self._triple("Acme Corp supplies Globex worldwide. / Beta provides Gamma chips.")
        items = sc._build_items([t], {}, {}, {}, schema_nodes={}, doc_text=doc, excerpt_len=10000)
        assert items[0]["quote_verbatim"] is True


# ── Single-subject document detection ─────────────────────────────────────────

class TestDocumentSubjectDetection:
    """_build_items flags the sole entity of its label as the document subject.

    Fixes the over-red regression on drug labels: an adverse-effect quote ("the most
    common adverse reactions are nausea, ...") grounds the AE but not the drug, yet the
    whole label is about that one drug, so the relation IS supported. The signal is
    domain-agnostic — 'sole entity of its label' — so it fires for a one-drug label but
    NOT for a many-company supply-chain article (where CoolIT->NVIDIA must stay red).
    """

    @staticmethod
    def _ae_triple(tid, drug, ae):
        return {
            "_id": tid, "rel_type": "HAS_ADVERSE_EFFECT",
            "from_label": "Substance", "to_label": "AdverseEffect",
            "from_pk": "name", "to_pk": "name",
            "from_props": {"name": drug}, "to_props": {"name": ae},
            "supporting_quote": f"adverse reactions include {ae}",
        }

    @staticmethod
    def _supply_triple(tid, frm, to):
        return {
            "_id": tid, "rel_type": "PROVIDES",
            "from_label": "Corporation", "to_label": "Corporation",
            "from_pk": "name", "to_pk": "name",
            "from_props": {"name": frm}, "to_props": {"name": to},
            "supporting_quote": f"{frm} supplies {to}",
        }

    def test_sole_drug_is_subject_many_aes_are_not(self):
        import agents.semantic_check_agent as sc
        triples = [
            self._ae_triple("t1", "fidaxomicin", "Nausea"),
            self._ae_triple("t2", "fidaxomicin", "Vomiting"),
            self._ae_triple("t3", "fidaxomicin", "Anemia"),
        ]
        items = {i["triple_id"]: i for i in sc._build_items(triples, {}, {}, {}, schema_nodes={})}
        # The one drug (sole Substance, recurs in 3 triples) is the subject.
        assert all(items[t].get("from_is_document_subject") for t in ("t1", "t2", "t3"))
        # The adverse effects are many distinct entities → none is the subject.
        assert all("to_is_document_subject" not in items[t] for t in ("t1", "t2", "t3"))

    def test_multi_company_article_has_no_subject(self):
        import agents.semantic_check_agent as sc
        triples = [
            self._supply_triple("t1", "CoolIT", "NVIDIA"),
            self._supply_triple("t2", "Asetek", "NVIDIA"),
            self._supply_triple("t3", "TSMC", "NVIDIA"),
        ]
        items = {i["triple_id"]: i for i in sc._build_items(triples, {}, {}, {}, schema_nodes={})}
        # Many distinct Corporations → no sole label → no entity flagged as subject,
        # so CoolIT->NVIDIA is judged strictly (stays red), per the documented case.
        for t in ("t1", "t2", "t3"):
            assert "from_is_document_subject" not in items[t]
            assert "to_is_document_subject" not in items[t]

    def test_single_triple_does_not_make_both_sides_subject(self):
        import agents.semantic_check_agent as sc
        # A lone triple: each label has 1 entity but appears only once (<2) — the ≥2
        # recurrence guard prevents treating a one-off entity as a document subject.
        items = sc._build_items([self._ae_triple("t1", "fidaxomicin", "Nausea")],
                                {}, {}, {}, schema_nodes={})
        assert "from_is_document_subject" not in items[0]
        assert "to_is_document_subject" not in items[0]


class TestInstructionsInUserMsg:
    """Extraction instructions are shown to the grader only when provided (variant B).

    The A/B experiment (gpt-5.5, n=3) picked B 9/9 vs A 7/9: passing the meta
    instructions lets the grader green a primary drug's adverse effects even when a
    stray comparator (a 2nd Substance) defeats the deterministic 'sole entity' flag,
    while 'capture regardless of primary subject' wording keeps multi-subject docs strict.
    """

    def test_instructions_block_present_when_passed(self):
        import agents.semantic_check_agent as sc
        msg = sc._build_user_msg([{"triple_id": "t1"}], "DOC EXCERPT",
                                  "Extract for the PRIMARY substance.")
        assert "Extraction instructions for this document" in msg
        assert "PRIMARY substance" in msg
        assert "DOC EXCERPT" in msg

    def test_no_block_when_instructions_empty(self):
        import agents.semantic_check_agent as sc
        msg = sc._build_user_msg([{"triple_id": "t1"}], "DOC EXCERPT", "")
        assert "Extraction instructions for this document" not in msg
        # whitespace-only is treated as empty too
        msg2 = sc._build_user_msg([{"triple_id": "t1"}], "DOC EXCERPT", "   \n ")
        assert "Extraction instructions for this document" not in msg2


class TestEntityPresenceValidator:
    """Deterministic green anchor: an entity located verbatim in its own quote is
    pinned green (from_color_anchor / to_color_anchor) and the LLM cannot lower it.

    This removes the false-red entity class. Scope is tight: matches the RESOLVED
    name only (never the raw term, so a suspicious resolution is left to flag), the
    entity axis only (constraint_violated / direction stays the LLM's), and a red
    color_floor (semantic-type mismatch, fabricated quote) still wins the EDGE color.
    """

    @staticmethod
    def _triple(tid="t1", frm="Acme Components Inc.", to="Globex Systems",
                quote="Acme Components Inc. supplies memory modules to Globex Systems."):
        return {
            "_id": tid, "rel_type": "PROVIDES",
            "from_label": "Corporation", "to_label": "Corporation",
            "from_pk": "name", "to_pk": "name",
            "from_props": {"name": frm}, "to_props": {"name": to},
            "supporting_quote": quote,
        }

    def test_presence_both_sides(self):
        import agents.semantic_check_agent as sc
        f, t = sc.EntityPresenceValidator()._presence(self._triple())
        assert (f, t) == (True, True)

    def test_presence_subject_absent_from_quote(self):
        import agents.semantic_check_agent as sc
        # A drug-label AE quote names the AE but not the drug → only the to-side is
        # proven present; the from-side (sole subject) is left to the LLM/subject rule.
        t = {
            "_id": "ae", "rel_type": "HAS_ADVERSE_EFFECT",
            "from_label": "Substance", "to_label": "AdverseEffect",
            "from_pk": "name", "to_pk": "name",
            "from_props": {"name": "fidaxomicin"}, "to_props": {"name": "Nausea"},
            "supporting_quote": "the most common adverse reactions are nausea, vomiting",
        }
        assert sc.EntityPresenceValidator()._presence(t) == (False, True)

    def test_resolved_name_absent_no_raw_term_is_not_anchored(self):
        import agents.semantic_check_agent as sc
        # Resolved "QUANTA LYON" is not in the quote and no raw term is supplied →
        # nothing to anchor on the from side.
        t = self._triple(frm="QUANTA LYON", to="NVIDIA",
                         quote="Quanta supplies servers to NVIDIA.")
        f, to = sc.EntityPresenceValidator()._presence(t)
        assert f is False      # "QUANTA LYON" not contiguous in the quote, no raw term
        assert to is True      # NVIDIA is

    # ── #2: extend the anchor to the document's own surface form (raw term) + synonyms ──
    _UMLS_CTX_KEY = {"properties": [{"source": "umls"}]}
    _GLEIF_CTX_KEY = {"properties": [{"source": "gleif"}]}

    def _ctx(self, **labels):
        import agents.semantic_check_agent as sc
        return sc.ValidatorContext(schema_nodes=labels)

    def test_raw_term_is_valid_umls_synonym(self):
        import agents.semantic_check_agent as sc
        # Doc said "kidney stone"; resolver rewrote to "Nephrolithiasis". The raw term is
        # the document's surface form → a valid synonym for a same-concept (UMLS) source.
        t = {
            "_id": "u", "rel_type": "MAY_TREAT",
            "from_label": "Substance", "to_label": "Indication",
            "from_pk": "name", "to_pk": "name",
            "from_props": {"name": "atorvastatin"}, "to_props": {"name": "Nephrolithiasis"},
            "from_term": "atorvastatin", "to_term": "kidney stone",
            "supporting_quote": "atorvastatin reduces the risk of kidney stone",
        }
        ctx = self._ctx(Substance=self._UMLS_CTX_KEY, Indication=self._UMLS_CTX_KEY)
        assert sc.EntityPresenceValidator()._presence(t, ctx) == (True, True)

    def test_raw_term_gated_for_suspicious_gleif_resolution(self):
        import agents.semantic_check_agent as sc
        # GLEIF "Quanta" -> "QUANTA LYON" is suspicious (a likely subsidiary), so the raw
        # term is NOT used to anchor — the resolution is left for its validator / the LLM.
        t = self._triple(frm="QUANTA LYON", to="NVIDIA",
                         quote="Quanta supplies servers to NVIDIA.")
        t["from_term"] = "Quanta"; t["to_term"] = "NVIDIA"
        ctx = self._ctx(Corporation=self._GLEIF_CTX_KEY)
        f, to = sc.EntityPresenceValidator()._presence(t, ctx)
        assert f is False      # gated: suspicious GLEIF resolution not greened on "Quanta"
        assert to is True

    def test_raw_term_ok_for_nonsuspicious_gleif_resolution(self):
        import agents.semantic_check_agent as sc
        # "Quanta" -> "Quanta Computer Inc." is NOT suspicious (resolved is the same company
        # plus a legal suffix), so the raw term legitimately anchors the from side green.
        t = self._triple(frm="Quanta Computer Inc.", to="NVIDIA",
                         quote="Quanta supplies servers to NVIDIA.")
        t["from_term"] = "Quanta"; t["to_term"] = "NVIDIA"
        ctx = self._ctx(Corporation=self._GLEIF_CTX_KEY)
        assert sc.EntityPresenceValidator()._presence(t, ctx) == (True, True)

    def test_resolver_synonyms_meta_hook(self):
        import agents.semantic_check_agent as sc
        # A resolver-supplied synonym in to_meta is always trusted (vetted).
        t = {
            "_id": "s", "rel_type": "MAY_TREAT",
            "from_label": "Substance", "to_label": "Indication",
            "from_pk": "name", "to_pk": "name",
            "from_props": {"name": "atorvastatin"}, "to_props": {"name": "Nephrolithiasis"},
            "to_meta": {"synonyms": ["renal calculi"]},
            "supporting_quote": "reduces renal calculi formation",
        }
        ctx = self._ctx(Substance=self._UMLS_CTX_KEY, Indication=self._UMLS_CTX_KEY)
        f, to = sc.EntityPresenceValidator()._presence(t, ctx)
        assert to is True      # matched via the synonym, not the resolved name

    def test_multi_segment_best_of_segment(self):
        import agents.semantic_check_agent as sc
        # Entity present in ONE ' / '-joined segment is enough.
        t = self._triple(quote="some unrelated intro / Acme Components Inc. ships to Globex Systems")
        assert sc.EntityPresenceValidator()._presence(t) == (True, True)

    def test_annotate_item_injects_flags(self):
        import agents.semantic_check_agent as sc
        item = {}
        sc.EntityPresenceValidator().annotate_item(self._triple(), item, sc.ValidatorContext())
        assert item.get("from_entity_in_quote") is True
        assert item.get("to_entity_in_quote") is True

    def test_check_returns_green_anchors(self):
        import agents.semantic_check_agent as sc
        out = sc.EntityPresenceValidator().check([self._triple()], sc.ValidatorContext())
        v = out["t1"]
        assert v.from_color_anchor == "green"
        assert v.to_color_anchor == "green"
        # Anchors never touch the relation axis.
        assert v.constraint_violated is False
        assert v.color_floor is None

    def test_anchor_overrides_llm_red_on_entity_axis(self, monkeypatch):
        import agents.semantic_check_agent as sc
        # LLM wrongly reds both entities though both are verbatim in the quote.
        def _fake(batch, doc_excerpt, instructions=""):
            return {i["triple_id"]: sc._TripleGrounding(
                triple_id=i["triple_id"], from_color="red", to_color="red",
                constraint_violated=False, opinion="entities not seen") for i in batch}
        monkeypatch.setattr(sc, "_call_llm_batch", _fake)
        out = {t["_id"]: t for t in sc.check_triples([self._triple()], "doc")}
        t = out["t1"]
        # Proven presence pins both entities green; with no cv the edge is green too.
        assert t["from_color"] == "green"
        assert t["to_color"] == "green"
        assert t["triple_color"] == "green"

    def test_anchor_does_not_override_constraint_violation(self, monkeypatch):
        import agents.semantic_check_agent as sc
        # Both entities present (anchored green) BUT the LLM finds the relation reversed
        # (constraint_violated). The edge must stay red — anchors are entity-axis only.
        def _fake(batch, doc_excerpt, instructions=""):
            return {i["triple_id"]: sc._TripleGrounding(
                triple_id=i["triple_id"], from_color="red", to_color="red",
                constraint_violated=True, opinion="direction reversed") for i in batch}
        monkeypatch.setattr(sc, "_call_llm_batch", _fake)
        out = {t["_id"]: t for t in sc.check_triples([self._triple()], "doc")}
        t = out["t1"]
        assert t["from_color"] == "green" and t["to_color"] == "green"  # entity axis anchored
        assert t["triple_color"] == "red"            # relation axis red wins the edge
        assert t["constraint_violated"] is True

    def test_red_color_floor_still_wins_edge_over_anchor(self, monkeypatch):
        import agents.semantic_check_agent as sc
        # A deterministic red floor (simulate via a fabricated agent_retry quote with
        # an unlocatable evidence span) must still red the EDGE even though the entity
        # names appear in the (fabricated) quote and get anchored green.
        def _fake(batch, doc_excerpt, instructions=""):
            return {i["triple_id"]: sc._TripleGrounding(
                triple_id=i["triple_id"], from_color="green", to_color="green",
                constraint_violated=False, opinion="ok") for i in batch}
        monkeypatch.setattr(sc, "_call_llm_batch", _fake)
        t = self._triple()
        t["extraction_source"] = "agent_retry"
        t["evidence"] = [{"start": None, "end": None, "text": t["supporting_quote"]}]
        out = {x["_id"]: x for x in sc.check_triples([t], "doc")}
        # FabricatedQuoteValidator red-floors the edge (no cv); anchors keep entity colors green.
        assert out["t1"]["triple_color"] == "red"


class TestRelationSupportHintValidator:
    """#3 — deterministic, ADVISORY signals for the LLM's relation-support judgment.

    Annotate-only: it injects precomputed facts (endpoints colocated in one segment;
    a negation cue present) and never returns a verdict — support/direction stays the LLM's.
    """

    @staticmethod
    def _t(quote, frm="Acme", to="Globex"):
        return {
            "_id": "h", "rel_type": "PROVIDES",
            "from_label": "Corporation", "to_label": "Corporation",
            "from_pk": "name", "to_pk": "name",
            "from_props": {"name": frm}, "to_props": {"name": to},
            "supporting_quote": quote,
        }

    def _ctx(self):
        import agents.semantic_check_agent as sc
        return sc.ValidatorContext(schema_nodes={"Corporation": {"properties": [{"source": "gleif"}]}})

    def test_colocated_single_segment(self):
        import agents.semantic_check_agent as sc
        item = {}
        sc.RelationSupportHintValidator().annotate_item(
            self._t("Acme supplies memory to Globex."), item, self._ctx())
        assert item.get("relation_endpoints_colocated") is True

    def test_not_colocated_when_endpoints_in_different_segments(self):
        import agents.semantic_check_agent as sc
        item = {}
        sc.RelationSupportHintValidator().annotate_item(
            self._t("Acme makes chips / Globex builds servers"), item, self._ctx())
        assert "relation_endpoints_colocated" not in item

    def test_negation_cue_flagged(self):
        import agents.semantic_check_agent as sc
        item = {}
        sc.RelationSupportHintValidator().annotate_item(
            self._t("Nephrotoxicity was not observed in humans."), item, self._ctx())
        assert item.get("possible_negation_in_quote") is True

    def test_no_negation_false_positive_on_substring(self):
        import agents.semantic_check_agent as sc
        item = {}
        # "another"/"cannot"-style substrings must not trip the word-bounded cue; this
        # quote has no real negation, only "Globex" and ordinary supply prose.
        sc.RelationSupportHintValidator().annotate_item(
            self._t("Acme supplies modules to Globex for another product line."), item, self._ctx())
        assert "possible_negation_in_quote" not in item

    def test_annotate_only_no_verdict(self):
        import agents.semantic_check_agent as sc
        out = sc.RelationSupportHintValidator().check([self._t("Acme supplies Globex")], sc.ValidatorContext())
        assert out == {}   # never returns a verdict


# ── #3: composable deterministic validators ──────────────────────────────────

class TestDeterministicValidators:
    def _triple(self, tid="t1", from_name="Drug A", to_name="Disease B", rel="MAY_TREAT"):
        return {
            "_id": tid, "rel_type": rel,
            "from_label": "Drug", "to_label": "Disease",
            "from_pk": "name", "to_pk": "name",
            "from_props": {"name": from_name}, "to_props": {"name": to_name},
            "supporting_quote": "q",
        }

    def test_structural_validator_flags_empty_pk(self):
        import agents.semantic_check_agent as sc
        t = self._triple()
        t["to_props"] = {}  # empty to PK
        out = sc.StructuralValidator().check([t], sc.ValidatorContext())
        assert "t1" in out
        assert out["t1"].note.startswith("[Structural:")
        assert out["t1"].color_floor == "red"         # empty PK = structurally invalid -> hard red floor
        assert "is empty" in out["t1"].note

    def test_structural_validator_flags_duplicate(self):
        import agents.semantic_check_agent as sc
        a = self._triple(tid="a")
        b = self._triple(tid="b")  # identical (fpv, tpv, rel)
        out = sc.StructuralValidator().check([a, b], sc.ValidatorContext())
        assert "Duplicate" in out["b"].note
        assert "a" not in out  # first occurrence is clean

    def test_schema_conformance_validator(self):
        import agents.semantic_check_agent as sc
        rels = [
            {"rel_type": "MAY_TREAT", "from_node": "Substance", "to_node": "Indication"},
            {"rel_type": "HAS_ADVERSE_EFFECT", "from_node": "Substance", "to_node": "AdverseEffect"},
        ]
        ctx = sc.ValidatorContext(schema_rels=rels)
        v = sc.SchemaConformanceValidator()

        def T(tid, rel, fl, tl):
            return {"_id": tid, "rel_type": rel, "from_label": fl, "to_label": tl}

        # legal edge -> clean
        assert v.check([T("a", "MAY_TREAT", "Substance", "Indication")], ctx) == {}
        # undeclared label / rel_type / reversed all flagged
        assert "b" in v.check([T("b", "HAS_ADVERSE_EFFECT", "Drug", "AdverseEffect")], ctx)
        assert "c" in v.check([T("c", "CAUSES", "Substance", "AdverseEffect")], ctx)
        assert "e" in v.check([T("e", "HAS_ADVERSE_EFFECT", "AdverseEffect", "Substance")], ctx)
        # legal parts, illegal combination -> hard red floor + constraint_violated
        r = v.check([T("d", "MAY_TREAT", "Substance", "AdverseEffect")], ctx)
        assert r["d"].color_floor == "red" and r["d"].constraint_violated
        # no schema_rels -> no-op (cannot check without the edge list)
        assert v.check([T("f", "CAUSES", "X", "Y")], sc.ValidatorContext()) == {}
        # schema_rels present but no rel carries from_node/to_node (e.g. a hints-only
        # fixture) -> no derivable legal set -> must NOT red-floor everything.
        hints_only = sc.ValidatorContext(schema_rels=[
            {"rel_type": "MAY_TREAT", "from_hint": "drug", "to_hint": "disease"},
        ])
        assert v.check([T("g", "MAY_TREAT", "Substance", "Indication")], hints_only) == {}

    def test_harvest_validator_forces_red(self, tmp_path):
        import agents.semantic_check_agent as sc
        hd = tmp_path / "harvest"
        hd.mkdir()
        (hd / "MAY_TREAT.jsonl").write_text(
            json.dumps({"source": "rejected", "rel_type": "MAY_TREAT",
                        "from_display": "Drug A", "to_display": "Disease B",
                        "doc_name": "doc1"}) + "\n",
            encoding="utf-8",
        )
        t = self._triple()
        ctx = sc.ValidatorContext(harvest_dir=hd, doc_name="doc1")
        out = sc.HarvestRejectionValidator().check([t], ctx)
        assert out["t1"].color_floor == "red"
        assert out["t1"].constraint_violated is True
        assert out["t1"].note == "[Previously rejected by human reviewer]"

    def test_harvest_validator_doc_scoped(self, tmp_path):
        import agents.semantic_check_agent as sc
        hd = tmp_path / "harvest"
        hd.mkdir()
        (hd / "MAY_TREAT.jsonl").write_text(
            json.dumps({"source": "rejected", "rel_type": "MAY_TREAT",
                        "from_display": "Drug A", "to_display": "Disease B",
                        "doc_name": "other_doc"}) + "\n",
            encoding="utf-8",
        )
        t = self._triple()
        ctx = sc.ValidatorContext(harvest_dir=hd, doc_name="doc1")
        assert sc.HarvestRejectionValidator().check([t], ctx) == {}  # different doc → no match

    def test_check_triples_no_llm_applies_deterministic(self, tmp_path, monkeypatch):
        # Force the single LLM batch to fail → exercises the no-LLM merge branch.
        import agents.semantic_check_agent as sc

        def _boom(*a, **k):
            raise RuntimeError("no model in test")
        monkeypatch.setattr(sc, "_call_llm_batch", _boom)

        hd = tmp_path / "harvest"
        hd.mkdir()
        (hd / "MAY_TREAT.jsonl").write_text(
            json.dumps({"source": "rejected", "rel_type": "MAY_TREAT",
                        "from_display": "Drug A", "to_display": "Disease B"}) + "\n",
            encoding="utf-8",
        )
        rejected = self._triple(tid="r")
        dup_a = self._triple(tid="da", from_name="X", to_name="Y", rel="R2")
        dup_b = self._triple(tid="db", from_name="X", to_name="Y", rel="R2")

        out = sc.check_triples(
            [rejected, dup_a, dup_b], doc_text="some doc",
            harvest_dir=hd,
        )
        by_id = {t["_id"]: t for t in out}
        # harvest rejection forces red + cv even without an LLM verdict
        assert by_id["r"]["triple_color"] == "red"
        assert by_id["r"]["constraint_violated"] is True
        assert "[Previously rejected by human reviewer]" in by_id["r"]["ai_opinion"]
        # structural note-only: duplicate flagged, but no recolor/cv on the dup
        assert "[Structural:" in by_id["db"]["ai_opinion"]
        assert "Duplicate" in by_id["db"]["ai_opinion"]
        assert "triple_color" not in by_id["db"]      # note-only never sets color
        assert by_id["db"].get("_ai_reviewed") is False


from extract import _rescue_to_segment


class TestRescueToSegment:
    DOC = (
        "## ADVERSE REACTIONS\n"
        "Most common adverse reactions are nasopharyngitis, arthralgia, diarrhea.\n"
        "|Nasopharyngitis|8.2|12.9|5.3|\n"
    )

    def test_skips_when_to_already_in_emitted_segment(self):
        # emitted quote already names the target -> nothing to rescue
        assert _rescue_to_segment(
            "nasopharyngitis", self.DOC, ["|Nasopharyngitis|8.2|12.9|5.3|"]
        ) is None

    def test_rescues_full_line_when_to_absent_from_segments(self):
        # emitted quote was a bad caption -> rescue a verbatim line naming `to`
        out = _rescue_to_segment("nasopharyngitis", self.DOC, ["## ADVERSE REACTIONS"])
        assert out is not None
        assert "nasopharyngitis" in out.lower()
        assert "\n" not in out  # single line, trimmed

    def test_none_when_to_not_in_doc(self):
        assert _rescue_to_segment("myalgia", self.DOC, ["## ADVERSE REACTIONS"]) is None

    def test_none_on_empty_term(self):
        assert _rescue_to_segment("", self.DOC, []) is None


from extract import _build_evidence, _harvest_signature, _cache_key


class TestBuildEvidence:
    DOC = "Acme Corp supplies Globex worldwide. Other text here."

    def test_located_quote_gets_offsets_and_verbatim_text(self):
        ev = _build_evidence(["Acme Corp supplies Globex"], self.DOC)
        assert len(ev) == 1
        assert ev[0]["start"] is not None and ev[0]["end"] is not None
        assert self.DOC[ev[0]["start"]:ev[0]["end"]] == ev[0]["text"]
        assert "Acme Corp supplies Globex" in ev[0]["text"]

    def test_unlocatable_quote_keeps_null_offsets(self):
        ev = _build_evidence(["nonexistent phrase xyz"], self.DOC)
        assert ev == [{"start": None, "end": None, "text": "nonexistent phrase xyz"}]

    def test_dedup_and_blank_skip(self):
        ev = _build_evidence(["Acme Corp", "Acme Corp", "  ", ""], self.DOC)
        assert len(ev) == 1


class TestSemanticCheckEvidence:
    def _triple(self, evidence, quote="x"):
        return {
            "_id": "t1", "rel_type": "R", "from_label": "O", "to_label": "O",
            "from_pk": "name", "to_pk": "name",
            "from_props": {"name": "Acme Corp"}, "to_props": {"name": "Globex"},
            "supporting_quote": quote, "evidence": evidence,
        }

    def test_verbatim_true_when_all_spans_located(self):
        import agents.semantic_check_agent as sc
        doc = "Acme Corp supplies Globex worldwide."
        ev = [{"start": 0, "end": 9, "text": "Acme Corp"},
              {"start": 18, "end": 24, "text": "Globex"}]
        items = sc._build_items([self._triple(ev)], {}, {}, {}, schema_nodes={}, doc_text=doc, excerpt_len=10000)
        assert items[0]["quote_verbatim"] is True

    def test_verbatim_false_when_a_span_has_null_offset(self):
        import agents.semantic_check_agent as sc
        doc = "Acme Corp supplies Globex worldwide."
        ev = [{"start": 0, "end": 9, "text": "Acme Corp"},
              {"start": None, "end": None, "text": "missing"}]
        items = sc._build_items([self._triple(ev)], {}, {}, {}, schema_nodes={}, doc_text=doc, excerpt_len=10000)
        assert items[0]["quote_verbatim"] is False

    def test_context_sliced_from_offsets_beyond_excerpt(self):
        import agents.semantic_check_agent as sc
        doc = ("HEADER. " * 50) + "Acme Corp supplies Globex."
        s = doc.index("Acme Corp")
        ev = [{"start": s, "end": s + 26, "text": "Acme Corp supplies Globex."}]
        items = sc._build_items([self._triple(ev)], {}, {}, {}, schema_nodes={}, doc_text=doc, excerpt_len=80)
        assert "Acme Corp supplies Globex" in items[0]["quote_context"]


class TestHarvestSignatureCacheKey:
    def test_signature_changes_with_content(self, tmp_path):
        (tmp_path / "R.jsonl").write_text('{"a":1}\n')
        sig1 = _harvest_signature(tmp_path)
        (tmp_path / "R.jsonl").write_text('{"a":2}\n')
        sig2 = _harvest_signature(tmp_path)
        assert sig1 and sig2 and sig1 != sig2

    def test_empty_when_no_dir(self, tmp_path):
        assert _harvest_signature(tmp_path / "nope") == ""

    def test_cache_key_differs_on_harvest_sig(self):
        a = _cache_key("doc", "schema", harvest_sig="aaa")
        b = _cache_key("doc", "schema", harvest_sig="bbb")
        assert a != b


def test_harvest_block_labels_examples_as_patterns():
    from agents.harvest import format_examples_block
    block = format_examples_block(
        [{"source": "batch", "from_display": "A", "to_display": "B", "supporting_quote": "q"}],
        "R",
    )
    assert "not an exhaustive list" in block.lower()


class TestFabricatedQuoteValidator:
    def _v(self):
        import agents.semantic_check_agent as sc
        return sc.FabricatedQuoteValidator(), sc.ValidatorContext()

    def _t(self, tid, evidence, source="agent_retry", quote="some quote"):
        return {"_id": tid, "rel_type": "R", "from_props": {}, "to_props": {},
                "from_pk": "name", "to_pk": "name", "supporting_quote": quote,
                "evidence": evidence, "extraction_source": source}

    def test_agent_quote_all_null_spans_red(self):
        v, ctx = self._v()
        t = self._t("a", [{"start": None, "end": None, "text": "fabricated"}])
        out = v.check([t], ctx)
        assert out["a"].color_floor == "red"
        assert out["a"].constraint_violated is False
        assert "verbatim" in out["a"].note.lower()

    def test_agent_quote_with_located_span_not_flagged(self):
        v, ctx = self._v()
        t = self._t("a", [{"start": 0, "end": 5, "text": "real"}])
        assert v.check([t], ctx) == {}

    def test_batch_triple_excluded(self):
        v, ctx = self._v()
        t = self._t("a", [{"start": None, "end": None, "text": "x"}], source=None)
        assert v.check([t], ctx) == {}

    def test_no_quote_not_flagged(self):
        v, ctx = self._v()
        t = self._t("a", [], quote="")
        assert v.check([t], ctx) == {}

    def test_source_in_rel_props_also_detected(self):
        v, ctx = self._v()
        t = {"_id": "a", "rel_type": "R", "from_props": {}, "to_props": {},
             "from_pk": "name", "to_pk": "name", "supporting_quote": "q",
             "evidence": [{"start": None, "end": None, "text": "q"}],
             "rel_props": {"extraction_source": "agent_retry"}}
        assert v.check([t], ctx)["a"].color_floor == "red"


class TestAddTripleVerbatimGuard:
    def _session(self):
        import agents.extraction_agent as ea
        s = ea.ExtractionSession()
        s.doc_text = "Nirsevimab is an F protein-directed fusion inhibitor produced in cells."
        s.schema_nodes = {"Substance": {}, "MOA": {}}
        s.schema_rels = [{"rel_type": "HAS_MOA", "from_node": "Substance", "to_node": "MOA"}]
        return s

    def _add(self, s, quote):
        return s.add_triple(
            "HAS_MOA", "Substance", "name", '{"name":"Nirsevimab"}',
            "MOA", "name", '{"name":"Fusion Inhibitor"}', supporting_quote=quote,
        )

    def test_fabricated_quote_flagged_and_warned(self):
        s = self._session()
        r = self._add(s, "Nirsevimab neutralizes RSV via the prefusion epitope")
        assert "NOT found verbatim" in r
        assert s.triples[0].get("quote_unlocatable") is True

    def test_replace_on_retry_clears_flag_no_append(self):
        s = self._session()
        self._add(s, "Nirsevimab neutralizes RSV via the prefusion epitope")  # fabricated
        self._add(s, "F protein-directed fusion inhibitor")                    # verbatim retry
        t = s.triples[0]
        assert len(s.triples) == 1
        assert "quote_unlocatable" not in t
        assert " / " not in t["supporting_quote"]
        assert "fusion inhibitor" in t["supporting_quote"].lower()

    def test_verbatim_quote_not_flagged(self):
        s = self._session()
        self._add(s, "F protein-directed fusion inhibitor")
        assert "quote_unlocatable" not in s.triples[0]

    def test_fabrication_does_not_pollute_existing_good_quote(self):
        s = self._session()
        self._add(s, "F protein-directed fusion inhibitor")                    # good first
        self._add(s, "Nirsevimab neutralizes RSV via the prefusion epitope")   # fabricated dup
        t = s.triples[0]
        assert " / " not in t["supporting_quote"]
        assert "prefusion" not in t["supporting_quote"].lower()
        assert "quote_unlocatable" not in t


class TestAddTripleZeroGroundingReject:
    def _session(self):
        import agents.extraction_agent as ea
        s = ea.ExtractionSession()
        s.doc_text = "BEYFORTUS (nirsevimab) — most common adverse reactions were rash and anaphylaxis."
        s.schema_nodes = {"Drug": {}, "AE": {}}
        s.schema_rels = [{"rel_type": "HAS_AE", "from_node": "Drug", "to_node": "AE"}]
        return s

    def _add(self, s, to_term, quote):
        return s.add_triple(
            "HAS_AE", "Drug", "name", '{"name":"Nirsevimab"}',
            "AE", "name", '{"name":"Resolved Name"}',
            from_raw_term="BEYFORTUS", to_raw_term=to_term, supporting_quote=quote,
        )

    def test_invented_entity_hard_rejected(self):
        s = self._session()
        r = self._add(s, "cyanosis", "Hypersensitivity reactions including cyanosis were observed")
        assert "Rejected" in r and "cyanosis" in r
        assert len(s.triples) == 0  # not stored

    def test_real_entity_bad_quote_kept_red_not_rejected(self):
        # term IS in doc but quote is not verbatim → keep-and-red, not hard reject
        s = self._session()
        r = self._add(s, "rash", "patients experienced a mild rash after dosing")
        assert "Rejected" not in r
        assert len(s.triples) == 1
        assert s.triples[0].get("quote_unlocatable") is True

    def test_real_entity_real_quote_stored_clean(self):
        s = self._session()
        r = self._add(s, "rash", "most common adverse reactions were rash")
        assert "Rejected" not in r
        assert len(s.triples) == 1
        assert "quote_unlocatable" not in s.triples[0]

    def test_no_raw_term_falls_back_to_keep_and_red(self):
        # agent omitted raw term → cannot verify entity → keep-and-red, no hard reject
        s = self._session()
        r = s.add_triple("HAS_AE", "Drug", "name", '{"name":"Nirsevimab"}',
                         "AE", "name", '{"name":"Resolved Name"}',
                         supporting_quote="some fabricated text not in the document at all")
        assert "Rejected" not in r
        assert len(s.triples) == 1


class TestCliSemanticCheckDocName:
    """_cli_semantic_check must pass the *_raw.json filename as doc_name (matching
    harvest.py's raw_path.name convention), not the source .md name stored in data["doc"] —
    otherwise the harvest rejection doc-scope filter never matches and a rejected triple
    silently comes back green on CLI agent-retry."""

    def test_rejected_triple_forced_red_on_cli_retry(self, tmp_path, monkeypatch):
        import agents.extraction_agent as ea
        import agents.semantic_check_agent as sc
        import yaml as _yaml

        monkeypatch.setattr(sc, "_call_llm_batch", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no model in test")))

        hd = tmp_path / "harvest"
        hd.mkdir()
        (hd / "MAY_TREAT.jsonl").write_text(
            json.dumps({"source": "rejected", "rel_type": "MAY_TREAT",
                        "from_display": "Drug A", "to_display": "Disease B",
                        "doc_name": "doc1_raw.json"}) + "\n",
            encoding="utf-8",
        )

        schema_path = tmp_path / "schema.yaml"
        schema_path.write_text(_yaml.dump({"nodes": {}, "relationships": []}))

        raw_path = tmp_path / "doc1_raw.json"
        raw_path.write_text(json.dumps({
            "doc": "doc1.md",  # source .md name — must NOT be used as doc_name
            "doc_text": "irrelevant",
            "triples": [{
                "_id": "t1", "rel_type": "MAY_TREAT",
                "from_label": "Substance", "to_label": "Indication",
                "from_pk": "name", "to_pk": "name",
                "from_props": {"name": "Drug A"}, "to_props": {"name": "Disease B"},
                "supporting_quote": "x", "evidence": [],
            }],
        }))

        ea._cli_semantic_check(str(raw_path), str(schema_path), "moderate")

        out = json.loads(raw_path.read_text())
        t = out["triples"][0]
        assert t["triple_color"] == "red"
        assert t["constraint_violated"] is True
        assert "[Previously rejected by human reviewer]" in t["ai_opinion"]


class TestSearchDocument:
    def _session(self):
        import agents.extraction_agent as ea
        s = ea.ExtractionSession()
        s.doc_text = (
            "## ADVERSE REACTIONS\n"
            "Most common adverse reactions were rash and anaphylaxis.\n"
            "|Nasopharyngitis|8.2|12.9|5.3|\n"
        )
        return s

    def test_finds_verbatim_line(self):
        import json as _j
        s = self._session()
        out = _j.loads(s.search_document("nasopharyngitis"))
        assert out["count"] == 1
        assert out["results"][0] == "|Nasopharyngitis|8.2|12.9|5.3|"

    def test_case_insensitive_phrase(self):
        import json as _j
        s = self._session()
        out = _j.loads(s.search_document("RASH and anaphylaxis"))
        assert out["count"] == 1
        assert "rash and anaphylaxis" in out["results"][0].lower()

    def test_no_match_warns_not_invent(self):
        import json as _j
        s = self._session()
        out = _j.loads(s.search_document("cyanosis"))
        assert out["count"] == 0
        assert "do NOT invent" in out["message"]

    def test_empty_query(self):
        import json as _j
        s = self._session()
        assert _j.loads(s.search_document(""))["results"] == []


class TestHardFloors:
    """note-only deterministic checks promoted to hard red floors (item 2)."""

    def test_duplicate_stays_note_only(self):
        import agents.semantic_check_agent as sc
        t = lambda tid: {"_id": tid, "rel_type": "R", "from_pk": "name", "to_pk": "name",
                         "from_props": {"name": "A"}, "to_props": {"name": "B"}}
        out = sc.StructuralValidator().check([t("a"), t("b")], sc.ValidatorContext())
        assert out["b"].color_floor is None           # duplicate is advisory, never recolors
        assert "Duplicate" in out["b"].note

    def _umls_triple(self, actual_types):
        return {
            "_id": "u1", "rel_type": "HAS_AE",
            "from_label": "Drug", "to_label": "AdverseEffect",
            "from_pk": "cui", "to_pk": "cui",
            "from_props": {"cui": "C1", "name": "drug"},
            "to_props": {"cui": "C2", "name": "thing"},
            "to_meta": {"types": actual_types},
        }

    def _umls_ctx(self):
        import agents.semantic_check_agent as sc
        return sc.ValidatorContext(schema_nodes={
            "Drug": {},
            "AdverseEffect": {"semantic_types": ["Disease or Syndrome", "Sign or Symptom"]},
        })

    def test_type_mismatch_is_hard_red_floor(self):
        from agents.umls_check import UMLSSemanticValidator
        t = self._umls_triple(["Pharmacologic Substance"])   # zero overlap
        out = UMLSSemanticValidator().check([t], self._umls_ctx())
        assert out["u1"].color_floor == "red"
        assert out["u1"].constraint_violated is True
        assert "Type mismatch" in out["u1"].note

    def test_type_overlap_passes(self):
        from agents.umls_check import UMLSSemanticValidator
        t = self._umls_triple(["Disease or Syndrome"])       # overlaps expected
        out = UMLSSemanticValidator().check([t], self._umls_ctx())
        assert out == {}                                     # no flag

    def test_vocab_mismatch_stays_note_only(self):
        import agents.semantic_check_agent as sc
        from agents.umls_check import UMLSSemanticValidator
        t = {
            "_id": "v1", "rel_type": "HAS_AE",
            "from_label": "Drug", "to_label": "AdverseEffect",
            "from_pk": "cui", "to_pk": "cui",
            "from_props": {"cui": "C1", "name": "drug"},
            "to_props": {"cui": "C2", "name": "thing"},
            "to_meta": {"types": ["Disease or Syndrome"], "root_source": "GO"},  # overlap OK, vocab bad
        }
        ctx = sc.ValidatorContext(schema_nodes={
            "Drug": {},
            "AdverseEffect": {"semantic_types": ["Disease or Syndrome"], "umls_vocabs": ["MED-RT"]},
        })
        out = UMLSSemanticValidator().check([t], ctx)
        assert out["v1"].color_floor is None                 # vocab-only = advisory
        assert out["v1"].constraint_violated is False
        assert "Vocab mismatch" in out["v1"].note


class TestGleifNameSuspicious:
    """Regression coverage for _gleif_name_suspicious's slash-suffix tokenization.

    'A/S' (Danish/Norwegian 'Aktieselskab') is a standard legal-form suffix already
    whitelisted as the single token 'as' in _GLEIF_FORM_WORDS. The naive regex
    re.sub(r"[^\\w]", " ", ...) treated '/' as a separator, splitting 'A/S' into the
    two stray tokens 'a' and 's', neither of which matched the whitelist — a false
    positive for every company using this common suffix (Maersk A/S, Asetek A/S, ...).
    """

    def test_slash_suffix_not_suspicious(self):
        from agents.gleif_check import _gleif_name_suspicious
        assert _gleif_name_suspicious("Asetek", "ASETEK A/S") is None
        assert _gleif_name_suspicious("Maersk", "MAERSK A/S") is None

    def test_slash_suffix_equivalent_to_no_slash(self):
        from agents.gleif_check import _gleif_name_suspicious
        assert _gleif_name_suspicious("Asetek", "ASETEK AS") is None

    def test_genuine_subsidiary_still_flagged(self):
        from agents.gleif_check import _gleif_name_suspicious
        warn = _gleif_name_suspicious("Quanta", "QUANTA LYON")
        assert warn is not None
        assert "lyon" in warn.lower()

    def test_genuine_legal_form_still_unflagged(self):
        from agents.gleif_check import _gleif_name_suspicious
        assert _gleif_name_suspicious("TDK", "TDK CORPORATION") is None


class TestGLEIFResolutionValidator:
    """Class-level coverage for GLEIFResolutionValidator: the annotate_item wiring
    around _gleif_name_suspicious (function-level cases covered above), plus the
    contract that this validator is advisory-only — no deterministic check() verdict,
    the LLM makes the color call from the injected warning.
    """

    @staticmethod
    def _triple(from_term="Quanta", from_name="QUANTA LYON", from_lei="LEI1",
                to_term="NVIDIA", to_name="NVIDIA CORPORATION", to_lei="LEI2"):
        return {
            "_id": "t1", "rel_type": "PROVIDES",
            "from_label": "Corporation", "to_label": "Corporation",
            "from_pk": "lei", "to_pk": "lei",
            "from_term": from_term, "to_term": to_term,
            "from_props": {"lei": from_lei, "name": from_name},
            "to_props": {"lei": to_lei, "name": to_name},
        }

    def test_annotates_suspicious_gleif_resolution(self):
        from agents.gleif_check import GLEIFResolutionValidator
        import agents.semantic_check_agent as sc
        item = {}
        GLEIFResolutionValidator().annotate_item(self._triple(), item, sc.ValidatorContext())
        assert "from_gleif_resolution_warning" in item
        assert "lyon" in item["from_gleif_resolution_warning"].lower()
        assert "to_gleif_resolution_warning" not in item  # NVIDIA -> NVIDIA CORPORATION is clean

    def test_no_warning_when_resolution_is_clean(self):
        from agents.gleif_check import GLEIFResolutionValidator
        import agents.semantic_check_agent as sc
        t = self._triple(from_term="TDK", from_name="TDK CORPORATION")
        item = {}
        GLEIFResolutionValidator().annotate_item(t, item, sc.ValidatorContext())
        assert "from_gleif_resolution_warning" not in item

    def test_skips_non_gleif_resolved_nodes(self):
        from agents.gleif_check import GLEIFResolutionValidator
        import agents.semantic_check_agent as sc
        # Same suspicious name pair, but no 'lei' on the node -> not GLEIF-resolved,
        # so annotate_item must skip it regardless of how the name looks.
        t = self._triple()
        del t["from_props"]["lei"]
        item = {}
        GLEIFResolutionValidator().annotate_item(t, item, sc.ValidatorContext())
        assert "from_gleif_resolution_warning" not in item

    def test_skips_short_uppercase_abbreviation(self):
        from agents.gleif_check import GLEIFResolutionValidator
        import agents.semantic_check_agent as sc
        # "TSMC" is <=6 chars all-caps -> treated as an abbreviation and never
        # flagged, even though "TSMC" -> "TSMC ARIZONA" would otherwise look suspicious.
        t = self._triple(from_term="TSMC", from_name="TSMC ARIZONA")
        item = {}
        GLEIFResolutionValidator().annotate_item(t, item, sc.ValidatorContext())
        assert "from_gleif_resolution_warning" not in item

    def test_both_sides_checked_independently(self):
        from agents.gleif_check import GLEIFResolutionValidator
        import agents.semantic_check_agent as sc
        t = self._triple(to_term="Foxconn", to_name="FOXCONN SHENZHEN")
        item = {}
        GLEIFResolutionValidator().annotate_item(t, item, sc.ValidatorContext())
        assert "from_gleif_resolution_warning" in item  # Quanta -> QUANTA LYON
        assert "to_gleif_resolution_warning" in item    # Foxconn -> FOXCONN SHENZHEN

    def test_check_is_inert_no_deterministic_verdict(self):
        from agents.gleif_check import GLEIFResolutionValidator
        import agents.semantic_check_agent as sc
        # Advisory-only: check() never returns a verdict, even for an obviously
        # suspicious resolution -- the LLM, not a rule, makes the color call.
        out = GLEIFResolutionValidator().check([self._triple()], sc.ValidatorContext())
        assert out == {}

    def test_prompt_fragment_present_in_assembled_system_prompt(self):
        import agents.semantic_check_agent as sc
        assert "gleif_resolution_warning" in sc._SYSTEM_PROMPT
        assert "QUANTA LYON" in sc._SYSTEM_PROMPT


class TestUseModelEnv:
    """The use_model_env context manager: SEMANTIC_CHECK_MODEL routing that lets
    the semantic-check grader run on a different model and provider than the
    extractor, restoring the previous environment on exit (incl. on error).
    """

    def test_none_is_noop(self, monkeypatch):
        import llm_client
        monkeypatch.setenv("LLM_MODEL", "gpt-5.5")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        with llm_client.use_model_env(None) as m:
            assert m == "gpt-5.5"
            assert os.environ.get("LLM_PROVIDER") == "openai"
        assert os.environ.get("LLM_MODEL") == "gpt-5.5"

    def test_full_swap_applies_and_restores_different_provider(self, monkeypatch):
        import llm_client
        monkeypatch.setenv("LLM_MODEL", "gpt-5.5")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_API_KEY", "openai-key")
        monkeypatch.setenv("LLM_ENDPOINT", "https://azure.example/")  # should be cleared inside
        monkeypatch.setitem(llm_client._models_config, "claude-judge", {
            "LLM_PROVIDER": "anthropic",
            "LLM_API_KEY": "anthropic-key",
        })
        with llm_client.use_model_env("claude-judge"):
            assert llm_client.get_provider() == "anthropic"
            assert os.environ.get("LLM_MODEL") == "claude-judge"
            assert os.environ.get("LLM_API_KEY") == "anthropic-key"
            # azure endpoint from the outer env must not leak into the anthropic call
            assert os.environ.get("LLM_ENDPOINT") is None
        # everything restored
        assert os.environ.get("LLM_PROVIDER") == "openai"
        assert os.environ.get("LLM_MODEL") == "gpt-5.5"
        assert os.environ.get("LLM_API_KEY") == "openai-key"
        assert os.environ.get("LLM_ENDPOINT") == "https://azure.example/"

    def test_bare_name_swaps_only_model(self, monkeypatch):
        import llm_client
        monkeypatch.setenv("LLM_MODEL", "gpt-5.5")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_API_KEY", "openai-key")
        # a model name NOT in _models_config keeps the current provider/creds
        llm_client._models_config.pop("gpt-5.4-mini", None)
        with llm_client.use_model_env("gpt-5.4-mini"):
            assert os.environ.get("LLM_MODEL") == "gpt-5.4-mini"
            assert os.environ.get("LLM_PROVIDER") == "openai"
            assert os.environ.get("LLM_API_KEY") == "openai-key"
        assert os.environ.get("LLM_MODEL") == "gpt-5.5"

    def test_restores_on_exception(self, monkeypatch):
        import llm_client
        monkeypatch.setenv("LLM_MODEL", "gpt-5.5")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        try:
            with llm_client.use_model_env("gpt-5.4-mini"):
                assert os.environ.get("LLM_MODEL") == "gpt-5.4-mini"
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert os.environ.get("LLM_MODEL") == "gpt-5.5"

    def test_call_llm_batch_routes_via_kwargs_without_env_mutation(self, monkeypatch):
        """SEMANTIC_CHECK_MODEL at a different-provider entry makes _call_llm_batch
        issue the litellm call on that model + provider + key via explicit KWARGS,
        WITHOUT mutating os.environ. The non-mutation is the thread-safety fix: the
        grader runs in asyncio.to_thread alongside extraction that reads os.environ.
        """
        import sys, types
        import llm_client
        import agents.semantic_check_agent as sc

        monkeypatch.setitem(llm_client._models_config, "claude-judge", {
            "LLM_PROVIDER": "anthropic",
            "LLM_API_KEY": "anthropic-key",
        })
        monkeypatch.setenv("SEMANTIC_CHECK_MODEL", "claude-judge")
        monkeypatch.setenv("LLM_MODEL", "gpt-5.5")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_API_KEY", "openai-key")

        captured: dict = {}

        class _Msg:
            content = '{"results": []}'

        class _Choice:
            message = _Msg()

        class _Comp:
            choices = [_Choice()]

        fake = types.ModuleType("litellm")
        fake.suppress_debug_info = False

        def _completion(**kwargs):
            captured["model"] = kwargs.get("model")
            captured["api_key"] = kwargs.get("api_key")          # routed via kwargs
            # os.environ as a concurrent extraction thread would observe it:
            captured["env_provider"] = llm_client.get_provider()
            captured["env_model"] = os.environ.get("LLM_MODEL")
            captured["env_key"] = os.environ.get("LLM_API_KEY")
            return _Comp()

        fake.completion = _completion
        monkeypatch.setitem(sys.modules, "litellm", fake)

        out = sc._call_llm_batch([], "doc excerpt", "")
        assert out == {}
        # routed to the grader model + key via kwargs
        assert captured["model"] == "anthropic/claude-judge"
        assert captured["api_key"] == "anthropic-key"
        # GLOBAL env was NOT swapped during the call — a concurrent extraction
        # reading os.environ still sees the pipeline (extraction) model/provider.
        assert captured["env_provider"] == "openai"
        assert captured["env_model"] == "gpt-5.5"
        assert captured["env_key"] == "openai-key"
        # and unchanged afterwards
        assert os.environ.get("LLM_PROVIDER") == "openai"
        assert os.environ.get("LLM_MODEL") == "gpt-5.5"
        assert os.environ.get("LLM_API_KEY") == "openai-key"


class TestOpenAICompatibleProvider:
    """openai_compatible provider: OpenAI-shaped gateways (OpenCode Go, OpenRouter,
    vLLM) reached via a custom LLM_ENDPOINT base URL, without disturbing the plain
    `openai` (api.openai.com) or `azure` paths.
    """

    def test_model_string_keeps_openai_prefix(self, monkeypatch):
        import llm_client
        monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
        monkeypatch.setenv("LLM_ENDPOINT", "https://opencode.ai/zen/go/v1")
        monkeypatch.setenv("LLM_MODEL", "deepseek-v4-pro")
        assert llm_client._get_litellm_model() == "openai/deepseek-v4-pro"

    def test_kwargs_carry_api_base_and_key(self, monkeypatch):
        import llm_client
        monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
        monkeypatch.setenv("LLM_ENDPOINT", "https://opencode.ai/zen/go/v1/")  # trailing slash
        monkeypatch.setenv("LLM_API_KEY", "zen-key")
        kw = llm_client._litellm_kwargs()
        assert kw["api_base"] == "https://opencode.ai/zen/go/v1"  # stripped
        assert kw["api_key"] == "zen-key"

    def test_ensure_env_sets_openai_key(self, monkeypatch):
        import llm_client
        monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
        monkeypatch.setenv("LLM_API_KEY", "zen-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        llm_client._ensure_litellm_env()
        assert os.environ.get("OPENAI_API_KEY") == "zen-key"

    def test_plain_openai_unchanged_no_api_base(self, monkeypatch):
        """Regression: real OpenAI (no endpoint) stays a bare model with no api_base."""
        import llm_client
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.delenv("LLM_ENDPOINT", raising=False)
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "sk-real")
        assert llm_client._get_litellm_model() == "gpt-4o"
        assert "api_base" not in llm_client._litellm_kwargs()

    def test_azure_unchanged(self, monkeypatch):
        """Regression: azure provider still prefixes azure/ and is unaffected."""
        import llm_client
        monkeypatch.setenv("LLM_PROVIDER", "azure")
        monkeypatch.setenv("LLM_ENDPOINT", "https://r.cognitiveservices.azure.com/openai/deployments/gpt-5.5")
        monkeypatch.setenv("LLM_MODEL", "gpt-5.5")
        assert llm_client._get_litellm_model() == "azure/gpt-5.5"

    def test_use_model_env_full_swap_to_compatible_gateway(self, monkeypatch):
        """SEMANTIC_CHECK_MODEL pointed at an openai_compatible config entry swaps
        provider + endpoint + key for the grader call, then restores."""
        import llm_client
        monkeypatch.setenv("LLM_PROVIDER", "azure")
        monkeypatch.setenv("LLM_MODEL", "gpt-5.5")
        monkeypatch.setenv("LLM_ENDPOINT", "https://r.cognitiveservices.azure.com/openai/deployments/gpt-5.5")
        monkeypatch.setitem(llm_client._models_config, "deepseek-v4-pro", {
            "LLM_PROVIDER": "openai_compatible",
            "LLM_ENDPOINT": "https://opencode.ai/zen/go/v1",
            "LLM_API_KEY": "zen-key",
        })
        with llm_client.use_model_env("deepseek-v4-pro"):
            assert llm_client.get_provider() == "openai_compatible"
            assert llm_client._get_litellm_model() == "openai/deepseek-v4-pro"
            kw = llm_client._litellm_kwargs()
            assert kw["api_base"] == "https://opencode.ai/zen/go/v1"
            assert kw["api_key"] == "zen-key"
        # azure restored
        assert llm_client.get_provider() == "azure"
        assert llm_client._get_litellm_model() == "azure/gpt-5.5"


class TestModelCallParams:
    """model_call_params: thread-safe model routing that computes (model, kwargs)
    WITHOUT mutating os.environ — the fix for the SEMANTIC_CHECK_MODEL race where a
    grader env swap corrupted concurrent extraction calls.
    """

    def test_none_uses_current_env(self, monkeypatch):
        import llm_client
        monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
        monkeypatch.setenv("LLM_ENDPOINT", "https://opencode.ai/zen/go/v1")
        monkeypatch.setenv("LLM_MODEL", "mimo-v2.5-pro")
        monkeypatch.setenv("LLM_API_KEY", "zen-key")
        model, kw = llm_client.model_call_params(None)
        assert model == "openai/mimo-v2.5-pro"
        assert kw["api_base"] == "https://opencode.ai/zen/go/v1"
        assert kw["api_key"] == "zen-key"

    def test_config_entry_routes_without_mutating_env(self, monkeypatch):
        import llm_client
        # pipeline (extraction) env = MiMo gateway
        monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
        monkeypatch.setenv("LLM_ENDPOINT", "https://opencode.ai/zen/go/v1")
        monkeypatch.setenv("LLM_MODEL", "mimo-v2.5-pro")
        monkeypatch.setenv("LLM_API_KEY", "zen-key")
        # grader = an Azure gpt-5.5 entry
        monkeypatch.setitem(llm_client._models_config, "gpt-5.5", {
            "LLM_PROVIDER": "azure",
            "LLM_ENDPOINT": "https://r.cognitiveservices.azure.com",
            "LLM_API_KEY": "azure-key",
            "LLM_MODEL": "gpt-5.5",
        })
        model, kw = llm_client.model_call_params("gpt-5.5")
        # routed to azure gpt-5.5
        assert model == "azure/gpt-5.5"
        assert kw["api_key"] == "azure-key"
        assert kw["api_base"] == "https://r.cognitiveservices.azure.com"
        # CRITICAL: os.environ was NOT touched — extraction still sees MiMo
        assert os.environ.get("LLM_PROVIDER") == "openai_compatible"
        assert os.environ.get("LLM_MODEL") == "mimo-v2.5-pro"
        assert os.environ.get("LLM_ENDPOINT") == "https://opencode.ai/zen/go/v1"
        assert os.environ.get("LLM_API_KEY") == "zen-key"
        assert llm_client._get_litellm_model() == "openai/mimo-v2.5-pro"

    def test_full_swap_does_not_leak_stale_endpoint(self, monkeypatch):
        """A grader entry with no endpoint (anthropic) must not inherit the
        pipeline's azure LLM_ENDPOINT."""
        import llm_client
        monkeypatch.setenv("LLM_PROVIDER", "azure")
        monkeypatch.setenv("LLM_ENDPOINT", "https://r.cognitiveservices.azure.com/openai/deployments/gpt-5.5")
        monkeypatch.setenv("LLM_MODEL", "gpt-5.5")
        monkeypatch.setitem(llm_client._models_config, "judge", {
            "LLM_PROVIDER": "anthropic",
            "LLM_API_KEY": "anthropic-key",
        })
        model, kw = llm_client.model_call_params("judge")
        assert model == "anthropic/judge"
        assert kw == {"api_key": "anthropic-key"}   # no api_base leaked
        assert "api_base" not in kw

    def test_bare_name_swaps_model_keeps_provider(self, monkeypatch):
        import llm_client
        monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
        monkeypatch.setenv("LLM_ENDPOINT", "https://opencode.ai/zen/go/v1")
        monkeypatch.setenv("LLM_API_KEY", "zen-key")
        llm_client._models_config.pop("glm-5.2", None)
        model, kw = llm_client.model_call_params("glm-5.2")
        assert model == "openai/glm-5.2"
        assert kw["api_base"] == "https://opencode.ai/zen/go/v1"
        assert kw["api_key"] == "zen-key"


class TestReviewQuoteOverride:
    """Editable supporting-quote in the review UI: an OVERRIDE quote is re-anchored
    against the source .md and stored (as supporting_quote + evidence) ONLY when it
    is verbatim-in-document; otherwise it is rejected with a warning and the
    original quote is kept (preserves the grounding invariant).
    """

    import pytest

    @pytest.fixture
    def run_dir(self, monkeypatch):
        import json, shutil, uuid
        from pathlib import Path
        # /review/save spawns a background daemon thread that runs harvest_project
        # on the project dir. Stub it so that thread can't re-create the temp dir
        # after teardown's rmtree (which previously left an orphan harvest/ folder).
        import agents.harvest
        monkeypatch.setattr(agents.harvest, "harvest_project", lambda *a, **k: None)
        root = Path(__file__).resolve().parent.parent
        rd = root / "projects" / f"_qtest_{uuid.uuid4().hex[:8]}" / "runs" / "r1"
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "doc.md").write_text(
            "BEYFORTUS (nirsevimab-alip) injection, for intramuscular use "
            "Initial U.S. Approval: 2023\n\n"
            "BEYFORTUS is a monoclonal antibody with anti-RSV activity.\n"
        )
        (rd / "doc_raw.json").write_text(json.dumps({"triples": [{
            "_id": "t1", "rel_type": "HAS_TRADENAME",
            "from_label": "Substance", "to_label": "Tradename",
            "from_pk": "name", "to_pk": "name",
            "from_props": {"name": "nirsevimab"}, "to_props": {"name": "Beyfortus"},
            "rel_props": {}, "triple_color": "red",
            "supporting_quote": "BEYFORTUS is a monoclonal antibody with anti-RSV activity",
        }]}))
        yield rd
        shutil.rmtree(rd.parent.parent, ignore_errors=True)

    def _post(self, rd, new_quote):
        from fastapi.testclient import TestClient
        import htmx_app.main as m
        import review_layer as rl
        c = TestClient(m.app)
        r = c.post("/review/save", json={
            "raw_path": str(rd / "doc_raw.json"),
            "rev_path": str(rd / "doc_review.json"),
            "triples": [{
                "_id": "t1", "action": "OVERRIDE", "triple_color": "green",
                "from_props": {"name": "nirsevimab"}, "to_props": {"name": "Beyfortus"},
                "rel_props": {}, "supporting_quote": new_quote,
            }],
        })
        assert r.status_code == 200, r.text
        ev = rl.load_events(rd / "doc_review.json")["t1"]
        return r.json(), ev

    def test_quote_in_document_is_stored_with_evidence(self, run_dir):
        good = "BEYFORTUS (nirsevimab-alip) injection, for intramuscular use Initial U.S. Approval: 2023"
        resp, ev = self._post(run_dir, good)
        assert resp["quote_warnings"] == []
        # stored verbatim + evidence span located
        assert "Initial U.S. Approval: 2023" in ev["supporting_quote"]
        assert ev["evidence"][0]["start"] is not None
        assert ev["evidence"][0]["end"] > ev["evidence"][0]["start"]
        assert ev["triple_color"] == "green"

    def test_quote_not_in_document_is_warned_and_not_stored(self, run_dir):
        bogus = "This sentence is absolutely not present anywhere in the label text."
        resp, ev = self._post(run_dir, bogus)
        assert len(resp["quote_warnings"]) == 1
        assert resp["quote_warnings"][0]["id"] == "t1"
        # the override is still saved (color), but the quote was NOT overridden
        assert "supporting_quote" not in ev
        assert "evidence" not in ev
        assert ev["triple_color"] == "green"

    def test_materialize_applies_overridden_quote(self, run_dir):
        good = "BEYFORTUS (nirsevimab-alip) injection, for intramuscular use Initial U.S. Approval: 2023"
        self._post(run_dir, good)
        import json, review_layer as rl
        raw = json.loads((run_dir / "doc_raw.json").read_text())
        events = rl.load_events(run_dir / "doc_review.json")
        mat = rl.materialize(raw, events)
        t = next(t for t in mat if t["_id"] == "t1")
        assert "Initial U.S. Approval: 2023" in t["supporting_quote"]
        assert t["evidence"][0]["start"] is not None


class TestRunLlmModels:
    """run_config.json records which LLM served each task (extract._run_llm_models)."""

    def test_separate_extraction_and_check_models(self, monkeypatch):
        import extract
        monkeypatch.setenv("LLM_MODEL", "mimo-v2.5-pro")
        monkeypatch.setenv("SEMANTIC_CHECK_MODEL", "gpt-5.5")
        m = extract._run_llm_models()
        assert m == {"extraction": "mimo-v2.5-pro",
                     "node_resolution": "mimo-v2.5-pro",
                     "semantic_check": "gpt-5.5"}

    def test_check_falls_back_to_extraction_when_unset(self, monkeypatch):
        import extract
        monkeypatch.setenv("LLM_MODEL", "mimo-v2.5-pro")
        monkeypatch.delenv("SEMANTIC_CHECK_MODEL", raising=False)
        m = extract._run_llm_models()
        assert m["semantic_check"] == "mimo-v2.5-pro"

    def test_azure_prefix_stripped(self, monkeypatch):
        import extract
        monkeypatch.setenv("LLM_MODEL", "azure/gpt-5.5")
        monkeypatch.setenv("SEMANTIC_CHECK_MODEL", "")
        m = extract._run_llm_models()
        assert m["extraction"] == "gpt-5.5"
        assert m["semantic_check"] == "gpt-5.5"


class TestCacheKeyModelAware:
    """The chunk cache is model-AWARE: a different extraction LLM yields a different
    key, so switching models forces genuine re-extraction (no silent reuse of another
    model's triples while run_config claims the new model).
    """

    def test_different_model_different_key(self):
        from extract import _cache_key
        a = _cache_key("doc", "schema", model="mimo-v2.5-pro")
        b = _cache_key("doc", "schema", model="gpt-5.5")
        assert a != b

    def test_same_model_same_key(self):
        from extract import _cache_key
        assert (_cache_key("doc", "schema", model="gpt-5.5")
                == _cache_key("doc", "schema", model="gpt-5.5"))

    def test_default_model_is_stable(self):
        from extract import _cache_key
        # back-compat: default model="" still deterministic
        assert _cache_key("doc", "schema") == _cache_key("doc", "schema")

    def test_model_independent_of_other_fields(self):
        from extract import _cache_key
        # model differs but everything else identical -> different key
        base = dict(doc_text="d", schema_yaml="s", instructions="i", abbr_str="a", harvest_sig="h")
        assert _cache_key(**base, model="x") != _cache_key(**base, model="y")


class TestListProjectsFilter:
    """_list_projects hides dot/underscore-prefixed dirs so temp/test scratch
    (e.g. the _qtest_* dirs the review tests create) never appears as a project.
    """

    def test_underscore_and_dot_dirs_excluded(self):
        import uuid, shutil
        from pathlib import Path
        import htmx_app.main as m
        root = Path(m._ROOT) / "projects"
        tmp = root / ("_listtest_" + uuid.uuid4().hex[:8])
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            listed = m._list_projects()
            assert tmp.name not in listed          # underscore-prefixed → hidden
            assert "drug" in listed                # real projects still listed
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestMaterializeQuoteDerivedFromEvidence:
    """Invariant #6: supporting_quote is DERIVED from evidence. An OVERRIDE event that
    sets evidence alone (e.g. a hand-edited _review.json) must not leave a stale
    supporting_quote — materialize() derives it from the evidence spans.
    """

    def test_evidence_only_override_derives_supporting_quote(self):
        import review_layer as rl
        raw = {"triples": [{
            "_id": "t1", "rel_type": "R",
            "from_props": {"name": "A"}, "to_props": {"name": "B"},
            "supporting_quote": "STALE OLD QUOTE",
            "evidence": [{"start": 0, "end": 3, "text": "STALE OLD QUOTE"}],
        }]}
        events = {"t1": {"action": "OVERRIDE", "evidence": [
            {"start": 10, "end": 20, "text": "fresh span one"},
            {"start": 30, "end": 40, "text": "fresh span two"},
        ]}}
        out = rl.materialize(raw, events)
        assert out[0]["supporting_quote"] == "fresh span one / fresh span two"
        assert out[0]["evidence"] == events["t1"]["evidence"]

    def test_quote_and_evidence_both_present_are_left_untouched(self):
        import review_layer as rl
        raw = {"triples": [{
            "_id": "t1", "rel_type": "R",
            "from_props": {"name": "A"}, "to_props": {"name": "B"},
            "supporting_quote": "old", "evidence": [{"start": 0, "end": 1, "text": "old"}],
        }]}
        # both supplied — materialize must not re-derive (save path already paired them)
        events = {"t1": {"action": "OVERRIDE",
                         "supporting_quote": "explicit quote",
                         "evidence": [{"start": 5, "end": 9, "text": "span"}]}}
        out = rl.materialize(raw, events)
        assert out[0]["supporting_quote"] == "explicit quote"

    def test_quote_only_override_leaves_evidence_alone(self):
        # Reverse case: evidence offsets can't be reconstructed without the source
        # doc, so materialize leaves evidence as-is (re-anchoring lives in save path).
        import review_layer as rl
        raw = {"triples": [{
            "_id": "t1", "rel_type": "R",
            "from_props": {"name": "A"}, "to_props": {"name": "B"},
            "supporting_quote": "old", "evidence": [{"start": 0, "end": 3, "text": "old"}],
        }]}
        events = {"t1": {"action": "OVERRIDE", "supporting_quote": "new quote"}}
        out = rl.materialize(raw, events)
        assert out[0]["supporting_quote"] == "new quote"
        assert out[0]["evidence"] == [{"start": 0, "end": 3, "text": "old"}]


class TestCompletionStructured:
    """completion_structured: the shared SYNC structured-output retry helper used by
    the semantic-check grader. Mirrors _acreate_litellm's reliability features
    (reasoning_effort auto-recovery, rate-limit Retry-After, fatal auth) WITHOUT
    mutating os.environ (DESIGN_INVARIANTS #1 — safe on the concurrent grader path).
    """

    import pytest
    from pydantic import BaseModel as _BaseModel

    class _Res(_BaseModel):
        ok: bool = True

    def _fake_litellm(self, monkeypatch, completion_fn, exceptions=None):
        import sys, types
        fake = types.ModuleType("litellm")
        fake.suppress_debug_info = False
        fake.completion = completion_fn
        if exceptions is not None:
            fake.exceptions = exceptions
        monkeypatch.setitem(sys.modules, "litellm", fake)
        return fake

    def test_passes_model_and_call_kwargs_explicitly(self, monkeypatch):
        import llm_client
        captured = {}

        class _Msg:  content = '{"ok": true}'
        class _Choice: message = _Msg()
        class _Comp:  choices = [_Choice()]

        def _completion(**kw):
            captured.update(kw)
            return _Comp()

        self._fake_litellm(monkeypatch, _completion)
        out = llm_client.completion_structured(
            model="anthropic/x", system_prompt="sys", user_msg="u",
            response_model=self._Res, max_completion_tokens=100,
            call_kwargs={"api_key": "k", "api_base": "http://b"},
        )
        assert out.ok is True
        assert captured["model"] == "anthropic/x"
        assert captured["api_key"] == "k"
        assert captured["api_base"] == "http://b"

    def test_reasoning_effort_rejection_recovers(self, monkeypatch):
        import llm_client
        # reasoning_effort must be live for the recovery branch to trigger
        monkeypatch.setattr(llm_client, "_reasoning_effort_ok", True)
        monkeypatch.setattr(llm_client, "LLM_REASONING_EFFORT", "low")
        calls = []

        class _Msg:  content = '{"ok": true}'
        class _Choice: message = _Msg()
        class _Comp:  choices = [_Choice()]

        def _completion(**kw):
            calls.append("reasoning_effort" in kw)
            if "reasoning_effort" in kw:
                raise Exception("Unsupported parameter: reasoning_effort")
            return _Comp()

        self._fake_litellm(monkeypatch, _completion)
        out = llm_client.completion_structured(
            model="m", system_prompt="s", user_msg="u",
            response_model=self._Res, max_completion_tokens=10,
        )
        assert out.ok is True
        assert calls == [True, False]          # first WITH effort, retry WITHOUT
        assert llm_client.is_reasoning_effort_ok() is False  # auto-disabled

    def test_auth_error_is_fatal_no_retry(self, monkeypatch):
        import llm_client, types
        attempts = []

        class AuthErr(Exception): pass
        class BadReq(Exception): pass
        class RateErr(Exception): pass
        exc_ns = types.SimpleNamespace(
            AuthenticationError=AuthErr, BadRequestError=BadReq, RateLimitError=RateErr)

        def _completion(**kw):
            attempts.append(1)
            raise AuthErr("401")

        self._fake_litellm(monkeypatch, _completion, exceptions=exc_ns)
        with self.pytest.raises(AuthErr):
            llm_client.completion_structured(
                model="m", system_prompt="s", user_msg="u",
                response_model=self._Res, max_completion_tokens=10, retries=3,
            )
        assert len(attempts) == 1              # fatal — not retried 3×

    def test_rate_limit_uses_retry_after_header(self, monkeypatch):
        import llm_client, types

        class AuthErr(Exception): pass
        class BadReq(Exception): pass
        class RateErr(Exception):
            def __init__(self):
                self.response = types.SimpleNamespace(headers={"retry-after": "7"})
        exc_ns = types.SimpleNamespace(
            AuthenticationError=AuthErr, BadRequestError=BadReq, RateLimitError=RateErr)

        state = {"n": 0}
        class _Msg:  content = '{"ok": true}'
        class _Choice: message = _Msg()
        class _Comp:  choices = [_Choice()]

        def _completion(**kw):
            state["n"] += 1
            if state["n"] == 1:
                raise RateErr()
            return _Comp()

        self._fake_litellm(monkeypatch, _completion, exceptions=exc_ns)
        import time as _t
        captured_delays = []
        monkeypatch.setattr(_t, "sleep", lambda d: captured_delays.append(d))
        out = llm_client.completion_structured(
            model="m", system_prompt="s", user_msg="u",
            response_model=self._Res, max_completion_tokens=10, retries=3,
        )
        assert out.ok is True
        assert captured_delays == [7.0]        # Retry-After header honoured, not backoff
