"""Contract test for the *_raw.json shape (schema_raw.RawFile).

Locks the file every pipeline stage reads/writes: a writer that drifts (unknown
field, bad color, missing structural key) fails here instead of downstream.
"""
import glob
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from schema_raw import RawFile, RawTriple, validate_file  # noqa: E402

_ROOT = Path(__file__).parent.parent
_RAW_FILES = sorted(glob.glob(str(_ROOT / "projects" / "*" / "runs" / "*" / "*_raw.json")))

_BASE_TRIPLE = dict(
    _id="f1", rel_type="R", from_label="A", to_label="B",
    from_pk="name", to_pk="name", from_props={"name": "x"}, to_props={"name": "y"},
)


@pytest.mark.skipif(not _RAW_FILES, reason="no *_raw.json run files on disk")
@pytest.mark.parametrize("path", _RAW_FILES, ids=lambda p: Path(p).name)
def test_existing_raw_files_satisfy_contract(path):
    assert validate_file(path) == []


def test_minimal_triple_valid_before_coloring():
    # A freshly-extracted triple has no color/evidence yet — still valid.
    RawTriple.model_validate(_BASE_TRIPLE)


def test_underscore_aliases_round_trip():
    t = RawTriple.model_validate({**_BASE_TRIPLE, "_ai_reviewed": True})
    assert t.id_ == "f1" and t.ai_reviewed is True


def test_unknown_triple_field_rejected():
    with pytest.raises(Exception):
        RawTriple.model_validate({**_BASE_TRIPLE, "from_colour": "red"})  # typo


def test_bad_color_literal_rejected():
    with pytest.raises(Exception):
        RawTriple.model_validate({**_BASE_TRIPLE, "triple_color": "blue"})


def test_unknown_top_level_key_rejected():
    with pytest.raises(Exception):
        RawFile.model_validate(
            {"dataset_name": "d", "schema_version": "1", "triples": [], "junk": 1}
        )
