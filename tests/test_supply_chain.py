"""Supply-chain guard for dependency pins.

litellm 1.107.0 on PyPI (2026-10-21) shipped a credential stealer + backdoor
(cook.pth / cook.py / sitecustomize.py). requirements.txt used `litellm>=1.30`,
which would have installed it. These tests keep litellm exactly pinned, keep
known-compromised versions out of every requirements file and the hash lock,
and keep the lock hash-verified.
"""
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
_REQ_FILES = [
    _ROOT / "requirements.txt",
    _ROOT / "requirements-dev.txt",
    _ROOT / "htmx_app" / "requirements_extra.txt",
]
_LOCK = _ROOT / "requirements.lock"

# Known-malicious releases. Add new advisories here.
_COMPROMISED = {
    "litellm": {"1.107.0"},
}


def _pins(path: Path) -> dict[str, str]:
    """name -> specifier string for every requirement line in *path*."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip().rstrip("\\").strip()
        if not line or line.startswith("-"):
            continue
        m = re.match(r"([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?\s*(.*)", line)
        if m:
            out[m.group(1).lower().replace("_", "-")] = m.group(2).strip()
    return out


def test_litellm_is_exactly_pinned():
    spec = _pins(_ROOT / "requirements.txt")["litellm"]
    assert re.fullmatch(r"==\s*[0-9][0-9A-Za-z.]*", spec), (
        f"litellm must be an exact `==` pin, got {spec!r}"
    )


@pytest.mark.parametrize("path", _REQ_FILES + [_LOCK], ids=lambda p: p.name)
def test_no_compromised_versions(path):
    pins = _pins(path)
    for name, bad in _COMPROMISED.items():
        spec = pins.get(name, "")
        m = re.fullmatch(r"==\s*(\S+)", spec)
        assert not (m and m.group(1) in bad), f"{path.name}: {name}{spec} is compromised"


def test_lock_matches_requirements_and_is_hashed():
    lock = _pins(_LOCK)
    assert lock.get("litellm") == _pins(_ROOT / "requirements.txt")["litellm"], (
        "requirements.lock is stale — re-run the `uv pip compile` command in its header"
    )
    text = _LOCK.read_text()
    n_pkgs = len(re.findall(r"^\S+==\S+ \\$", text, flags=re.M))
    assert n_pkgs and n_pkgs == len(re.findall(r"^\S+==\S+ \\\n\s+--hash=sha256:", text, flags=re.M)), (
        "every locked package must carry --hash entries"
    )
