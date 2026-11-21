"""Frozen v1.1 artifacts must not change after the locked test run."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import freeze_results as fr  # noqa: E402

MANIFEST = ROOT / fr.MANIFEST_REL


@pytest.mark.skipif(not MANIFEST.exists(), reason="no freeze manifest yet")
def test_consistency_checks_passed_at_freeze():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert all(m["consistency_checks"].values()), m["consistency_checks"]


@pytest.mark.skipif(not MANIFEST.exists(), reason="no freeze manifest yet")
def test_frozen_files_unchanged():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for rel, meta in m["files"].items():
        p = ROOT / rel
        if not p.exists():  # data/ and runs/ may be absent on a fresh clone
            pytest.skip(f"{rel} not present on this machine")
        assert fr.sha256_file(p) == meta["sha256"], f"frozen artifact changed: {rel}"


def test_sha256_file_roundtrip(tmp_path):
    import hashlib
    p = tmp_path / "x.bin"
    p.write_bytes(b"gaplink" * 300000)  # > 1 chunk
    assert fr.sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()
