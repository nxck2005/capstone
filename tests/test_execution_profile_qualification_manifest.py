from __future__ import annotations

import copy
import hashlib

import pytest

from config.execution_profiles import canonical_json_bytes
from verify_execution_profile_qualification_manifest import QualificationManifestError, verify


def test_manifest_mutation_status_fails(tmp_path, monkeypatch):
    import verify_execution_profile_qualification_manifest as module
    from pathlib import Path

    path = tmp_path / "manifest.json"
    source = Path("results/execution_profiles/qualification/manifest.json")
    path.write_bytes(source.read_bytes())
    verify(path)
    mutated = __import__("json").loads(path.read_bytes())
    mutated["g8_coverage"] = 1
    mutated["manifest_sha256"] = hashlib.sha256(canonical_json_bytes({k: v for k, v in mutated.items() if k != "manifest_sha256"})).hexdigest()
    path.write_bytes(canonical_json_bytes(mutated))
    with pytest.raises(QualificationManifestError):
        verify(path)
