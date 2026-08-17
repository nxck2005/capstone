from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path

import pytest

from baseline.g8_pascal_merge import load_required_authority
from baseline.g8_pascal_portable import (
    PORTABLE_MANIFEST_PATH,
    PortableVerificationError,
    build_scientific_manifest,
    verify_portable_scientific_manifest,
)
from baseline.g8_pascal_production import SUCCESSOR_ROOT, unit_digest


RUNTIME = SUCCESSOR_ROOT / "runtime"


def _read_manifest() -> dict:
    return json.loads(PORTABLE_MANIFEST_PATH.read_bytes())


@pytest.fixture(scope="module")
def fresh_runtime_copy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    destination = tmp_path_factory.mktemp("pascal-runtime") / "runtime"
    shutil.copytree(RUNTIME, destination)
    return destination


def _tiny_runtime(tmp_path: Path) -> Path:
    root = tmp_path / "runtime"
    (root / "aa").mkdir(parents=True)
    (root / "campaign_state.json").write_bytes(b'{"state":1}')
    (root / "aa" / "request.json").write_bytes(b'{"request":1}')
    (root / "aa" / "result.json").write_bytes(b'{"result":1}')
    (root / "aa" / "state.json").write_bytes(b'{"state":1}')
    return root


def test_modes_and_coordination_paths_do_not_enter_portable_identity(fresh_runtime_copy: Path) -> None:
    manifest = _read_manifest()
    verify_portable_scientific_manifest(manifest, fresh_runtime_copy)
    original_digest = manifest["scientific_runtime_sha256"]

    for path in [fresh_runtime_copy, *fresh_runtime_copy.rglob("*")]:
        os.chmod(path, 0o755 if path.is_dir() else 0o644)
    verify_portable_scientific_manifest(manifest, fresh_runtime_copy)

    authority, _authority_digest, _authority_file_digest = load_required_authority()
    lock_digest = unit_digest(manifest["bindings"]["campaign_id"], authority[0]["work_unit_id"])
    locks = fresh_runtime_copy / ".locks"
    locks.mkdir()
    (locks / f"{lock_digest}.lock").write_bytes(b"")
    (fresh_runtime_copy / ".campaign.lock").write_bytes(b"")
    verified = verify_portable_scientific_manifest(manifest, fresh_runtime_copy)
    assert verified["scientific_runtime_sha256"] == original_digest


@pytest.mark.parametrize("filename", ["request.json", "result.json", "state.json", "campaign_state.json"])
def test_mutated_scientific_byte_fails(tmp_path: Path, filename: str) -> None:
    root = _tiny_runtime(tmp_path)
    manifest = build_scientific_manifest(root, validate_namespace=False)
    target = root / filename if filename == "campaign_state.json" else root / "aa" / filename
    target.write_bytes(target.read_bytes() + b"x")
    with pytest.raises(PortableVerificationError):
        verify_portable_scientific_manifest(manifest, root, validate_namespace=False)


def test_missing_and_extra_scientific_files_fail(tmp_path: Path) -> None:
    root = _tiny_runtime(tmp_path)
    manifest = build_scientific_manifest(root, validate_namespace=False)
    (root / "aa" / "result.json").unlink()
    with pytest.raises(PortableVerificationError):
        verify_portable_scientific_manifest(manifest, root, validate_namespace=False)

    root = _tiny_runtime(tmp_path / "extra")
    manifest = build_scientific_manifest(root, validate_namespace=False)
    (root / "aa" / "foreign.json").write_bytes(b"foreign")
    with pytest.raises(PortableVerificationError):
        verify_portable_scientific_manifest(manifest, root, validate_namespace=False)


def test_path_substitution_fails(tmp_path: Path) -> None:
    root = _tiny_runtime(tmp_path)
    manifest = build_scientific_manifest(root, validate_namespace=False)
    (root / "aa" / "request.json").rename(root / "aa" / "substituted.json")
    with pytest.raises(PortableVerificationError):
        verify_portable_scientific_manifest(manifest, root, validate_namespace=False)


@pytest.mark.parametrize("binding", ["campaign_id", "execution_profile_id", "production_contract_sha256", "required_authority_identity_set_sha256"])
def test_wrong_runtime_binding_fails(tmp_path: Path, binding: str) -> None:
    root = _tiny_runtime(tmp_path)
    manifest = build_scientific_manifest(root, validate_namespace=False)
    mutant = copy.deepcopy(manifest)
    mutant["bindings"][binding] = "0" * 64 if binding.endswith("sha256") else "wrong"
    with pytest.raises(PortableVerificationError):
        verify_portable_scientific_manifest(mutant, root, validate_namespace=False)


def test_wrong_trial_contract_fails(tmp_path: Path) -> None:
    root = _tiny_runtime(tmp_path)
    manifest = build_scientific_manifest(root, validate_namespace=False)
    mutant = copy.deepcopy(manifest)
    mutant["trials_per_identity"] = 4999
    with pytest.raises(PortableVerificationError):
        verify_portable_scientific_manifest(mutant, root, validate_namespace=False)
