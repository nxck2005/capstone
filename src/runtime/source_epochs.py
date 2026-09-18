"""The prospective W10/PAPR downstream source epoch (AM-98).

The historical W9 v4 manifest (`results/learned/w9/downstream_source_manifest_v4.json`)
remains immutable.  This module builds and authenticates its additive successor,
which governs PAPR-constrained training and the W10 validation rehearsal, and it
provides the active-epoch closure check used by the closed-authority verifiers so
that historical v4 authorities keep verifying against their own embedded bytes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from runtime.source_guard import (
    PROTECTED_PREFIXES,
    assert_manifest_commit_bytes,
    build_manifest,
    committed_source_differences,
    git_tree_hashes,
    working_tree_source_differences,
)
from training.deterministic_core import canonical_sha256

W10_SOURCE_PATH = "results/learned/w10/w10_downstream_source_manifest.json"
W10_MANIFEST_KIND = "W10_PREPARATORY_SOURCE_SUCCESSOR_V1"
W10_MANIFEST_PREFIX = "w10downstreamsource-"
HISTORICAL_SOURCE_PATH = "results/learned/w9/downstream_source_manifest_v4.json"
HISTORICAL_SOURCE_COMMIT = "22fde3e0ba0c8ad7a92356587eb95780ded89ee6"

W10_RELEVANT_CONFIG_PATHS = (
    "configs/er9-digital-pascal-v4.yaml",
    "configs/learned-er2-randomized-pascal-v4.yaml",
    "configs/learned-papr-constrained-r1-6.yaml",
    "spec/params.generated.yaml",
)

W10_ALLOWED_EVIDENCE_RUNTIME_PREFIXES = (
    "results/learned/er9/",
    "results/learned/er2_randomized/",
    "results/learned/g11/",
    "results/learned/w9/",
    "results/learned/w10/",
    "checkpoints/er9_pascal_v4/",
    "checkpoints/er2_randomized_pascal_v4/",
    "checkpoints/papr_constrained_pascal_v4/",
    "checkpoints/smoke/",
    "checkpoints/er9_production_pascal_v4/",
)

W10_GOVERNS = (
    "papr_constrained_training",
    "w10_validation_rehearsal",
)

WORKING_TREE_GUARD = {
    "checks_committed_source_commit_to_head": True,
    "checks_unstaged_protected_source": True,
    "checks_staged_protected_source": True,
    "checks_untracked_protected_source": True,
    "runtime_and_evidence_only_after_freeze": True,
}


class SourceEpochHold(RuntimeError):
    """The active downstream source epoch is missing or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceEpochHold(message)


def build_w10_manifest(root: Path, *, source_commit: str) -> dict[str, Any]:
    """Build the successor epoch at the exact clean HEAD it binds."""

    base = build_manifest(
        root,
        source_commit=source_commit,
        relevant_config_paths=W10_RELEVANT_CONFIG_PATHS,
    )
    base.pop("manifest_id")
    base["manifest_kind"] = W10_MANIFEST_KIND
    base["historical_w9_downstream_manifest"] = {
        "path": HISTORICAL_SOURCE_PATH,
        "source_commit": HISTORICAL_SOURCE_COMMIT,
        "manifest_id": "w9downstreamsource-89bdc14e154a6a9e9d4ea3ba75fc05f5b4cff6474cb3e2e3133fd21c48d5da33",
    }
    base["allowed_evidence_runtime_prefixes"] = list(W10_ALLOWED_EVIDENCE_RUNTIME_PREFIXES)
    base["governs"] = list(W10_GOVERNS)
    base["pre_science_state"] = {
        "papr_constrained_training_runs": 0,
        "w10_authority_frozen": False,
        "w10_scientific_units": 0,
        "g12_freeze_manifest": False,
        "test": "SEALED",
        "test_access": 0,
    }
    base["manifest_id"] = W10_MANIFEST_PREFIX + hashlib.sha256(
        (json.dumps(base, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
    ).hexdigest()
    return base


def assert_w10_manifest_contract(manifest: Mapping[str, Any]) -> None:
    _require(manifest.get("schema_version") == 2, "W10 source manifest schema differs")
    _require(manifest.get("manifest_kind") == W10_MANIFEST_KIND, "W10 source manifest kind differs")
    _require(manifest.get("source_commit_comparison") == "exact_clean_HEAD_at_freeze", "W10 source freeze rule differs")
    _require(manifest.get("protected_source_prefixes") == list(PROTECTED_PREFIXES), "W10 protected source boundary differs")
    _require(manifest.get("allowed_evidence_runtime_prefixes") == list(W10_ALLOWED_EVIDENCE_RUNTIME_PREFIXES), "W10 evidence boundary differs")
    _require(manifest.get("working_tree_guard") == WORKING_TREE_GUARD, "W10 working-tree guard differs")
    _require(manifest.get("governs") == list(W10_GOVERNS), "W10 governed lifecycle differs")
    _require(manifest.get("historical_w9_downstream_manifest", {}).get("source_commit") == HISTORICAL_SOURCE_COMMIT, "W10 historical source binding differs")
    source_commit = manifest.get("source_commit")
    _require(isinstance(source_commit, str) and len(source_commit) == 40, "W10 source commit is malformed")  # literal-ok: Git SHA-1 width
    trees = manifest.get("tree_hashes")
    _require(
        isinstance(trees, Mapping)
        and set(trees) == {"repository", "src", "tools", "configs", "spec", "tests"}
        and all(isinstance(value, str) and len(value) == 40 for value in trees.values()),  # literal-ok: Git SHA-1 width
        "W10 tree closure differs",
    )
    relevant = manifest.get("relevant_config_sha256")
    _require(
        isinstance(relevant, Mapping)
        and tuple(sorted(relevant)) == tuple(sorted(W10_RELEVANT_CONFIG_PATHS)),
        "W10 relevant config closure differs",
    )
    lock = manifest.get("requirements_pascal_lock_sha256")
    _require(isinstance(lock, str) and len(lock) == 64, "W10 Pascal lock identity is malformed")  # literal-ok: SHA-256 width
    pre_science = manifest.get("pre_science_state")
    _require(
        isinstance(pre_science, Mapping)
        and pre_science.get("papr_constrained_training_runs") == 0
        and pre_science.get("w10_authority_frozen") is False
        and pre_science.get("w10_scientific_units") == 0
        and pre_science.get("g12_freeze_manifest") is False
        and pre_science.get("test") == "SEALED"
        and pre_science.get("test_access") == 0,
        "W10 pre-science state differs",
    )


def successor_path(root: Path) -> Path:
    return Path(root) / W10_SOURCE_PATH


def load_w10_manifest(root: Path, *, live: bool = True) -> dict[str, Any]:
    path = successor_path(root)
    _require(path.is_file() and not path.is_symlink(), "W10 successor source manifest is missing")
    value = json.loads(path.read_bytes())
    body = dict(value)
    identifier = body.pop("manifest_id", None)
    _require(identifier == W10_MANIFEST_PREFIX + canonical_sha256(body), "W10 source manifest ID differs")
    assert_w10_manifest_contract(value)
    if live:
        assert_clean_active_source_closure(root, value)
    return value


def source_record(root: Path, manifest: Mapping[str, Any], path: Path | None = None) -> dict[str, Any]:
    target = path or successor_path(root)
    return {
        "path": str(target.relative_to(root)),
        "manifest_id": str(manifest["manifest_id"]),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }


def assert_clean_active_source_closure(root: Path, authority: Mapping[str, Any]) -> dict[str, Any]:
    """Live worktree closure against one manifest's own commit and trees."""

    root = Path(root).resolve()
    source_commit = str(authority.get("source_commit", ""))
    _require(len(source_commit) == 40, "active source authority has no full commit")  # literal-ok: SHA-1 commit length
    committed = committed_source_differences(root, source_commit)
    working = working_tree_source_differences(root)
    _require(not committed, f"protected committed source drift: {committed}")
    _require(not any(working.values()), f"protected working-tree source drift: {working}")
    _require(dict(authority.get("tree_hashes", {})) == git_tree_hashes(root, source_commit), "active source Git tree closure differs")
    lock_path = root / "requirements-pascal.lock"
    _require(lock_path.is_file(), "Pascal lock is missing")
    _require(
        hashlib.sha256(lock_path.read_bytes()).hexdigest() == authority.get("requirements_pascal_lock_sha256"),
        "Pascal lock SHA-256 differs from the active authority",
    )
    return {"committed": committed, **working}


def assert_active_epoch_closure(root: Path, historical: Mapping[str, Any]) -> None:
    """Closed-authority closure: historical bytes plus live active closure.

    The historical v4 manifest promised that no protected source would change
    after its commit.  A successor epoch is exactly that permitted change, so
    the live check moves to the successor while the historical manifest is
    still authenticated against its own commit and never rewritten.
    """

    assert_manifest_commit_bytes(root, historical)
    if successor_path(root).is_file() and not successor_path(root).is_symlink():
        successor = load_w10_manifest(root, live=True)
        _require(
            successor.get("historical_w9_downstream_manifest", {}).get("path") == HISTORICAL_SOURCE_PATH,
            "active successor does not name the historical epoch",
        )
    else:
        from runtime.source_guard import assert_clean_source_closure  # noqa: PLC0415

        assert_clean_source_closure(root, historical)


__all__ = [
    "HISTORICAL_SOURCE_COMMIT",
    "HISTORICAL_SOURCE_PATH",
    "W10_ALLOWED_EVIDENCE_RUNTIME_PREFIXES",
    "W10_GOVERNS",
    "W10_MANIFEST_KIND",
    "W10_MANIFEST_PREFIX",
    "W10_RELEVANT_CONFIG_PATHS",
    "W10_SOURCE_PATH",
    "SourceEpochHold",
    "assert_active_epoch_closure",
    "assert_clean_active_source_closure",
    "assert_w10_manifest_contract",
    "build_w10_manifest",
    "load_w10_manifest",
    "source_record",
    "successor_path",
]
