"""The prospective W10/PAPR downstream source epochs (AM-98, AM-99).

The historical W9 v4 manifest (`results/learned/w9/downstream_source_manifest_v4.json`)
remains immutable.  Successor-v1 is the AM-98 freeze that produced zero PAPR and
zero W10 science; it is preserved as superseded-before-science evidence.
Successor-v2 (AM-99) remains immutable history.  Successor-v3 is the active
pre-science epoch over the repaired implementation.  The active-epoch closure
check lets closed-authority verifiers keep verifying against their own embedded
bytes while the successor governs new work.
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
W10_V2_SOURCE_PATH = "results/learned/w10/w10_downstream_source_manifest_v2.json"
W10_V2_MANIFEST_KIND = "W10_PREPARATORY_SOURCE_SUCCESSOR_V2"
W10_V2_MANIFEST_PREFIX = "w10downstreamsourcev2-"
W10_V3_SOURCE_PATH = "results/learned/w10/w10_downstream_source_manifest_v3.json"
W10_V3_MANIFEST_KIND = "W10_PREPARATORY_SOURCE_SUCCESSOR_V3"
W10_V3_MANIFEST_PREFIX = "w10downstreamsourcev3-"
W10_MANIFEST_KINDS = (W10_MANIFEST_KIND, W10_V2_MANIFEST_KIND, W10_V3_MANIFEST_KIND)
HISTORICAL_SOURCE_PATH = "results/learned/w9/downstream_source_manifest_v4.json"
HISTORICAL_SOURCE_COMMIT = "22fde3e0ba0c8ad7a92356587eb95780ded89ee6"

W10_RELEVANT_CONFIG_PATHS = (
    "configs/er9-digital-pascal-v4.yaml",
    "configs/learned-er2-randomized-pascal-v4.yaml",
    "configs/learned-papr-constrained-r1-6.yaml",
    "spec/params.generated.yaml",
)

# Historical successor-v1 declared this exact boundary and must keep verifying
# against its own bytes; successor-v2 adds the worker-local W10 rehearsal root.
W10_V1_ALLOWED_EVIDENCE_RUNTIME_PREFIXES = (
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

W10_ALLOWED_EVIDENCE_RUNTIME_PREFIXES = (
    *W10_V1_ALLOWED_EVIDENCE_RUNTIME_PREFIXES,
    "checkpoints/w10_rehearsal/",
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
    base["allowed_evidence_runtime_prefixes"] = list(W10_V1_ALLOWED_EVIDENCE_RUNTIME_PREFIXES)
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
    _require(manifest.get("manifest_kind") in W10_MANIFEST_KINDS, "W10 source manifest kind differs")
    _require(manifest.get("source_commit_comparison") == "exact_clean_HEAD_at_freeze", "W10 source freeze rule differs")
    _require(manifest.get("protected_source_prefixes") == list(PROTECTED_PREFIXES), "W10 protected source boundary differs")
    if manifest.get("manifest_kind") in {W10_V2_MANIFEST_KIND, W10_V3_MANIFEST_KIND}:
        _require(manifest.get("allowed_evidence_runtime_prefixes") == list(W10_ALLOWED_EVIDENCE_RUNTIME_PREFIXES), "W10 v2 evidence boundary differs")
    else:
        _require(manifest.get("allowed_evidence_runtime_prefixes") == list(W10_V1_ALLOWED_EVIDENCE_RUNTIME_PREFIXES), "W10 v1 evidence boundary differs")
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


def successor_path(root: Path, *, epoch: str = "v2") -> Path:
    """Return one explicitly named successor path; no implicit fallback."""

    names = {
        "v1": W10_SOURCE_PATH,
        "v2": W10_V2_SOURCE_PATH,
        "v3": W10_V3_SOURCE_PATH,
    }
    _require(epoch in names, f"unknown W10 successor epoch: {epoch}")
    return Path(root) / names[epoch]


def manifest_prefix(kind: str) -> str:
    if kind == W10_V3_MANIFEST_KIND:
        return W10_V3_MANIFEST_PREFIX
    if kind == W10_V2_MANIFEST_KIND:
        return W10_V2_MANIFEST_PREFIX
    return W10_MANIFEST_PREFIX


def active_manifest_path(root: Path) -> Path:
    v3 = successor_path(root, epoch="v3")
    if v3.is_file() and not v3.is_symlink():
        return v3
    v2 = successor_path(root, epoch="v2")
    if v2.is_file() and not v2.is_symlink():
        return v2
    return successor_path(root, epoch="v1")


def load_w10_manifest(root: Path, *, live: bool = True, epoch: str | None = None) -> dict[str, Any]:
    path = successor_path(root, epoch=epoch) if epoch is not None else active_manifest_path(root)
    _require(path.is_file() and not path.is_symlink(), "W10 successor source manifest is missing")
    value = json.loads(path.read_bytes())
    body = dict(value)
    identifier = body.pop("manifest_id", None)
    kind = value.get("manifest_kind")
    _require(kind in W10_MANIFEST_KINDS, "W10 successor manifest kind differs")
    _require(identifier == manifest_prefix(str(kind)) + canonical_sha256(body), "W10 source manifest ID differs")
    assert_w10_manifest_contract(value)
    if kind == W10_V2_MANIFEST_KIND:
        _require_w10_v2_supersession(root, value)
    if kind == W10_V3_MANIFEST_KIND:
        _require_w10_v3_supersession(root, value)
    if live:
        assert_clean_active_source_closure(root, value)
    return value


def _require_w10_v2_supersession(root: Path, value: Mapping[str, Any]) -> None:
    """Successor-v2 names the exact superseded-before-science v1 bytes."""

    superseded = value.get("superseded_successor")
    _require(isinstance(superseded, Mapping), "W10 v2 supersession record is missing")
    v1_path = successor_path(root, epoch="v1")
    _require(v1_path.is_file() and not v1_path.is_symlink(), "superseded W10 v1 manifest is missing")
    raw = v1_path.read_bytes()
    _require(superseded.get("path") == W10_SOURCE_PATH, "W10 v2 superseded path differs")
    _require(superseded.get("manifest_id") == json.loads(raw).get("manifest_id"), "W10 v2 superseded ID differs")
    _require(superseded.get("sha256") == hashlib.sha256(raw).hexdigest(), "W10 v2 superseded bytes differ")
    _require(
        superseded.get("superseded_before_science") is True
        and superseded.get("papr_constrained_training_runs") == 0
        and superseded.get("w10_scientific_units") == 0
        and superseded.get("test_access") == 0,
        "W10 v2 supersession is not zero-science",
    )


def _require_w10_v3_supersession(root: Path, value: Mapping[str, Any]) -> None:
    """Successor-v3 names the exact superseded-before-science v2 bytes."""

    superseded = value.get("superseded_successor")
    _require(isinstance(superseded, Mapping), "W10 v3 supersession record is missing")
    v2_path = successor_path(root, epoch="v2")
    _require(v2_path.is_file() and not v2_path.is_symlink(), "superseded W10 v2 manifest is missing")
    raw = v2_path.read_bytes()
    predecessor = json.loads(raw)
    _require(superseded.get("path") == W10_V2_SOURCE_PATH, "W10 v3 superseded path differs")
    _require(superseded.get("manifest_kind") == W10_V2_MANIFEST_KIND, "W10 v3 superseded kind differs")
    _require(superseded.get("manifest_id") == predecessor.get("manifest_id"), "W10 v3 superseded ID differs")
    _require(superseded.get("sha256") == hashlib.sha256(raw).hexdigest(), "W10 v3 superseded bytes differ")
    _require(superseded.get("source_commit") == predecessor.get("source_commit"), "W10 v3 predecessor source commit differs")
    _require(
        superseded.get("superseded_before_science") is True
        and superseded.get("papr_constrained_training_runs") == 0
        and superseded.get("w10_scientific_units") == 0
        and superseded.get("test_access") == 0,
        "W10 v3 supersession is not zero-science",
    )


def source_record(root: Path, manifest: Mapping[str, Any], path: Path | None = None) -> dict[str, Any]:
    if path is None:
        kind = str(manifest.get("manifest_kind"))
        candidates = {
            W10_MANIFEST_KIND: W10_SOURCE_PATH,
            W10_V2_MANIFEST_KIND: W10_V2_SOURCE_PATH,
            W10_V3_MANIFEST_KIND: W10_V3_SOURCE_PATH,
        }
        candidate = root / candidates[kind]
        if not candidate.is_file():
            candidate = active_manifest_path(root)
        target = candidate
    else:
        target = path
    return {
        "path": str(target.relative_to(root)),
        "manifest_id": str(manifest["manifest_id"]),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }


def build_w10_manifest_v2(root: Path, *, source_commit: str) -> dict[str, Any]:
    """Build the corrected AM-99 successor epoch at the exact clean HEAD it binds."""

    base = build_manifest(
        root,
        source_commit=source_commit,
        relevant_config_paths=W10_RELEVANT_CONFIG_PATHS,
    )
    base.pop("manifest_id")
    base["manifest_kind"] = W10_V2_MANIFEST_KIND
    base["historical_w9_downstream_manifest"] = {
        "path": HISTORICAL_SOURCE_PATH,
        "source_commit": HISTORICAL_SOURCE_COMMIT,
        "manifest_id": "w9downstreamsource-89bdc14e154a6a9e9d4ea3ba75fc05f5b4cff6474cb3e2e3133fd21c48d5da33",
    }
    v1_path = successor_path(Path(root), epoch="v1")
    if v1_path.is_file() and not v1_path.is_symlink():
        raw = v1_path.read_bytes()
        base["superseded_successor"] = {
            "path": W10_SOURCE_PATH,
            "manifest_kind": W10_MANIFEST_KIND,
            "manifest_id": json.loads(raw)["manifest_id"],
            "sha256": hashlib.sha256(raw).hexdigest(),
            "source_commit": json.loads(raw)["source_commit"],
            "superseded_before_science": True,
            "papr_constrained_training_runs": 0,
            "w10_scientific_units": 0,
            "test_access": 0,
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
    base["manifest_id"] = W10_V2_MANIFEST_PREFIX + hashlib.sha256(
        (json.dumps(base, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
    ).hexdigest()
    return base


def build_w10_manifest_v3(root: Path, *, source_commit: str) -> dict[str, Any]:
    """Build the repaired successor-v3 epoch over immutable v2 bytes."""

    base = build_manifest(
        root,
        source_commit=source_commit,
        relevant_config_paths=W10_RELEVANT_CONFIG_PATHS,
    )
    base.pop("manifest_id")
    base["manifest_kind"] = W10_V3_MANIFEST_KIND
    base["historical_w9_downstream_manifest"] = {
        "path": HISTORICAL_SOURCE_PATH,
        "source_commit": HISTORICAL_SOURCE_COMMIT,
        "manifest_id": "w9downstreamsource-89bdc14e154a6a9e9d4ea3ba75fc05f5b4cff6474cb3e2e3133fd21c48d5da33",
    }
    v2_path = successor_path(Path(root), epoch="v2")
    _require(v2_path.is_file() and not v2_path.is_symlink(), "W10 v3 requires the frozen v2 predecessor")
    raw = v2_path.read_bytes()
    predecessor = json.loads(raw)
    base["superseded_successor"] = {
        "path": W10_V2_SOURCE_PATH,
        "manifest_kind": W10_V2_MANIFEST_KIND,
        "manifest_id": predecessor["manifest_id"],
        "sha256": hashlib.sha256(raw).hexdigest(),
        "source_commit": predecessor["source_commit"],
        "superseded_before_science": True,
        "papr_constrained_training_runs": 0,
        "w10_scientific_units": 0,
        "test_access": 0,
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
    base["manifest_id"] = W10_V3_MANIFEST_PREFIX + hashlib.sha256(
        (json.dumps(base, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
    ).hexdigest()
    return base


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
    the live check moves to the active successor while the historical manifest
    is still authenticated against its own commit and never rewritten.
    """

    assert_manifest_commit_bytes(root, historical)
    active = active_manifest_path(root)
    if active.is_file() and not active.is_symlink():
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
    "W10_V1_ALLOWED_EVIDENCE_RUNTIME_PREFIXES",
    "W10_MANIFEST_KIND",
    "W10_MANIFEST_KINDS",
    "W10_MANIFEST_PREFIX",
    "W10_RELEVANT_CONFIG_PATHS",
    "W10_SOURCE_PATH",
    "W10_V2_MANIFEST_KIND",
    "W10_V2_MANIFEST_PREFIX",
    "W10_V2_SOURCE_PATH",
    "W10_V3_MANIFEST_KIND",
    "W10_V3_MANIFEST_PREFIX",
    "W10_V3_SOURCE_PATH",
    "SourceEpochHold",
    "active_manifest_path",
    "assert_active_epoch_closure",
    "assert_clean_active_source_closure",
    "assert_w10_manifest_contract",
    "build_w10_manifest",
    "build_w10_manifest_v2",
    "build_w10_manifest_v3",
    "load_w10_manifest",
    "manifest_prefix",
    "source_record",
    "successor_path",
]
