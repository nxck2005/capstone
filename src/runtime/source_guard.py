"""Full-tree and live-working-tree source closure for W9 v4 launches."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


class SourceGuardHold(RuntimeError):
    """The frozen scientific source closure is not the live source."""


PROTECTED_PREFIXES = (
    "src/",
    "tools/",
    "configs/",
    "spec/",
    "tests/",
    "requirements-pascal.lock",
)
ALLOWED_EVIDENCE_PREFIXES = (
    "results/learned/er9/",
    "results/learned/er2_randomized/",
    "results/learned/g11/",
    "results/learned/w9/",
    "checkpoints/er9_pascal_v4/",
    "checkpoints/er2_randomized_pascal_v4/",
    "checkpoints/smoke/",
)
V4_RELEVANT_CONFIG_PATHS = (
    "configs/er9-digital-pascal-v4.yaml",
    "configs/learned-er2-randomized-pascal-v4.yaml",
    "spec/params.generated.yaml",
)
DOWNSTREAM_RELEVANT_CONFIG_PATHS = V4_RELEVANT_CONFIG_PATHS
DOWNSTREAM_MANIFEST_KIND = "W9_V4_FINAL_DOWNSTREAM_SOURCE_SUCCESSOR"
DOWNSTREAM_ALLOWED_EVIDENCE_PREFIXES = (
    *ALLOWED_EVIDENCE_PREFIXES,
    "results/learned/w10/",
    "checkpoints/er9_production_pascal_v4/",
)
V4_WORKING_TREE_GUARD = {
    "checks_committed_source_commit_to_head": True,
    "checks_unstaged_protected_source": True,
    "checks_staged_protected_source": True,
    "checks_untracked_protected_source": True,
    "runtime_and_evidence_only_after_freeze": True,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceGuardHold(message)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    _require(result.returncode == 0, f"git command failed: git {' '.join(args)}")
    return result.stdout.strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)
    _require(result.returncode == 0, f"git byte query failed: git {' '.join(args)}")
    return result.stdout


def _git_names(root: Path, *args: str) -> list[str]:
    result = subprocess.run(["git", *args, "-z"], cwd=root, capture_output=True, check=False)
    _require(result.returncode == 0, f"git path query failed: git {' '.join(args)}")
    return [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def _protected(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix) for prefix in PROTECTED_PREFIXES)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_v4_manifest_contract(manifest: Mapping[str, Any]) -> None:
    """Authenticate the structural meaning of a W9 v4 source manifest.

    The manifest digest authenticates bytes, but it does not by itself say
    that those bytes describe the intended closure.  Keep this contract next
    to the source guard so producers, active executables and verifiers share
    the same protected/allowed boundary and exact config set.
    """

    _require(manifest.get("schema_version") == 2, "W9 v4 source manifest schema differs")
    _require(
        manifest.get("manifest_kind") == "W9_V4_FULL_SCIENTIFIC_SOURCE_CLOSURE",
        "W9 v4 source manifest kind differs",
    )
    source_commit = manifest.get("source_commit")
    _require(
        isinstance(source_commit, str) and len(source_commit) == 40,  # literal-ok: full Git SHA-1 length
        "W9 v4 source manifest commit is not a full SHA-1",
    )
    _require(
        manifest.get("source_commit_comparison") == "exact_clean_HEAD_at_freeze",
        "W9 v4 source manifest freeze rule differs",
    )
    _require(
        manifest.get("protected_source_prefixes") == list(PROTECTED_PREFIXES),
        "W9 v4 protected source boundary differs",
    )
    _require(
        manifest.get("allowed_evidence_runtime_prefixes") == list(ALLOWED_EVIDENCE_PREFIXES),
        "W9 v4 evidence/runtime boundary differs",
    )
    _require(
        manifest.get("working_tree_guard") == V4_WORKING_TREE_GUARD,
        "W9 v4 working-tree guard differs",
    )
    tree_hashes = manifest.get("tree_hashes")
    _require(
        isinstance(tree_hashes, Mapping)
        and set(tree_hashes) == {"repository", "src", "tools", "configs", "spec", "tests"}
        and all(isinstance(value, str) and len(value) == 40 for value in tree_hashes.values()),  # literal-ok: Git tree SHA-1 length
        "W9 v4 source tree hashes are malformed",
    )
    relevant = manifest.get("relevant_config_sha256")
    _require(
        isinstance(relevant, Mapping)
        and tuple(sorted(relevant)) == V4_RELEVANT_CONFIG_PATHS
        and all(isinstance(value, str) and len(value) == 64 for value in relevant.values()),  # literal-ok: SHA-256 length
        "W9 v4 relevant config closure differs",
    )
    lock_sha = manifest.get("requirements_pascal_lock_sha256")
    _require(
        isinstance(lock_sha, str) and len(lock_sha) == 64,  # literal-ok: SHA-256 length
        "W9 v4 Pascal lock identity is malformed",
    )


def assert_downstream_manifest_contract(manifest: Mapping[str, Any]) -> None:
    """Authenticate the final post-search execution-source successor."""

    _require(manifest.get("schema_version") == 2, "downstream source manifest schema differs")
    _require(manifest.get("manifest_kind") == DOWNSTREAM_MANIFEST_KIND, "downstream source manifest kind differs")
    _require(manifest.get("source_commit_comparison") == "exact_clean_HEAD_at_freeze", "downstream source freeze rule differs")
    _require(manifest.get("historical_stage1_source_commit") == "a66592f7e53f8bffa1c5747e2ae9af71673ae67e", "historical Stage-1 source binding differs")
    _require(manifest.get("protected_source_prefixes") == list(PROTECTED_PREFIXES), "downstream protected source boundary differs")
    _require(manifest.get("allowed_evidence_runtime_prefixes") == list(DOWNSTREAM_ALLOWED_EVIDENCE_PREFIXES), "downstream output boundary differs")
    _require(manifest.get("working_tree_guard") == V4_WORKING_TREE_GUARD, "downstream working-tree guard differs")
    source_commit = manifest.get("source_commit")
    _require(isinstance(source_commit, str) and len(source_commit) == 40, "downstream source commit is malformed")  # literal-ok: Git SHA-1 width
    trees = manifest.get("tree_hashes")
    _require(
        isinstance(trees, Mapping)
        and set(trees) == {"repository", "src", "tools", "configs", "spec", "tests"}
        and all(isinstance(value, str) and len(value) == 40 for value in trees.values()),  # literal-ok: Git tree SHA-1 width
        "downstream tree closure differs",
    )
    relevant = manifest.get("relevant_config_sha256")
    _require(
        isinstance(relevant, Mapping)
        and tuple(sorted(relevant)) == DOWNSTREAM_RELEVANT_CONFIG_PATHS
        and all(isinstance(value, str) and len(value) == 64 for value in relevant.values()),  # literal-ok: SHA-256 width
        "downstream config closure differs",
    )
    _require(isinstance(manifest.get("requirements_pascal_lock_sha256"), str) and len(str(manifest["requirements_pascal_lock_sha256"])) == 64, "downstream Pascal lock identity differs")  # literal-ok: SHA-256 width
    _require(manifest.get("governs") == [
        "fresh_production_er9",
        "randomized_er2",
        "final_er9_validation",
        "g11_h4",
        "w10_validation_rehearsal",
    ], "downstream governed lifecycle differs")
    _require(manifest.get("test") == "SEALED" and manifest.get("test_access") == 0, "downstream source crossed test boundary")


def assert_manifest_commit_bytes(root: Path, manifest: Mapping[str, Any]) -> None:
    """Authenticate a historical manifest against its commit, not live HEAD."""

    source_commit = str(manifest.get("source_commit", ""))
    _require(len(source_commit) == 40, "historical source commit is malformed")  # literal-ok: Git SHA-1 width
    _require(git_tree_hashes(root, source_commit) == dict(manifest.get("tree_hashes", {})), "historical Git tree closure differs")
    configs = manifest.get("relevant_config_sha256", {})
    _require(isinstance(configs, Mapping), "historical config closure is malformed")
    for relative, digest in configs.items():
        _require(git_blob_sha256(root, source_commit, str(relative)) == digest, f"historical config differs: {relative}")
    _require(git_blob_sha256(root, source_commit, "requirements-pascal.lock") == manifest.get("requirements_pascal_lock_sha256"), "historical Pascal lock differs")


def git_blob_bytes(root: Path, commit: str, relative: str) -> bytes:
    """Read one exact tracked file from a Git commit, never from the worktree."""

    return _git_bytes(root, "show", f"{commit}:{relative}")


def git_blob_sha256(root: Path, commit: str, relative: str) -> str:
    return hashlib.sha256(git_blob_bytes(root, commit, relative)).hexdigest()


def git_tree_hashes(root: Path, commit: str) -> dict[str, str]:
    values = {"repository": _git(root, "rev-parse", f"{commit}^{{tree}}")}
    for name in ("src", "tools", "configs", "spec", "tests"):
        values[name] = _git(root, "rev-parse", f"{commit}:{name}")
    return values


def committed_source_differences(root: Path, source_commit: str, head: str | None = None) -> list[str]:
    head = head or _git(root, "rev-parse", "HEAD")
    if source_commit == head:
        return []
    return sorted(path for path in _git_names(root, "diff", "--name-only", f"{source_commit}..{head}") if _protected(path))


def working_tree_source_differences(root: Path) -> dict[str, list[str]]:
    return {
        "unstaged": sorted(path for path in _git_names(root, "diff", "--name-only") if _protected(path)),
        "staged": sorted(path for path in _git_names(root, "diff", "--cached", "--name-only") if _protected(path)),
        "untracked": sorted(path for path in _git_names(root, "ls-files", "--others", "--exclude-standard") if _protected(path)),
    }


def assert_clean_source_closure(root: Path, authority: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(root).resolve()
    source_commit = str(authority.get("source_commit", ""))
    _require(len(source_commit) == 40, "source authority does not bind a full source commit")  # literal-ok: SHA-1 commit length
    committed = committed_source_differences(root, source_commit)
    working = working_tree_source_differences(root)
    _require(not committed, f"protected committed source drift: {committed}")
    _require(not any(working.values()), f"protected working-tree source drift: {working}")
    expected = authority.get("tree_hashes", {})
    _require(isinstance(expected, Mapping), "source authority tree hashes are missing")
    actual = git_tree_hashes(root, source_commit)
    _require(dict(expected) == actual, "source authority Git tree closure differs")
    expected_configs = authority.get("relevant_config_sha256", {})
    if expected_configs:
        _require(isinstance(expected_configs, Mapping), "source authority config hashes are malformed")
        actual_configs = {
            str(relative): git_blob_sha256(root, source_commit, str(relative))
            for relative in sorted(expected_configs)
        }
        _require(dict(expected_configs) == actual_configs, "source authority config closure differs")
    lock_path = root / "requirements-pascal.lock"
    _require(lock_path.is_file(), "Pascal lock is missing")
    _require(
        git_blob_sha256(root, source_commit, "requirements-pascal.lock")
        == authority.get("requirements_pascal_lock_sha256"),
        "Pascal lock SHA-256 differs from the source commit",
    )
    _require(
        _sha256_file(lock_path) == authority.get("requirements_pascal_lock_sha256"),
        "Pascal lock SHA-256 differs from the live worktree",
    )
    return {"committed": committed, **working, "tree_hashes": actual}


def _tracked_paths_at_commit(root: Path, commit: str, prefix: str) -> list[str]:
    raw = _git(root, "ls-tree", "-r", "--name-only", commit, "--", prefix)
    return [line for line in raw.splitlines() if line]


def build_manifest(root: Path, *, source_commit: str, relevant_config_paths: Iterable[str]) -> dict[str, Any]:
    root = Path(root).resolve()
    _require(len(source_commit) == 40, "source manifest requires a full source commit")  # literal-ok: SHA-1 commit length
    head = _git(root, "rev-parse", "HEAD")
    _require(source_commit == head, "source manifest source_commit must equal HEAD at freeze time")
    working = working_tree_source_differences(root)
    _require(not any(working.values()), f"source manifest freeze requires a clean protected worktree: {working}")
    config_hashes: dict[str, str] = {}
    for relative in sorted(set(relevant_config_paths)):
        _require(
            _git(root, "cat-file", "-e", f"{source_commit}:{relative}") == "",
            f"relevant config is missing from source commit: {relative}",
        )
        config_hashes[relative] = git_blob_sha256(root, source_commit, relative)
    lock_path = root / "requirements-pascal.lock"
    _require(lock_path.is_file() and not lock_path.is_symlink(), "Pascal lock is missing")
    manifest = {
        "schema_version": 2,
        "manifest_kind": "W9_V4_FULL_SCIENTIFIC_SOURCE_CLOSURE",
        "source_commit": source_commit,
        "source_commit_comparison": "exact_clean_HEAD_at_freeze",
        "tree_hashes": git_tree_hashes(root, source_commit),
        "requirements_pascal_lock_sha256": git_blob_sha256(root, source_commit, "requirements-pascal.lock"),
        "relevant_config_sha256": config_hashes,
        "protected_source_prefixes": list(PROTECTED_PREFIXES),
        "allowed_evidence_runtime_prefixes": list(ALLOWED_EVIDENCE_PREFIXES),
        "working_tree_guard": {
            "checks_committed_source_commit_to_head": True,
            "checks_unstaged_protected_source": True,
            "checks_staged_protected_source": True,
            "checks_untracked_protected_source": True,
            "runtime_and_evidence_only_after_freeze": True,
        },
    }
    manifest["manifest_id"] = "er9sourcev4-" + hashlib.sha256(
        (json.dumps({key: value for key, value in manifest.items()}, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
    ).hexdigest()
    return manifest


def build_downstream_manifest(root: Path, *, source_commit: str) -> dict[str, Any]:
    """Build the one final source successor without rewriting Stage-1."""

    base = build_manifest(
        root,
        source_commit=source_commit,
        relevant_config_paths=DOWNSTREAM_RELEVANT_CONFIG_PATHS,
    )
    base.pop("manifest_id")
    base["manifest_kind"] = DOWNSTREAM_MANIFEST_KIND
    base["allowed_evidence_runtime_prefixes"] = list(DOWNSTREAM_ALLOWED_EVIDENCE_PREFIXES)
    base["historical_stage1_source_commit"] = "a66592f7e53f8bffa1c5747e2ae9af71673ae67e"
    base["governs"] = [
        "fresh_production_er9",
        "randomized_er2",
        "final_er9_validation",
        "g11_h4",
        "w10_validation_rehearsal",
    ]
    base["test"] = "SEALED"
    base["test_access"] = 0
    base["manifest_id"] = "w9downstreamsource-" + hashlib.sha256(
        (json.dumps(base, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
    ).hexdigest()
    return base


__all__ = [
    "ALLOWED_EVIDENCE_PREFIXES",
    "V4_RELEVANT_CONFIG_PATHS",
    "V4_WORKING_TREE_GUARD",
    "PROTECTED_PREFIXES",
    "SourceGuardHold",
    "assert_v4_manifest_contract",
    "assert_downstream_manifest_contract",
    "assert_manifest_commit_bytes",
    "assert_clean_source_closure",
    "build_manifest",
    "build_downstream_manifest",
    "committed_source_differences",
    "git_tree_hashes",
    "git_blob_bytes",
    "git_blob_sha256",
    "working_tree_source_differences",
]
