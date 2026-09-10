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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceGuardHold(message)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    _require(result.returncode == 0, f"git command failed: git {' '.join(args)}")
    return result.stdout.strip()


def _git_names(root: Path, *args: str) -> list[str]:
    result = subprocess.run(["git", *args, "-z"], cwd=root, capture_output=True, check=False)
    _require(result.returncode == 0, f"git path query failed: git {' '.join(args)}")
    return [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def _protected(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix) for prefix in PROTECTED_PREFIXES)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    lock_path = root / "requirements-pascal.lock"
    _require(lock_path.is_file(), "Pascal lock is missing")
    _require(_sha256_file(lock_path) == authority.get("requirements_pascal_lock_sha256"), "Pascal lock SHA-256 differs")
    return {"committed": committed, **working, "tree_hashes": actual}


def _tracked_paths_at_commit(root: Path, commit: str, prefix: str) -> list[str]:
    raw = _git(root, "ls-tree", "-r", "--name-only", commit, "--", prefix)
    return [line for line in raw.splitlines() if line]


def build_manifest(root: Path, *, source_commit: str, relevant_config_paths: Iterable[str]) -> dict[str, Any]:
    root = Path(root).resolve()
    _require(len(source_commit) == 40, "source manifest requires a full source commit")  # literal-ok: SHA-1 commit length
    config_hashes: dict[str, str] = {}
    for relative in sorted(set(relevant_config_paths)):
        path = root / relative
        _require(path.is_file() and not path.is_symlink(), f"relevant config is missing: {relative}")
        config_hashes[relative] = _sha256_file(path)
    lock_path = root / "requirements-pascal.lock"
    _require(lock_path.is_file(), "Pascal lock is missing")
    manifest = {
        "schema_version": 2,
        "manifest_kind": "W9_V4_FULL_SCIENTIFIC_SOURCE_CLOSURE",
        "source_commit": source_commit,
        "tree_hashes": git_tree_hashes(root, source_commit),
        "requirements_pascal_lock_sha256": _sha256_file(lock_path),
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


__all__ = [
    "ALLOWED_EVIDENCE_PREFIXES",
    "PROTECTED_PREFIXES",
    "SourceGuardHold",
    "assert_clean_source_closure",
    "build_manifest",
    "committed_source_differences",
    "git_tree_hashes",
    "working_tree_source_differences",
]
