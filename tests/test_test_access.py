"""Structural and runtime tests for the SR-22 test-access boundary."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import data.test_access as test_access
from config.params import REPO_ROOT, get
from data.test_access import TestAccessError as AccessDenied
from data.test_access import load_test_sample


def _write_manifest(root: Path, body: dict[str, object]) -> Path:
    path = root / get("artifacts.freeze_manifest_file")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def _complete_manifest() -> dict[str, object]:
    return {
        field: f"frozen-{field}"
        for field in get("evaluation.freeze_manifest_covers")
    }


def test_loading_without_freeze_manifest_raises_before_loader_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []
    monkeypatch.setattr(test_access, "REPO_ROOT", tmp_path)

    with pytest.raises(AccessDenied, match="test access remains sealed"):
        load_test_sample(
            lambda sample_id: calls.append(sample_id),
            "sample-one",
        )

    assert calls == []


def test_incomplete_freeze_manifest_raises_before_loader_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []
    manifest = _complete_manifest()
    del manifest[get("evaluation.freeze_manifest_covers")[0]]
    _write_manifest(tmp_path, manifest)
    monkeypatch.setattr(test_access, "REPO_ROOT", tmp_path)

    with pytest.raises(AccessDenied, match="does not cover required fields"):
        load_test_sample(
            lambda sample_id: calls.append(sample_id),
            "sample-one",
        )

    assert calls == []


def test_complete_freeze_manifest_releases_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _write_manifest(tmp_path, _complete_manifest())
    monkeypatch.setattr(test_access, "REPO_ROOT", tmp_path)

    sample = load_test_sample(
        lambda sample_id: {"stable_sample_id": sample_id},
        "sample-one",
    )

    assert sample == {"stable_sample_id": "sample-one"}


def test_no_other_source_module_imports_test_access():
    guarded_module = (REPO_ROOT / "src/data/test_access.py").resolve()
    violations: list[str] = []
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        if path.resolve() == guarded_module:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports_guard = any(
                    alias.name == "data.test_access"
                    or alias.name.startswith("data.test_access.")
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports_guard = (
                    module == "data.test_access"
                    or module.startswith("data.test_access.")
                    or (
                        module in {"data", ""}
                        and any(alias.name == "test_access" for alias in node.names)
                    )
                    or (
                        module == "test_access"
                        and path.parent.resolve() == guarded_module.parent
                    )
                )
            else:
                continue
            if imports_guard:
                violations.append(str(path.relative_to(REPO_ROOT)))

    assert violations == []
