"""The G-12 model-facing test loader requires a complete, *committed* freeze manifest."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import data.test_access as test_access
from config.params import get


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture()
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "guard@test.invalid")
    _git(tmp_path, "config", "user.name", "guard")
    monkeypatch.setattr(test_access, "REPO_ROOT", tmp_path)
    return tmp_path


def _write_manifest(root: Path) -> Path:
    path = root / str(get("artifacts.freeze_manifest_file"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({field: f"bound-{field}" for field in get("evaluation.freeze_manifest_covers")}))
    return path


def test_missing_manifest_keeps_test_sealed(repo: Path) -> None:
    with pytest.raises(test_access.TestAccessError, match="sealed"):
        test_access._committed_freeze_manifest()


def test_uncommitted_manifest_is_refused(repo: Path) -> None:
    _write_manifest(repo)
    with pytest.raises(test_access.TestAccessError, match="not committed"):
        test_access._committed_freeze_manifest()


def test_committed_manifest_releases_and_later_edits_are_refused(repo: Path) -> None:
    path = _write_manifest(repo)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "freeze")
    assert set(get("evaluation.freeze_manifest_covers")) <= set(test_access._committed_freeze_manifest())
    path.write_text(path.read_text() + " ")
    with pytest.raises(test_access.TestAccessError, match="differs from its committed bytes"):
        test_access._committed_freeze_manifest()


def test_load_test_dataset_checks_the_freeze_before_any_data(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import data.adapters as adapters

    monkeypatch.setattr(adapters, "_adapter", lambda *_args: pytest.fail("adapter reached before the freeze check"))
    with pytest.raises(test_access.TestAccessError):
        test_access.load_test_dataset("imagenette160")
