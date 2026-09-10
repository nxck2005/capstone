"""Prelaunch verifier must remain visible and model/data-free."""

from __future__ import annotations

import ast
from pathlib import Path

from runtime.w9_authority import resolve_runtime_root


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools/verify_er9_pascal_v4_ready.py"


def test_prelaunch_verifier_does_not_import_model_or_dataset_runtime() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(name.startswith(("torch", "models", "data")) for name in imported)


def test_authority_root_is_explicit_and_repository_contained(tmp_path: Path) -> None:
    authority = {"runtime_root": "checkpoints/er9_pascal_v4"}
    assert resolve_runtime_root(tmp_path, authority) == (tmp_path / "checkpoints/er9_pascal_v4").resolve()
