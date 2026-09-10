"""Static guards for the non-scientific W9 CUDA lifecycle smoke."""

from __future__ import annotations

import ast
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools/run_w9_pascal_lifecycle_smoke.py"


def test_smoke_has_no_scientific_dataset_import_or_loader() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(name.startswith("data") or name.startswith("torchvision") for name in imported)


def test_smoke_marks_permanent_ineligibility_and_test_sealed() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    for marker in ("NON_SCIENTIFIC", "SYNTHETIC_ONLY", "INELIGIBLE_FOR_SELECTION", "INELIGIBLE_FOR_ER9_SEARCH", "INELIGIBLE_FOR_ER2_RESULT", "TEST_NOT_ACCESSED"):
        assert marker in text
    assert '"test": "SEALED"' in text
