from __future__ import annotations

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_er9_and_randomized_er2_production_modules_have_no_test_boundary_import() -> None:
    paths = (
        REPO / "src/evaluation/er9_protocol.py",
        REPO / "src/evaluation/er9_search.py",
        REPO / "src/evaluation/er9_transport.py",
        REPO / "src/evaluation/er9_campaign.py",
        REPO / "src/models/er9_digital.py",
        REPO / "src/training/er9.py",
        REPO / "src/training/er2_randomized.py",
    )
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        assert all("data.test_access" not in name for name in imported)
        assert all("test" not in name.split(".")[-1] for name in imported if name)


def test_er9_source_does_not_name_a_model_facing_test_split() -> None:
    paths = (
        REPO / "src/evaluation/er9_campaign.py",
        REPO / "src/training/er9.py",
        REPO / "src/training/er2_randomized.py",
    )
    for path in paths:
        text = path.read_text()
        assert "load_dataset(\"test\"" not in text
        assert "ValidationDJSCCDataset(\"test\"" not in text
