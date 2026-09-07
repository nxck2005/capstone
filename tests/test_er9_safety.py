from __future__ import annotations

import ast
import json
from pathlib import Path

from training.er9 import ER9Trainer, _publish_json_immutable


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


def test_er9_checkpoint_json_publication_serializes_sidecar_bytes(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.sidecar.json"
    _publish_json_immutable(path, {"z": 1, "a": [2, 3]})
    assert json.loads(path.read_bytes()) == {"a": [2, 3], "z": 1}


def test_er9_checkpoint_publication_returns_the_published_sidecar_path(tmp_path: Path) -> None:
    trainer = object.__new__(ER9Trainer)
    trainer.runtime_root = tmp_path / "runtime"
    trainer.campaign_id = "test-campaign"
    trainer.run_id = "test-run"
    trainer.config_hash = "config"
    trainer.recipe_sha256 = "recipe"
    trainer.transmit_dim = 64
    trainer.quantiser_bits = 2
    trainer.global_optimizer_step = 1
    trainer.predecessor_checkpoint_id = None
    trainer._checkpoint_payload = lambda record, checkpoint_id: {  # type: ignore[attr-defined]
        "record": dict(record),
        "checkpoint_id": checkpoint_id,
    }
    result = trainer.save_checkpoint(
        {"epoch": 0, "global_optimizer_step": 1},
        {"n_correct": 1, "n_total": 1},
    )
    assert result["sidecar_path"] == "checkpoints/epoch-0000.sidecar.json"
    assert (trainer.runtime_root / result["sidecar_path"]).is_file()
