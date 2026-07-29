"""Reference-classifier CLI output routing tests without a dataset run."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from config.params import REPO_ROOT, get


def _cli_module():
    path = REPO_ROOT / "tools/train_reference_classifier.py"
    spec = importlib.util.spec_from_file_location("train_reference_classifier_tool", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_production_artifact_paths_are_configured():
    module = _cli_module()
    paths = module._paths_for_run(smoke=False, output_root=None)

    assert paths["resolved_config"] == REPO_ROOT / get("artifacts.classifier_resolved_config_file")
    assert paths["checkpoint_dir"] == REPO_ROOT / get("artifacts.classifier_checkpoint_dir")


def test_smoke_output_is_confined_to_ignored_smoke_root(tmp_path: Path):
    module = _cli_module()
    root = REPO_ROOT / "checkpoints/smoke/reference-classifier/test-output"
    paths = module._paths_for_run(smoke=True, output_root=root)

    assert paths["artifact_dir"] == root.resolve()
    with pytest.raises(ValueError, match="smoke output root"):
        module._paths_for_run(smoke=True, output_root=tmp_path)
    with pytest.raises(ValueError, match="smoke-only"):
        module._paths_for_run(smoke=False, output_root=root)


def test_epoch_log_keeps_small_summary_not_full_permutation(tmp_path: Path):
    module = _cli_module()
    path = tmp_path / "epochs.jsonl"
    module._write_epoch_log(
        path,
        [{"epoch": 0, "lr": 0.1, "loss": 1.0, "steps": 1, "sample_order": [3, 2, 1]}],
        [{"epoch": 0, "n_correct": 1, "n_total": 2, "top1_accuracy": 0.5}],
    )

    record = json.loads(path.read_text())
    assert record["training"] == {"epoch": 0, "loss": 1.0, "lr": 0.1, "steps": 1}
    assert record["validation"]["n_total"] == 2


def test_cli_applies_deterministic_backend_before_trainer_construction(monkeypatch):
    module = _cli_module()
    events: list[str] = []

    monkeypatch.setattr(module, "set_deterministic_backend", lambda: events.append("backend"))
    monkeypatch.setattr(
        module,
        "load_reference_classifier_config",
        lambda *_args, **_kwargs: events.append("config") or object(),
    )

    class ConstructionProbe:
        def __init__(self, *_args, **_kwargs):
            assert events == ["backend", "config"]
            raise RuntimeError("constructed after deterministic backend")

    monkeypatch.setattr(module, "ReferenceClassifierTrainer", ConstructionProbe)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_reference_classifier.py", "--config", "x.yaml", "--dataset", "cifar10"],
    )

    with pytest.raises(RuntimeError, match="constructed after deterministic backend"):
        module.main()


def test_cli_rejects_incompatible_full_smoke_arguments_before_construction(monkeypatch):
    module = _cli_module()
    constructed = False
    monkeypatch.setattr(module, "set_deterministic_backend", lambda: pytest.fail("backend applied"))
    monkeypatch.setattr(
        module,
        "load_reference_classifier_config",
        lambda *_args, **_kwargs: pytest.fail("config loaded"),
    )

    class ConstructionProbe:
        def __init__(self, *_args, **_kwargs):
            nonlocal constructed
            constructed = True

    monkeypatch.setattr(module, "ReferenceClassifierTrainer", ConstructionProbe)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_reference_classifier.py", "--config", "x.yaml", "--dataset", "cifar10",
            "--full-run", "--smoke-steps", "1",
        ],
    )

    with pytest.raises(ValueError, match="incompatible"):
        module.main()

    assert not constructed


def test_full_cli_resume_requests_full_lineage(monkeypatch, tmp_path: Path):
    module = _cli_module()
    monkeypatch.setattr(module, "set_deterministic_backend", lambda: None)
    monkeypatch.setattr(module, "load_reference_classifier_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        module,
        "_paths_for_run",
        lambda **_kwargs: {
            "artifact_dir": tmp_path,
            "resolved_config": tmp_path / "resolved.json",
            "epochs": tmp_path / "epochs.jsonl",
            "validation_summary": tmp_path / "validation.json",
            "best_checkpoint": tmp_path / "best.json",
            "checkpoint_dir": tmp_path / "checkpoints",
        },
    )

    class SmokeCheckpointProbe:
        def __init__(self, *_args, **_kwargs):
            self.state = object()

        def resume(self, _checkpoint, *, execution_mode):
            assert execution_mode == "full"
            raise ValueError("cannot resume smoke checkpoint in full mode")

    monkeypatch.setattr(module, "ReferenceClassifierTrainer", SmokeCheckpointProbe)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_reference_classifier.py", "--config", "x.yaml", "--dataset", "cifar10",
            "--full-run", "--resume", str(tmp_path / "smoke.pt"),
        ],
    )

    with pytest.raises(ValueError, match="cannot resume smoke checkpoint in full mode"):
        module.main()


def test_full_cli_defers_production_artifact_creation_until_official_run(monkeypatch, tmp_path: Path):
    module = _cli_module()
    artifact_dir = tmp_path / "production"
    monkeypatch.setattr(module, "set_deterministic_backend", lambda: None)
    monkeypatch.setattr(module, "load_reference_classifier_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        module,
        "_paths_for_run",
        lambda **_kwargs: {
            "artifact_dir": artifact_dir,
            "resolved_config": artifact_dir / "resolved.json",
            "epochs": artifact_dir / "epochs.jsonl",
            "validation_summary": artifact_dir / "validation.json",
            "best_checkpoint": artifact_dir / "best.json",
            "checkpoint_dir": artifact_dir / "checkpoints",
        },
    )

    class FullRunProbe:
        def __init__(self, *_args, **_kwargs):
            pass

        def run_epochs(self, **kwargs):
            assert kwargs["execution_mode"] == "full"
            assert kwargs["full_run_requested"] is True
            assert not artifact_dir.exists()
            raise RuntimeError("official full run reached before artifacts")

    monkeypatch.setattr(module, "ReferenceClassifierTrainer", FullRunProbe)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_reference_classifier.py", "--config", "x.yaml", "--dataset", "cifar10", "--full-run"],
    )

    with pytest.raises(RuntimeError, match="official full run reached before artifacts"):
        module.main()
