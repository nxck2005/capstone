"""Dedicated clean-reference-classifier configuration contract tests."""

from __future__ import annotations

from config.params import REPO_ROOT, get
from config.run_config import RunConfig, config_hash, load_reference_classifier_config


CONFIG = REPO_ROOT / get("config.dir") / "reference-classifier-clean.yaml"


def test_classifier_config_hash_is_mapping_order_invariant_and_sensitive():
    cfg = load_reference_classifier_config(CONFIG, dataset="imagenette160")
    body = cfg.to_dict()
    reordered = {key: body[key] for key in reversed(tuple(body))}
    reordered["resolved"] = {
        key: body["resolved"][key] for key in reversed(tuple(body["resolved"]))
    }
    changed = cfg.to_dict()
    changed["resolved"]["architecture"] = "resnet34"

    assert config_hash(RunConfig.from_dict(reordered)) == config_hash(cfg)
    assert config_hash(RunConfig.from_dict(changed)) != config_hash(cfg)


def test_classifier_config_resolved_identity_is_complete_and_channel_free():
    cfg = load_reference_classifier_config(CONFIG, dataset="stl10")

    assert set(cfg.resolved) == {
        "project_id",
        "task",
        "dataset",
        "dataset_version",
        "split_manifest_hash",
        "classifier_variant",
        "train_seed",
        "architecture",
        "analysis_version",
    }
