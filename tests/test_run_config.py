"""Run-configuration resolution and hashing tests (SR-1, SR-13)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

import config.run_config as run_config
from config.params import REPO_ROOT, get
from config.run_config import (
    RunConfig,
    config_hash,
    load_experiment,
    load_reference_classifier_config,
)


LEARNED = REPO_ROOT / get("config.dir") / "learned-headline.yaml"
CLASSICAL = REPO_ROOT / get("config.dir") / "classical-headline.yaml"
REFERENCE_CLASSIFIER = REPO_ROOT / get("config.dir") / "reference-classifier-clean.yaml"


def _learned_config() -> RunConfig:
    return load_experiment(
        LEARNED,
        train_seed=get("evaluation.train_seeds")[0],
        channel_seed=get("evaluation.channel_seeds")[0],
        test_snr_db=get("channel.test_snr_grid_db")[0],
    )


def _load_mutated_learned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    **choice_updates: object,
) -> RunConfig:
    document = yaml.safe_load(LEARNED.read_text(encoding="utf-8"))
    document["choices"].update(choice_updates)
    path = tmp_path / "mutated-learned.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(run_config, "_experiment_path", lambda _: path)
    return run_config.load_experiment(
        path.name,
        train_seed=get("evaluation.train_seeds")[0],
        channel_seed=get("evaluation.channel_seeds")[0],
        test_snr_db=get("channel.test_snr_grid_db")[0],
    )


def test_round_trip_is_exact_and_frozen():
    cfg = _learned_config()
    restored = RunConfig.from_dict(cfg.to_dict())

    assert restored == cfg
    with pytest.raises(FrozenInstanceError):
        restored.experiment = "mutated"  # type: ignore[misc]
    with pytest.raises(TypeError):
        restored.resolved["dataset"] = "stl10"  # type: ignore[index]
    with pytest.raises(TypeError):
        restored.parameters["channel"] = {}  # type: ignore[index]


def test_symbolic_choices_resolve_and_are_retained():
    cfg = _learned_config()

    assert cfg.choices["bw_ratio"] == "crossover_ratio"
    assert cfg.resolved["bw_ratio"] == get("bandwidth.crossover_ratio")
    assert cfg.choices["train_snr_db"] == "train_snr_db_fixed"
    assert cfg.resolved["train_snr_db"] == get("channel.train_snr_db_fixed")
    assert cfg.choices["lambda"] == "lambda_core"
    assert cfg.resolved["lambda"] == get("learned_system.lambda_core")
    assert cfg.resolved["k"] == get(
        f"bandwidth.k_symbols.imagenette160.{get('bandwidth.crossover_ratio')}"
    )
    assert cfg.resolved["dataset_version"] == get(
        "datasets.imagenette160.archive_sha256"
    )
    assert cfg.resolved["analysis_version"] == get("config.analysis_version")


def test_unresolvable_train_snr_db_symbol_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    value = "train_snr_db_fixedd"

    with pytest.raises(ValueError) as caught:
        _load_mutated_learned(
            tmp_path,
            monkeypatch,
            train_snr_db=value,
        )

    message = str(caught.value)
    assert "train_snr_db" in message
    assert value in message
    assert f"params.channel.{value}" in message


def test_unresolvable_lambda_symbol_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    value = "lambda_coree"

    with pytest.raises(ValueError) as caught:
        _load_mutated_learned(
            tmp_path,
            monkeypatch,
            **{"lambda": value},
        )

    message = str(caught.value)
    assert "lambda" in message
    assert value in message
    assert f"params.learned_system.{value}" in message


def test_numeric_train_snr_db_passes_through(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = _load_mutated_learned(
        tmp_path,
        monkeypatch,
        train_snr_db=7,
    )

    assert cfg.choices["train_snr_db"] == 7
    assert cfg.resolved["train_snr_db"] == 7


def test_all_committed_experiment_configs_still_load():
    loaded = {
        path.name: load_experiment(
            path,
            **(
                {
                    "train_seed": get("evaluation.train_seeds")[0],
                    "channel_seed": get("evaluation.channel_seeds")[0],
                    "test_snr_db": get("channel.test_snr_grid_db")[0],
                }
                if path == LEARNED
                else {
                    "channel_seed": get("evaluation.channel_seeds")[0],
                    "test_snr_db": get("channel.test_snr_grid_db")[0],
                }
            ),
        )
        for path in sorted(LEARNED.parent.glob("*-headline.yaml"))
    }

    assert set(loaded) == {LEARNED.name, CLASSICAL.name}


def test_reference_classifier_config_resolves_without_channel_fields():
    cfg = load_reference_classifier_config(REFERENCE_CLASSIFIER, dataset="imagenette160")

    assert cfg.resolved["train_seed"] == get("reference_classifier.clean_train_seed")
    assert cfg.resolved["classifier_variant"] == "clean"
    assert cfg.resolved["split_manifest_hash"] == get("datasets.imagenette160.manifest_sha256")
    assert cfg.resolved["architecture"] == get("reference_classifier.arch")
    assert not {"channel", "bw_ratio", "k", "test_snr_db"} & set(cfg.resolved)
    assert RunConfig.from_dict(cfg.to_dict()) == cfg


def test_reference_classifier_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="unknown classifier dataset"):
        load_reference_classifier_config(REFERENCE_CLASSIFIER, dataset="unknown")


@pytest.mark.parametrize(
    "dataset",
    tuple(
        name
        for name, value in get("datasets").items()
        if isinstance(value, dict) and "loader" in value
    ),
)
def test_reference_classifier_config_resolves_every_configured_dataset(dataset):
    cfg = load_reference_classifier_config(REFERENCE_CLASSIFIER, dataset=dataset)

    assert cfg.resolved["dataset"] == dataset
    assert cfg.resolved["dataset_version"] == get(
        f"datasets.{dataset}.archive_sha256"
    )
    assert cfg.resolved["split_manifest_hash"] == get(
        f"datasets.{dataset}.manifest_sha256"
    )
    assert set(cfg.parameters) == set(get("config.fingerprint_parameter_roots"))
    with pytest.raises(TypeError):
        cfg.parameters["reference_classifier"]["arch"] = "resnet50"  # type: ignore[index]


def test_reference_classifier_rejects_overrides():
    with pytest.raises(ValueError, match="accepts no overrides"):
        load_reference_classifier_config(
            REFERENCE_CLASSIFIER,
            dataset="imagenette160",
            channel_seed=0,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda body: body.update({"unexpected": True}), "keys differ"),
        (lambda body: body["choices"].update({"channel": "awgn"}), "choices"),
        (
            lambda body: body["sweep_axes"].update({"train_seed": "train_seeds"}),
            "sweep_axes",
        ),
    ),
)
def test_reference_classifier_rejects_malformed_choice_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: object,
    message: str,
):
    body = yaml.safe_load(REFERENCE_CLASSIFIER.read_text(encoding="utf-8"))
    mutation(body)
    path = tmp_path / "malformed-reference-classifier.yaml"
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(run_config, "_experiment_path", lambda _: path)

    with pytest.raises(ValueError, match=message):
        run_config.load_reference_classifier_config(path.name, dataset="imagenette160")


def test_hash_is_deterministic_under_serialised_key_reordering():
    cfg = _learned_config()
    body = cfg.to_dict()
    reordered = {key: body[key] for key in reversed(tuple(body))}
    reordered["resolved"] = {
        key: body["resolved"][key] for key in reversed(tuple(body["resolved"]))
    }
    reordered["parameters"] = {
        key: body["parameters"][key]
        for key in reversed(tuple(body["parameters"]))
    }

    assert config_hash(RunConfig.from_dict(reordered)) == config_hash(cfg)
    assert len(config_hash(cfg)) == 64


def test_hash_changes_when_a_resolved_setting_changes():
    cfg = _learned_config()
    body = cfg.to_dict()
    body["resolved"]["train_seed"] = get("evaluation.train_seeds")[1]
    changed = RunConfig.from_dict(body)

    assert config_hash(changed) != config_hash(cfg)


def test_hash_changes_when_fingerprint_schema_version_changes():
    cfg = _learned_config()
    body = cfg.to_dict()
    body["fingerprint_schema_version"] += 1

    assert config_hash(RunConfig.from_dict(body)) != config_hash(cfg)


@pytest.mark.parametrize(
    "root",
    (
        "project",
        "datasets",
        "preprocessing",
        "bandwidth",
        "channel",
        "learned_system",
        "baseline",
        "reference_classifier",
        "digital_semantic_control",
        "evaluation",
        "compute",
        "artifacts",
        "environment",
    ),
)
def test_every_scientific_runtime_parameter_root_affects_hash(root):
    cfg = _learned_config()
    body = cfg.to_dict()
    body["parameters"][root]["_fingerprint_test_mutation"] = root

    assert config_hash(RunConfig.from_dict(body)) != config_hash(cfg)


def test_fingerprint_snapshot_excludes_administrative_roots():
    cfg = _learned_config()

    assert set(cfg.parameters) == {
        "project",
        "datasets",
        "preprocessing",
        "bandwidth",
        "channel",
        "learned_system",
        "baseline",
        "reference_classifier",
        "digital_semantic_control",
        "evaluation",
        "compute",
        "artifacts",
        "environment",
    }
    assert not {
        "config",
        "demo",
        "hardware_tier23",
        "deliverables",
    } & set(cfg.parameters)
    assert cfg.resolved["analysis_version"] == get("config.analysis_version")
    assert cfg.resolved["dataset_version"] == get(
        "datasets.imagenette160.archive_sha256"
    )


def test_every_sweep_axis_requires_one_valid_override():
    with pytest.raises(ValueError, match="missing=.*test_snr_db"):
        load_experiment(
            LEARNED,
            train_seed=get("evaluation.train_seeds")[0],
            channel_seed=get("evaluation.channel_seeds")[0],
        )
    with pytest.raises(ValueError, match="outside params.channel.test_snr_grid_db"):
        load_experiment(
            CLASSICAL,
            channel_seed=get("evaluation.channel_seeds")[0],
            test_snr_db=10_000,
        )


def test_classical_choice_file_resolves_without_learned_only_fields():
    cfg = load_experiment(
        CLASSICAL,
        channel_seed=get("evaluation.channel_seeds")[0],
        test_snr_db=get("channel.test_snr_grid_db")[0],
    )

    assert cfg.resolved["system"] == "classical_adaptive"
    assert "lambda" not in cfg.resolved
    assert "train_seed" not in cfg.resolved
