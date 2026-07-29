"""Mutation coverage for the fail-closed transparency-probe verifier."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from config.params import REPO_ROOT
from probes.transparency_bitrate import load_design
from run_transparency_bitrate_probe import (
    ProbeRunError,
    _consistent_shard_lineage,
)
import verify_transparency_bitrate_probe as verifier
from verify_transparency_bitrate_probe import VerificationError, verify

SOURCE = REPO_ROOT / "results/probes/transparency_bitrate"


def _lineage_evidence_present() -> bool:
    try:
        return (
            json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))[
                "schema_version"
            ]
            == 2
        )
    except (OSError, KeyError, json.JSONDecodeError):
        return False


pytestmark = pytest.mark.skipif(
    not _lineage_evidence_present(),
    reason="lineage-bound transparency-probe evidence has not been generated",
)
_FILES = (
    "summary.json",
    "resolved_config.json",
    "aggregate.csv",
    "per_image.csv",
    "cache_manifest.json",
)


def _evidence(tmp_path: Path, *, copy: tuple[str, ...]) -> Path:
    destination = tmp_path / "evidence"
    destination.mkdir()
    copied = set(copy)
    if copied - {"summary.json"}:
        copied.add("summary.json")
    for name in _FILES:
        target = destination / name
        if name in copied:
            shutil.copyfile(SOURCE / name, target)
        else:
            target.symlink_to(SOURCE / name)
    return destination


def _mutate_json(path: Path, mutation) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutation(value)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _mutate_csv(path: Path, mutation) -> None:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    assert fields is not None
    mutation(rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _refresh_file_hash(evidence: Path, name: str) -> None:
    field = {
        "per_image.csv": "per_image_file_hash",
        "aggregate.csv": "aggregate_file_hash",
        "cache_manifest.json": "cache_manifest_hash",
        "resolved_config.json": "resolved_config_hash",
    }[name]
    digest = hashlib.sha256((evidence / name).read_bytes()).hexdigest()
    _mutate_json(
        evidence / "summary.json",
        lambda value: value.update({field: digest}),
    )


def test_committed_probe_evidence_verifies():
    result = verify()
    assert result["status"] == "COMPLETE"
    assert result["cells"] == 68_000
    assert result["implementation_commit"]
    assert result["measurement_commit"]
    assert result["evidence_commit"]
    assert result["cache_verification"]["portable_evidence_verified"] is True


@pytest.mark.parametrize("mutation", ["missing", "unexpected"])
def test_missing_or_unexpected_summary_field_fails(tmp_path: Path, mutation: str):
    evidence = _evidence(tmp_path, copy=("summary.json",))
    if mutation == "missing":
        _mutate_json(evidence / "summary.json", lambda value: value.pop("dataset"))
    else:
        _mutate_json(
            evidence / "summary.json",
            lambda value: value.update(unexpected=True),
        )
    with pytest.raises(VerificationError, match="fields differ"):
        verify(evidence_dir=evidence)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(measurement_commit="0" * 40), "measurement commit"),
        (lambda value: value.update(git_dirty_state=True), "dirty"),
        (lambda value: value.update(dataset="stl10"), "wrong dataset"),
        (lambda value: value.update(split="test"), "non-validation"),
        (
            lambda value: value["test_isolation_declaration"].update(
                test_accessed=True
            ),
            "test-access claim",
        ),
        (
            lambda value: value.update(classifier_checkpoint_identity="0" * 64),
            "checkpoint identity",
        ),
        (
            lambda value: value.update(classifier_config_identity="0" * 64),
            "config identity",
        ),
        (lambda value: value.update(dataset_identity="0" * 64), "dataset/archive"),
        (lambda value: value.update(manifest_identity="0" * 64), "manifest"),
        (
            lambda value: value.update(
                clean_validation={
                    "n_correct": 897,
                    "n_total": 1000,
                    "top1_accuracy": 0.897,
                }
            ),
            "898/1000",
        ),
        (
            lambda value: value.update(bootstrap_resamples=9999),
            "resample count",
        ),
        (
            lambda value: value.update(g8_status="selected"),
            "G-8 selection",
        ),
        (
            lambda value: value["provisional_bandwidth_parameters"].update(
                crossover_ratio="r_1_2"
            ),
            "provisional bandwidth",
        ),
    ],
)
def test_summary_identity_and_scope_mutations_fail(
    tmp_path: Path, mutation, message: str
):
    evidence = _evidence(tmp_path, copy=("summary.json",))
    _mutate_json(evidence / "summary.json", mutation)
    with pytest.raises(VerificationError, match=message):
        verify(evidence_dir=evidence)


def test_missing_validation_cell_fails(tmp_path: Path):
    evidence = _evidence(tmp_path, copy=("per_image.csv",))
    _mutate_csv(evidence / "per_image.csv", lambda rows: rows.pop())
    _refresh_file_hash(evidence, "per_image.csv")
    with pytest.raises(VerificationError, match="missing validation cells"):
        verify(evidence_dir=evidence)


def test_duplicate_budget_axis_sample_cell_fails(tmp_path: Path):
    evidence = _evidence(tmp_path, copy=("per_image.csv",))

    def mutation(rows):
        rows[-1] = dict(rows[0])

    _mutate_csv(evidence / "per_image.csv", mutation)
    _refresh_file_hash(evidence, "per_image.csv")
    with pytest.raises(VerificationError, match="duplicate budget/axis/sample"):
        verify(evidence_dir=evidence)


def test_stable_id_outside_validation_manifest_fails(tmp_path: Path):
    evidence = _evidence(tmp_path, copy=("per_image.csv",))
    _mutate_csv(
        evidence / "per_image.csv",
        lambda rows: rows[0].update(stable_sample_id="ffffffffffffffff"),
    )
    _refresh_file_hash(evidence, "per_image.csv")
    with pytest.raises(VerificationError, match="outside validation manifest"):
        verify(evidence_dir=evidence)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("emitted_bytes", "664", "exceed budget"),
        ("realized_bpp", "0.0", "realized bpp"),
        ("correct", "toggle", "correctness"),
        ("feasible", "false", "infeasible/decode status"),
        ("decode_success", "false", "feasible/decode status"),
        ("psnr", "nan", "PSNR/SSIM"),
        ("ssim", "2.0", "PSNR/SSIM"),
    ],
)
def test_per_image_numeric_and_status_mutations_fail(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
):
    evidence = _evidence(tmp_path, copy=("per_image.csv",))
    def mutation(rows):
        replacement = value
        if value == "toggle":
            replacement = "false" if rows[0][field] == "true" else "true"
        rows[0].update({field: replacement})

    _mutate_csv(evidence / "per_image.csv", mutation)
    _refresh_file_hash(evidence, "per_image.csv")
    with pytest.raises(VerificationError, match=message):
        verify(evidence_dir=evidence)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("n_correct", "0", "n_correct"),
        ("mean_psnr", "0", "mean_psnr"),
        ("mean_ssim", "0", "mean_ssim"),
    ],
)
def test_accuracy_and_metric_aggregate_mutations_fail(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
):
    evidence = _evidence(tmp_path, copy=("aggregate.csv",))
    _mutate_csv(
        evidence / "aggregate.csv",
        lambda rows: rows[0].update({field: value}),
    )
    _refresh_file_hash(evidence, "aggregate.csv")
    with pytest.raises(VerificationError, match=message):
        verify(evidence_dir=evidence)


def test_best_axis_mutation_fails(tmp_path: Path):
    evidence = _evidence(tmp_path, copy=("summary.json",))
    _mutate_json(
        evidence / "summary.json",
        lambda value: value["point_estimate_best_axes"][0].update(encode_axis=160),
    )
    with pytest.raises(VerificationError, match="best-axis"):
        verify(evidence_dir=evidence)


def test_file_hash_disagreement_fails(tmp_path: Path):
    evidence = _evidence(tmp_path, copy=("summary.json",))
    _mutate_json(
        evidence / "summary.json",
        lambda value: value.update(per_image_file_hash="0" * 64),
    )
    with pytest.raises(VerificationError, match="file hash disagreement"):
        verify(evidence_dir=evidence)


def test_bootstrap_result_mutation_fails(tmp_path: Path):
    evidence = _evidence(tmp_path, copy=("summary.json",))
    _mutate_json(
        evidence / "summary.json",
        lambda value: value["bootstrap"]["budgets"][0].update(
            one_sided_95_lower_bound=0.0
        ),
    )
    with pytest.raises(VerificationError, match="bootstrap result"):
        verify(evidence_dir=evidence)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("probe_efficiency_threshold", "5 pp"),
        ("probe_crossover_threshold", "2 pp"),
    ],
)
def test_threshold_forecast_mutation_fails(
    tmp_path: Path, field: str, message: str
):
    evidence = _evidence(tmp_path, copy=("summary.json",))
    _mutate_json(
        evidence / "summary.json",
        lambda value: value[field].update(status="left_censored"),
    )
    with pytest.raises(VerificationError, match=message):
        verify(evidence_dir=evidence)


def test_cache_manifest_missing_entry_fails(tmp_path: Path):
    evidence = _evidence(tmp_path, copy=("cache_manifest.json",))

    def mutation(value):
        value["entries"].pop()
        value["entry_count"] -= 1

    _mutate_json(evidence / "cache_manifest.json", mutation)
    _refresh_file_hash(evidence, "cache_manifest.json")
    with pytest.raises(VerificationError, match="cache manifest entry count"):
        verify(evidence_dir=evidence)


def test_cache_manifest_root_mutation_fails(tmp_path: Path):
    evidence = _evidence(tmp_path, copy=("cache_manifest.json",))
    _mutate_json(
        evidence / "cache_manifest.json",
        lambda value: value.update(entries_root_sha256="0" * 64),
    )
    _refresh_file_hash(evidence, "cache_manifest.json")
    with pytest.raises(VerificationError, match="deterministic root"):
        verify(evidence_dir=evidence)


def _mutate_resolved_lineage(
    evidence: Path,
    mutation,
) -> None:
    _mutate_json(evidence / "resolved_config.json", mutation)
    resolved = json.loads(
        (evidence / "resolved_config.json").read_text(encoding="utf-8")
    )
    _mutate_json(
        evidence / "summary.json",
        lambda value: value.update(
            execution_sources_hash=hashlib.sha256(
                verifier.canonical_json(resolved["execution_sources"])
            ).hexdigest()
        ),
    )
    _refresh_file_hash(evidence, "resolved_config.json")


def test_unreachable_implementation_commit_fails(tmp_path: Path):
    config = tmp_path / "probe.yaml"
    config.write_text(
        (
            REPO_ROOT / "configs/transparency-bitrate-probe.yaml"
        ).read_text(encoding="utf-8").replace(
            f"implementation_commit: {load_design(REPO_ROOT / 'configs/transparency-bitrate-probe.yaml')['implementation_commit']}",
            f"implementation_commit: {'0' * 40}",
        ),
        encoding="utf-8",
    )
    with pytest.raises(VerificationError, match="cat-file"):
        verify(config_path=config)


def test_unreachable_measurement_commit_fails(tmp_path: Path):
    evidence = _evidence(
        tmp_path,
        copy=("resolved_config.json", "summary.json"),
    )
    _mutate_json(
        evidence / "resolved_config.json",
        lambda value: (
            value.update(measurement_commit="0" * 40),
            value["execution_sources"].update(measurement_commit="0" * 40),
        ),
    )
    _mutate_json(
        evidence / "summary.json",
        lambda value: value.update(measurement_commit="0" * 40),
    )
    with pytest.raises(VerificationError, match="cat-file"):
        verify(evidence_dir=evidence)


def test_wrong_commit_ancestry_fails(tmp_path: Path):
    evidence = _evidence(
        tmp_path,
        copy=("resolved_config.json", "summary.json"),
    )
    unrelated_ancestor = "f3875a8e5c52ebea6ede4aa734859a727159688a"
    _mutate_json(
        evidence / "resolved_config.json",
        lambda value: (
            value.update(measurement_commit=unrelated_ancestor),
            value["execution_sources"].update(
                measurement_commit=unrelated_ancestor
            ),
        ),
    )
    _mutate_json(
        evidence / "summary.json",
        lambda value: value.update(measurement_commit=unrelated_ancestor),
    )
    with pytest.raises(VerificationError, match="wrong ancestry"):
        verify(evidence_dir=evidence)


def test_critical_source_changed_between_a_and_b_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    evidence = _evidence(tmp_path, copy=())
    resolved = json.loads(
        (SOURCE / "resolved_config.json").read_text(encoding="utf-8")
    )
    measurement_commit = resolved["measurement_commit"]
    original = verifier._git_bytes

    def changed(commit: str, relative_path: str) -> bytes:
        value = original(commit, relative_path)
        if (
            commit == measurement_commit
            and relative_path == "src/env.py"
        ):
            return value + b"\n"
        return value

    monkeypatch.setattr(verifier, "_git_bytes", changed)
    with pytest.raises(VerificationError, match="changed between A and B"):
        verify(evidence_dir=evidence)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda source: source.update(
                resolved_runtime_path="/outside/measurement/src/env.py"
            ),
            "outside the measurement checkout",
        ),
        (
            lambda source: source.update(executed_byte_sha256="0" * 64),
            "runtime-byte mismatch",
        ),
        (
            lambda source: source.update(
                implementation_git_blob_sha="0" * 40
            ),
            "git-blob mismatch",
        ),
    ],
)
def test_execution_source_identity_mutations_fail(
    tmp_path: Path,
    mutation,
    message: str,
):
    evidence = _evidence(tmp_path, copy=("resolved_config.json",))

    def change(value):
        mutation(
            value["execution_sources"]["critical_files"]["src/env.py"]
        )

    _mutate_resolved_lineage(evidence, change)
    with pytest.raises(VerificationError, match=message):
        verify(evidence_dir=evidence)


@pytest.mark.parametrize("kind", ["missing", "unexpected"])
def test_missing_or_unexpected_execution_source_fails(
    tmp_path: Path,
    kind: str,
):
    evidence = _evidence(tmp_path, copy=("resolved_config.json",))

    def change(value):
        files = value["execution_sources"]["critical_files"]
        if kind == "missing":
            files.pop("src/env.py")
        else:
            files["src/unexpected.py"] = dict(files["src/env.py"])

    _mutate_resolved_lineage(evidence, change)
    with pytest.raises(VerificationError, match="critical source entries differ"):
        verify(evidence_dir=evidence)


def test_shards_from_differing_commits_fail():
    resolved = json.loads(
        (SOURCE / "resolved_config.json").read_text(encoding="utf-8")
    )
    first = {
        "measurement_commit": resolved["measurement_commit"],
        "execution_sources": resolved["execution_sources"],
    }
    second = json.loads(json.dumps(first))
    second["measurement_commit"] = (
        "f3875a8e5c52ebea6ede4aa734859a727159688a"
    )
    with pytest.raises(ProbeRunError, match="measurement commit"):
        _consistent_shard_lineage([first, second])


def test_tracked_cache_or_codestream_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
):
    design = load_design(
        REPO_ROOT / "configs/transparency-bitrate-probe.yaml",
        repo_root=REPO_ROOT,
    )
    monkeypatch.setattr(
        verifier.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout=f"{design['cache_root']}/committed.j2kcache\n",
            stderr="",
        ),
    )
    with pytest.raises(VerificationError, match="cache or codestream"):
        verifier._verify_no_tracked_cache_or_codestream(design)
