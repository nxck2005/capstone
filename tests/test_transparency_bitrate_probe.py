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


@pytest.mark.external_dataset
def test_committed_probe_evidence_verifies():
    result = verify()
    assert result["status"] == "COMPLETE"
    assert result["cells"] == 68_000
    assert result["implementation_commit"]
    assert result["measurement_commit"]
    assert result["evidence_commit"]
    assert result["cache_verification"]["portable_evidence_verified"] is True


@pytest.mark.parametrize("mutation", ["missing", "unexpected"])
@pytest.mark.external_dataset
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
@pytest.mark.external_dataset
def test_summary_identity_and_scope_mutations_fail(
    tmp_path: Path, mutation, message: str
):
    evidence = _evidence(tmp_path, copy=("summary.json",))
    _mutate_json(evidence / "summary.json", mutation)
    with pytest.raises(VerificationError, match=message):
        verify(evidence_dir=evidence)


@pytest.mark.external_dataset
def test_missing_validation_cell_fails(tmp_path: Path):
    evidence = _evidence(tmp_path, copy=("per_image.csv",))
    _mutate_csv(evidence / "per_image.csv", lambda rows: rows.pop())
    _refresh_file_hash(evidence, "per_image.csv")
    with pytest.raises(VerificationError, match="missing validation cells"):
        verify(evidence_dir=evidence)


@pytest.mark.external_dataset
def test_duplicate_budget_axis_sample_cell_fails(tmp_path: Path):
    evidence = _evidence(tmp_path, copy=("per_image.csv",))

    def mutation(rows):
        rows[-1] = dict(rows[0])

    _mutate_csv(evidence / "per_image.csv", mutation)
    _refresh_file_hash(evidence, "per_image.csv")
    with pytest.raises(VerificationError, match="duplicate budget/axis/sample"):
        verify(evidence_dir=evidence)


@pytest.mark.external_dataset
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
@pytest.mark.external_dataset
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
@pytest.mark.external_dataset
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


@pytest.mark.external_dataset
def test_best_axis_mutation_fails(tmp_path: Path):
    evidence = _evidence(tmp_path, copy=("summary.json",))
    _mutate_json(
        evidence / "summary.json",
        lambda value: value["point_estimate_best_axes"][0].update(encode_axis=160),
    )
    with pytest.raises(VerificationError, match="best-axis"):
        verify(evidence_dir=evidence)


@pytest.mark.external_dataset
def test_file_hash_disagreement_fails(tmp_path: Path):
    evidence = _evidence(tmp_path, copy=("summary.json",))
    _mutate_json(
        evidence / "summary.json",
        lambda value: value.update(per_image_file_hash="0" * 64),
    )
    with pytest.raises(VerificationError, match="file hash disagreement"):
        verify(evidence_dir=evidence)


@pytest.mark.external_dataset
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
@pytest.mark.external_dataset
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


@pytest.mark.external_dataset
def test_cache_manifest_missing_entry_fails(tmp_path: Path):
    evidence = _evidence(tmp_path, copy=("cache_manifest.json",))

    def mutation(value):
        value["entries"].pop()
        value["entry_count"] -= 1

    _mutate_json(evidence / "cache_manifest.json", mutation)
    _refresh_file_hash(evidence, "cache_manifest.json")
    with pytest.raises(VerificationError, match="cache manifest entry count"):
        verify(evidence_dir=evidence)


@pytest.mark.external_dataset
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


@pytest.mark.external_dataset
def test_unreachable_implementation_commit_fails(tmp_path: Path):
    config = tmp_path / "probe.yaml"
    config.write_text(
        (
            REPO_ROOT / "configs/transparency-bitrate-probe.yaml"
        ).read_text(encoding="utf-8").replace(
            f"implementation_commit: {load_design(REPO_ROOT / 'configs/transparency-bitrate-probe.yaml')['implementation_commit']}",
            f"implementation_commit: {'f' * 40}",
        ),
        encoding="utf-8",
    )
    with pytest.raises(VerificationError, match="cat-file"):
        verify(config_path=config)


@pytest.mark.external_dataset
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


@pytest.mark.external_dataset
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


@pytest.mark.external_dataset
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
@pytest.mark.external_dataset
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
@pytest.mark.external_dataset
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


@pytest.mark.external_dataset
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


@pytest.mark.external_dataset
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


# ---------------------------------------------------------------------------
# AM-82 — the codec configuration is bound as history, and the single pinned
# readjudication is the only thing permitting it to differ from HEAD.
#
# These exercise `_verify_codec_configuration_binding` directly against a fake
# codec, so each mutation fails for its own independent property rather than
# because some unrelated aggregate hash moved.
# ---------------------------------------------------------------------------


class _FakeCodec:
    """Just enough of `J2KCodec` to stand in for HEAD's codec configuration."""

    def __init__(self, snapshot: dict) -> None:
        self.snapshot = snapshot
        self.configuration_hash = hashlib.sha256(
            verifier.canonical_json(snapshot)
        ).hexdigest()


def _archived_summary() -> dict:
    summary = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
    return {
        "codec_configuration": summary["codec_configuration"],
        "codec_configuration_hash": summary["codec_configuration_hash"],
        "dataset": summary["dataset"],
        "encode_axis_order": summary["encode_axis_order"],
    }


def _current_snapshot() -> dict:
    """The archived snapshot with exactly the AM-80 change applied."""

    snapshot = json.loads(
        json.dumps(_archived_summary()["codec_configuration"])
    )
    snapshot["baseline"]["downsample_axis_px"]["cifar10"] = [32]
    return snapshot


def _record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation=None) -> None:
    value = json.loads(
        (SOURCE / "codec_configuration_readjudication.json").read_text(
            encoding="utf-8"
        )
    )
    if mutation is not None:
        mutation(value)
    path = tmp_path / "readjudication.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(verifier, "READJUDICATION", path)


@pytest.mark.external_dataset
def test_codec_binding_accepts_the_declared_am80_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _record(tmp_path, monkeypatch)
    verifier._verify_codec_configuration_binding(
        _archived_summary(), _FakeCodec(_current_snapshot())
    )


@pytest.mark.external_dataset
def test_codec_binding_accepts_no_drift_without_a_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(verifier, "READJUDICATION", tmp_path / "absent.json")
    summary = _archived_summary()
    verifier._verify_codec_configuration_binding(
        summary, _FakeCodec(summary["codec_configuration"])
    )


@pytest.mark.external_dataset
def test_codec_binding_rejects_a_stale_record_when_nothing_drifted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A record left behind must not sit ready to cover the next change."""

    _record(tmp_path, monkeypatch)
    summary = _archived_summary()
    with pytest.raises(VerificationError, match="nothing has drifted"):
        verifier._verify_codec_configuration_binding(
            summary, _FakeCodec(summary["codec_configuration"])
        )


@pytest.mark.external_dataset
def test_codec_binding_rejects_undeclared_drift_without_a_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(verifier, "READJUDICATION", tmp_path / "absent.json")
    with pytest.raises(VerificationError, match="no readjudication is recorded"):
        verifier._verify_codec_configuration_binding(
            _archived_summary(), _FakeCodec(_current_snapshot())
        )


@pytest.mark.external_dataset
def test_codec_binding_rejects_an_archive_that_fails_its_own_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _record(tmp_path, monkeypatch)
    summary = _archived_summary()
    summary["codec_configuration_hash"] = "0" * 64
    with pytest.raises(VerificationError, match="reproduce its own recorded hash"):
        verifier._verify_codec_configuration_binding(
            summary, _FakeCodec(_current_snapshot())
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("baseline", "j2k_resolutions"), 5),
        (("baseline", "j2k_wavelet"), "reversible_5_3"),
        (("baseline", "j2k_progression_order"), "LRCP"),
        (("baseline", "j2k_code_block_size"), [32, 32]),
        (("baseline", "j2k_tile_size"), "fixed_512"),
        (("baseline", "j2k_rate_control"), "fixed_compression_ratio"),
        (("baseline", "j2k_rate_control_method"), "single_shot"),
        (("baseline", "j2k_search_tolerance_bytes"), 32),
        (("baseline", "j2k_cache_key"), ["canonical_pixels_sha256"]),
        (("baseline", "j2k_impl_version"), "2.5.3"),
        (("preprocessing", "codec_downsample_interpolation"), "nearest"),
        (("environment", "openjpeg"), "2.5.3"),
    ],
)
@pytest.mark.external_dataset
def test_codec_binding_rejects_any_additional_codec_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: tuple[str, ...], value
):
    """The AM-82 record covers one parameter. Anything else must still fail."""

    _record(tmp_path, monkeypatch)
    snapshot = _current_snapshot()
    snapshot[path[0]][path[1]] = value
    with pytest.raises(VerificationError, match="but the archived and current"):
        verifier._verify_codec_configuration_binding(
            _archived_summary(), _FakeCodec(snapshot)
        )


@pytest.mark.external_dataset
def test_codec_binding_rejects_a_change_to_the_probes_own_axis_ladder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The reachability argument dies the moment Imagenette's ladder moves."""

    snapshot = _current_snapshot()
    snapshot["baseline"]["downsample_axis_px"]["imagenette160"] = [160, 128, 96]
    _record(
        tmp_path,
        monkeypatch,
        lambda value: value.update(
            changed_parameter_paths=[
                "baseline.downsample_axis_px.cifar10",
                "baseline.downsample_axis_px.imagenette160",
            ],
            old_values={
                "baseline.downsample_axis_px.cifar10": [32, 24, 16],
                "baseline.downsample_axis_px.imagenette160": [160, 128, 96, 64],
            },
            new_values={
                "baseline.downsample_axis_px.cifar10": [32],
                "baseline.downsample_axis_px.imagenette160": [160, 128, 96],
            },
        ),
    )
    with pytest.raises(VerificationError, match="not identical"):
        verifier._verify_codec_configuration_binding(
            _archived_summary(), _FakeCodec(snapshot)
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value.update(superseded_codec_configuration_hash="0" * 64),
            "supersedes a codec hash",
        ),
        (
            lambda value: value.update(current_codec_configuration_hash="0" * 64),
            "does not cover the current",
        ),
        (
            lambda value: value.update(
                old_values={"baseline.downsample_axis_px.cifar10": [32]}
            ),
            "wrong superseded value",
        ),
        (
            lambda value: value.update(
                new_values={"baseline.downsample_axis_px.cifar10": [32, 24, 16]}
            ),
            "wrong current value",
        ),
        (
            lambda value: value.update(probe_datasets=["cifar10"]),
            "datasets other than the one",
        ),
        (
            lambda value: value.update(unchanged={"campaign_rerun": True}),
            "claims the campaign was rerun",
        ),
        (
            lambda value: value.update(justification="because"),
            "no substantive justification",
        ),
        (lambda value: value.update(scope_limits=[]), "no scope limits"),
        (
            lambda value: value.update(reachability_argument={"selector": "nothing"}),
            "no code-backed reachability argument",
        ),
        (
            lambda value: value.update(
                reachability_argument={
                    "selector": "configured_axes",
                    "probe_dataset_ladder_unchanged": False,
                }
            ),
            "no code-backed reachability argument",
        ),
        (lambda value: value.update(amendment="none"), "names no amendment"),
        (
            lambda value: value.update(kind="something_else"),
            "not a recognised codec-configuration record",
        ),
    ],
)
@pytest.mark.external_dataset
def test_codec_binding_rejects_a_defective_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation, message: str
):
    _record(tmp_path, monkeypatch, mutation)
    with pytest.raises(VerificationError, match=message):
        verifier._verify_codec_configuration_binding(
            _archived_summary(), _FakeCodec(_current_snapshot())
        )
