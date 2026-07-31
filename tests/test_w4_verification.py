"""Mutation coverage for the W4 bounded-evidence verifier.

The fixture builds a complete, valid evidence directory from scratch rather than
depending on the committed bounded run, so these tests prove the verifier's
*discrimination* — that each defect class is caught — independently of whether
the real evidence happens to exist or verify at the time they run.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

import run_classical_baseline_w4_smoke as runner
import verify_w4_baseline_integration as verifier
from baseline.classical.outage import EVIDENCE_LABELS, OutagePolicyError
from baseline.classical.pipeline import (
    CODEC_INFEASIBILITY,
    DECODE_FAILURE,
    STRUCTURAL_INFEASIBILITY,
)
from baseline.classical.records import aggregate_schema, per_image_schema
from config.params import REPO_ROOT, get
from artifacts.ids import (
    make_analysis_cell_id,
    make_noise_id,
    make_pair_id,
    make_run_id,
)
from config.run_config import canonical_sha256, config_hash, load_experiment
from data.registry import manifest_sha256
from models.frozen_reference_classifier import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CONFIG_HASH,
)

DATASET = "imagenette160"
#: A small, stable set of real tracked files, so the source-binding checks run
#: against genuine Git blobs without hashing the whole runtime.
#: `record`-role paths, because only non-record roles are asserted byte-identical
#: against the working tree — binding a source under active edit would make the
#: fixture fail for an unrelated reason. The drift branch itself is covered by
#: `test_runtime_drift_since_the_evidence_is_caught`.
FIXTURE_SOURCES = {
    "results/reference_classifier/g1_adjudication.json": "record",
    "results/baseline/g2/g2_adjudication.json": "record",
}


@pytest.fixture(scope="module")
def head_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _csv_bytes(schema: tuple[str, ...], rows: list[dict[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(schema), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in schema})
    return buffer.getvalue().encode("utf-8")


def _outage_record() -> dict[str, Any]:
    """The real frozen artifact, so the manifest re-derivation genuinely agrees."""

    return json.loads(
        (REPO_ROOT / "results/baseline/w4/outage_policy.json").read_text(encoding="utf-8")
    )


def _scheduled_noise(sample_id: str) -> str:
    """The real scheduled identity, so the verifier's recomputation agrees."""

    return make_noise_id(
        {
            "dataset_version": get(f"datasets.{DATASET}.archive_sha256"),
            "split_manifest_hash": manifest_sha256(DATASET),
            "stable_sample_id": sample_id,
            "test_snr_db": 18.0,
            "channel_seed": 0,
            "channel": "awgn",
            "k": 3200,
            "block_index": 0,
            "rng_purpose": "channel_noise",
        }
    )


def _cell_id() -> str:
    return make_analysis_cell_id({"train_seed": 0, "channel_seed": 0})


def _pair(sample_id: str) -> str:
    return make_pair_id(
        {
            "analysis_cell_id": _cell_id(),
            "stable_sample_id": sample_id,
            "bw_ratio": "r_1_24",
            "test_snr_db": 18.0,
            "noise_id": _scheduled_noise(sample_id),
        }
    )


def _per_image_rows(selected_class: int) -> list[dict[str, Any]]:
    """Three rows: one delivered-correct, one delivered-wrong, one outage."""

    base = {
        "run_id": FIXTURE_RUN_ID,
        "analysis_cell_id": _cell_id(),
        "dataset": DATASET,
        "dataset_version": get(f"datasets.{DATASET}.archive_sha256"),
        "split": "val",
        "bw_ratio": "r_1_24",
        "test_snr_db": 18.0,
        "source_bytes": 1063,
    }
    return [
        base
        | {
            "pair_id": _pair("0" * 16),
            "noise_id": _scheduled_noise("0" * 16),
            "stable_sample_id": "0" * 16,
            "true_label": 4,
            "pred_label": 4,
            "correct": "true",
            "outage": "false",
            "outage_reason": "",
        },
        base
        | {
            "pair_id": _pair("1" * 16),
            "noise_id": _scheduled_noise("1" * 16),
            "stable_sample_id": "1" * 16,
            "true_label": 5,
            "pred_label": 6,
            "correct": "false",
            "outage": "false",
            "outage_reason": "",
        },
        base
        | {
            "pair_id": _pair("2" * 16),
            "noise_id": _scheduled_noise("2" * 16),
            "stable_sample_id": "2" * 16,
            "true_label": selected_class,
            "pred_label": selected_class,
            "correct": "true",
            "outage": "true",
            "outage_reason": DECODE_FAILURE,
        },
    ]


def _aggregate_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    n_correct = sum(1 for row in rows if row["correct"] == "true")
    delivered = [row for row in rows if row["outage"] == "false"]
    delivered_correct = sum(1 for row in delivered if row["correct"] == "true")
    decode = sum(1 for row in rows if row["outage_reason"] == DECODE_FAILURE)
    values = {
        "run_id": FIXTURE_RUN_ID,
        "timestamp": "2026-07-31T00:00:00+00:00",
        "git_commit": "b" * 40,
        "git_dirty": "false",
        "config_hash": FIXTURE_CONFIG_HASH,
        "checkpoint_id": EXPECTED_CHECKPOINT_SHA256,
        "system": "classical_fixed_mcs",
        "dataset": DATASET,
        "split": "val",
        "n": n,
        "k": 3200,
        "bw_ratio": "r_1_24",
        "channel": "awgn",
        "train_snr_db": "",
        "test_snr_db": 18.0,
        "train_seed": 0,
        "channel_seed": 0,
        "lambda": "",
        "source_codec": "jpeg2000",
        "jpeg_quality": "",
        "j2k_target_bytes": 1063,
        "ldpc_rate": "2/3",
        "modulation": "qam16",
        "top1_acc": n_correct / n,
        "n_correct": n_correct,
        "n_test": n,
        "psnr_db": 27.0,
        "ssim": 0.77,
        "bytes_sent": 1063,
        "header_bytes": 157.0,
        "payload_bytes": 905.0,
        "papr_db": 2.69,
        "decode_failure_rate": decode / n,
        "infeasible_rate": 0.0,
        "coverage_rate": len(delivered) / n,
        "acc_given_delivery": delivered_correct / len(delivered),
        "test_subset": "",
        "wall_clock_s": 1.0,
        "peak_vram_gb": "",
        "classifier_variant": "clean",
        "quantiser_bits": "",
        "transmit_dim": "",
        "entropy_stream_bytes": "",
        "entropy_table_bytes": "",
        "side_information_bytes": "",
        "tb_crc_type": "crc24a",
        "base_graph": 1,
        "lifting_size": 160,
        "num_codeblocks": 2,
        "filler_bits": 576,
        "effective_code_rate": 0.5,
        "model_param_count": "",
    }
    return values


#: A real resolved configuration for the fixture cell, so the verifier's
#: reconstruct-and-rehash path is exercised rather than stubbed.
FIXTURE_RUN_CONFIG = load_experiment(
    "configs/classical-baseline-w4-imagenette.yaml",
    train_seed=0,
    channel_seed=0,
    test_snr_db=18,
)
FIXTURE_CONFIG_HASH = config_hash(FIXTURE_RUN_CONFIG)
FIXTURE_RUN_ID = make_run_id(
    {
        "system": "classical_fixed_mcs",
        "dataset": DATASET,
        "dataset_version": get(f"datasets.{DATASET}.archive_sha256"),
        "split": "val",
        "split_manifest_hash": manifest_sha256(DATASET),
        "bw_ratio": "r_1_24",
        "test_snr_db": 18.0,
        "train_seed": 0,
        "channel_seed": 0,
        "config_hash": FIXTURE_CONFIG_HASH,
        "checkpoint_id": EXPECTED_CHECKPOINT_SHA256,
        "classifier_variant": "clean",
        "ldpc_rate": "2/3",
        "modulation": "qam16",
        "quantiser_bits": None,
        "transmit_dim": None,
        "lambda": None,
        "analysis_version": get("config.analysis_version"),
    }
)


def _raw_rows(per_image: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The final raw-row artifact the verifier recomputes byte and timing facts from."""

    rows = []
    for index, row in enumerate(per_image):
        outage = row["outage"] == "true"
        verdict = DECODE_FAILURE if outage else "delivered"
        rows.append(
            {
                "work_id": f"{index:064x}",
                "group": "imagenette160_task_scored",
                "task_scored": True,
                "verdict": verdict,
                "wall_clock_s": 0.25,
                "k_symbols": 3200,
                "config_hash": FIXTURE_CONFIG_HASH,
                "scheduled_noise_id": row["noise_id"],
                "actual_noise_id": row["noise_id"],
                "noise_consumed": True,
                "summary": {"accounting": {"payload_bytes": 1063}},
                "source_coding": {
                    "emitted_bytes": 1062,
                    "header_bytes": 157,
                    "payload_bytes": 905,
                    "payload_filler_bytes": 1,
                },
                "per_image": row,
            }
        )
    return rows


def _write_run_configs(directory: Path) -> tuple[list[dict[str, Any]], str]:
    """Archive the fixture configuration exactly as the runner would."""

    configs = {("fixture", 0, 0, 18.0): (FIXTURE_RUN_CONFIG, FIXTURE_CONFIG_HASH)}
    index = runner.write_run_config_artifacts(
        configs, directory / "run_configs", relative_to=directory
    )
    return index, runner.config_hash_root(configs)


@pytest.fixture
def evidence(tmp_path: Path, head_commit: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(verifier, "EXPECTED_SOURCES", FIXTURE_SOURCES)
    directory = tmp_path / "w4"
    directory.mkdir()
    run_config_index, config_root = _write_run_configs(directory)

    outage = _outage_record()
    rows = _per_image_rows(int(outage["selected_class"]))
    aggregate = _aggregate_row(rows)

    raw_rows = _raw_rows(rows)
    raw_body = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in raw_rows
    ).encode("utf-8")
    (directory / "smoke_rows.jsonl").write_bytes(raw_body)

    per_image_bytes = _csv_bytes(per_image_schema(), rows)
    aggregate_bytes = _csv_bytes(aggregate_schema(), [aggregate])
    (directory / "per_image.csv").write_bytes(per_image_bytes)
    (directory / "aggregate.csv").write_bytes(aggregate_bytes)

    summary = {
        "complete": True,
        "evidence_labels": list(EVIDENCE_LABELS),
        "git_dirty": False,
        "execution_source_commit": head_commit,
        "config_hash_root": config_root,
        "config_hashes": {"imagenette160_task_scored/18.0": FIXTURE_CONFIG_HASH},
        "plan_sha256": "p" * 64,
        "checkpoint_id": EXPECTED_CHECKPOINT_SHA256,
        "classifier_config_hash": EXPECTED_CONFIG_HASH,
        "br4_sweep_completed": False,
        "operating_point_selected": False,
        "g8_status": "unresolved",
        "j2k_resolutions_issue_status": "resolved_by_am80",
        "training_performed": False,
        "test_split_access": {
            "test_accessed": False,
            "test_inference": False,
            "test_accuracy_computed": False,
            "test_split_sealed": True,
        },
        "per_image_csv_sha256": hashlib.sha256(per_image_bytes).hexdigest(),
        "aggregate_csv_sha256": hashlib.sha256(aggregate_bytes).hexdigest(),
        "raw_rows_sha256": hashlib.sha256(raw_body).hexdigest(),
        "raw_rows_count": len(raw_rows),
        "worklist_sha256": canonical_sha256([row["work_id"] for row in raw_rows]),
        "wall_clock_s": sum(row["wall_clock_s"] for row in raw_rows),
        "openjpeg_version": get("environment.openjpeg"),
        "openjpeg_preflight_preceded_artifacts": True,
        "cifar10_transport_only": {
            "declaration": (
                "transport-only plumbing smoke\nno task accuracy\n"
                "no frozen-classifier inference\n"
                "not comparable to Imagenette task labels"
            ),
            "classifier_inference_performed": False,
            "task_accuracy": None,
            "top1_acc": None,
            "n_correct": None,
            "encode_axis_px": 32,
            "sample_count": 5,
        },
        "imagenette160_task_scored": {
            "dataset": DATASET,
            "cells": [
                {
                    "dataset": DATASET,
                    "test_snr_db": 18.0,
                    "n": len(rows),
                    "top1_acc": aggregate["top1_acc"],
                }
            ],
        },
    }
    resolved = {
        "complete": True,
        "evidence_labels": list(EVIDENCE_LABELS),
        "config_hash_root": config_root,
        "plan_sha256": "p" * 64,
        "run_configs": run_config_index,
        "field_semantics": {
            "aggregate": dict.fromkeys(aggregate_schema(), {}),
            "per_image": dict.fromkeys(per_image_schema(), {}),
        },
    }
    manifest = {
        "execution_source_commit": head_commit,
        "sources": [
            {
                "path": path,
                "role": role,
                "git_blob_sha": subprocess.run(
                    ["git", "rev-parse", f"{head_commit}:{path}"],
                    cwd=REPO_ROOT, capture_output=True, text=True, check=True
                ).stdout.strip(),
                "sha256": hashlib.sha256(
                    verifier.git_bytes(head_commit, path)
                ).hexdigest(),
                "bytes": len(verifier.git_bytes(head_commit, path)),
                "source_commit": head_commit,
            }
            for path, role in sorted(FIXTURE_SOURCES.items())
        ],
    }
    accounting = {"evidence_labels": list(EVIDENCE_LABELS), "examples": []}

    for name, payload in (
        ("smoke_summary.json", summary),
        ("resolved_config.json", resolved),
        ("outage_policy.json", outage),
        ("accounting_examples.json", accounting),
        ("execution_source_manifest.json", manifest),
    ):
        (directory / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return directory


def _run_all(evidence: Path) -> None:
    payloads = verifier.check_presence_and_labels(evidence)
    policy = verifier.check_outage_policy(payloads["outage"])
    verifier.check_records(evidence, policy, payloads["summary"])
    verifier.check_cifar_separation(payloads["summary"])
    records = verifier.check_records(evidence, policy, payloads["summary"])
    cell_hashes = verifier.check_configuration(
        evidence, payloads["resolved"], payloads["summary"]
    )
    raw_rows = verifier.check_raw_rows(evidence, payloads["summary"], cell_hashes)
    verifier.check_identities(
        raw_rows,
        payloads["resolved"]["run_configs"],
        records["aggregates"],
        records["per_image"],
    )
    verifier.check_byte_accounting(evidence, raw_rows, records["aggregates"])
    verifier.check_sources(evidence, payloads["summary"])


def _rewrite(evidence: Path, name: str, mutate) -> None:
    path = evidence / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _rewrite_csv(evidence: Path, name: str, schema, mutate) -> None:
    path = evidence / name
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    rows = mutate(rows)
    body = _csv_bytes(tuple(schema), rows)
    path.write_bytes(body)
    field = "per_image_csv_sha256" if "per_image" in name else "aggregate_csv_sha256"
    _rewrite(
        evidence,
        "smoke_summary.json",
        lambda payload: payload.__setitem__(field, hashlib.sha256(body).hexdigest()),
    )


# ---------------------------------------------------------------------------


def test_the_valid_fixture_verifies(evidence: Path) -> None:
    _run_all(evidence)


def test_missing_evidence_fails(evidence: Path) -> None:
    (evidence / "accounting_examples.json").unlink()
    with pytest.raises(verifier.VerificationError, match="missing W4 evidence file"):
        _run_all(evidence)


def test_partial_evidence_relabelled_complete_is_caught(evidence: Path) -> None:
    (evidence / "smoke_progress.json").write_text(
        json.dumps({"complete": False}), encoding="utf-8"
    )
    with pytest.raises(verifier.VerificationError, match="still marked partial"):
        _run_all(evidence)


def test_incomplete_summary_fails(evidence: Path) -> None:
    _rewrite(evidence, "smoke_summary.json", lambda p: p.__setitem__("complete", False))
    with pytest.raises(verifier.VerificationError, match="not complete"):
        _run_all(evidence)


def test_wrong_evidence_labels_fail(evidence: Path) -> None:
    _rewrite(
        evidence,
        "smoke_summary.json",
        lambda p: p.__setitem__("evidence_labels", ["bounded"]),
    )
    with pytest.raises(verifier.VerificationError, match="wrong evidence labels"):
        _run_all(evidence)


def test_a_dirty_worktree_fails(evidence: Path) -> None:
    _rewrite(evidence, "smoke_summary.json", lambda p: p.__setitem__("git_dirty", True))
    with pytest.raises(verifier.VerificationError, match="dirty worktree"):
        _run_all(evidence)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("br4_sweep_completed", True, "BR-4 or G-8 completion"),
        ("operating_point_selected", True, "BR-4 or G-8 completion"),
        ("g8_status", "resolved", "BR-4 or G-8 completion"),
        ("j2k_resolutions_issue_status", "unresolved", "AM-80 resolution"),
        ("training_performed", True, "records training"),
    ],
)
def test_claims_of_completion_are_rejected(evidence, field, value, message) -> None:
    _rewrite(evidence, "smoke_summary.json", lambda p: p.__setitem__(field, value))
    with pytest.raises(verifier.VerificationError, match=message):
        _run_all(evidence)


def test_wording_that_claims_a_full_sweep_is_rejected(evidence: Path) -> None:
    _rewrite(
        evidence,
        "smoke_summary.json",
        lambda p: p.__setitem__("note", "BR-4 sweep complete and operating point selected"),
    )
    with pytest.raises(verifier.VerificationError, match="evidence claims"):
        _run_all(evidence)


def test_an_unreachable_source_commit_fails(evidence: Path) -> None:
    for name in ("smoke_summary.json", "execution_source_manifest.json"):
        _rewrite(evidence, name, lambda p: p.__setitem__(
            "execution_source_commit", "0" * 40))
    with pytest.raises(verifier.VerificationError, match="not reachable"):
        _run_all(evidence)


def test_a_changed_source_commit_between_files_fails(evidence: Path) -> None:
    _rewrite(
        evidence,
        "execution_source_manifest.json",
        lambda p: p.__setitem__("execution_source_commit", "0" * 40),
    )
    with pytest.raises(verifier.VerificationError, match="different execution commits"):
        _run_all(evidence)


def test_one_changed_runtime_byte_is_caught(evidence: Path) -> None:
    def mutate(payload):
        payload["sources"][0]["sha256"] = "0" * 64

    _rewrite(evidence, "execution_source_manifest.json", mutate)
    with pytest.raises(verifier.VerificationError, match="differ from the manifest"):
        _run_all(evidence)


def test_runtime_drift_since_the_evidence_is_caught(
    evidence: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A runtime source edited after the run must invalidate the evidence."""

    path = next(iter(FIXTURE_SOURCES))
    monkeypatch.setattr(verifier, "EXPECTED_SOURCES", {path: "runtime"})
    _rewrite(
        evidence,
        "execution_source_manifest.json",
        lambda p: p.__setitem__(
            "sources", [e for e in p["sources"] if e["path"] == path]
        ),
    )
    _rewrite(
        evidence,
        "execution_source_manifest.json",
        lambda p: p["sources"][0].__setitem__("role", "runtime"),
    )
    monkeypatch.setattr(verifier, "_current_bytes", lambda _path: b"drifted")
    with pytest.raises(verifier.VerificationError, match="has drifted"):
        _run_all(evidence)


def test_an_omitted_bound_source_is_caught(evidence: Path) -> None:
    _rewrite(
        evidence,
        "execution_source_manifest.json",
        lambda p: p.__setitem__("sources", p["sources"][:1]),
    )
    with pytest.raises(verifier.VerificationError, match="bound source set differs"):
        _run_all(evidence)


def test_a_deleted_per_image_row_fails_reconciliation(evidence: Path) -> None:
    _rewrite_csv(evidence, "per_image.csv", per_image_schema(), lambda rows: rows[:-1])
    with pytest.raises(verifier.VerificationError, match="does not reconcile"):
        _run_all(evidence)


def test_a_duplicated_per_image_row_is_caught(evidence: Path) -> None:
    _rewrite_csv(
        evidence, "per_image.csv", per_image_schema(), lambda rows: rows + [rows[0]]
    )
    with pytest.raises(verifier.VerificationError, match="duplicate per-image identity"):
        _run_all(evidence)


def test_an_altered_aggregate_value_is_caught(evidence: Path) -> None:
    def mutate(rows):
        rows[0]["top1_acc"] = "0.99"
        return rows

    _rewrite_csv(evidence, "aggregate.csv", aggregate_schema(), mutate)
    with pytest.raises(verifier.VerificationError, match="does not reconcile"):
        _run_all(evidence)


def test_an_altered_selected_outage_class_is_caught(evidence: Path) -> None:
    _rewrite(
        evidence,
        "outage_policy.json",
        lambda p: p.__setitem__("selected_class", 1),
    )
    with pytest.raises(OutagePolicyError, match="tie-break|selected class"):
        _run_all(evidence)


def test_a_hardcoded_uniform_outage_accuracy_is_caught(evidence: Path) -> None:
    """Halving one count leaves 1/n unchanged but moves the measured value."""

    def mutate(payload):
        counts = list(payload["class_counts"])
        counts[payload["selected_class"]] = counts[payload["selected_class"]] // 2
        payload["class_counts"] = counts
        payload["selected_count"] = counts[payload["selected_class"]]
        payload["numerator"] = counts[payload["selected_class"]]
        payload["validation_count"] = sum(counts)
        payload["denominator"] = sum(counts)
        payload["measured_validation_accuracy"] = 1 / payload["class_count"]

    _rewrite(evidence, "outage_policy.json", mutate)
    with pytest.raises(
        (OutagePolicyError, verifier.VerificationError),
        # Recomputation from counts is what catches it — which of the
        # count-derived invariants fires first is not the point.
        match="tied-maximum classes|not a maximum|not selected_count / validation_count",
    ):
        _run_all(evidence)


def test_a_moved_schema_column_is_caught(evidence: Path) -> None:
    path = evidence / "per_image.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    header[0], header[1] = header[1], header[0]
    path.write_text("\n".join([",".join(header), *lines[1:]]) + "\n", encoding="utf-8")
    _rewrite(
        evidence,
        "smoke_summary.json",
        lambda p: p.__setitem__(
            "per_image_csv_sha256", hashlib.sha256(path.read_bytes()).hexdigest()
        ),
    )
    with pytest.raises(verifier.VerificationError, match="not the configured schema"):
        _run_all(evidence)


def test_a_dropped_schema_column_is_caught(evidence: Path) -> None:
    path = evidence / "per_image.csv"
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    schema = tuple(f for f in per_image_schema() if f != "correct")
    path.write_bytes(_csv_bytes(schema, rows))
    _rewrite(
        evidence,
        "smoke_summary.json",
        lambda p: p.__setitem__(
            "per_image_csv_sha256", hashlib.sha256(path.read_bytes()).hexdigest()
        ),
    )
    with pytest.raises(verifier.VerificationError, match="not the configured schema"):
        _run_all(evidence)


def test_a_changed_checkpoint_identity_is_caught(evidence: Path) -> None:
    def mutate(rows):
        rows[0]["checkpoint_id"] = "0" * 64
        return rows

    _rewrite_csv(evidence, "aggregate.csv", aggregate_schema(), mutate)
    with pytest.raises(verifier.VerificationError, match="other than the frozen G-1"):
        _run_all(evidence)


def test_a_test_split_row_is_caught(evidence: Path) -> None:
    def mutate(rows):
        rows[0]["split"] = "test"
        return rows

    _rewrite_csv(evidence, "per_image.csv", per_image_schema(), mutate)
    with pytest.raises(verifier.VerificationError, match="claims split=test"):
        _run_all(evidence)


def test_a_cifar_row_carrying_task_accuracy_is_caught(evidence: Path) -> None:
    _rewrite(
        evidence,
        "smoke_summary.json",
        lambda p: p["cifar10_transport_only"].__setitem__("top1_acc", 0.6),
    )
    with pytest.raises(verifier.VerificationError, match="across class vocabularies"):
        _run_all(evidence)


def test_cifar_classifier_inference_is_caught(evidence: Path) -> None:
    _rewrite(
        evidence,
        "smoke_summary.json",
        lambda p: p["cifar10_transport_only"].__setitem__(
            "classifier_inference_performed", True
        ),
    )
    with pytest.raises(verifier.VerificationError, match="reports frozen-classifier"):
        _run_all(evidence)


def test_a_cifar_per_image_task_row_is_caught(evidence: Path) -> None:
    def mutate(rows):
        rows[0]["dataset"] = "cifar10"
        return rows

    _rewrite_csv(evidence, "per_image.csv", per_image_schema(), mutate)
    with pytest.raises(verifier.VerificationError, match="only 'imagenette160'"):
        _run_all(evidence)


def test_an_unconfigured_cifar_axis_is_caught(evidence: Path) -> None:
    _rewrite(
        evidence,
        "smoke_summary.json",
        lambda p: p["cifar10_transport_only"].__setitem__("encode_axis_px", 28),
    )
    with pytest.raises(verifier.VerificationError, match="unconfigured encode axis"):
        _run_all(evidence)


def test_an_invalid_system_value_is_caught(evidence: Path) -> None:
    def mutate(rows):
        rows[0]["system"] = "classical_someday"
        return rows

    _rewrite_csv(evidence, "aggregate.csv", aggregate_schema(), mutate)
    with pytest.raises(verifier.VerificationError, match="unsupported system"):
        _run_all(evidence)


def test_an_incorrect_outage_reason_is_caught(evidence: Path) -> None:
    def mutate(rows):
        rows[-1]["outage_reason"] = "gremlins"
        return rows

    _rewrite_csv(evidence, "per_image.csv", per_image_schema(), mutate)
    with pytest.raises(verifier.VerificationError, match="invalid reason"):
        _run_all(evidence)


def test_an_outage_row_not_using_the_frozen_class_is_caught(evidence: Path) -> None:
    def mutate(rows):
        rows[-1]["pred_label"] = str(int(rows[-1]["pred_label"]) + 1)
        return rows

    _rewrite_csv(evidence, "per_image.csv", per_image_schema(), mutate)
    with pytest.raises(verifier.VerificationError, match="frozen constant prediction"):
        _run_all(evidence)


def test_non_binary_correctness_is_caught(evidence: Path) -> None:
    def mutate(rows):
        rows[0]["correct"] = "0.5"
        return rows

    _rewrite_csv(evidence, "per_image.csv", per_image_schema(), mutate)
    with pytest.raises(verifier.VerificationError, match="not strictly boolean"):
        _run_all(evidence)


def test_a_delivered_row_with_an_outage_reason_is_caught(evidence: Path) -> None:
    def mutate(rows):
        rows[0]["outage_reason"] = CODEC_INFEASIBILITY
        return rows

    _rewrite_csv(evidence, "per_image.csv", per_image_schema(), mutate)
    with pytest.raises(verifier.VerificationError, match="delivered row carries"):
        _run_all(evidence)


def test_a_config_hash_root_mismatch_is_caught(evidence: Path) -> None:
    _rewrite(
        evidence,
        "resolved_config.json",
        lambda p: p.__setitem__("config_hash_root", "0" * 64),
    )
    with pytest.raises(
        verifier.VerificationError, match="disagree on config_hash_root"
    ):
        _run_all(evidence)


def test_a_params_snapshot_mismatch_is_caught(evidence: Path) -> None:
    """The snapshot now lives inside each archived RunConfig, not beside it."""

    entry = json.loads(
        (evidence / "resolved_config.json").read_text(encoding="utf-8")
    )["run_configs"][0]
    path = evidence / entry["relative_path"]
    body = json.loads(path.read_text(encoding="utf-8"))
    body["parameters"]["bandwidth"] = {"tampered": True}
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    with pytest.raises(verifier.VerificationError, match="does not match its"):
        _run_all(evidence)


def test_a_csv_hash_mismatch_is_caught(evidence: Path) -> None:
    _rewrite(
        evidence,
        "smoke_summary.json",
        lambda p: p.__setitem__("per_image_csv_sha256", "0" * 64),
    )
    with pytest.raises(verifier.VerificationError, match="does not match the hash"):
        _run_all(evidence)


def test_an_incomplete_field_semantics_table_is_caught(evidence: Path) -> None:
    def mutate(payload):
        payload["field_semantics"]["per_image"].pop("correct")

    _rewrite(evidence, "resolved_config.json", mutate)
    with pytest.raises(verifier.VerificationError, match="field-semantics"):
        _run_all(evidence)


# ---------------------------------------------------------------------------
# Per-cell RunConfig provenance (PB_2C/C2.2)
#
# The old verifier compared `resolved_config.json`'s config_hash against
# `smoke_summary.json`'s. Both carried the same wrong hash, so the check passed
# while one 18 dB fingerprint stood in for every cell. These mutations attack
# the property that check could not see: that the hash *comes out of* a concrete
# configuration describing the cell that actually ran.
# ---------------------------------------------------------------------------


def _index_entry(evidence: Path) -> dict[str, Any]:
    return json.loads(
        (evidence / "resolved_config.json").read_text(encoding="utf-8")
    )["run_configs"][0]


def test_one_config_hash_reused_for_two_snr_points_is_caught(evidence: Path) -> None:
    """The exact PB_2 defect: 18 dB and -8 dB sharing one fingerprint."""

    def mutate(payload):
        digest = payload["config_hashes"]["imagenette160_task_scored/18.0"]
        payload["config_hashes"]["imagenette160_task_scored/-8.0"] = digest

    _rewrite(evidence, "smoke_summary.json", mutate)
    with pytest.raises(verifier.VerificationError, match="share one config hash"):
        _run_all(evidence)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("test_snr_db", -8.0),
        ("bw_ratio", "r_1_48"),
        ("modulation", "bpsk"),
        ("ldpc_rate", "1/3"),
        ("dataset", "cifar10"),
        ("encode_axis_px", 64),
        ("train_seed", 1),
        ("channel_seed", 2),
    ],
)
def test_an_index_entry_that_misdescribes_its_configuration_is_caught(
    evidence: Path, field: str, value: Any
) -> None:
    def mutate(payload):
        payload["run_configs"][0][field] = value

    _rewrite(evidence, "resolved_config.json", mutate)
    with pytest.raises(verifier.VerificationError, match="but the configuration resolves"):
        _run_all(evidence)


def test_a_missing_run_config_artifact_is_caught(evidence: Path) -> None:
    (evidence / _index_entry(evidence)["relative_path"]).unlink()
    with pytest.raises(verifier.VerificationError, match="archived run config .* is missing"):
        _run_all(evidence)


def test_run_config_file_hash_drift_is_caught(evidence: Path) -> None:
    path = evidence / _index_entry(evidence)["relative_path"]
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(verifier.VerificationError, match="recorded file hash"):
        _run_all(evidence)


def test_a_config_hash_that_does_not_reproduce_is_caught(evidence: Path) -> None:
    """Recording a hash beside a configuration is not the same as deriving it."""

    entry = _index_entry(evidence)
    path = evidence / entry["relative_path"]
    body = json.loads(path.read_text(encoding="utf-8"))
    body["resolved"]["test_snr_db"] = -8
    payload = json.dumps(body, indent=2)
    path.write_text(payload, encoding="utf-8")

    def mutate(index):
        index["run_configs"][0]["file_sha256"] = hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()
        index["run_configs"][0]["test_snr_db"] = -8

    _rewrite(evidence, "resolved_config.json", mutate)
    with pytest.raises(
        verifier.VerificationError, match="does not reproduce its own config hash"
    ):
        _run_all(evidence)


def test_an_artifact_not_stored_under_its_own_hash_is_caught(evidence: Path) -> None:
    entry = _index_entry(evidence)
    source = evidence / entry["relative_path"]
    renamed = source.with_name("some-other-name.json")
    source.rename(renamed)

    def mutate(index):
        index["run_configs"][0]["relative_path"] = str(
            renamed.relative_to(evidence)
        )

    _rewrite(evidence, "resolved_config.json", mutate)
    with pytest.raises(verifier.VerificationError, match="not stored under its own"):
        _run_all(evidence)


def test_the_execution_plan_hash_substituted_for_a_config_hash_is_caught(
    evidence: Path,
) -> None:
    digest = _index_entry(evidence)["config_hash"]
    _rewrite(evidence, "smoke_summary.json", lambda p: p.__setitem__("plan_sha256", digest))
    _rewrite(evidence, "resolved_config.json", lambda p: p.__setitem__("plan_sha256", digest))
    with pytest.raises(
        verifier.VerificationError, match="execution-plan hash is being used"
    ):
        _run_all(evidence)


def test_the_root_digest_substituted_for_a_cell_config_hash_is_caught(
    evidence: Path,
) -> None:
    root = json.loads(
        (evidence / "resolved_config.json").read_text(encoding="utf-8")
    )["config_hash_root"]

    def mutate(payload):
        payload["config_hashes"] = {"imagenette160_task_scored/18.0": root}

    _rewrite(evidence, "smoke_summary.json", mutate)
    with pytest.raises(verifier.VerificationError, match="disagree on the cell config"):
        _run_all(evidence)


def test_an_empty_run_config_index_is_caught(evidence: Path) -> None:
    _rewrite(evidence, "resolved_config.json", lambda p: p.__setitem__("run_configs", []))
    with pytest.raises(verifier.VerificationError, match="no run-config index"):
        _run_all(evidence)


def test_a_config_hash_root_that_does_not_reproduce_is_caught(evidence: Path) -> None:
    """The root must follow from the cell hashes, not merely agree between files."""

    forged = "f" * 64

    def mutate(payload):
        payload["config_hash_root"] = forged

    _rewrite(evidence, "resolved_config.json", mutate)
    _rewrite(evidence, "smoke_summary.json", mutate)
    with pytest.raises(verifier.VerificationError, match="does not reproduce from"):
        _run_all(evidence)


# ---------------------------------------------------------------------------
# Identity, byte, timing and preflight verification (PB_2C/C2.5)
#
# The old verifier recomputed aggregate *rates* but never rebuilt an identity,
# never parsed a codestream, never read a row timing and never opened a raw-row
# file. These mutations attack exactly those blind spots.
# ---------------------------------------------------------------------------


def _raw(evidence: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (evidence / "smoke_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _rewrite_raw(evidence: Path, mutate) -> None:
    rows = _raw(evidence)
    rows = mutate(rows) or rows
    body = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")
    (evidence / "smoke_rows.jsonl").write_bytes(body)
    _rewrite(
        evidence,
        "smoke_summary.json",
        lambda payload: payload.update(
            raw_rows_sha256=hashlib.sha256(body).hexdigest(),
            raw_rows_count=len(rows),
            worklist_sha256=canonical_sha256([row["work_id"] for row in rows]),
            wall_clock_s=sum(row["wall_clock_s"] for row in rows),
        ),
    )


def test_a_missing_final_raw_row_artifact_is_caught(evidence: Path) -> None:
    (evidence / "smoke_rows.jsonl").unlink()
    with pytest.raises(verifier.VerificationError, match="missing W4 evidence file"):
        _run_all(evidence)


def test_partial_raw_rows_relabelled_complete_are_caught(evidence: Path) -> None:
    """A truncated run must never pass as a finished one."""

    rows = _raw(evidence)
    body = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows[:-1]
    ).encode("utf-8")
    (evidence / "smoke_rows.jsonl").write_bytes(body)
    with pytest.raises(verifier.VerificationError, match="does not match the hash"):
        _run_all(evidence)


def test_a_deleted_final_raw_row_is_caught(evidence: Path) -> None:
    _rewrite_raw(evidence, lambda rows: rows[:-1])
    with pytest.raises(verifier.VerificationError, match="describe different rows"):
        _run_all(evidence)


def test_a_duplicated_final_raw_row_is_caught(evidence: Path) -> None:
    _rewrite_raw(evidence, lambda rows: rows + [rows[0]])
    with pytest.raises(verifier.VerificationError, match="duplicated work item"):
        _run_all(evidence)


def test_a_null_noise_identity_on_an_infeasible_row_is_caught(evidence: Path) -> None:
    """The PB_2 defect: an infeasible row that cannot pair with anything."""

    def mutate(rows):
        rows[-1].update(
            verdict=CODEC_INFEASIBILITY,
            scheduled_noise_id=None,
            actual_noise_id=None,
            noise_consumed=False,
        )

    _rewrite_raw(evidence, mutate)
    with pytest.raises(
        verifier.VerificationError, match="carries no scheduled noise identity"
    ):
        _run_all(evidence)


def test_an_infeasible_row_claiming_noise_was_consumed_is_caught(evidence: Path) -> None:
    def mutate(rows):
        rows[-1].update(verdict=STRUCTURAL_INFEASIBILITY, noise_consumed=True)

    _rewrite_raw(evidence, mutate)
    with pytest.raises(verifier.VerificationError, match="claims a channel realisation"):
        _run_all(evidence)


def test_transmitted_noise_differing_from_the_schedule_is_caught(evidence: Path) -> None:
    _rewrite_raw(evidence, lambda rows: rows[0].update(actual_noise_id="f" * 64))
    with pytest.raises(verifier.VerificationError, match="does not reconcile"):
        _run_all(evidence)


def test_a_per_image_row_not_carrying_the_scheduled_identity_is_caught(
    evidence: Path,
) -> None:
    def mutate(rows):
        rows[0]["per_image"]["noise_id"] = "e" * 64

    _rewrite_raw(evidence, mutate)
    with pytest.raises(verifier.VerificationError, match="scheduled noise identity"):
        _run_all(evidence)


def test_a_pair_id_that_does_not_recompute_is_caught(evidence: Path) -> None:
    def mutate(rows):
        rows[0]["per_image"]["pair_id"] = "d" * 64

    _rewrite_raw(evidence, mutate)
    with pytest.raises(verifier.VerificationError, match="pair_id does not recompute"):
        _run_all(evidence)


def test_an_analysis_cell_id_that_does_not_recompute_is_caught(evidence: Path) -> None:
    def mutate(rows):
        rows[0]["per_image"]["analysis_cell_id"] = "9" * 64

    _rewrite_raw(evidence, mutate)
    with pytest.raises(
        verifier.VerificationError, match="analysis_cell_id does not recompute"
    ):
        _run_all(evidence)


def test_a_run_id_that_does_not_recompute_is_caught(evidence: Path) -> None:
    """Changing a keyed selection must move run_id; a stale one is a mismatch."""

    _rewrite_csv(evidence, "aggregate.csv", aggregate_schema(),
                 lambda rows: [dict(row, train_seed=1) for row in rows])
    with pytest.raises(verifier.VerificationError, match="run_id does not recompute"):
        _run_all(evidence)


def test_a_decode_failure_aggregate_with_null_header_bytes_is_caught(
    evidence: Path,
) -> None:
    """AM-81: overhead is measurable whenever a codestream was emitted."""

    _rewrite_csv(evidence, "aggregate.csv", aggregate_schema(),
                 lambda rows: [dict(row, header_bytes="") for row in rows])
    with pytest.raises(verifier.VerificationError, match="leaves header_bytes blank"):
        _run_all(evidence)


def test_a_decode_failure_aggregate_with_null_payload_bytes_is_caught(
    evidence: Path,
) -> None:
    _rewrite_csv(evidence, "aggregate.csv", aggregate_schema(),
                 lambda rows: [dict(row, payload_bytes="") for row in rows])
    with pytest.raises(verifier.VerificationError, match="leaves payload_bytes blank"):
        _run_all(evidence)


def test_a_wrong_aggregation_denominator_is_caught(evidence: Path) -> None:
    """Averaging delivered rows only -- the PB_2 behaviour -- must now fail."""

    rows = _raw(evidence)
    delivered = [row for row in rows if row["verdict"] != DECODE_FAILURE]
    wrong = sum(row["source_coding"]["payload_bytes"] for row in delivered) / len(
        delivered
    )
    # The fixture's rows all carry the same split, so shift one to make the
    # delivered-only and all-emitted denominators disagree.
    rows[-1]["source_coding"].update(
        payload_bytes=800, emitted_bytes=957, payload_filler_bytes=106
    )
    body = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")
    (evidence / "smoke_rows.jsonl").write_bytes(body)
    _rewrite(
        evidence,
        "smoke_summary.json",
        lambda p: p.update(raw_rows_sha256=hashlib.sha256(body).hexdigest()),
    )
    _rewrite_csv(evidence, "aggregate.csv", aggregate_schema(),
                 lambda csv_rows: [dict(row, payload_bytes=wrong) for row in csv_rows])
    with pytest.raises(verifier.VerificationError, match="recomputes to"):
        _run_all(evidence)


def test_payload_filler_folded_into_the_payload_column_is_caught(
    evidence: Path,
) -> None:
    def mutate(rows):
        for row in rows:
            source = row["source_coding"]
            source["payload_bytes"] = source["payload_bytes"] + source[
                "payload_filler_bytes"
            ]

    _rewrite_raw(evidence, mutate)
    with pytest.raises(verifier.VerificationError, match="!= emitted bytes"):
        _run_all(evidence)


def test_a_row_reporting_bytes_without_a_codestream_is_caught(evidence: Path) -> None:
    def mutate(rows):
        rows[-1]["source_coding"]["emitted_bytes"] = None

    _rewrite_raw(evidence, mutate)
    with pytest.raises(verifier.VerificationError, match="without a codestream"):
        _run_all(evidence)


def test_a_negative_row_timing_is_caught(evidence: Path) -> None:
    _rewrite_raw(evidence, lambda rows: rows[0].update(wall_clock_s=-1.0))
    with pytest.raises(verifier.VerificationError, match="non-negative elapsed time"):
        _run_all(evidence)


def test_a_wall_clock_excluding_resumed_rows_is_caught(evidence: Path) -> None:
    """Reporting only the resumed session's elapsed time -- the PB_2 behaviour."""

    rows = _raw(evidence)
    resumed_only = rows[-1]["wall_clock_s"]
    _rewrite(
        evidence, "smoke_summary.json", lambda p: p.update(wall_clock_s=resumed_only)
    )
    with pytest.raises(
        verifier.VerificationError, match="not the sum of the durable row timings"
    ):
        _run_all(evidence)


def test_a_wrong_actual_openjpeg_version_is_caught(evidence: Path) -> None:
    _rewrite(
        evidence, "smoke_summary.json", lambda p: p.update(openjpeg_version="2.4.0")
    )
    with pytest.raises(
        verifier.VerificationError, match="OpenJPEG version other than the configured"
    ):
        _run_all(evidence)


def test_evidence_not_declaring_preflight_ordering_is_caught(evidence: Path) -> None:
    _rewrite(
        evidence,
        "smoke_summary.json",
        lambda p: p.update(openjpeg_preflight_preceded_artifacts=False),
    )
    with pytest.raises(
        verifier.VerificationError, match="preflight preceded artifact creation"
    ):
        _run_all(evidence)


def test_raw_rows_out_of_worklist_order_are_caught(evidence: Path) -> None:
    rows = _raw(evidence)
    reordered = list(reversed(rows))
    body = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in reordered
    ).encode("utf-8")
    (evidence / "smoke_rows.jsonl").write_bytes(body)
    _rewrite(
        evidence,
        "smoke_summary.json",
        lambda p: p.update(raw_rows_sha256=hashlib.sha256(body).hexdigest()),
    )
    with pytest.raises(verifier.VerificationError, match="deterministic order"):
        _run_all(evidence)


def test_a_raw_row_naming_an_unarchived_config_hash_is_caught(evidence: Path) -> None:
    _rewrite_raw(evidence, lambda rows: rows[0].update(config_hash="7" * 64))
    with pytest.raises(verifier.VerificationError, match="not archived"):
        _run_all(evidence)
