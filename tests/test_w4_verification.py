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

import verify_w4_baseline_integration as verifier
from baseline.classical.outage import EVIDENCE_LABELS, OutagePolicyError
from baseline.classical.pipeline import (
    CODEC_INFEASIBILITY,
    DECODE_FAILURE,
    STRUCTURAL_INFEASIBILITY,
)
from baseline.classical.records import aggregate_schema, per_image_schema
from config.params import REPO_ROOT, get
from models.frozen_reference_classifier import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CONFIG_HASH,
)

DATASET = "imagenette160"
#: A small, stable set of real tracked files, so the source-binding checks run
#: against genuine Git blobs without hashing the whole runtime.
FIXTURE_SOURCES = {
    "src/baseline/classical/outage.py": "runtime",
    "configs/classical-baseline-w4-smoke-plan.yaml": "configuration",
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


def _per_image_rows(selected_class: int) -> list[dict[str, Any]]:
    """Three rows: one delivered-correct, one delivered-wrong, one outage."""

    base = {
        "run_id": "a" * 64,
        "analysis_cell_id": "c" * 64,
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
            "pair_id": "p1" + "0" * 62,
            "noise_id": "n1" + "0" * 62,
            "stable_sample_id": "0" * 16,
            "true_label": 4,
            "pred_label": 4,
            "correct": "true",
            "outage": "false",
            "outage_reason": "",
        },
        base
        | {
            "pair_id": "p2" + "0" * 62,
            "noise_id": "n2" + "0" * 62,
            "stable_sample_id": "1" * 16,
            "true_label": 5,
            "pred_label": 6,
            "correct": "false",
            "outage": "false",
            "outage_reason": "",
        },
        base
        | {
            "pair_id": "p3" + "0" * 62,
            "noise_id": "n3" + "0" * 62,
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
        "run_id": "a" * 64,
        "timestamp": "2026-07-31T00:00:00+00:00",
        "git_commit": "b" * 40,
        "git_dirty": "false",
        "config_hash": "cfg" + "0" * 61,
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
        "payload_bytes": 892.0,
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


@pytest.fixture
def evidence(tmp_path: Path, head_commit: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(verifier, "EXPECTED_SOURCES", FIXTURE_SOURCES)
    directory = tmp_path / "w4"
    directory.mkdir()

    outage = _outage_record()
    rows = _per_image_rows(int(outage["selected_class"]))
    aggregate = _aggregate_row(rows)

    per_image_bytes = _csv_bytes(per_image_schema(), rows)
    aggregate_bytes = _csv_bytes(aggregate_schema(), [aggregate])
    (directory / "per_image.csv").write_bytes(per_image_bytes)
    (directory / "aggregate.csv").write_bytes(aggregate_bytes)

    summary = {
        "complete": True,
        "evidence_labels": list(EVIDENCE_LABELS),
        "git_dirty": False,
        "execution_source_commit": head_commit,
        "config_hash": aggregate["config_hash"],
        "checkpoint_id": EXPECTED_CHECKPOINT_SHA256,
        "classifier_config_hash": EXPECTED_CONFIG_HASH,
        "br4_sweep_completed": False,
        "operating_point_selected": False,
        "g8_status": "unresolved",
        "j2k_resolutions_issue_status": "unresolved",
        "training_performed": False,
        "test_split_access": {
            "test_accessed": False,
            "test_inference": False,
            "test_accuracy_computed": False,
            "test_split_sealed": True,
        },
        "per_image_csv_sha256": hashlib.sha256(per_image_bytes).hexdigest(),
        "aggregate_csv_sha256": hashlib.sha256(aggregate_bytes).hexdigest(),
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
        "config_hash": aggregate["config_hash"],
        "parameters": {root: get(root) for root in get("config.fingerprint_parameter_roots")},
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
    verifier.check_configuration(payloads["resolved"], payloads["summary"])
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
        ("j2k_resolutions_issue_status", "resolved", "marked resolved"),
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


def test_a_config_hash_mismatch_is_caught(evidence: Path) -> None:
    _rewrite(
        evidence, "resolved_config.json", lambda p: p.__setitem__("config_hash", "0" * 64)
    )
    with pytest.raises(verifier.VerificationError, match="disagree on config_hash"):
        _run_all(evidence)


def test_a_params_snapshot_mismatch_is_caught(evidence: Path) -> None:
    def mutate(payload):
        payload["parameters"]["bandwidth"] = {"tampered": True}

    _rewrite(evidence, "resolved_config.json", mutate)
    with pytest.raises(verifier.VerificationError, match="snapshot differs"):
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
