#!/usr/bin/env python3
"""Verify the bounded W4 classical-baseline integration evidence.

Network-free. Recomputes rather than trusts: the outage class is re-derived from
the committed validation manifest, every aggregate rate is recomputed from the
per-image rows, and every bound source is re-hashed at the declared execution
commit. A summary field that merely *asserts* a number is never accepted as
evidence of it.

The checks are fail-closed and deliberately blunt about the two mistakes that
would be invisible in the numbers themselves:

* a CIFAR-10 row carrying a task score, which would mean the Imagenette-160
  frozen classifier was applied across class vocabularies; and
* an outage accuracy of ``1/n_classes`` that does not follow from the manifest's
  own label counts. The committed value happens to equal ``0.1`` because the
  validation split is exactly stratified, so a float comparison alone could
  never distinguish a measurement from a hardcoded constant.

Usage:
    .venv/bin/python tools/verify_w4_baseline_integration.py
    .venv/bin/python tools/verify_w4_baseline_integration.py --evidence-dir DIR
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from baseline.classical import composition  # noqa: E402
from baseline.classical.outage import (  # noqa: E402
    EVIDENCE_LABELS,
    count_validation_labels,
    policy_from_record,
)
from baseline.classical.pipeline import (  # noqa: E402
    CODEC_INFEASIBILITY,
    DECODE_FAILURE,
    STRUCTURAL_INFEASIBILITY,
)
from baseline.classical.records import (  # noqa: E402
    FROZEN_CLASSIFIER_DATASET,
    aggregate_schema,
    per_image_schema,
)
from config.params import get  # noqa: E402
from artifacts.ids import (  # noqa: E402
    make_analysis_cell_id,
    make_noise_id,
    make_pair_id,
    make_run_id,
)
from data.registry import manifest_sha256  # noqa: E402
from config.run_config import RunConfig, canonical_sha256, config_hash  # noqa: E402
from models.frozen_reference_classifier import (  # noqa: E402
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CONFIG_HASH,
)

sys.path.insert(0, str(REPO / "tools"))
from gen_w4_source_manifest import (  # noqa: E402
    EXPECTED_SOURCES,
    git_bytes,
    sha256_bytes,
)
from gen_w4_integration_adjudication import (  # noqa: E402
    ADJUDICATION_SCHEMA_VERSION,
    BOUND_EVIDENCE_FILES,
    PROVISIONAL_OPERATING_POINTS,
    SELECTION_SOURCES,
    bler_characterization,
    SELECTION_POLICY_FIELDS,
    SELECTION_POLICY_FREEZE_GATE,
    TIE_EQUALITY,
    selection_machinery,
    selection_policy_fingerprint,
    sweep_guard,
    worked_composition_example,
)

EVIDENCE_DIR = Path("results/baseline/w4")
REQUIRED_FILES = (
    "resolved_config.json",
    "outage_policy.json",
    "smoke_summary.json",
    "accounting_examples.json",
    "execution_source_manifest.json",
    "per_image.csv",
    "aggregate.csv",
    "smoke_rows.jsonl",
    "overhead_table.json",
)

ADJUDICATION_FILE = "integration_adjudication.json"

#: Phrases that would claim the bounded run did something it did not.
FORBIDDEN_CLAIMS = (
    "full validation sweep completed",
    "br-4 sweep complete",
    "operating point selected",
    "g8 resolved",
    "g-8 resolved",
    "g-8 selection complete",
)


class VerificationError(RuntimeError):
    """A W4 evidence contract violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read {path.name}: {exc}") from None
    _require(isinstance(value, dict), f"{path.name} must contain an object")
    return value


def _csv_rows(path: Path, schema: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise VerificationError(f"cannot read {path.name}: {exc}") from None
    reader = csv.reader(text.splitlines())
    try:
        header = tuple(next(reader))
    except StopIteration:
        raise VerificationError(f"{path.name} is empty") from None
    _require(
        header == schema,
        f"{path.name} header is not the configured schema: "
        f"missing={sorted(set(schema) - set(header))}, "
        f"unexpected={sorted(set(header) - set(schema))}, "
        f"order_differs={header != schema}",
    )
    rows = []
    for number, fields in enumerate(reader, start=2):
        _require(
            len(fields) == len(schema),
            f"{path.name} line {number} has {len(fields)} fields, expected {len(schema)}",
        )
        rows.append(dict(zip(schema, fields, strict=True)))
    return rows


def _bool(value: str, field: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise VerificationError(f"{field} is not strictly boolean: {value!r}")


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def _labels_ok(payload: Any) -> bool:
    return tuple(payload.get("evidence_labels", ())) == EVIDENCE_LABELS


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise VerificationError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_presence_and_labels(evidence: Path) -> dict[str, Any]:
    for name in REQUIRED_FILES:
        _require((evidence / name).is_file(), f"missing W4 evidence file: {name}")

    summary = _json(evidence / "smoke_summary.json")
    resolved = _json(evidence / "resolved_config.json")
    outage = _json(evidence / "outage_policy.json")
    accounting = _json(evidence / "accounting_examples.json")

    for name, payload in (
        ("smoke_summary.json", summary),
        ("resolved_config.json", resolved),
        ("outage_policy.json", outage),
        ("accounting_examples.json", accounting),
    ):
        _require(_labels_ok(payload), f"{name} carries the wrong evidence labels")

    _require(summary.get("complete") is True, "smoke_summary.json is not complete")
    _require(resolved.get("complete") is True, "resolved_config.json is not complete")
    _require(
        summary.get("git_dirty") is False,
        "the bounded evidence was produced from a dirty worktree",
    )
    _require(
        summary.get("br4_sweep_completed") is False
        and summary.get("operating_point_selected") is False
        and summary.get("g8_status") == "unresolved",
        "smoke_summary.json claims BR-4 or G-8 completion",
    )
    _require(
        summary.get("j2k_resolutions_issue_status") == "resolved_by_am80",
        "the evidence does not record the AM-80 resolution of the JPEG-2000 "
        "resolution issue",
    )
    _require(
        summary.get("training_performed") is False,
        "smoke_summary.json records training",
    )
    access = summary.get("test_split_access") or {}
    _require(
        access.get("test_split_sealed") is True
        and access.get("test_accessed") is False
        and access.get("test_inference") is False
        and access.get("test_accuracy_computed") is False,
        "smoke_summary.json does not declare the test split sealed",
    )

    blob = json.dumps(summary, sort_keys=True).lower()
    for claim in FORBIDDEN_CLAIMS:
        _require(claim not in blob, f"evidence claims {claim!r}")

    progress = evidence / "smoke_progress.json"
    if progress.is_file():
        _require(
            _json(progress).get("complete") is True,
            "the run is still marked partial: smoke_progress.json says complete=false",
        )
    return {
        "summary": summary,
        "resolved": resolved,
        "outage": outage,
        "accounting": accounting,
    }


def check_outage_policy(outage: dict[str, Any]) -> Any:
    """Re-derive the frozen class from the manifest rather than trusting it."""

    policy = policy_from_record(outage)
    _require(
        policy.dataset == FROZEN_CLASSIFIER_DATASET,
        f"the outage class was selected on {policy.dataset!r}, not the frozen "
        f"classifier's {FROZEN_CLASSIFIER_DATASET!r}",
    )
    counts = count_validation_labels(policy.dataset, REPO)
    _require(
        counts.manifest_sha256 == policy.manifest_sha256,
        "the frozen outage policy was selected on a different split manifest",
    )
    _require(
        tuple(counts.class_counts) == tuple(policy.class_counts),
        "recomputed validation class counts differ from the frozen artifact",
    )
    _require(
        counts.selected_class() == policy.selected_class,
        f"the frozen selected class {policy.selected_class} is not the class the "
        f"manifest selects ({counts.selected_class()})",
    )
    _require(
        policy.selected_count == counts.class_counts[policy.selected_class]
        and policy.validation_count == counts.validation_count,
        "the frozen numerator/denominator disagree with the manifest",
    )
    # The measured value coincides with 1/n only because the split is exactly
    # stratified. Accept it only when the counts produce it.
    _require(
        float(outage["measured_validation_accuracy"])
        == policy.selected_count / policy.validation_count,
        "the recorded outage accuracy is not selected_count / validation_count "
        "(a hardcoded 1/n_classes would land here)",
    )
    _require(
        outage.get("test_split_access", {}).get("test_split_sealed") is True,
        "the outage artifact does not declare the test split sealed",
    )
    return policy


def check_records(evidence: Path, policy, summary: dict[str, Any]) -> dict[str, Any]:
    per_image = _csv_rows(evidence / "per_image.csv", per_image_schema())
    aggregates = _csv_rows(evidence / "aggregate.csv", aggregate_schema())
    _require(bool(per_image), "per_image.csv has no rows")
    _require(bool(aggregates), "aggregate.csv has no rows")

    _require(
        hashlib.sha256((evidence / "per_image.csv").read_bytes()).hexdigest()
        == summary.get("per_image_csv_sha256"),
        "per_image.csv does not match the hash recorded in smoke_summary.json",
    )
    _require(
        hashlib.sha256((evidence / "aggregate.csv").read_bytes()).hexdigest()
        == summary.get("aggregate_csv_sha256"),
        "aggregate.csv does not match the hash recorded in smoke_summary.json",
    )

    systems = set(get("artifacts.system_values"))
    identities: set[tuple[str, str]] = set()
    for row in per_image:
        _require(
            row["split"] != "test",
            f"per-image row {row['stable_sample_id']} claims split=test",
        )
        _require(
            row["dataset"] == FROZEN_CLASSIFIER_DATASET,
            f"a task-scored per-image row is for {row['dataset']!r}; only "
            f"{FROZEN_CLASSIFIER_DATASET!r} may be scored with the frozen classifier",
        )
        key = (row["run_id"], row["pair_id"])
        _require(key not in identities, f"duplicate per-image identity {key}")
        identities.add(key)
        outage_flag = _bool(row["outage"], "outage")
        correct = _bool(row["correct"], "correct")
        reason = row["outage_reason"]
        if outage_flag:
            _require(
                reason
                in (STRUCTURAL_INFEASIBILITY, CODEC_INFEASIBILITY, DECODE_FAILURE),
                f"outage row carries an invalid reason {reason!r}",
            )
            _require(
                int(row["pred_label"]) == policy.selected_class,
                "an outage row does not carry the frozen constant prediction",
            )
            _require(
                correct == (int(row["true_label"]) == policy.selected_class),
                "an outage row's correctness does not follow from the frozen class",
            )
        else:
            _require(reason == "", "a delivered row carries an outage reason")

    for aggregate in aggregates:
        _require(
            aggregate["system"] in systems,
            f"aggregate row carries an unsupported system {aggregate['system']!r}",
        )
        _require(aggregate["split"] != "test", "an aggregate row claims split=test")
        _require(
            aggregate["checkpoint_id"] == EXPECTED_CHECKPOINT_SHA256,
            "an aggregate row records a checkpoint other than the frozen G-1 one",
        )
        matching = [
            row
            for row in per_image
            if row["run_id"] == aggregate["run_id"]
        ]
        _require(
            bool(matching),
            f"aggregate run_id {aggregate['run_id'][:12]} has no per-image rows",
        )
        _reconcile(aggregate, matching)
    return {"per_image": per_image, "aggregates": aggregates}


def _reconcile(aggregate: dict[str, str], rows: list[dict[str, str]]) -> None:
    """Recompute every rate from the rows themselves."""

    n = len(rows)
    n_correct = sum(1 for row in rows if _bool(row["correct"], "correct"))
    delivered = [row for row in rows if not _bool(row["outage"], "outage")]
    reasons = [row["outage_reason"] for row in rows if _bool(row["outage"], "outage")]
    decode = reasons.count(DECODE_FAILURE)
    structural = reasons.count(STRUCTURAL_INFEASIBILITY)
    codec = reasons.count(CODEC_INFEASIBILITY)
    _require(
        len(delivered) + decode + structural + codec == n,
        "verdict counts do not sum to the row count",
    )

    expected: dict[str, Any] = {
        "n": float(n),
        "n_test": float(n),
        "n_correct": float(n_correct),
        "top1_acc": n_correct / n,
        "coverage_rate": len(delivered) / n,
        "decode_failure_rate": decode / n,
        "infeasible_rate": (structural + codec) / n,
    }
    for field, value in expected.items():
        actual = _optional_float(aggregate[field])
        _require(
            actual is not None and abs(actual - value) < 1e-12,
            f"aggregate {field}={aggregate[field]!r} does not reconcile with its "
            f"{n} rows (expected {value})",
        )
    delivered_correct = sum(1 for row in delivered if _bool(row["correct"], "correct"))
    actual = _optional_float(aggregate["acc_given_delivery"])
    if delivered:
        _require(
            actual is not None
            and abs(actual - delivered_correct / len(delivered)) < 1e-12,
            "aggregate acc_given_delivery does not reconcile with its delivered rows",
        )
    else:
        _require(
            actual is None,
            "acc_given_delivery must use the null representation with no delivered rows",
        )
        _require(
            aggregate["psnr_db"] == "" and aggregate["ssim"] == "",
            "PSNR/SSIM must be null when nothing was delivered",
        )


def check_cifar_separation(summary: dict[str, Any]) -> int:
    cifar = summary.get("cifar10_transport_only") or {}
    _require(bool(cifar), "smoke_summary.json has no cifar10_transport_only section")
    _require(
        "transport-only plumbing smoke" in str(cifar.get("declaration", "")),
        "the CIFAR-10 section does not carry the transport-only declaration",
    )
    _require(
        cifar.get("classifier_inference_performed") is False,
        "the CIFAR-10 smoke reports frozen-classifier inference",
    )
    for field in ("task_accuracy", "top1_acc", "n_correct"):
        _require(
            cifar.get(field) is None,
            f"the CIFAR-10 transport-only smoke reports {field}, which would mean "
            "the Imagenette-160 classifier was applied across class vocabularies",
        )
    _require(
        cifar.get("encode_axis_px") in get("baseline.downsample_axis_px")["cifar10"],
        "the CIFAR-10 smoke used an unconfigured encode axis",
    )
    task = summary.get("imagenette160_task_scored") or {}
    _require(bool(task), "smoke_summary.json has no imagenette160_task_scored section")
    _require(
        task.get("dataset") == FROZEN_CLASSIFIER_DATASET,
        "the task-scored section is not Imagenette-160",
    )
    for cell in task.get("cells", ()):
        _require(
            cell.get("dataset") == FROZEN_CLASSIFIER_DATASET,
            "a task-scored cell is not Imagenette-160",
        )
    return int(cifar.get("sample_count", 0))


def _current_bytes(path: str) -> bytes:
    """Read a bound source as it exists now.

    Its own function so the drift branch below can be exercised directly by a
    test, rather than only when a real file happens to be dirty.
    """

    return (REPO / path).read_bytes()


def check_sources(evidence: Path, summary: dict[str, Any]) -> dict[str, Any]:
    manifest = _json(evidence / "execution_source_manifest.json")
    commit = manifest.get("execution_source_commit")
    _require(
        commit == summary.get("execution_source_commit"),
        "the source manifest and the summary name different execution commits",
    )
    try:
        _git("cat-file", "-e", f"{commit}^{{commit}}")
    except VerificationError:
        raise VerificationError(f"execution commit is not reachable: {commit}") from None

    recorded = {entry["path"]: entry for entry in manifest.get("sources", ())}
    missing = sorted(set(EXPECTED_SOURCES) - set(recorded))
    unexpected = sorted(set(recorded) - set(EXPECTED_SOURCES))
    _require(
        not missing and not unexpected,
        f"bound source set differs: missing={missing}, unexpected={unexpected}",
    )
    for path, role in sorted(EXPECTED_SOURCES.items()):
        entry = recorded[path]
        _require(entry["role"] == role, f"{path} is bound with role {entry['role']!r}")
        at_commit = git_bytes(commit, path)
        _require(
            sha256_bytes(at_commit) == entry["sha256"],
            f"{path} bytes at {commit[:12]} differ from the manifest",
        )
        _require(
            entry["source_commit"] == commit,
            f"{path} is bound to a different commit",
        )
        if role != "record":
            current = _current_bytes(path)
            _require(
                sha256_bytes(current) == entry["sha256"],
                f"{path} has drifted since the bounded evidence was produced; "
                "rerun the bounded run rather than regenerating the manifest",
            )
    return manifest


_RUN_CONFIG_INDEX_FIELDS = (
    "config_hash",
    "relative_path",
    "file_sha256",
    "run_config",
    "dataset",
    "bw_ratio",
    "test_snr_db",
    "train_seed",
    "channel_seed",
    "modulation",
    "ldpc_rate",
    "encode_axis_px",
)


def check_configuration(
    evidence: Path, resolved: dict[str, Any], summary: dict[str, Any]
) -> dict[str, str]:
    """Every emitted cell must have a concrete, reproducible `RunConfig`.

    PB_2 resolved one configuration at 18 dB and reused its hash for the -8 dB
    cell, both fixtures and the CIFAR-10 rows.  The old check could not see it,
    because it compared two files that both carried the same wrong hash.  This
    instead reconstructs each archived configuration through the ordinary
    `RunConfig.from_dict`/`config_hash` pair and requires the hash to *come out*
    of the configuration rather than to be recorded next to it.
    """

    _require(
        summary.get("checkpoint_id") == EXPECTED_CHECKPOINT_SHA256,
        "the evidence records a checkpoint other than the frozen G-1 one",
    )
    _require(
        summary.get("classifier_config_hash") == EXPECTED_CONFIG_HASH,
        "the evidence records a classifier config hash other than the frozen one",
    )

    index = resolved.get("run_configs")
    _require(
        isinstance(index, list) and index,
        "resolved_config.json carries no run-config index",
    )
    roots = get("config.fingerprint_parameter_roots")
    hashes: dict[str, str] = {}
    for entry in index:
        _require(
            isinstance(entry, dict)
            and set(entry) == set(_RUN_CONFIG_INDEX_FIELDS),
            f"a run-config index entry does not carry exactly "
            f"{sorted(_RUN_CONFIG_INDEX_FIELDS)}",
        )
        digest = entry["config_hash"]
        path = evidence / entry["relative_path"]
        _require(
            path.is_file(),
            f"the archived run config {entry['relative_path']} is missing",
        )
        body = path.read_bytes()
        _require(
            hashlib.sha256(body).hexdigest() == entry["file_sha256"],
            f"the archived run config {entry['relative_path']} does not match its "
            "recorded file hash",
        )
        try:
            run_config = RunConfig.from_dict(json.loads(body))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise VerificationError(
                f"{entry['relative_path']} does not reconstruct as a RunConfig: {exc}"
            ) from None
        _require(
            config_hash(run_config) == digest,
            f"{entry['relative_path']} does not reproduce its own config hash",
        )
        _require(
            path.name == f"{digest}.json",
            "an archived run config is not stored under its own config hash",
        )
        for field in (
            "dataset",
            "bw_ratio",
            "test_snr_db",
            "train_seed",
            "channel_seed",
            "modulation",
            "ldpc_rate",
            "encode_axis_px",
        ):
            _require(
                run_config.resolved[field] == entry[field],
                f"{entry['relative_path']}: the index records {field}={entry[field]!r} "
                f"but the configuration resolves {run_config.resolved[field]!r}",
            )
        snapshot = run_config.to_dict()["parameters"]
        _require(
            set(snapshot) == set(roots),
            f"{entry['relative_path']}: the parameter snapshot does not cover the "
            "fingerprint roots",
        )
        for root in roots:
            _require(
                snapshot[root] == get(root),
                f"{entry['relative_path']}: the recorded params.{root} snapshot "
                "differs from the current spec",
            )
        _require(
            digest not in hashes,
            f"config hash {digest[:12]} is archived twice",
        )
        hashes[digest] = entry["relative_path"]

    recorded = summary.get("config_hashes")
    _require(
        isinstance(recorded, dict) and set(recorded.values()) == set(hashes),
        "smoke_summary.json and resolved_config.json disagree on the cell config hashes",
    )
    _require(
        len(set(recorded.values())) == len(recorded),
        "two bounded cells share one config hash",
    )
    root_digest = summary.get("config_hash_root")
    _require(
        root_digest == resolved.get("config_hash_root"),
        "resolved_config.json and smoke_summary.json disagree on config_hash_root",
    )
    _require(
        canonical_sha256(sorted(hashes)) == root_digest,
        "config_hash_root does not reproduce from the archived cell config hashes",
    )
    _require(
        root_digest not in hashes,
        "the execution-level config_hash_root is being used as a cell config hash",
    )
    _require(
        summary.get("plan_sha256") == resolved.get("plan_sha256")
        and summary["plan_sha256"] not in hashes,
        "the execution-plan hash is being used as a run-config hash",
    )

    # Coverage is compared as a set: the artifact is written with sorted JSON
    # keys for byte determinism, and field *order* is a CSV contract, enforced
    # against the emitted headers rather than against this annotation map.
    semantics = resolved.get("field_semantics") or {}
    for section, schema in (
        ("aggregate", aggregate_schema()),
        ("per_image", per_image_schema()),
    ):
        annotated = set(semantics.get(section, {}))
        missing = sorted(set(schema) - annotated)
        unexpected = sorted(annotated - set(schema))
        _require(
            not missing and not unexpected,
            f"the committed field-semantics table does not cover {section} "
            f"exactly: missing={missing}, unexpected={unexpected}",
        )
    return hashes
    # Coverage is compared as a set: the artifact is written with sorted JSON
    # keys for byte determinism, and field *order* is a CSV contract, enforced
    # against the emitted headers rather than against this annotation map.
    semantics = resolved.get("field_semantics") or {}
    for section, schema in (
        ("aggregate", aggregate_schema()),
        ("per_image", per_image_schema()),
    ):
        annotated = set(semantics.get(section, {}))
        missing = sorted(set(schema) - annotated)
        unexpected = sorted(annotated - set(schema))
        _require(
            not missing and not unexpected,
            f"the committed field-semantics table does not cover {section} "
            f"exactly: missing={missing}, unexpected={unexpected}",
        )


def check_raw_rows(
    evidence: Path, summary: dict[str, Any], cell_hashes: dict[str, str]
) -> list[dict[str, Any]]:
    """Read the final raw rows and recompute what the CSVs only assert.

    The partial file is execution state. Accepting it as evidence would let a
    truncated run pass as a finished one, so the *final* artifact is required
    and is re-hashed against the summary before anything is derived from it.
    """

    path = evidence / "smoke_rows.jsonl"
    _require(path.is_file(), "missing final raw-row artifact: smoke_rows.jsonl")
    body = path.read_bytes()
    _require(
        hashlib.sha256(body).hexdigest() == summary.get("raw_rows_sha256"),
        "smoke_rows.jsonl does not match the hash recorded in smoke_summary.json",
    )
    try:
        rows = [
            json.loads(line)
            for line in body.decode("utf-8").splitlines()
            if line.strip()
        ]
    except json.JSONDecodeError as exc:
        raise VerificationError(f"smoke_rows.jsonl is not valid JSONL: {exc}") from None
    _require(
        len(rows) == summary.get("raw_rows_count"),
        f"smoke_rows.jsonl holds {len(rows)} rows, the summary declares "
        f"{summary.get('raw_rows_count')}",
    )
    work_ids = [row["work_id"] for row in rows]
    _require(
        len(set(work_ids)) == len(work_ids),
        "smoke_rows.jsonl contains a duplicated work item",
    )
    _require(
        summary.get("worklist_sha256")
        == canonical_sha256(work_ids),
        "smoke_rows.jsonl is not the committed worklist in its deterministic order",
    )

    for row in rows:
        _require(
            row.get("config_hash") in cell_hashes,
            f"raw row {row['work_id'][:12]} names a config hash that is not archived",
        )
        elapsed = row.get("wall_clock_s")
        _require(
            isinstance(elapsed, int | float) and elapsed >= 0,
            f"raw row {row['work_id'][:12]} has no non-negative elapsed time",
        )
        # Three separate facts. An infeasible row keeps its schedule and is
        # honest that nothing was drawn; a transmitted row must reconcile.
        scheduled = row.get("scheduled_noise_id")
        _require(
            isinstance(scheduled, str) and len(scheduled) == 64,
            f"raw row {row['work_id'][:12]} carries no scheduled noise identity",
        )
        consumed = row.get("noise_consumed")
        _require(
            consumed is True or consumed is False,
            f"raw row {row['work_id'][:12]} does not state whether noise was consumed",
        )
        actual = row.get("actual_noise_id")
        if row["verdict"] in (STRUCTURAL_INFEASIBILITY, CODEC_INFEASIBILITY):
            _require(
                actual is None and consumed is False,
                f"raw row {row['work_id'][:12]} is {row['verdict']} but claims a "
                "channel realisation was drawn",
            )
        else:
            _require(
                consumed is True and actual == scheduled,
                f"raw row {row['work_id'][:12]} transmitted but its realised "
                "identity does not reconcile with the scheduled one",
            )

    total = sum(row["wall_clock_s"] for row in rows)
    _require(
        abs(total - float(summary.get("wall_clock_s", -1))) <= 1e-9,
        "the summary wall clock is not the sum of the durable row timings; a "
        "resumed session's elapsed time is not the bounded run's total",
    )
    _require(
        summary.get("openjpeg_version") == get("environment.openjpeg"),
        "the evidence records an OpenJPEG version other than the configured one",
    )
    _require(
        summary.get("openjpeg_preflight_preceded_artifacts") is True,
        "the evidence does not record that the OpenJPEG preflight preceded "
        "artifact creation",
    )
    return rows


def check_byte_accounting(
    evidence: Path, raw_rows: list[dict[str, Any]], aggregates: list[dict[str, str]]
) -> dict[str, int]:
    """Independently recompute the BR-11 columns from the raw codestreams.

    BR-11/AM-81 aggregates over every row that emitted a codestream, delivered
    and decode-failure alike. PB_2 averaged delivered rows only, so a cell whose
    rows all failed to decode reported no overhead at all.
    """

    emitted = 0
    for row in raw_rows:
        source_coding = row.get("source_coding") or {}
        header = source_coding.get("header_bytes")
        payload = source_coding.get("payload_bytes")
        if source_coding.get("emitted_bytes") is None:
            _require(
                header is None and payload is None,
                f"raw row {row['work_id'][:12]} reports byte columns without a "
                "codestream",
            )
            continue
        emitted += 1
        _require(
            isinstance(header, int) and isinstance(payload, int),
            f"raw row {row['work_id'][:12]} emitted a codestream but carries no "
            "header/payload split",
        )
        _require(
            header + payload == source_coding["emitted_bytes"],
            f"raw row {row['work_id'][:12]}: header + payload != emitted bytes",
        )
        filler = source_coding.get("payload_filler_bytes")
        accounting = (row.get("summary") or {}).get("accounting") or {}
        bytes_sent = accounting.get("payload_bytes")
        if filler is not None and bytes_sent is not None:
            _require(
                header + payload + filler == bytes_sent,
                f"raw row {row['work_id'][:12]}: header + payload + filler != "
                "bytes_sent",
            )
            _require(
                filler == 0 or payload != bytes_sent,
                f"raw row {row['work_id'][:12]}: payload filler is folded into "
                "the BR-11 payload column",
            )

    # And the aggregate means must follow from those rows, per cell.
    by_hash: dict[str, list[dict[str, Any]]] = {}
    for row in raw_rows:
        by_hash.setdefault(row["config_hash"], []).append(row)
    for aggregate in aggregates:
        rows = by_hash.get(aggregate["config_hash"], [])
        values = [
            (row["source_coding"]["header_bytes"], row["source_coding"]["payload_bytes"])
            for row in rows
            if (row.get("source_coding") or {}).get("header_bytes") is not None
        ]
        for column, index in (("header_bytes", 0), ("payload_bytes", 1)):
            recorded = _optional_float(aggregate[column])
            if not values:
                _require(
                    recorded is None,
                    f"aggregate {aggregate['config_hash'][:12]} reports "
                    f"{column} with no emitted codestream behind it",
                )
                continue
            _require(
                recorded is not None,
                f"aggregate {aggregate['config_hash'][:12]} leaves {column} blank "
                f"while {len(values)} of its rows emitted a codestream",
            )
            expected = sum(value[index] for value in values) / len(values)
            _require(
                abs(recorded - expected) <= 1e-9,
                f"aggregate {aggregate['config_hash'][:12]}: {column} is "
                f"{recorded}, recomputes to {expected} over {len(values)} "
                "emitted-codestream rows",
            )
    return {"emitted_rows": emitted}


def check_identities(
    raw_rows: list[dict[str, Any]],
    index: list[dict[str, Any]],
    aggregates: list[dict[str, str]],
    per_image_rows: list[dict[str, str]],
) -> None:
    """Recompute every identity rather than trusting the row that carries it.

    PB_2's verifier never rebuilt `noise_id`, `pair_id`, `analysis_cell_id` or
    `run_id`, so a null realisation on an infeasible row -- which silently
    removes that image from the paired comparison ER-10 exists to make -- was
    invisible to it.
    """

    by_hash = {entry["config_hash"]: entry for entry in index}

    for row in raw_rows:
        per = row.get("per_image")
        if per is None:
            continue
        entry = by_hash[row["config_hash"]]
        scheduled = row["scheduled_noise_id"]

        _require(
            per["noise_id"] == scheduled,
            f"raw row {row['work_id'][:12]}: the emitted per-image row does not "
            "carry the scheduled noise identity",
        )
        _require(
            len(per["noise_id"]) == 64,
            f"per-image row {per['stable_sample_id']} carries no noise identity; "
            "a row that did not transmit still has a scheduled one",
        )
        expected_noise = make_noise_id(
            {
                "dataset_version": per["dataset_version"],
                "split_manifest_hash": manifest_sha256(per["dataset"]),
                "stable_sample_id": per["stable_sample_id"],
                "test_snr_db": float(per["test_snr_db"]),
                "channel_seed": int(entry["channel_seed"]),
                "channel": "awgn",
                "k": int(row["k_symbols"]),
                "block_index": 0,
                "rng_purpose": "channel_noise",
            }
        )
        _require(
            per["noise_id"] == expected_noise,
            f"per-image row {per['stable_sample_id']}: noise_id does not recompute "
            "from the scheduled evaluation cell",
        )
        expected_cell = make_analysis_cell_id(
            {
                "train_seed": int(entry["train_seed"]),
                "channel_seed": int(entry["channel_seed"]),
            }
        )
        _require(
            per["analysis_cell_id"] == expected_cell,
            f"per-image row {per['stable_sample_id']}: analysis_cell_id does not "
            "recompute from the archived configuration",
        )
        expected_pair = make_pair_id(
            {
                "analysis_cell_id": expected_cell,
                "stable_sample_id": per["stable_sample_id"],
                "bw_ratio": per["bw_ratio"],
                "test_snr_db": float(per["test_snr_db"]),
                "noise_id": per["noise_id"],
            }
        )
        _require(
            per["pair_id"] == expected_pair,
            f"per-image row {per['stable_sample_id']}: pair_id does not recompute "
            "from its own scheduled noise identity",
        )

    # A raw file that dropped a row would otherwise be self-consistent, because
    # its own count and worklist digest are recomputed from what remains. The
    # per-image CSV is the independent witness.
    raw_samples = {
        (row["per_image"]["run_id"], row["per_image"]["stable_sample_id"])
        for row in raw_rows
        if row.get("per_image") is not None
    }
    csv_samples = {
        (row["run_id"], row["stable_sample_id"]) for row in per_image_rows
    }
    _require(
        raw_samples == csv_samples,
        "per_image.csv and smoke_rows.jsonl describe different rows: "
        f"only in the CSV={sorted(sample for sample in csv_samples - raw_samples)}, "
        f"only in the raw rows={sorted(sample for sample in raw_samples - csv_samples)}",
    )

    for aggregate in aggregates:
        entry = by_hash.get(aggregate["config_hash"])
        _require(
            entry is not None,
            f"aggregate row names config hash {aggregate['config_hash'][:12]}, "
            "which is not archived",
        )
        expected_run = make_run_id(
            {
                "system": aggregate["system"],
                "dataset": aggregate["dataset"],
                "dataset_version": get(
                    f"datasets.{aggregate['dataset']}."
                    f"{get('config.dataset_version_rule')}"
                ),
                "split": aggregate["split"],
                "split_manifest_hash": manifest_sha256(aggregate["dataset"]),
                "bw_ratio": aggregate["bw_ratio"],
                "test_snr_db": float(aggregate["test_snr_db"]),
                "train_seed": int(aggregate["train_seed"]),
                "channel_seed": int(aggregate["channel_seed"]),
                "config_hash": aggregate["config_hash"],
                "checkpoint_id": aggregate["checkpoint_id"],
                "classifier_variant": aggregate["classifier_variant"],
                "ldpc_rate": aggregate["ldpc_rate"],
                "modulation": aggregate["modulation"],
                "quantiser_bits": None,
                "transmit_dim": None,
                "lambda": None,
                "analysis_version": get("config.analysis_version"),
            }
        )
        _require(
            aggregate["run_id"] == expected_run,
            f"aggregate {aggregate['config_hash'][:12]}: run_id does not recompute "
            "from its own recorded selections",
        )
        for field in ("test_snr_db", "bw_ratio", "modulation", "ldpc_rate"):
            recorded = aggregate[field]
            expected = entry[field]
            if field == "test_snr_db":
                recorded, expected = float(recorded), float(expected)
            _require(
                recorded == expected,
                f"aggregate {aggregate['config_hash'][:12]}: {field} is {recorded!r} "
                f"but its configuration resolves {expected!r}",
            )


def check_overhead_table(
    evidence: Path, raw_rows: list[dict[str, Any]]
) -> int:
    """BR-11's archived overhead table, recomputed and scope-checked.

    BR-11 has always required this artifact. It is bounded evidence, so it must
    say so: a table that silently looked like the full validation grid would be
    the BR-4 sweep claim PB_2C forbids.
    """

    table = _json(evidence / "overhead_table.json")
    _require(_labels_ok(table), "overhead_table.json carries the wrong evidence labels")
    _require(
        table.get("evidence_scope") == "bounded_integration",
        "overhead_table.json does not declare its bounded scope",
    )
    _require(
        table.get("complete_for_full_validation_grid") is False,
        "overhead_table.json claims to cover the full validation grid",
    )
    cells = table.get("cells")
    _require(
        isinstance(cells, list) and cells,
        "overhead_table.json lists no executed cells",
    )
    _require(
        table.get("executed_cells") == len(cells),
        "overhead_table.json miscounts its own cells",
    )

    observed: dict[tuple, list[dict[str, Any]]] = {}
    for row in raw_rows:
        source_coding = row.get("source_coding") or {}
        if source_coding.get("emitted_bytes") is None:
            continue
        per = row.get("per_image") or {}
        key = (
            row["config_hash"],
            per.get("stable_sample_id"),
        )
        observed.setdefault(row["config_hash"], []).append(source_coding)

    declared = {cell["config_hash"] for cell in cells}
    _require(
        declared <= set(observed),
        "overhead_table.json names a cell no row executed; unexecuted "
        f"combinations must not be synthesised: {sorted(declared - set(observed))}",
    )
    _require(
        set(observed) <= declared,
        "overhead_table.json omits an executed cell: "
        f"{sorted(set(observed) - declared)}",
    )

    for cell in cells:
        rows = observed[cell["config_hash"]]
        _require(
            cell["emitted_rows"] == len(rows),
            f"overhead cell {cell['config_hash'][:12]} claims "
            f"{cell['emitted_rows']} emitted rows, {len(rows)} were emitted",
        )
        for column, field in (
            ("mean_header_bytes", "header_bytes"),
            ("mean_payload_bytes", "payload_bytes"),
        ):
            expected = sum(row[field] for row in rows) / len(rows)
            _require(
                abs(cell[column] - expected) <= 1e-9,
                f"overhead cell {cell['config_hash'][:12]}: {column} is "
                f"{cell[column]}, recomputes to {expected}",
            )
        total = cell["mean_header_bytes"] + cell["mean_payload_bytes"]
        _require(
            abs(cell["mean_emitted_codestream_bytes"] - total) <= 1e-9
            and abs(cell["overhead_fraction_of_emitted"]
                    - cell["mean_header_bytes"] / total) <= 1e-9,
            f"overhead cell {cell['config_hash'][:12]} does not reconcile",
        )
    return len(cells)


def check_integration_adjudication(evidence: Path) -> dict[str, Any]:
    """Verify the W4 closing adjudication by recomputing what it claims.

    Everything structural is re-derived: the bound evidence hashes from the
    files on disk, the selection-machinery description from the module itself,
    the worked composition example from the real composition function, and the
    characterised BLER identities from the committed G-2 curves. A field that
    merely states a number is never accepted as evidence of it.
    """

    path = evidence / ADJUDICATION_FILE
    _require(path.is_file(), f"missing W4 evidence file: {ADJUDICATION_FILE}")
    payload = _json(path)

    _require(
        payload.get("schema_version") == ADJUDICATION_SCHEMA_VERSION,
        f"{ADJUDICATION_FILE} has an unexpected schema_version",
    )
    _require(
        _labels_ok(payload), f"{ADJUDICATION_FILE} carries the wrong evidence labels"
    )
    declaration = str(payload.get("prominent_declaration", "")).lower()
    for phrase in (
        "bounded validation/plumbing integration",
        "not the br-4 full validation sweep",
        "not a g-8 operating-point selection",
        "not test evidence",
    ):
        _require(
            phrase in declaration,
            f"{ADJUDICATION_FILE} does not state {phrase!r} in prose",
        )

    claims = payload.get("claims") or {}
    _require(
        claims.get("bounded_validation_plumbing_integration") is True,
        f"{ADJUDICATION_FILE} does not declare itself bounded integration",
    )
    for field in (
        "br4_full_validation_sweep",
        "g8_operating_point_selection",
        "g8_characterization_started",
        "test_evidence",
        "training_performed",
        "artifact_finetuning_performed",
        "lambda_calibration_performed",
        "er9_implemented",
    ):
        _require(
            claims.get(field) is False,
            f"{ADJUDICATION_FILE} claims {field}",
        )
    _require(
        claims.get("g8_status") == "unresolved",
        f"{ADJUDICATION_FILE} does not record G-8 as unresolved",
    )
    _require(
        claims.get("test_split_sealed") is True,
        f"{ADJUDICATION_FILE} does not declare the test split sealed",
    )
    remaining = payload.get("remaining") or {}
    _require(
        remaining.get("full_br4_sweep_started") is False
        and remaining.get("g8_started") is False,
        f"{ADJUDICATION_FILE} claims the BR-4 sweep or G-8 has started",
    )

    blob = json.dumps(payload, sort_keys=True).lower()
    for claim in FORBIDDEN_CLAIMS:
        _require(claim not in blob, f"{ADJUDICATION_FILE} claims {claim!r}")

    _require(
        "evidence_commit" not in payload,
        "the adjudication records an evidence_commit; a file cannot contain the "
        "hash of the commit that adds it, so the commit is resolved from Git "
        "path history and never stored",
    )
    _require(
        bool(str(payload.get("evidence_commit_resolution", "")).strip()),
        f"{ADJUDICATION_FILE} states no evidence-commit resolution policy",
    )

    # Bound evidence hashes, recomputed from disk.
    recorded_files = payload.get("evidence_files") or {}
    missing = sorted(set(BOUND_EVIDENCE_FILES) - set(recorded_files))
    unexpected = sorted(set(recorded_files) - set(BOUND_EVIDENCE_FILES))
    _require(
        not missing and not unexpected,
        f"bound evidence set differs: missing={missing}, unexpected={unexpected}",
    )
    for name, digest in sorted(recorded_files.items()):
        actual = sha256_bytes((evidence / name).read_bytes())
        _require(
            actual == digest,
            f"{name} does not match the hash bound by {ADJUDICATION_FILE}",
        )

    # Selection sources, re-hashed at HEAD.
    sources = {entry["path"]: entry for entry in payload.get("selection_sources", ())}
    missing = sorted(set(SELECTION_SOURCES) - set(sources))
    unexpected = sorted(set(sources) - set(SELECTION_SOURCES))
    _require(
        not missing and not unexpected,
        f"bound selection sources differ: missing={missing}, unexpected={unexpected}",
    )
    for source_path, role in sorted(SELECTION_SOURCES.items()):
        entry = sources[source_path]
        _require(
            entry.get("role") == role,
            f"{source_path} is bound with role {entry.get('role')!r}",
        )
        current = (REPO / source_path).read_bytes()
        _require(
            sha256_bytes(current) == entry.get("sha256")
            and len(current) == entry.get("bytes"),
            f"{source_path} has drifted since the adjudication was written; "
            f"regenerate it with tools/gen_w4_integration_adjudication.py",
        )
    _require(
        "src/baseline/classical/composition.py" not in EXPECTED_SOURCES,
        "the selection module must not be bound into the bounded run's "
        "execution manifest: it postdates that measurement",
    )

    # The worked example, recomputed by the real composition function.
    recorded_example = (payload.get("composition") or {}).get("worked_example")
    _require(
        recorded_example == worked_composition_example(),
        "the adjudication's worked composition example does not reproduce; the "
        "composition arithmetic or a measured input has changed",
    )

    # The BLER characterisation, re-derived from the committed curves.
    _require(
        payload.get("bler_characterization") == bler_characterization(),
        "the adjudication's BLER characterisation does not match the committed "
        "G-2 curves",
    )
    characterization = payload["bler_characterization"]
    _require(
        characterization.get("extrapolation_permitted") is False
        and characterization.get("absent_evidence_treated_as_zero_bler") is False,
        "the adjudication permits extrapolation or zero-BLER defaults",
    )

    # The selection machinery and sweep guard, read back out of the module.
    _require(
        payload.get("selection_machinery") == selection_machinery(),
        "the adjudication's selection-machinery description does not match "
        "src/baseline/classical/composition.py",
    )
    _require(
        payload["selection_machinery"]["passes_executed"] == 0,
        "the adjudication records selection passes as executed",
    )

    # ------------------------------------------------------------------
    # The pre-G-8 selection-policy freeze, checked by name rather than only
    # through the whole-dict equality above.  A named failure says which rule
    # moved; "the description does not match" does not.
    # ------------------------------------------------------------------
    machinery = payload["selection_machinery"]
    _require(
        list(machinery.get("tie_break_order") or [])
        == list(composition.TIE_BREAK_ORDER),
        "the adjudication's tie-break order is not the implementation's: "
        f"recorded {machinery.get('tie_break_order')!r}, "
        f"module {list(composition.TIE_BREAK_ORDER)!r}",
    )
    _require(
        machinery.get("tie_break_frozen_before_g8") is True,
        "the adjudication does not declare the tie-break order frozen before "
        "G-8; a selection rule that may be revised after the validation table "
        "exists is not preregistered",
    )
    _require(
        machinery.get("tie_break_frozen_against_gate")
        == SELECTION_POLICY_FREEZE_GATE,
        "the adjudication freezes the tie-break order against the wrong gate: "
        f"recorded {machinery.get('tie_break_frozen_against_gate')!r}, "
        f"expected {SELECTION_POLICY_FREEZE_GATE!r}",
    )
    _require(
        machinery.get("tie_equality") == TIE_EQUALITY,
        "the adjudication no longer defines a tie as exact equality; a "
        "tolerance parameter would make the tie-break order negotiable",
    )
    _require(
        machinery.get("resumed_state_validation") == "exact_ordered_prefix",
        "the adjudication does not record that resumed selection state is "
        "validated as an exact ordered prefix of the permitted passes",
    )

    # The fixed-modulation reference (BR-9), by name.
    fixed = machinery.get("fixed_modulation") or {}
    _require(
        fixed.get("source") == composition.CORE_MODULATION_SOURCE
        == "params.baseline.core_modulation",
        "the adjudication does not name params.baseline.core_modulation as the "
        "source of the fixed-modulation reference curve (BR-9)",
    )
    _require(
        fixed.get("configured_value") == composition.core_modulation(),
        "the adjudication records fixed modulation "
        f"{fixed.get('configured_value')!r}, but "
        f"{composition.CORE_MODULATION_SOURCE} resolves to "
        f"{composition.core_modulation()!r}",
    )
    _require(
        fixed.get("searches_modulations") is False,
        "the adjudication claims the fixed-modulation curve searches "
        "modulations; BR-9 makes it a reference, not a second optimizer",
    )

    # The fingerprint, recomputed here rather than read.  The generator and the
    # implementation could in principle be edited together; this digest is the
    # single value a future G-8 campaign manifest binds, so it is derived
    # independently from the named fields.
    _require(
        list(machinery.get("selection_policy_fields") or [])
        == list(SELECTION_POLICY_FIELDS),
        "the adjudication's selection-policy fingerprint covers different "
        f"fields: recorded {machinery.get('selection_policy_fields')!r}, "
        f"expected {list(SELECTION_POLICY_FIELDS)!r}",
    )
    _require(
        machinery.get("selection_policy_sha256")
        == selection_policy_fingerprint(machinery),
        "the adjudication's selection_policy_sha256 does not reproduce from "
        "the policy fields it covers; the frozen selection policy has moved",
    )
    _require(
        payload.get("sweep_guard") == sweep_guard(),
        "the adjudication's sweep-guard limits do not match the module",
    )

    # The three provisional operating points must be exactly where G-8 left them.
    provisional = payload.get("provisional_operating_points") or {}
    _require(
        sorted(provisional) == sorted(PROVISIONAL_OPERATING_POINTS),
        f"{ADJUDICATION_FILE} does not record all three provisional operating "
        "points",
    )
    for name in PROVISIONAL_OPERATING_POINTS:
        entry = provisional[name]
        _require(
            entry.get("value") == get(f"bandwidth.{name}"),
            f"the provisional {name} has moved: recorded {entry.get('value')!r}, "
            f"params says {get(f'bandwidth.{name}')!r}",
        )
        _require(
            entry.get("status") == get(f"bandwidth.{name}_status")
            == "provisional_until_G-8",
            f"{name} is no longer provisional_until_G-8",
        )
        _require(
            entry.get("selected_by_w4") is False,
            f"the adjudication claims W4 selected {name}",
        )

    # The outage measurement, cross-checked against the frozen artifact.
    outage = payload.get("outage") or {}
    frozen = _json(evidence / "outage_policy.json")
    for field in ("selected_class", "numerator", "denominator"):
        _require(
            outage.get(field) == frozen.get(field),
            f"the adjudication's outage {field} disagrees with outage_policy.json",
        )
    _require(
        outage.get("measured_validation_accuracy")
        == outage["numerator"] / outage["denominator"],
        "the adjudication's outage accuracy is not its own numerator/denominator",
    )
    _require(
        outage.get("assumed_uniform_accuracy_rejected") is True,
        "the adjudication does not reject the assumed 1/n_classes outage accuracy",
    )

    # Test-split counters, all zero.
    access = payload.get("test_split_access") or {}
    _require(
        access.get("test_split_sealed") is True
        and access.get("test_accessed") is False
        and access.get("test_inference") is False
        and access.get("test_accuracy_computed") is False,
        f"{ADJUDICATION_FILE} does not declare the test split sealed",
    )
    for counter in (
        "decoder_calls",
        "canonicalization_calls",
        "inference_calls",
        "accuracy_calls",
    ):
        _require(
            access.get(counter) == 0,
            f"{ADJUDICATION_FILE} records a non-zero {counter}",
        )
    _require(
        access.get("release_gate") == get("evaluation.test_access_gate"),
        "the adjudication names the wrong test-access gate",
    )

    _require(
        payload.get("bounded_evidence_execution_commit")
        == _json(evidence / "smoke_summary.json").get("execution_source_commit"),
        "the adjudication names a different bounded-run execution commit",
    )
    return payload


def check_selection_machinery_behaviour() -> None:
    """Exercise the selection module's fail-closed behaviours, live.

    The adjudication describes them; this proves they still hold at HEAD.  Each
    probe is unit-scale — no dataset, no codec, no channel — and the sweep-guard
    probe is deliberately the *refusing* path, so running the verifier can never
    start a sweep.
    """

    table = composition.g2_bler_table()
    identity = {
        "k_and_n": (128, 256),
        "base_graph": 2,
        "lifting_size": 22,
        "modulation": "qpsk",
        "decoder_algorithm": "offset_min_sum",
        "decoder_offset": 0.5,
        "iterations": 50,
        "snr_convention": "eb_n0_per_information_bit",
        "rate": "0.5",
    }

    partial = dict(identity)
    partial.pop("lifting_size")
    try:
        table.lookup(partial, 2.5)
    except composition.BlerLookupError:
        pass
    else:
        raise VerificationError("an incomplete BLER lookup key was accepted")

    wrong = dict(identity, base_graph=1)
    _require(
        not table.lookup(wrong, 2.5).characterized,
        "an uncharacterized BLER identity was treated as characterized",
    )
    outside = table.lookup(identity, 18.0)
    _require(
        not outside.characterized and outside.bler is None,
        "a BLER was extrapolated beyond the characterized support",
    )

    # No cross-configuration cache collision.
    base = composition.Candidate("imagenette160", "r_1_6", "qpsk", "1/2", 160, 12.0)
    for field, value in (
        ("dataset", "stl10"),
        ("ratio", "r_1_12"),
        ("modulation", "qam16"),
        ("ldpc_rate", "2/3"),
        ("encode_axis_px", 128),
    ):
        other = composition.Candidate(
            **{**{f: getattr(base, f) for f in base.__dataclass_fields__}, field: value}
        )
        _require(
            other.feasibility_key() != base.feasibility_key(),
            f"the feasibility cache key collides on {field}",
        )

    # Deterministic tie-breaking, independent of enumeration order.
    outage = composition.measured_outage_accuracy_from_record(
        _json(REPO / EVIDENCE_DIR / "outage_policy.json")
    )
    codec = composition.MeasuredCodecAccuracy(
        correct=870,
        total=1000,
        split=composition.SELECTION_SPLIT,
        source="verifier tie-break probe",
    )
    tied = [
        composition.CandidateEvaluation(
            candidate=composition.Candidate(
                "imagenette160", "r_1_6", modulation, "1/2", 160, 12.0
            ),
            status=composition.ELIGIBLE,
            composition=composition.compose(
                [0.01], codec_accuracy=codec, outage_accuracy=outage
            ),
        )
        for modulation in ("bpsk", "qpsk", "qam16")
    ]
    chosen = {
        composition.select_best(list(order)).selected.candidate
        for order in (tied, list(reversed(tied)), [tied[1], tied[2], tied[0]])
    }
    _require(len(chosen) == 1, "tie-breaking is not order-independent")

    # Two passes and then stop, and no pass-state leak.
    campaign = composition.SelectionCampaign(composition.CLASSICAL_ADAPTIVE)

    def _probe(context: Any) -> tuple[()]:
        try:
            context.result_of(context.pass_id)
        except composition.SelectionPassError:
            return ()
        raise VerificationError("a selection pass can read its own or a later pass")

    campaign.run_pass(1, _probe, scorer="clean")
    campaign.run_pass(2, _probe, scorer="artifact_finetuned")
    for third in (3, 1, 2):
        try:
            campaign.run_pass(third, _probe, scorer=f"extra-{third}")
        except composition.SelectionPassError:
            continue
        raise VerificationError(f"a further selection pass {third} was permitted")

    # BR-9's fixed-modulation reference curve holds the *configured* modulation
    # even when another one dominates the grid outright.  A recorded claim that
    # the curve does not search is not proof that it does not; this is.
    configured = composition.core_modulation()
    dominant = next(
        name for name in ("bpsk", "qam16", "qpsk") if name != configured
    )

    def _cell(modulation: str, correct: int, bler: float) -> Any:
        return composition.CandidateEvaluation(
            candidate=composition.Candidate(
                dataset="imagenette160",
                ratio="r_1_6",
                modulation=modulation,
                ldpc_rate="1/2",
                encode_axis_px=160,
                snr_db=12.0,
            ),
            status=composition.ELIGIBLE,
            composition=composition.compose(
                [bler], codec_accuracy=codec, outage_accuracy=outage
            ),
        )

    grid = {
        snr: [_cell(dominant, 990, 0.001), _cell(configured, 600, 0.50)]
        for snr in (-6.0, 18.0)
    }
    curve = composition.resolve_curve(composition.CLASSICAL_FIXED_MOD, grid)
    _require(
        curve.held_fixed.get("modulation") == configured,
        f"the fixed-modulation curve held {curve.held_fixed.get('modulation')!r} "
        f"rather than the configured {configured!r} (BR-9)",
    )
    _require(
        all(
            selection.selected.candidate.modulation == configured
            for _, selection in curve.per_snr
        ),
        "the fixed-modulation curve selected a non-core modulation at some SNR",
    )
    # ...and the adaptive curve really would have taken the other one, so the
    # check above is discriminating rather than vacuous.
    adaptive = composition.resolve_curve(composition.CLASSICAL_ADAPTIVE, grid)
    _require(
        all(
            selection.selected.candidate.modulation == dominant
            for _, selection in adaptive.per_snr
        ),
        "the fixed-modulation probe is not discriminating: the adaptive curve "
        f"did not prefer {dominant!r}",
    )

    # Resumed state is an exact ordered prefix of the permitted passes.
    allowed = composition.selection_passes()
    good = composition.SelectionCampaign(composition.CLASSICAL_ADAPTIVE)
    good.run_pass(allowed[0], _probe, scorer="clean")
    [stored_first] = good.state()
    resumable = composition.SelectionCampaign(
        composition.CLASSICAL_ADAPTIVE, completed=(stored_first,)
    )
    _require(
        resumable.completed_passes == (allowed[0],)
        and resumable.exhausted is False,
        "a valid pass-one resume was not admitted",
    )

    def _malformed(state: Any, what: str) -> None:
        try:
            composition.SelectionCampaign(
                composition.CLASSICAL_ADAPTIVE, completed=state
            )
        except composition.SelectionPassError:
            return
        raise VerificationError(f"resumed state admitted {what}")

    second = composition.PassResult(
        pass_id=allowed[-1],
        mode=composition.CLASSICAL_ADAPTIVE,
        scorer="artifact_finetuned",
        selections=stored_first.selections,
    )
    _malformed((second,), "pass two with no pass one")
    _malformed((second, stored_first), "a reversed pass sequence")
    _malformed(
        (
            stored_first,
            composition.PassResult(
                pass_id=allowed[-1],
                mode=composition.CLASSICAL_ADAPTIVE,
                scorer=stored_first.scorer,
                selections=stored_first.selections,
            ),
        ),
        "the same scorer reused across both passes",
    )
    _malformed(
        (
            composition.PassResult(
                pass_id=allowed[0],
                mode=composition.CLASSICAL_ADAPTIVE,
                scorer="clean",
                selections=("not a selection",),
            ),
        ),
        "a stored pass holding a malformed selection",
    )

    # The sweep guard refuses an unauthorized sweep-scale workload.
    budget = composition.sweep_budget(None)
    try:
        composition.check_sweep_budget(
            candidates=budget.max_candidates + 1, samples=1
        )
    except composition.SweepBudgetError:
        pass
    else:
        raise VerificationError("the full-sweep guard permitted an over-budget run")

    source = (REPO / "src/baseline/classical/composition.py").read_text()
    for hatch in ("os.environ", "getenv", "environ[", "putenv"):
        _require(
            hatch not in source,
            f"the selection module consults {hatch}: an environment variable "
            "must not be able to disarm the sweep guard",
        )

    # Assembled at runtime so this scanner does not match its own source.
    needle = "G8" + "Authorization" + "("
    tracked = _git("ls-files").split()
    offenders = [
        name
        for name in tracked
        if not name.startswith("tests/")
        and Path(name).suffix in {".py", ".json", ".yaml", ".yml", ".toml"}
        and needle in (REPO / name).read_text(errors="ignore")
    ]
    _require(
        not offenders,
        f"a G-8 sweep authorization is constructed in {offenders}",
    )


def check_gates() -> None:
    for tool in ("tools/verify_g1_adjudication.py", "tools/verify_g2_adjudication.py"):
        result = subprocess.run(
            [sys.executable, str(REPO / tool)],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        _require(
            result.returncode == 0,
            f"{tool} failed: {(result.stderr or result.stdout).strip()}",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", default=str(EVIDENCE_DIR))
    parser.add_argument("--skip-gates", action="store_true",
                        help="skip the G-1/G-2 sub-verifiers (used by mutation tests)")
    arguments = parser.parse_args()
    evidence = (REPO / arguments.evidence_dir).resolve()

    try:
        payloads = check_presence_and_labels(evidence)
        policy = check_outage_policy(payloads["outage"])
        records = check_records(evidence, policy, payloads["summary"])
        cifar_samples = check_cifar_separation(payloads["summary"])
        cell_hashes = check_configuration(
            evidence, payloads["resolved"], payloads["summary"]
        )
        raw_rows = check_raw_rows(evidence, payloads["summary"], cell_hashes)
        check_identities(
            raw_rows,
            payloads["resolved"]["run_configs"],
            records["aggregates"],
            records["per_image"],
        )
        byte_totals = check_byte_accounting(
            evidence, raw_rows, records["aggregates"]
        )
        overhead_cells = check_overhead_table(evidence, raw_rows)
        manifest = check_sources(evidence, payloads["summary"])
        adjudication = check_integration_adjudication(evidence)
        check_selection_machinery_behaviour()
        if not arguments.skip_gates:
            check_gates()
    except (VerificationError, composition.CompositionError) as exc:
        print(f"W4 baseline integration verification FAIL: {exc}", file=sys.stderr)
        return 1

    task = payloads["summary"]["imagenette160_task_scored"]
    cells = ", ".join(
        f"{cell['test_snr_db']}dB n={cell['n']} top1={cell['top1_acc']}"
        for cell in task["cells"]
    )
    print(
        "W4 baseline integration verification PASS: "
        f"execution={payloads['summary']['execution_source_commit'][:12]}, "
        f"sources={len(manifest['sources'])}, "
        f"per_image_rows={len(records['per_image'])}, "
        f"aggregate_rows={len(records['aggregates'])}, "
        f"outage_class={policy.selected_class} "
        f"({policy.selected_count}/{policy.validation_count}), "
        f"cifar_transport_only={cifar_samples} (no task score), "
        f"imagenette[{cells}], raw_rows={len(raw_rows)} "
        f"(emitted={byte_totals['emitted_rows']}), "
        f"openjpeg={payloads['summary']['openjpeg_version']}, "
        f"overhead_cells={overhead_cells}, "
        f"bler_curves={len(adjudication['bler_characterization']['characterized'])}, "
        f"selection_sources={len(adjudication['selection_sources'])}, "
        f"sweep_guard=<={adjudication['sweep_guard']['max_candidates']}"
        f"x{adjudication['sweep_guard']['max_samples']}"
        f"/{adjudication['sweep_guard']['max_workload']} unauthorized, "
        f"passes_executed=0, g8={adjudication['claims']['g8_status']}, "
        "test_split_access=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
