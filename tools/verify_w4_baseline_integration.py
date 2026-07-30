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

EVIDENCE_DIR = Path("results/baseline/w4")
REQUIRED_FILES = (
    "resolved_config.json",
    "outage_policy.json",
    "smoke_summary.json",
    "accounting_examples.json",
    "execution_source_manifest.json",
    "per_image.csv",
    "aggregate.csv",
)

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
        summary.get("j2k_resolutions_issue_status") == "unresolved",
        "the JPEG-2000 resolution issue is marked resolved without a spec decision",
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


def check_configuration(resolved: dict[str, Any], summary: dict[str, Any]) -> None:
    _require(
        resolved.get("config_hash") == summary.get("config_hash"),
        "resolved_config.json and smoke_summary.json disagree on config_hash",
    )
    _require(
        summary.get("checkpoint_id") == EXPECTED_CHECKPOINT_SHA256,
        "the evidence records a checkpoint other than the frozen G-1 one",
    )
    _require(
        summary.get("classifier_config_hash") == EXPECTED_CONFIG_HASH,
        "the evidence records a classifier config hash other than the frozen one",
    )
    parameters = resolved.get("parameters") or {}
    roots = get("config.fingerprint_parameter_roots")
    _require(
        set(parameters) == set(roots),
        "the recorded parameter snapshot does not cover the fingerprint roots",
    )
    for root in roots:
        _require(
            parameters[root] == get(root),
            f"the recorded params.{root} snapshot differs from the current spec",
        )
    semantics = resolved.get("field_semantics") or {}
    _require(
        tuple(semantics.get("aggregate", {})) == aggregate_schema()
        and tuple(semantics.get("per_image", {})) == per_image_schema(),
        "the committed field-semantics table does not cover both schemas exactly",
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
        check_configuration(payloads["resolved"], payloads["summary"])
        manifest = check_sources(evidence, payloads["summary"])
        if not arguments.skip_gates:
            check_gates()
    except VerificationError as exc:
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
        f"imagenette[{cells}], test_split_access=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
