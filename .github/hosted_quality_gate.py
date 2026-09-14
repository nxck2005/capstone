#!/usr/bin/env python3
"""Run CI without pretending worker-local transactional runtimes are portable.

The scientific terminal verifiers remain the worker-side authority.  Hosted CI
replaces only their exact runtime-consuming commands with checks over the
content-addressed evidence that is actually committed to Git.  Every other
quality-gate command is preserved verbatim.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import run_quality_gate as gate  # noqa: E402
from evaluation.downstream_v4 import (  # noqa: E402
    ER2_RUNTIME_ROOT,
    FINAL_PAIR,
    PRODUCTION_CELLS,
    read_json,
    sha256_file,
    validate_final_er9_cell,
    verify_er2_authority,
    verify_production_authority,
)
from training.deterministic_core import canonical_sha256  # noqa: E402
from verify_er2_randomized import verify_audit, verify_validation  # noqa: E402


ER9_CLOSEOUT = REPO / "results/learned/er9/er9_production_closeout_v4.json"
ER2_ROOT = REPO / "results/learned/er2_randomized"
ER2_COMPLETION = ER2_ROOT / "er2_randomized_completion_v4.json"


class HostedEvidenceHold(RuntimeError):
    """Committed evidence or hosted routing differs from the frozen contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HostedEvidenceHold(message)


def full_sha256(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and value != "0" * 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} is not a full non-sentinel SHA-256",
    )
    return value


def content_address(value: Mapping[str, Any], field: str, prefix: str, label: str) -> None:
    body = dict(value)
    identifier = body.pop(field, None)
    require(identifier == prefix + canonical_sha256(body), f"{label} content address differs")


def verify_er9_published() -> None:
    """Authenticate only immutable ER-9 evidence available in a clean clone."""

    authority = verify_production_authority(REPO)
    closeout = read_json(ER9_CLOSEOUT, "ER-9 production closeout")
    content_address(closeout, "closeout_id", "er9productionv4closeout-", "ER-9 closeout")
    require(closeout.get("schema_version") == 1, "ER-9 closeout schema differs")
    require(closeout.get("artifact_role") == "ER9_PRODUCTION_V4_CLOSEOUT", "ER-9 closeout role differs")
    require(closeout.get("status") == "THREE_REQUIRED_CELLS_COMPLETE_VALIDATION_ONLY", "ER-9 closeout status differs")
    require(closeout.get("authority_id") == authority["authority_id"], "ER-9 closeout authority differs")
    require(closeout.get("source_commit") == authority["source_commit"], "ER-9 closeout source differs")
    require(closeout.get("source_manifest_id") == authority["source_binding"]["manifest_id"], "ER-9 closeout manifest differs")
    require(closeout.get("selected_pair") == FINAL_PAIR, "ER-9 closeout pair differs")
    require(closeout.get("training_count") == 3, "ER-9 closeout training count differs")
    require(closeout.get("stage1_promotion_count") == 0, "ER-9 closeout promoted Stage 1")
    require(closeout.get("best_seed_selection") is False, "ER-9 closeout selected a best seed")
    records = closeout.get("seed_cells")
    require(
        isinstance(records, list)
        and [(item.get("train_seed"), item.get("channel_seed")) for item in records] == list(PRODUCTION_CELLS),
        "ER-9 closeout cell set/order differs",
    )
    cross_cell_ids: list[list[str]] = []
    for cell, record in zip(PRODUCTION_CELLS, records, strict=True):
        runtime_root = f"checkpoints/er9_production_pascal_v4/train{cell[0]}_channel{cell[1]}"
        terminal_path = runtime_root + "/run_terminal.json"
        validation_path = f"results/learned/er9/final_validation/train{cell[0]}_channel{cell[1]}.json"
        require(record.get("candidate") == FINAL_PAIR, f"ER-9 closeout candidate differs at {cell}")
        require(record.get("runtime_root") == runtime_root, f"ER-9 runtime identity differs at {cell}")
        require(record.get("terminal_path") == terminal_path, f"ER-9 terminal path differs at {cell}")
        full_sha256(record.get("terminal_sha256"), f"ER-9 terminal {cell}")
        require(record.get("promoted_stage1") is False, f"ER-9 Stage-1 promotion differs at {cell}")
        require(record.get("test_access") == 0, f"ER-9 closeout test access differs at {cell}")
        epoch = record.get("selected_epoch")
        require(isinstance(epoch, int) and not isinstance(epoch, bool) and 0 <= epoch < 100, f"ER-9 selected epoch differs at {cell}")
        checkpoint_sha = full_sha256(record.get("selected_checkpoint_sha256"), f"ER-9 checkpoint {cell}")
        require(record.get("validation_path") == validation_path, f"ER-9 validation path differs at {cell}")
        validation_file = REPO / validation_path
        require(sha256_file(validation_file) == record.get("validation_sha256"), f"ER-9 validation SHA differs at {cell}")
        validation = read_json(validation_file, f"ER-9 validation {cell}")
        content_address(validation, "validation_id", "er9productionvalidation-", f"ER-9 validation {cell}")
        validate_final_er9_cell(validation, cell=cell)
        require(validation.get("checkpoint") == {
            "epoch": epoch,
            "path": f"{runtime_root}/epochs/epoch-{epoch:04d}/checkpoint.pt",
            "sha256": checkpoint_sha,
        }, f"ER-9 selected validation checkpoint differs at {cell}")
        ids = [str(row["stable_sample_id"]) for row in validation["points"][0]["per_image"]]
        cross_cell_ids.append(ids)
    require(all(ids == cross_cell_ids[0] for ids in cross_cell_ids), "ER-9 cross-cell stable-ID order differs")
    require(closeout.get("test") == "SEALED" and closeout.get("test_access") == 0, "ER-9 closeout crossed test boundary")
    print("ER-9 production v4 published-evidence verifier PASS: three closeout cells and full validation; worker runtime not inspected")


def verify_er2_published() -> None:
    """Authenticate immutable ER-2 evidence without worker checkpoint bytes."""

    authority = verify_er2_authority(REPO)
    selected_path = ER2_ROOT / "er2_selected_checkpoint_v4.json"
    audit_path = ER2_ROOT / "er2_snr_assignment_audit_v4.json"
    validation_path = ER2_ROOT / "er2_randomized_validation_v4.json"
    selected = read_json(selected_path, "ER-2 selected checkpoint")
    content_address(selected, "selection_id", "er2selectedv4-", "ER-2 selection")
    require(selected.get("artifact_role") == "ER2_RANDOMIZED_SELECTED_CHECKPOINT_V4", "ER-2 selection role differs")
    require(selected.get("authority_id") == authority["authority_id"] and selected.get("source_commit") == authority["source_commit"], "ER-2 selection authority/source differs")
    require((selected.get("train_seed"), selected.get("channel_seed")) == (0, 0), "ER-2 selection seed differs")
    epoch = selected.get("selected_epoch")
    require(isinstance(epoch, int) and not isinstance(epoch, bool) and 0 <= epoch < 100, "ER-2 selected epoch differs")
    checkpoint_sha = full_sha256(selected.get("checkpoint_sha256"), "ER-2 selected checkpoint")
    require(selected.get("checkpoint_path") == f"{ER2_RUNTIME_ROOT}/epochs/epoch-{epoch:04d}/checkpoint.pt", "ER-2 selected checkpoint path differs")
    require(selected.get("selection_metric") == "validation_n_correct" and selected.get("tie_break") == "earliest_epoch", "ER-2 checkpoint-selection rule differs")
    require(str(selected.get("task_head_identity", "")).startswith("er9taskhead-") and len(str(selected["task_head_identity"])) == 76, "ER-2 task-head identity differs")
    require(selected.get("test") == "SEALED" and selected.get("test_access") == 0, "ER-2 selection crossed test boundary")
    # Rebuilding the keyed assignment digest requires the worker-local
    # Imagenette training corpus.  The unchanged terminal verifier performs
    # that strong recomputation on Confessor; hosted CI authenticates the
    # committed audit itself and all arithmetic that can be derived from it.
    audit = verify_audit(audit_path, authority, recompute=False)
    require(audit.get("schema_version") == 2, "ER-2 assignment-audit schema differs")
    require(audit.get("system") == "learned_snr_randomised", "ER-2 assignment-audit system differs")
    require(audit.get("rng_purpose") == "er2_snr_randomised_v1", "ER-2 assignment-audit RNG purpose differs")
    require(audit.get("identity_fields") == [
        "dataset_version", "split_manifest_hash", "stable_sample_id", "train_seed", "epoch",
    ], "ER-2 assignment-audit identity fields differ")
    require(audit.get("config_hash") == authority.get("config_hash"), "ER-2 assignment-audit config differs")
    full_sha256(audit.get("assignment_digest"), "ER-2 assignment digest")
    sample_count = audit.get("sample_count")
    epoch_count = audit.get("epoch_count")
    domain_keys = [str(value) for value in authority["randomized_snr_domain"]]
    require(sample_count == 8469 and epoch_count == 100, "ER-2 assignment-audit dimensions differ")
    rows = audit.get("per_epoch_counts")
    require(isinstance(rows, list) and [row.get("epoch") for row in rows] == list(range(epoch_count)), "ER-2 per-epoch assignment order differs")
    aggregate = {key: 0 for key in domain_keys}
    for row in rows:
        counts = row.get("counts")
        require(isinstance(counts, Mapping) and list(counts) == domain_keys, "ER-2 per-epoch assignment domain differs")
        require(all(isinstance(counts[key], int) and not isinstance(counts[key], bool) and counts[key] >= 0 for key in domain_keys), "ER-2 per-epoch assignment count is invalid")
        require(sum(counts[key] for key in domain_keys) == sample_count, "ER-2 per-epoch assignment total differs")
        for key in domain_keys:
            aggregate[key] += counts[key]
    require(audit.get("global_counts") == aggregate, "ER-2 global assignment counts differ")
    require(sum(aggregate.values()) == audit.get("assignment_count"), "ER-2 global assignment total differs")
    require(audit.get("runtime_root") == ER2_RUNTIME_ROOT, "ER-2 assignment-audit runtime differs")
    validation = verify_validation(validation_path, authority, selected)
    require(all(row.get("checkpoint_id") == checkpoint_sha for curve in validation["curves"] for row in curve["outcomes"]), "ER-2 validation checkpoint binding differs")
    completion = read_json(ER2_COMPLETION, "ER-2 completion")
    content_address(completion, "completion_id", "er2completionv4-", "ER-2 completion")
    require(completion.get("schema_version") == 2 and completion.get("artifact_role") == "ER2_RANDOMIZED_V4_COMPLETION", "ER-2 completion role/schema differs")
    require(completion.get("status") == "ONE_RANDOMIZED_RUN_COMPLETE_VALIDATION_ONLY", "ER-2 completion status differs")
    require(completion.get("authority_id") == authority["authority_id"] and completion.get("source_commit") == authority["source_commit"], "ER-2 completion authority/source differs")
    require(completion.get("runtime_root") == ER2_RUNTIME_ROOT and completion.get("scientific_training_run_count") == 1, "ER-2 completion scope differs")
    expected_records = {
        "terminal": (f"{ER2_RUNTIME_ROOT}/run_terminal.json", None),
        "selected_checkpoint": (str(selected_path.relative_to(REPO)), sha256_file(selected_path)),
        "assignment_audit": (str(audit_path.relative_to(REPO)), sha256_file(audit_path)),
        "validation": (str(validation_path.relative_to(REPO)), sha256_file(validation_path)),
    }
    for field, (path, digest) in expected_records.items():
        record = completion.get(field)
        require(isinstance(record, Mapping) and record.get("path") == path, f"ER-2 completion {field} path differs")
        full_sha256(record.get("sha256"), f"ER-2 completion {field}")
        if digest is not None:
            require(record["sha256"] == digest, f"ER-2 completion {field} SHA differs")
    require(completion.get("test") == "SEALED" and completion.get("test_access") == 0, "ER-2 completion crossed test boundary")
    print("randomized ER-2 v4 published-evidence verifier PASS: one run and full validation; worker runtime not inspected")


def hosted_commands(profile: str) -> tuple[list[str], ...]:
    """Replace only exact worker-runtime terminal commands, failing closed."""

    commands = gate.profile_commands(profile)
    replaced_er9 = 0
    replaced_er2 = 0
    result: list[list[str]] = []
    for command in commands:
        tool = Path(command[1]).name if len(command) >= 2 else ""
        arguments = command[2:]
        if tool == "verify_er9_production_v4.py" and arguments == ["--terminal"]:
            require(ER9_CLOSEOUT.is_file() and not ER9_CLOSEOUT.is_symlink(), "ER-9 terminal routing lacks a safe closeout")
            result.append([sys.executable, str(Path(__file__).resolve()), "verify-er9-published"])
            replaced_er9 += 1
        elif tool == "verify_er2_randomized.py" and not arguments:
            require(ER2_COMPLETION.is_file() and not ER2_COMPLETION.is_symlink(), "ER-2 terminal routing lacks a safe completion")
            result.append([sys.executable, str(Path(__file__).resolve()), "verify-er2-published"])
            replaced_er2 += 1
        else:
            result.append(command)
    expected_er9 = int(ER9_CLOSEOUT.is_file() and not ER9_CLOSEOUT.is_symlink())
    expected_er2 = int(ER2_COMPLETION.is_file() and not ER2_COMPLETION.is_symlink())
    require(replaced_er9 == expected_er9, "hosted ER-9 runtime-command replacement count differs")
    require(replaced_er2 == expected_er2, "hosted ER-2 runtime-command replacement count differs")
    return tuple(result)


def self_test() -> None:
    original = gate.profile_commands("ci-cpu")
    hosted = hosted_commands("ci-cpu")
    require(len(original) == len(hosted), "hosted routing changed command count")
    differences = [(before, after) for before, after in zip(original, hosted, strict=True) if before != after]
    expected = 1 + int(ER2_COMPLETION.is_file() and not ER2_COMPLETION.is_symlink())
    require(len(differences) == expected, "hosted routing changed an unexpected command")
    for before, after in differences:
        require(Path(before[1]).name in {"verify_er9_production_v4.py", "verify_er2_randomized.py"}, "hosted routing replaced a non-runtime command")
        require(Path(after[1]).resolve() == Path(__file__).resolve(), "hosted routing replacement is not this audited adapter")
    verify_er9_published()
    if ER2_COMPLETION.is_file() and not ER2_COMPLETION.is_symlink():
        verify_er2_published()
    print(f"hosted quality-gate routing self-test PASS: replacements={len(differences)}")


def run(profile: str) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), str(REPO / "tools"), environment.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    for command in hosted_commands(profile):
        print("$ " + " ".join(command), flush=True)
        subprocess.run(command, cwd=REPO, env=environment, check=True)
    gate._check_clean_checkout()
    print(f"hosted quality gate PASS: {profile}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("self-test", "verify-er9-published", "verify-er2-published", "static", "ci-cpu"))
    args = parser.parse_args(argv)
    if args.action == "self-test":
        self_test()
    elif args.action == "verify-er9-published":
        verify_er9_published()
    elif args.action == "verify-er2-published":
        verify_er2_published()
    else:
        run(args.action)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
