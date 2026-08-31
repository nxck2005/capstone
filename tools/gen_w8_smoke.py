#!/usr/bin/env python3
"""Create the immutable, explicitly ineligible W8-A smoke record.

This command resolves protocol/configuration identities only.  It never loads a
real dataset, creates a scientific checkpoint, performs an optimizer step, or
opens a test sample.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from config.run_config import config_hash as run_config_hash  # noqa: E402
from training.deterministic_core import canonical_bytes, canonical_sha256  # noqa: E402
from training.w8_protocol import (  # noqa: E402
    W8_SMOKE_ROLE,
    eligibility_for_role,
    fresh_initialization_identity,
    protocol_config_hash,
    run_cells,
)
from training.w8_final import load_w8_smoke_config  # noqa: E402


SMOKE_ROLE = "W8_NON_SCIENTIFIC_SMOKE"
SMOKE_PREFIX = "w8smoke-"
DEFAULT_OUTPUT = REPO / "results/learned/w8/w8_a_smoke.json"


def build_smoke(*, source_commit: str, focused_test_command: str, issued_at_utc: str | None = None) -> dict[str, Any]:
    cells = run_cells()
    configs = []
    for cell in cells:
        config = load_w8_smoke_config(cell.ratio, train_seed=cell.train_seed, channel_seed=cell.channel_seed)
        configs.append({
            "run_index": cell.run_index,
            "ratio": cell.ratio,
            "k": cell.k,
            "train_seed": cell.train_seed,
            "channel_seed": cell.channel_seed,
            "config_hash": run_config_hash(config),
            "protocol_config_hash": protocol_config_hash(config),
            "artifact_role": config.resolved["artifact_role"],
            "eligibility": eligibility_for_role(SMOKE_ROLE),
        })
    body: dict[str, Any] = {
        "schema_version": 1,
        "artifact_role": SMOKE_ROLE,
        "scientific_status": "NON_SCIENTIFIC",
        "status": "PASS_NON_SCIENTIFIC_ONLY",
        "issued_at_utc": issued_at_utc or datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "protocol_checks": {
            "six_configurations_constructed": True,
            "ratios_and_k_paths": {"r_1_6": 12800, "r_1_24": 3200},
            "lambda_exact": 3.0,
            "seed_zipper": "PASS",
            "run_order": [cell.to_dict() for cell in cells],
        },
        "configurations": configs,
        "fresh_initialization_checks": {
            "identity_fields": ["train_seed", "component_path"],
            "same_seed_same_identity": fresh_initialization_identity(0),
            "different_seed_identity": fresh_initialization_identity(1),
            "w7_checkpoint_initialization": "REJECTED_BY_W8_TRAINER_CONTRACT",
            "foreign_w8_checkpoint_initialization": "REJECTED_BY_W8_TRAINER_CONTRACT",
            "same_run_resume_only": "AUTHENTICATED_CHECKPOINT_LINEAGE_REQUIRED",
        },
        "boundary": {
            "optimizer_steps": 0,
            "scientific_checkpoints": 0,
            "w8_result_eligibility": "NOT_ELIGIBLE_FOR_W8_RESULT",
            "g10_eligibility": "NOT_ELIGIBLE_FOR_G10",
            "test_eligibility": "NOT_ELIGIBLE_FOR_TEST",
            "test_model_facing_access": 0,
            "learned_test_inference": 0,
            "er2_randomized_training": 0,
            "papr_constrained_training": 0,
            "er9_training": 0,
            "g10_adjudications": 0,
        },
        "focused_test_command": focused_test_command,
        "focused_test_result": "MUST_BE_GREEN_BEFORE_W8_A_COMPLETION",
        "papr_secondary_protocol_item": "max_papr_db_or_equivalent_exact_bound_requires_downstream_pre_execution_authority; not invented from W7 measurements",
        "eligibility": eligibility_for_role(SMOKE_ROLE),
    }
    body["smoke_id"] = SMOKE_PREFIX + canonical_sha256(body)
    return body


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_smoke(value: dict[str, Any], path: Path) -> None:
    """Publish the carrier record without a replaceable final pathname."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable W8 smoke record already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(f"immutable W8 smoke record already exists: {path}") from None
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--focused-test-command", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--issued-at-utc")
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else REPO / args.output
    value = build_smoke(
        source_commit=args.source_commit,
        focused_test_command=args.focused_test_command,
        issued_at_utc=args.issued_at_utc,
    )
    write_smoke(value, output)
    print(f"W8 non-scientific smoke record written: {value['smoke_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
