#!/usr/bin/env python3
"""Publish the immutable, zero-coverage W8-A pre-execution completion record."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from config.run_config import config_hash as run_config_hash  # noqa: E402
from gen_w8_data_verification import verify_data_verification  # noqa: E402
from gen_w8_runtime_estimate import verify_runtime_estimate  # noqa: E402
from gen_w8_execution_authorization import (  # noqa: E402
    AUTHORIZATION_ROLE,
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    G4_ID,
    PASCAL_LOCK_SHA256,
    W7_TERMINAL_ID,
    verify_authorization,
)
from gen_w8_source_manifest import verify_manifest  # noqa: E402
from training.deterministic_core import canonical_bytes, canonical_sha256  # noqa: E402
from training.w8_protocol import (  # noqa: E402
    W8_ACCUMULATION_FACTOR,
    W8_EFFECTIVE_BATCH_SIZE,
    W8_PHYSICAL_BATCH_SIZE,
    W8_SELECTED_GPU_NAME,
    W8_SELECTED_GPU_UUID,
    W8_VALIDATION_BATCH_SIZE,
    run_cells,
)


COMPLETION_ROLE = "W8_A_PRE_EXECUTION_COMPLETION"
COMPLETION_PREFIX = "w8acompletion-"
DEFAULT_OUTPUT = REPO / "results/learned/w8/w8_a_completion.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def build_completion(
    *,
    source_manifest_path: Path,
    authorization_path: Path,
    smoke_path: Path,
    data_verification: dict[str, Any],
    runtime_estimate: dict[str, Any],
    issued_at_utc: str | None = None,
) -> dict[str, Any]:
    source_manifest = verify_manifest(source_manifest_path)
    verify_data_verification(data_verification, repo=REPO)
    verify_runtime_estimate(runtime_estimate)
    authorization = verify_authorization(
        authorization_path,
        expected_source_commit=source_manifest["source_commit"],
        expected_source_manifest_path=source_manifest_path,
    )
    smoke = _read(smoke_path)
    if smoke.get("artifact_role") != "W8_NON_SCIENTIFIC_SMOKE" or smoke.get("scientific_status") != "NON_SCIENTIFIC":
        raise ValueError("W8 smoke record is not the required non-scientific artifact")
    body: dict[str, Any] = {
        "schema_version": 1,
        "artifact_role": COMPLETION_ROLE,
        "status": "GREEN_PRE_EXECUTION",
        "issued_at_utc": issued_at_utc or datetime.now(timezone.utc).isoformat(),
        "upstream": {
            "w7_terminal_completion_id": W7_TERMINAL_ID,
            "g4_adjudication_id": G4_ID,
            "lambda_core": 3.0,
            "lambda_status": "selected_at_G-4",
        },
        "scientific_source": {
            "source_commit": source_manifest["source_commit"],
            "source_manifest_id": source_manifest["manifest_id"],
            "source_manifest_sha256": _sha(source_manifest_path),
            "authorization_role": AUTHORIZATION_ROLE,
            "authorization_id": authorization["authorization_id"],
            "authorization_sha256": _sha(authorization_path),
        },
        "campaign": {
            "campaign_id": CAMPAIGN_ID,
            "campaign_root": CAMPAIGN_ROOT,
            "run_cells": [cell.to_dict() for cell in run_cells()],
            "run_count": 6,  # literal-ok: frozen W8 six-cell matrix
            "unique_ratios": ["r_1_6", "r_1_24"],
            "seed_pairing": "zipped_not_cross_product",
        },
        "profile": {
            "execution_profile_id": "confessor_pascal_cu126",
            "scientific_writer_host": "confessor",
            "gpu_name": W8_SELECTED_GPU_NAME,
            "gpu_uuid": W8_SELECTED_GPU_UUID,
            "requirements_lock_sha256": PASCAL_LOCK_SHA256,
            "physical_batch_size": W8_PHYSICAL_BATCH_SIZE,
            "accumulation_factor": W8_ACCUMULATION_FACTOR,
            "effective_batch_size": W8_EFFECTIVE_BATCH_SIZE,
            "validation_batch_size": W8_VALIDATION_BATCH_SIZE,
        },
        "smoke": {
            "path": str(smoke_path.relative_to(REPO)) if smoke_path.is_relative_to(REPO) else str(smoke_path),
            "file_sha256": _sha(smoke_path),
            "smoke_id": smoke["smoke_id"],
            "scientific_status": "NON_SCIENTIFIC",
            "w8_result_eligibility": "NOT_ELIGIBLE_FOR_W8_RESULT",
            "g10_eligibility": "NOT_ELIGIBLE_FOR_G10",
            "test_eligibility": "NOT_ELIGIBLE_FOR_TEST",
        },
        "data_verification": data_verification,
        "runtime_estimate": runtime_estimate,
        "pre_execution_counters": {
            "w8_scientific_optimizer_steps": 0,
            "w8_final_training_runs": 0,
            "w8_completed_runs": 0,
            "w8_scientific_checkpoints": 0,
            "g10_adjudications": 0,
            "er2_randomized_training": 0,
            "papr_constrained_training": 0,
            "er9_training": 0,
            "test_model_facing_access": 0,
            "learned_test_inference": 0,
        },
        "boundary": {
            "scientific_execution": "REQUIRES_SEPARATE_W8_B_OWNER_AUTHORIZATION",
            "w8_b_launch_authorization_present": False,
            "source_contains_scientific_w8_results": False,
            "campaign_root_created_at_freeze": False,
            "test": "SEALED",
            "g10": "NOT_AUTHORIZED",
            "er2_randomized_training": "NOT_AUTHORIZED",
            "papr_constrained_training": "NOT_AUTHORIZED_BOUND_REQUIRES_DOWNSTREAM_PROTOCOL_ITEM",
            "er9": "NOT_AUTHORIZED",
        },
        "smoke_optimizer_steps_are_scientific_zero": True,
    }
    body["completion_id"] = COMPLETION_PREFIX + canonical_sha256(body)
    return body


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_completion(value: dict[str, Any], path: Path) -> None:
    """Publish the zero-coverage completion without replacing its final name."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable W8-A completion already exists: {path}")
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
            raise FileExistsError(f"immutable W8-A completion already exists: {path}") from None
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--data-verification", type=Path, required=True)
    parser.add_argument("--runtime-estimate", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--issued-at-utc")
    args = parser.parse_args(argv)
    paths = {
        key: value if value.is_absolute() else REPO / value
        for key, value in {
            "source": args.source_manifest,
            "authorization": args.authorization,
            "smoke": args.smoke,
            "data": args.data_verification,
            "estimate": args.runtime_estimate,
            "output": args.output,
        }.items()
    }
    value = build_completion(
        source_manifest_path=paths["source"], authorization_path=paths["authorization"],
        smoke_path=paths["smoke"], data_verification=_read(paths["data"]),
        runtime_estimate=_read(paths["estimate"]), issued_at_utc=args.issued_at_utc,
    )
    write_completion(value, paths["output"])
    print(f"W8-A completion written: {value['completion_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
