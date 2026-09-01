#!/usr/bin/env python3
"""Generate or verify the detached owner-issued W8-B launch authority only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from gen_w8_execution_authorization import (  # noqa: E402
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    DEFAULT_OUTPUT as W8_A_PATH,
    PASCAL_LOCK_SHA256,
)
from gen_w8_source_manifest import DEFAULT_OUTPUT as SOURCE_MANIFEST_PATH  # noqa: E402
from run_w8_campaign import (  # noqa: E402
    LAUNCH_AUTHORIZATION_ROLE,
    load_authority,
    verify_launch_authorization,
)
from training.deterministic_core import canonical_bytes, canonical_sha256  # noqa: E402
from training.w8_protocol import (  # noqa: E402
    W8_ACCUMULATION_FACTOR,
    W8_EFFECTIVE_BATCH_SIZE,
    W8_PHYSICAL_BATCH_SIZE,
    W8_PROFILE_ID,
    W8_SELECTED_GPU_NAME,
    W8_SELECTED_GPU_UUID,
    W8_VALIDATION_BATCH_SIZE,
)

OUTPUT_PATH = REPO / "results/learned/w8/w8_b_launch_authorization.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_launch_authorization(
    authorization_path: Path,
    source_manifest_path: Path,
    *,
    issued_at_utc: str,
) -> dict[str, Any]:
    """Build authority bytes; this function cannot create campaign state."""

    if not issued_at_utc:
        raise ValueError("issued_at_utc must be explicit")
    authorization, manifest = load_authority(authorization_path, source_manifest_path)
    body: dict[str, Any] = {
        "schema_version": 1,
        "artifact_role": LAUNCH_AUTHORIZATION_ROLE,
        "status": "AUTHORIZED",
        "authorization_scope": "W8_SIX_CORE_RUNS_ONLY",
        "issued_at_utc": issued_at_utc,
        "w8_a_authorization_id": authorization["authorization_id"],
        "w8_a_authorization_sha256": _sha(authorization_path),
        "source_commit": manifest["source_commit"],
        "source_manifest_id": manifest["manifest_id"],
        "source_manifest_sha256": _sha(source_manifest_path),
        "campaign_id": CAMPAIGN_ID,
        "campaign_root": CAMPAIGN_ROOT,
        "profile": {
            "execution_profile_id": W8_PROFILE_ID,
            "gpu_name": W8_SELECTED_GPU_NAME,
            "gpu_uuid": W8_SELECTED_GPU_UUID,
            "device": "cuda:0",
            "requirements_lock": "requirements-pascal.lock",
            "requirements_lock_sha256": PASCAL_LOCK_SHA256,
            "physical_batch_size": W8_PHYSICAL_BATCH_SIZE,
            "accumulation_factor": W8_ACCUMULATION_FACTOR,
            "effective_batch_size": W8_EFFECTIVE_BATCH_SIZE,
            "validation_batch_size": W8_VALIDATION_BATCH_SIZE,
        },
        "scope": {
            "core_runs": 6,
            "er2_randomized_training": False,
            "papr_constrained_training": False,
            "er9_training": False,
            "g10": False,
        },
        "test": {
            "status": "SEALED",
            "model_facing_access": 0,
            "learned_inference": 0,
        },
        "owner_authorization": True,
    }
    value = dict(body)
    value["authorization_id"] = "w8blaunch-" + canonical_sha256(body)
    return value


def _verify(
    path: Path, authorization_path: Path, source_manifest_path: Path
) -> dict[str, Any]:
    authorization, manifest = load_authority(authorization_path, source_manifest_path)
    return verify_launch_authorization(
        path,
        w8_authorization=authorization,
        w8_authorization_path=authorization_path,
        source_manifest=manifest,
        source_manifest_path=source_manifest_path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, default=W8_A_PATH)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST_PATH)
    parser.add_argument("--path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--issued-at-utc")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    authorization_path = args.authorization.resolve()
    source_manifest_path = args.source_manifest.resolve()
    output_path = args.path.resolve()
    if args.check:
        if args.issued_at_utc is not None:
            parser.error("--check does not accept --issued-at-utc")
    else:
        if args.issued_at_utc is None:
            parser.error("generation requires explicit --issued-at-utc")
        value = build_launch_authorization(
            authorization_path,
            source_manifest_path,
            issued_at_utc=args.issued_at_utc,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("xb") as stream:
            stream.write(canonical_bytes(value))

    value = _verify(output_path, authorization_path, source_manifest_path)
    print(
        json.dumps(
            {
                "status": "PASS",
                "path": str(output_path.relative_to(REPO)),
                "authorization_id": value["authorization_id"],
                "file_sha256": _sha(output_path),
                "w8_a_authorization_id": value["w8_a_authorization_id"],
                "source_commit": value["source_commit"],
                "source_manifest_id": value["source_manifest_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
