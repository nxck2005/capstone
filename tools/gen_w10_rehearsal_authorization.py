#!/usr/bin/env python3
"""Freeze or preflight the W10 validation-only rehearsal authority (AM-98).

``--preflight`` validates the scope, the active successor source epoch and every
binding without writing anything and without touching a GPU.  A real freeze
additionally requires all bindings FROZEN (including the future PAPR checkpoint
and the two pre-W10 validation selections) and the exact Confessor TITAN Xp.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402
from evaluation.downstream_v4 import TITAN_XP_NAME, TITAN_XP_UUID, immutable_write  # noqa: E402
from evaluation.w10_bindings import pending_states, resolve_scope_bindings  # noqa: E402
from evaluation.w10_scope import (  # noqa: E402
    SCOPE,
    W10_CELL,
    W10_DATASET,
    W10_SPLIT,
    W10_VALIDATION_DENOMINATOR,
    scope_identity,
    scope_sha256,
    snr_grid,
    unit_count,
    work_units,
)
from runtime.source_epochs import load_w10_manifest, source_record  # noqa: E402
from runtime.w9_authority import authenticate_live_w9_pascal  # noqa: E402
from training.deterministic_core import canonical_sha256  # noqa: E402
from verify_g11 import verify_authority as verify_g11_authority, verify_terminal as verify_g11_terminal  # noqa: E402

TARGET = REPO / "results/learned/w10/w10_rehearsal_authorization.json"
W10_RUNTIME_ROOT = "checkpoints/w10_rehearsal"


def g11_closure() -> dict:
    """Authenticate and content-bind the closed G11 terminal via its own verifier."""

    authority_path = REPO / "results/learned/g11/g11_execution_authorization_v4.json"
    terminal_path = REPO / "results/learned/g11/g11_terminal_closeout.json"
    authority = verify_g11_authority()
    terminal = verify_g11_terminal(terminal_path)
    if terminal.get("decision") != "GREEN" or terminal.get("test_access") != 0:
        raise SystemExit("G11 terminal is not GREEN with zero test access")
    return {
        "authority_id": str(authority["authority_id"]),
        "authority_path": str(authority_path.relative_to(REPO)),
        "authority_sha256": hashlib.sha256(authority_path.read_bytes()).hexdigest(),
        "terminal_id": str(terminal["terminal_id"]),
        "terminal_path": str(terminal_path.relative_to(REPO)),
        "terminal_sha256": hashlib.sha256(terminal_path.read_bytes()).hexdigest(),
        "decision": "GREEN",
        "test": "SEALED",
        "test_access": 0,
    }


def build_body(*, gpu_name: str, gpu_uuid: str, source: dict, bindings: list[dict], cuda_mapping: dict | None) -> dict:
    profile = get("environment.execution_profiles.confessor_pascal_cu126")
    body: dict = {
        "schema_version": 2,
        "authority_kind": "W10_VALIDATION_REHEARSAL_AUTHORITY",
        "status": "FROZEN_W10_ONLY_PRE_EXECUTION",
        "source_manifest": source_record(REPO, source),
        "source_commit": source["source_commit"],
        "source_binding": source,
        "scope_version": int(get("evaluation.w10_scope_version")),
        "scope": scope_identity(),
        "scope_sha256": scope_sha256(),
        "scope_table_module": "src/evaluation/w10_scope.py",
        "snr_grid_db": list(snr_grid()),
        "systems": list(dict.fromkeys(entry.system for entry in SCOPE)),
        "ratios": sorted({entry.bw_ratio for entry in SCOPE}),
        "learned_ratios": list(get("evaluation.w10_learned_ratios")),
        "cell": {"train_seed": W10_CELL[0], "channel_seed": W10_CELL[1]},
        "split": W10_SPLIT,
        "dataset": W10_DATASET,
        "denominator": W10_VALIDATION_DENOMINATOR,
        "unit_count": unit_count(),
        "unit_count_derived_from_scope": True,
        "bindings": bindings,
        "g11_closure": g11_closure(),
        "runtime_root": W10_RUNTIME_ROOT,
        "execution_profile_id": "confessor_pascal_cu126",
        "host": "confessor",
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "compute_capability": str(profile["compute_capability"]),
        "device": "cuda:0",
        "cuda_visible_devices": gpu_uuid,
        "papr_training_authorized": False,
        "production_training_authorized": False,
        "er2_training_authorized": False,
        "g11_authorized": False,
        "papr_training_run_count": 1,
        "papr_lifecycle_bound": True,
        "test_authorized": False,
        "validation_only": True,
        "test": "SEALED",
        "test_access": 0,
    }
    if cuda_mapping is not None:
        body["cuda_mapping"] = cuda_mapping
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-name", default=TITAN_XP_NAME)
    parser.add_argument("--gpu-uuid", default=TITAN_XP_UUID)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if (args.gpu_name, args.gpu_uuid) != (TITAN_XP_NAME, TITAN_XP_UUID):
        raise SystemExit("W10 requires the exact Confessor TITAN Xp")
    source = load_w10_manifest(REPO, live=True)
    bindings = resolve_scope_bindings(REPO)
    pending = pending_states(bindings)
    if args.preflight:
        print(f"W10 preflight: scope_sha256={scope_sha256()} units={unit_count()} pending={pending or 'none'}")
        return 0
    if pending:
        raise SystemExit("W10 authority cannot freeze while bindings are pending: " + ", ".join(pending))
    body = build_body(gpu_name=args.gpu_name, gpu_uuid=args.gpu_uuid, source=source, bindings=bindings, cuda_mapping=None)
    live = authenticate_live_w9_pascal(
        REPO,
        body,
        config_hash=canonical_sha256({"scope": scope_identity(), "units": work_units()}),
    )
    body["cuda_mapping"] = live["environment"]["cuda_mapping"]
    body["authority_id"] = "w10rehearsalauth-" + canonical_sha256(body)
    immutable_write(TARGET, body)
    print(f"W10 rehearsal authority: {body['authority_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
