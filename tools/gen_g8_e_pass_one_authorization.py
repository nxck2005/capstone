#!/usr/bin/env python3
"""Freeze the narrow owner-authorized G8_E E5 artifacts (authorization + marker).

Generates, from the authenticated frozen chain only:

  results/baseline/g8_e/e2_confessor_successor/e5_pass_one_authorization.json
  results/baseline/g8_e/e2_confessor_successor/e5_pre_execution_marker.json

The authorization binds the exact worker-successor campaign/contract identity,
the scientific data identity, the E2 completion/E3/E4 lifecycle artifacts by
their exact SHA-256 values, the frozen successor BLER table, the W4 selection
policy fingerprint and call plan, the candidate authority and outage policy,
and the single pre-registered pass-one output path.  Its scope permits verify +
pass-one-once + freeze/closeout publication and refuses training, pass two and
three, fallback, ratio adjudication, test access, learned-system training and
G8_F execution.  Both files are refused if they already exist or if a pass-one
completion record already exists anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline import g8_e_pass_one as pass_one  # noqa: E402
from baseline import g8_e_corrected_v3 as v3  # noqa: E402

AUTHORIZED_BY = (
    "repository owner/operator via the explicit G8_E E5-E7 takeover execution "
    "prompt of 2026-08-22, authorizing selection pass one EXACTLY ONCE plus its "
    "verification, freeze and closeout"
)
EXACT_COMMAND = ".venv/bin/python tools/run_g8_e_pass_one.py --execute"
OUTPUT_RULE = (
    "one immutable content-addressed completion record at the pre-registered "
    "path results/baseline/g8_e/pass_one_state.json; refuse whenever it exists"
)


def _atomic_publish(path: Path, payload: bytes) -> None:
    staging = path.parent / f".{path.name}.staging-{os.getpid()}"
    try:
        with open(staging, "wb") as stream:
            stream.write(payload)
            stream.flush()
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    if pass_one.PASS_ONE_STATE_PATH.exists():
        print("FAIL: a pass-one completion record already exists", file=sys.stderr)
        return 2
    if pass_one.E5_AUTHORIZATION_PATH.exists() or pass_one.E5_MARKER_PATH.exists():
        print("FAIL: E5 authorization/marker already exists", file=sys.stderr)
        return 2

    context = pass_one.authenticate_frozen_chain()
    chain = context["chain"]
    contract = context["contract"]
    historical_e2_issued = json.loads(
        pass_one.v3s.V3S_AUTHORIZATION_PATH.read_bytes()
    )["issued_sha256"]

    authorization: dict[str, object] = {
        "schema_version": pass_one.PASS_ONE_SCHEMA_VERSION,
        "artifact_role": pass_one.AUTHORIZATION_ROLE,
        "status": "AUTHORIZED",
        "authorized_by": AUTHORIZED_BY,
        "reason": (
            "Narrow E5/pass-one authorization over the completed verified E2-E4 "
            f"worker-successor campaign {contract['campaign_id']}: verify the "
            "E4 prerequisites, execute the frozen BR-4 selection call plan "
            "exactly once through the frozen scorer, freeze the immutable "
            "pass-one state and the training-only corpus-spec lineage binding, "
            "and close out G8_E through E7 verification. Supersedes nothing: "
            f"the historical {historical_e2_issued} E2-E4-only authorization "
            "remains immutable history."
        ),
        "campaign_id": contract["campaign_id"],
        "contract_id": contract["contract_id"],
        "contract_sha256": v3.sha256_bytes(
            pass_one.v3s.V3S_CONTRACT_PATH.read_bytes()
        ),
        "source_manifest_id": contract["source_manifest"]["id"],
        "source_manifest_sha256": contract["source_manifest"]["sha256"],
        "data_identity_id": contract["scientific_data_identity"]["id"],
        "data_identity_sha256": contract["scientific_data_identity"]["sha256"],
        **{key: chain[key] for key in (
            "e2_completion_sha256",
            "e3_id",
            "e3_sha256",
            "e4_id",
            "e4_sha256",
            "bler_table_id",
            "bler_table_sha256",
            "w4_integration_adjudication_sha256",
            "selection_policy_sha256",
            "selection_call_plan_sha256",
            "candidate_authority_file_sha256",
            "outage_policy_file_sha256",
        )},
        "state_path": str(pass_one.PASS_ONE_STATE_PATH.relative_to(pass_one.REPO_ROOT)),
        "scope": dict(pass_one.AUTHORIZED_SCOPE),
    }
    authorization["issued_sha256"] = v3.sha256_bytes(
        v3.canonical_json(authorization)
    )

    marker = {
        "schema_version": pass_one.PASS_ONE_SCHEMA_VERSION,
        "artifact_role": pass_one.MARKER_ROLE,
        "status": "MARKED_PRE_EXECUTION",
        "authorization_path": str(pass_one.E5_AUTHORIZATION_PATH.relative_to(pass_one.REPO_ROOT)),
        "authorization_sha256": authorization["issued_sha256"],
        "scorer_module": pass_one.SCORER_MODULE,
        "selection_policy_sha256": chain["selection_policy_sha256"],
        "e4_input_id": chain["e4_id"],
        "e4_input_sha256": chain["e4_sha256"],
        "intended_output_path": str(pass_one.PASS_ONE_STATE_PATH.relative_to(pass_one.REPO_ROOT)),
        "intended_output_rule": OUTPUT_RULE,
        "exact_command": EXACT_COMMAND,
        "restart_command": (
            EXACT_COMMAND
            + "  # idempotent: refuses when the completion record already exists"
        ),
        "pre_execution_pass_one_count": 0,
    }
    marker["issued_sha256"] = v3.sha256_bytes(v3.canonical_json(marker))

    _atomic_publish(
        pass_one.E5_AUTHORIZATION_PATH, v3.rendered_json(authorization)
    )
    _atomic_publish(pass_one.E5_MARKER_PATH, v3.rendered_json(marker))
    print(json.dumps({
        "status": "FROZEN",
        "authorization_path": str(pass_one.E5_AUTHORIZATION_PATH.relative_to(pass_one.REPO_ROOT)),
        "authorization_issued_sha256": authorization["issued_sha256"],
        "marker_path": str(pass_one.E5_MARKER_PATH.relative_to(pass_one.REPO_ROOT)),
        "marker_issued_sha256": marker["issued_sha256"],
        "state_path": str(pass_one.PASS_ONE_STATE_PATH.relative_to(pass_one.REPO_ROOT)),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
