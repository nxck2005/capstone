#!/usr/bin/env python3
"""Freeze the G8_E E6 artifacts after a verified pass one.

Generates results/baseline/g8_e/e2_confessor_successor/e6_pass_one_freeze.json:
the additive completion of the frozen E1 corpus specification's pass-one
lineage (whose own bytes stay untouched — its verifier requires
``state_sha256 is None`` forever), binding the immutable pass-one state, the
unchanged corpus specification, the training-only scope proof and zero
prohibited counters.  Refuses unless the pass-one state authenticates against
a full recomputation and no freeze exists yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline import g8_e_corrected_v3 as v3  # noqa: E402
from baseline import g8_e_pass_one as pass_one  # noqa: E402

E6_FREEZE_PATH = pass_one.v3s.V3S_ROOT / "e6_pass_one_freeze.json"
E6_FREEZE_ROLE = "g8_e_v3_e6_pass_one_freeze"
CORPUS_SPEC_PATH = pass_one.REPO_ROOT / "results/baseline/g8_e/corpus_spec.json"


def build_freeze(context: dict, verified: dict, marker_digest: str) -> dict:
    corpus_raw = CORPUS_SPEC_PATH.read_bytes()
    corpus = json.loads(corpus_raw)
    from baseline.g8_e import verify_e1_corpus_spec_file

    verify_e1_corpus_spec_file(CORPUS_SPEC_PATH)
    lineage = corpus["selected_pass_one_lineage"]
    if (
        lineage["required"] is not True
        or lineage["state_path"] != str(pass_one.PASS_ONE_STATE_PATH.relative_to(pass_one.REPO_ROOT))
        or lineage["state_sha256"] is not None
        or lineage["selection_state_must_be_immutable"] is not True
    ):
        raise SystemExit("corpus-spec lineage schema differs from the pre-registered binding")
    if corpus["training_only"] is not True or corpus["materialized"] is not False:
        raise SystemExit("corpus specification is not an unmaterialized training-only spec")

    body: dict = {
        "schema_version": pass_one.PASS_ONE_SCHEMA_VERSION,
        "artifact_role": E6_FREEZE_ROLE,
        "status": "E6_PASS_ONE_FROZEN",
        "campaign_id": context["contract"]["campaign_id"],
        "contract_id": context["contract"]["contract_id"],
        "authorization_issued_sha256": context["authorization"]["issued_sha256"],
        "marker_issued_sha256": marker_digest,
        "pass_one_state": {
            "path": str(pass_one.PASS_ONE_STATE_PATH.relative_to(pass_one.REPO_ROOT)),
            "state_id": verified["state_id"],
            "state_content_sha256": verified["state_sha256"],
            "state_file_sha256": verified["file_sha256"],
            "immutable": True,
            "executed_exactly_once": True,
        },
        "corpus_specification_binding": {
            "path": str(CORPUS_SPEC_PATH.relative_to(pass_one.REPO_ROOT)),
            "corpus_spec_id": corpus["corpus_spec_id"],
            "sha256": v3.sha256_bytes(corpus_raw),
            "bytes": len(corpus_raw),
            "frozen_spec_bytes_untouched": True,
            "resolved_lineage": {
                "state_path": lineage["state_path"],
                "state_sha256": verified["state_sha256"],
                "selection_record_field": lineage["selection_record_field"],
            },
            "training_only": True,
            "materialized": False,
            "materialized_object_count": corpus["materialized_object_count"],
            "validation_ids_forbidden": corpus["forbidden_membership"]["validation_ids_forbidden"],
            "test_ids_forbidden": corpus["forbidden_membership"]["test_ids_forbidden"],
        },
        "counters": dict(verified["counters"]),
    }
    body["e6_freeze_id"] = v3._id(
        "g8ee6freeze-",
        {key: child for key, child in body.items() if key != "e6_freeze_id"},
    )
    body["e6_freeze_sha256"] = v3.sha256_bytes(v3.canonical_json(body))
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    if E6_FREEZE_PATH.exists():
        print("FAIL: E6 freeze already exists", file=sys.stderr)
        return 2
    context = pass_one.authenticate_inputs()
    marker = pass_one.authenticate_marker(
        pass_one.E5_MARKER_PATH, pass_one.E5_AUTHORIZATION_PATH, context["authorization"]
    )
    verified = pass_one.verify_pass_one_state()
    body = build_freeze(context, verified, marker["issued_sha256"])
    payload = v3.rendered_json(body)
    staging = E6_FREEZE_PATH.parent / f".{E6_FREEZE_PATH.name}.staging"
    staging.write_bytes(payload)
    staging.replace(E6_FREEZE_PATH)
    print(json.dumps({
        "status": "FROZEN",
        "phase": "E6_PASS_ONE_FROZEN",
        "e6_freeze_id": body["e6_freeze_id"],
        "e6_freeze_sha256": v3.sha256_bytes(payload),
        "pass_one_state_id": verified["state_id"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
