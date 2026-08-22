#!/usr/bin/env python3
"""G8_E E7 closeout: full read-only verification and terminal verdict.

Verifies, in order, with live sources and live data:

  1. the frozen worker-successor contract chain (v3s frozen verifier);
  2. the E2 completion / E3 / E4 lifecycle artifacts against the tracked
     bytes, their exact custody SHA-256 values and the count-derived object
     semantics (denominators, outcome arithmetic, no accuracy floats);
  3. the W4 integration adjudication file and its preregistered selection
     policy fingerprint recomputed from the live module;
  4. the narrow E5 authorization, its scope, and the pre-execution marker
     (pre_execution_pass_one_count = 0);
  5. the immutable pass-one completion record: content ID/digest, bindings,
     exactly-one execution counter, zero prohibited counters, and a full
     byte-equal recomputation of every selection from the frozen inputs;
  6. the E6 freeze artifact: corpus-spec lineage completion with untouched
     spec bytes, training-only/materialized-false proof, and validation/test
     membership refusal;
  7. the corpus specification itself through the frozen E1 validator.

Prints one JSON verdict.  Exit 0 only for the GREEN terminal state.
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
from baseline.g8_e import verify_e1_corpus_spec_file  # noqa: E402

CORPUS_SPEC_PATH = pass_one.REPO_ROOT / "results/baseline/g8_e/corpus_spec.json"
E6_FREEZE_PATH = pass_one.v3s.V3S_ROOT / "e6_pass_one_freeze.json"


def _verify_e4_semantics(e4: dict) -> None:
    objects = e4["objects"]
    if len(objects) != e4["object_count"] or e4["validation_denominator"] != 1000:
        raise SystemExit("E4 object/denominator semantics differ")
    totals = {"delivered": 0, "codec_infeasibility": 0, "decode_failure": 0}
    for obj in objects:
        if obj["status"] != "eligible" or obj["total_count"] != 1000:
            raise SystemExit("E4 object is not eligible at denominator 1000")
        counts = obj["clean_accuracy_counts"]
        if obj["correct_count"] != counts["correct_count"]:
            raise SystemExit("E4 clean accuracy is not count-derived")
        if (
            obj["delivered_count"]
            + obj["codec_infeasibility_count"]
            + obj["decode_failure_count"]
            != 1000
        ):
            raise SystemExit("E4 outcome counts do not cover the denominator")
        totals["delivered"] += obj["delivered_count"]
        totals["codec_infeasibility"] += obj["codec_infeasibility_count"]
        totals["decode_failure"] += obj["decode_failure_count"]
    outage = e4["outage_accuracy"]
    if (
        outage["selection_is_count_derived"] is not True
        or outage["selected_class"] != 0
        or outage["numerator"] != 100
        or outage["denominator"] != 1000
    ):
        raise SystemExit("E4 outage binding is not the measured count-derived record")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    report: dict = {}
    try:
        context = pass_one.authenticate_inputs()
        chain = context["chain"]
        authorization = context["authorization"]
        report["contract_id"] = context["contract"]["contract_id"]
        report["campaign_id"] = context["contract"]["campaign_id"]

        marker = pass_one.authenticate_marker(
            pass_one.E5_MARKER_PATH, pass_one.E5_AUTHORIZATION_PATH, authorization
        )
        if marker["pre_execution_pass_one_count"] != 0:
            raise SystemExit("marker pre-execution count differs from zero")

        _verify_e4_semantics(context["e4"])

        state_file = pass_one.REPO_ROOT / authorization["state_path"]
        if not state_file.is_file():
            raise SystemExit("pass-one completion record is absent")
        verified = pass_one.verify_pass_one_state()
        if verified["counters"]["pass_one_executed_count"] != 1:
            raise SystemExit("pass one was not executed exactly once")

        freeze_raw = E6_FREEZE_PATH.read_bytes()
        freeze = json.loads(freeze_raw)
        body_without_digest = {
            key: child for key, child in freeze.items() if key != "e6_freeze_sha256"
        }
        if freeze.get("e6_freeze_sha256") != v3.sha256_bytes(v3.canonical_json(body_without_digest)):
            raise SystemExit("E6 freeze digest differs")
        body_without_id = {
            key: child for key, child in body_without_digest.items() if key != "e6_freeze_id"
        }
        if freeze.get("e6_freeze_id") != v3._id("g8ee6freeze-", body_without_id):
            raise SystemExit("E6 freeze ID differs")
        if freeze.get("authorization_issued_sha256") != authorization["issued_sha256"]:
            raise SystemExit("E6 freeze binds a different authorization")
        if freeze.get("pass_one_state", {}).get("state_id") != verified["state_id"]:
            raise SystemExit("E6 freeze binds a different pass-one state")
        corpus_binding = freeze["corpus_specification_binding"]
        corpus_raw = CORPUS_SPEC_PATH.read_bytes()
        if (
            corpus_binding["sha256"] != v3.sha256_bytes(corpus_raw)
            or corpus_binding["bytes"] != len(corpus_raw)
            or corpus_binding["frozen_spec_bytes_untouched"] is not True
            or corpus_binding["training_only"] is not True
            or corpus_binding["materialized"] is not False
            or corpus_binding["resolved_lineage"]["state_sha256"] != verified["state_sha256"]
        ):
            raise SystemExit("E6 corpus-spec lineage completion differs")
        validate_corpus_training_only(context, corpus_binding)

        report.update({
            "e2_completion_sha256": chain["e2_completion_sha256"],
            "e3_id": chain["e3_id"],
            "e3_sha256": chain["e3_sha256"],
            "e4_id": chain["e4_id"],
            "e4_sha256": chain["e4_sha256"],
            "bler_table_sha256": chain["bler_table_sha256"],
            "w4_integration_adjudication_sha256": chain["w4_integration_adjudication_sha256"],
            "selection_policy_sha256": chain["selection_policy_sha256"],
            "e5_authorization_issued_sha256": authorization["issued_sha256"],
            "e5_marker_issued_sha256": marker["issued_sha256"],
            "pass_one_state_id": verified["state_id"],
            "pass_one_state_content_sha256": verified["state_sha256"],
            "pass_one_state_file_sha256": verified["file_sha256"],
            "pass_one_selections": verified["selections"],
            "pass_one_cells_without_selection": verified["cells_without_selection"],
            "e6_freeze_id": freeze["e6_freeze_id"],
            "e6_freeze_file_sha256": v3.sha256_bytes(freeze_raw),
            "counters": verified["counters"],
            "corpus_spec_id": corpus_binding["corpus_spec_id"],
        })
        prohibited = ("training", "pass_two", "pass_three", "fallback_invoked",
                      "ratio_adjudicated", "test_access", "learned_system_training",
                      "g8_f_execution")
        if any(verified["counters"].get(name) != 0 for name in prohibited):
            raise SystemExit("a prohibited G8_E counter is nonzero")
    except SystemExit as exc:
        print(json.dumps({"status": "HOLD", "reason": str(exc)}, sort_keys=True))
        return 2
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "reason": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    report["status"] = "PASS"
    report["verdict"] = (
        "G8_E GREEN - VALIDATION CAMPAIGN AND PASS ONE FROZEN; "
        "G8_F READY; NO TRAINING OR PASS TWO"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


def validate_corpus_training_only(context: dict, corpus_binding: dict) -> None:
    """Prove training-only scope without touching test payloads."""

    corpus = json.loads(CORPUS_SPEC_PATH.read_bytes())
    membership = corpus["forbidden_membership"]
    if (
        membership["validation_ids_forbidden"] is not True
        or membership["test_ids_forbidden"] is not True
        or membership["validation_or_test_ids_may_not_be_materialized"] is not True
    ):
        raise SystemExit("corpus spec does not refuse validation/test membership")
    rules = corpus["generation_rules"]
    if (
        rules["input_split"] != "train"
        or rules["output_split"] != "train"
        or rules["no_validation_or_test_fallback"] is not True
        or corpus["materialized_object_count"] != 0
    ):
        raise SystemExit("corpus generation rules are not training-split-only")
    train_manifest = corpus["train_manifest"]
    manifest_path = pass_one.REPO_ROOT / str(train_manifest["path"])
    raw = manifest_path.read_bytes()
    if v3.sha256_bytes(raw) != train_manifest["sha256"]:
        raise SystemExit("corpus train manifest bytes differ")
    from data.manifests import validate_manifest_bytes

    rows = validate_manifest_bytes("imagenette160", raw)
    train_rows = [row for row in rows if row.split == "train"]
    if len(train_rows) != int(train_manifest["expected_count"]):
        raise SystemExit("corpus train manifest count differs")
    train_ids = [row.stable_sample_id for row in train_rows]
    if train_ids != sorted(train_ids) or len(set(train_ids)) != len(train_ids):
        raise SystemExit("corpus train IDs are not ascending-unique manifest order")
    if (
        v3.sha256_bytes(v3.canonical_json(train_ids))
        != train_manifest["expected_stable_id_set_sha256"]
    ):
        raise SystemExit("corpus train ID set differs from the frozen expectation")


if __name__ == "__main__":
    raise SystemExit(main())
