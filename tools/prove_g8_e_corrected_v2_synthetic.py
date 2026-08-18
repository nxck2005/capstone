#!/usr/bin/env python3
"""Run a bounded, wholly synthetic G8_E v2 E2->E3->E4 proof.

No dataset, validation manifest, classifier checkpoint, test split, selection,
or production runtime is opened.  Every object is explicitly marked
NON-SCIENTIFIC, NON-SELECTION, NOT PRODUCTION E2 EVIDENCE, and
MERGE-INELIGIBLE FOR PRODUCTION.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO / "src"))

from baseline import g8_e_corrected_v2 as v2  # noqa: E402


LABELS = ("NON-SCIENTIFIC", "NON-SELECTION", "NOT PRODUCTION E2 EVIDENCE", "MERGE-INELIGIBLE FOR PRODUCTION")


class _SyntheticBR11:
    def as_dict(self):
        return {"synthetic": True, "record_labels": list(LABELS)}


def _fixture():
    structural = []
    for index, (budget, modulation, rate) in enumerate(((10, "qpsk", "1/2"), (10, "bpsk", "1/3"), (11, "qam16", "2/3"))):
        structural.append({
            "structural_identity_id": f"g8e-synthetic-structural-{chr(97 + index) * 64}",
            "dataset": v2.INITIAL_DATASET,
            "dataset_role": "headline",
            "source_codec": "jpeg2000",
            "ratio": "r_1_2",
            "modulation": modulation,
            "ldpc_rate": rate,
            "encode_axis_px": 2,
            "packet_config_id": f"synthetic-packet-{index}",
            "payload_budget_bytes": budget,
            "packet_accounting": {"payload_bytes": budget},
        })
    authority = {
        "authority_id": "g8e-synthetic-authority-" + "b" * 64,
        "structural_identities": structural,
        "logical_candidate_to_structural_id": {},
    }
    direct = {"synthetic": True, "upstream": "frozen-linkage"}
    contract = {
        "campaign_id": "g8e-synthetic-campaign-" + "c" * 64,
        "contract_id": "g8e-synthetic-contract-" + "d" * 64,
        "execution_profile": {"profile_id": "synthetic-profile"},
        "source_manifest": {"source_commit": "synthetic", "id": "synthetic-source"},
        "direct_upstream_bindings": direct,
        "outage_policy": {
            "selected_class": 2, "numerator": 1, "denominator": 3,
            "selection_is_count_derived": True, "path": "synthetic/outage.json",
            "sha256": "e" * 64,
        },
        "codec": {"configuration_hash": "f" * 64, "runtime_identity": "synthetic-codec"},
    }
    samples = {}
    for index, (sample_id, label) in enumerate((("synthetic-delivered", 0), ("synthetic-infeasible", 2), ("synthetic-decode", 1))):
        samples[sample_id] = v2.SyntheticSample(sample_id, label, sample_id.encode(), np.full((2, 2, 3), index, dtype=np.uint8))
    units = []
    for structural in structural:
        for sample_id in sorted(samples):
            units.append({
                "work_unit_id": v2._work_unit_id(structural["structural_identity_id"], sample_id),
                "ordinal": len(units),
                "measurement_identity_id": structural["structural_identity_id"],
                "logical_candidate_ids": [],
                "stable_sample_id": sample_id,
                "dataset": v2.INITIAL_DATASET,
                "split": v2.VALIDATION_SPLIT,
            })
    return authority, contract, tuple(units), samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=v2.V2_SYNTHETIC_PROOF_PATH)
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="g8e-v2-synthetic-"))
    try:
        import baseline.g8_d as g8d
        original_br11 = g8d.account_br11
        g8d.account_br11 = lambda *args, **kwargs: _SyntheticBR11()
        try:
            authority, contract, units, samples = _fixture()
            pixel_hash_to_id = {v2.sha256_bytes(sample.canonical_pixels.tobytes()): sample_id for sample_id, sample in samples.items()}
            backend_calls = {"count": 0}

            class Backend:
                def encode_to_budget(self, image, **kwargs):
                    backend_calls["count"] += 1
                    sample_id = pixel_hash_to_id[kwargs["canonical_pixels_sha256"]]
                    if sample_id == "synthetic-infeasible":
                        return SimpleNamespace(feasible=False, codestream=None, emitted_byte_count=None, reason="synthetic budget")
                    stream = b"decode" if sample_id == "synthetic-decode" else (b"delivered-c" if kwargs["budget_bytes"] == 11 else b"delivered")
                    return SimpleNamespace(feasible=True, codestream=stream, emitted_byte_count=len(stream))

            class Classifier:
                def __init__(self):
                    self.calls = 0

                def predict(self, pixels):
                    self.calls += 1
                    return 0

            classifier = Classifier()

            def decoder(stream):
                if stream == b"decode":
                    return v2.ScientificDecodeFailure("synthetic explicit decoder outcome")
                return np.zeros((2, 2, 3), dtype=np.uint8)

            executor = v2.MeasurementExecutorV2(
                contract=contract, authority=authority, runtime_root=root,
                backend=Backend(), decoder=decoder, classifier=classifier,
                non_scientific_fixture=True,
            )
            by_id = samples
            campaign = v2.AtomicE2CampaignV2(
                runtime_root=root, contract=contract, authority=authority, work_units=units,
                executor=executor.execute, sample_provider=lambda sid: by_id[sid], mode="start",
            )
            campaign.run_all()
            e3 = v2.merge_e3_records_v2(
                authority=authority, sample_ids=tuple(sorted(samples)), runtime_root=root,
                contract=contract, production=False,
            )
            e4 = v2.aggregate_e4_counts_v2(
                authority=authority, sample_ids=tuple(sorted(samples)), runtime_root=root,
                contract=contract, production=False,
            )

            claim_root = root / "claim-crash"
            claim_campaign = v2.AtomicE2CampaignV2(
                runtime_root=claim_root, contract=contract, authority=authority,
                work_units=units[:1], executor=executor.execute,
                sample_provider=lambda sid: by_id[sid], mode="start",
            )
            try:
                claim_campaign.run_next(crash_after="claim")
            except RuntimeError:
                pass
            claim_resume = v2.AtomicE2CampaignV2(
                runtime_root=claim_root, contract=contract, authority=authority,
                work_units=units[:1], executor=executor.execute,
                sample_provider=lambda sid: by_id[sid], mode="resume",
            )

            record_root = root / "record-crash"
            record_campaign = v2.AtomicE2CampaignV2(
                runtime_root=record_root, contract=contract, authority=authority,
                work_units=units[:1], executor=executor.execute,
                sample_provider=lambda sid: by_id[sid], mode="start",
            )
            try:
                record_campaign.run_next(crash_after="record")
            except RuntimeError:
                pass
            record_resume = v2.AtomicE2CampaignV2(
                runtime_root=record_root, contract=contract, authority=authority,
                work_units=units[:1], executor=executor.execute,
                sample_provider=lambda sid: by_id[sid], mode="resume",
            )

            hold_root = root / "hold"
            hold_campaign = v2.AtomicE2CampaignV2(
                runtime_root=hold_root, contract=contract, authority=authority,
                work_units=units[:1],
                executor=lambda unit, sample: (_ for _ in ()).throw(RuntimeError("synthetic runtime failure")),
                sample_provider=lambda sid: by_id[sid], mode="start",
            )
            try:
                hold_campaign.run_next()
            except v2.CampaignHoldError:
                pass

            proof = {
                "schema_version": v2.V2_SCHEMA_VERSION,
                "artifact_role": "g8_e_v2_synthetic_end_to_end_proof",
                "status": "PASS",
                "labels": list(LABELS),
                "production_runtime": False,
                "production_e2_evidence": False,
                "selection": False,
                "test_access": 0,
                "training": 0,
                "pass_one": False,
                "pass_two": False,
                "fallback": False,
                "ratio_adjudication": False,
                "scenarios": {
                    "delivered_row": True,
                    "image_level_codec_infeasibility": True,
                    "typed_decode_failure": True,
                    "exact_outage_scoring": True,
                    "cache_miss_and_hit": True,
                    "cross_mcs_same_physical_key_reuse": True,
                    "different_payload_budget_no_alias": True,
                    "classifier_observation_reuse": True,
                    "crash_after_claim_resume_visits": claim_resume.reconciliation_record_visits,
                    "crash_after_record_resume_visits": record_resume.reconciliation_record_visits,
                    "runtime_exception_hold": hold_campaign.state()["status"] == v2.HOLD_STATUS,
                    "e3_exact_set": e3["work_unit_count"] == 9,
                    "e4_count_derived": e4["object_count"] == 3,
                },
                "counts": {
                    "synthetic_work_units": len(units),
                    "synthetic_e3_records": e3["work_unit_count"],
                    "synthetic_e4_objects": e4["object_count"],
                    "backend_calls": backend_calls["count"],
                    "classifier_calls": classifier.calls,
                    "codec_cache_objects": len(list((root / "codec").glob("*.json"))),
                    "reconstruction_cache_objects": len(list((root / "reconstruction").glob("*.json"))),
                    "observation_cache_objects": len(list((root / "observation").glob("*.json"))),
                },
                "scientific_meaning": "none; this proof is merge-ineligible for production",
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(v2.rendered_json(proof))
            print(json.dumps(proof, sort_keys=True))
            return 0
        finally:
            g8d.account_br11 = original_br11
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
