#!/usr/bin/env python3
"""Prove the complete v3 lifecycle with a tiny merge-ineligible fixture."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from baseline import g8_e_corrected_v3 as v3  # noqa: E402
import aggregate_g8_e_corrected_v3 as aggregate_cli  # noqa: E402
import merge_g8_e_corrected_v3 as merge_cli  # noqa: E402
import run_g8_e_corrected_v3 as runner_cli  # noqa: E402


LABELS = (
    "NON-SCIENTIFIC",
    "NON-SELECTION",
    "NOT PRODUCTION E2 EVIDENCE",
    "MERGE-INELIGIBLE FOR PRODUCTION",
)


class _SyntheticBR11:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def as_dict(self) -> dict[str, Any]:
        return self.payload


def fixture_context() -> dict[str, Any]:
    structural: list[dict[str, Any]] = []
    for index, (budget, modulation, rate) in enumerate(
        ((10, "qpsk", "1/2"), (10, "bpsk", "1/3"), (11, "qam16", "2/3"))
    ):
        structural.append({
            "structural_identity_id": f"g8e-v3-synthetic-structural-{chr(97 + index) * 64}",
            "dataset": v3.INITIAL_DATASET,
            "dataset_role": "headline",
            "source_codec": "jpeg2000",
            "ratio": "r_1_2",
            "modulation": modulation,
            "ldpc_rate": rate,
            "encode_axis_px": 2,
            "packet_config_id": f"v3-synthetic-packet-{index}",
            "payload_budget_bytes": budget,
            "packet_accounting": {"payload_bytes": budget},
        })
    authority = {
        "authority_id": "g8e-v3-synthetic-authority-" + "b" * 64,
        "structural_identities": structural,
        "logical_candidate_to_structural_id": {},
    }
    source_sha = "7" * 64
    data = {
        "data_identity_id": "g8e-v3-synthetic-data-" + "8" * 64,
    }
    direct = {"synthetic": True, "labels": list(LABELS)}
    classifier_sha = v3.sha256_bytes(b"synthetic-or-contract-bound-classifier")
    contract = {
        "campaign_id": "g8e-v3-synthetic-campaign-" + "c" * 64,
        "contract_id": "g8e-v3-synthetic-contract-" + "d" * 64,
        "execution_profile": {"profile_id": "synthetic-profile"},
        "source_manifest": {"source_commit": "synthetic-v3", "id": "synthetic-source", "sha256": source_sha},
        "authority": {"sha256": v3.sha256_bytes(v3.canonical_json(authority))},
        "scientific_data_identity": {"id": data["data_identity_id"], "sha256": "9" * 64, "manifest_sha256": "6" * 64},
        "direct_upstream_bindings": direct,
        "outage_policy": {
            "selected_class": 2,
            "numerator": 1,
            "denominator": 3,
            "selection_is_count_derived": True,
            "path": "synthetic/outage.json",
            "sha256": "e" * 64,
        },
        "codec": {"configuration_hash": "f" * 64, "runtime_identity": "synthetic-codec"},
        "classifier": {
            "checkpoint_sha256": classifier_sha,
            "config_identity": "synthetic-g1-config",
            "runtime_identity": "synthetic-g1-runtime",
        },
        "authorization": {"scope": {"phase": "E2", "synthetic_fixture_only": True}},
    }
    samples: list[v3.SyntheticSample] = []
    for index, (sample_id, label) in enumerate(
        (("v3-synthetic-delivered", 0), ("v3-synthetic-infeasible", 2), ("v3-synthetic-decode", 1))
    ):
        samples.append(v3.SyntheticSample(
            sample_id,
            label,
            sample_id.encode(),
            np.full((2, 2, 3), index, dtype=np.uint8),
        ))
    pixel_to_id = {
        v3.sha256_bytes(sample.canonical_pixels.tobytes()): sample.stable_sample_id
        for sample in samples
    }

    class Backend:
        calls = 0

        def encode_to_budget(self, image: np.ndarray, **kwargs: Any) -> SimpleNamespace:
            self.calls += 1
            sample_id = pixel_to_id[kwargs["canonical_pixels_sha256"]]
            if sample_id == "v3-synthetic-infeasible":
                return SimpleNamespace(feasible=False, codestream=None, emitted_byte_count=None, reason="synthetic budget")
            stream = b"decode" if sample_id == "v3-synthetic-decode" else (
                b"delivered-c" if kwargs["budget_bytes"] == 11 else b"delivered"
            )
            return SimpleNamespace(feasible=True, codestream=stream, emitted_byte_count=len(stream), reason=None)

    class Classifier:
        calls = 0

        def predict(self, pixels: np.ndarray) -> int:
            self.calls += 1
            return 0

    def decoder(stream: bytes) -> np.ndarray | v3.ScientificDecodeFailure:
        if stream == b"decode":
            return v3.ScientificDecodeFailure("synthetic explicit decoder outcome")
        return np.zeros((2, 2, 3), dtype=np.uint8)

    sample_ids = tuple(sorted(sample.stable_sample_id for sample in samples))
    work_units = v3.expected_work_units(authority, sample_ids)
    contract["transaction"] = {
        "production_total_required": len(work_units),
        "production_authority_order_sha256": v3.sha256_bytes(
            v3.canonical_json([unit["work_unit_id"] for unit in work_units])
        ),
    }

    return {
        "labels": LABELS,
        "authority": authority,
        "contract": contract,
        "data_identity": data,
        "samples": tuple(samples),
        "sample_ids": sample_ids,
        "sample_labels": {sample.stable_sample_id: sample.label for sample in samples},
        "backend": Backend(),
        "decoder": decoder,
        "classifier": Classifier(),
    }


def _authorization(path: Path, fixture: dict[str, Any]) -> None:
    contract = fixture["contract"]
    data = fixture["data_identity"]
    body = {
        "schema_version": v3.V3_SCHEMA_VERSION,
        "artifact_role": "g8_e_v3_owner_e2_authorization",
        "status": "AUTHORIZED",
        "authorized_by": "synthetic-lifecycle-proof",
        "reason": "NON-SCIENTIFIC fixture lifecycle only",
        "campaign_id": contract["campaign_id"],
        "contract_id": contract["contract_id"],
        "source_manifest_id": contract["source_manifest"]["id"],
        "source_manifest_sha256": contract["source_manifest"]["sha256"],
        "data_identity_id": data["data_identity_id"],
        "data_identity_sha256": contract["scientific_data_identity"]["sha256"],
        "profile_id": contract["execution_profile"]["profile_id"],
        "scope": contract["authorization"]["scope"],
    }
    body["issued_sha256"] = v3.sha256_bytes(v3.canonical_json(body))
    v3._atomic_publish(path, v3.rendered_json(body))


def _br11_stub(*args: Any, **kwargs: Any) -> _SyntheticBR11:
    stream = args[0]
    identity = kwargs["emitted_file_identity"]
    return _SyntheticBR11({
        "verdict": kwargs["verdict"],
        "emitted_file_identity": identity.payload(),
        "bytes_sent": kwargs["bytes_sent"],
        "emitted_codestream_bytes": len(stream),
        "header_bytes": 0,
        "payload_bytes": len(stream),
        "payload_filler_bytes": kwargs["bytes_sent"] - len(stream),
        "codestream_sha256": v3.sha256_bytes(stream),
        "denominator": 1,
        "record_labels": list(LABELS),
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=v3.V3_SYNTHETIC_PROOF_PATH)
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="g8e-v3-lifecycle-"))
    try:
        import baseline.g8_d as g8d

        original = g8d.account_br11
        g8d.account_br11 = _br11_stub
        try:
            predata = v3.verify_v3_predata_zero_state()
            fixture = fixture_context()
            authorization_path = root / "synthetic-owner-authorization.json"
            runtime = root / "runtime"
            _authorization(authorization_path, fixture)
            frozen_after_authorization = v3.verify_v3_frozen_contract()
            common = [
                "--campaign-id", fixture["contract"]["campaign_id"],
                "--runtime-root", str(runtime),
                "--authorization", str(authorization_path),
            ]
            partial_fixture = dict(fixture)
            partial_fixture["max_units"] = 3
            if runner_cli.main(["--start", *common], fixture=partial_fixture) != 0:
                raise v3.G8EV3Error("synthetic production entry-point start failed")
            partial_state = v3._state_for_runtime(runtime)
            del partial_fixture  # process interruption: no in-memory campaign survives
            frozen_during_e2 = v3.verify_v3_frozen_contract()
            resume_fixture = dict(fixture)
            resume_fixture["max_units"] = None
            if runner_cli.main(["--resume", *common], fixture=resume_fixture) != 0:
                raise v3.G8EV3Error("synthetic production entry-point resume failed")
            completed, completion_sha = v3.verify_e2_completion_artifact(
                runtime_root=runtime,
                contract=fixture["contract"],
                authority=fixture["authority"],
                production=False,
            )
            merge_fixture = {
                "contract": fixture["contract"],
                "authority": fixture["authority"],
                "sample_ids": fixture["sample_ids"],
                "sample_labels": fixture["sample_labels"],
            }
            if merge_cli.main(["--execute", "--runtime-root", str(runtime)], fixture=merge_fixture) != 0:
                raise v3.G8EV3Error("synthetic production E3 entry point failed")
            e3_path = runtime / "e3_exact_set_closure.json"
            e3_sha = v3.sha256_file(e3_path)
            e3 = v3.verify_e3_artifact(e3_path, contract=fixture["contract"], expected_sha256=e3_sha)
            if aggregate_cli.main(
                ["--execute", "--runtime-root", str(runtime), "--e3", str(e3_path), "--e3-sha256", e3_sha],
                fixture=merge_fixture,
            ) != 0:
                raise v3.G8EV3Error("synthetic production E4 entry point failed")
            e4_path = runtime / "e4_count_derived.json"
            e4 = v3.verify_e4_artifact(
                e4_path,
                contract=fixture["contract"],
                e3_path=e3_path,
                e3_sha256=e3_sha,
            )
            try:
                v3.build_e3_artifact(
                    authority=fixture["authority"],
                    sample_ids=fixture["sample_ids"],
                    sample_labels=fixture["sample_labels"],
                    runtime_root=runtime,
                    contract=fixture["contract"],
                    production=True,
                )
            except v3.G8EV3Error:
                production_rejection = True
            else:
                production_rejection = False
            if not production_rejection:
                raise v3.G8EV3Error("production E3 accepted merge-ineligible fixture evidence")
            proof = {
                "schema_version": v3.V3_SCHEMA_VERSION,
                "artifact_role": "g8_e_v3_non_scientific_full_lifecycle_proof",
                "status": "PASS",
                "record_labels": list(LABELS),
                "production_runtime": False,
                "production_e2_evidence": False,
                "stages": {
                    "predata_zero_state": predata["phase"] == "PRE_DATA_ZERO",
                    "synthetic_authorization_authenticated": True,
                    "actual_runner_start": partial_state["completed_prefix_count"] == 3,
                    "interruption_after_partial_e2": True,
                    "immutable_verifier_after_authorization": frozen_after_authorization["contract_sha256"] == predata["contract_sha256"],
                    "immutable_verifier_during_e2": frozen_during_e2["contract_sha256"] == predata["contract_sha256"],
                    "actual_runner_resume": True,
                    "exact_e2_completion": completed["status"] == "E2_COMPLETE",
                    "actual_e3_cli": e3["status"] == "E3_COMPLETE",
                    "actual_e4_cli": e4["status"] == "E4_COMPLETE",
                    "production_campaign_rejects_fixture": production_rejection,
                },
                "counts": {
                    "work_units": completed["completed_work_unit_count"],
                    "e3_records": e3["observed_work_unit_count"],
                    "e4_objects": e4["object_count"],
                    "e4_record_traversals": e4["record_traversal_count"],
                },
                "bindings": {
                    "completion_sha256": completion_sha,
                    "e3_id": e3["e3_id"],
                    "e3_sha256": e3_sha,
                    "e4_id": e4["e4_id"],
                    "e4_sha256": v3.sha256_file(e4_path),
                },
                "safety": {
                    "pass_one": False,
                    "training": 0,
                    "pass_two": 0,
                    "fallback": False,
                    "ratio_adjudication": False,
                    "test_access": 0,
                },
                "scientific_meaning": "none; fixture is synthetic and merge-ineligible for production",
            }
            v3._atomic_publish(args.output, v3.rendered_json(proof))
            print({
                "status": proof["status"],
                "work_units": proof["counts"]["work_units"],
                "e3_id": e3["e3_id"],
                "e4_id": e4["e4_id"],
                "output": str(args.output),
            })
            return 0
        finally:
            g8d.account_br11 = original
    except (OSError, v3.G8EV3Error) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
