#!/usr/bin/env python3
"""Benchmark v3 E2/E3/E4 bookkeeping with merge-ineligible records only."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline import g8_e_corrected_v3 as v3  # noqa: E402


LABELS = (
    "NON-SCIENTIFIC",
    "NON-SELECTION",
    "NOT PRODUCTION E2 EVIDENCE",
    "MERGE-INELIGIBLE FOR PRODUCTION",
)


def _context(n: int) -> tuple[dict[str, Any], dict[str, Any], tuple[dict[str, Any], ...], dict[str, v3.SyntheticSample]]:
    structural = {
        "structural_identity_id": "g8e-v3-scale-structural-" + "b" * 64,
        "dataset": v3.INITIAL_DATASET,
        "dataset_role": "headline",
        "source_codec": "jpeg2000",
        "ratio": "r_1_2",
        "modulation": "qpsk",
        "ldpc_rate": "1/2",
        "encode_axis_px": 2,
        "packet_config_id": "v3-scale-packet",
        "payload_budget_bytes": 4,
        "packet_accounting": {"payload_bytes": 4},
    }
    authority = {
        "authority_id": "g8e-v3-scale-authority-" + "a" * 64,
        "structural_identities": [structural],
        "logical_candidate_to_structural_id": {},
    }
    contract = {
        "campaign_id": "g8e-v3-scale-campaign-" + "c" * 64,
        "contract_id": "g8e-v3-scale-contract-" + "d" * 64,
        "execution_profile": {"profile_id": "synthetic-scale-profile"},
        "source_manifest": {"source_commit": "synthetic-scale", "id": "synthetic-scale-source", "sha256": "1" * 64},
        "authority": {"sha256": v3.sha256_bytes(v3.canonical_json(authority))},
        "scientific_data_identity": {"id": "synthetic-scale-data", "sha256": "2" * 64, "manifest_sha256": "3" * 64},
        "direct_upstream_bindings": {"synthetic": True, "labels": list(LABELS)},
        "outage_policy": {
            "selected_class": 0,
            "numerator": 1,
            "denominator": 10,
            "selection_is_count_derived": True,
            "path": "synthetic",
            "sha256": "e" * 64,
        },
        "codec": {"configuration_hash": "f" * 64, "runtime_identity": "synthetic-scale-codec"},
    }
    samples: dict[str, v3.SyntheticSample] = {}
    for ordinal in range(n):
        sample_id = f"v3-scale-{ordinal:08d}"
        samples[sample_id] = v3.SyntheticSample(
            sample_id,
            ordinal % 10,
            b"source-" + sample_id.encode(),
            np.zeros((2, 2, 3), dtype=np.uint8),
        )
    units = v3.expected_work_units(authority, tuple(sorted(samples)))
    return authority, contract, units, samples


def _run_one(n: int) -> dict[str, Any]:
    root = Path(tempfile.mkdtemp(prefix=f"g8e-v3-scale-{n}-"))
    try:
        authority, contract, units, samples = _context(n)
        structural = authority["structural_identities"][0]
        key = v3.PhysicalCacheKey("1" * 64, "2" * 64, (2, 2, 3), 4, 2, "f" * 64, "synthetic-scale-codec")
        codec = v3.v2.CodecArtifactV2(
            key,
            v3.v2.OUTCOME_CODEC_INFEASIBILITY,
            "synthetic budget",
            None,
            None,
            "g8e-v3-scale-codec-" + "0" * 64,
            False,
        )

        def executor(unit: dict[str, Any], sample: v3.SyntheticSample) -> v3.MeasurementRecordV3:
            return v3.MeasurementRecordV3.build(
                campaign_id=contract["campaign_id"],
                contract_id=contract["contract_id"],
                authority=authority,
                work_unit=unit,
                structural=structural,
                sample=sample,
                physical_key=key,
                codec=codec,
                reconstruction=None,
                observation=None,
                outage_policy=contract["outage_policy"],
                profile_id=contract["execution_profile"]["profile_id"],
                source_commit=contract["source_manifest"]["source_commit"],
                g8_c_linkage_digest=v3.sha256_bytes(v3.canonical_json(contract["direct_upstream_bindings"])),
                record_labels=LABELS,
            )

        start = time.perf_counter()
        campaign = v3.AtomicE2CampaignV3(
            runtime_root=root,
            contract=contract,
            authority=authority,
            work_units=units,
            executor=executor,
            sample_provider=lambda sample_id: samples[sample_id],
            mode="start",
        )
        campaign.run_all()
        e2_seconds = time.perf_counter() - start
        e2 = campaign.instrumentation()
        e3_counters: dict[str, int] = defaultdict(int)
        start = time.perf_counter()
        e3, e3_path, e3_sha = v3.publish_e3_artifact(
            authority=authority,
            sample_ids=tuple(sorted(samples)),
            sample_labels={sample_id: sample.label for sample_id, sample in samples.items()},
            runtime_root=root,
            contract=contract,
            production=False,
            authenticate_caches=False,
            instrumentation=e3_counters,
        )
        e3_seconds = time.perf_counter() - start
        e4_counters: dict[str, int] = defaultdict(int)
        start = time.perf_counter()
        e4 = v3.build_e4_artifact(
            authority=authority,
            sample_ids=tuple(sorted(samples)),
            runtime_root=root,
            contract=contract,
            e3_path=e3_path,
            e3_sha256=e3_sha,
            production=False,
            instrumentation=e4_counters,
        )
        e4_seconds = time.perf_counter() - start
        return {
            "n": n,
            "e2": {
                "runtime_seconds": e2_seconds,
                "runtime_seconds_per_unit": e2_seconds / n,
                **e2,
            },
            "e3": {
                "runtime_seconds": e3_seconds,
                "runtime_seconds_per_unit": e3_seconds / n,
                **dict(e3_counters),
                "required": e3["required_work_unit_count"],
                "observed": e3["observed_work_unit_count"],
            },
            "e4": {
                "runtime_seconds": e4_seconds,
                "runtime_seconds_per_unit": e4_seconds / n,
                **dict(e4_counters),
                "object_count": e4["object_count"],
            },
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=(2500, 5000, 10000, 20000))
    parser.add_argument("--output", type=Path, default=v3.V3_COMPLEXITY_PATH)
    args = parser.parse_args()
    if tuple(args.sizes) != tuple(sorted(set(args.sizes))) or any(size <= 0 for size in args.sizes):
        parser.error("--sizes must be unique positive increasing integers")
    rows = [_run_one(size) for size in args.sizes]
    for row in rows:
        e2 = row["e2"]
        e3 = row["e3"]
        e4 = row["e4"]
        if e2["authority_order_digest_computations"] != 1 or e2["full_authority_id_visits_during_normal_progression"] != 0:
            raise v3.G8EV3Error("v3 transaction authority instrumentation violates the O(N) contract")
        if e3["records_parsed"] != row["n"] or e3["structural_lookup_operations"] != row["n"]:
            raise v3.G8EV3Error("v3 E3 instrumentation violates the linear operation contract")
        if e4["e3_artifact_verifications"] != 1 or e4["record_traversals"] != row["n"] or e4["object_aggregation_operations"] != row["n"]:
            raise v3.G8EV3Error("v3 E4 instrumentation violates the linear operation contract")
    largest = rows[-1]
    n = largest["n"]
    target = 288000
    evidence = {
        "schema_version": v3.V3_SCHEMA_VERSION,
        "artifact_role": "g8_e_v3_linear_scale_evidence",
        "status": "PASS",
        "record_labels": list(LABELS),
        "sizes": list(args.sizes),
        "rows": rows,
        "code_level_proof": {
            "e2": "one cached authority-order digest at construction; _load_state compares persisted digest to the cached scalar; normal progression performs zero full-authority visits",
            "e3": "one expected_by_id hash map, one structural_by_id hash map, one record scan and one authority-order scan; every membership/structural lookup is O(1)",
            "e4": "one O(1) E3 artifact authentication and one authority-order record traversal; no second complete E3 ingest",
            "e2_total": "O(N)",
            "e3_total": "O(N)",
            "e4_total": "O(N)",
        },
        "extrapolation_288000": {
            "basis_n": n,
            "e2_seconds_linear": largest["e2"]["runtime_seconds"] * target / n,
            "e3_seconds_linear": largest["e3"]["runtime_seconds"] * target / n,
            "e4_seconds_linear": largest["e4"]["runtime_seconds"] * target / n,
            "authority_digest_computations_per_process": 1,
            "full_authority_id_visits_during_normal_progression": 0,
            "normal_state_publications": 2 * target + 1,
            "checkpoint_count": (target + v3.v2.CHECKPOINT_INTERVAL - 1) // v3.v2.CHECKPOINT_INTERVAL,
        },
        "scientific_meaning": "none; no codec, classifier, dataset, selection or production runtime was opened",
    }
    v3._atomic_publish(args.output, v3.rendered_json(evidence))
    print({
        "status": "PASS",
        "sizes": list(args.sizes),
        "e2_authority_digest_computations": [row["e2"]["authority_order_digest_computations"] for row in rows],
        "e3_records_parsed": [row["e3"]["records_parsed"] for row in rows],
        "e4_record_traversals": [row["e4"]["record_traversals"] for row in rows],
        "output": str(args.output),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
