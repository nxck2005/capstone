#!/usr/bin/env python3
"""Synthetic no-codec/no-classifier transaction scale benchmark.

Every object created by this tool is NON-SCIENTIFIC, NON-SELECTION, and
MERGE-INELIGIBLE FOR PRODUCTION.  It never opens a dataset or production
runtime; its purpose is only to measure compact state bookkeeping.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline import g8_e_corrected_v2 as v2  # noqa: E402


def _fixture(n: int) -> tuple[dict[str, object], dict[str, object], tuple[dict[str, object], ...], dict[str, v2.SyntheticSample]]:
    authority = {
        "authority_id": "g8e-synthetic-authority-" + "a" * 64,
        "structural_identities": [{
            "structural_identity_id": "g8e-synthetic-structural-" + "b" * 64,
            "dataset": v2.INITIAL_DATASET,
            "dataset_role": "headline",
            "source_codec": "synthetic",
            "ratio": "r_1_2",
            "modulation": "qpsk",
            "ldpc_rate": "1/2",
            "encode_axis_px": 2,
            "packet_config_id": "synthetic-packet",
            "payload_budget_bytes": 4,
            "packet_accounting": {"payload_bytes": 4},
        }],
        "logical_candidate_to_structural_id": {},
    }
    samples: dict[str, v2.SyntheticSample] = {}
    units = []
    for ordinal in range(n):
        sample_id = f"synthetic-{ordinal:08d}"
        samples[sample_id] = v2.SyntheticSample(sample_id, ordinal % 10, b"source-" + sample_id.encode(), np.zeros((2, 2, 3), dtype=np.uint8))
        units.append({
            "work_unit_id": v2._work_unit_id(authority["structural_identities"][0]["structural_identity_id"], sample_id),
            "ordinal": ordinal,
            "measurement_identity_id": authority["structural_identities"][0]["structural_identity_id"],
            "logical_candidate_ids": [],
            "stable_sample_id": sample_id,
            "dataset": v2.INITIAL_DATASET,
            "split": v2.VALIDATION_SPLIT,
        })
    contract = {
        "campaign_id": "g8e-synthetic-campaign-" + "c" * 64,
        "contract_id": "g8e-synthetic-contract-" + "d" * 64,
        "execution_profile": {"profile_id": "synthetic-profile"},
        "source_manifest": {"source_commit": "synthetic"},
        "direct_upstream_bindings": {"synthetic": True},
        "outage_policy": {"selected_class": 0, "selection_is_count_derived": True, "path": "synthetic", "sha256": "e" * 64},
        "codec": {"configuration_hash": "f" * 64, "runtime_identity": "synthetic"},
    }
    return authority, contract, tuple(units), samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.n < 10000:
        parser.error("--n must be at least 10000")
    owned = args.root is None
    root = args.root or Path(tempfile.mkdtemp(prefix="g8e-v2-scale-"))
    try:
        authority, contract, units, samples = _fixture(args.n)
        key = v2.PhysicalCacheKey("1" * 64, "2" * 64, (2, 2, 3), 4, 2, "f" * 64, "synthetic")
        codec = v2.CodecArtifactV2(key, v2.OUTCOME_CODEC_INFEASIBILITY, "synthetic_budget", None, None, "g8e-synthetic-codec-" + "0" * 64, False)

        def executor(unit, sample):
            return v2.MeasurementRecordV2.build(
                campaign_id=contract["campaign_id"], contract_id=contract["contract_id"], authority=authority,
                work_unit=unit, structural=authority["structural_identities"][0], sample=sample,
                physical_key=key, codec=codec, reconstruction=None, observation=None,
                outage_policy=contract["outage_policy"], profile_id="synthetic-profile", source_commit="synthetic",
                g8_c_linkage_digest=v2.sha256_bytes(v2.canonical_json(contract["direct_upstream_bindings"])),
                record_labels=("NON-SCIENTIFIC", "NON-SELECTION", "NOT PRODUCTION E2 EVIDENCE", "MERGE-INELIGIBLE FOR PRODUCTION"),
            )

        start = time.perf_counter()
        campaign = v2.AtomicE2CampaignV2(
            runtime_root=root, contract=contract, authority=authority, work_units=units,
            executor=executor, sample_provider=lambda sample_id: samples[sample_id], mode="start",
        )
        campaign.run_all()
        elapsed = time.perf_counter() - start
        state_size = (root / "campaign_state.json").stat().st_size
        checkpoint_size = sum(path.stat().st_size for path in (root / "checkpoints").glob("*.json"))
        record_size = sum(path.stat().st_size for path in (root / "records").glob("*.json"))
        resume = v2.AtomicE2CampaignV2(
            runtime_root=root, contract=contract, authority=authority, work_units=units,
            executor=executor, sample_provider=lambda sample_id: samples[sample_id], mode="resume",
        )
        result = {
            "status": "PASS",
            "labels": ["NON-SCIENTIFIC", "NON-SELECTION", "NOT PRODUCTION E2 EVIDENCE", "MERGE-INELIGIBLE FOR PRODUCTION"],
            "n": args.n,
            "runtime_seconds": elapsed,
            "state_size_bytes": state_size,
            "checkpoint_bytes_on_disk": checkpoint_size,
            "scientific_record_bytes_on_disk": record_size,
            "state_publications": campaign.state_publications,
            "state_bytes_written": campaign.state_bytes_written,
            "checkpoint_bytes_written": campaign.checkpoint_bytes_written,
            "record_validation_visits_normal": campaign.reconciliation_record_visits,
            "record_validation_visits_one_resume_reconciliation": resume.reconciliation_record_visits,
            "aggregate_directory_present": (root / "aggregates").exists(),
            "normal_transaction_complexity": "O(1) per unit; O(N) cumulative",
            "resume_reconciliation_complexity": "O(N) once",
            "extrapolation_288000": {
                # Each unit has an atomic claim publication and an atomic
                # completion publication, in addition to the initial state.
                "normal_state_publications": 2 * 288000 + 1,
                "normal_prefix_record_validation_visits": 0,
                "one_resume_reconciliation_record_visits": 288000,
                "state_bytes_written_linear_estimate": campaign.state_bytes_written * 288000 // args.n,
                "checkpoint_count": (288000 + v2.CHECKPOINT_INTERVAL - 1) // v2.CHECKPOINT_INTERVAL,
            },
        }
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered)
        print(rendered, end="")
        return 0
    finally:
        if owned:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
