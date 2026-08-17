#!/usr/bin/env python3
"""Run the explicitly non-scientific G8_D pipeline smoke.

This command uses synthetic pixels, a synthetic JPEG 2000 backend and a
synthetic decoder.  It never opens a dataset payload, loads a checkpoint,
invokes a classifier, constructs a selection authorization, or writes a
production campaign root.  Its output is a bounded plumbing witness only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline import g8_d  # noqa: E402


SMOKE_SCHEMA_VERSION = 1
SMOKE_ARTIFACT_ROLE = "g8_d_bounded_non_scientific_smoke"
SMOKE_SOURCE = "d6-synthetic-non-scientific-smoke"
MUTATION_CASES = (
    "wrong_g8_c_table_binding",
    "predecessor_table_instead_of_pascal_successor",
    "wrong_validation_manifest",
    "test_split_requested",
    "changed_classifier_checkpoint",
    "codec_configuration_mutation",
    "image_content_mutation",
    "cache_key_alias_attempt",
    "emitted_bytes_exceed_budget",
    "structural_infeasibility",
    "codec_infeasibility",
    "reconstruction_cache_corruption",
    "br11_accounting_mutation",
    "incorrect_accuracy_counts",
    "assumed_bare_accuracy",
    "duplicate_work_unit",
    "missing_work_unit",
    "stale_aggregate",
    "interrupted_publication",
    "changed_source_or_config_on_resume",
)


def _codestream(payload: bytes = b"abc") -> bytes:
    sot = b"\xff\x90\x00\x0a\x00\x00" + (17).to_bytes(4, "big") + b"\x00\x00"
    return b"\xff\x4f" + sot + b"\xff\x93" + payload + b"\xff\xd9"


class _SmokeCodec:
    def __init__(self, *, infeasible: bool = False, codestream: bytes | None = None) -> None:
        self.infeasible = infeasible
        self.codestream = _codestream() if codestream is None else codestream
        self.calls = 0
        self.snapshot = {
            "baseline": {
                "source_codec": "jpeg2000",
                "rate_control": "largest_codestream_within_budget",
                "smoke_backend": "synthetic",
            },
            "preprocessing": {
                "downsample": "bilinear",
                "upsample": "bicubic",
                "preserves_aspect": True,
            },
            "environment": {"openjpeg": "synthetic", "binding": "synthetic"},
        }
        self.configuration_hash = g8_d.sha256_bytes(g8_d.canonical_json(self.snapshot))

    def encode_to_budget(self, image: np.ndarray, **kwargs: object) -> SimpleNamespace:
        del image
        self.calls += 1
        budget = int(kwargs["budget_bytes"])
        if self.infeasible:
            return SimpleNamespace(
                feasible=False,
                codestream=None,
                emitted_byte_count=None,
                compression_ratio_argument=4000.0,
                search_trace=(),
                cache_key="synthetic-infeasible",
            )
        return SimpleNamespace(
            feasible=True,
            codestream=self.codestream,
            emitted_byte_count=len(self.codestream),
            compression_ratio_argument=4000.0,
            search_trace=(
                SimpleNamespace(
                    iteration=1,
                    compression_ratio=4000.0,
                    emitted_bytes=len(self.codestream),
                    within_budget=len(self.codestream) <= budget,
                ),
            ),
            cache_key="synthetic-backend-cache",
        )


class _SmokeDecoder:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, codestream: bytes) -> np.ndarray:
        if not codestream:
            raise ValueError("empty synthetic codestream")
        self.calls += 1
        return np.full((4, 4, 3), 7, dtype=np.uint8)


def _synthetic_context(contract: dict[str, Any]) -> dict[str, Any]:
    split = g8_d.ValidationSplitIdentity.from_mapping(
        next(item for item in contract["validation_split_bindings"] if item["dataset"] == "imagenette160")
    )
    classifier = g8_d.ClassifierIdentity.from_mapping(contract["classifier_binding"])
    table = g8_d.G8CTableIdentity.from_mapping(contract["g8_c_binding"])
    image = g8_d.ImageIdentity.from_pixels(
        split_identity=split,
        stable_sample_id="d6-synthetic-image-0",
        source_bytes=b"synthetic-validation-placeholder-not-dataset-bytes",
        canonical_pixels=np.zeros((8, 8, 3), dtype=np.uint8),
    )
    budget = g8_d.BudgetIdentity(
        bw_ratio="d6-synthetic-budget",
        bytes_sent=30,
        payload_bytes=30,
        packet_accounting={"payload_bytes": 30, "channel_bits": 240, "synthetic": True},
    )
    backend = _SmokeCodec()
    codec = g8_d.CodecConfigurationIdentity(
        backend.snapshot,
        backend.configuration_hash,
        "synthetic-openjpeg",
    )
    candidate = g8_d.CandidateIdentity(
        image_identity_id=image.identity_id,
        budget_identity_id=budget.identity_id,
        codec_configuration_id=codec.identity_id,
        g8_c_table_identity_id=table.identity_id,
        bler_identity={
            "k_and_n": [128, 256],
            "base_graph": 2,
            "lifting_size": 22,
            "modulation": "qpsk",
            "decoder_algorithm": "offset_min_sum",
            "decoder_offset": 0.5,
            "iterations": 50,
            "snr_convention": "es_n0_per_symbol",
            "rate": "1/2",
        },
        snr_db=0.0,
        encode_axis_px=8,
    )
    return {
        "split": split,
        "classifier": classifier,
        "table": table,
        "image": image,
        "budget": budget,
        "backend": backend,
        "codec": codec,
        "candidate": candidate,
    }


def run_smoke(output_path: Path, repo_root: Path = REPO) -> dict[str, Any]:
    """Run the bounded synthetic pipeline and publish one canonical witness."""

    repo_root = Path(repo_root).resolve()
    output_path = Path(output_path).resolve()
    contract = g8_d.build_g8_d_contract(repo_root)
    context = _synthetic_context(contract)
    image = context["image"]
    budget = context["budget"]

    with tempfile.TemporaryDirectory(prefix="g8-d-smoke-") as temporary:
        scratch = Path(temporary)
        engine = g8_d.CodecSearchEngine(
            scratch / "codec",
            backend=context["backend"],
            codec_identity=context["codec"],
        )
        search_kwargs = {
            "image_identity": image,
            "encoded_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "budget": budget,
            "encode_axis_px": 8,
        }
        feasible = engine.search(**search_kwargs)
        cached_feasible = engine.search(**search_kwargs)
        structural = engine.search(**search_kwargs, structurally_feasible=False, structural_reason="synthetic packet plan")

        infeasible_backend = _SmokeCodec(infeasible=True)
        infeasible_engine = g8_d.CodecSearchEngine(
            scratch / "codec-infeasible",
            backend=infeasible_backend,
            codec_identity=context["codec"],
        )
        codec_infeasible = infeasible_engine.search(**search_kwargs)

        over_budget_backend = _SmokeCodec(codestream=b"x" * 31)
        over_budget_engine = g8_d.CodecSearchEngine(
            scratch / "codec-over-budget",
            backend=over_budget_backend,
            codec_identity=context["codec"],
        )
        try:
            over_budget_engine.search(**search_kwargs)
        except g8_d.G8DContractError as exc:
            over_budget_rejected = "exceeds payload budget" in str(exc)
        else:
            over_budget_rejected = False

        if feasible.emitted_codestream is None or feasible.emitted_identity is None:
            raise g8_d.G8DContractError("synthetic feasible smoke search produced no emitted bytes")
        delivered = g8_d.account_br11(
            feasible.emitted_codestream,
            emitted_file_identity=feasible.emitted_identity,
            bytes_sent=budget.bytes_sent,
            verdict="delivered",
        )
        decode_failure = g8_d.account_br11(
            feasible.emitted_codestream,
            emitted_file_identity=feasible.emitted_identity,
            bytes_sent=budget.bytes_sent,
            verdict="decode_failure",
        )
        br11 = g8_d.aggregate_br11((delivered, decode_failure))

        decoder = _SmokeDecoder()
        reconstruction_cache = g8_d.ReconstructionCache(
            scratch / "reconstruction",
            context["codec"],
            decoder=decoder,
        )
        reconstruction = reconstruction_cache.get_or_create(
            image_identity=image,
            emitted_file_identity=feasible.emitted_identity,
            codestream=feasible.emitted_codestream,
            output_shape=image.canonical_shape,
        )
        reconstruction_cached = reconstruction_cache.get_or_create(
            image_identity=image,
            emitted_file_identity=feasible.emitted_identity,
            codestream=feasible.emitted_codestream,
            output_shape=image.canonical_shape,
        )

        work_unit = g8_d.WorkUnitIdentity(contract["campaign_id"], 0, context["candidate"].identity_id)
        record = g8_d.CleanClassifierMeasurementRecord.from_outcomes(
            work_unit=work_unit,
            candidate=context["candidate"],
            image=image,
            validation_split=context["split"],
            classifier=context["classifier"],
            g8_c_table=context["table"],
            reconstruction=reconstruction.identity,
            reconstruction_cache_object_id=reconstruction.cache_object_id,
            outcomes=[True, False, True],
            source=SMOKE_SOURCE,
        )
        campaign = g8_d.AtomicMeasurementCampaign(
            scratch / "resume",
            contract_id=contract["contract_id"],
            campaign_id=contract["campaign_id"],
            work_units=[work_unit],
            record_factory=lambda _work_unit: record,
        )
        campaign.initialize()
        first_run = campaign.run_all()
        reused = campaign.run_next()
        final_state = campaign.read_state()

        body: dict[str, Any] = {
            "schema_version": SMOKE_SCHEMA_VERSION,
            "artifact_role": SMOKE_ARTIFACT_ROLE,
            "contract_id": contract["contract_id"],
            "campaign_id": contract["campaign_id"],
            "smoke_label": "NON-SCIENTIFIC BOUNDED SMOKE",
            "non_scientific": True,
            "non_selection": True,
            "non_headline": True,
            "merge_eligible": False,
            "validation_campaign_started": False,
            "pass_one_started": False,
            "training_started": False,
            "test_split_accessed": False,
            "samples": 1,
            "candidates": 4,
            "cells": 4,
            "measurement_work_units": 1,
            "statuses": [
                feasible.status,
                structural.status,
                codec_infeasible.status,
                "decode_failure",
            ],
            "codec_backend_calls": context["backend"].calls + infeasible_backend.calls + over_budget_backend.calls,
            "codec_cache_hit": cached_feasible.cache_hit,
            "reconstruction_cache_hit": reconstruction_cached.cache_hit,
            "reconstruction_decoder_calls": decoder.calls,
            "emitted_bytes_authoritative": feasible.emitted_identity.emitted_bytes == len(feasible.emitted_codestream),
            "requested_ratio_is_provenance_only": feasible.requested_compression_ratio == 4000.0,
            "over_budget_rejected": over_budget_rejected,
            "br11": br11.as_dict(),
            "clean_measurement": {
                "record_id": record.record_id,
                "correct_count": record.correct_count,
                "total_count": record.total_count,
                "accuracy_derivation": "correct_count / total_count",
                "validation_only": True,
                "test_access": 0,
                "training": False,
                "scientific_evidence": False,
                "merge_eligible": False,
                "reconstruction_cache_object_id": record.reconstruction_cache_object_id,
            },
            "resume": {
                "completed_count": len(final_state["completed_work_unit_ids"]),
                "reused_complete_output": reused.reused,
                "first_run_count": len(first_run),
                "state_schema_version": final_state["schema_version"],
                "in_progress_work_unit_id": final_state["in_progress_work_unit_id"],
                "aggregate_record_count": final_state["aggregate_ref"]["record_count"],
            },
            "protected_counters": {
                "inference": 0,
                "training": 0,
                "validation_decoding": 0,
                "test_access": 0,
            },
            "mutation_case_names": list(MUTATION_CASES),
            "source": SMOKE_SOURCE,
        }
    body["artifact_id"] = "g8dsmoke-" + g8_d.sha256_bytes(g8_d.canonical_json(body))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = g8_d.rendered_json(body)
    output_path.write_bytes(payload)
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/baseline/g8_d/bounded_smoke.json",
    )
    args = parser.parse_args()
    artifact = run_smoke(args.output, REPO)
    print(json.dumps({"status": "PASS", "artifact_id": artifact["artifact_id"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
