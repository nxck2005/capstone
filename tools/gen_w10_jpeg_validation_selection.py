#!/usr/bin/env python3
"""Freeze the W10 JPEG-secondary validation selection (owner-invoked only).

The selector searches the prospectively frozen candidate space and executes
each candidate at its *exact* JPEG quality; frozen W10 execution later uses
only the recorded winner.  ``--preflight`` writes nothing and touches no GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline.classical.channel_transport import build_accounting
from baseline.classical.records import classify_reconstruction
from baseline.jpeg import JpegCodec, JpegCodecError, decode_codestream
from baseline.ldpc.transport import build_packet_plan
from config.params import get
from data.preprocessing import codec_downsample, codec_input, codec_upsample
from evaluation.downstream_v4 import immutable_write
from evaluation.w10_backends import ValidationView
from evaluation.w10_classical import _outage_policy, load_artifact_classifier
from evaluation.w10_scope import HEADLINE_RATIO, W10_DATASET
from evaluation.w10_selections import (
    JPEG_SELECTION_PATH,
    bind_candidate_evidence,
    build_selection_artifact,
    jpeg_analytic_evidence,
    jpeg_candidate_space,
    jpeg_selection_contract,
    rank_candidates,
    scorer_binding,
    score_vector_digest,
)
from runtime.source_epochs import load_w10_manifest, source_record
from runtime.w9_authority import authenticate_live_w9_pascal
from evaluation.downstream_v4 import TITAN_XP_NAME, TITAN_XP_UUID

TARGET = REPO / JPEG_SELECTION_PATH
K = int(get(f"bandwidth.k_symbols.{W10_DATASET}.{HEADLINE_RATIO}"))


def _authority(source: dict) -> dict:
    profile = get("environment.execution_profiles.confessor_pascal_cu126")
    return {
        "execution_profile_id": "confessor_pascal_cu126",
        "host": "confessor",
        "device": "cuda:0",
        "gpu_uuid": TITAN_XP_UUID,
        "gpu_name": TITAN_XP_NAME,
        "compute_capability": str(profile["compute_capability"]),
        "cuda_visible_devices": TITAN_XP_UUID,
        "source_binding": source,
    }


def _score_candidate(
    *,
    view: ValidationView,
    classifier,
    policy,
    codec: JpegCodec,
    snr_db: int,
    phy: dict,
    quality: int,
    device: str,
    source_epoch: dict,
) -> tuple[dict, list[bool]]:
    correct: list[bool] = []
    outage_correct = 0
    delivered = decode_failures = infeasible = 0
    emitted_bytes: list[int | None] = []
    packet = build_packet_plan(K, str(phy["modulation"]), str(phy["ldpc_rate"]))
    if not packet.feasible:
        raise RuntimeError("JPEG selector received a structurally infeasible candidate")
    accounting = build_accounting(packet)
    scorer = scorer_binding(REPO)
    for stable_id in view.stable_ids:
        product = view.product(stable_id)
        label = view.label(stable_id)
        canonical_image = codec_input(product)
        try:
            encoded = codec.encode_exact_quality(
                codec_downsample(canonical_image, int(phy["encode_axis_px"])),
                canonical_pixels_sha256=hashlib.sha256(canonical_image.tobytes()).hexdigest(),
                budget_bytes=accounting.payload_bytes,
                encode_axis_px=int(phy["encode_axis_px"]),
                quality=int(quality),
            )
        except JpegCodecError:
            encoded = None
        if encoded is None or not encoded.feasible or encoded.codestream is None:
            infeasible += 1
            emitted_bytes.append(None)
            outage_correct += int(policy.is_correct(label))
            correct.append(False)
            continue
        emitted_bytes.append(int(encoded.emitted_byte_count))
        delivered += 1
        try:
            restored = codec_upsample(
                decode_codestream(encoded.codestream),
                tuple(int(value) for value in canonical_image.shape[:2]),
            )
            prediction = classify_reconstruction(classifier, restored, device=device)
            correct.append(prediction == int(label))
        except Exception as exc:
            if isinstance(exc, (ValueError, JpegCodecError, OSError)):
                decode_failures += 1
                delivered -= 1
                correct.append(False)
            else:
                raise
    entry = {
        "snr_db": int(snr_db),
        "modulation": str(phy["modulation"]),
        "ldpc_rate": str(phy["ldpc_rate"]),
        "encode_axis_px": int(phy["encode_axis_px"]),
        "quality": int(quality),
        "n_correct": sum(correct),
        "n_delivered": delivered,
        "n_decode_failure": decode_failures,
        "n_infeasible": infeasible,
        "n_total": len(correct),
        "per_image_correct_digest": score_vector_digest(correct),
    }
    clean = {
        "correct": int(sum(correct)),
        "total": len(correct),
        "split": "val",
        "source": "worker_clean_jpeg_validation_codec_measurement",
        "correct_digest": score_vector_digest(correct),
        "outage_correct_count": int(outage_correct),
        "feasible_count": int(delivered),
        "decode_failure_count": int(decode_failures),
        "infeasible_count": int(infeasible),
        "emitted_bytes_digest": canonical_sha256({"emitted_bytes": emitted_bytes}),
        "scorer_binding": scorer,
    }
    analytic = jpeg_analytic_evidence(REPO, entry, clean=clean)
    entry["analytic_evidence"] = analytic
    entry["status"] = str(analytic["eligibility"])
    entry["expected_accuracy"] = (
        None if analytic["composition"] is None else float(analytic["composition"]["expected_accuracy"])
    )
    entry["success_probability"] = (
        None if analytic["composition"] is None else float(analytic["composition"]["success_probability"])
    )
    entry["candidate_evidence"] = bind_candidate_evidence({
        "schema_version": 1,
        "method": "jpeg_clean_codec_measurement",
        "snr_db": int(snr_db),
        "modulation": str(phy["modulation"]),
        "ldpc_rate": str(phy["ldpc_rate"]),
        "encode_axis_px": int(phy["encode_axis_px"]),
        "quality": int(quality),
        "clean": clean,
        "scorer_binding": scorer,
        "source_epoch": dict(source_epoch),
        "selection_contract_sha256": jpeg_selection_contract().sha256(),
    })
    return entry, correct


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    source = load_w10_manifest(REPO, live=True)
    contract = jpeg_selection_contract().identity()
    candidate_space = jpeg_candidate_space(REPO)
    if args.preflight:
        print(
            f"JPEG selection preflight: contract={jpeg_selection_contract().sha256()[:16]} "
            f"snrs={len(contract['snr_grid_db'])} candidates_per_snr={len(candidate_space[str(int(contract['snr_grid_db'][0]))])} "
            f"source={source_record(REPO, source)['manifest_id']}"
        )
        return 0
    if TARGET.is_file() or TARGET.is_symlink():
        raise SystemExit("JPEG validation selection already exists and is immutable")
    if contract["source_epoch"] != source_record(REPO, source):
        raise SystemExit("JPEG selection contract source epoch differs from the live source")
    authenticate_live_w9_pascal(
        REPO,
        _authority(source),
        config_hash=jpeg_selection_contract().sha256(),
    )
    view = ValidationView()
    classifier = load_artifact_classifier(REPO, args.device)
    policy = _outage_policy(REPO)
    codec = JpegCodec()
    selections = []
    candidate_scores = []
    for snr in contract["snr_grid_db"]:
        entries = []
        print(f"JPEG selection SNR {snr} dB: analytic BR-4 candidate grid")
        for candidate in candidate_space[str(int(snr))]:
            entry, _correct = _score_candidate(
                view=view,
                classifier=classifier,
                policy=policy,
                codec=codec,
                snr_db=int(snr),
                phy=candidate,
                quality=int(candidate["quality"]),
                device=args.device,
                source_epoch=source_record(REPO, source),
            )
            entries.append(entry)
        winner = rank_candidates("jpeg_secondary", entries)
        selections.append(winner)
        candidate_scores.append({"snr_db": int(snr), "candidates": entries})
        print(
            f"  winner quality={winner['quality']} objective={winner['expected_accuracy']:.6f} "
            f"tie={winner['tie_break_applied']}"
        )
    artifact = build_selection_artifact(
        "jpeg_secondary",
        selections=selections,
        candidate_scores=candidate_scores,
    )
    immutable_write(TARGET, artifact)
    print(f"JPEG validation selection: {artifact['selection_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
