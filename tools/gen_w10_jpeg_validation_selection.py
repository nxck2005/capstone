#!/usr/bin/env python3
"""Freeze the W10 JPEG-secondary validation selection (owner-invoked only).

The selector searches the prospectively frozen candidate space and executes
each candidate at its *exact* JPEG quality; frozen W10 execution later uses
only the recorded winner.  ``--preflight`` writes nothing and touches no GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline.classical.jpeg_pipeline import run_jpeg_pipeline
from baseline.classical.pipeline import DECODE_FAILURE, DELIVERED, CODEC_INFEASIBILITY, STRUCTURAL_INFEASIBILITY, ChannelIdentity
from baseline.classical.records import score_result
from baseline.jpeg import JpegCodec
from config.params import get
from data.preprocessing import codec_input
from evaluation.downstream_v4 import immutable_write
from evaluation.w10_backends import ValidationView
from evaluation.w10_classical import _outage_policy, load_artifact_classifier
from evaluation.w10_evidence import scheduled_noise_id
from evaluation.w10_scope import HEADLINE_RATIO, W10_DATASET
from evaluation.w10_selections import (
    JPEG_SELECTION_PATH,
    build_selection_artifact,
    jpeg_phy_shortlist,
    jpeg_selection_contract,
    rank_candidates,
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
) -> tuple[dict, list[bool]]:
    channel_identity = ChannelIdentity(
        dataset_version=str(get(f"datasets.{W10_DATASET}.{get('config.dataset_version_rule')}")),
        split_manifest_hash=str(get(f"datasets.{W10_DATASET}.manifest_sha256")),
        channel_seed=0,
    )
    correct: list[bool] = []
    delivered = decode_failures = infeasible = 0
    for stable_id in view.stable_ids:
        product = view.product(stable_id)
        label = view.label(stable_id)
        result = run_jpeg_pipeline(
            product,
            dataset=W10_DATASET,
            k_symbols=K,
            modulation=phy["modulation"],
            ldpc_rate=phy["ldpc_rate"],
            snr_db=float(snr_db),
            quality=int(quality),
            codec=codec,
            channel_identity=channel_identity,
            encode_axis_px=int(phy["encode_axis_px"]),
            device=device,
        )
        canonical_image = codec_input(product)
        outcome = score_result(
            result,
            true_label=label,
            policy=policy,
            canonical_image=canonical_image if result.verdict == DELIVERED else None,
            classifier=classifier if result.verdict == DELIVERED else None,
            device=device,
        )
        correct.append(bool(outcome.correct))
        if result.verdict == DELIVERED:
            delivered += 1
        elif result.verdict == DECODE_FAILURE:
            decode_failures += 1
        elif result.verdict in (STRUCTURAL_INFEASIBILITY, CODEC_INFEASIBILITY):
            infeasible += 1
        else:  # pragma: no cover - the verdict set is closed
            raise RuntimeError(f"unknown classical verdict {result.verdict!r}")
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
    return entry, correct


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    source = load_w10_manifest(REPO, live=True)
    contract = jpeg_selection_contract().identity()
    shortlist = jpeg_phy_shortlist(REPO)
    qualities = tuple(int(value) for value in get("baseline.jpeg_quality_grid"))
    if args.preflight:
        print(
            f"JPEG selection preflight: contract={jpeg_selection_contract().sha256()[:16]} "
            f"snrs={len(contract['snr_grid_db'])} qualities={len(qualities)} "
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
        phy = shortlist[int(snr)]
        entries = []
        print(f"JPEG selection SNR {snr} dB: phy={phy['modulation']}/{phy['ldpc_rate']}/axis{phy['encode_axis_px']}")
        for quality in qualities:
            entry, _correct = _score_candidate(
                view=view,
                classifier=classifier,
                policy=policy,
                codec=codec,
                snr_db=int(snr),
                phy=phy,
                quality=int(quality),
                device=args.device,
            )
            entries.append(entry)
        winner = rank_candidates("jpeg_secondary", entries)
        selections.append(winner)
        candidate_scores.append({"snr_db": int(snr), "candidates": entries})
        print(
            f"  winner quality={winner['quality']} n_correct={winner['n_correct']} "
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
