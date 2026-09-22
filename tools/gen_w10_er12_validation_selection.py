#!/usr/bin/env python3
"""Freeze the W10 ER-12 label-transmission PHY selection (owner-invoked only).

The transmitter runs the frozen ER-9 trunk/task head over its own quantised
representation, transmits only the one-byte predicted-label frame over the
configured matched-k digital chain, and scores decoded labels against the true
labels.  ``--preflight`` writes nothing and touches no GPU.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.params import get
from evaluation.downstream_v4 import TITAN_XP_NAME, TITAN_XP_UUID, immutable_write
from evaluation.er9_search import configured_phy_candidates
from evaluation.er9_transport import ER9TransportBatch
from evaluation.w10_backends import (
    DECODE_FAILURE,
    W10Execution,
    ValidationView,
    decode_label_payload,
    label_payload,
)
from evaluation.w10_bindings import resolve_binding
from evaluation.w10_dispatch import _load_er9_assets
from evaluation.w10_evidence import scheduled_noise_id
from evaluation.w10_scope import HEADLINE_RATIO, W10_DATASET, entry_for
from evaluation.w10_selections import (
    ER12_SELECTION_PATH,
    bind_candidate_evidence,
    build_selection_artifact,
    er12_selection_contract,
    rank_candidates,
    score_vector_digest,
)
from runtime.source_epochs import load_w10_manifest, source_record
from runtime.w9_authority import authenticate_live_w9_pascal
from training.deterministic_core import canonical_sha256

TARGET = REPO / ER12_SELECTION_PATH
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


def _er12_checkpoint_id(er9_binding: Mapping[str, object], assets: Mapping[str, object]) -> str:
    """Require the loaded ER-9 checkpoint to match its frozen W10 binding."""

    try:
        bound_checkpoint_id = er9_binding["checkpoint"]["selected_checkpoint_sha256"]
    except (KeyError, TypeError):
        raise RuntimeError(
            "ER-12 frozen ER-9 binding is missing selected_checkpoint_sha256"
        ) from None
    try:
        loaded_checkpoint_id = assets["checkpoint_id"]
    except (KeyError, TypeError):
        raise RuntimeError("ER-12 loaded ER-9 assets are missing checkpoint_id") from None

    if not isinstance(bound_checkpoint_id, str) or not bound_checkpoint_id:
        raise RuntimeError("ER-12 frozen ER-9 checkpoint identity is invalid")
    if not isinstance(loaded_checkpoint_id, str) or not loaded_checkpoint_id:
        raise RuntimeError("ER-12 loaded ER-9 checkpoint identity is invalid")
    if loaded_checkpoint_id != bound_checkpoint_id:
        raise RuntimeError(
            "ER-12 loaded ER-9 checkpoint identity differs from the frozen binding"
        )
    return loaded_checkpoint_id


def _predicted_labels(model, view: ValidationView, device: str) -> dict[str, int]:
    predicted: dict[str, int] = {}
    with torch.inference_mode():
        for start in range(0, len(view.stable_ids), 32):  # literal-ok: frozen W8 validation batch size
            chunk = view.stable_ids[start : start + 32]
            inputs = torch.stack([view.canonical_tensor(stable_id) for stable_id in chunk]).to(device)
            logits = model(inputs).logits
            for stable_id, value in zip(chunk, logits.argmax(dim=1).cpu(), strict=True):
                predicted[stable_id] = int(value)
    return predicted


def _score_candidate(
    *,
    view: ValidationView,
    predicted: dict[str, int],
    policy,
    modulation: str,
    ldpc_rate: str,
    packet,
    snr_db: int,
    device: str,
    source_epoch: dict,
    checkpoint_id: str,
    selection_contract_sha256: str,
) -> tuple[dict, list[bool]]:
    layout = packet.segmentation
    if layout is None:
        raise RuntimeError("ER-12 candidate has no packet segmentation")
    payload_bytes = int(layout.payload_bits) // 8  # literal-ok: bits-per-octet conversion
    session = ER9TransportBatch(packet, device=device)
    correct: list[bool] = []
    delivered = 0
    all_noise_ids: list[str] = []
    for start in range(0, len(view.stable_ids), 32):  # literal-ok: frozen W8 validation batch size
        chunk = view.stable_ids[start : start + 32]
        payloads = []
        for stable_id in chunk:
            frame = np.zeros(payload_bytes, dtype=np.uint8)
            frame[:1] = label_payload(predicted[stable_id])[0]
            payloads.append(np.unpackbits(frame))
        chunk_noise_ids = [
            scheduled_noise_id(
                stable_sample_id=stable_id,
                bw_ratio=HEADLINE_RATIO,
                test_snr_db=snr_db,
                k=K,
            )
            for stable_id in chunk
        ]
        all_noise_ids.extend(chunk_noise_ids)
        result = session.round_trip(payloads=payloads, snr_db=float(snr_db), noise_ids=chunk_noise_ids)
        for stable_id, payload in zip(chunk, result.payloads, strict=True):
            true_label = view.label(stable_id)
            decoded = None if payload is None else decode_label_payload(payload, payload_bits=int(layout.payload_bits))
            if decoded is None:
                correct.append(policy.is_correct(true_label))
            else:
                delivered += 1
                correct.append(decoded == true_label)
    outcome_counts = {
        "delivered": delivered,
        "decode_failure": len(correct) - delivered,
        "infeasible": 0,
    }
    worker_evidence = {
        "artifact_role": "W10_ER12_CANDIDATE_WORKER_EVIDENCE",
        "source_epoch": dict(source_epoch),
        "execution_profile_id": "confessor_pascal_cu126",
        "checkpoint_id": str(checkpoint_id),
        "selection_contract_sha256": str(selection_contract_sha256),
        "snr_db": int(snr_db),
        "modulation": str(modulation),
        "ldpc_rate": str(ldpc_rate),
        "correct": int(sum(correct)),
        "total": len(correct),
        "correct_digest": score_vector_digest(correct),
        "outcome_counts": outcome_counts,
        "noise_ids_digest": canonical_sha256({"noise_ids": all_noise_ids}),
        "test_access": 0,
    }
    worker_evidence["worker_evidence_id"] = "w10er12workerevidence-" + canonical_sha256(worker_evidence)
    entry = {
        "snr_db": int(snr_db),
        "modulation": str(modulation),
        "ldpc_rate": str(ldpc_rate),
        "n_correct": sum(correct),
        "n_delivered": delivered,
        "n_decode_failure": len(correct) - delivered,
        "n_infeasible": 0,
        "n_total": len(correct),
        "per_image_correct_digest": score_vector_digest(correct),
        "status": "eligible",
    }
    entry["candidate_evidence"] = bind_candidate_evidence({
        "schema_version": 1,
        "method": "er12_per_image_channel_simulation",
        "snr_db": int(snr_db),
        "modulation": str(modulation),
        "ldpc_rate": str(ldpc_rate),
        "protocol_version": int(get("evaluation.w10_er12_protocol_version")),
        "label_bits": int(get("evaluation.w10_er12_label_bits")),
        "payload_frame": str(get("evaluation.w10_er12_payload_frame")),
        "true_label_in_payload": False,
        "outcome_counts": outcome_counts,
        "correct": int(sum(correct)),
        "total": len(correct),
        "correct_digest": score_vector_digest(correct),
        "noise_convention": "w10_scheduled_noise_id_per_stable_image_snr_k_shared_across_candidates",
        "noise_ids_digest": canonical_sha256({"noise_ids": all_noise_ids}),
        "worker_evidence": worker_evidence,
    })
    return entry, correct


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    source = load_w10_manifest(REPO, live=True)
    contract = er12_selection_contract().identity()
    if args.preflight:
        pairs = configured_phy_candidates(K)
        print(
            f"ER-12 selection preflight: contract={er12_selection_contract().sha256()[:16]} "
            f"snrs={len(contract['snr_grid_db'])} candidates={len(pairs)} source={contract['source_epoch']['manifest_id']}"
        )
        return 0
    if TARGET.is_file() or TARGET.is_symlink():
        raise SystemExit("ER-12 validation selection already exists and is immutable")
    if contract["source_epoch"] != source_record(REPO, source):
        raise SystemExit("ER-12 selection contract source epoch differs from the live source")
    authenticate_live_w9_pascal(
        REPO,
        _authority(source),
        config_hash=er12_selection_contract().sha256(),
    )
    view = ValidationView()
    context = W10Execution(root=REPO, device=args.device, view=view)
    er9_binding = resolve_binding(REPO, entry_for("er9_digital", HEADLINE_RATIO))
    assets = _load_er9_assets(context, er9_binding["checkpoint"], root=REPO)
    checkpoint_id = _er12_checkpoint_id(er9_binding, assets)
    from evaluation.er9_campaign import authenticated_er9_outage_policy

    policy = authenticated_er9_outage_policy(assets["config"])
    predicted = _predicted_labels(assets["model"], view, args.device)
    candidates = {(str(item["modulation"]), str(item["ldpc_rate"])): item for item in assets["phy_candidates"]}
    source_epoch = source_record(REPO, source)
    selection_contract_sha256 = er12_selection_contract().sha256()
    selections = []
    candidate_scores = []
    for snr in contract["snr_grid_db"]:
        entries = []
        for (modulation, rate), item in candidates.items():
            entry, _correct = _score_candidate(
                view=view,
                predicted=predicted,
                policy=policy,
                modulation=modulation,
                ldpc_rate=rate,
                packet=item["packet"],
                snr_db=int(snr),
                device=args.device,
                source_epoch=source_epoch,
                checkpoint_id=checkpoint_id,
                selection_contract_sha256=selection_contract_sha256,
            )
            entries.append(entry)
        winner = rank_candidates("er12_label_bound", entries)
        selections.append(winner)
        candidate_scores.append({"snr_db": int(snr), "candidates": entries})
        print(f"ER-12 SNR {snr}: winner {winner['modulation']}/{winner['ldpc_rate']} n_correct={winner['n_correct']}")
    artifact = build_selection_artifact(
        "er12_label_bound",
        selections=selections,
        candidate_scores=candidate_scores,
    )
    immutable_write(TARGET, artifact)
    print(f"ER-12 validation selection: {artifact['selection_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
