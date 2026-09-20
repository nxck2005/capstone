"""W10 per-image classical, JPEG-secondary and artifact-scored evaluators (AM-98).

BR-4's analytic composition is selection only; W10 reports actual per-image
channel simulation at the already-frozen operating points, scored through the
frozen clean and BR-12 artifact-finetuned classifiers.
"""

from __future__ import annotations

import hashlib
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from baseline.classical.pipeline import (
    CODEC_INFEASIBILITY,
    DECODE_FAILURE,
    DELIVERED,
    STRUCTURAL_INFEASIBILITY,
    ChannelIdentity,
    ClassicalResult,
    run_classical_pipeline,
)
from baseline.classical.records import score_result
from baseline.classical.jpeg_pipeline import run_jpeg_pipeline
from baseline.j2k import J2KCodec
from baseline.jpeg import JpegCodec
from baseline.classical.outage import load_outage_policy
from config.params import get
from evaluation.w10_backends import W10Execution, _aggregate, _apply_papr
from evaluation.w10_evidence import run_identity, scheduled_noise_id
from evaluation.w10_scope import W10_DATASET
from models.frozen_reference_classifier import load_frozen_reference_classifier
from training.deterministic_core import canonical_sha256

OUTAGE_POLICY_PATH = "results/baseline/w4/outage_policy.json"
# The W10 runtime root (checkpoints/w10_rehearsal) is worker-local and ignored;
# the J2K cache is content-addressed by canonical pixels and codec configuration,
# so it is deterministic and never scientific evidence.
W10_J2K_CACHE_DIR = "checkpoints/w10_rehearsal/j2k_cache"
OUTAGE_POLICY_SHA256 = "ebcc34133f7a1e38635e8a958cb41a4b8f019b02fd97bd2f0ad606a0a1396121"
ARTIFACT_CHECKPOINT_PATH = "checkpoints/artifact_classifier/epoch-17.pt"
G1_BEST_PATH = "results/reference_classifier/best_checkpoint.json"


def load_artifact_classifier(root: Path, device: torch.device | str) -> torch.nn.Module:
    """Load exactly the frozen BR-12 artifact-finetuned classifier."""

    checkpoint_path = Path(root) / ARTIFACT_CHECKPOINT_PATH
    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        _download_artifact_checkpoint(root, checkpoint_path)
    from training.g8_f_f2_closeout import verify_checkpoint

    verify_checkpoint(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = load_frozen_reference_classifier(device, allow_download=True)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model


def _download_artifact_checkpoint(root: Path, destination: Path) -> None:
    import json

    freeze_path = Path(root) / "results/baseline/g8_f/artifact_classifier_freeze.json"
    if not freeze_path.is_file():
        raise RuntimeError("BR-12 freeze is missing; cannot locate the artifact checkpoint")
    freeze = json.loads(freeze_path.read_bytes())
    external = freeze.get("checkpoint_external_artifact")
    if not isinstance(external, Mapping):
        raise RuntimeError("BR-12 checkpoint is not local and carries no hosted artifact record")
    url = (
        f"https://github.com/{external['repository']}/releases/download/"
        f"{external['release_tag']}/{external['asset_name']}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):  # literal-ok: one-MiB stream chunk
            output.write(chunk)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if temporary.stat().st_size != int(external["bytes"]) or digest != external["sha256"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("downloaded BR-12 checkpoint bytes differ from the freeze")
    temporary.replace(destination)


def _outage_policy(root: Path) -> Any:
    path = Path(root) / OUTAGE_POLICY_PATH
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != OUTAGE_POLICY_SHA256:
        raise RuntimeError("authenticated BR-13 outage policy bytes differ")
    return load_outage_policy(
        path,
        expected_dataset=W10_DATASET,
        expected_manifest_sha256=str(get(f"datasets.{W10_DATASET}.manifest_sha256")),
    )


def _candidate_table(root: Path) -> dict[str, dict[str, Any]]:
    import json

    value = json.loads((Path(root) / "results/baseline/g8_e/candidate_authority.json").read_bytes())
    return {str(item["candidate_id"]): item for item in value["candidates"]}


def _selection_map(binding: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    if binding.get("kind") == "br16_fixed_mcs":
        fixed = binding["fixed_configuration"]
        return {
            int(snr): {
                "snr_db": int(snr),
                "encode_axis_px": int(fixed["encode_axis_px"]),
                "ldpc_rate": str(fixed["ldpc_rate"]),
                "modulation": str(fixed["modulation"]),
            }
            for snr in get("channel.test_snr_grid_db")
        }
    if binding.get("kind") == "classical_pass_two":
        table = {str(item["snr_db"]): item for item in binding["selections"]}
        return {int(float(snr)): {"authority_candidate_id": item["authority_candidate_id"]} for snr, item in table.items()}
    if binding.get("kind") == "w10_jpeg_validation_selection":
        # Frozen W10 JPEG execution uses exactly the selected quality and PHY.
        return {int(float(item["snr_db"])): dict(item) for item in binding["selections"]}
    raise RuntimeError(f"unsupported classical selection binding: {binding.get('kind')}")


def _classical_scoring(
    result: ClassicalResult,
    *,
    canonical_image: np.ndarray,
    label: int,
    policy: Any,
    classifiers: Mapping[str, torch.nn.Module],
    device: torch.device | str,
) -> dict[str, Any]:
    outcomes = {
        variant: score_result(
            result,
            true_label=label,
            policy=policy,
            canonical_image=canonical_image if result.verdict == DELIVERED else None,
            classifier=classifiers.get(variant) if result.verdict == DELIVERED else None,
            device=device,
        )
        for variant in classifiers
    }
    return outcomes


def classical_unit(
    context: W10Execution,
    unit: Mapping[str, Any],
    *,
    root: Path,
    checkpoint_id: str,
    binding: Mapping[str, Any],
    codec_kind: str = "jpeg2000",
    quality: int | None = None,
) -> dict[str, Any]:
    """One per-image classical (or JPEG) validation unit at a frozen point."""

    view = context.validation()
    policy = _outage_policy(root)
    classifiers = {
        "artifact_finetuned": context.artifact_classifier(lambda: load_artifact_classifier(root, context.device)),
    }
    if "clean" in unit_classifier_variants(unit):
        classifiers["clean"] = context.classifier("clean")
    candidates = _candidate_table(root)
    selection = _selection_map(binding)
    point = selection[int(unit["snr_db"])]
    if codec_kind == "jpeg2000":
        candidate = candidates[str(point["authority_candidate_id"])]
        modulation = str(candidate["modulation"])
        ldpc_rate = str(candidate["ldpc_rate"])
        encode_axis = int(candidate["encode_axis_px"])
        config_hash = canonical_sha256(candidate)
    else:
        modulation = str(point["modulation"])
        ldpc_rate = str(point["ldpc_rate"])
        encode_axis = int(point["encode_axis_px"])
        config_hash = canonical_sha256(point)
    k = int(get(f"bandwidth.k_symbols.{W10_DATASET}.{unit['bw_ratio']}"))
    codec = (
        J2KCodec(Path(root) / W10_J2K_CACHE_DIR)
        if codec_kind == "jpeg2000"
        else JpegCodec()
    )
    streams: dict[str, list[dict[str, Any]]] = {variant: [] for variant in classifiers}
    papr_values: list[float] = []
    for stable_id in view.stable_ids:
        product = view.product(stable_id)
        label = view.label(stable_id)
        channel_identity = ChannelIdentity(
            dataset_version=str(get(f"datasets.{W10_DATASET}.{get('config.dataset_version_rule')}")),
            split_manifest_hash=str(get(f"datasets.{W10_DATASET}.manifest_sha256")),
            channel_seed=int(unit["channel_seed"]),
        )
        if codec_kind == "jpeg2000":
            result = run_classical_pipeline(
                product,
                dataset=W10_DATASET,
                k_symbols=k,
                modulation=modulation,
                ldpc_rate=ldpc_rate,
                snr_db=float(unit["snr_db"]),
                codec=codec,
                channel_identity=channel_identity,
                encode_axis_px=encode_axis,
                device=str(context.device),
            )
        else:
            if quality is None:
                raise RuntimeError("JPEG unit requires its frozen quality")
            result = run_jpeg_pipeline(
                product,
                dataset=W10_DATASET,
                k_symbols=k,
                modulation=modulation,
                ldpc_rate=ldpc_rate,
                snr_db=float(unit["snr_db"]),
                quality=int(quality),
                codec=codec,
                channel_identity=channel_identity,
                encode_axis_px=encode_axis,
                device=str(context.device),
            )
        if result.transport is not None:
            papr_values.append(float(result.transport.papr_db))
        from data.preprocessing import codec_input

        canonical_image = codec_input(product)
        outcomes = _classical_scoring(result, canonical_image=canonical_image, label=label, policy=policy, classifiers=classifiers, device=context.device)
        for variant, outcome in outcomes.items():
            identity = run_identity(
                system=unit["system"],
                bw_ratio=unit["bw_ratio"],
                snr_db=unit["snr_db"],
                config_hash=config_hash,
                checkpoint_id=checkpoint_id,
                classifier_variant=variant,
                ldpc_rate=ldpc_rate,
                modulation=modulation,
                quantiser_bits=None,
                transmit_dim=None,
                reconstruction_weight=None,
            )
            from baseline.classical.records import per_image_row

            streams[variant].append(
                per_image_row(
                    result,
                    outcome,
                    identity=identity,
                    true_label=label,
                    run_id=identity.run_id(),
                    scheduled_noise_id=scheduled_noise_id(
                        stable_sample_id=stable_id,
                        bw_ratio=unit["bw_ratio"],
                        test_snr_db=unit["snr_db"],
                        k=k,
                    ),
                )
            )
    primary_variant = unit_primary_variant(unit)
    primary = streams[primary_variant]
    expected_digest = point.get("per_image_correct_digest") if codec_kind != "jpeg2000" else None
    if expected_digest is not None:
        from evaluation.w10_selections import score_vector_digest

        require_digest = score_vector_digest([bool(row["correct"]) for row in primary])
        if require_digest != str(expected_digest):
            raise RuntimeError(
                "W10 JPEG arm does not reproduce its frozen selection candidate digest"
            )
    aggregate = _aggregate(primary, system=unit["system"])
    _apply_papr(aggregate, papr_values, denominator=len(view.stable_ids))
    aggregate["binding"] = {
        "kind": "jpeg_secondary_validation_evaluation" if codec_kind != "jpeg2000" else "classical_per_image_validation_evaluation",
        "codec": codec_kind,
        "selection_kind": binding.get("kind"),
        "selection_id": binding.get("selection_id"),
        "checkpoint_id": checkpoint_id,
        "modulation": modulation,
        "ldpc_rate": ldpc_rate,
        "encode_axis_px": encode_axis,
        "analytic_composition_used": False,
    }
    aggregate["secondary_streams"] = [
        {
            "classifier_variant": variant,
            "per_image": rows,
            "n_correct": sum(int(bool(row["correct"])) for row in rows),
            "n_total": len(rows),
        }
        for variant, rows in streams.items()
        if variant != primary_variant
    ]
    aggregate["primary_classifier_variant"] = primary_variant
    return aggregate


def unit_classifier_variants(unit: Mapping[str, Any]) -> tuple[str, ...]:
    from evaluation.w10_scope import entry_for

    return entry_for(str(unit["system"]), str(unit["bw_ratio"])).classifier_variants


def unit_primary_variant(unit: Mapping[str, Any]) -> str:
    return unit_classifier_variants(unit)[0]


__all__ = [
    "ARTIFACT_CHECKPOINT_PATH",
    "G1_BEST_PATH",
    "OUTAGE_POLICY_PATH",
    "OUTAGE_POLICY_SHA256",
    "classical_unit",
    "load_artifact_classifier",
    "unit_classifier_variants",
    "unit_primary_variant",
]
