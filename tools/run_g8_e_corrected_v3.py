#!/usr/bin/env python3
"""Owner-gated corrected-v3 E2 runner; real validation remains closed."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline import g8_e_corrected_v3 as v3  # noqa: E402


def _production_samples() -> tuple[v3.SyntheticSample, ...]:
    """Open validation payloads only after every outer identity gate passes."""

    from data.registry import load_dataset

    dataset = load_dataset(v3.INITIAL_DATASET, v3.VALIDATION_SPLIT)
    result: list[v3.SyntheticSample] = []
    for index in range(len(dataset)):
        source = dataset.source_sample(index)
        product, label = dataset[index]
        result.append(v3.SyntheticSample(
            stable_sample_id=source.stable_sample_id,
            label=int(label),
            source_bytes=source.source_bytes,
            canonical_pixels=product.canonical_image.copy(),
        ))
    return tuple(result)


def run_campaign_core(
    *,
    mode: str,
    runtime_root: Path,
    contract: Mapping[str, Any],
    authority: Mapping[str, Any],
    samples: Sequence[v3.SyntheticSample],
    backend: Any,
    decoder: Any,
    classifier: Any,
    non_scientific_fixture: bool = False,
    max_units: int | None = None,
) -> v3.AtomicE2CampaignV3:
    sample_by_id = {sample.stable_sample_id: sample for sample in samples}
    sample_ids = tuple(sorted(sample_by_id))
    if len(sample_by_id) != len(samples):
        raise v3.G8EV3Error("runner sample IDs are duplicated")
    work_units = v3.expected_work_units(authority, sample_ids)
    executor = v3.MeasurementExecutorV3(
        contract=contract,
        authority=authority,
        runtime_root=runtime_root,
        backend=backend,
        decoder=decoder,
        classifier=classifier,
        non_scientific_fixture=non_scientific_fixture,
    )

    def provider(sample_id: str) -> v3.SyntheticSample:
        try:
            return sample_by_id[sample_id]
        except KeyError:
            raise v3.G8EV3Error("provider requested a foreign validation identity") from None

    campaign = v3.AtomicE2CampaignV3(
        runtime_root=runtime_root,
        contract=contract,
        authority=authority,
        work_units=work_units,
        executor=executor.execute,
        sample_provider=provider,
        mode=mode,
    )
    run_constructed_campaign(campaign, max_units=max_units)
    return campaign


def run_constructed_campaign(campaign: v3.AtomicE2CampaignV3, *, max_units: int | None = None) -> None:
    if max_units is None:
        campaign.run_all()
    else:
        for _ in range(max_units):
            if not campaign.run_next():
                break


def main(argv: Sequence[str] | None = None, *, fixture: Mapping[str, Any] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--start", action="store_true")
    group.add_argument("--resume", action="store_true")
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--contract", type=Path, default=v3.V3_CONTRACT_PATH)
    parser.add_argument("--runtime-root", type=Path, default=v3.V3_RUNTIME_ROOT)
    parser.add_argument("--profile", default=v3.PRODUCTION_PROFILE_ID)
    parser.add_argument("--device", default=v3.PRODUCTION_DEVICE)
    parser.add_argument("--authorization", type=Path, default=v3.V3_AUTHORIZATION_PATH)
    args = parser.parse_args(argv)
    mode = "start" if args.start else "resume"
    try:
        v3.reject_superseded_campaign(args.campaign_id)
        if fixture is None:
            if args.contract.resolve() != v3.V3_CONTRACT_PATH.resolve():
                raise v3.G8EV3Error("production runner must use the current v3 contract")
            if args.runtime_root.resolve() != v3.V3_RUNTIME_ROOT.resolve():
                raise v3.G8EV3Error("production runner must use the current v3 runtime root")
            bundle = v3.verify_v3_frozen_contract(verify_live_sources=True, verify_live_data=False)
            contract = bundle["contract"]
            data_identity = bundle["scientific_data_identity"]
            authority = v3.load_measurement_authority()
            if args.campaign_id != contract["campaign_id"]:
                raise v3.G8EV3Error("runner campaign ID differs from the current v3 contract")
            profile = contract["execution_profile"]
            if args.profile != profile["profile_id"] or args.device != profile["device"]:
                raise v3.G8EV3Error("runner profile/device differs from the frozen v3 profile")
            v3.v2.check_runtime_mode(mode, args.runtime_root)
            # Authorization and every frozen identity are authenticated before
            # load_dataset or canonical image decode can be reached.
            v3.authenticate_owner_authorization_v3(args.authorization, contract, data_identity)
            from config.execution_profiles import authenticate_execution_profile

            live_profile = authenticate_execution_profile(
                args.profile,
                device=args.device,
                config_hash=profile["config_hash"],
                require_openjpeg=True,
            )
            if live_profile["execution_profile_id"] != profile["profile_id"]:
                raise v3.G8EV3Error("live execution profile differs")
            v3.storage_preflight(bundle["storage_plan"], args.runtime_root)
            frozen_ids = v3.verify_live_validation_identity(data_identity)
            _, frozen_labels = v3.frozen_validation_metadata(data_identity)
            from baseline.j2k import J2KCodec
            from models.frozen_reference_classifier import load_frozen_reference_classifier

            codec = J2KCodec(args.runtime_root / "backend")
            model = load_frozen_reference_classifier(args.device, allow_download=False)

            class ClassifierAdapter:
                def predict(self, pixels: Any) -> int:
                    import torch
                    from data.preprocessing import reconstruction_input

                    with torch.inference_mode():
                        return int(model(reconstruction_input(pixels)[None].to(args.device)).argmax(dim=1).item())

            # Constructing the transaction authenticates/reconciles the exact
            # current prefix before _production_samples can open one image.
            sample_by_id: dict[str, v3.SyntheticSample] = {}
            work_units = v3.expected_work_units(authority, frozen_ids)
            executor = v3.MeasurementExecutorV3(
                contract=contract,
                authority=authority,
                runtime_root=args.runtime_root,
                backend=codec,
                decoder=codec.decode_codestream,
                classifier=ClassifierAdapter(),
                non_scientific_fixture=False,
            )

            def provider(sample_id: str) -> v3.SyntheticSample:
                try:
                    return sample_by_id[sample_id]
                except KeyError:
                    raise v3.G8EV3Error("provider requested a foreign validation identity") from None

            campaign = v3.AtomicE2CampaignV3(
                runtime_root=args.runtime_root,
                contract=contract,
                authority=authority,
                work_units=work_units,
                executor=executor.execute,
                sample_provider=provider,
                mode=mode,
            )
            samples = _production_samples()
            if tuple(sorted(sample.stable_sample_id for sample in samples)) != tuple(frozen_ids):
                raise v3.G8EV3Error("model-facing validation loader differs from the frozen identity")
            if {sample.stable_sample_id: sample.label for sample in samples} != frozen_labels:
                raise v3.G8EV3Error("model-facing validation labels differ from the frozen manifest")
            sample_by_id.update((sample.stable_sample_id, sample) for sample in samples)
            if len(sample_by_id) != len(samples):
                raise v3.G8EV3Error("model-facing validation loader contains duplicate identities")
            run_constructed_campaign(campaign)
        else:
            labels = set(fixture.get("labels", ()))
            required_labels = {"NON-SCIENTIFIC", "NON-SELECTION", "NOT PRODUCTION E2 EVIDENCE", "MERGE-INELIGIBLE FOR PRODUCTION"}
            if labels != required_labels:
                raise v3.G8EV3Error("synthetic runner fixture labels differ")
            contract = fixture["contract"]
            authority = fixture["authority"]
            data_identity = fixture["data_identity"]
            if args.campaign_id != contract["campaign_id"]:
                raise v3.G8EV3Error("synthetic runner campaign differs")
            v3.authenticate_owner_authorization_v3(args.authorization, contract, data_identity)
            campaign = run_campaign_core(
                mode=mode,
                runtime_root=args.runtime_root,
                contract=contract,
                authority=authority,
                samples=fixture["samples"],
                backend=fixture["backend"],
                decoder=fixture["decoder"],
                classifier=fixture["classifier"],
                non_scientific_fixture=True,
                max_units=fixture.get("max_units"),
            )
        state = campaign.state()
        if state["status"] == v3.v2.COMPLETE_STATUS:
            completion, _, completion_sha = v3.publish_e2_completion(
                runtime_root=args.runtime_root,
                contract=contract,
                authority=authority,
                production=fixture is None,
            )
            status = "E2_COMPLETE"
        else:
            completion = None
            completion_sha = None
            status = "E2_PARTIAL"
        print({
            "status": status,
            "campaign_id": contract["campaign_id"],
            "completed": state["completed_prefix_count"],
            "required": state["total_required"],
            "completion_id": None if completion is None else completion["completion_id"],
            "completion_sha256": completion_sha,
        })
        return 0
    except (OSError, v3.G8EV3Error) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
