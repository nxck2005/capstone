#!/usr/bin/env python3
"""Owner-gated runner for the G8_E corrected-v2 epoch.

This command is intentionally closed in the repository's pre-data state: no
owner authorization artifact exists here.  Authentication, source/profile
checks, and disk preflight all happen before the validation registry is opened.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline import g8_e_corrected_v2 as v2  # noqa: E402


def _production_samples() -> tuple[v2.SyntheticSample, ...]:
    """Open validation payloads only after all outer gates have passed."""

    from data.registry import load_dataset

    dataset = load_dataset(v2.INITIAL_DATASET, v2.VALIDATION_SPLIT)
    result: list[v2.SyntheticSample] = []
    for index in range(len(dataset)):
        source = dataset.source_sample(index)
        product, label = dataset[index]
        result.append(v2.SyntheticSample(
            stable_sample_id=source.stable_sample_id,
            label=int(label),
            source_bytes=source.source_bytes,
            canonical_pixels=product.canonical_image.copy(),
        ))
    return tuple(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--start", action="store_true")
    group.add_argument("--resume", action="store_true")
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--contract", type=Path, default=v2.V2_CONTRACT_PATH)
    parser.add_argument("--runtime-root", type=Path, default=v2.V2_RUNTIME_ROOT)
    parser.add_argument("--profile", default=v2.PRODUCTION_PROFILE_ID)
    parser.add_argument("--device", default=v2.PRODUCTION_DEVICE)
    parser.add_argument("--authorization", type=Path, default=v2.V2_ROOT / "e2_execution_authorization.json")
    args = parser.parse_args()
    mode = "start" if args.start else "resume"
    try:
        v2.reject_superseded_campaign(args.campaign_id)
        if args.contract.resolve() != v2.V2_CONTRACT_PATH.resolve():
            raise v2.G8EV2Error("runner must use the current corrected-v2 contract")
        if args.runtime_root.resolve() != v2.V2_RUNTIME_ROOT.resolve():
            raise v2.G8EV2Error("runner must use the current corrected-v2 runtime root")
        bundle = v2.verify_bundle(verify_live_sources=True)
        contract = bundle["contract"]
        if args.campaign_id != contract["campaign_id"]:
            raise v2.G8EV2Error("runner campaign ID differs from the current v2 contract")
        profile = contract["execution_profile"]
        if args.profile != profile["profile_id"] or args.device != profile["device"]:
            raise v2.G8EV2Error("runner profile/device differs from the frozen v2 profile")
        runtime = args.runtime_root.resolve()
        v2.check_runtime_mode(mode, runtime)
        # This refusal is intentionally before _production_samples().
        if not args.authorization.is_file():
            raise v2.G8EV2Error("v2 owner E2 authorization is absent; refusing before validation payload decode")
        v2.authenticate_owner_authorization_v2(args.authorization, contract)
        from config.execution_profiles import authenticate_execution_profile

        profile_result = authenticate_execution_profile(
            args.profile,
            device=args.device,
            config_hash=profile["config_hash"],
            require_openjpeg=True,
        )
        if profile_result["execution_profile_id"] != profile["profile_id"]:
            raise v2.G8EV2Error("live execution profile differs from the frozen v2 profile")
        v2.storage_preflight(bundle["storage_plan"], runtime)

        # No validation payload is opened above this point.
        samples = _production_samples()
        sample_by_id = {sample.stable_sample_id: sample for sample in samples}
        sample_ids = tuple(sample.stable_sample_id for sample in samples)
        authority = v2.load_measurement_authority()
        work_units = v2.expected_work_units(authority, sample_ids)
        from baseline.j2k import J2KCodec
        from models.frozen_reference_classifier import load_frozen_reference_classifier

        codec = J2KCodec(runtime / "backend")
        classifier = load_frozen_reference_classifier(args.device, allow_download=False)

        class ClassifierAdapter:
            def predict(self, pixels):
                import torch
                from data.preprocessing import reconstruction_input

                with torch.inference_mode():
                    return int(classifier(reconstruction_input(pixels)[None].to(args.device)).argmax(dim=1).item())

        executor = v2.MeasurementExecutorV2(
            contract=contract,
            authority=authority,
            runtime_root=runtime,
            backend=codec,
            decoder=codec.decode_codestream,
            classifier=ClassifierAdapter(),
        )

        def provider(sample_id: str) -> v2.SyntheticSample:
            try:
                return sample_by_id[sample_id]
            except KeyError:
                raise v2.G8EV2Error("provider requested a foreign validation sample") from None

        campaign = v2.AtomicE2CampaignV2(
            runtime_root=runtime,
            contract=contract,
            authority=authority,
            work_units=work_units,
            executor=executor.execute,
            sample_provider=provider,
            mode=mode,
        )
        campaign.run_all()
        print({"status": "E2_COMPLETE", "campaign_id": contract["campaign_id"], "work_units": len(work_units)})
    except (OSError, v2.G8EV2Error) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
