#!/usr/bin/env python3
"""Run the complete corrected E2 stack, gated by owner authorization.

The gate is checked after contract/source/profile authentication and before
the validation registry is opened.  In this pre-data epoch no authorization
artifact exists, so both ``--start`` and ``--resume`` refuse without decoding
any validation payload.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline import g8_e_corrected as corrected  # noqa: E402


def _production_samples() -> tuple[dict[str, object], ...]:
    """Open and canonicalize validation samples only after the owner gate."""

    from data.registry import load_dataset

    dataset = load_dataset(corrected.INITIAL_DATASET, corrected.VALIDATION_SPLIT)
    result = []
    for index in range(len(dataset)):
        sample = dataset.source_sample(index)
        product, label = dataset[index]
        result.append({
            "stable_sample_id": sample.stable_sample_id,
            "label": int(label),
            "source_bytes": sample.source_bytes,
            "canonical_pixels": product.canonical_image.copy(),
        })
    return tuple(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--start", action="store_true")
    group.add_argument("--resume", action="store_true")
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--contract", type=Path, default=corrected.CORRECTED_CONTRACT_PATH)
    parser.add_argument("--runtime-root", type=Path, default=corrected.CORRECTED_RUNTIME_ROOT)
    parser.add_argument("--profile", default=corrected.PRODUCTION_PROFILE_ID)
    parser.add_argument("--device", default=corrected.PRODUCTION_DEVICE)
    parser.add_argument("--authorization", type=Path, default=corrected.CORRECTED_ROOT / "e2_execution_authorization.json")
    args = parser.parse_args()
    try:
        corrected.reject_old_campaign(args.campaign_id)
        if args.contract.resolve() != corrected.CORRECTED_CONTRACT_PATH.resolve():
            raise corrected.CorrectedG8EError("runner must use the current corrected E1 contract")
        bundle = corrected.verify_corrected_bundle(verify_live_sources=True)
        contract = bundle["contract"]
        if args.campaign_id != contract["campaign_id"]:
            raise corrected.CorrectedG8EError("runner campaign ID differs from corrected contract")
        if args.profile != contract["execution_profile"]["profile_id"] or args.device != contract["execution_profile"]["device"]:
            raise corrected.CorrectedG8EError("runner profile/device differs from frozen execution profile")
        # This is deliberately before _production_samples(): absence is the
        # normal pre-data result and does not create a runtime directory.
        if not args.authorization.is_file():
            raise corrected.CorrectedG8EError(
                "E2 execution authorization is absent; refusing before validation payload decode"
            )
        authorization = corrected.authenticate_owner_authorization(args.authorization, contract)
        del authorization

        from config.execution_profiles import authenticate_execution_profile

        profile_authentication = authenticate_execution_profile(
            args.profile,
            device=args.device,
            config_hash=contract["execution_profile"]["config_hash"],
            require_openjpeg=True,
        )
        if profile_authentication["execution_profile_id"] != contract["execution_profile"]["profile_id"]:
            raise corrected.CorrectedG8EError("live execution profile differs from corrected contract")

        from baseline.j2k import J2KCodec

        samples = _production_samples()
        ids = tuple(str(sample["stable_sample_id"]) for sample in samples)
        by_id = {str(sample["stable_sample_id"]): sample for sample in samples}
        work_units = corrected.expected_work_units(bundle["authority"], ids)
        codec = J2KCodec(args.runtime_root / "backend")
        engine = corrected.MeasurementExecutor(
            bundle=bundle,
            runtime_root=args.runtime_root,
            backend=codec,
            decoder=codec.decode_codestream,
            classifier=corrected.FrozenG1Classifier(args.device),
        )

        def provider(sample_id: str) -> corrected.SyntheticSample:
            sample = by_id[sample_id]
            return corrected.SyntheticSample(
                stable_sample_id=str(sample["stable_sample_id"]),
                label=int(sample["label"]),
                source_bytes=sample["source_bytes"],
                canonical_pixels=sample["canonical_pixels"],
            )

        campaign = corrected.AtomicE2Campaign(
            runtime_root=args.runtime_root,
            contract=contract,
            authority=bundle["authority"],
            work_units=work_units,
            executor=engine,
            sample_provider=provider,
        )
        campaign.run_all()
        print({"status": "E2_COMPLETE", "campaign_id": contract["campaign_id"], "work_units": len(work_units)})
    except (OSError, corrected.CorrectedG8EError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
