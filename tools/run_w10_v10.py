#!/usr/bin/env python3
"""Stage, execute, close, or verify the separately authorized W10 v10 suffix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evaluation.downstream_v4 import immutable_write, read_json, require  # noqa: E402
from evaluation.w10_backends import ValidationView  # noqa: E402
from evaluation.w10_dispatch import dispatch  # noqa: E402
from evaluation.w10_evidence import unit_relative_path  # noqa: E402
from evaluation.w10_rehearsal import execute, validate_unit, work_units  # noqa: E402
from evaluation.w10_successor_v10 import (  # noqa: E402
    V10_AUTHORITY_PATH,
    V11_AUTHORITY_PATH,
    V10_CLOSEOUT_FILES,
    V10_LAUNCH_PATH,
    V11_LAUNCH_PATH,
    V10_PLAN_PATH,
    V11_PLAN_PATH,
    V10_START,
    V10_STOP,
    V9_CUSTODY_PATH,
    build_v10_authority,
    build_v10_closeout,
    build_v10_launch,
    build_v10_plan,
    build_v9_custody,
    published_v10_units,
    verify_v10_authority,
    verify_v10_evidence_set,
    verify_v10_launch,
    verify_v10_plan,
    verify_v10_published,
)
from runtime.source_epochs import load_w10_manifest  # noqa: E402
from runtime.w9_authority import authenticate_live_w9_pascal  # noqa: E402
from training.deterministic_core import canonical_bytes, canonical_sha256  # noqa: E402

PUBLISHED = REPO / "results/learned/w10"
PUBLISHED_FILES = {
    "units": PUBLISHED / "w10_continuation_units_v10.json",
    "unit_manifest": PUBLISHED / "w10_continuation_unit_manifest_v10.json",
    "images": PUBLISHED / "w10_continuation_per_image_manifest_v10.json",
    "closeout": PUBLISHED / "w10_continuation_closeout_v10.json",
}


def _complete_results(runtime: Path) -> list[dict]:
    results = []
    for unit in work_units():
        path = runtime / unit_relative_path(unit, "json")
        require(path.is_file() and not path.is_symlink(), "W10 v10 cannot close incomplete scope")
        value = read_json(path, "W10 v10 unit")
        validate_unit(value, unit)
        results.append(value)
    return results


def main(argv: list[str] | None = None, *, epoch: str = "v10") -> int:
    require(epoch in {"v10", "v11"}, "unknown W10 suffix epoch")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("custody", "authority", "plan", "launch", "execute", "closeout", "verify"))
    parser.add_argument("--preflight", action="store_true", help="authenticate custody/authority/launch without writing")
    parser.add_argument("--published", action="store_true", help="verify committed closeout without worker files")
    args = parser.parse_args(argv)
    require(not args.preflight or args.action in {"custody", "authority", "launch"}, "--preflight is unavailable for this action")
    require(not args.published or args.action == "verify", "--published requires verify")
    if args.action != "verify":
        require(not (REPO / "results/freeze_manifest.json").exists(), "G12 is already open")

    if args.published:
        result = verify_v10_published(REPO, epoch=epoch)
        print(f"W10 v10 published verifier PASS: {result['closeout_id']}")
        return 0

    if args.action == "custody":
        require(epoch == "v10", "v9 custody was already frozen before v11")
        custody = build_v9_custody(REPO, stable_ids=ValidationView().stable_ids)
        if not args.preflight:
            immutable_write(REPO / V9_CUSTODY_PATH, custody)
        print(f"W10 failed v9 custody: {custody['custody_id']}; units=105; streams=168; written={not args.preflight}")
        return 0

    if args.action == "authority":
        source = load_w10_manifest(REPO, live=True, epoch=epoch)
        body = build_v10_authority(REPO, source)
        live = authenticate_live_w9_pascal(
            REPO, body, config_hash=canonical_sha256({"scope": body["scope"], "units": work_units()})
        )
        require(body["cuda_mapping"] == live["environment"]["cuda_mapping"], "W10 v10 CUDA mapping differs")
        if not args.preflight:
            body["authority_id"] = f"w10continuationauth{epoch}-" + canonical_sha256(body)
            immutable_write(REPO / (V11_AUTHORITY_PATH if epoch == "v11" else V10_AUTHORITY_PATH), body)
        print(f"W10 v10 authority ready: new ordinals={V10_START}–{V10_STOP - 1}; written={not args.preflight}")
        return 0

    authority = verify_v10_authority(REPO, epoch=epoch)
    runtime = REPO / authority["runtime_root"]
    stable_ids = ValidationView().stable_ids
    count = verify_v10_evidence_set(
        REPO, authority=authority, stable_ids=stable_ids,
        phase="closeout" if args.action in {"closeout", "verify"} else args.action if args.action == "plan" else "execute",
    )
    if args.action == "plan":
        require(count == 0, "W10 v10 plan must precede new evidence")
        immutable_write(runtime / authority["plan_path"], build_v10_plan(authority))
        print("W10 v10 plan ready: historical 0–230; new 231–251")
        return 0

    plan = verify_v10_plan(runtime, authority)
    if args.action == "launch":
        require(count == 0, "W10 v10 launch grant must precede new evidence")
        body = build_v10_launch(authority, plan)
        if not args.preflight:
            immutable_write(REPO / (V11_LAUNCH_PATH if epoch == "v11" else V10_LAUNCH_PATH), body)
        print(f"W10 v10 launch grant: {body['launch_id']}; written={not args.preflight}")
        return 0
    verify_v10_launch(REPO, authority, plan)
    if args.action == "execute":
        authenticate_live_w9_pascal(
            REPO, authority, config_hash=canonical_sha256({"scope": authority["scope"], "units": work_units()})
        )
        backend = dispatch(REPO, device="cuda:0", authority=authority)
        results = execute(runtime, authority=authority, backend=backend, expected_stable_ids=stable_ids)
        print(f"W10 v10 suffix executed/resumed: {len(results)}/{V10_STOP} units")
        return 0

    require(count == V10_STOP - V10_START, "W10 v10 terminal scope is incomplete")
    results = _complete_results(runtime)
    closeout, units, images = build_v10_closeout(runtime, results, authority)
    published = published_v10_units(results, authority)
    if args.action == "closeout":
        for name, value in (
            (f"continuation_unit_manifest_{epoch}.json", units),
            (f"continuation_per_image_manifest_{epoch}.json", images),
            (f"continuation_closeout_{epoch}.json", closeout),
        ):
            immutable_write(runtime / name, value)
        for name, value in (
            (f"w10_continuation_unit_manifest_{epoch}.json", units),
            (f"w10_continuation_per_image_manifest_{epoch}.json", images),
            (f"w10_continuation_units_{epoch}.json", published),
            (f"w10_continuation_closeout_{epoch}.json", closeout),
        ):
            immutable_write(PUBLISHED / name, value)
        print(f"W10 v10 closeout: {closeout['closeout_id']}")
        return 0

    require({path.name for path in runtime.iterdir()} >= {f"continuation_unit_manifest_{epoch}.json", f"continuation_per_image_manifest_{epoch}.json", f"continuation_closeout_{epoch}.json"}, "W10 suffix runtime closeout is missing")
    for name, value in (
        (f"continuation_unit_manifest_{epoch}.json", units),
        (f"continuation_per_image_manifest_{epoch}.json", images),
        (f"continuation_closeout_{epoch}.json", closeout),
    ):
        path = runtime / name
        require(path.read_bytes() == canonical_bytes(value), f"W10 v10 runtime {name} differs")
    for key, value in (("units", published), ("unit_manifest", units), ("images", images), ("closeout", closeout)):
        require((PUBLISHED / PUBLISHED_FILES[key].name.replace("_v10.json", f"_{epoch}.json")).read_bytes() == canonical_bytes(value), f"W10 suffix published {key} differs")
    print(f"W10 v10 terminal verifier PASS: {closeout['closeout_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
