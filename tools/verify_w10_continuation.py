#!/usr/bin/env python3
"""Verify W10 v9 continuation on the worker or from committed published evidence."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evaluation.downstream_v4 import read_json, require  # noqa: E402
from evaluation.w10_backends import ValidationView  # noqa: E402
from evaluation.w10_continuation import (  # noqa: E402
    CONTINUATION_START,
    CONTINUATION_STOP,
    build_continuation_closeout,
    build_continuation_plan,
    published_continuation_units,
    verify_continuation_authority,
    verify_continuation_plan,
    verify_launch_authorization,
    verify_suffix_evidence_set,
    verify_suffix_streams,
    verify_worker_prefix,
)
from evaluation.w10_evidence import per_image_relative_path, unit_relative_path  # noqa: E402
from evaluation.w10_rehearsal import validate_unit, work_units  # noqa: E402
from runtime.source_epochs import W10_V8_AUTHORITY_ID, W10_V8_MANIFEST_ID, v8_continuation_custody  # noqa: E402
from training.deterministic_core import canonical_bytes, canonical_sha256  # noqa: E402

PUBLISHED_ROOT = REPO / "results/learned/w10"
PUBLISHED_PATHS = {
    "units": PUBLISHED_ROOT / "w10_continuation_units_v9.json",
    "unit_manifest": PUBLISHED_ROOT / "w10_continuation_unit_manifest_v9.json",
    "images": PUBLISHED_ROOT / "w10_continuation_per_image_manifest_v9.json",
    "closeout": PUBLISHED_ROOT / "w10_continuation_closeout_v9.json",
}


def _address(value: dict, field: str, prefix: str) -> None:
    body = dict(value)
    identifier = body.pop(field, None)
    require(identifier == prefix + canonical_sha256(body), f"W10 v9 {field} differs")


def verify_published() -> dict:
    """Authenticate committed bodies and digests; do not claim worker-byte access."""

    authority = verify_continuation_authority(REPO)
    verify_launch_authorization(REPO, authority, build_continuation_plan(authority))
    custody = v8_continuation_custody(REPO)
    values = {key: read_json(path, f"W10 published {key}") for key, path in PUBLISHED_PATHS.items()}
    published = values["units"]
    unit_manifest = values["unit_manifest"]
    images = values["images"]
    closeout = values["closeout"]
    for value, field, prefix in (
        (published, "published_units_id", "w10continuationunits-"),
        (unit_manifest, "unit_manifest_id", "w10unitmanifest-"),
        (images, "per_image_manifest_id", "w10imagemanifest-"),
        (closeout, "closeout_id", "w10continuationcloseout-"),
    ):
        _address(value, field, prefix)
    require(published["unit_count"] == CONTINUATION_STOP and len(published["units"]) == CONTINUATION_STOP, "W10 v9 published scope differs")
    require(published["artifact_role"] == "W10_VALIDATION_CONTINUATION_PUBLISHED_UNITS_V9" and published["validation_only"] is True and published["test"] == "SEALED" and published["test_access"] == 0, "W10 v9 published boundary differs")
    require(published["historical_custody_id"] == custody["custody_id"] and published["original_authority_id"] == W10_V8_AUTHORITY_ID and published["continuation_authority_id"] == authority["authority_id"], "W10 v9 published provenance differs")
    require(unit_manifest["unit_count"] == CONTINUATION_STOP and len(unit_manifest["units"]) == CONTINUATION_STOP, "W10 v9 unit manifest scope differs")
    require(unit_manifest["artifact_role"] == "W10_VALIDATION_REHEARSAL_UNIT_MANIFEST" and images["artifact_role"] == "W10_VALIDATION_REHEARSAL_PER_IMAGE_MANIFEST", "W10 v9 manifest roles differ")
    expected_streams = []
    expected_provenance = []
    for index, (unit, value, row) in enumerate(zip(work_units(), published["units"], unit_manifest["units"], strict=True)):
        validate_unit(value, unit)
        expected_row = {
            "ordinal": index,
            "unit_id": value["unit_id"],
            "unit_path": unit_relative_path(unit, "json"),
            "unit_sha256": hashlib.sha256(canonical_bytes(value)).hexdigest(),
            "per_image_path": value["per_image_path"],
            "per_image_sha256": value["per_image_sha256"],
            "n_correct": value["n_correct"],
            "n_total": value["n_total"],
            "scorer_variants": value["scorer_variants"],
        }
        require(row == expected_row, "W10 v9 published unit manifest row differs")
        if index < CONTINUATION_START:
            require(row["unit_sha256"] == custody["units"][index]["sha256"] and value["unit_id"] == custody["units"][index]["unit_id"], "historical W10 published unit differs from v8 custody")
            require("execution_authority_id" not in value["binding"], "historical W10 published unit was relabelled")
        else:
            require(value["binding"].get("execution_authority_id") == authority["authority_id"], "v9 published unit has wrong execution authority")
        expected_provenance.append({
            "ordinal": index,
            "source_manifest_id": W10_V8_MANIFEST_ID if index < CONTINUATION_START else authority["source_manifest"]["manifest_id"],
            "authority_id": W10_V8_AUTHORITY_ID if index < CONTINUATION_START else authority["authority_id"],
            "unit_id": value["unit_id"],
        })
        for variant_index, variant in enumerate(value["scorer_variants"]):
            require(variant["path"] == per_image_relative_path(unit, variant_index), "W10 v9 scorer path differs")
            expected_streams.append({
                "ordinal": index,
                "classifier_variant": variant["classifier_variant"],
                "per_image_path": variant["path"],
                "per_image_sha256": variant["sha256"],
                "n_correct": variant["n_correct"],
                "n_total": variant["n_total"],
            })
    require(unit_manifest["ordered_unit_ids_digest"] == canonical_sha256({"unit_ids": [unit["unit_id"] for unit in published["units"]]}), "W10 v9 ordered unit digest differs")
    require(images["stream_count"] == len(expected_streams) and images["streams"] == expected_streams, "W10 v9 published stream manifest differs")
    require(images["ordered_per_image_digest"] == canonical_sha256({"streams": expected_streams}), "W10 v9 ordered stream digest differs")
    for actual, frozen in zip(expected_streams[:custody["stream_count"]], custody["streams"], strict=True):
        require(actual["ordinal"] == frozen["ordinal"] and actual["per_image_path"] == frozen["path"] and actual["per_image_sha256"] == frozen["canonical_sha256"], "historical W10 published stream differs from v8 custody")
    require(closeout["artifact_role"] == "W10_VALIDATION_CONTINUATION_CLOSEOUT_V9" and closeout["status"] == "COMPLETE", "W10 v9 closeout role/status differs")
    require(closeout["ordinal_provenance"] == expected_provenance and closeout["unit_count"] == CONTINUATION_STOP and closeout["stream_count"] == len(expected_streams), "W10 v9 closeout provenance differs")
    require(closeout["original_authority_id"] == W10_V8_AUTHORITY_ID and closeout["continuation_authority_id"] == authority["authority_id"] and closeout["scope_sha256"] == authority["scope_sha256"], "W10 v9 closeout authority/scope differs")
    require(closeout["historical_custody_id"] == custody["custody_id"] and closeout["historical_custody_sha256"] == authority["historical_custody_sha256"], "W10 v9 closeout custody differs")
    require(closeout["ordered_unit_ids_digest"] == unit_manifest["ordered_unit_ids_digest"] and closeout["ordered_per_image_digest"] == images["ordered_per_image_digest"], "W10 v9 closeout ordered digests differ")
    require(closeout["unit_manifest_sha256"] == canonical_sha256(unit_manifest) and closeout["per_image_manifest_sha256"] == canonical_sha256(images), "W10 v9 closeout manifest digests differ")
    require(closeout["validation_only"] is True and closeout["test"] == "SEALED" and closeout["test_access"] == 0, "W10 v9 closeout crossed test boundary")
    return {"closeout_id": closeout["closeout_id"], "unit_count": CONTINUATION_STOP, "stream_count": len(expected_streams)}


def verify_worker() -> dict:
    authority = verify_continuation_authority(REPO)
    runtime = REPO / authority["runtime_root"]
    stable_ids = ValidationView().stable_ids
    custody = verify_worker_prefix(REPO, expected_stable_ids=stable_ids, allow_suffix=True)
    plan = verify_continuation_plan(runtime, authority)
    verify_launch_authorization(REPO, authority, plan)
    verify_suffix_evidence_set(runtime, custody, authority)
    verify_suffix_streams(runtime, expected_stable_ids=stable_ids, authority=authority)
    results = []
    for unit in work_units():
        value = read_json(runtime / unit_relative_path(unit, "json"), "W10 v9 worker unit")
        validate_unit(value, unit)
        results.append(value)
    closeout, units, images = build_continuation_closeout(runtime, results, authority)
    require(read_json(runtime / "continuation_closeout_v9.json", "W10 v9 runtime closeout") == closeout, "W10 v9 runtime closeout differs")
    require(read_json(runtime / "continuation_unit_manifest_v9.json", "W10 v9 runtime unit manifest") == units, "W10 v9 runtime unit manifest differs")
    require(read_json(runtime / "continuation_per_image_manifest_v9.json", "W10 v9 runtime image manifest") == images, "W10 v9 runtime image manifest differs")
    require(read_json(PUBLISHED_PATHS["units"], "W10 v9 published units") == published_continuation_units(results, authority), "W10 v9 published unit bodies differ")
    verify_published()
    return {"closeout_id": closeout["closeout_id"], "unit_count": CONTINUATION_STOP}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published", action="store_true", help="verify committed evidence without worker bytes")
    args = parser.parse_args(argv)
    result = verify_published() if args.published else verify_worker()
    print(f"W10 v9 continuation verifier PASS: {result['closeout_id']}; worker_bytes={not args.published}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
