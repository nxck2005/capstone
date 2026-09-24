#!/usr/bin/env python3
"""Verify W10's exact validation-only scope, bindings, published evidence and closeout.

``--terminal`` is a worker-custody verifier: it requires the per-image runtime.
``verify_published`` is the hosted published-evidence path: it authenticates the
committed authority, closeout, published unit bodies and manifests without the
worker runtime bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402
from evaluation.downstream_v4 import read_json, require  # noqa: E402
from evaluation.w10_bindings import pending_states, resolve_scope_bindings  # noqa: E402
from evaluation.w10_rehearsal import build_closeout, published_units, validate_unit, W10_SYSTEMS, work_units  # noqa: E402
from evaluation.w10_scope import (  # noqa: E402
    SCOPE,
    W10_DATASET,
    W10_SPLIT,
    W10_VALIDATION_DENOMINATOR,
    scope_identity,
    scope_sha256,
    snr_grid,
    unit_count,
)
from runtime.source_epochs import W10_V9_SOURCE_PATH, assert_active_epoch_closure, load_w10_manifest, source_record  # noqa: E402
from training.deterministic_core import canonical_sha256  # noqa: E402

AUTHORITY = REPO / "results/learned/w10/w10_rehearsal_authorization.json"
CLOSEOUT = REPO / "results/learned/w10/w10_rehearsal_closeout.json"
UNIT_MANIFEST = REPO / "results/learned/w10/w10_rehearsal_unit_manifest.json"
PER_IMAGE_MANIFEST = REPO / "results/learned/w10/w10_rehearsal_per_image_manifest.json"
PUBLISHED_UNITS = REPO / "results/learned/w10/w10_rehearsal_units.json"
W10_RUNTIME_ROOT = "checkpoints/w10_rehearsal"
G11_AUTHORITY = REPO / "results/learned/g11/g11_execution_authorization_v4.json"
G11_TERMINAL = REPO / "results/learned/g11/g11_terminal_closeout.json"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _g11_closure(*, published: bool = False) -> dict:
    if published:
        from verify_g11_published import verify_published as verify_g11_published

        published_value = verify_g11_published()
        return {
            "authority_id": str(published_value["authority_id"]),
            "authority_path": str(G11_AUTHORITY.relative_to(REPO)),
            "authority_sha256": _sha256_file(G11_AUTHORITY),
            "terminal_id": str(published_value["terminal_id"]),
            "terminal_path": str(G11_TERMINAL.relative_to(REPO)),
            "terminal_sha256": _sha256_file(G11_TERMINAL),
            "decision": "GREEN",
            "test": "SEALED",
            "test_access": 0,
        }
    from verify_g11 import verify_authority as verify_g11_authority, verify_terminal as verify_g11_terminal

    authority = verify_g11_authority()
    terminal = verify_g11_terminal(G11_TERMINAL)
    require(terminal.get("decision") == "GREEN" and terminal.get("test_access") == 0, "G11 terminal is not GREEN")
    return {
        "authority_id": str(authority["authority_id"]),
        "authority_path": str(G11_AUTHORITY.relative_to(REPO)),
        "authority_sha256": _sha256_file(G11_AUTHORITY),
        "terminal_id": str(terminal["terminal_id"]),
        "terminal_path": str(G11_TERMINAL.relative_to(REPO)),
        "terminal_sha256": _sha256_file(G11_TERMINAL),
        "decision": "GREEN",
        "test": "SEALED",
        "test_access": 0,
    }


def verify_authority(
    path: Path = AUTHORITY,
    *,
    verify_bindings_runtime: bool = True,
    verify_g11_runtime: bool = True,
) -> dict:
    value = read_json(path, "W10 authority")
    body = dict(value)
    identifier = body.pop("authority_id", None)
    require(identifier == "w10rehearsalauth-" + canonical_sha256(body), "W10 authority ID differs")
    require(value.get("schema_version") == 2 and value.get("authority_kind") == "W10_VALIDATION_REHEARSAL_AUTHORITY", "W10 authority role differs")
    require(value.get("status") == "FROZEN_W10_ONLY_PRE_EXECUTION", "W10 authority status differs")
    if (REPO / W10_V9_SOURCE_PATH).is_file():
        source = load_w10_manifest(REPO, live=False, epoch="v8")
        assert_active_epoch_closure(REPO, source)
    else:
        source = load_w10_manifest(REPO, live=True)
    require(value.get("source_manifest") == source_record(REPO, source) and value.get("source_binding") == source and value.get("source_commit") == source["source_commit"], "W10 source binding differs")
    require(value.get("scope") == scope_identity(), "W10 authority scope differs")
    require(value.get("scope_sha256") == scope_sha256(), "W10 authority scope digest differs")
    require(value.get("scope_table_module") == "src/evaluation/w10_scope.py", "W10 scope table module differs")
    require(value.get("snr_grid_db") == list(snr_grid()), "W10 authority SNR grid differs")
    require(value.get("systems") == list(dict.fromkeys(entry.system for entry in SCOPE)), "W10 authority systems differ")
    require(value.get("ratios") == sorted({entry.bw_ratio for entry in SCOPE}), "W10 authority ratios differ")
    require(value.get("learned_ratios") == list(get("evaluation.w10_learned_ratios")), "W10 authority learned ratios differ")
    require(value.get("cell") == {"train_seed": 0, "channel_seed": 0} and value.get("split") == W10_SPLIT and value.get("dataset") == W10_DATASET, "W10 authority cell/split/dataset differs")
    require(value.get("denominator") == W10_VALIDATION_DENOMINATOR, "W10 authority denominator differs")
    require(value.get("unit_count") == unit_count() and value.get("unit_count_derived_from_scope") is True, "W10 authority unit count differs")
    bindings = resolve_scope_bindings(REPO, verify_runtime=verify_bindings_runtime)
    require(value.get("bindings") == bindings, "W10 authority bindings differ from the live frozen artifacts")
    require(not pending_states(bindings), "W10 authority carries a pending binding")
    expected_g11 = _g11_closure() if verify_g11_runtime else _g11_closure(published=True)
    require(value.get("g11_closure") == expected_g11, "W10 authority G11 closure differs")
    require(value.get("validation_only") is True and value.get("test_authorized") is False and value.get("test_access") == 0 and value.get("test") == "SEALED", "W10 authority test boundary differs")
    require(value.get("papr_training_authorized") is False and value.get("papr_training_run_count") == 1 and value.get("papr_lifecycle_bound") is True, "W10 authority PAPR lifecycle binding differs")
    require(value.get("runtime_root") == W10_RUNTIME_ROOT, "W10 authority runtime root differs")
    require(value.get("host") == "confessor" and value.get("device") == "cuda:0", "W10 authority host/device differs")
    require(value.get("gpu_name") == "NVIDIA TITAN Xp" and value.get("gpu_uuid") == "GPU-46acd0f2-2ff5-1a43-cac9-2ae20e56dc9a", "W10 authority worker is not the exact Confessor TITAN Xp")  # literal-ok: frozen profile identity
    require(value.get("cuda_visible_devices") == value.get("gpu_uuid"), "W10 authority CUDA binding differs")
    mapping = value.get("cuda_mapping")
    require(mapping == {
        "cuda_visible_devices": value.get("gpu_uuid"),
        "logical_device": "cuda:0",
        "cuda0_gpu_uuid": value.get("gpu_uuid"),
        "cuda0_gpu_name": value.get("gpu_name"),
        "cuda0_compute_capability": value.get("compute_capability"),
        "device_count": 1,
    }, "W10 authority CUDA mapping differs")
    return value


def _published_records() -> tuple[dict, dict, dict]:
    units = read_json(PUBLISHED_UNITS, "W10 published units")
    body = dict(units)
    identifier = body.pop("published_units_id", None)
    require(identifier == "w10publishedunits-" + canonical_sha256(body), "W10 published units ID differs")
    expected = work_units()
    require(units.get("unit_count") == len(expected) and len(units.get("units", [])) == len(expected), "W10 published unit count differs")
    require(units.get("test") == "SEALED" and units.get("test_access") == 0, "W10 published units crossed test boundary")
    for value, unit in zip(units["units"], expected, strict=True):
        validate_unit(value, unit)
    unit_manifest = read_json(UNIT_MANIFEST, "W10 published unit manifest")
    images = read_json(PER_IMAGE_MANIFEST, "W10 published per-image manifest")
    unit_manifest_body = dict(unit_manifest)
    unit_manifest_id = unit_manifest_body.pop("unit_manifest_id", None)
    require(unit_manifest_id == "w10unitmanifest-" + canonical_sha256(unit_manifest_body), "W10 unit manifest ID differs")
    images_body = dict(images)
    images_id = images_body.pop("per_image_manifest_id", None)
    require(images_id == "w10imagemanifest-" + canonical_sha256(images_body), "W10 per-image manifest ID differs")
    require(unit_manifest.get("unit_count") == len(expected), "W10 unit manifest count differs")
    expected_stream_count = sum(len(value.get("scorer_variants", ())) for value in units["units"])
    require(images.get("stream_count") == expected_stream_count, "W10 per-image manifest count differs")
    require(
        unit_manifest.get("ordered_unit_ids_digest")
        == canonical_sha256({"unit_ids": [value["unit_id"] for value in units["units"]]}),
        "W10 unit manifest ordered digest differs",
    )
    manifest_units = unit_manifest.get("units")
    require(isinstance(manifest_units, list) and len(manifest_units) == len(expected), "W10 unit manifest rows differ")
    for published, manifest_row in zip(units["units"], manifest_units, strict=True):
        require(manifest_row.get("unit_id") == published["unit_id"], "W10 unit manifest unit binding differs")
        require(manifest_row.get("scorer_variants") == [
            {
                "classifier_variant": variant["classifier_variant"],
                "path": variant["path"],
                "sha256": variant["sha256"],
                "n_correct": variant["n_correct"],
                "n_total": variant["n_total"],
            }
            for variant in published["scorer_variants"]
        ], "W10 unit manifest scorer custody differs")
    return units, unit_manifest, images


def verify_terminal() -> None:
    authority = verify_authority()
    runtime = REPO / str(authority["runtime_root"])
    units = []
    for expected in work_units():
        path = runtime / "units" / f"{expected['ordinal']:03d}-{expected['system']}-{expected['bw_ratio']}-snr{int(expected['snr_db']):+03d}.json"
        value = read_json(path, f"W10 unit {expected['ordinal']}")
        validate_unit(value, expected)
        units.append(value)
    body, units_manifest, images_manifest = build_closeout(runtime, units, authority=authority)
    stored_units = read_json(runtime / "unit_manifest.json", "W10 unit manifest")
    stored_images = read_json(runtime / "per_image_manifest.json", "W10 per-image manifest")
    require(stored_units == units_manifest and stored_images == images_manifest, "W10 closeout manifests differ from the rebuilt evidence")
    require(read_json(UNIT_MANIFEST, "W10 published unit manifest") == units_manifest, "W10 published unit manifest differs from the runtime")
    require(read_json(PER_IMAGE_MANIFEST, "W10 published per-image manifest") == images_manifest, "W10 published per-image manifest differs from the runtime")
    published = published_units(units, authority=authority)
    require(read_json(PUBLISHED_UNITS, "W10 published units") == published, "W10 published unit bodies differ from the runtime")
    stored = read_json(CLOSEOUT, "W10 closeout")
    require(stored == body, "W10 closeout does not content-bind the exact completed evidence")
    require(stored.get("unit_count") == unit_count(), "W10 closeout unit count differs")
    require(stored.get("ordered_unit_ids_digest") == units_manifest["ordered_unit_ids_digest"], "W10 closeout unit digest differs")
    require(stored.get("ordered_per_image_digest") == images_manifest["ordered_per_image_digest"], "W10 closeout per-image digest differs")
    require(stored.get("test") == "SEALED" and stored.get("test_access") == 0, "W10 closeout crossed test boundary")


def verify_published(*, verify_bindings_runtime: bool = False) -> dict:
    """Hosted published-evidence authentication without worker runtime bytes."""

    if not verify_bindings_runtime:
        # W10's PAPR prerequisite has its own published-evidence verifier.  Run
        # that explicit boundary before resolving W10 bindings so this hosted
        # path never treats the worker terminal chain as a substitute.
        from verify_papr_training_published import verify_published as verify_papr_published

        verify_papr_published()
    authority = verify_authority(
        verify_bindings_runtime=verify_bindings_runtime,
        verify_g11_runtime=False,
    )
    units, unit_manifest, images = _published_records()
    require(units["authority_id"] == authority["authority_id"] and units["scope_sha256"] == authority["scope_sha256"], "W10 published units authority/scope differs")
    require(unit_manifest.get("ordered_unit_ids_digest") == canonical_sha256({"unit_ids": [value["unit_id"] for value in units["units"]]}), "W10 published unit digest differs")
    require(images.get("ordered_per_image_digest") == canonical_sha256({"streams": images["streams"]}), "W10 published per-image digest differs")
    closeout_value = read_json(CLOSEOUT, "W10 closeout")
    body = dict(closeout_value)
    identifier = body.pop("closeout_id", None)
    require(identifier == "w10closeout-" + canonical_sha256(body), "W10 closeout ID differs")
    require(closeout_value.get("authority_id") == authority["authority_id"], "W10 closeout authority differs")
    require(closeout_value.get("scope_sha256") == authority["scope_sha256"], "W10 closeout scope differs")
    require(closeout_value.get("unit_count") == len(units["units"]), "W10 closeout unit count differs")
    require(closeout_value.get("ordered_unit_ids_digest") == unit_manifest["ordered_unit_ids_digest"], "W10 closeout unit digest differs")
    require(closeout_value.get("ordered_per_image_digest") == images["ordered_per_image_digest"], "W10 closeout per-image digest differs")
    require(closeout_value.get("test") == "SEALED" and closeout_value.get("test_access") == 0, "W10 closeout crossed test boundary")
    return {
        "authority_id": authority["authority_id"],
        "closeout_id": closeout_value["closeout_id"],
        "unit_count": closeout_value["unit_count"],
        "test_access": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal", action="store_true")
    parser.add_argument("--published", action="store_true", help="use the clean-clone published-evidence path")
    args = parser.parse_args(argv)
    if args.published:
        value = verify_published()
        print(f"W10 rehearsal published-evidence verifier PASS: {value['closeout_id']}")
        return 0
    verify_authority()
    if args.terminal:
        verify_terminal()
        print("W10 rehearsal verifier PASS: terminal")
        return 0
    print("W10 rehearsal verifier PASS: authority only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
