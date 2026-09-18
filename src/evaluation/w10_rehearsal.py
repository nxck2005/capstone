"""Frozen W10 validation-only rehearsal orchestration (AM-98).

The orchestration owns exact scope, ordering, idempotence, authority checks,
per-image custody and closeout content binding.  The production scientific
dispatcher lives in :mod:`evaluation.w10_dispatch`; this module never loads a
model or a dataset itself, so its contracts stay source- and test-independent.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from evaluation.downstream_v4 import immutable_write, read_json, require
from evaluation.w10_evidence import (
    W10_UNIT_PREFIX,
    W10_UNIT_ROLE,
    build_per_image_record,
    per_image_relative_path,
    recompute_n_correct,
    unit_relative_path,
    validate_per_image,
)
from evaluation.w10_scope import (
    W10_CELL,
    W10_CHANNEL_SEED,
    W10_SPLIT,
    W10_TRAIN_SEED,
    W10_VALIDATION_DENOMINATOR,
    SCOPE,
    scope_evidence_requirement,
    scope_sha256,
    unit_count,
    work_units,
)
from training.deterministic_core import canonical_sha256

W10_SYSTEMS = tuple(dict.fromkeys(entry.system for entry in SCOPE))

W10_UNIT_KEYS = (
    "ordinal",
    "system",
    "bw_ratio",
    "role",
    "backend",
    "snr_db",
    "train_seed",
    "channel_seed",
    "split",
)


def unit_evidence_requirement(expected: Mapping[str, Any]) -> dict[str, Any]:
    """The scope contract a unit must satisfy, keyed by role."""

    for entry in SCOPE:
        if (
            entry.system == expected["system"]
            and entry.bw_ratio == expected["bw_ratio"]
            and entry.role == expected["role"]
        ):
            return scope_evidence_requirement(entry)
    raise ValueError(f"no W10 scope entry for {expected!r}")


def unit_body(
    *,
    expected: Mapping[str, Any],
    evidence: Mapping[str, Any],
    per_image: Mapping[str, Any],
    per_image_path: str,
    scorer_variants: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the immutable unit record around one primary per-image stream."""

    rows = evidence["per_image"]
    body = {
        "schema_version": 1,
        "artifact_role": W10_UNIT_ROLE,
        **{key: expected[key] for key in W10_UNIT_KEYS},
        "validation_only": True,
        "test": "SEALED",
        "test_access": 0,
        "denominator": W10_VALIDATION_DENOMINATOR,
        "n_correct": int(evidence["n_correct"]),
        "n_total": int(evidence["n_total"]),
        "coverage_rate": evidence.get("coverage_rate"),
        "acc_given_delivery": evidence.get("acc_given_delivery"),
        "decode_failure_count": int(evidence.get("decode_failure_count", 0)),
        "infeasible_count": int(evidence.get("infeasible_count", 0)),
        "mean_papr_db": evidence.get("mean_papr_db"),
        "source_bytes": evidence.get("source_bytes"),
        "binding": dict(evidence["binding"]),
        "per_image_path": per_image_path,
        "per_image_sha256": per_image["_sha256"],
        "scorer_variants": scorer_variants,
    }
    body["unit_id"] = W10_UNIT_PREFIX + canonical_sha256(body)
    return body


def validate_unit(value: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Fail closed on any scope, boundary, denominator or identity divergence."""

    require(value.get("artifact_role") == W10_UNIT_ROLE, "W10 unit role differs")
    require(value.get("schema_version") == 1, "W10 unit schema differs")
    for key in W10_UNIT_KEYS:
        require(value.get(key) == expected[key], f"W10 unit {key} differs")
    require(value.get("validation_only") is True and value.get("test_access") == 0 and value.get("test") == "SEALED", "W10 unit crossed test boundary")
    require(value.get("split") == W10_SPLIT and value.get("train_seed") == W10_TRAIN_SEED and value.get("channel_seed") == W10_CHANNEL_SEED, "W10 unit cell/split differs")
    require(value.get("denominator") == W10_VALIDATION_DENOMINATOR and value.get("n_total") == W10_VALIDATION_DENOMINATOR, "W10 unit denominator differs")
    require(isinstance(value.get("n_correct"), int) and 0 <= value["n_correct"] <= value["n_total"], "W10 unit correctness is invalid")
    require(isinstance(value.get("binding"), Mapping) and value["binding"], "W10 unit scientific binding is missing")
    require(value.get("per_image_path") == per_image_relative_path(expected, 0), "W10 unit per-image path differs")
    require(isinstance(value.get("per_image_sha256"), str) and len(value["per_image_sha256"]) == 64, "W10 unit per-image digest is malformed")  # literal-ok: SHA-256 width
    variants = value.get("scorer_variants")
    require(isinstance(variants, list) and variants, "W10 unit scorer variants are missing")
    requirement = unit_evidence_requirement(expected)
    require(variants[0].get("classifier_variant") == requirement["primary_classifier_variant"], "W10 unit primary scorer variant differs")
    for variant in variants:
        require(isinstance(variant, Mapping) and variant.get("n_total") == W10_VALIDATION_DENOMINATOR, "W10 scorer variant denominator differs")
        require(isinstance(variant.get("sha256"), str) and len(variant["sha256"]) == 64, "W10 scorer variant digest is malformed")  # literal-ok: SHA-256 width
    require(variants[0].get("n_correct") == value.get("n_correct"), "W10 unit aggregate is not the primary stream aggregate")
    body = dict(value)
    identifier = body.pop("unit_id", None)
    require(identifier == W10_UNIT_PREFIX + canonical_sha256(body), "W10 unit identity differs")


def _write_per_image(
    runtime: Path,
    expected: Mapping[str, Any],
    *,
    variant_index: int,
    classifier_variant: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    relative = per_image_relative_path(expected, variant_index)
    record = build_per_image_record(
        ordinal=int(expected["ordinal"]), unit_key=relative, rows=rows
    )
    path = runtime / relative
    immutable_write(path, record)
    digest = canonical_sha256(record)
    return {
        "classifier_variant": classifier_variant,
        "path": relative,
        "sha256": digest,
        "n_correct": recompute_n_correct(rows),
        "n_total": len(rows),
        "_sha256": digest,
        "_rows": rows,
    }


def execute(
    runtime_root: Path,
    *,
    authority: Mapping[str, Any],
    backend: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    expected_stable_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Execute/resume the exact authority scope through a scientific backend.

    The backend owns model and dataset loading and returns one evidence
    mapping per unit: ``n_correct``, ``n_total``, ``binding``, ``per_image``
    (1000 SR-18 rows) and optional additional scorer streams.  This layer
    owns scope, ordering, idempotence, per-image custody and the validation
    boundary.
    """

    require(authority.get("authority_kind") == "W10_VALIDATION_REHEARSAL_AUTHORITY", "W10 execution authority role differs")
    require(authority.get("cell") == {"train_seed": W10_TRAIN_SEED, "channel_seed": W10_CHANNEL_SEED}, "W10 execution authority cell differs")
    require(authority.get("split") == W10_SPLIT, "W10 execution authority split differs")
    require(authority.get("systems") == list(W10_SYSTEMS), "W10 execution authority systems differ")
    require(authority.get("unit_count") == unit_count(), "W10 execution authority unit count differs")
    require(authority.get("scope_sha256") == scope_sha256(), "W10 execution authority scope digest differs")
    require(authority.get("validation_only") is True and authority.get("test_authorized") is False and authority.get("test_access") == 0, "W10 execution authority crosses test boundary")
    units_dir = runtime_root / "units"
    results: list[dict[str, Any]] = []
    for expected in work_units():
        path = units_dir / f"{expected['ordinal']:03d}-{expected['system']}-{expected['bw_ratio']}-snr{int(expected['snr_db']):+03d}.json"
        relative = per_image_relative_path(expected, 0)
        per_image_path = runtime_root / relative
        if path.is_file() and not path.is_symlink():
            value = read_json(path, f"W10 unit {expected['ordinal']}")
            validate_unit(value, expected)
            require(value["per_image_path"] == relative, "W10 resumed unit per-image path differs")
            require(per_image_path.is_file() and not per_image_path.is_symlink(), "W10 resumed per-image evidence is missing")
            require(canonical_sha256(read_json(per_image_path, "W10 per-image")) == value["per_image_sha256"], "W10 resumed per-image digest differs")
            results.append(value)
            continue
        supplied = backend(expected)
        rows = [dict(row) for row in supplied["per_image"]]
        if expected_stable_ids is not None:
            validate_per_image(
                rows,
                expected_stable_ids=expected_stable_ids,
                system=str(expected["system"]),
                bw_ratio=str(expected["bw_ratio"]),
                snr_db=expected["snr_db"],
            )
        require(recompute_n_correct(rows) == int(supplied["n_correct"]), "W10 aggregate does not recompute from per-image rows")
        require(len(rows) == W10_VALIDATION_DENOMINATOR, "W10 backend per-image denominator differs")
        primary = _write_per_image(
            runtime_root,
            expected,
            variant_index=0,
            classifier_variant=str(supplied["primary_classifier_variant"]),
            rows=rows,
        )
        variants = [primary]
        for index, extra in enumerate(supplied.get("secondary_streams", ()), start=1):
            extra_rows = [dict(row) for row in extra["per_image"]]
            if expected_stable_ids is not None:
                validate_per_image(
                    extra_rows,
                    expected_stable_ids=expected_stable_ids,
                    system=str(expected["system"]),
                    bw_ratio=str(expected["bw_ratio"]),
                    snr_db=expected["snr_db"],
                )
            variants.append(
                _write_per_image(
                    runtime_root,
                    expected,
                    variant_index=index,
                    classifier_variant=str(extra["classifier_variant"]),
                    rows=extra_rows,
                )
            )
        public_variants = [{key: value for key, value in variant.items() if not key.startswith("_")} for variant in variants]
        value = unit_body(
            expected=expected,
            evidence=supplied,
            per_image=primary,
            per_image_path=relative,
            scorer_variants=public_variants,
        )
        validate_unit(value, expected)
        immutable_write(path, value)
        results.append(value)
    require(len(results) == unit_count(), "W10 execution did not cover the exact scope")
    return results


def unit_manifest(runtime_root: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for value, expected in zip(results, work_units(), strict=True):
        validate_unit(value, expected)
        unit_path = runtime_root / "units" / f"{expected['ordinal']:03d}-{expected['system']}-{expected['bw_ratio']}-snr{int(expected['snr_db']):+03d}.json"
        per_image_path = runtime_root / str(value["per_image_path"])
        require(unit_path.is_file() and per_image_path.is_file(), "W10 closeout evidence file is missing")
        rows.append({
            "ordinal": int(value["ordinal"]),
            "unit_id": str(value["unit_id"]),
            "unit_path": str(unit_path.relative_to(runtime_root)),
            "unit_sha256": hashlib.sha256(unit_path.read_bytes()).hexdigest(),
            "per_image_path": str(value["per_image_path"]),
            "per_image_sha256": str(value["per_image_sha256"]),
            "n_correct": int(value["n_correct"]),
            "n_total": int(value["n_total"]),
        })
    body = {
        "schema_version": 1,
        "artifact_role": "W10_VALIDATION_REHEARSAL_UNIT_MANIFEST",
        "unit_count": len(rows),
        "ordered_unit_ids_digest": canonical_sha256({"unit_ids": [row["unit_id"] for row in rows]}),
        "units": rows,
    }
    body["unit_manifest_id"] = "w10unitmanifest-" + canonical_sha256(body)
    return body


def per_image_manifest(runtime_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    streams = []
    for row in manifest["units"]:
        path = runtime_root / str(row["per_image_path"])
        streams.append({
            "ordinal": row["ordinal"],
            "per_image_path": row["per_image_path"],
            "per_image_sha256": row["per_image_sha256"],
            "n_total": row["n_total"],
        })
    body = {
        "schema_version": 1,
        "artifact_role": "W10_VALIDATION_REHEARSAL_PER_IMAGE_MANIFEST",
        "stream_count": len(streams),
        "ordered_per_image_digest": canonical_sha256({"streams": streams}),
        "streams": streams,
    }
    body["per_image_manifest_id"] = "w10imagemanifest-" + canonical_sha256(body)
    return body


def build_closeout(
    runtime_root: Path,
    results: list[Mapping[str, Any]],
    *,
    authority: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the exact scope and build the three content-bound records."""

    expected = work_units()
    require(len(results) == len(expected), "W10 unit count differs")
    for value, unit in zip(results, expected, strict=True):
        validate_unit(value, unit)
    units = unit_manifest(runtime_root, [dict(value) for value in results])
    images = per_image_manifest(runtime_root, units)
    require(units["unit_count"] == len(expected), "W10 unit manifest count differs")
    body = {
        "schema_version": 1,
        "artifact_role": "W10_VALIDATION_ONLY_REHEARSAL_CLOSEOUT",
        "status": "COMPLETE",
        "authority_id": authority["authority_id"],
        "source_manifest_id": authority["source_manifest"]["manifest_id"],
        "source_commit": authority["source_commit"],
        "scope_sha256": authority["scope_sha256"],
        "split": W10_SPLIT,
        "cell": list(W10_CELL),
        "systems": list(W10_SYSTEMS),
        "ratios": sorted({entry.bw_ratio for entry in SCOPE}),
        "unit_count": len(results),
        "unit_manifest": {
            "path": "unit_manifest.json",
            "unit_manifest_id": units["unit_manifest_id"],
            "sha256": canonical_sha256(units),
        },
        "per_image_manifest": {
            "path": "per_image_manifest.json",
            "per_image_manifest_id": images["per_image_manifest_id"],
            "sha256": canonical_sha256(images),
        },
        "ordered_unit_ids_digest": units["ordered_unit_ids_digest"],
        "ordered_per_image_digest": images["ordered_per_image_digest"],
        "validation_only": True,
        "test": "SEALED",
        "test_access": 0,
    }
    body["closeout_id"] = "w10closeout-" + canonical_sha256(body)
    return body, units, images


def closeout(
    runtime_root: Path,
    results: list[Mapping[str, Any]],
    *,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Content-bind the exact completed scope; never a bare numeric count."""

    body, units, images = build_closeout(runtime_root, results, authority=authority)
    immutable_write(runtime_root / "unit_manifest.json", units)
    immutable_write(runtime_root / "per_image_manifest.json", images)
    return body


__all__ = [
    "W10_SYSTEMS",
    "closeout",
    "execute",
    "unit_body",
    "unit_evidence_requirement",
    "validate_unit",
    "work_units",
]
