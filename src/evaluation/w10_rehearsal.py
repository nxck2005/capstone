"""Frozen W10 validation-only one-cell/full-grid/all-systems orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from config.params import get
from evaluation.downstream_v4 import EXPECTED_SNR_GRID, immutable_write, require
from training.deterministic_core import canonical_sha256

W10_CELL = (0, 0)
W10_SYSTEMS = tuple(str(value) for value in get("artifacts.system_values"))


def work_units() -> tuple[dict[str, Any], ...]:
    return tuple(
        {"ordinal": ordinal, "system": system, "snr_db": snr, "train_seed": 0, "channel_seed": 0, "split": "val"}
        for ordinal, (system, snr) in enumerate((system, snr) for system in W10_SYSTEMS for snr in EXPECTED_SNR_GRID)
    )


def validate_unit(value: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    require(value.get("artifact_role") == "W10_VALIDATION_REHEARSAL_UNIT", "W10 unit role differs")
    for key in ("ordinal", "system", "snr_db", "train_seed", "channel_seed", "split"):
        require(value.get(key) == expected[key], f"W10 unit {key} differs")
    require(value.get("validation_only") is True and value.get("test_access") == 0 and value.get("test") == "SEALED", "W10 unit crossed test boundary")
    require(isinstance(value.get("n_total"), int) and value["n_total"] > 0, "W10 unit denominator is invalid")
    require(isinstance(value.get("n_correct"), int) and 0 <= value["n_correct"] <= value["n_total"], "W10 unit correctness is invalid")
    body = dict(value); identifier = body.pop("unit_id", None)
    require(identifier == "w10unit-" + canonical_sha256(body), "W10 unit identity differs")


def execute(
    output_root: Path,
    *,
    authority: Mapping[str, Any],
    evaluator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Execute/resume exact units through a caller-supplied scientific backend.

    The backend owns model loading; this layer owns scope, ordering,
    idempotence, and the validation/test boundary. Synthetic tests use the
    same interface without importing any scientific dataset.
    """

    require(authority.get("authority_kind") == "W10_VALIDATION_REHEARSAL_AUTHORITY", "W10 execution authority role differs")
    require(authority.get("cell") == {"train_seed": 0, "channel_seed": 0}, "W10 execution authority cell differs")
    require(authority.get("snr_grid_db") == list(EXPECTED_SNR_GRID) and authority.get("systems") == list(W10_SYSTEMS), "W10 execution authority grid/systems differ")
    require(authority.get("validation_only") is True and authority.get("test_authorized") is False and authority.get("test_access") == 0, "W10 execution authority crosses test boundary")
    results = []
    for expected in work_units():
        path = output_root / "units" / f"{expected['ordinal']:03d}-{expected['system']}-snr{expected['snr_db']:+03d}.json"
        if path.is_file() and not path.is_symlink():
            import json
            value = json.loads(path.read_bytes())
        else:
            supplied = dict(evaluator(expected))
            value = {**expected, **supplied, "artifact_role": "W10_VALIDATION_REHEARSAL_UNIT", "validation_only": True, "test": "SEALED", "test_access": 0}
            value["unit_id"] = "w10unit-" + canonical_sha256(value)
            validate_unit(value, expected)
            immutable_write(path, value)
        validate_unit(value, expected)
        results.append(value)
    return results


def closeout(results: list[Mapping[str, Any]], *, source_commit: str, authority_id: str) -> dict[str, Any]:
    expected = work_units()
    require(len(results) == len(expected), "W10 unit count differs")
    for value, unit in zip(results, expected, strict=True):
        validate_unit(value, unit)
    body = {
        "schema_version": 1, "artifact_role": "W10_VALIDATION_ONLY_REHEARSAL_CLOSEOUT",
        "status": "COMPLETE", "source_commit": source_commit, "authority_id": authority_id,
        "split": "val", "cell": [0, 0], "snr_grid_db": list(EXPECTED_SNR_GRID),
        "systems": list(W10_SYSTEMS), "unit_count": len(results),
        "validation_only": True, "test": "SEALED", "test_access": 0,
    }
    body["closeout_id"] = "w10closeout-" + canonical_sha256(body)
    return body


__all__ = ["W10_CELL", "W10_SYSTEMS", "closeout", "execute", "validate_unit", "work_units"]
