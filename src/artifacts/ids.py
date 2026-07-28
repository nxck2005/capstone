"""Content-addressed artifact identities and strict paired joins (SR-18)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from config.params import get
from config.run_config import canonical_sha256


def _key_fields(parameter_path: str) -> tuple[str, ...]:
    fields = get(parameter_path)
    if not isinstance(fields, list) or not all(isinstance(field, str) for field in fields):
        raise TypeError(f"params.{parameter_path} must be a list of field names")
    return tuple(fields)


def _content_id(parameter_path: str, values: Mapping[str, Any]) -> str:
    fields = _key_fields(parameter_path)
    missing = set(fields) - set(values)
    if missing:
        raise ValueError(
            f"cannot build ID from params.{parameter_path}; "
            f"missing fields: {sorted(missing)}"
        )
    return canonical_sha256({field: values[field] for field in fields})


def make_run_id(values: Mapping[str, Any]) -> str:
    """Identify one system run over every declared run-key field."""

    form = get("artifacts.run_id_form")
    if form != "content_addressed_sha256_over_sorted_key_value_pairs":
        raise NotImplementedError(f"unsupported params.artifacts.run_id_form: {form}")
    return _content_id("artifacts.run_id_key", values)


def make_analysis_cell_id(values: Mapping[str, Any]) -> str:
    """Identify one compound train-seed/channel-seed replicate."""

    return _content_id("artifacts.analysis_cell_id_key", values)


def make_noise_id(values: Mapping[str, Any]) -> str:
    """Identify one channel realisation independently of the evaluated system."""

    return _content_id("artifacts.noise_id_key", values)


def make_pair_id(values: Mapping[str, Any]) -> str:
    """Identify one system-independent ER-10 pairing row."""

    fields = set(_key_fields("artifacts.pair_id_key"))
    excluded = set(get("artifacts.pair_id_excludes"))
    overlap = fields & excluded
    if overlap:
        raise ValueError(
            "params.artifacts.pair_id_key contains excluded fields: "
            f"{sorted(overlap)}"
        )
    return _content_id("artifacts.pair_id_key", values)


def _index_trajectory(
    rows: Iterable[Mapping[str, Any]],
    arm: str,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"{arm} trajectory row has no valid pair_id")
        if pair_id in indexed:
            raise ValueError(f"{arm} trajectory duplicates pair_id {pair_id}")
        indexed[pair_id] = row
    return indexed


def join_pair_trajectories(
    left_rows: Iterable[Mapping[str, Any]],
    right_rows: Iterable[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], Mapping[str, Any]], ...]:
    """Strict one-to-one ER-10 join, rejecting missing and duplicate rows."""

    left = _index_trajectory(left_rows, "left")
    right = _index_trajectory(right_rows, "right")
    missing_from_left = set(right) - set(left)
    missing_from_right = set(left) - set(right)
    if missing_from_left or missing_from_right:
        raise ValueError(
            "pair trajectories differ: "
            f"missing_from_left={sorted(missing_from_left)}, "
            f"missing_from_right={sorted(missing_from_right)}"
        )
    return tuple((left[pair_id], right[pair_id]) for pair_id in sorted(left))
