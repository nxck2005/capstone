"""Successor-only merge boundary for the future C3/C5 table phase.

The historical G8 table tooling remains bound to the predecessor campaign.
This module is the only successor input adapter: it accepts complete,
accepted Pascal records from the successor runtime and rejects every other
campaign or artifact role.  It intentionally does not construct a BlerTable
until the owner opens the later selection/table phase.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from baseline.g8_pascal_production import (
    RESULT_ARTIFACT_ROLE,
    audit_campaign,
    inspect_unit,
    successor_bindings,
    validate_runtime_namespace,
)


class SuccessorMergeError(RuntimeError):
    """Successor evidence is not an eligible isolated table input."""


def collect_successor_results(root: Path | str) -> list[dict[str, Any]]:
    """Return only accepted successor result records, in authority order."""

    try:
        summary = audit_campaign(root)
        validate_runtime_namespace(root)
    except Exception as exc:
        if isinstance(exc, SuccessorMergeError):
            raise
        raise SuccessorMergeError(f"successor evidence cannot be audited: {exc}") from exc
    bindings = successor_bindings()
    records: list[dict[str, Any]] = []
    for ordinal in summary["accepted_authority_ordinals"]:
        report = inspect_unit(root, ordinal)
        if not report["accepted"]:
            raise SuccessorMergeError("accepted coverage is not reproducible from a terminal unit state")
        complete = [item["result"] for item in report["validated_results"].values() if item["result"]["status"] == "complete"]
        if len(complete) != 1:
            raise SuccessorMergeError("accepted successor unit does not have exactly one complete result")
        result = complete[0]
        if result["artifact_role"] != RESULT_ARTIFACT_ROLE or result["identity"]["campaign_id"] != bindings["campaign_id"]:
            raise SuccessorMergeError("predecessor or foreign result reached the successor merge boundary")
        if result["identity"]["execution_profile_id"] != bindings["execution_profile_id"]:
            raise SuccessorMergeError("successor result profile binding differs")
        records.append(result)
    if len(records) != len(summary["accepted_authority_ordinals"]) or len({record["identity"]["authority_ordinal"] for record in records}) != len(records):
        raise SuccessorMergeError("successor merge input has duplicate accepted ordinals")
    if any(record["identity"]["authority_ordinal"] != ordinal for record, ordinal in zip(records, summary["accepted_authority_ordinals"], strict=True)):
        raise SuccessorMergeError("successor merge input order differs from authority order")
    return records


def build_successor_bler_table(*_args: Any, **_kwargs: Any) -> None:
    """Later-phase gate; no table is authorized by this readiness repair."""

    raise SuccessorMergeError(
        "successor BlerTable construction is gated to the owner-authorized C3/C5 phase"
    )


__all__ = ["SuccessorMergeError", "collect_successor_results", "build_successor_bler_table"]
