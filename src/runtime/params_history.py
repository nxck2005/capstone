"""Exact historical parameter-snapshot projections for closed verifiers.

``config_hash`` covers the complete parameter snapshot, so adding a parameter
changes every run/config fingerprint.  Closed authorities keep verifying with
the projection that removes only the parameters named by a later amendment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from config.run_config import RunConfig
from training.deterministic_core import canonical_sha256

AM98_ADDED_PARAMETER_PATHS: tuple[str, ...] = (
    "evaluation.w10_scope_version",
    "evaluation.w10_validation_denominator",
    "evaluation.w10_learned_ratios",
    "evaluation.w10_per_image_required",
    "evaluation.w10_analysis_version_policy",
    "evaluation.w10_papr_protocol_version",
    "evaluation.w10_papr_cap_db",
    "evaluation.w10_papr_training_runs",
    "evaluation.w10_er12_protocol_version",
    "evaluation.w10_er12_label_bits",
    "evaluation.w10_er12_payload_frame",
    "evaluation.w10_er12_declared_role",
)


def strip_paths(value: Any, paths: Sequence[str]) -> Any:
    """Return a copy of ``value`` with each dotted leaf path removed."""

    if not paths:
        return value
    return _strip(value, {tuple(path.split(".")) for path in paths})


def _strip(node: Any, paths: set[tuple[str, ...]]) -> Any:
    if not isinstance(node, Mapping):
        return node
    result: dict[str, Any] = {}
    for key, child in node.items():
        if (key,) in paths:
            continue
        descendants = {path[1:] for path in paths if len(path) > 1 and path[0] == key}
        result[key] = _strip(child, descendants) if descendants and isinstance(child, Mapping) else child
    return result


def projected_config_hash(config: RunConfig, *, removed_paths: Sequence[str]) -> str:
    """Recompute the run/config fingerprint under one historical projection."""

    return canonical_sha256(
        {
            "fingerprint_schema_version": config.fingerprint_schema_version,
            "resolved": config.resolved.to_dict(),
            "parameters": strip_paths(config.parameters.to_dict(), removed_paths),
        }
    )


__all__ = ["AM98_ADDED_PARAMETER_PATHS", "projected_config_hash", "strip_paths"]
