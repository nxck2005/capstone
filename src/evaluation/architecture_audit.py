"""Pragmatic computed architecture-difference audit for ER-9/G-11."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from training.deterministic_core import canonical_sha256


class ArchitectureAuditError(ValueError):
    """Architecture comparison input is incomplete or inconsistent."""


SHARED_FIELDS = (
    "encoder_arch",
    "encoder_trunk",
    "preprocessing",
    "task_head_arch",
    "train_split",
    "augmentation",
    "optimizer",
    "epochs",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArchitectureAuditError(message)


def compare_architectures(
    learned: Mapping[str, Any],
    er9: Mapping[str, Any],
    *,
    declared_difference: str = "channel_interface",
    allowed_er9_only: tuple[str, ...] = ("channel_interface",),
) -> dict[str, Any]:
    """Derive the difference status from field comparison, never a declaration."""

    _require(isinstance(learned, Mapping) and isinstance(er9, Mapping), "architecture contracts must be mappings")
    shared_comparison = {
        field: {"learned": learned.get(field), "er9": er9.get(field), "equal": learned.get(field) == er9.get(field)}
        for field in SHARED_FIELDS
    }
    unexpected_shared = [field for field, result in shared_comparison.items() if not result["equal"]]
    learned_interface = learned.get("interface", {})
    er9_interface = er9.get("interface", {})
    _require(isinstance(learned_interface, Mapping) and isinstance(er9_interface, Mapping), "interface contracts must be mappings")
    observed_interface_differences = sorted(
        key for key in set(learned_interface) | set(er9_interface) if learned_interface.get(key) != er9_interface.get(key)
    )
    allowed = set(allowed_er9_only)
    unexpected_interface = [path for path in observed_interface_differences if path not in allowed]
    computed_only_declared = not unexpected_shared and not unexpected_interface and bool(observed_interface_differences)
    result = {
        "schema_version": 2,
        "artifact_role": "ER9_ARCHITECTURE_DIFFERENCE_AUDIT",
        "declared_difference": declared_difference,
        "shared_component_comparison": shared_comparison,
        "observed_shared_differences": unexpected_shared,
        "observed_interface_differences": observed_interface_differences,
        "unexpected_difference_paths": sorted(set(unexpected_shared) | set(unexpected_interface)),
        "allowed_er9_only_components": sorted(allowed),
        "only_declared_difference": computed_only_declared,
        "comparison_method": "fieldwise_shared-contract-and-interface-comparison",
        "test_access": 0,
    }
    result["audit_id"] = "er9archdiffv2-" + canonical_sha256(result)
    return result


__all__ = ["ArchitectureAuditError", "SHARED_FIELDS", "compare_architectures"]
