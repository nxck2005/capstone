"""The G-12 single test campaign's explicit scope (SR-22, ER-1, ER-6, AM-100).

Every arm that reads the test split is listed here, once, with its seed cells.
The ER-1 systems and H2's fixed-MCS reference run in all three zipped cells so
§2's seed-conditional estimand can be formed for H1–H4; every other arm runs in
the first cell.  All arms use the full test split: ER-6 permits a subset for
sweeps, but the full split costs little once the classical transport is batched
and removes a subset-selection rule.  The scope is frozen into the G-12 freeze
manifest by digest before any test sample is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config.params import get
from training.deterministic_core import canonical_sha256

G12_SCOPE_VERSION = 1
G12_SPLIT = "test"
G12_DATASET = "imagenette160"
G12_TEST_SUBSET = False
ALL_CELLS = ((0, 0), (1, 1), (2, 2))
FIRST_CELL = ((0, 0),)


@dataclass(frozen=True)
class G12Arm:
    """One scientific arm of the single test campaign."""

    system: str
    bw_ratio: str
    role: str
    backend: str
    w10_role: str | None
    cells: tuple[tuple[int, int], ...]
    classifier_variants: tuple[str, ...]
    hypotheses: tuple[str, ...] = ()

    def identity(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "bw_ratio": self.bw_ratio,
            "role": self.role,
            "backend": self.backend,
            "w10_role": self.w10_role,
            "cells": [list(cell) for cell in self.cells],
            "classifier_variants": list(self.classifier_variants),
            "hypotheses": list(self.hypotheses),
        }


# ``w10_role`` names the frozen W10 binding that supplies this arm's selection
# and scorer; checkpoints for cells beyond (0, 0) are resolved per cell.
SCOPE: tuple[G12Arm, ...] = (
    G12Arm("learned", "r_1_6", "er1_headline_learned", "learned", "er1_headline_learned", ALL_CELLS, ("own_task_head",), ("H1", "H2", "H3", "H4")),
    G12Arm("classical_adaptive", "r_1_6", "er1_headline_classical", "classical", "er1_headline_classical", ALL_CELLS, ("artifact_finetuned", "clean"), ("H1", "H3")),
    G12Arm("er9_digital", "r_1_6", "er9_control_h4", "er9_digital", "er9_control_h4", ALL_CELLS, ("own_task_head",), ("H4",)),
    G12Arm("classical_fixed_mcs", "r_1_6", "br16_fixed_mcs_h2", "classical", "br16_fixed_mcs", ALL_CELLS, ("artifact_finetuned", "clean"), ("H2",)),
    G12Arm("learned", "r_1_24", "er11_efficiency_learned", "learned", "er11_efficiency_learned", FIRST_CELL, ("own_task_head",)),
    G12Arm("classical_adaptive", "r_1_24", "er11_efficiency_classical", "classical", "er11_efficiency_classical", FIRST_CELL, ("artifact_finetuned", "clean")),
    G12Arm("learned_snr_randomised", "r_1_6", "er2_snr_randomised", "er2_randomised", "er2_snr_randomised", FIRST_CELL, ("own_task_head",)),
    G12Arm("learned_papr_constrained", "r_1_6", "sr16_papr_secondary", "learned", "sr16_papr_secondary", FIRST_CELL, ("own_task_head",)),
    G12Arm("classical_fixed_mod", "r_1_6", "br9_fixed_modulation", "classical", "br9_fixed_modulation", FIRST_CELL, ("artifact_finetuned", "clean")),
    G12Arm("classical_jpeg_secondary", "r_1_6", "dec9_jpeg_secondary", "jpeg_secondary", "dec9_jpeg_secondary", FIRST_CELL, ("artifact_finetuned", "clean")),
    G12Arm("label_transmission_bound", "r_1_6", "er12_label_upper_bound", "label_bound", "er12_label_upper_bound", FIRST_CELL, ("predicted_label",)),
    G12Arm("semantic_recon_ablation", "r_1_6", "er4_reconstruction_ablation", "recon_ablation", "er4_reconstruction_ablation", FIRST_CELL, ("clean",)),
    G12Arm("er9_digital_low_rate", "r_1_6", "am100_low_rate_secondary", "er9_low_rate", None, FIRST_CELL, ("own_task_head",)),
)


def snr_grid() -> tuple[int, ...]:
    return tuple(int(value) for value in get("channel.test_snr_grid_db"))


def headline_ratio() -> str:
    """``params.bandwidth.headline_ratio`` names another bandwidth key; resolve it."""

    value = str(get("bandwidth.headline_ratio"))
    return str(get(f"bandwidth.{value}")) if not value.startswith("r_") else value


def test_denominator() -> int:
    return int(get(f"datasets.{G12_DATASET}.test_images"))


def validate_scope(arms: tuple[G12Arm, ...] = SCOPE) -> None:
    allowed = set(str(value) for value in get("artifacts.system_values"))
    seen: set[tuple[str, str]] = set()
    for arm in arms:
        if arm.system not in allowed:
            raise ValueError(f"G-12 system {arm.system!r} is not a legal system value")
        if (arm.system, arm.bw_ratio) in seen:
            raise ValueError(f"duplicate G-12 arm {(arm.system, arm.bw_ratio)}")
        seen.add((arm.system, arm.bw_ratio))
        if any(cell not in ALL_CELLS for cell in arm.cells):
            raise ValueError(f"{arm.role} names a cell outside the zipped seed pairing")
    full = {arm.system for arm in arms if arm.bw_ratio == headline_ratio() and arm.cells == ALL_CELLS}
    required = {"learned", "classical_adaptive", get("evaluation.h4_comparator"), get("evaluation.cliff_reference_system")}
    if not required <= full:
        raise ValueError(f"ER-1/H2/H4 arms must run in all three cells: missing {sorted(required - full)}")
    covered = {arm.role for arm in arms}
    for role in ("er2_snr_randomised", "er4_reconstruction_ablation", "er11_efficiency_learned", "er11_efficiency_classical", "er12_label_upper_bound"):
        if role not in covered:
            raise ValueError(f"params.evaluation.test_campaign_covers requires {role}")


def work_units() -> tuple[dict[str, Any], ...]:
    validate_scope()
    units: list[dict[str, Any]] = []
    for arm in SCOPE:
        for train_seed, channel_seed in arm.cells:
            for snr_db in snr_grid():
                units.append({
                    "ordinal": len(units),
                    "system": arm.system,
                    "bw_ratio": arm.bw_ratio,
                    "role": arm.role,
                    "backend": arm.backend,
                    "snr_db": snr_db,
                    "train_seed": train_seed,
                    "channel_seed": channel_seed,
                    "split": G12_SPLIT,
                })
    return tuple(units)


def arm_for(role: str) -> G12Arm:
    for arm in SCOPE:
        if arm.role == role:
            return arm
    raise KeyError(role)


def unit_stem(unit: dict[str, Any]) -> str:
    return (
        f"{int(unit['ordinal']):03d}-{unit['system']}-{unit['bw_ratio']}"
        f"-c{int(unit['train_seed'])}-snr{int(unit['snr_db']):+03d}"
    )


def scope_identity() -> dict[str, Any]:
    return {
        "scope_version": G12_SCOPE_VERSION,
        "dataset": G12_DATASET,
        "split": G12_SPLIT,
        "test_subset": G12_TEST_SUBSET,
        "denominator": test_denominator(),
        "snr_grid_db": list(snr_grid()),
        "arms": [arm.identity() for arm in SCOPE],
        "unit_count": len(work_units()),
    }


def scope_sha256() -> str:
    return canonical_sha256(scope_identity())


__all__ = [
    "ALL_CELLS",
    "FIRST_CELL",
    "G12Arm",
    "G12_DATASET",
    "G12_SCOPE_VERSION",
    "G12_SPLIT",
    "G12_TEST_SUBSET",
    "SCOPE",
    "arm_for",
    "headline_ratio",
    "scope_identity",
    "scope_sha256",
    "snr_grid",
    "test_denominator",
    "unit_stem",
    "validate_scope",
    "work_units",
]
