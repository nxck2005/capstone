"""The prospective W10 validation-rehearsal system/ratio scope (AM-98).

W10 is not a blind ``system × SNR`` Cartesian product.  Each entry below is one
explicit scientific arm with its own role, ratio, backend, binding source and
scorer contract.  The unit count is derived from this table, never hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config.params import get
from training.deterministic_core import canonical_sha256

W10_CELL = (0, 0)
W10_TRAIN_SEED = 0
W10_CHANNEL_SEED = 0
W10_SPLIT = "val"
W10_DATASET = "imagenette160"
W10_VALIDATION_DENOMINATOR = 1000  # literal-ok: frozen Imagenette-160 validation split size

W10_BACKENDS = (
    "learned",
    "er2_randomised",
    "er9_digital",
    "recon_ablation",
    "label_bound",
    "classical",
    "jpeg_secondary",
)

LEARNED_RATIOS = ("r_1_6", "r_1_24")
HEADLINE_RATIO = "r_1_6"


@dataclass(frozen=True)
class W10ScopeEntry:
    """One explicit W10 scientific arm."""

    system: str
    bw_ratio: str
    role: str
    backend: str
    checkpoint_source: str
    scorer_source: str
    selection_source: str
    classifier_variants: tuple[str, ...]
    per_image_required: bool = True
    denominator: int = W10_VALIDATION_DENOMINATOR

    def identity(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "bw_ratio": self.bw_ratio,
            "role": self.role,
            "backend": self.backend,
            "checkpoint_source": self.checkpoint_source,
            "scorer_source": self.scorer_source,
            "selection_source": self.selection_source,
            "classifier_variants": list(self.classifier_variants),
            "per_image_required": self.per_image_required,
            "denominator": self.denominator,
        }


# ``classical_finetune_scored`` is deliberately absent.  ``params.reference_classifier.headline_scorer``
# is ``artifact_finetuned``, so the headline classical arm already IS the artifact-finetuned
# scorer; the clean-scorer contrast is the ``clean`` classifier variant of the same physical run.
# Enumerating it as a second arm would run the identical channel simulation twice (AM-98).
SCOPE: tuple[W10ScopeEntry, ...] = (
    W10ScopeEntry(
        system="learned",
        bw_ratio="r_1_6",
        role="er1_headline_learned",
        backend="learned",
        checkpoint_source="w8_reconciliation:r_1_6:train0_channel0",
        scorer_source="own_task_head",
        selection_source="not_applicable_no_selection",
        classifier_variants=("own_task_head",),
    ),
    W10ScopeEntry(
        system="learned",
        bw_ratio="r_1_24",
        role="er11_efficiency_learned",
        backend="learned",
        checkpoint_source="w8_reconciliation:r_1_24:train0_channel0",
        scorer_source="own_task_head",
        selection_source="not_applicable_no_selection",
        classifier_variants=("own_task_head",),
    ),
    W10ScopeEntry(
        system="learned_snr_randomised",
        bw_ratio="r_1_6",
        role="er2_snr_randomised",
        backend="er2_randomised",
        checkpoint_source="er2_selected_checkpoint_v4",
        scorer_source="own_task_head",
        selection_source="not_applicable_no_selection",
        classifier_variants=("own_task_head",),
    ),
    W10ScopeEntry(
        system="learned_papr_constrained",
        bw_ratio="r_1_6",
        role="sr16_papr_secondary",
        backend="learned",
        checkpoint_source="pending_papr_training",
        scorer_source="own_task_head",
        selection_source="not_applicable_no_selection",
        classifier_variants=("own_task_head",),
    ),
    W10ScopeEntry(
        system="classical_adaptive",
        bw_ratio="r_1_6",
        role="er1_headline_classical",
        backend="classical",
        checkpoint_source="not_applicable_untrained_codec",
        scorer_source="br12_artifact_finetuned_reference_classifier",
        selection_source="pass_two_state:classical_adaptive:r_1_6",
        classifier_variants=("artifact_finetuned", "clean"),
    ),
    W10ScopeEntry(
        system="classical_adaptive",
        bw_ratio="r_1_24",
        role="er11_efficiency_classical",
        backend="classical",
        checkpoint_source="not_applicable_untrained_codec",
        scorer_source="br12_artifact_finetuned_reference_classifier",
        selection_source="pass_two_state:classical_adaptive:r_1_24",
        classifier_variants=("artifact_finetuned", "clean"),
    ),
    W10ScopeEntry(
        system="classical_fixed_mcs",
        bw_ratio="r_1_6",
        role="br16_fixed_mcs",
        backend="classical",
        checkpoint_source="not_applicable_untrained_codec",
        scorer_source="br12_artifact_finetuned_reference_classifier",
        selection_source="br16_fixed_configuration:r_1_6",
        classifier_variants=("artifact_finetuned", "clean"),
    ),
    W10ScopeEntry(
        system="classical_fixed_mod",
        bw_ratio="r_1_6",
        role="br9_fixed_modulation",
        backend="classical",
        checkpoint_source="not_applicable_untrained_codec",
        scorer_source="br12_artifact_finetuned_reference_classifier",
        selection_source="pass_two_state:classical_fixed_mod:r_1_6",
        classifier_variants=("artifact_finetuned", "clean"),
    ),
    W10ScopeEntry(
        system="classical_jpeg_secondary",
        bw_ratio="r_1_6",
        role="dec9_jpeg_secondary",
        backend="jpeg_secondary",
        checkpoint_source="not_applicable_untrained_codec",
        scorer_source="br12_artifact_finetuned_reference_classifier",
        selection_source="w10_jpeg_validation_selection",
        classifier_variants=("artifact_finetuned", "clean"),
    ),
    W10ScopeEntry(
        system="er9_digital",
        bw_ratio="r_1_6",
        role="er9_control_h4",
        backend="er9_digital",
        checkpoint_source="er9_production_closeout:train0_channel0",
        scorer_source="own_task_head",
        selection_source="er9_stage2_selection_v4:2048x2",
        classifier_variants=("own_task_head",),
    ),
    W10ScopeEntry(
        system="label_transmission_bound",
        bw_ratio="r_1_6",
        role="er12_label_upper_bound",
        backend="label_bound",
        checkpoint_source="er9_production_closeout:train0_channel0",
        scorer_source="transmitted_predicted_label_then_true_label_scoring_only",
        selection_source="w10_er12_validation_selection",
        classifier_variants=("predicted_label",),
    ),
    W10ScopeEntry(
        system="semantic_recon_ablation",
        bw_ratio="r_1_6",
        role="er4_reconstruction_ablation",
        backend="recon_ablation",
        checkpoint_source="w8_reconciliation:r_1_6:train0_channel0",
        scorer_source="frozen_g1_reference_classifier",
        selection_source="not_applicable_no_selection",
        classifier_variants=("clean",),
    ),
)


def snr_grid() -> tuple[int, ...]:
    return tuple(int(value) for value in get("channel.test_snr_grid_db"))


def scope_identity() -> dict[str, Any]:
    return {
        "scope_version": int(get("evaluation.w10_scope_version")),
        "cell": {"train_seed": W10_TRAIN_SEED, "channel_seed": W10_CHANNEL_SEED},
        "dataset": W10_DATASET,
        "split": W10_SPLIT,
        "denominator": W10_VALIDATION_DENOMINATOR,
        "snr_grid_db": list(snr_grid()),
        "entries": [entry.identity() for entry in SCOPE],
    }


def scope_sha256() -> str:
    return canonical_sha256(scope_identity())


def validate_scope(entries: tuple[W10ScopeEntry, ...] = SCOPE) -> None:
    allowed_systems = tuple(str(value) for value in get("artifacts.system_values"))
    legal_ratios = tuple(str(value) for value in get("bandwidth.ratios"))
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        if entry.system not in allowed_systems:
            raise ValueError(f"W10 system {entry.system!r} is not a legal system value")
        if entry.backend not in W10_BACKENDS:
            raise ValueError(f"W10 backend {entry.backend!r} is not supported")
        if entry.bw_ratio not in legal_ratios:
            raise ValueError(f"W10 ratio {entry.bw_ratio!r} is not configured")
        if entry.denominator != W10_VALIDATION_DENOMINATOR:
            raise ValueError(f"W10 denominator for {entry.system} differs")
        if not entry.per_image_required:
            raise ValueError("every scientific W10 unit requires per-image evidence")
        key = (entry.system, entry.bw_ratio)
        if key in seen:
            raise ValueError(f"duplicate W10 scope entry: {key}")
        seen.add(key)
    learned_ratios = {entry.bw_ratio for entry in entries if entry.system == "learned"}
    if learned_ratios != set(LEARNED_RATIOS):
        raise ValueError(
            "ER-11 requires learned evaluation at exactly the trained ratios "
            f"{sorted(LEARNED_RATIOS)}, found {sorted(learned_ratios)}"
        )
    if ("classical_adaptive", "r_1_24") not in seen:
        raise ValueError("ER-11 requires the classical companion at r_1_24")
    if "classical_finetune_scored" in {entry.system for entry in entries}:
        raise ValueError(
            "classical_finetune_scored must not be a duplicate W10 arm (AM-98)"
        )


def scope_evidence_requirement(entry: W10ScopeEntry) -> dict[str, Any]:
    """The primary scorer variant and denominator a scope entry requires."""

    return {
        "system": entry.system,
        "bw_ratio": entry.bw_ratio,
        "role": entry.role,
        "backend": entry.backend,
        "primary_classifier_variant": entry.classifier_variants[0],
        "classifier_variants": list(entry.classifier_variants),
        "denominator": entry.denominator,
        "per_image_required": entry.per_image_required,
    }


def entry_for(system: str, bw_ratio: str) -> W10ScopeEntry:
    for entry in SCOPE:
        if entry.system == system and entry.bw_ratio == bw_ratio:
            return entry
    raise KeyError(f"no W10 scope entry for ({system!r}, {bw_ratio!r})")


def work_units() -> tuple[dict[str, Any], ...]:
    """Exact deterministic unit enumeration derived from the scope table."""

    validate_scope()
    grid = snr_grid()
    units: list[dict[str, Any]] = []
    ordinal = 0
    for entry in SCOPE:
        for snr_db in grid:
            units.append({
                "ordinal": ordinal,
                "system": entry.system,
                "bw_ratio": entry.bw_ratio,
                "role": entry.role,
                "backend": entry.backend,
                "snr_db": snr_db,
                "train_seed": W10_TRAIN_SEED,
                "channel_seed": W10_CHANNEL_SEED,
                "split": W10_SPLIT,
            })
            ordinal += 1
    return tuple(units)


def unit_count() -> int:
    return len(work_units())


__all__ = [
    "HEADLINE_RATIO",
    "LEARNED_RATIOS",
    "SCOPE",
    "W10_BACKENDS",
    "W10_CELL",
    "W10_CHANNEL_SEED",
    "W10_DATASET",
    "W10_SPLIT",
    "W10_TRAIN_SEED",
    "W10_VALIDATION_DENOMINATOR",
    "W10ScopeEntry",
    "entry_for",
    "scope_evidence_requirement",
    "scope_identity",
    "scope_sha256",
    "snr_grid",
    "unit_count",
    "validate_scope",
    "work_units",
]
