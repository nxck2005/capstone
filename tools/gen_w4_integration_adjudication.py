#!/usr/bin/env python3
"""Generate the W4 classical-baseline integration adjudication.

Closes W4: the bounded validation/plumbing integration built across PA, PB_1
(+PB_1C), PB_2 (+PB_2C) and PB_3, together with the BR-4 selection machinery
PB_3 built and did **not** run.

Everything the adjudication asserts is derived here rather than typed: the
evidence hashes are computed from the files on disk, the selection-machinery
description is read out of `baseline.classical.composition` itself, the worked
composition example is computed by the real composition function, and the
characterised BLER identities are enumerated from the committed G-2 curves.
`tools/verify_w4_baseline_integration.py` then recomputes all of it.

Two things are deliberately *not* recorded:

* an `evidence_commit`. A file cannot contain the hash of the commit that adds
  it, so the commit is **resolved** from Git path history the way G-2's is, and
  the verifier rejects any stored value;
* `src/baseline/classical/composition.py` in the W4 *execution* source manifest.
  That manifest binds the sources that participated in the bounded measurement
  at commit `76e789c9f3d0`, where this module did not exist. It is bound here
  instead, under its own `selection_sources` role, at HEAD.

Usage:
    .venv/bin/python tools/gen_w4_integration_adjudication.py
    .venv/bin/python tools/gen_w4_integration_adjudication.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from baseline.classical import composition  # noqa: E402
from baseline.classical.outage import (  # noqa: E402
    EVIDENCE_LABELS,
    write_json_atomically,
)
from baseline.classical.records import FROZEN_CLASSIFIER_DATASET  # noqa: E402
from config.params import get  # noqa: E402
from models.frozen_reference_classifier import (  # noqa: E402
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CONFIG_HASH,
)

EVIDENCE_DIR = REPO / "results" / "baseline" / "w4"
ADJUDICATION = EVIDENCE_DIR / "integration_adjudication.json"

ADJUDICATION_SCHEMA_VERSION = 1

#: The bounded evidence this adjudication stands on, hashed from disk.
BOUND_EVIDENCE_FILES = (
    "resolved_config.json",
    "outage_policy.json",
    "smoke_summary.json",
    "accounting_examples.json",
    "execution_source_manifest.json",
    "per_image.csv",
    "aggregate.csv",
    "smoke_rows.jsonl",
    "overhead_table.json",
)

#: The PB_3 selection machinery, bound at HEAD under its own role. These files
#: did not exist at the bounded run's execution commit and must never be added
#: to `execution_source_manifest.json`, which would claim they participated in
#: a measurement they postdate.
SELECTION_SOURCES = {
    "src/baseline/classical/composition.py": "selection_implementation",
    "tests/test_classical_composition.py": "selection_test",
}

#: The three bandwidth values G-8 will decide. PB_3 records them so the
#: verifier can prove they did not move; it does not select any of them.
PROVISIONAL_OPERATING_POINTS = (
    "efficiency_ratio",
    "crossover_ratio",
    "low_ratio_operating_point",
)

PROMINENT_DECLARATION = (
    "Bounded validation/plumbing integration. This is not the BR-4 full "
    "validation sweep, not a G-8 operating-point selection, and not test "
    "evidence. W4 built the classical transport path, the frozen outage "
    "policy, the record layer, the bounded evidence and the BR-4 selection "
    "machinery; it executed only bounded, unit-scale workloads. G-8 remains "
    "unresolved, no operating ratio was chosen, no model was trained or "
    "fine-tuned, and the test split stayed sealed."
)

EVIDENCE_COMMIT_RESOLUTION = (
    "resolved, never recorded: the commit that introduced this file is "
    "`git log -1 --format=%H -- results/baseline/w4/integration_adjudication.json`. "
    "A file cannot contain the hash of the commit that adds it, so a stored "
    "evidence_commit would be either null, a guess or circular. The verifier "
    "rejects any stored value."
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def worked_composition_example() -> dict[str, Any]:
    """A composition computed by the real function, for the verifier to redo.

    Two code blocks at the committed QPSK point, the frozen outage measurement,
    and a codec accuracy stated as the counts it would be measured from. If the
    arithmetic in `composition.py` changes, the number recorded here stops
    matching what the verifier recomputes.
    """

    outage = composition.measured_outage_accuracy_from_record(
        json.loads((EVIDENCE_DIR / "outage_policy.json").read_text())
    )
    codec = composition.MeasuredCodecAccuracy(
        correct=870,
        total=1000,
        split=composition.SELECTION_SPLIT,
        source=(
            "illustrative measured-input placeholder for the worked example; "
            "the real acc_clean per cached codec configuration is a G-8 input"
        ),
    )
    identity = {
        "k_and_n": [128, 256],
        "base_graph": 2,
        "lifting_size": 22,
        "modulation": "qpsk",
        "decoder_algorithm": "offset_min_sum",
        "decoder_offset": 0.5,
        "iterations": 50,
        "snr_convention": "eb_n0_per_information_bit",
        "rate": "0.5",
    }
    snr_db = 2.5
    table = composition.g2_bler_table()
    bler = table.require(identity, snr_db)
    result = composition.compose(
        [bler, bler], codec_accuracy=codec, outage_accuracy=outage
    )
    return {
        "identity": identity,
        "snr_db": snr_db,
        "code_blocks": result.code_blocks,
        "block_bler": bler,
        "success_probability": result.success_probability,
        "codec_accuracy": {
            "correct": codec.correct,
            "total": codec.total,
            "value": codec.value,
            "split": codec.split,
        },
        "outage_accuracy": {
            "selected_class": outage.selected_class,
            "numerator": outage.numerator,
            "denominator": outage.denominator,
            "value": outage.value,
        },
        "expected_accuracy": result.expected_accuracy,
    }


def bler_characterization() -> dict[str, Any]:
    """What the committed evidence does and does not characterise."""

    table = composition.g2_bler_table()
    identities = []
    for identity in table.identities:
        curve = table._curves[identity]  # noqa: SLF001 - same package, read-only
        identities.append(
            {
                **identity.as_key(),
                "support_db": list(curve.support),
                "measured_points": len(curve.snr_db),
                "trials_per_point": curve.trials,
            }
        )
    return {
        "source": "results/baseline/g2/bler_results.csv",
        "measurement_arm": composition._MEASUREMENT_SYSTEM,  # noqa: SLF001
        "required_identity_fields": list(composition.BLER_REQUIRED_FIELDS),
        "spec_required_identity_fields": list(composition.BLER_IDENTITY_FIELDS),
        "additional_required_fields": list(
            composition.BLER_EXTRA_IDENTITY_FIELDS
        ),
        "interpolation": composition._INTERPOLATION,  # noqa: SLF001
        "extrapolation_permitted": False,
        "absent_evidence_treated_as_zero_bler": False,
        "characterized": identities,
        "uncharacterized_examples": [
            {
                "reason": "identity_not_characterized",
                "example": "base_graph 1 at the otherwise-committed QPSK identity",
            },
            {
                "reason": "identity_not_characterized",
                "example": "any (K, N) other than (128, 256)",
            },
            {
                "reason": "identity_not_characterized",
                "example": "any LDPC rate other than 1/2",
            },
            {
                "reason": "snr_outside_characterized_support",
                "example": "the committed QPSK curve at 18 dB, above its 2.75 dB span",
            },
            {
                "reason": "snr_outside_characterized_support",
                "example": "16-QAM at 2.5 dB Eb/N0, below its 4.0 dB span",
            },
        ],
    }


def selection_machinery() -> dict[str, Any]:
    return {
        "module": "src/baseline/classical/composition.py",
        "composition_formula": {
            "tb_success": "product over code blocks of (1 - BLER_r)",
            "expected_accuracy": (
                "P(TB success) * measured codec accuracy + "
                "(1 - P(TB success)) * measured outage accuracy"
            ),
            "both_inputs_measured": True,
            "assumed_uniform_outage_accuracy_rejected": True,
        },
        "feasibility_cache": {
            "key_fields": list(composition.FEASIBILITY_KEY_FIELDS),
            "excluded_fields": dict(composition.FEASIBILITY_KEY_EXCLUSIONS),
        },
        "tie_break_order": list(composition.TIE_BREAK_ORDER),
        "tie_equality": "exact float equality; no tolerance parameter",
        "system_modes": list(composition.SYSTEM_MODES),
        "uncharacterized_candidates_are": "ineligible, not low-scoring",
        "selection_passes": {
            "permitted": list(composition.selection_passes()),
            "terminates_after_pass": get(
                "reference_classifier.br4_selection_terminates_after_pass"
            ),
            "enforcement": "structural: SelectionCampaign state machine",
            "artifact_finetune_gate": get(
                "reference_classifier.artifact_finetune_gate"
            ),
            "artifact_finetuned_classifier_trained": False,
        },
        "passes_executed": 0,
    }


def sweep_guard() -> dict[str, Any]:
    budget = composition.sweep_budget(None)
    return {
        "entry_point": "select_operating_points",
        "max_candidates": budget.max_candidates,
        "max_samples": budget.max_samples,
        "max_workload": budget.max_workload,
        "authorization_type": "G8Authorization",
        "authorization_gate": composition.G8_GATE,
        "authorization_default": None,
        "environment_variable_bypass": False,
        "repository_default_authorized": False,
        "authorizations_in_repository": 0,
    }


def bounded_executions() -> list[dict[str, Any]]:
    summary = json.loads((EVIDENCE_DIR / "smoke_summary.json").read_text())
    executions = [
        {
            "phase": "PB_2C",
            "kind": "bounded classical-baseline smoke over real validation data",
            "rows": summary["raw_rows_count"],
            "wall_time_s": summary["wall_clock_s"],
            "split": "val",
            "detail": (
                "5 CIFAR-10 transport-only samples with no classifier inference "
                "and no task score, 24 Imagenette-160 validation images at 18 dB "
                "and 24 at -8 dB, plus structural- and codec-infeasibility "
                "fixtures"
            ),
        }
    ]
    executions.append(
        {
            "phase": "PB_3",
            "kind": "unit-scale exercises of the BR-4 selection machinery",
            "rows": 0,
            "wall_time_s": None,
            "split": "none: synthetic and fixture inputs only",
            "detail": (
                "no dataset was loaded, no image was encoded, no channel was "
                "run and no classifier was evaluated; the selection machinery "
                "was exercised only over synthetic candidates and the committed "
                "G-2 curves, under the bounded sweep budget"
            ),
        }
    )
    return executions


def build() -> dict[str, Any]:
    evidence_files = {
        name: sha256_bytes((EVIDENCE_DIR / name).read_bytes())
        for name in BOUND_EVIDENCE_FILES
    }
    selection_sources = []
    head = git("rev-parse", "HEAD")
    for path, role in sorted(SELECTION_SOURCES.items()):
        payload = (REPO / path).read_bytes()
        selection_sources.append(
            {
                "path": path,
                "role": role,
                "sha256": sha256_bytes(payload),
                "bytes": len(payload),
                "bound_at": "HEAD",
            }
        )
    outage = json.loads((EVIDENCE_DIR / "outage_policy.json").read_text())
    summary = json.loads((EVIDENCE_DIR / "smoke_summary.json").read_text())

    provisional = {}
    for name in PROVISIONAL_OPERATING_POINTS:
        provisional[name] = {
            "value": get(f"bandwidth.{name}"),
            "status": get(f"bandwidth.{name}_status"),
            "selected_by_w4": False,
        }

    return {
        "schema_version": ADJUDICATION_SCHEMA_VERSION,
        "gate": "W4 bounded classical-baseline integration",
        "verdict": "bounded_integration_complete",
        "prominent_declaration": PROMINENT_DECLARATION,
        "evidence_labels": list(EVIDENCE_LABELS),
        "claims": {
            "bounded_validation_plumbing_integration": True,
            "br4_full_validation_sweep": False,
            "g8_operating_point_selection": False,
            "test_evidence": False,
            "g8_status": "unresolved",
            "training_performed": False,
            "artifact_finetuning_performed": False,
            "lambda_calibration_performed": False,
            "er9_implemented": False,
            "test_split_sealed": True,
        },
        "evidence_commit_resolution": EVIDENCE_COMMIT_RESOLUTION,
        "head_at_generation": head,
        "w4_implementation_commits": {
            "pa_green": "031becc",
            "pb_1_green": "e47913c52e9117179691b70b29a289880b22dbdd",
            "pb_1c_green": "4eda158145595de0f2e9aa92456ee4a052db74b0",
            "pb_2_green_invalidated": "50de80364c2546463918387a8f335ea36107bde0",
            "pb_2c_green": "3324393a3e1692478bba8cf1020708bf52947f6d",
            "pre_b3_green": "81372a5f1139bbfa9e086d229bf807c7cf6a8bce",
            "note": (
                "the PB_3 implementation/evidence green is the commit this file "
                "is introduced in, resolved from Git path history rather than "
                "recorded here"
            ),
        },
        "bounded_evidence_execution_commit": summary["execution_source_commit"],
        "evidence_files": evidence_files,
        "selection_sources": selection_sources,
        "frozen_classifier": {
            "dataset": FROZEN_CLASSIFIER_DATASET,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "config_hash": EXPECTED_CONFIG_HASH,
            "gate": "G-1",
            "retrained_by_w4": False,
        },
        "outage": {
            "selection_policy": outage["selection_policy"],
            "selected_class": outage["selected_class"],
            "numerator": outage["numerator"],
            "denominator": outage["denominator"],
            "measured_validation_accuracy": outage["measured_validation_accuracy"],
            "assumed_uniform_accuracy_rejected": True,
            "derivation": "selected_count / validation_count over the committed "
            "validation manifest; equals 1/n_classes only because the split is "
            "exactly stratified",
        },
        "bounded_executions": bounded_executions(),
        "composition": {
            "worked_example": worked_composition_example(),
        },
        "bler_characterization": bler_characterization(),
        "selection_machinery": selection_machinery(),
        "sweep_guard": sweep_guard(),
        "provisional_operating_points": provisional,
        "test_split_access": {
            "test_split_sealed": True,
            "test_accessed": False,
            "test_inference": False,
            "test_accuracy_computed": False,
            "decoder_calls": 0,
            "canonicalization_calls": 0,
            "inference_calls": 0,
            "accuracy_calls": 0,
            "release_gate": get("evaluation.test_access_gate"),
        },
        "remaining": {
            "next_engineering_task": "G-8 classical validation work: the full "
            "BR-4 validation sweep and the operating-point decision",
            "full_br4_sweep_started": False,
            "g8_started": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate in memory and compare against the committed bytes",
    )
    arguments = parser.parse_args()

    payload = build()
    if arguments.check:
        if not ADJUDICATION.is_file():
            print(f"missing {ADJUDICATION.name}", file=sys.stderr)
            return 1
        committed = json.loads(ADJUDICATION.read_text())
        regenerated = json.loads(json.dumps(payload))
        # `head_at_generation` moves with every commit and is informational.
        committed.pop("head_at_generation", None)
        regenerated.pop("head_at_generation", None)
        if committed != regenerated:
            differing = sorted(
                key
                for key in set(committed) | set(regenerated)
                if committed.get(key) != regenerated.get(key)
            )
            print(
                f"{ADJUDICATION.name} is stale; differing keys: {differing}",
                file=sys.stderr,
            )
            return 1
        print(
            f"ok: {ADJUDICATION.name} matches its regenerated content "
            f"({len(payload['evidence_files'])} evidence files, "
            f"{len(payload['selection_sources'])} selection sources)"
        )
        return 0

    digest = write_json_atomically(ADJUDICATION, payload)
    print(
        f"wrote {ADJUDICATION.relative_to(REPO)} sha256={digest} "
        f"({len(payload['evidence_files'])} evidence files, "
        f"{len(payload['selection_sources'])} selection sources)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
