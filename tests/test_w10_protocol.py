"""Synthetic-only checks for the prospectively frozen W10 protocol (AM-98)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from evaluation import w10_rehearsal as rehearsal
from evaluation.downstream_v4 import DownstreamHold
from evaluation.w10_evidence import per_image_relative_path, recompute_n_correct, run_identity, sr18_row
from evaluation.w10_scope import (
    LEARNED_RATIOS,
    SCOPE,
    scope_sha256,
    snr_grid,
    unit_count,
    validate_scope,
    work_units,
)
from training.deterministic_core import canonical_bytes


def _primary_path(expected: dict) -> str:
    return per_image_relative_path(expected, 0)


def _synthetic_backend(expected: dict, *, n_correct: int = 500, n_total: int = 1000) -> dict:
    identity = run_identity(
        system=expected["system"],
        bw_ratio=expected["bw_ratio"],
        snr_db=expected["snr_db"],
        config_hash="a" * 64,
        checkpoint_id="b" * 64,
        classifier_variant="own_task_head",
        ldpc_rate="1/2",
        modulation="qpsk",
        quantiser_bits=None,
        transmit_dim=None,
        reconstruction_weight=None,
    )
    rows = []
    for index in range(n_total):
        correct = index < n_correct
        rows.append(
            sr18_row(
                identity=identity,
                stable_sample_id=f"val-{index:04d}",
                true_label=index % 10,
                pred_label=(index % 10) if correct else (index % 10) + 1,
                correct=correct,
                outage=False,
                outage_reason=None,
                source_bytes=None,
            )
        )
    return {
        "n_correct": recompute_n_correct(rows),
        "n_total": n_total,
        "binding": {"unit": expected["role"], "synthetic": True},
        "per_image": rows,
        "primary_classifier_variant": "own_task_head",
    }


def _authority() -> dict:
    return {
        "authority_id": "w10rehearsalauth-synthetic",
        "authority_kind": "W10_VALIDATION_REHEARSAL_AUTHORITY",
        "cell": {"train_seed": 0, "channel_seed": 0},
        "split": "val",
        "systems": list(rehearsal.W10_SYSTEMS),
        "unit_count": unit_count(),
        "scope_sha256": scope_sha256(),
        "source_manifest": {"manifest_id": "w10downstreamsource-synthetic", "path": "x", "sha256": "e" * 64},
        "source_commit": "f" * 40,
        "validation_only": True,
        "test_authorized": False,
        "test_access": 0,
    }


def test_w10_scope_is_explicit_ratio_aware_and_not_a_cartesian_system_product() -> None:
    validate_scope()
    units = work_units()
    assert len(units) == len(SCOPE) * len(snr_grid())
    assert unit_count() == 252  # literal-ok: AM-98 derived 12 arms x 21 SNRs
    assert [unit["ordinal"] for unit in units] == list(range(unit_count()))
    pairs = [(entry.system, entry.bw_ratio) for entry in SCOPE]
    assert len(set(pairs)) == len(pairs)
    learned_ratios = {entry.bw_ratio for entry in SCOPE if entry.system == "learned"}
    assert learned_ratios == set(LEARNED_RATIOS)
    assert ("classical_adaptive", "r_1_24") in pairs
    assert "classical_finetune_scored" not in {entry.system for entry in SCOPE}
    for unit in units:
        assert unit["train_seed"] == 0 and unit["channel_seed"] == 0
        assert unit["split"] == "val"
        assert unit["system"] != "classical_finetune_scored"


def test_w10_scope_digest_changes_with_scope_semantics() -> None:
    before = scope_sha256()
    assert before == scope_sha256()
    assert len(before) == 64


def test_w10_unit_identity_binds_ratio_and_binding() -> None:
    expected = work_units()[0]
    evidence = _synthetic_backend(expected)
    primary = {
        "classifier_variant": "own_task_head",
        "path": _primary_path(expected),
        "sha256": "c" * 64,
        "n_correct": evidence["n_correct"],
        "n_total": 1000,
        "_sha256": "c" * 64,
    }
    body = rehearsal.unit_body(
        expected=expected,
        evidence=evidence,
        per_image=primary,
        per_image_path=_primary_path(expected),
        scorer_variants=[{k: v for k, v in primary.items() if not k.startswith("_")}],
    )
    other_ratio = dict(expected)
    other_ratio["bw_ratio"] = "r_1_24"
    other = rehearsal.unit_body(
        expected=other_ratio,
        evidence=evidence,
        per_image=primary,
        per_image_path=_primary_path(expected),
        scorer_variants=[{k: v for k, v in primary.items() if not k.startswith("_")}],
    )
    assert body["unit_id"] != other["unit_id"]
    changed_binding = dict(evidence)
    changed_binding["binding"] = {"unit": expected["role"], "synthetic": True, "checkpoint": "d" * 64}
    third = rehearsal.unit_body(
        expected=expected,
        evidence=changed_binding,
        per_image=primary,
        per_image_path=_primary_path(expected),
        scorer_variants=[{k: v for k, v in primary.items() if not k.startswith("_")}],
    )
    assert third["unit_id"] != body["unit_id"]


def test_w10_synthetic_execute_and_closeout_bind_ordered_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = work_units()[0]
    monkeypatch.setattr(rehearsal, "work_units", lambda: (expected,))
    monkeypatch.setattr(rehearsal, "unit_count", lambda: 1)
    authority = _authority()
    authority["unit_count"] = 1
    stable_ids = [f"val-{index:04d}" for index in range(1000)]
    results = rehearsal.execute(
        tmp_path,
        authority=authority,
        backend=_synthetic_backend,
        expected_stable_ids=stable_ids,
    )
    assert len(results) == 1
    unit = results[0]
    assert unit["n_total"] == 1000 and unit["n_correct"] == 500
    assert unit["split"] == "val" and unit["test_access"] == 0
    closeout = rehearsal.closeout(tmp_path, results, authority=authority)
    assert closeout["unit_count"] == 1
    assert closeout["ordered_unit_ids_digest"] == hashlib.sha256(
        canonical_bytes({"unit_ids": [unit["unit_id"]]})
    ).hexdigest()
    assert closeout["test"] == "SEALED" and closeout["test_access"] == 0


def test_w10_rejects_wrong_denominator_and_test_authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = work_units()[0]
    monkeypatch.setattr(rehearsal, "work_units", lambda: (expected,))
    monkeypatch.setattr(rehearsal, "unit_count", lambda: 1)
    authority = _authority()
    authority["unit_count"] = 1
    with pytest.raises(Exception):
        rehearsal.execute(
            tmp_path,
            authority=authority,
            backend=lambda unit: _synthetic_backend(unit, n_total=999),
            expected_stable_ids=[f"val-{index:04d}" for index in range(999)],
        )
    authority["test_authorized"] = True
    with pytest.raises(DownstreamHold):
        rehearsal.execute(tmp_path, authority=authority, backend=_synthetic_backend)


def test_w10_unit_validation_rejects_test_access_and_unknown_role() -> None:
    expected = work_units()[0]
    evidence = _synthetic_backend(expected)
    primary = {
        "classifier_variant": "own_task_head",
        "path": _primary_path(expected),
        "sha256": "c" * 64,
        "n_correct": evidence["n_correct"],
        "n_total": 1000,
        "_sha256": "c" * 64,
    }
    body = rehearsal.unit_body(
        expected=expected,
        evidence=evidence,
        per_image=primary,
        per_image_path=_primary_path(expected),
        scorer_variants=[{k: v for k, v in primary.items() if not k.startswith("_")}],
    )
    rehearsal.validate_unit(body, expected)
    tampered = dict(body)
    tampered["test_access"] = 1
    with pytest.raises(DownstreamHold):
        rehearsal.validate_unit(tampered, expected)
    tampered = dict(body)
    tampered["bw_ratio"] = "r_1_24"
    with pytest.raises(DownstreamHold):
        rehearsal.validate_unit(tampered, expected)
