"""W10 prospective-selection contracts and mutation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

import evaluation.w10_selections as selections
from evaluation.w10_selections import (
    ER12_SELECTION_PREFIX,
    JPEG_SELECTION_PREFIX,
    W10SelectionHold,
    build_selection_artifact,
    er12_rank_key,
    jpeg_candidate_space,
    jpeg_phy_shortlist,
    jpeg_rank_key,
    rank_candidates,
    score_vector_digest,
    verify_selection_artifact,
)

REPO = Path(__file__).resolve().parents[1]
SYNTHETIC_MANIFEST = {
    "manifest_kind": "W10_PREPARATORY_SOURCE_SUCCESSOR_V2",
    "manifest_id": "w10downstreamsourcev2-synthetic",
    "source_commit": "a" * 40,
}
SYNTHETIC_SOURCE_RECORD = {
    "path": "results/learned/w10/w10_downstream_source_manifest_v2.json",
    "manifest_id": SYNTHETIC_MANIFEST["manifest_id"],
    "sha256": "b" * 64,
}
GRID = list(selections.snr_grid())
JPEG_CANDIDATES = {
    int(snr): [
        {
            "snr_db": int(snr),
            "modulation": "qpsk",
            "ldpc_rate": "1/2",
            "encode_axis_px": 32,
            "quality": quality,
        }
        for quality in (10, 50, 90)
    ]
    for snr in GRID
}
ER12_CANDIDATES = {
    int(snr): [
        {"snr_db": int(snr), "modulation": "bpsk", "ldpc_rate": "1/3"},
        {"snr_db": int(snr), "modulation": "qpsk", "ldpc_rate": "1/2"},
    ]
    for snr in GRID
}


@pytest.fixture(autouse=True)
def _synthetic_source(monkeypatch):
    monkeypatch.setattr(selections, "load_w10_manifest", lambda root, live=True: SYNTHETIC_MANIFEST)
    monkeypatch.setattr(selections, "source_record", lambda root, manifest, path=None: SYNTHETIC_SOURCE_RECORD)


@pytest.fixture
def _synthetic_candidate_sets(monkeypatch):
    def expected(kind: str, root: Path) -> dict[str, set[str]]:
        source = JPEG_CANDIDATES if kind == "jpeg_secondary" else ER12_CANDIDATES
        return {
            str(snr): {selections._candidate_id(entry) for entry in entries}
            for snr, entries in source.items()
        }

    monkeypatch.setattr(selections, "_expected_candidate_set", expected)
    monkeypatch.setattr(selections, "_verify_jpeg_analytic_candidate", lambda root, entry: None)
    monkeypatch.setattr(selections, "_verify_er12_candidate_evidence", lambda root, entry: None)


def _candidate_evidence(entry: dict, *, kind: str) -> dict:
    body = {
        "schema_version": 1,
        "snr_db": entry["snr_db"],
        "modulation": entry["modulation"],
        "ldpc_rate": entry["ldpc_rate"],
        "method": (
            "jpeg_clean_codec_measurement"
            if kind == "jpeg_secondary"
            else "er12_per_image_channel_simulation"
        ),
    }
    if kind == "jpeg_secondary":
        body.update({"encode_axis_px": entry["encode_axis_px"], "quality": entry["quality"]})
    return selections.bind_candidate_evidence(body)


def _score_entry(entry: dict, *, kind: str, index: int) -> dict:
    n_correct = 500 + index
    n_delivered = 900
    value = {
        **entry,
        "n_correct": n_correct,
        "n_delivered": n_delivered,
        "n_decode_failure": 0,
        "n_infeasible": 100,
        "n_total": 1000,
        "per_image_correct_digest": score_vector_digest(
            [True] * n_correct + [False] * (1000 - n_correct)
        ),
        "status": "eligible",
        "candidate_evidence": _candidate_evidence(entry, kind=kind),
    }
    if kind == "jpeg_secondary":
        value.update(
            {
                "expected_accuracy": n_correct / 1000,
                "success_probability": 0.9,
                "analytic_evidence": {
                    "schema_version": 1,
                    "method": "br4_analytic_composition",
                    "synthetic": True,
                },
            }
        )
    return value


def _build(kind: str) -> dict:
    source = JPEG_CANDIDATES if kind == "jpeg_secondary" else ER12_CANDIDATES
    score_table = []
    winners = []
    for snr in GRID:
        entries = [
            _score_entry(entry, kind=kind, index=index)
            for index, entry in enumerate(source[int(snr)])
        ]
        score_table.append({"snr_db": int(snr), "candidates": entries})
        winners.append(rank_candidates(kind, entries))
    return build_selection_artifact(kind, selections=winners, candidate_scores=score_table)


def _resign(value: dict, *, kind: str) -> dict:
    body = dict(value)
    prefix = JPEG_SELECTION_PREFIX if kind == "jpeg_secondary" else ER12_SELECTION_PREFIX
    body["selection_id"] = prefix + selections.canonical_sha256(
        {key: item for key, item in body.items() if key != "selection_id"}
    )
    return body


def test_jpeg_candidate_space_is_the_complete_configured_cartesian_product() -> None:
    space = jpeg_candidate_space(REPO)
    expected_qualities = set(selections.jpeg_selection_contract().candidate_qualities)
    expected_axes = set(selections.jpeg_selection_contract().candidate_encode_axes)
    expected_modulations = set(selections.jpeg_selection_contract().candidate_modulations)
    expected_rates = set(selections.jpeg_selection_contract().candidate_ldpc_rates)

    assert set(space) == {str(int(snr)) for snr in GRID}
    for snr in GRID:
        candidates = space[str(int(snr))]
        assert len(candidates) == 912
        assert {item["quality"] for item in candidates} == expected_qualities
        assert {item["encode_axis_px"] for item in candidates} == expected_axes
        assert {item["modulation"] for item in candidates} == expected_modulations
        assert {item["ldpc_rate"] for item in candidates} == expected_rates
        assert len({selections._candidate_id(item) for item in candidates}) == len(candidates)


def test_jpeg_candidate_space_mutations_change_the_frozen_set() -> None:
    expected = {
        selections._candidate_id(item)
        for item in jpeg_candidate_space(REPO)[str(int(GRID[0]))]
    }
    candidates = list(jpeg_candidate_space(REPO)[str(int(GRID[0]))])
    for field in ("quality", "modulation", "ldpc_rate", "encode_axis_px"):
        reduced = [item for item in candidates if item[field] != candidates[0][field]]
        assert {selections._candidate_id(item) for item in reduced} != expected
    unauthorized = [*candidates, {**candidates[0], "quality": 1}]
    assert {selections._candidate_id(item) for item in unauthorized} != expected


def test_jpeg_phy_shortcut_is_rejected() -> None:
    with pytest.raises(W10SelectionHold, match="JPEG2000 pass-two PHY shortlist"):
        jpeg_phy_shortlist(REPO)


def test_selection_contract_proves_analytic_jpeg_selection() -> None:
    contract = selections.jpeg_selection_contract()
    assert contract.candidate_space == (
        "jpeg_quality_x_configured_encode_axis_x_packet_feasible_modulation_x_ldpc_rate"
    )
    assert contract.selection_method == "br4_analytic_composition_no_per_candidate_channel_simulation"
    assert contract.noise_convention == (
        "none_for_br4_analytic_selection_actual_channel_only_after_freeze"
    )
    assert "expected_accuracy_descending" in contract.tie_break


def test_jpeg_selection_generator_has_no_noisy_channel_simulation() -> None:
    source = (REPO / "tools/gen_w10_jpeg_validation_selection.py").read_text()
    assert "run_jpeg_pipeline" not in source
    assert "scheduled_noise_id" not in source
    assert "ER9Transport" not in source


def test_rank_keys_are_total_and_tie_ordered() -> None:
    a = {
        "snr_db": 0,
        "modulation": "qpsk",
        "ldpc_rate": "1/2",
        "encode_axis_px": 32,
        "quality": 50,
        "expected_accuracy": 0.5,
        "success_probability": 0.9,
    }
    b = dict(a, quality=10)
    assert jpeg_rank_key(a) < jpeg_rank_key(b)
    c = {"snr_db": 0, "modulation": "bpsk", "ldpc_rate": "1/2", "n_correct": 10, "n_delivered": 5}
    d = dict(c, modulation="qpsk")
    assert er12_rank_key(c) < er12_rank_key(d)
    assert jpeg_rank_key(a) != jpeg_rank_key(dict(a, quality=51))


@pytest.mark.parametrize("kind", ["jpeg_secondary", "er12_label_bound"])
def test_verified_artifacts_recompute_and_reject_grid_and_score_mutations(
    kind: str, _synthetic_candidate_sets
) -> None:
    artifact = _build(kind)
    path = REPO / (
        "results/learned/w10/jpeg_validation_selection.json"
        if kind == "jpeg_secondary"
        else "results/learned/w10/er12_validation_selection.json"
    )
    verify_selection_artifact(REPO, kind, artifact, path=path)

    with pytest.raises(W10SelectionHold, match="selection ID"):
        verify_selection_artifact(REPO, kind, {**artifact, "selection_id": "invalid"}, path=path)

    dropped = [dict(item) for item in artifact["candidate_scores"]]
    dropped[0] = {"snr_db": dropped[0]["snr_db"], "candidates": dropped[0]["candidates"][:-1]}
    with pytest.raises(W10SelectionHold):
        verify_selection_artifact(
            REPO, kind, _resign({**artifact, "candidate_scores": dropped}, kind=kind), path=path
        )

    added_entry = dict(artifact["candidate_scores"][0]["candidates"][0])
    if kind == "jpeg_secondary":
        added_entry["quality"] = 7
    else:
        added_entry["modulation"] = "qam16"
    added = [dict(item) for item in artifact["candidate_scores"]]
    added[0] = {
        "snr_db": added[0]["snr_db"],
        "candidates": [*added[0]["candidates"], added_entry],
    }
    with pytest.raises(W10SelectionHold):
        verify_selection_artifact(
            REPO, kind, _resign({**artifact, "candidate_scores": added}, kind=kind), path=path
        )

    scores = [dict(item) for item in artifact["candidate_scores"]]
    changed = [dict(entry) for entry in scores[0]["candidates"]]
    changed[0] = {**changed[0], "n_correct": 1000}
    if kind == "jpeg_secondary":
        changed[0]["expected_accuracy"] = 0.99
    scores[0] = {"snr_db": scores[0]["snr_db"], "candidates": changed}
    with pytest.raises(W10SelectionHold, match="recomputed frozen winner"):
        verify_selection_artifact(
            REPO, kind, _resign({**artifact, "candidate_scores": scores}, kind=kind), path=path
        )


def test_jpeg_winner_recomputation_does_not_trust_supplied_score(_synthetic_candidate_sets) -> None:
    artifact = _build("jpeg_secondary")
    point = artifact["candidate_scores"][0]
    original_winner = rank_candidates("jpeg_secondary", point["candidates"])
    loser = dict(point["candidates"][0], expected_accuracy=0.99)
    changed = [
        {"snr_db": point["snr_db"], "candidates": [loser, *point["candidates"][1:]]},
        *artifact["candidate_scores"][1:],
    ]
    assert original_winner["candidate_id"] != rank_candidates("jpeg_secondary", changed[0]["candidates"])["candidate_id"]
    with pytest.raises(W10SelectionHold, match="recomputed frozen winner"):
        verify_selection_artifact(
            REPO,
            "jpeg_secondary",
            _resign({**artifact, "candidate_scores": changed}, kind="jpeg_secondary"),
            path=REPO / "results/learned/w10/jpeg_validation_selection.json",
        )


def test_er12_contract_keeps_full_feasible_phy_set_and_protocol() -> None:
    contract = selections.er12_selection_contract()
    assert contract.candidate_space == "configured_packet_feasible_modulation_x_ldpc_rate"
    assert contract.candidate_encode_axes == ()
    assert contract.noise_convention == selections.NOISE_CONVENTION
    assert int(contract.identity()["er12_protocol_version"]) == 1


def _er12_evidence_entry() -> dict:
    correct_digest = selections.score_vector_digest([True] * 100 + [False] * 900)
    outcome_counts = {"delivered": 900, "decode_failure": 100, "infeasible": 0}
    worker = {
        "artifact_role": "W10_ER12_CANDIDATE_WORKER_EVIDENCE",
        "source_epoch": dict(SYNTHETIC_SOURCE_RECORD),
        "execution_profile_id": "confessor_pascal_cu126",
        "checkpoint_id": "c" * 64,
        "selection_contract_sha256": selections._contract_sha256("er12_label_bound"),
        "snr_db": 0,
        "modulation": "bpsk",
        "ldpc_rate": "1/3",
        "correct": 100,
        "total": 1000,
        "correct_digest": correct_digest,
        "outcome_counts": outcome_counts,
        "noise_ids_digest": "d" * 64,
        "test_access": 0,
    }
    worker["worker_evidence_id"] = "w10er12workerevidence-" + selections.canonical_sha256(worker)
    evidence = selections.bind_candidate_evidence({
        "schema_version": 1,
        "method": "er12_per_image_channel_simulation",
        "snr_db": 0,
        "modulation": "bpsk",
        "ldpc_rate": "1/3",
        "protocol_version": 1,
        "label_bits": 4,
        "payload_frame": "one_byte_low_nibble_predicted_label_high_nibble_zero",
        "true_label_in_payload": False,
        "outcome_counts": outcome_counts,
        "correct": 100,
        "total": 1000,
        "correct_digest": correct_digest,
        "noise_convention": selections.NOISE_CONVENTION,
        "noise_ids_digest": "d" * 64,
        "worker_evidence": worker,
    })
    return {
        "snr_db": 0,
        "modulation": "bpsk",
        "ldpc_rate": "1/3",
        "n_correct": 100,
        "n_delivered": 900,
        "n_decode_failure": 100,
        "n_infeasible": 0,
        "n_total": 1000,
        "per_image_correct_digest": correct_digest,
        "status": "eligible",
        "candidate_evidence": evidence,
    }


def test_er12_worker_evidence_is_independently_bound_and_cross_checked() -> None:
    entry = _er12_evidence_entry()
    selections._verify_er12_candidate_evidence(REPO, entry)

    evidence = dict(entry["candidate_evidence"])
    worker = dict(evidence["worker_evidence"])
    worker["correct"] = 101
    worker["worker_evidence_id"] = "w10er12workerevidence-" + selections.canonical_sha256(
        {key: item for key, item in worker.items() if key != "worker_evidence_id"}
    )
    evidence["worker_evidence"] = worker
    evidence["evidence_id"] = "w10candidateevidence-" + selections.canonical_sha256(
        {key: item for key, item in evidence.items() if key != "evidence_id"}
    )
    mutated = dict(entry, candidate_evidence=evidence)
    with pytest.raises(W10SelectionHold, match="worker evidence outcome"):
        selections._verify_er12_candidate_evidence(REPO, mutated)
