"""W10 JPEG/ER-12 selection contract and mutation tests (synthetic artifacts)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import evaluation.w10_selections as selections
from evaluation.w10_selections import (
    ER12_SELECTION_PREFIX,
    JPEG_SELECTION_PREFIX,
    W10SelectionHold,
    build_selection_artifact,
    er12_rank_key,
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


def _score_entry(entry: dict, *, n_correct: int, n_delivered: int, decode: int = 0) -> dict:
    scored = {
        **entry,
        "n_correct": n_correct,
        "n_delivered": n_delivered,
        "n_decode_failure": decode,
        "n_infeasible": 1000 - n_delivered - decode,
        "n_total": 1000,
        "per_image_correct_digest": score_vector_digest([True] * n_correct + [False] * (1000 - n_correct)),
    }
    return scored


def _build(kind: str) -> dict:
    source = JPEG_CANDIDATES if kind == "jpeg_secondary" else ER12_CANDIDATES
    score_table = []
    winners = []
    for snr in GRID:
        entries = []
        for index, entry in enumerate(source[int(snr)]):
            entries.append(_score_entry(entry, n_correct=500 + index, n_delivered=900 + index))
        score_table.append({"snr_db": int(snr), "candidates": entries})
        winners.append(rank_candidates(kind, entries))
    return build_selection_artifact(kind, selections=winners, candidate_scores=score_table)


def test_rank_keys_are_total_and_tie_ordered() -> None:
    a = {"snr_db": 0, "modulation": "qpsk", "ldpc_rate": "1/2", "encode_axis_px": 32, "quality": 50, "n_correct": 10, "n_delivered": 5}
    b = dict(a, quality=10)
    assert jpeg_rank_key(a) < jpeg_rank_key(b)  # higher quality wins an exact tie
    c = {"snr_db": 0, "modulation": "bpsk", "ldpc_rate": "1/2", "n_correct": 10, "n_delivered": 5}
    d = {"snr_db": 0, "modulation": "qpsk", "ldpc_rate": "1/2", "n_correct": 10, "n_delivered": 5}
    assert er12_rank_key(c) < er12_rank_key(d)
    assert jpeg_rank_key(a) != jpeg_rank_key({"snr_db": 0, "modulation": "qpsk", "ldpc_rate": "1/2", "encode_axis_px": 32, "quality": 51, "n_correct": 10, "n_delivered": 5})


def test_jpeg_phy_shortlist_matches_the_frozen_pass_two_selection() -> None:
    shortlist = jpeg_phy_shortlist(REPO)
    assert sorted(shortlist) == [int(snr) for snr in GRID]
    table = json.loads((REPO / "results/baseline/g8_e/candidate_authority.json").read_bytes())
    ids = {str(item["candidate_id"]) for item in table["candidates"]}
    state = json.loads((REPO / "results/baseline/g8_f/pass_two_state.json").read_bytes())
    call = next(item for item in state["calls"] if item["mode"] == "classical_adaptive" and item["ratio"] == "r_1_6")
    assert [point["authority_candidate_id"] for point in call["per_snr"]] == [shortlist[snr]["authority_candidate_id"] for snr in sorted(shortlist)]
    assert all(phy["authority_candidate_id"] in ids for phy in shortlist.values())


@pytest.mark.parametrize("kind", ["jpeg_secondary", "er12_label_bound"])
def test_verified_artifacts_recompute_and_reject_every_mutation(kind: str, _synthetic_candidate_sets) -> None:
    artifact = _build(kind)
    path = REPO / ("results/learned/w10/jpeg_validation_selection.json" if kind == "jpeg_secondary" else "results/learned/w10/er12_validation_selection.json")
    verify_selection_artifact(REPO, kind, artifact, path=path)

    def mutate(field: str, value) -> dict:
        return {**artifact, field: value}

    def resign(value: dict) -> dict:
        body = dict(value)
        prefix = JPEG_SELECTION_PREFIX if kind == "jpeg_secondary" else ER12_SELECTION_PREFIX
        body["selection_id"] = prefix + selections.canonical_sha256({key: item for key, item in body.items() if key != "selection_id"})
        return body

    with pytest.raises(W10SelectionHold, match="selection ID"):
        verify_selection_artifact(
            REPO,
            kind,
            mutate("selection_id", "w10jpegselection-00" if kind == "jpeg_secondary" else "w10er12selection-00"),
            path=path,
        )

    # Every semantic mutation is resigned so it fails on its own contract check.
    semantic = [
        ("contract", dict(artifact["contract"], scorer="changed")),
        ("contract_sha256", "0" * 64),
        ("source_epoch", {"path": "other", "manifest_id": "x", "sha256": "y"}),
        ("denominator", 999),
        ("test_access", 1),
        ("test", "OPEN"),
    ]
    if kind == "er12_label_bound":
        semantic.append(("contract", dict(artifact["contract"], noise_convention="changed")))
    for field, value in semantic:
        with pytest.raises(W10SelectionHold):
            verify_selection_artifact(REPO, kind, resign(mutate(field, value)), path=path)

    # SNR order mutation.
    swapped = [dict(item) for item in artifact["candidate_scores"]]
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(W10SelectionHold, match="SNR order"):
        verify_selection_artifact(REPO, kind, resign(mutate("candidate_scores", swapped)), path=path)

    # Candidate grid mutation: dropping or adding a candidate at one SNR.
    dropped = [dict(item) for item in artifact["candidate_scores"]]
    dropped[0] = {"snr_db": dropped[0]["snr_db"], "candidates": dropped[0]["candidates"][:-1]}
    with pytest.raises(W10SelectionHold):
        verify_selection_artifact(REPO, kind, resign(mutate("candidate_scores", dropped)), path=path)
    added = [dict(item) for item in artifact["candidate_scores"]]
    added[0] = {
        "snr_db": added[0]["snr_db"],
        "candidates": added[0]["candidates"] + [{**added[0]["candidates"][0], "quality": 7} if kind == "jpeg_secondary" else {**added[0]["candidates"][0], "modulation": "qam16"}],
    }
    with pytest.raises(W10SelectionHold):
        verify_selection_artifact(REPO, kind, resign(mutate("candidate_scores", added)), path=path)

    # Changing the frozen modulation/rate/axis of every candidate at one SNR.
    altered = [dict(item) for item in artifact["candidate_scores"]]
    changed = [dict(entry) for entry in altered[0]["candidates"]]
    for entry in changed:
        if kind == "jpeg_secondary":
            entry["ldpc_rate"] = "2/3" if entry["ldpc_rate"] != "2/3" else "1/2"
        else:
            entry["modulation"] = "qam16" if entry["modulation"] != "qam16" else "bpsk"
    altered[0] = {"snr_db": altered[0]["snr_db"], "candidates": changed}
    with pytest.raises(W10SelectionHold):
        verify_selection_artifact(REPO, kind, resign(mutate("candidate_scores", altered)), path=path)

    # Winner mutation: changing the stored quality/modulation must fail.
    winners = [dict(item) for item in artifact["selections"]]
    if kind == "jpeg_secondary":
        winners[0] = {**winners[0], "quality": winners[0]["quality"] + 1}
    else:
        winners[0] = {**winners[0], "modulation": "qam16" if winners[0]["modulation"] != "qam16" else "bpsk"}
    with pytest.raises(W10SelectionHold, match="recomputed frozen winner"):
        verify_selection_artifact(REPO, kind, resign(mutate("selections", winners)), path=path)

    # Tie-break flag mutation.
    winners = [dict(item) for item in artifact["selections"]]
    winners[0] = {**winners[0], "tie_break_applied": not winners[0]["tie_break_applied"]}
    with pytest.raises(W10SelectionHold, match="recomputed frozen winner"):
        verify_selection_artifact(REPO, kind, resign(mutate("selections", winners)), path=path)

    # Score mutation: inflating a losing candidate flips the recomputed winner.
    scores = [dict(item) for item in artifact["candidate_scores"]]
    bad = [dict(entry) for entry in scores[0]["candidates"]]
    bad[0] = {**bad[0], "n_correct": 1000}
    scores[0] = {"snr_db": scores[0]["snr_db"], "candidates": bad}
    with pytest.raises(W10SelectionHold, match="recomputed frozen winner"):
        verify_selection_artifact(REPO, kind, resign(mutate("candidate_scores", scores)), path=path)
    scores = [dict(item) for item in artifact["candidate_scores"]]
    bad = [dict(entry) for entry in scores[0]["candidates"]]
    bad[0] = {**bad[0], "n_total": 999}
    scores[0] = {"snr_db": scores[0]["snr_db"], "candidates": bad}
    with pytest.raises(W10SelectionHold, match="denominator"):
        verify_selection_artifact(REPO, kind, resign(mutate("candidate_scores", scores)), path=path)
