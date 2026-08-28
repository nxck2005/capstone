"""G-4 adjudicator fixtures only; no scientific candidate is loaded."""

from __future__ import annotations

import copy

import pytest

from adjudication.w7_g4 import G4Hold, adjudicate_g4, fixture_candidate
from training.w7_protocol import W7_LAMBDA_GRID


def _set(values):
    return [fixture_candidate(lambda_value, **kwargs) for lambda_value, kwargs in zip(W7_LAMBDA_GRID, values)]


def test_primary_uses_numeric_smallest_qualifying_lambda():
    candidates = _set([
        {"top1": 0.8, "psnr_db": 19.9},
        {"top1": 0.8, "psnr_db": 20.0},
        {"top1": 0.8, "psnr_db": 21.0},
        {"top1": 0.8, "psnr_db": 22.0},
        {"top1": 0.8, "psnr_db": 23.0},
    ])
    result = adjudicate_g4(list(reversed(candidates)))
    assert result["status"] == "G4_ADJUDICATED_PRIMARY"
    assert result["selected_lambda"] == 0.1


def test_relaxed_floor_is_used_only_when_primary_is_empty():
    candidates = _set([
        {"top1": 0.8, "psnr_db": 15.9},
        {"top1": 0.8, "psnr_db": 16.0},
        {"top1": 0.8, "psnr_db": 17.0},
        {"top1": 0.8, "psnr_db": 18.0},
        {"top1": 0.8, "psnr_db": 19.0},
    ])
    result = adjudicate_g4(candidates)
    assert result["status"] == "G4_ADJUDICATED_RELAXED"
    assert result["selection_tier"] == "RELAXED"
    assert result["selected_lambda"] == 0.1


def test_no_solution_is_an_explicit_hold():
    result = adjudicate_g4(_set([{"top1": 0.8, "psnr_db": 15.0}] * len(W7_LAMBDA_GRID)))
    assert result["status"] == "G4_HOLD_DEC2_REVERSAL_REPLAN_REQUIRED"
    assert result["selected_lambda"] is None


def test_percentage_point_tolerance_is_exact():
    candidates = _set([
        {"top1": 0.8, "psnr_db": 19.0},
        {"top1": 0.79, "psnr_db": 20.0},
        {"top1": 0.789, "psnr_db": 21.0},
        {"top1": 0.8, "psnr_db": 22.0},
        {"top1": 0.8, "psnr_db": 23.0},
    ])
    result = adjudicate_g4(candidates)
    assert result["selected_lambda"] == 0.1


def test_candidate_completeness_and_homogeneity_are_fail_closed():
    candidates = _set([{"top1": 0.8, "psnr_db": 20.0}] * len(W7_LAMBDA_GRID))
    with pytest.raises(G4Hold, match="exactly one"):
        adjudicate_g4(candidates[:-1])
    duplicate = copy.deepcopy(candidates)
    duplicate[-1]["lambda"] = 0.1
    with pytest.raises(G4Hold, match="duplicated|incomplete"):
        adjudicate_g4(duplicate)
    mixed = copy.deepcopy(candidates)
    mixed[1]["lineage"]["gpu_uuid"] = "GPU-other"
    with pytest.raises(G4Hold, match="homogeneity"):
        adjudicate_g4(mixed)
    profile = copy.deepcopy(candidates)
    profile[0]["eligibility"]["w7_g4_eligibility"] = "NOT_ELIGIBLE_FOR_W7_G4"
    with pytest.raises(G4Hold, match="eligibility"):
        adjudicate_g4(profile)
