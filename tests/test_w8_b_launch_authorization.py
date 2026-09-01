from __future__ import annotations

import hashlib
from pathlib import Path

from gen_w8_b_launch_authorization import OUTPUT_PATH, _verify


def test_committed_w8_b_launch_authorization_authenticates_exactly() -> None:
    repo = Path(__file__).resolve().parents[1]
    value = _verify(
        OUTPUT_PATH,
        repo / "results/learned/w8/w8_execution_authorization.json",
        repo / "results/learned/w8/w8_source_manifest.json",
    )
    assert value["authorization_id"].startswith("w8blaunch-")
    assert value["scope"] == {
        "core_runs": 6,
        "er2_randomized_training": False,
        "papr_constrained_training": False,
        "er9_training": False,
        "g10": False,
    }
    assert value["test"] == {
        "status": "SEALED",
        "model_facing_access": 0,
        "learned_inference": 0,
    }
    assert len(hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest()) == 64
