"""W10 successor source-epoch custody and zero-science boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from runtime.source_epochs import (
    W10_V2_MANIFEST_KIND,
    W10_V2_SOURCE_PATH,
    W10_V3_MANIFEST_KIND,
    W10_V3_SOURCE_PATH,
    W10_V4_MANIFEST_KIND,
    W10_V4_SOURCE_PATH,
    active_manifest_path,
    load_w10_manifest,
)

REPO = Path(__file__).resolve().parents[1]
V1_PATH = REPO / "results/learned/w10/w10_downstream_source_manifest.json"
V2_PATH = REPO / W10_V2_SOURCE_PATH
V3_PATH = REPO / W10_V3_SOURCE_PATH
V4_PATH = REPO / W10_V4_SOURCE_PATH


def test_successor_v1_and_v2_bytes_remain_immutable() -> None:
    assert hashlib.sha256(V1_PATH.read_bytes()).hexdigest() == (
        "dedc2c894dad47b24ead8cdbc22513d2dbe54945e41415e4e9d3692421eab3ea"
    )
    assert hashlib.sha256(V2_PATH.read_bytes()).hexdigest() == (
        "d2ce8f2ee88d97a11acdae5bc60e1dcdb35e5181c2a017e0a68a66e292f975e3"
    )
    assert hashlib.sha256(V3_PATH.read_bytes()).hexdigest() == (
        "0ab6be914f5a02e772b10827e61785d6699c6dd4c3b916a6c2caca484d3e40f8"
    )


def test_successor_v4_is_active_and_supersedes_v3_before_science() -> None:
    value = load_w10_manifest(REPO, live=True)
    assert active_manifest_path(REPO) == V4_PATH
    assert value["manifest_kind"] == W10_V4_MANIFEST_KIND
    predecessor = value["superseded_successor"]
    v3 = json.loads(V3_PATH.read_bytes())
    assert predecessor == {
        "path": W10_V3_SOURCE_PATH,
        "manifest_kind": W10_V3_MANIFEST_KIND,
        "manifest_id": v3["manifest_id"],
        "sha256": hashlib.sha256(V3_PATH.read_bytes()).hexdigest(),
        "source_commit": v3["source_commit"],
        "superseded_before_science": True,
        "papr_constrained_training_runs": 0,
        "w10_scientific_units": 0,
        "test_access": 0,
    }


def test_successor_v3_remains_exact_superseded_history() -> None:
    value = load_w10_manifest(REPO, live=False, epoch="v3")
    predecessor = value["superseded_successor"]
    v2 = json.loads(V2_PATH.read_bytes())
    assert predecessor == {
        "path": W10_V2_SOURCE_PATH,
        "manifest_kind": W10_V2_MANIFEST_KIND,
        "manifest_id": v2["manifest_id"],
        "sha256": hashlib.sha256(V2_PATH.read_bytes()).hexdigest(),
        "source_commit": v2["source_commit"],
        "superseded_before_science": True,
        "papr_constrained_training_runs": 0,
        "w10_scientific_units": 0,
        "test_access": 0,
    }
    assert value["pre_science_state"] == {
        "papr_constrained_training_runs": 0,
        "w10_authority_frozen": False,
        "w10_scientific_units": 0,
        "g12_freeze_manifest": False,
        "test": "SEALED",
        "test_access": 0,
    }
