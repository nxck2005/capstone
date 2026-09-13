#!/usr/bin/env python3
"""Verify the final post-search execution-source successor."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evaluation.downstream_v4 import PRODUCTION_CELLS, read_json, require  # noqa: E402
from runtime.source_guard import assert_clean_source_closure, assert_downstream_manifest_contract  # noqa: E402
from training.deterministic_core import canonical_sha256  # noqa: E402

TARGET = REPO / "results/learned/w9/downstream_source_manifest_v4.json"
SMOKE = REPO / "results/learned/w9/downstream_successor_synthetic_smoke.json"


def verify_smoke(path: Path = SMOKE) -> dict:
    value = read_json(path, "downstream synthetic smoke")
    body = dict(value)
    identifier = body.pop("smoke_id", None)
    require(identifier == "downstreamsmoke-" + canonical_sha256(body), "downstream smoke ID differs")
    require(value.get("artifact_role") == "FINAL_DOWNSTREAM_V4_SYNTHETIC_LIFECYCLE_SMOKE", "downstream smoke role differs")
    require(value.get("status") == "NON_SCIENTIFIC" and value.get("eligibility") == {"synthetic_only": True, "scientific_use": False, "selection_use": False}, "downstream smoke eligibility differs")
    cells = value.get("production_cells")
    require(isinstance(cells, list) and [tuple(item.get("cell", ())) for item in cells] == list(PRODUCTION_CELLS), "downstream smoke cells differ")
    require(len({item.get("runtime_root") for item in cells}) == 3 and len({item.get("selected_checkpoint_sha256") for item in cells}) == 3, "downstream smoke cell isolation differs")  # literal-ok: exact production cells
    require(all(item.get("selected_checkpoint_loaded") is True for item in cells), "downstream smoke selected checkpoint was not loaded")
    for field in ("fresh_start", "transactional_checkpoint", "exact_resume", "terminalization", "selected_checkpoint_loading", "final_validation_plumbing"):
        require(value.get(field) is True, f"downstream smoke proof differs: {field}")
    require(value.get("scientific_dataset_reads") == 0 and value.get("scientific_training") == 0, "downstream smoke performed scientific work")
    require(value.get("test") == "SEALED" and value.get("test_access") == 0, "downstream smoke crossed test boundary")
    return value


def verify(path: Path = TARGET) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ValueError("downstream source manifest is missing or unsafe")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("downstream source manifest is not an object")
    body = dict(value)
    identifier = body.pop("manifest_id", None)
    if identifier != "w9downstreamsource-" + canonical_sha256(body):
        raise ValueError("downstream source manifest ID differs")
    assert_downstream_manifest_contract(value)
    assert_clean_source_closure(REPO, value)
    verify_smoke()
    return value


def main() -> int:
    value = verify()
    print(f"downstream source manifest PASS: {value['manifest_id']}; sha256={hashlib.sha256(TARGET.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
