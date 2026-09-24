"""W10 authority contract tests: G11 closure, PAPR lifecycle, selections, successor, test."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import gen_w10_rehearsal_authorization as generator
import verify_w10_rehearsal as verifier
from config.params import get
from evaluation.w10_scope import SCOPE, unit_count
from training.deterministic_core import canonical_bytes

REPO = Path(__file__).resolve().parents[1]
G11_RECORD = {
    "authority_id": "g11h4auth-synthetic",
    "authority_path": "results/learned/g11/g11_execution_authorization_v4.json",
    "authority_sha256": "1" * 64,
    "terminal_id": "g11terminal-synthetic",
    "terminal_path": "results/learned/g11/g11_terminal_closeout.json",
    "terminal_sha256": "2" * 64,
    "decision": "GREEN",
    "test": "SEALED",
    "test_access": 0,
}
MANIFEST = {
    "manifest_kind": "W10_PREPARATORY_SOURCE_SUCCESSOR_V2",
    "manifest_id": "w10downstreamsourcev2-synthetic",
    "source_commit": "a" * 40,
}
SOURCE_RECORD = {
    "path": "results/learned/w10/w10_downstream_source_manifest_v2.json",
    "manifest_id": MANIFEST["manifest_id"],
    "sha256": "b" * 64,
}


def _fake_bindings() -> list[dict]:
    return [
        {
            "scope": entry.identity(),
            "binding_id": f"w10binding-{entry.role}",
            "checkpoint": {"state": "FROZEN"},
            "scorer": {"state": "FROZEN"},
            "selection": {"state": "FROZEN"},
        }
        for entry in SCOPE
    ]


def _authority(tmp_path: Path) -> dict:
    body = generator.build_body(
        gpu_name="NVIDIA TITAN Xp",
        gpu_uuid="GPU-46acd0f2-2ff5-1a43-cac9-2ae20e56dc9a",
        source=MANIFEST,
        bindings=_fake_bindings(),
        cuda_mapping={
            "cuda_visible_devices": "GPU-46acd0f2-2ff5-1a43-cac9-2ae20e56dc9a",
            "logical_device": "cuda:0",
            "cuda0_gpu_uuid": "GPU-46acd0f2-2ff5-1a43-cac9-2ae20e56dc9a",
            "cuda0_gpu_name": "NVIDIA TITAN Xp",
            "cuda0_compute_capability": "6.1",
            "device_count": 1,
        },
    )
    body["authority_id"] = "w10rehearsalauth-" + verifier.canonical_sha256(body)
    return body


@pytest.fixture(autouse=True)
def _synthetic_contract(monkeypatch):
    monkeypatch.setattr(verifier, "load_w10_manifest", lambda root, live=True, epoch=None: MANIFEST)
    monkeypatch.setattr(verifier, "assert_active_epoch_closure", lambda root, historical: None)
    monkeypatch.setattr(verifier, "source_record", lambda root, manifest: SOURCE_RECORD)
    monkeypatch.setattr(verifier, "resolve_scope_bindings", lambda root, verify_runtime=True: _fake_bindings())
    monkeypatch.setattr(verifier, "pending_states", lambda bindings: [])
    monkeypatch.setattr(generator, "source_record", lambda root, source: SOURCE_RECORD)
    monkeypatch.setattr(verifier, "_g11_closure", lambda: dict(G11_RECORD))
    monkeypatch.setattr(generator, "g11_closure", lambda: dict(G11_RECORD))


def _write(path: Path, body: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(body))
    return path


def test_valid_authority_passes_and_binds_g11_papr_and_selections(tmp_path: Path) -> None:
    value = verifier.verify_authority(_write(tmp_path / "w10_authority.json", _authority(tmp_path)))
    assert value["g11_closure"] == G11_RECORD
    assert value["g11_closure"]["decision"] == "GREEN" and value["g11_closure"]["test_access"] == 0
    assert value["papr_training_run_count"] == 1 and value["papr_lifecycle_bound"] is True
    assert value["runtime_root"] == "checkpoints/w10_rehearsal"


def test_authority_mutations_fail_closed(tmp_path: Path) -> None:
    base = _authority(tmp_path)

    def resign(body: dict) -> dict:
        body = dict(body)
        body["authority_id"] = "w10rehearsalauth-" + verifier.canonical_sha256(
            {key: item for key, item in body.items() if key != "authority_id"}
        )
        return body

    mutations = [
        ("g11_closure", {**G11_RECORD, "decision": "RED"}),
        ("g11_closure", {**G11_RECORD, "terminal_id": "changed"}),
        ("g11_closure", None),
        ("papr_training_run_count", 0),
        ("papr_lifecycle_bound", False),
        ("runtime_root", "results/learned/w10/runtime"),
        ("source_commit", "c" * 40),
        ("source_manifest", {"path": "other", "manifest_id": "x", "sha256": "y"}),
        ("unit_count", unit_count() + 1),
        ("scope_sha256", "0" * 64),
        ("test_access", 1),
        ("test", "OPEN"),
        ("systems", ["learned"]),
    ]
    for index, (field, value) in enumerate(mutations):
        path = _write(tmp_path / f"mut{index}.json", resign({**base, field: value}))
        with pytest.raises(Exception):
            verifier.verify_authority(path)

    # ID corruption is rejected before any semantic check.
    path = _write(tmp_path / "mut-id.json", {**base, "authority_id": "w10rehearsalauth-00"})
    with pytest.raises(Exception):
        verifier.verify_authority(path)
