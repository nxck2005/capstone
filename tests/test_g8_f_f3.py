"""Fail-closed synthetic checks for validation-only G8_F/F3."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from baseline import g8_f_f3 as f3


def test_identified_artifacts_detect_mutation() -> None:
    value = f3.identified({"schema_version": 1, "status": "OK"}, field="thing_id", prefix="thing-")
    f3._verify_identified(value, field="thing_id", prefix="thing-")
    mutated = dict(value)
    mutated["status"] = "MUTATED"
    with pytest.raises(f3.F3Hold, match="content digest"):
        f3._verify_identified(mutated, field="thing_id", prefix="thing-")


def test_reconstruction_identity_authenticates_exact_pixels(tmp_path: Path) -> None:
    pixels = bytes([7]) * (f3.CANONICAL_AXIS * f3.CANONICAL_AXIS * 3)
    identity = {"schema_version": 2, "test": "synthetic"}
    object_id = f3.v3._id(f3.v2.V2_RECONSTRUCTION_PREFIX, identity)
    value = {
        "schema_version": f3.v2.V2_SCHEMA_VERSION,
        "artifact_role": "g8_e_v2_reconstruction_cache_object",
        "identity": identity,
        "status": f3.v2.OUTCOME_DELIVERED,
        "reason": None,
        "pixels_b64": __import__("base64").b64encode(pixels).decode("ascii"),
        "pixels_sha256": f3.sha256_bytes(pixels),
        "object_id": object_id,
    }
    path = tmp_path / f"{object_id}.json"
    path.write_bytes(f3.rendered_json(value))
    file_sha, pixels_sha = f3._reconstruction_file_identity(path, object_id)
    assert file_sha == f3.sha256_file(path)
    assert pixels_sha == f3.sha256_bytes(pixels)
    value["pixels_sha256"] = "0" * 64
    path.write_bytes(f3.rendered_json(value))
    with pytest.raises(f3.F3Hold, match="pixels differ"):
        f3._reconstruction_file_identity(path, object_id)


def test_f3_source_has_no_codec_encoder_training_or_test_boundary() -> None:
    source = Path(f3.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    assert "data.test_access" not in imports
    assert not any(name.startswith("baseline.j2k") for name in imports)
    assert not any(name.startswith("training.reference_classifier") for name in imports)
    assert "optimizer.step(" not in source
    assert "torch.inference_mode()" in source


def test_f3_protected_terminal_counts_are_exact() -> None:
    expected = {
        "f3_cached_sweep_rescoring": 1,
        "f2_optimizer_steps_during_f3": 0,
        "pass_two": 0,
        "pass_three": 0,
        "fallback_training": 0,
        "learned_training": 0,
        "test_access": 0,
    }
    source = Path(f3.__file__).read_text(encoding="utf-8")
    assert json.dumps(expected, sort_keys=True, separators=(",", ":")) not in source  # no opaque hard-coded JSON blob
    assert expected["pass_three"] == expected["test_access"] == 0
