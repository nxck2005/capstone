"""C1 orchestration, provenance, and fail-closed table-loader tests."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from baseline import g8_bler_characterization as characterization
from baseline.g8_campaign import rendered_json, sha256_bytes


REPO = Path(__file__).resolve().parents[1]


def test_source_manifest_reconstructs_before_registration() -> None:
    payload = characterization.build_source_manifest()
    assert payload["manifest_id"].startswith("g8charsrc-")
    assert payload["scientific_execution_performed"] is False
    assert payload["characterization_started"] is False
    assert payload["runtime"] == {
        "logical_root": "results/baseline/g8/work_units",
        "absolute_paths_bound": False,
    }
    characterization.validate_source_manifest(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("campaign_id", "wrong-campaign"),
        ("full_strength_trials", 16),
        ("selection_policy_sha256", "0" * 64),
    ),
)
def test_source_manifest_mutations_fail_closed(field: str, value: object) -> None:
    payload = characterization.build_source_manifest()
    payload[field] = value
    with pytest.raises(characterization.CharacterizationError):
        characterization.validate_source_manifest(payload)


def test_source_manifest_never_binds_itself_or_absolute_paths() -> None:
    payload = characterization.build_source_manifest()
    paths = [entry["path"] for entry in payload["sources"] + payload["dependencies"]]
    assert characterization.SOURCE_MANIFEST_RELATIVE_PATH not in paths
    assert all(not isinstance(value, str) or not value.startswith("/") for value in _walk(payload))


def _walk(value: object):
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)
    else:
        yield value


def test_production_root_requires_exact_spelling(tmp_path: Path) -> None:
    exact = Path("/home/nick/projects/capstone/results/baseline/g8/work_units")
    assert characterization.validate_production_root(exact) == exact
    for candidate in (
        Path("results/baseline/g8/work_units"),
        Path("/tmp/other-root"),
        Path("/home/nick/projects/capstone/results/baseline/g8/work_units/../work_units"),
    ):
        with pytest.raises(characterization.CharacterizationError):
            characterization.validate_production_root(candidate)


def test_characterization_sources_have_no_prohibited_data_boundary_imports() -> None:
    prohibited = {
        "data",
        "data.manifests",
        "src.data.test_access",
        "baseline.classical.codec",
        "baseline.classifier",
    }
    paths = [REPO / relative for relative in characterization.CHARACTERIZATION_SOURCE_PATHS]
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(name in prohibited or name.startswith("data.") for name in names), path


def test_worker_authenticates_once_for_multiple_units(monkeypatch: pytest.MonkeyPatch) -> None:
    import run_g8_bler_characterization as cli

    calls = {"contexts": 0, "units": []}

    class FakeContext:
        resume_context = object()

    def fake_context() -> FakeContext:
        calls["contexts"] += 1
        return FakeContext()

    def fake_plan(*args, **kwargs):
        return {"remaining_work_unit_ids": ["u-1", "u-2"]}

    def fake_run(context, **kwargs):
        calls["units"].append(kwargs["work_unit_id"])
        return {
            "attempt": 1,
            "request_sha256": "1" * 64,
            "result_sha256": "2" * 64,
            "state_sha256": "3" * 64,
            "result": {"status": "complete", "measurement": {"trials_completed": 5000}},
        }

    monkeypatch.setattr(cli.runner, "AuthenticatedRunnerContext", fake_context)
    monkeypatch.setattr(cli.resume, "build_resume_plan", fake_plan)
    monkeypatch.setattr(cli.runner, "run_one_unit", fake_run)
    sink: list[dict] = []
    cli._worker_main(
        "/home/nick/projects/capstone/results/baseline/g8/work_units",
        "cuda",
        1,
        0,
        1,
        ["u-1", "u-2"],
        SimpleNamespace(put=sink.append),
    )
    assert calls == {"contexts": 1, "units": ["u-1", "u-2"]}
    assert sink[0]["ok"] is True


def test_table_loader_rejects_missing_table() -> None:
    with pytest.raises((FileNotFoundError, characterization.CharacterizationError)):
        characterization.load_bler_table(REPO / "results/baseline/g8/does-not-exist.json")


def test_canonical_self_hash_rule_is_stable() -> None:
    payload = characterization.build_source_manifest()
    raw = rendered_json(payload)
    assert sha256_bytes(raw) != payload["manifest_id"].split("-", 1)[1]
    assert json.loads(raw)["manifest_id"] == payload["manifest_id"]
