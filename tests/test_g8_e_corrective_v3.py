"""Corrected-v3 pre-data lifecycle, scale and evidence-integrity tests.

Every generated fixture is NON-SCIENTIFIC, NON-SELECTION, NOT PRODUCTION E2
EVIDENCE, and MERGE-INELIGIBLE FOR PRODUCTION.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from baseline import g8_e_corrected_v3 as v3
import prove_g8_e_corrected_v3_lifecycle as proof
import run_g8_e_corrected_v3 as runner


def _require_bundle() -> None:
    if not v3.V3_CONTRACT_PATH.is_file():
        pytest.skip("v3 artifacts are frozen after the code-bearing source commit")


@pytest.fixture
def fake_br11(monkeypatch: pytest.MonkeyPatch) -> None:
    import baseline.g8_d as g8d

    monkeypatch.setattr(g8d, "account_br11", proof._br11_stub)


def _authorize(path: Path, context: dict[str, Any]) -> None:
    proof._authorization(path, context)


def _runner_args(context: dict[str, Any], root: Path, auth: Path, mode: str) -> list[str]:
    return [
        f"--{mode}",
        "--campaign-id", context["contract"]["campaign_id"],
        "--runtime-root", str(root),
        "--authorization", str(auth),
    ]


@pytest.fixture
def completed_runtime(tmp_path: Path, fake_br11: None) -> tuple[dict[str, Any], Path]:
    context = proof.fixture_context()
    root = tmp_path / "runtime"
    auth = tmp_path / "authorization.json"
    _authorize(auth, context)
    assert runner.main(_runner_args(context, root, auth, "start"), fixture=context) == 0
    return context, root


def _publish_e3(context: dict[str, Any], root: Path, counters: dict[str, int] | None = None) -> tuple[dict[str, Any], Path, str]:
    return v3.publish_e3_artifact(
        authority=context["authority"],
        sample_ids=context["sample_ids"],
        sample_labels=context["sample_labels"],
        runtime_root=root,
        contract=context["contract"],
        production=False,
        instrumentation=counters,
    )


def _record_paths(root: Path) -> list[Path]:
    return sorted((root / "records").glob("*.json"))


def _rewrite_record(path: Path, mutate: Any) -> None:
    value = json.loads(path.read_text())
    mutate(value)
    value["record_id"] = v3._id(
        v3.v2.V2_RECORD_PREFIX,
        {key: child for key, child in value.items() if key != "record_id"},
    )
    path.write_bytes(v3.rendered_json(value))


def _copy_completed(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    for path in target.rglob("*"):
        if path.is_file():
            path.chmod(0o600)
    return target


def test_frozen_verifier_and_phase_transitioned_predata_refusal() -> None:
    _require_bundle()
    # Corrected-v3 is preserved history; terminal v3s independently admits
    # AM-87's exact post-D7 builder bytes. These tests exercise frozen logic.
    frozen = v3.verify_v3_frozen_contract(verify_live_sources=False, verify_live_data=False)
    # The owner-authorized local E2 campaign began and was aborted at a clean
    # partial prefix (PARTIAL_OWNER_ABORTED_PROFILE_RELOCATION), so the E1-only
    # pre-data verifier must now refuse while the phase-invariant frozen
    # verifier keeps authenticating the same immutable contract bytes.
    with pytest.raises(v3.G8EV3Error, match="zero state"):
        v3.verify_v3_predata_zero_state(verify_live_sources=False, verify_live_data=False)
    # The refusal must come from tracked lifecycle evidence on every host.
    assert v3.V3_AUTHORIZATION_PATH.is_file()
    # The preserved aborted local runtime is deliberately untracked custody
    # evidence: it exists only on the writer machine, where the
    # external_dataset-marked runner-refusal test pins its exact partial
    # prefix. A clean checkout cannot see it, so its presence is asserted
    # only where it exists.
    if v3.V3_RUNTIME_ROOT.exists():
        assert v3.V3_RUNTIME_ROOT.is_dir()
        state = json.loads((v3.V3_RUNTIME_ROOT / "campaign_state.json").read_text())
        assert 0 < state["completed_prefix_count"] < state["total_required"]


def test_predata_closes_after_transition_without_poisoning_immutable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _require_bundle()
    synthetic_runtime = tmp_path / "runtime"
    synthetic_runtime.mkdir()
    monkeypatch.setattr(v3, "V3_RUNTIME_ROOT", synthetic_runtime)
    with pytest.raises(v3.G8EV3Error, match="zero state"):
        v3.verify_v3_predata_zero_state(verify_live_sources=False, verify_live_data=False)
    assert v3.verify_v3_frozen_contract(
        verify_live_sources=False, verify_live_data=False
    )["contract"]["checkpoint"] == "E1_corrected_v3"


def test_immutable_verifier_survives_synthetic_authorization_and_runtime(tmp_path: Path, fake_br11: None) -> None:
    _require_bundle()
    context = proof.fixture_context()
    auth = tmp_path / "authorization.json"
    root = tmp_path / "runtime"
    _authorize(auth, context)
    before = v3.verify_v3_frozen_contract(
        verify_live_sources=False, verify_live_data=False
    )["contract_sha256"]
    partial = dict(context, max_units=2)
    assert runner.main(_runner_args(context, root, auth, "start"), fixture=partial) == 0
    assert v3.verify_v3_frozen_contract(
        verify_live_sources=False, verify_live_data=False
    )["contract_sha256"] == before


def test_actual_start_resume_e2_e3_e4_lifecycle(tmp_path: Path, fake_br11: None) -> None:
    context = proof.fixture_context()
    auth = tmp_path / "authorization.json"
    root = tmp_path / "runtime"
    _authorize(auth, context)
    assert runner.main(_runner_args(context, root, auth, "start"), fixture=dict(context, max_units=3)) == 0
    assert v3._state_for_runtime(root)["completed_prefix_count"] == 3
    assert runner.main(_runner_args(context, root, auth, "resume"), fixture=context) == 0
    completion, _ = v3.verify_e2_completion_artifact(
        runtime_root=root,
        contract=context["contract"],
        authority=context["authority"],
        production=False,
    )
    e3, e3_path, e3_sha = _publish_e3(context, root)
    e4 = v3.build_e4_artifact(
        authority=context["authority"],
        sample_ids=context["sample_ids"],
        runtime_root=root,
        contract=context["contract"],
        e3_path=e3_path,
        e3_sha256=e3_sha,
        production=False,
    )
    assert completion["completed_work_unit_count"] == 9
    assert e3["observed_work_unit_count"] == 9
    assert e4["record_traversal_count"] == 9
    assert e4["e3_id"] == e3["e3_id"]


def test_authority_digest_is_computed_once_and_loads_are_constant(tmp_path: Path) -> None:
    context = proof.fixture_context()
    units = v3.expected_work_units(context["authority"], context["sample_ids"])
    codec = v3.v2.CodecArtifactV2(
        v3.PhysicalCacheKey("1" * 64, "2" * 64, (2, 2, 3), 10, 2, "f" * 64, "synthetic-codec"),
        v3.v2.OUTCOME_CODEC_INFEASIBILITY,
        "synthetic",
        None,
        None,
        "codec-object",
        False,
    )
    samples = {sample.stable_sample_id: sample for sample in context["samples"]}

    def execute(unit: dict[str, Any], sample: v3.SyntheticSample) -> v3.MeasurementRecordV3:
        return v3.MeasurementRecordV3.build(
            campaign_id=context["contract"]["campaign_id"],
            contract_id=context["contract"]["contract_id"],
            authority=context["authority"],
            work_unit=unit,
            structural=context["authority"]["structural_identities"][0],
            sample=sample,
            physical_key=codec.key,
            codec=codec,
            reconstruction=None,
            observation=None,
            outage_policy=context["contract"]["outage_policy"],
            profile_id="synthetic-profile",
            source_commit="synthetic-v3",
            g8_c_linkage_digest=v3.sha256_bytes(v3.canonical_json(context["contract"]["direct_upstream_bindings"])),
            record_labels=proof.LABELS,
        )

    campaign = v3.AtomicE2CampaignV3(
        runtime_root=tmp_path,
        contract=context["contract"],
        authority=context["authority"],
        work_units=units,
        executor=execute,
        sample_provider=lambda sample_id: samples[sample_id],
        mode="start",
    )
    for _ in range(5):
        campaign.state()
    campaign.run_next()
    instrumentation = campaign.instrumentation()
    assert instrumentation["authority_order_digest_computations"] == 1
    assert instrumentation["full_authority_id_visits_initialization"] == len(units)
    assert instrumentation["full_authority_id_visits_during_normal_progression"] == 0


@pytest.mark.parametrize("mutation", ("missing", "extra", "substitution", "ordinal", "structural"))
def test_e3_exact_set_and_order_mutations_are_rejected(completed_runtime: tuple[dict[str, Any], Path], tmp_path: Path, mutation: str) -> None:
    context, original = completed_runtime
    root = _copy_completed(original, tmp_path / mutation)
    paths = _record_paths(root)
    if mutation == "missing":
        paths[0].unlink()
    elif mutation == "extra":
        shutil.copyfile(paths[0], root / "records" / "foreign.json")
    elif mutation == "substitution":
        paths[0].write_bytes(paths[1].read_bytes())
    elif mutation == "ordinal":
        _rewrite_record(paths[0], lambda value: value.__setitem__("authority_ordinal", 1))
    else:
        _rewrite_record(paths[0], lambda value: value["structural_identity"].__setitem__("modulation", "foreign"))
    with pytest.raises(v3.G8EV3Error):
        _publish_e3(context, root)


def test_e3_operations_are_indexed_and_linear(completed_runtime: tuple[dict[str, Any], Path]) -> None:
    context, root = completed_runtime
    counters: dict[str, int] = {}
    e3 = v3.build_e3_artifact(
        authority=context["authority"],
        sample_ids=context["sample_ids"],
        sample_labels=context["sample_labels"],
        runtime_root=root,
        contract=context["contract"],
        production=False,
        instrumentation=counters,
    )
    n = e3["required_work_unit_count"]
    assert counters["records_parsed"] == n
    assert counters["structural_lookup_operations"] == n
    assert counters["expected_id_lookup_operations"] == 2 * n


def test_e3_rejects_manifest_label_substitution(completed_runtime: tuple[dict[str, Any], Path], tmp_path: Path) -> None:
    context, original = completed_runtime
    root = _copy_completed(original, tmp_path / "label-substitution")
    path = _record_paths(root)[0]
    _rewrite_record(path, lambda value: value.__setitem__("label", (value["label"] + 1) % 10))
    with pytest.raises(v3.G8EV3Error, match="class label"):
        _publish_e3(context, root)


@pytest.mark.parametrize(
    "mutation",
    ("reconstruction_corrupt", "reconstruction_pixels", "observation_corrupt", "prediction", "classifier", "source", "pixels", "budget", "axis"),
)
def test_e3_fully_authenticates_cache_references(completed_runtime: tuple[dict[str, Any], Path], tmp_path: Path, mutation: str) -> None:
    context, original = completed_runtime
    root = _copy_completed(original, tmp_path / mutation)
    delivered = next(path for path in _record_paths(root) if json.loads(path.read_text())["outcome"] == v3.v2.OUTCOME_DELIVERED)
    record = json.loads(delivered.read_text())
    if mutation.startswith("reconstruction"):
        target = root / "reconstruction" / f"{record['reconstruction']['object_id']}.json"
        if mutation == "reconstruction_corrupt":
            target.write_text("{}")
        else:
            value = json.loads(target.read_text())
            value["pixels_sha256"] = "0" * 64
            target.write_bytes(v3.rendered_json(value))
    elif mutation.startswith("observation") or mutation == "classifier":
        target = root / "observation" / f"{record['classifier_observation']['object_id']}.json"
        if mutation == "observation_corrupt":
            target.write_text("{}")
        else:
            value = json.loads(target.read_text())
            value["identity"]["classifier_checkpoint_sha256"] = "0" * 64
            target.write_bytes(v3.rendered_json(value))
    elif mutation == "prediction":
        _rewrite_record(delivered, lambda value: value["classifier_observation"].__setitem__("predicted_label", 1))
    else:
        field = {"source": "source_bytes_sha256", "pixels": "canonical_pixels_sha256", "budget": "payload_budget_bytes", "axis": "encode_axis_px"}[mutation]
        replacement: Any = "0" * 64 if mutation in {"source", "pixels"} else 999
        _rewrite_record(delivered, lambda value: value["physical_cache_key"].__setitem__(field, replacement))
    with pytest.raises(v3.G8EV3Error):
        _publish_e3(context, root)


@pytest.mark.parametrize("case", ("absent", "wrong_sha", "wrong_campaign", "mutated"))
def test_e4_requires_exact_frozen_e3(completed_runtime: tuple[dict[str, Any], Path], tmp_path: Path, case: str) -> None:
    context, root = completed_runtime
    _, e3_path, e3_sha = _publish_e3(context, root)
    contract = context["contract"]
    selected = e3_path
    digest = e3_sha
    if case == "absent":
        selected = tmp_path / "absent.json"
    elif case == "wrong_sha":
        digest = "0" * 64
    elif case == "wrong_campaign":
        contract = dict(contract, campaign_id="foreign")
    else:
        value = json.loads(e3_path.read_text())
        value["observed_work_unit_count"] -= 1
        e3_path.write_bytes(v3.rendered_json(value))
    with pytest.raises((OSError, v3.G8EV3Error)):
        v3.build_e4_artifact(
            authority=context["authority"],
            sample_ids=context["sample_ids"],
            runtime_root=root,
            contract=contract,
            e3_path=selected,
            e3_sha256=digest,
            production=False,
        )


def test_e4_traverses_once_and_counts_exactly(completed_runtime: tuple[dict[str, Any], Path]) -> None:
    context, root = completed_runtime
    _, e3_path, e3_sha = _publish_e3(context, root)
    counters: dict[str, int] = {}
    e4 = v3.build_e4_artifact(
        authority=context["authority"],
        sample_ids=context["sample_ids"],
        runtime_root=root,
        contract=context["contract"],
        e3_path=e3_path,
        e3_sha256=e3_sha,
        production=False,
        instrumentation=counters,
    )
    assert counters == {"e3_artifact_verifications": 1, "record_traversals": 9, "object_aggregation_operations": 9}
    assert all(obj["total_count"] == 3 for obj in e4["objects"])
    assert e4["validation_denominator"] == 3


@pytest.mark.parametrize("field", ("manifest_sha256", "ordered_validation_stable_ids_sha256", "validation_stable_id_set_sha256", "archive_sha256", "class_mapping_sha256"))
def test_scientific_data_identity_mutations_are_rejected_before_payload(field: str) -> None:
    manifest = Path("data/manifests/imagenette160.csv").read_bytes()
    expected = v3.validation_identity_from_manifest_bytes(manifest)
    observed = json.loads(json.dumps(expected))
    observed[field] = "0" * 64
    observed["data_identity_id"] = v3._id(
        v3.V3_DATA_PREFIX,
        {key: child for key, child in observed.items() if key != "data_identity_id"},
    )
    with pytest.raises(v3.G8EV3Error, match="before payload decode"):
        v3.verify_scientific_data_identity(expected, observed)


@pytest.mark.parametrize("failure", ("codec", "decoder", "classifier"))
def test_unexpected_work_failure_creates_no_scientific_row(tmp_path: Path, fake_br11: None, failure: str) -> None:
    context = proof.fixture_context()
    sample = context["samples"][0]
    unit = v3.expected_work_units(context["authority"], (sample.stable_sample_id,))[0]
    backend: Any = context["backend"]
    decoder: Any = context["decoder"]
    classifier: Any = context["classifier"]
    if failure == "codec":
        backend = SimpleNamespace(encode_to_budget=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("codec")))
    elif failure == "decoder":
        decoder = lambda stream: (_ for _ in ()).throw(RuntimeError("decoder"))
    else:
        classifier = SimpleNamespace(predict=lambda pixels: (_ for _ in ()).throw(RuntimeError("classifier")))
    engine = v3.MeasurementExecutorV3(
        contract=context["contract"],
        authority=context["authority"],
        runtime_root=tmp_path,
        backend=backend,
        decoder=decoder,
        classifier=classifier,
        non_scientific_fixture=True,
    )
    with pytest.raises(v3.FatalExecutionError):
        engine.execute(unit, sample)
    assert not (tmp_path / "records").exists()


def test_post_record_state_interruption_reconciles(tmp_path: Path, fake_br11: None, monkeypatch: pytest.MonkeyPatch) -> None:
    context = proof.fixture_context()
    samples = {sample.stable_sample_id: sample for sample in context["samples"]}
    units = v3.expected_work_units(context["authority"], context["sample_ids"])
    engine = v3.MeasurementExecutorV3(
        contract=context["contract"],
        authority=context["authority"],
        runtime_root=tmp_path,
        backend=context["backend"],
        decoder=context["decoder"],
        classifier=context["classifier"],
        non_scientific_fixture=True,
    )
    campaign = v3.AtomicE2CampaignV3(
        runtime_root=tmp_path,
        contract=context["contract"],
        authority=context["authority"],
        work_units=units,
        executor=engine.execute,
        sample_provider=lambda sample_id: samples[sample_id],
        mode="start",
    )
    with pytest.raises(RuntimeError):
        campaign.run_next(crash_after="state")
    assert len(_record_paths(tmp_path)) == 1
    resumed = v3.AtomicE2CampaignV3(
        runtime_root=tmp_path,
        contract=context["contract"],
        authority=context["authority"],
        work_units=units,
        executor=engine.execute,
        sample_provider=lambda sample_id: samples[sample_id],
        mode="resume",
    )
    assert resumed.state()["completed_prefix_count"] == 1
    assert resumed.state()["status"] != v3.v2.HOLD_STATUS


def test_readonly_active_state_authenticates_exact_prefix_before_payload(tmp_path: Path, fake_br11: None) -> None:
    context = proof.fixture_context()
    samples = {sample.stable_sample_id: sample for sample in context["samples"]}
    units = v3.expected_work_units(context["authority"], context["sample_ids"])
    engine = v3.MeasurementExecutorV3(
        contract=context["contract"], authority=context["authority"], runtime_root=tmp_path,
        backend=context["backend"], decoder=context["decoder"], classifier=context["classifier"],
        non_scientific_fixture=True,
    )
    campaign = v3.AtomicE2CampaignV3(
        runtime_root=tmp_path, contract=context["contract"], authority=context["authority"],
        work_units=units, executor=engine.execute, sample_provider=lambda sample_id: samples[sample_id], mode="start",
    )
    campaign.run_next()
    state = v3.verify_runtime_prefix_readonly(
        runtime_root=tmp_path,
        contract=context["contract"],
        authority=context["authority"],
        sample_ids=context["sample_ids"],
    )
    assert state["completed_prefix_count"] == 1


def test_production_executor_rejects_source_bytes_not_matching_stable_id(tmp_path: Path) -> None:
    context = proof.fixture_context()
    sample = context["samples"][0]
    unit = v3.expected_work_units(context["authority"], (sample.stable_sample_id,))[0]
    engine = v3.MeasurementExecutorV3(
        contract=context["contract"], authority=context["authority"], runtime_root=tmp_path,
        backend=context["backend"], decoder=context["decoder"], classifier=context["classifier"],
        non_scientific_fixture=False,
    )
    with pytest.raises(v3.FatalExecutionError, match="stable sample identity"):
        engine.execute(unit, sample)


@pytest.mark.external_dataset
def test_production_runner_refuses_old_v2_and_missing_authorization_before_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_bundle()
    bundle = v3.verify_v3_frozen_contract(verify_live_sources=False)
    old_campaign = json.loads(v3.v2.V2_CONTRACT_PATH.read_text())["campaign_id"]
    missing_authorization = tmp_path / "missing-authorization.json"
    old_runtime = tmp_path / "old-runtime-must-not-be-created"
    old = subprocess.run(
        [
            sys.executable,
            "tools/run_g8_e_corrected_v3.py",
            "--start",
            "--campaign-id", old_campaign,
            "--runtime-root", str(old_runtime),
            "--authorization", str(missing_authorization),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert old.returncode == 2
    assert not old_runtime.exists()
    production_root = v3.V3_RUNTIME_ROOT
    production_state = production_root / "campaign_state.json"
    production_state_before = production_state.read_bytes() if production_state.is_file() else None
    isolated_runtime = tmp_path / "runtime-must-not-be-created"
    monkeypatch.setattr(v3, "V3_RUNTIME_ROOT", isolated_runtime)
    monkeypatch.setattr(v3, "V3_AUTHORIZATION_PATH", missing_authorization)

    def payload_boundary_reached(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("missing authorization reached a production payload boundary")

    monkeypatch.setattr(v3, "AtomicE2CampaignV3", payload_boundary_reached)
    monkeypatch.setattr(runner, "_production_samples", payload_boundary_reached)
    current = runner.main(
        [
            "--start",
            "--campaign-id", bundle["contract"]["campaign_id"],
            "--runtime-root", str(isolated_runtime),
            "--authorization", str(missing_authorization),
        ]
    )
    assert current == 2
    assert not isolated_runtime.exists()
    if production_state_before is None:
        assert not production_root.exists()
    else:
        assert production_state.read_bytes() == production_state_before


def test_frozen_boundaries_and_future_pass_one_plan() -> None:
    _require_bundle()
    contract = v3.verify_v3_frozen_contract(
        verify_live_sources=False, verify_live_data=False
    )["contract"]
    assert contract["safety"] == {
        "measurement_coverage": 0,
        "e2_completed_units": 0,
        "e3_present": False,
        "e4_present": False,
        "pass_one_started": False,
        "pass_one_completed": False,
        "training": 0,
        "pass_two": 0,
        "fallback_invoked": False,
        "ratio_adjudicated": False,
        "test_access": 0,
        "validation_decoding": 0,
    }
    plan = contract["selection_authorization"]
    assert plan == v3.v2._selection_call_plan()
    assert (plan["call_count"], plan["max_candidates"], plan["max_samples"]) == (18, 1008, 1000)
    assert plan["authorization_issued"] is False


def test_scale_evidence_has_all_required_sizes_and_linear_counters() -> None:
    _require_bundle()
    if not v3.V3_COMPLEXITY_PATH.is_file():
        pytest.skip("complexity evidence is generated after source freeze")
    value = json.loads(v3.V3_COMPLEXITY_PATH.read_text())
    assert value["sizes"] == [2500, 5000, 10000, 20000]
    for row in value["rows"]:
        assert row["e2"]["authority_order_digest_computations"] == 1
        assert row["e2"]["full_authority_id_visits_during_normal_progression"] == 0
        assert row["e3"]["records_parsed"] == row["n"]
        assert row["e4"]["record_traversals"] == row["n"]


def test_non_scientific_lifecycle_proof_is_complete() -> None:
    _require_bundle()
    if not v3.V3_SYNTHETIC_PROOF_PATH.is_file():
        pytest.skip("synthetic lifecycle proof is generated after source freeze")
    value = json.loads(v3.V3_SYNTHETIC_PROOF_PATH.read_text())
    assert value["status"] == "PASS"
    assert all(value["stages"].values())
    assert value["production_e2_evidence"] is False
    assert value["safety"]["test_access"] == 0
    assert value["safety"]["training"] == 0
