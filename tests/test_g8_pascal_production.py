from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from baseline import g8_pascal_production as production
from baseline.g8_pascal_successor import REQUIRED_COUNT, rendered_json


def _profile(device: str) -> dict[str, object]:
    worker = next(item for item in production.PRODUCTION_WORKERS if item["device"] == device)
    bindings = production.successor_bindings()
    return {
        "execution_profile_id": production.SUCCESSOR_PROFILE_ID,
        "lock_file": "requirements-pascal.lock",
        "lock_file_sha256": bindings["lock_file_sha256"],
        "python_version": "3.12.0",
        "torch_version": "2.13.0+cu126",
        "torch_cuda_build": "12.6",
        "torchvision_version": "0.28.0+cu126",
        "numpy_version": "2.0.0",
        "sionna_version": "1.2.1",
        "openjpeg_version": None,
        "deterministic_backend": {"torch": True},
        "amp": False,
        "gpu_name": worker["gpu_name"],
        "gpu_uuid": worker["gpu_uuid"],
        "gpu_vram_mib": "12288",
        "gpu_compute_capability": worker["gpu_compute_capability"],
        "gpu_index": worker["shard_index"],
        "nvidia_smi_index": worker["shard_index"],
        "driver_version": "555.42.02",
        "git_commit": "a" * 40,
        "git_dirty": False,
        "config_hash": bindings["production_contract_sha256"],
        "device": device,
    }


def _run(root: Path, ordinal: int = 0, *, device: str = "cuda:0") -> dict[str, object]:
    worker = next(item for item in production.PRODUCTION_WORKERS if item["device"] == device)
    return production.run_unit(
        root,
        ordinal=ordinal,
        shard_index=int(worker["shard_index"]),
        shard_count=2,
        device=device,
        gpu_uuid=str(worker["gpu_uuid"]),
        profile=_profile(device),
        batch_size=4,
    )


def _complete(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        production,
        "execute_frozen_measurement",
        lambda request, *, device, batch_size: {
            "status": "complete",
            "trials_completed": 5000,
            "bit_errors": 0,
            "block_errors": 0,
        },
    )


def test_transaction_reaches_accepted_and_reconciles_nonzero_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _complete(monkeypatch)
    root = tmp_path / "runtime"
    result = _run(root)
    assert result["status"] == production.STATUS_ACCEPTED
    assert _run(root)["attempt"] == 1
    report = production.inspect_unit(root, 0)
    assert report["classification"] == production.STATUS_ACCEPTED
    assert report["request_attempts"] == [1]
    assert report["result_attempts"] == [1]
    summary = production.reconcile_campaign(root)
    assert summary["accepted_authority_ordinals"] == [0]
    assert summary["accepted_count"] == 1
    audited = production.audit_campaign(root)
    assert audited["accepted_count"] == 1
    assert audited["terminal_invalid_authority_ordinals"] == []


def test_audit_rejects_stale_terminal_invalid_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _complete(monkeypatch)
    root = tmp_path / "runtime"
    _run(root)
    production.reconcile_campaign(root)
    state_path = root / production.PRODUCTION_STATE_FILENAME
    state = json.loads(state_path.read_bytes())
    state["terminal_invalid_authority_ordinals"] = [1]
    state["scientific_execution_performed"] = True
    state["state_sha256"] = production.digest_without_field(state, "state_sha256")
    state_path.write_bytes(rendered_json(state))
    with pytest.raises(production.RecoveryError, match="stale relative to durable evidence"):
        production.audit_campaign(root)


def test_request_only_crash_restarts_without_skipping_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _complete(monkeypatch)
    original = production._write_state_cas

    def interrupt(root, state, *, expected_sha256):
        if state["identity"]["status"] == production.STATUS_REQUEST_PUBLISHED:
            raise RuntimeError("simulated request-state crash")
        return original(root, state, expected_sha256=expected_sha256)

    monkeypatch.setattr(production, "_write_state_cas", interrupt)
    root = tmp_path / "runtime"
    with pytest.raises(RuntimeError, match="request-state crash"):
        _run(root)
    interrupted = production.inspect_unit(root, 0)
    assert interrupted["classification"] == production.STATUS_CLAIMED
    assert interrupted["request_attempts"] == [1]
    assert interrupted["result_attempts"] == []
    monkeypatch.setattr(production, "_write_state_cas", original)
    assert _run(root)["status"] == production.STATUS_ACCEPTED
    assert production.inspect_unit(root, 0)["request_attempts"] == [1]


def test_result_published_before_state_update_is_recovered_without_reexecution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def measure(request, *, device, batch_size):
        calls["count"] += 1
        return {"status": "complete", "trials_completed": 5000, "bit_errors": 0, "block_errors": 0}

    monkeypatch.setattr(production, "execute_frozen_measurement", measure)
    original = production._write_state_cas

    def interrupt(root, state, *, expected_sha256):
        if state["identity"]["status"] == production.STATUS_RESULT_PUBLISHED:
            raise RuntimeError("simulated result-state crash")
        return original(root, state, expected_sha256=expected_sha256)

    monkeypatch.setattr(production, "_write_state_cas", interrupt)
    root = tmp_path / "runtime"
    with pytest.raises(RuntimeError, match="result-state crash"):
        _run(root)
    assert calls["count"] == 1
    interrupted = production.inspect_unit(root, 0)
    assert interrupted["classification"] == production.STATUS_REQUEST_PUBLISHED
    assert interrupted["result_attempts"] == [1]
    monkeypatch.setattr(production, "_write_state_cas", original)
    assert production.reconcile_campaign(root)["accepted_count"] == 1
    assert calls["count"] == 1


def test_failed_attempt_is_immutable_and_retries_as_attempt_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def measure(request, *, device, batch_size):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"status": "failed", "trials_completed": 0, "bit_errors": 0, "block_errors": 0}
        return {"status": "complete", "trials_completed": 5000, "bit_errors": 0, "block_errors": 0}

    monkeypatch.setattr(production, "execute_frozen_measurement", measure)
    root = tmp_path / "runtime"
    assert _run(root)["status"] == production.STATUS_FAILED
    assert production.reconcile_campaign(root)["failed_authority_ordinals"] == [0]
    assert _run(root)["status"] == production.STATUS_ACCEPTED
    assert production.reconcile_campaign(root)["accepted_authority_ordinals"] == [0]
    report = production.inspect_unit(root, 0)
    assert report["request_attempts"] == [1, 2]
    assert report["result_attempts"] == [1, 2]
    assert report["state"]["identity"]["attempt"] == 2
    assert production.request_path(root, report["work_unit_id"], 1).read_bytes() == production.request_path(root, report["work_unit_id"], 2).read_bytes()


@pytest.mark.parametrize("mutation", ["campaign", "profile", "lock", "uuid", "driver", "swapped"])
def test_foreign_campaign_profile_lock_uuid_and_swapped_mapping_rejected(mutation: str) -> None:
    bindings = production.successor_bindings()
    request = production.build_request(bindings, ordinal=0, profile=_profile("cuda:0"))
    changed = copy.deepcopy(request)
    if mutation == "campaign":
        changed["identity"]["campaign_id"] = "g8p-foreign"
    elif mutation == "profile":
        changed["identity"]["profile_provenance"]["execution_profile_id"] = "local_4060_cu130"
    elif mutation == "lock":
        changed["identity"]["lock_file_sha256"] = "0" * 64
    elif mutation == "uuid":
        changed["identity"]["profile_provenance"]["gpu_uuid"] = "GPU-foreign"
    elif mutation == "driver":
        changed["identity"]["profile_provenance"]["driver_version"] = ""
    else:
        with pytest.raises(production.ProductionContractError):
            production.build_request(bindings, ordinal=0, profile=_profile("cuda:1"))
        return
    with pytest.raises(production.SuccessorProductionError):
        production.validate_request(changed, bindings=bindings)


def test_runtime_rejects_generic_cuda_duplicate_result_stale_state_and_partial_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _complete(monkeypatch)
    root = tmp_path / "runtime"
    _run(root)
    report = production.inspect_unit(root, 0)
    state = report["state"]
    with pytest.raises(production.ProductionContractError, match="explicit cuda:N"):
        production.run_unit(
            root,
            ordinal=0,
            shard_index=0,
            shard_count=2,
            device="cuda",
            gpu_uuid=str(production.PRODUCTION_WORKERS[0]["gpu_uuid"]),
            profile=_profile("cuda:0"),
        )
    result_path = production.result_path(root, report["work_unit_id"], 1)
    with pytest.raises(production.PublicationConflict):
        production.publish_immutable_json(result_path, {"tampered": True}, root=root)
    with pytest.raises(production.StaleStateError):
        production.publish_state(root, state, expected_sha256="0" * 64)
    bucket = result_path.parent
    (bucket / ".partial-artifact").write_bytes(b"partial")
    with pytest.raises(production.RecoveryError):
        production.validate_runtime_namespace(root)


def test_predecessor_result_and_old_runtime_subtree_are_uningestible(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    production.ensure_runtime_root(root)
    old_result = next((production.OLD_WORK_UNIT_ROOT).glob("[0-9a-f][0-9a-f]/*.result.json"))
    bucket = root / old_result.parent.name
    bucket.mkdir()
    shutil.copyfile(old_result, bucket / old_result.name)
    with pytest.raises(production.RecoveryError):
        production.validate_runtime_namespace(root)
    with pytest.raises(production.RuntimeRootError):
        production.ensure_runtime_root(production.OLD_WORK_UNIT_ROOT / "foreign-runtime")


def test_successor_merge_boundary_is_successor_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from baseline.g8_pascal_merge import SuccessorMergeError, build_successor_bler_table, collect_successor_results

    _complete(monkeypatch)
    root = tmp_path / "runtime"
    _run(root)
    production.reconcile_campaign(root)
    records = collect_successor_results(root)
    assert len(records) == 1
    with pytest.raises(SuccessorMergeError, match="gated"):
        build_successor_bler_table(records)

    foreign_root = tmp_path / "foreign"
    production.ensure_runtime_root(foreign_root)
    old_result = next(production.OLD_WORK_UNIT_ROOT.glob("[0-9a-f][0-9a-f]/*.result.json"))
    foreign_bucket = foreign_root / old_result.parent.name
    foreign_bucket.mkdir()
    shutil.copyfile(old_result, foreign_bucket / old_result.name)
    with pytest.raises(SuccessorMergeError):
        collect_successor_results(foreign_root)


def test_production_source_contract_drift_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.loads(production.PRODUCTION_SOURCE_MANIFEST.read_bytes())
    payload["sources"][0]["sha256"] = "0" * 64
    mutant = tmp_path / "production_source_manifest.json"
    mutant.write_bytes(rendered_json(payload))
    monkeypatch.setattr(production, "PRODUCTION_SOURCE_MANIFEST", mutant)
    production._successor_bindings_json.cache_clear()
    with pytest.raises(production.ProductionContractError):
        production.validate_production_contracts()
    production._successor_bindings_json.cache_clear()


def test_pascal_successor_custody_policy_is_narrow_and_final_publication_is_mandatory() -> None:
    expected = production.PASCAL_SUCCESSOR_CUSTODY_POLICY
    coordinator = json.loads(production.PRODUCTION_COORDINATOR_CONTRACT.read_bytes())
    contract = json.loads(production.PRODUCTION_CONTRACT.read_bytes())
    assert coordinator["evidence_custody_policy"] == expected
    assert contract["evidence_custody_policy"] == expected
    assert expected["scope"] == "owner_authorized_confessor_pascal_cu126_g8_c_successor_only"
    assert expected["prepublication_loss_risk"] == "explicitly_accepted_by_owner"
    assert "before_bler_table_freeze_or_g8_d" in expected["final_handoff"]


@pytest.mark.parametrize(
    "field",
    [
        "scope",
        "local_evidence_accumulation",
        "git_publication_timing",
        "prepublication_loss_risk",
        "scientific_validity_basis",
        "final_handoff",
    ],
)
def test_pascal_successor_custody_policy_mutation_fails_closed(field: str) -> None:
    mutant = dict(production.PASCAL_SUCCESSOR_CUSTODY_POLICY)
    mutant[field] = "broadened"
    with pytest.raises(production.ProductionContractError, match="custody policy differs"):
        production._validate_custody_policy(mutant)

    mutant = dict(production.PASCAL_SUCCESSOR_CUSTODY_POLICY)
    del mutant[field]
    with pytest.raises(production.ProductionContractError, match="custody policy differs"):
        production._validate_custody_policy(mutant)


def test_nonzero_successor_verifier_reads_only_successor_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _complete(monkeypatch)
    root = tmp_path / "runtime"
    _run(root)
    production.reconcile_campaign(root)
    checked = subprocess.run(
        [sys.executable, "tools/verify_g8_pascal_successor.py", "--root", str(root)],
        cwd=production.REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(checked.stdout)
    assert result["accepted"] == 1
    assert result["required"] == REQUIRED_COUNT


def test_exact_two_shard_union_has_no_overlap_or_omission() -> None:
    partitions = production.exact_shard_partition()
    assert len(partitions[0]) + len(partitions[1]) == REQUIRED_COUNT
    assert not set(partitions[0]) & set(partitions[1])
    assert set(partitions[0]) | set(partitions[1]) == set(range(REQUIRED_COUNT))


def test_coordinator_rejects_shard_overlap_and_omission() -> None:
    import run_g8_pascal_dual_gpu as coordinator

    partitions = production.exact_shard_partition()
    workers = [
        {
            **dict(worker),
            "assigned_authority_ordinals": list(partitions[int(worker["shard_index"])]),
        }
        for worker in production.PRODUCTION_WORKERS
    ]
    overlap = copy.deepcopy(workers)
    overlap[1]["assigned_authority_ordinals"] = list(overlap[0]["assigned_authority_ordinals"])
    with pytest.raises(RuntimeError, match="assignment differs|overlap or omission"):
        coordinator._validate_partition(overlap)
    omission = copy.deepcopy(workers)
    omission[1]["assigned_authority_ordinals"].pop()
    with pytest.raises(RuntimeError, match="assignment differs|overlap or omission"):
        coordinator._validate_partition(omission)


def test_coordinator_launches_two_explicit_children_and_isolates_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import run_g8_pascal_dual_gpu as coordinator

    class Child:
        def __init__(self, return_code: int) -> None:
            self.return_code = return_code

        def wait(self) -> int:
            return self.return_code

    commands: list[list[str]] = []

    def popen(command, *, cwd, text):
        commands.append(command)
        return Child(143 if command[command.index("--device") + 1] == "cuda:0" else 0)

    monkeypatch.setattr(coordinator.subprocess, "Popen", popen)
    workers = [dict(item) for item in production.PRODUCTION_WORKERS]
    failures = coordinator.launch_children(
        {"workers": workers},
        root=tmp_path / "runtime",
        batch_size=32,
        max_units=1,
    )
    assert len(commands) == 2
    assert {command[command.index("--device") + 1] for command in commands} == {"cuda:0", "cuda:1"}
    assert all("cuda" not in command for command in commands)
    assert failures == [{"device": "cuda:0", "return_code": 143}]


def test_worker_max_units_skips_accepted_ordinals_without_consuming_cap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import run_g8_pascal_dual_gpu as coordinator

    accepted: set[int] = set()
    calls: list[int] = []

    def inspect(root, ordinal):
        return {"classification": production.STATUS_ACCEPTED if ordinal in accepted else "available"}

    def complete_unit(root, *, ordinal, shard_index, shard_count, device, gpu_uuid, profile, batch_size):
        calls.append(ordinal)
        accepted.add(ordinal)
        return {"status": production.STATUS_ACCEPTED}

    monkeypatch.setattr(coordinator, "inspect_unit", inspect)
    monkeypatch.setattr(coordinator, "run_unit", complete_unit)
    root = tmp_path / "runtime"

    for expected_calls, expected_ordinal in [([0], 0), ([0, 2], 2), ([0, 2, 4], 4), ([0, 2, 4, 6], 6)]:
        assert coordinator._run_worker_batch(
            root=root,
            partition=[0, 2, 4, 6],
            shard_index=0,
            shard_count=2,
            device="cuda:0",
            gpu_uuid=str(production.PRODUCTION_WORKERS[0]["gpu_uuid"]),
            profile={},
            batch_size=32,
            max_units=1,
        ) == 1
        assert calls == expected_calls
        assert calls[-1] == expected_ordinal


def test_worker_max_units_counts_failed_attempt_and_stops_batch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import run_g8_pascal_dual_gpu as coordinator

    calls: list[int] = []

    def failed_unit(root, *, ordinal, shard_index, shard_count, device, gpu_uuid, profile, batch_size):
        calls.append(ordinal)
        return {"status": production.STATUS_FAILED}

    monkeypatch.setattr(coordinator, "run_unit", failed_unit)
    root = tmp_path / "runtime"
    production.ensure_runtime_root(root)
    attempted = coordinator._run_worker_batch(
        root=root,
        partition=[0, 2, 4],
        shard_index=0,
        shard_count=2,
        device="cuda:0",
        gpu_uuid=str(production.PRODUCTION_WORKERS[0]["gpu_uuid"]),
        profile={},
        batch_size=32,
        max_units=1,
    )
    assert attempted == 1
    assert calls == [0]


def test_worker_does_not_skip_terminal_invalid_evidence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import run_g8_pascal_dual_gpu as coordinator

    calls: list[int] = []

    def terminal_invalid(root, ordinal):
        return {"classification": production.STATUS_TERMINAL_INVALID}

    def should_not_run(root, *, ordinal, shard_index, shard_count, device, gpu_uuid, profile, batch_size):
        calls.append(ordinal)
        return {"status": production.STATUS_ACCEPTED}

    monkeypatch.setattr(coordinator, "inspect_unit", terminal_invalid)
    monkeypatch.setattr(coordinator, "run_unit", should_not_run)
    with pytest.raises(production.RecoveryError, match="terminal-invalid"):
        coordinator._run_worker_batch(
            root=tmp_path / "runtime",
            partition=[0, 2],
            shard_index=0,
            shard_count=2,
            device="cuda:0",
            gpu_uuid=str(production.PRODUCTION_WORKERS[0]["gpu_uuid"]),
            profile={},
            batch_size=32,
            max_units=1,
        )
    assert calls == []


def test_failed_run_unit_makes_worker_and_coordinator_nonpass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import run_g8_pascal_dual_gpu as coordinator

    worker = production.PRODUCTION_WORKERS[0]
    args = SimpleNamespace(
        device=worker["device"],
        gpu_uuid=worker["gpu_uuid"],
        shard_index=worker["shard_index"],
        shard_count=2,
        root=tmp_path / "runtime",
        batch_size=32,
        max_units=1,
    )
    monkeypatch.setattr(coordinator, "load_json", lambda path: {})
    monkeypatch.setattr(coordinator, "validate_successor_manifest", lambda payload: None)
    monkeypatch.setattr(coordinator, "validate_successor_state", lambda payload: None)
    monkeypatch.setattr(coordinator, "validate_production_contracts", lambda: {"workers": [dict(worker)]})
    monkeypatch.setattr(coordinator, "authenticate_worker_profile", lambda **kwargs: {})
    monkeypatch.setattr(coordinator, "ensure_runtime_root", lambda root: root)
    monkeypatch.setattr(coordinator, "exact_shard_partition", lambda: {0: [0], 1: []})
    monkeypatch.setattr(coordinator, "inspect_unit", lambda root, ordinal: {"classification": "available"})
    monkeypatch.setattr(coordinator, "run_unit", lambda *args, **kwargs: {"status": production.STATUS_FAILED})
    summaries = iter([
        {"failed_authority_ordinals": [], "terminal_invalid_authority_ordinals": [], "in_progress_authority_ordinals": []},
        {"failed_authority_ordinals": [0], "terminal_invalid_authority_ordinals": [], "in_progress_authority_ordinals": []},
    ])
    monkeypatch.setattr(coordinator, "reconcile_campaign", lambda root: next(summaries))

    assert coordinator._worker(args) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "FAIL"


def test_coordinator_reconciliation_failure_cannot_print_pass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import run_g8_pascal_dual_gpu as coordinator

    root = tmp_path / "runtime"
    monkeypatch.setattr(coordinator, "build_plan", lambda: {"workers": []})
    monkeypatch.setattr(coordinator, "_validate_launch_gate", lambda root, authorization, dry_run: {})
    monkeypatch.setattr(coordinator, "ensure_runtime_root", lambda path: path)
    monkeypatch.setattr(coordinator, "launch_children", lambda plan, *, root, batch_size, max_units: [])
    monkeypatch.setattr(coordinator, "reconcile_campaign", lambda path: {
        "failed_authority_ordinals": [0],
        "terminal_invalid_authority_ordinals": [],
        "in_progress_authority_ordinals": [],
    })
    monkeypatch.setattr(sys, "argv", ["run_g8_pascal_dual_gpu.py", "--execute", "--root", str(root)])

    assert coordinator.main() == 1
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "FAIL"
