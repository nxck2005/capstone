"""B4 runner, gate, RNG, publication and isolated transaction tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from baseline import g8_bler_contract as bler_contract
from baseline import g8_bler_runner as runner
from baseline import g8_bler_work_units as work_units
from baseline.ldpc import adapter as adapter_module
from baseline.g8_campaign import canonical_json
from config.params import get


@pytest.fixture(scope="module")
def auth_context() -> runner.AuthenticatedRunnerContext:
    return runner.AuthenticatedRunnerContext()


def _first_unit(context: runner.AuthenticatedRunnerContext) -> tuple[str, dict]:
    work_unit_id = context.ordered_work_unit_ids()[0]
    return work_unit_id, context.work_unit_record(work_unit_id)


def _bounded_request(context: runner.AuthenticatedRunnerContext, *, trials: int = 7) -> dict:
    work_unit_id, unit = _first_unit(context)
    return bler_contract.build_bounded_smoke_request(
        work_unit_id=work_unit_id,
        bler_identity=unit["identity"],
        snr_db=unit["snr_db"],
        source_packet_config_ids=unit["source_packet_config_ids"],
        trials_requested=trials,
    )


def test_context_authenticates_registered_candidate_once(auth_context):
    binding = auth_context.runner_contract_binding()
    assert binding["bler_runner_contract_id"].startswith("g8runner-")
    assert len(binding["bler_runner_contract_sha256"]) == 64
    first = auth_context.authority_binding()
    first["campaign_id"] = "mutated"
    assert auth_context.authority_binding()["campaign_id"] != "mutated"
    assert len(auth_context.ordered_work_unit_ids()) == 3213


def test_full_strength_is_rejected_before_root_or_adapter(auth_context, tmp_path, monkeypatch):
    root = tmp_path / "full-runtime"
    work_unit_id, _unit = _first_unit(auth_context)
    called = []

    class ForbiddenAdapter:
        def __init__(self, *args, **kwargs):
            called.append((args, kwargs))
            raise AssertionError("adapter must not be constructed by an unauthorized call")

    monkeypatch.setattr(adapter_module, "SionnaLDPCAdapter", ForbiddenAdapter)
    with pytest.raises(runner.RunnerAuthorizationError, match="full-strength execution"):
        runner.run_one_unit(
            auth_context,
            execution_class=runner.EXECUTION_CLASS_FULL_STRENGTH,
            root=root,
            work_unit_id=work_unit_id,
            shard_count=1,
            shard_index=0,
            batch_size=1,
            device="cpu",
        )
    assert not root.exists()
    assert called == []


def test_bounded_authorization_requires_explicit_fresh_nonproduction_root(auth_context, tmp_path):
    with pytest.raises(runner.RunnerAuthorizationError):
        runner.authorize_execution(
            auth_context,
            runner.EXECUTION_CLASS_BOUNDED_SMOKE,
            root=None,
        )
    with pytest.raises(runner.RunnerAuthorizationError):
        runner.authorize_execution(
            auth_context,
            runner.EXECUTION_CLASS_BOUNDED_SMOKE,
            root=work_units.DEFAULT_WORK_UNIT_ROOT,
        )
    alias = tmp_path / "production-alias"
    alias.symlink_to(work_units.DEFAULT_WORK_UNIT_ROOT, target_is_directory=True)
    with pytest.raises(runner.RunnerAuthorizationError):
        runner.authorize_execution(
            auth_context,
            runner.EXECUTION_CLASS_BOUNDED_SMOKE,
            root=alias,
        )
    existing = tmp_path / "already-existing"
    existing.mkdir()
    with pytest.raises(runner.RunnerAuthorizationError):
        runner.authorize_execution(
            auth_context,
            runner.EXECUTION_CLASS_BOUNDED_SMOKE,
            root=existing,
        )


def test_shard_bounds_and_unknown_execution_class_are_closed(auth_context, tmp_path):
    with pytest.raises(runner.RunnerAuthorizationError, match="shard_index"):
        runner.run_one_unit(
            auth_context,
            execution_class=runner.EXECUTION_CLASS_BOUNDED_SMOKE,
            root=tmp_path / "bad-shard",
            work_unit_id=auth_context.ordered_work_unit_ids()[0],
            shard_count=1,
            shard_index=1,
            batch_size=1,
            device="cpu",
        )
    with pytest.raises(runner.RunnerAuthorizationError, match="unknown execution class"):
        runner.run_one_unit(
            auth_context,
            execution_class="not-a-real-class",
            root=tmp_path / "bad-class",
            work_unit_id=auth_context.ordered_work_unit_ids()[0],
            shard_count=1,
            shard_index=0,
            batch_size=1,
            device="cpu",
        )


class _DeterministicAdapter:
    def __init__(self, k: int, n: int, q_m: int, base_graph: int, device: str = "cpu"):
        del q_m, base_graph, device
        self.k = k
        self.n = n
        self.lifting_size = 352

    def encode(self, bits: np.ndarray) -> np.ndarray:
        encoded = np.zeros((bits.shape[0], self.n), dtype=np.uint8)
        encoded[:, : self.k] = bits
        return encoded

    def decode(self, llr: np.ndarray) -> np.ndarray:
        # The fake is a deterministic adapter seam; it deliberately does not
        # claim a physical-layer result.  Its output is sufficient to prove
        # complete-K counting and batch-boundary invariance.
        return np.zeros((llr.shape[0], self.k), dtype=np.uint8)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 7, 32])
def test_measurement_is_batch_partition_invariant(auth_context, monkeypatch, batch_size):
    request = _bounded_request(auth_context, trials=13)
    monkeypatch.setattr(adapter_module, "SionnaLDPCAdapter", _DeterministicAdapter)
    observed = runner._execute_measurement(request, device="cpu", batch_size=batch_size)
    assert observed["status"] == bler_contract.STATUS_COMPLETE
    assert observed["trials_completed"] == 13
    assert 0 < observed["bit_errors"] < observed["block_errors"] * request["bler_identity"]["k_and_n"][0]
    assert observed["block_errors"] == 13


def test_measurement_rejects_bad_batch_size(auth_context):
    request = _bounded_request(auth_context, trials=2)
    with pytest.raises((ValueError, runner.G8BlerRunnerError)):
        runner._execute_measurement(request, device="cpu", batch_size=0)


def test_immutable_publication_is_idempotent_and_conflict_safe(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    target = root / "aa" / "request.json"
    payload = {"artifact_role": "test", "value": 1}
    digest = runner._publish_immutable_json(target, payload, root=root)
    assert digest == bler_contract.sha256_bytes(canonical_json(payload))
    assert target.read_bytes() == canonical_json(payload)
    assert runner._publish_immutable_json(target, payload, root=root) == digest
    with pytest.raises(runner.RunnerConflictError):
        runner._publish_immutable_json(target, {"artifact_role": "test", "value": 2}, root=root)
    assert not list(root.rglob("*.staging"))


def test_immutable_publication_rejects_symlink_and_hard_link_alias(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    bucket = root / "aa"
    bucket.mkdir()
    symlink_target = tmp_path / "symlink-target"
    symlink_target.write_bytes(b"foreign")
    (bucket / "symlink.json").symlink_to(symlink_target)
    with pytest.raises(runner.RunnerConflictError):
        runner._publish_immutable_json(bucket / "symlink.json", {"x": 1}, root=root)
    hard_target = tmp_path / "hard-target"
    hard_target.write_bytes(b"foreign")
    (bucket / "hard.json").hardlink_to(hard_target)
    with pytest.raises(runner.RunnerConflictError):
        runner._publish_immutable_json(bucket / "hard.json", {"x": 1}, root=root)


def test_one_bounded_unit_uses_claim_request_result_link_transaction(
    auth_context, tmp_path, monkeypatch
):
    monkeypatch.setattr(adapter_module, "SionnaLDPCAdapter", _DeterministicAdapter)
    work_unit_id, _unit = _first_unit(auth_context)
    root = tmp_path / "one-unit"
    outcome = runner.run_one_unit(
        auth_context,
        execution_class=runner.EXECUTION_CLASS_BOUNDED_SMOKE,
        root=root,
        work_unit_id=work_unit_id,
        shard_count=1,
        shard_index=0,
        batch_size=2,
        device="cpu",
    )
    assert outcome["state"]["identity"]["status"] == work_units.STATUS_RESULT_LINKED
    assert outcome["request"]["request"]["execution_class"] == runner.EXECUTION_CLASS_BOUNDED_SMOKE
    assert outcome["result"]["result"]["disposition"]["required_coverage_contribution"] == 0
    assert outcome["measurement"]["trials_completed"] == 16


def test_smoke_record_builder_is_path_and_time_free(auth_context, tmp_path, monkeypatch):
    monkeypatch.setattr(adapter_module, "SionnaLDPCAdapter", _DeterministicAdapter)
    work_unit_id, _unit = _first_unit(auth_context)
    outcome = runner.run_one_unit(
        auth_context,
        execution_class=runner.EXECUTION_CLASS_BOUNDED_SMOKE,
        root=tmp_path / "record-unit",
        work_unit_id=work_unit_id,
        shard_count=1,
        shard_index=0,
        batch_size=4,
        device="cpu",
    )
    record = runner.build_bounded_smoke_record(
        auth_context,
        [outcome],
        shard_count=1,
        shard_index=0,
        batch_size=4,
        production_root_used=False,
        temporary_root_removed=True,
    )
    rendered = canonical_json(record).decode("ascii")
    assert "hostname" not in rendered
    assert "timestamp" not in rendered
    assert str(tmp_path) not in rendered
    assert record["selected_work_units"][0]["required_coverage_contribution"] == 0
