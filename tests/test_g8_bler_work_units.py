"""B2C mutation, partition, path, concurrency, and crash-safety tests.

All publication tests use an isolated temporary work-unit root.  This file
deliberately contains no runner, channel, codec, dataset, classifier, or
test-split path.

Where process *lifetime* is the property under test — exactly-one-winner
races, hard exits before and after publication, and cross-process
linearizability — these tests fork real child processes.  Threads cannot
substitute: only a real ``os._exit`` proves what survives a hard kill.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import errno
import hashlib
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from baseline import g8_bler_contract
from baseline import g8_bler_work_units as units
import gen_g8_bler_state_contract as state_generator
import verify_g8_bler_state_contract as state_verifier
from baseline.g8_campaign import rendered_json, sha256_bytes


RACERS = 8
STRESS_ROUNDS = 5


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def context() -> units.AuthenticatedExecutionContext:
    return units.AuthenticatedExecutionContext()


@pytest.fixture(scope="module")
def state_context(
    context: units.AuthenticatedExecutionContext,
) -> units.AuthenticatedUnitStateContext:
    return units.AuthenticatedUnitStateContext(context)


def _plan(
    context: Any,
    *,
    shard_count: int = 7,
    shard_index: int = 0,
) -> dict[str, Any]:
    return units.build_shard_plan(context, shard_count, shard_index)


def _claim(
    state_context: units.AuthenticatedUnitStateContext,
    root: Path,
    *,
    shard_count: int = 7,
    shard_index: int = 0,
    unit: int = 0,
) -> tuple[dict[str, Any], Path]:
    plan = _plan(state_context, shard_count=shard_count, shard_index=shard_index)
    work_unit_id = plan["assigned_work_unit_ids"][unit]
    state = units.build_unit_state(state_context, work_unit_id, plan)
    return state, units.unit_state_path(state_context, work_unit_id, root=root)


def _linked(
    state_context: units.AuthenticatedUnitStateContext,
    work_unit_id: str,
    plan: dict[str, Any],
    **overrides: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": units.STATUS_RESULT_LINKED,
        "request_sha256": "a" * 64,
        "result_path": "results/baseline/g8/results/unit.json",
        "result_sha256": "b" * 64,
        "scientific_execution_performed": True,
        "trials_completed": 5,
    }
    payload.update(overrides)
    return units.build_unit_state(state_context, work_unit_id, plan, **payload)


def _forced(state: dict[str, Any], **identity_overrides: Any) -> dict[str, Any]:
    """Bypass build-time validation to isolate one transition rule."""

    forced = copy.deepcopy(state)
    forced["identity"].update(identity_overrides)
    forced["identity_sha256"] = units.state_identity_digest(forced)
    return forced


@contextlib.contextmanager
def _forking() -> Any:
    """Fork children that only touch an isolated root and then ``os._exit``."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        yield


def _run_child(work: Any) -> int:
    with _forking():
        pid = os.fork()
    if pid == 0:  # pragma: no cover - child process
        status = 0
        try:
            work()
        except units.StaleWriterError:
            status = 3
        except units.StateConflictError:
            status = 4
        except BaseException:
            status = 1
        os._exit(status)
    _pid, raw_status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(raw_status)


def _run_children(count: int, work: Any) -> list[int]:
    """Start ``count`` children that all block until one shared release."""

    read_fd, write_fd = os.pipe()
    children = []
    for _ in range(count):
        with _forking():
            pid = os.fork()
        if pid == 0:  # pragma: no cover - child process
            status = 0
            try:
                os.read(read_fd, 1)
                work()
            except units.StaleWriterError:
                status = 3
            except units.StateConflictError:
                status = 4
            except BaseException:
                status = 1
            os._exit(status)
        children.append(pid)
    os.write(write_fd, b"g" * count)
    codes = []
    for pid in children:
        _pid, raw_status = os.waitpid(pid, 0)
        codes.append(os.waitstatus_to_exitcode(raw_status))
    os.close(read_fd)
    os.close(write_fd)
    return codes


def _staged_contract(tmp_path: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    """Write a contract plus a campaign state that registers exactly its bytes."""

    body = rendered_json(payload)
    contract_path = tmp_path / "bler_state_contract.json"
    contract_path.write_bytes(body)
    state = json.loads(units.DEFAULT_CAMPAIGN_STATE_PATH.read_bytes())
    for entry in state["identity"]["produced_artifacts"]:
        if entry["path"] == units.STATE_CONTRACT_REPO_RELATIVE_PATH:
            entry["sha256"] = sha256_bytes(body)
            entry["bytes"] = len(body)
    state_path = tmp_path / "campaign_state.json"
    state_path.write_bytes(rendered_json(state))
    return contract_path, state_path


class _CloseTracker:
    """Detect any descriptor integer closed twice without an intervening open."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.double_closed: list[int] = []
        self._closed: set[int] = set()
        real_close = os.close
        real_open = os.open

        def close(descriptor: int) -> None:
            if descriptor in self._closed:
                self.double_closed.append(descriptor)
            self._closed.add(descriptor)
            return real_close(descriptor)

        def open_(*args: Any, **kwargs: Any) -> int:
            descriptor = real_open(*args, **kwargs)
            self._closed.discard(descriptor)
            return descriptor

        monkeypatch.setattr(os, "close", close)
        monkeypatch.setattr(os, "open", open_)


# ---------------------------------------------------------------------------
# Authenticated authority layers
# ---------------------------------------------------------------------------


def test_context_authenticates_frozen_bindings_and_full_authority(
    context: units.AuthenticatedExecutionContext,
) -> None:
    assert context.authority_binding() == {
        "campaign_id": units.EXPECTED_CAMPAIGN_ID,
        "campaign_manifest_sha256": units.EXPECTED_CAMPAIGN_MANIFEST_SHA256,
        "required_bler_artifact_sha256": units.EXPECTED_REQUIRED_IDENTITIES_SHA256,
        "selection_policy_sha256": units.EXPECTED_SELECTION_POLICY_SHA256,
        "bler_tooling_contract_id": units.EXPECTED_B1C_CONTRACT_ID,
        "bler_tooling_contract_sha256": units.EXPECTED_B1C_CONTRACT_SHA256,
        "tooling_schema_version": units.B1C_TOOLING_SCHEMA_VERSION,
        "request_schema_version": units.B1C_REQUEST_SCHEMA_VERSION,
        "result_schema_version": units.B1C_RESULT_SCHEMA_VERSION,
        "required_work_unit_count": units.EXPECTED_REQUIRED_WORK_UNIT_COUNT,
    }
    assert len(context.ordered_work_unit_ids) == units.EXPECTED_REQUIRED_WORK_UNIT_COUNT


def test_context_exposes_fresh_mutation_safe_records(
    context: units.AuthenticatedExecutionContext,
) -> None:
    work_unit_id = context.work_unit_ids[0]
    first = context.work_unit_record(work_unit_id)
    first["work_unit_id"] = "tampered"
    assert context.work_unit_record(work_unit_id)["work_unit_id"] == work_unit_id
    index = context.work_unit_index()
    index.clear()
    assert len(context.work_unit_index()) == units.EXPECTED_REQUIRED_WORK_UNIT_COUNT
    binding = context.authority_binding()
    binding["campaign_id"] = "tampered"
    assert context.campaign_id == units.EXPECTED_CAMPAIGN_ID


def test_unit_state_context_binds_the_registered_b2c_contract(
    state_context: units.AuthenticatedUnitStateContext,
) -> None:
    body = units.DEFAULT_STATE_CONTRACT_PATH.read_bytes()
    payload = json.loads(body)
    assert payload["checkpoint"] == "B2C"
    assert payload["schema_version"] == units.STATE_CONTRACT_SCHEMA_VERSION
    assert state_context.state_contract_id == payload["contract_id"]
    assert state_context.state_contract_sha256 == sha256_bytes(body)
    assert state_context.state_contract_id != units.SUPERSEDED_STATE_CONTRACT_ID
    assert state_context.state_contract_sha256 != units.SUPERSEDED_STATE_CONTRACT_SHA256


def test_contract_supersedes_the_exact_b2_contract() -> None:
    payload = json.loads(units.DEFAULT_STATE_CONTRACT_PATH.read_bytes())
    assert payload["supersedes"] == {
        "checkpoint": "B2",
        "contract_id": units.SUPERSEDED_STATE_CONTRACT_ID,
        "contract_sha256": units.SUPERSEDED_STATE_CONTRACT_SHA256,
        "contract_bytes": units.SUPERSEDED_STATE_CONTRACT_BYTES,
        "reason": payload["supersedes"]["reason"],
        "states_written_under_the_superseded_contract": 0,
        "per_unit_migration_required": False,
    }
    assert sha256_bytes(units.DEFAULT_STATE_CONTRACT_PATH.read_bytes()) not in json.dumps(payload)


def test_unregistered_state_contract_is_rejected(tmp_path: Path) -> None:
    payload = state_generator.build()
    contract_path, state_path = _staged_contract(tmp_path, payload)
    # Correct pair authenticates.
    units.AuthenticatedUnitStateContext(
        campaign_state_path=state_path, state_contract_path=contract_path
    )
    # The same artifact against the *production* registration does not, because
    # the registered byte count and SHA-256 no longer match.
    mutated = copy.deepcopy(payload)
    mutated["scope"]["runner_exists"] = True
    mutated["contract_id"] = state_generator.contract_identifier(mutated)
    contract_path.write_bytes(rendered_json(mutated))
    with pytest.raises(units.StateContractAuthenticationError):
        units.AuthenticatedUnitStateContext(
            campaign_state_path=state_path, state_contract_path=contract_path
        )


def test_superseded_b2_contract_is_rejected(tmp_path: Path) -> None:
    payload = state_generator.build()
    payload["contract_id"] = units.SUPERSEDED_STATE_CONTRACT_ID
    contract_path, state_path = _staged_contract(tmp_path, payload)
    with pytest.raises(units.StateContractAuthenticationError):
        units.AuthenticatedUnitStateContext(
            campaign_state_path=state_path, state_contract_path=contract_path
        )


@pytest.mark.parametrize(
    "operation",
    [
        lambda ctx, state, path, root: units.build_unit_state(
            ctx, _plan(ctx)["assigned_work_unit_ids"][0], _plan(ctx)
        ),
        lambda ctx, state, path, root: units.validate_unit_state(ctx, state),
        lambda ctx, state, path, root: units.canonical_state_bytes(ctx, state),
        lambda ctx, state, path, root: units.read_unit_state(ctx, path, root=root),
        lambda ctx, state, path, root: units.create_unit_state_exclusive(ctx, state, root=root),
        lambda ctx, state, path, root: units.replace_unit_state(ctx, path, state, "0" * 64, root=root),
    ],
)
def test_plain_execution_context_is_rejected_for_unit_state_operations(
    context: units.AuthenticatedExecutionContext,
    state_context: units.AuthenticatedUnitStateContext,
    tmp_path: Path,
    operation: Any,
) -> None:
    state, path = _claim(state_context, tmp_path)
    with pytest.raises(units.UnitStateContextRequiredError):
        operation(context, state, path, tmp_path)


def test_wrong_b1c_tooling_binding_fails_closed(tmp_path: Path) -> None:
    forged = tmp_path / "bler_tooling_contract.json"
    payload = json.loads(units.DEFAULT_STATE_CONTRACT_PATH.read_bytes())
    forged.write_bytes(rendered_json(payload))
    with pytest.raises(units.AuthorityAuthenticationError):
        units.AuthenticatedExecutionContext(tooling_contract_path=forged)


# ---------------------------------------------------------------------------
# Deterministic ordinal-modulo sharding
# ---------------------------------------------------------------------------


def test_one_shard_preserves_all_canonical_ids(
    context: units.AuthenticatedExecutionContext,
) -> None:
    plan = _plan(context, shard_count=1, shard_index=0)
    assert plan["assigned_work_unit_ids"] == list(context.ordered_work_unit_ids)
    assert plan["assigned_work_unit_count"] == units.EXPECTED_REQUIRED_WORK_UNIT_COUNT


@pytest.mark.parametrize("count", [1, 2, 3, 7, 11, 32, 64, 127])
def test_ordinal_modulo_partition_is_complete_disjoint_and_balanced(
    context: units.AuthenticatedExecutionContext,
    count: int,
) -> None:
    ids = list(context.ordered_work_unit_ids)
    seen: list[str] = []
    sizes = []
    for index in range(count):
        assigned = _plan(context, shard_count=count, shard_index=index)["assigned_work_unit_ids"]
        assert assigned == [unit for ordinal, unit in enumerate(ids) if ordinal % count == index]
        sizes.append(len(assigned))
        seen.extend(assigned)
    assert sorted(seen) == sorted(ids)
    assert len(seen) == len(set(seen))
    assert max(sizes) - min(sizes) <= 1


def test_shard_plan_is_byte_identical_and_does_not_depend_on_mapping_order(
    context: units.AuthenticatedExecutionContext,
) -> None:
    first = _plan(context, shard_count=5, shard_index=2)
    second = units.build_shard_plan(
        units.AuthenticatedExecutionContext(), 5, 2
    )
    assert units.shard_plan_bytes(first) == units.shard_plan_bytes(second)
    assert units.shard_plan_digest(first) == first["plan_digest"]


@pytest.mark.parametrize("count", [0, -1, True, 1.0, "3", None])
def test_invalid_shard_counts_are_rejected(
    context: units.AuthenticatedExecutionContext,
    count: Any,
) -> None:
    with pytest.raises(units.ShardPlanError):
        units.build_shard_plan(context, count, 0)


@pytest.mark.parametrize("index", [-1, True, 1.0, "0", None])
def test_invalid_shard_indices_are_rejected(
    context: units.AuthenticatedExecutionContext,
    index: Any,
) -> None:
    with pytest.raises(units.ShardPlanError):
        units.build_shard_plan(context, 3, index)


def test_out_of_range_shard_index_is_rejected(
    context: units.AuthenticatedExecutionContext,
) -> None:
    with pytest.raises(units.ShardPlanError):
        units.build_shard_plan(context, 3, 3)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.__setitem__("shard_index", 1),
        lambda p: p.__setitem__("assigned_work_unit_count", 0),
        lambda p: p["assigned_work_unit_ids"].reverse(),
        lambda p: p["assigned_work_unit_ids"].pop(),
        lambda p: p.__setitem__("sharding_algorithm", "hash_v2"),
        lambda p: p.__setitem__("campaign_id", "other"),
        lambda p: p.__setitem__("plan_digest", "0" * 64),
        lambda p: p.pop("plan_digest"),
        lambda p: p.__setitem__("unexpected", 1),
    ],
)
def test_shard_plan_mutations_fail_closed(
    context: units.AuthenticatedExecutionContext,
    mutation: Any,
) -> None:
    plan = _plan(context, shard_count=3, shard_index=0)
    mutated = copy.deepcopy(plan)
    mutation(mutated)
    with pytest.raises(units.ShardPlanError):
        units.validate_shard_plan(context, mutated)


def test_shard_layout_never_enters_seed_derivation(
    context: units.AuthenticatedExecutionContext,
) -> None:
    work_unit_id = context.work_unit_ids[100]
    seeds = {
        purpose: context.seed(work_unit_id, purpose)
        for purpose in g8_bler_contract.SEED_PURPOSES
    }
    assert _plan(context, shard_count=17, shard_index=100 % 17)["assigned_work_unit_ids"]
    assert seeds == {
        purpose: context.seed(work_unit_id, purpose)
        for purpose in g8_bler_contract.SEED_PURPOSES
    }


# ---------------------------------------------------------------------------
# Safe paths and no-follow inspection
# ---------------------------------------------------------------------------


def test_safe_path_is_digest_derived_and_id_bound(
    context: units.AuthenticatedExecutionContext,
    tmp_path: Path,
) -> None:
    work_unit_id = context.work_unit_ids[0]
    digest = hashlib.sha256(work_unit_id.encode("utf-8")).hexdigest()
    path = units.unit_state_path(context, work_unit_id, root=tmp_path)
    assert path == tmp_path / digest[:2] / f"{digest}.state.json"
    assert units.validate_unit_state_path(context, path, work_unit_id, root=tmp_path) == path
    with pytest.raises(units.UnsafeUnitStatePathError):
        units.validate_unit_state_path(context, path, context.work_unit_ids[1], root=tmp_path)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda p: p.parent.parent / "outside" / p.name,
        lambda p: p.parent / ".." / p.parent.name / p.name,
        lambda p: p.with_name(p.name.removesuffix(".state.json") + ".state"),
        lambda p: p.parent / p.name.upper(),
        lambda p: p.parent.parent / "00" / p.name,
    ],
)
def test_unsafe_paths_fail_closed(
    context: units.AuthenticatedExecutionContext,
    tmp_path: Path,
    mutator: Any,
) -> None:
    work_unit_id = context.work_unit_ids[0]
    path = units.unit_state_path(context, work_unit_id, root=tmp_path)
    with pytest.raises(units.UnsafeUnitStatePathError):
        units.validate_unit_state_path(context, mutator(path), work_unit_id, root=tmp_path)


def test_relative_absolute_alias_and_normalized_paths_fail(
    context: units.AuthenticatedExecutionContext,
    tmp_path: Path,
) -> None:
    work_unit_id = context.work_unit_ids[0]
    path = units.unit_state_path(context, work_unit_id, root=tmp_path)
    with pytest.raises(units.UnsafeUnitStatePathError):
        units.validate_unit_state_path(
            context, str(path.relative_to(tmp_path)), work_unit_id, root=tmp_path
        )
    with pytest.raises(units.UnsafeUnitStatePathError):
        units.validate_unit_state_path(
            context,
            str(path).replace(str(tmp_path), str(tmp_path / "missing")),
            work_unit_id,
            root=tmp_path,
        )


def test_symlink_escape_is_rejected(
    context: units.AuthenticatedExecutionContext,
    tmp_path: Path,
) -> None:
    work_unit_id = context.work_unit_ids[0]
    path = units.unit_state_path(context, work_unit_id, root=tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    path.parent.symlink_to(outside, target_is_directory=True)
    with pytest.raises(units.UnsafeUnitStatePathError):
        units.validate_unit_state_path(context, path, work_unit_id, root=tmp_path)


def test_dangling_symlinks_are_detected_rather_than_reported_absent(
    context: units.AuthenticatedExecutionContext,
    tmp_path: Path,
) -> None:
    work_unit_id = context.work_unit_ids[0]
    path = units.unit_state_path(context, work_unit_id, root=tmp_path)
    path.parent.mkdir(parents=True)
    path.symlink_to(tmp_path / "does-not-exist")
    # This is exactly the fail-open case: exists() follows the link and is False.
    assert not path.exists()
    assert path.is_symlink()
    with pytest.raises(units.UnsafeUnitStatePathError):
        units.validate_unit_state_path(context, path, work_unit_id, root=tmp_path)


def test_dangling_symlink_root_is_rejected(
    context: units.AuthenticatedExecutionContext,
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(units.UnsafeUnitStatePathError):
        units.unit_state_path(context, context.work_unit_ids[0], root=root)


def test_non_directory_parent_component_is_rejected(
    context: units.AuthenticatedExecutionContext,
    tmp_path: Path,
) -> None:
    work_unit_id = context.work_unit_ids[0]
    path = units.unit_state_path(context, work_unit_id, root=tmp_path)
    path.parent.write_bytes(b"not a directory")
    with pytest.raises(units.UnsafeUnitStatePathError):
        units.validate_unit_state_path(context, path, work_unit_id, root=tmp_path)


# ---------------------------------------------------------------------------
# Closed state schema and B2C contract binding
# ---------------------------------------------------------------------------


def test_state_identity_binds_the_registered_state_contract(
    state_context: units.AuthenticatedUnitStateContext,
) -> None:
    plan = _plan(state_context, shard_count=3, shard_index=0)
    state = units.build_unit_state(state_context, plan["assigned_work_unit_ids"][0], plan)
    assert set(state["identity"]) == set(units.UNIT_STATE_IDENTITY_FIELDS)
    assert state["schema_version"] == units.UNIT_STATE_SCHEMA_VERSION == 2
    assert state["identity"]["bler_state_contract_id"] == state_context.state_contract_id
    assert state["identity"]["bler_state_contract_sha256"] == state_context.state_contract_sha256


@pytest.mark.parametrize(
    "field,value",
    [
        ("bler_state_contract_id", units.SUPERSEDED_STATE_CONTRACT_ID),
        ("bler_state_contract_sha256", units.SUPERSEDED_STATE_CONTRACT_SHA256),
        ("bler_state_contract_id", "g8state-" + "0" * 64),
        ("bler_state_contract_sha256", "0" * 64),
    ],
)
def test_state_with_wrong_state_contract_binding_is_rejected(
    state_context: units.AuthenticatedUnitStateContext,
    field: str,
    value: str,
) -> None:
    plan = _plan(state_context, shard_count=3, shard_index=0)
    state = units.build_unit_state(state_context, plan["assigned_work_unit_ids"][0], plan)
    with pytest.raises(units.UnitStateError):
        units.validate_unit_state(state_context, _forced(state, **{field: value}))


def test_claim_failed_and_result_linked_states_are_closed(
    state_context: units.AuthenticatedUnitStateContext,
) -> None:
    plan = _plan(state_context, shard_count=3, shard_index=0)
    work_unit_id = plan["assigned_work_unit_ids"][0]
    claim = units.build_unit_state(state_context, work_unit_id, plan)
    failed = units.build_unit_state(state_context, work_unit_id, plan, status=units.STATUS_FAILED)
    linked = _linked(state_context, work_unit_id, plan)
    assert claim["identity"]["status"] == units.STATUS_CLAIMED
    assert failed["identity"]["status"] == units.STATUS_FAILED
    assert linked["identity"]["status"] == units.STATUS_RESULT_LINKED
    for state in (claim, failed, linked):
        assert units.validate_unit_state(state_context, state) == state


def test_result_linked_requires_a_bound_request_sha(
    state_context: units.AuthenticatedUnitStateContext,
) -> None:
    plan = _plan(state_context, shard_count=3, shard_index=0)
    work_unit_id = plan["assigned_work_unit_ids"][0]
    with pytest.raises(units.UnitStateError):
        _linked(state_context, work_unit_id, plan, request_sha256=None)
    linked = _linked(state_context, work_unit_id, plan)
    with pytest.raises(units.UnitStateError):
        units.validate_unit_state(state_context, _forced(linked, request_sha256=None))


@pytest.mark.parametrize("value", ["A" * 64, "z" * 64, "abc", 1, ""])
def test_request_sha_must_be_lowercase_hexadecimal(
    state_context: units.AuthenticatedUnitStateContext,
    value: Any,
) -> None:
    plan = _plan(state_context, shard_count=3, shard_index=0)
    with pytest.raises(units.UnitStateError):
        _linked(state_context, plan["assigned_work_unit_ids"][0], plan, request_sha256=value)


def test_no_result_may_exist_without_a_request_binding(
    state_context: units.AuthenticatedUnitStateContext,
) -> None:
    plan = _plan(state_context, shard_count=3, shard_index=0)
    work_unit_id = plan["assigned_work_unit_ids"][0]
    claim = units.build_unit_state(state_context, work_unit_id, plan)
    orphaned = _forced(
        claim,
        status=units.STATUS_FAILED,
        result_path="results/baseline/g8/results/unit.json",
        result_sha256="b" * 64,
    )
    with pytest.raises(units.UnitStateError):
        units.validate_unit_state(state_context, orphaned)


def test_state_identity_digest_excludes_runtime_metadata(
    state_context: units.AuthenticatedUnitStateContext,
) -> None:
    plan = _plan(state_context, shard_count=3, shard_index=0)
    work_unit_id = plan["assigned_work_unit_ids"][0]
    bare = units.build_unit_state(state_context, work_unit_id, plan)
    annotated = units.build_unit_state(
        state_context,
        work_unit_id,
        plan,
        runtime_metadata={
            "hostname": "host",
            "process_id": 17,
            "device": "cpu",
            "wall_clock_annotation": "2026-08-02T00:00:00Z",
            "update_annotation": "test",
        },
    )
    assert units.state_identity_digest(bare) == units.state_identity_digest(annotated)
    assert units.unit_state_sha256(state_context, bare) != units.unit_state_sha256(
        state_context, annotated
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.pop("identity_sha256"),
        lambda s: s.__setitem__("unexpected", True),
        lambda s: s.__setitem__("schema_version", 1),
        lambda s: s.__setitem__("artifact_role", "other"),
        lambda s: s["identity"].__setitem__("campaign_id", "other-campaign"),
        lambda s: s["identity"].__setitem__("request_schema_version", 1),
        lambda s: s["identity"].__setitem__("result_schema_version", 1),
        lambda s: s["identity"].__setitem__("canonical_ordinal", 1),
        lambda s: s["identity"].__setitem__("required_work_unit_record_sha256", "0" * 64),
        lambda s: s["identity"].__setitem__("shard_plan_digest", "0" * 64),
        lambda s: s["identity"].pop("bler_state_contract_id"),
        lambda s: s.__setitem__("identity_sha256", "0" * 64),
    ],
)
def test_state_identity_mutations_fail_closed(
    state_context: units.AuthenticatedUnitStateContext,
    mutation: Any,
) -> None:
    plan = _plan(state_context, shard_count=3, shard_index=0)
    state = units.build_unit_state(state_context, plan["assigned_work_unit_ids"][0], plan)
    mutated = copy.deepcopy(state)
    mutation(mutated)
    with pytest.raises(units.UnitStateError):
        units.validate_unit_state(state_context, mutated)


@pytest.mark.parametrize(
    "field,value",
    [
        ("attempt", True),
        ("attempt", 0),
        ("attempt", -1),
        ("trials_completed", True),
        ("trials_completed", -1),
        ("test_split_access", True),
        ("test_split_access", 1),
        ("scientific_execution_performed", 1),
        ("process_id", float("nan")),
        ("process_id", float("inf")),
        ("process_id", -1),
    ],
)
def test_state_numeric_and_boolean_mutations_fail_closed(
    state_context: units.AuthenticatedUnitStateContext,
    field: str,
    value: Any,
) -> None:
    plan = _plan(state_context, shard_count=3, shard_index=0)
    state = units.build_unit_state(state_context, plan["assigned_work_unit_ids"][0], plan)
    mutated = copy.deepcopy(state)
    target = mutated["runtime_metadata"] if field == "process_id" else mutated["identity"]
    target[field] = value
    with pytest.raises(units.UnitStateError):
        units.validate_unit_state(state_context, mutated)


def test_status_contradictions_fail_closed(
    state_context: units.AuthenticatedUnitStateContext,
) -> None:
    plan = _plan(state_context, shard_count=3, shard_index=0)
    work_unit_id = plan["assigned_work_unit_ids"][0]
    claim = units.build_unit_state(state_context, work_unit_id, plan)
    failed = units.build_unit_state(state_context, work_unit_id, plan, status=units.STATUS_FAILED)
    for mutated in (
        _forced(claim, trials_completed=1),
        _forced(
            failed,
            request_sha256="a" * 64,
            result_path="results/x.json",
            result_sha256="a" * 64,
        ),
        _forced(claim, status=units.STATUS_RESULT_LINKED),
    ):
        with pytest.raises(units.UnitStateError):
            units.validate_unit_state(state_context, mutated)


# ---------------------------------------------------------------------------
# Transition legality
# ---------------------------------------------------------------------------


def test_next_attempt_clean_claim_may_reshard(
    state_context: units.AuthenticatedUnitStateContext,
) -> None:
    plan = _plan(state_context, shard_count=7, shard_index=0)
    work_unit_id = plan["assigned_work_unit_ids"][0]
    failed = units.build_unit_state(state_context, work_unit_id, plan, status=units.STATUS_FAILED)
    index = next(
        i
        for i in range(11)
        if work_unit_id in _plan(state_context, shard_count=11, shard_index=i)["assigned_work_unit_ids"]
    )
    other_plan = _plan(state_context, shard_count=11, shard_index=index)
    retry = units.build_unit_state(state_context, work_unit_id, other_plan, attempt=2)
    assert retry["identity"]["shard_count"] == 11
    units.validate_state_transition(failed, retry)


def test_illegal_transitions_are_rejected(
    state_context: units.AuthenticatedUnitStateContext,
) -> None:
    plan = _plan(state_context, shard_count=7, shard_index=0)
    work_unit_id = plan["assigned_work_unit_ids"][0]
    claim = units.build_unit_state(state_context, work_unit_id, plan)
    failed = units.build_unit_state(state_context, work_unit_id, plan, status=units.STATUS_FAILED)
    linked = _linked(state_context, work_unit_id, plan)
    index = next(
        i
        for i in range(11)
        if work_unit_id in _plan(state_context, shard_count=11, shard_index=i)["assigned_work_unit_ids"]
    )
    other_plan = _plan(state_context, shard_count=11, shard_index=index)

    cases = {
        "changed result sha": (linked, _linked(state_context, work_unit_id, plan, result_sha256="c" * 64)),
        "changed result path": (
            linked,
            _linked(state_context, work_unit_id, plan, result_path="results/other.json"),
        ),
        "result regression": (linked, claim),
        "result-linked reassignment": (
            linked,
            units.build_unit_state(state_context, work_unit_id, other_plan, attempt=2),
        ),
        "failed to claimed": (failed, claim),
        "failed to result-linked": (failed, linked),
        "same-attempt reshard": (
            claim,
            units.build_unit_state(state_context, work_unit_id, other_plan),
        ),
        "attempt skip": (
            failed,
            units.build_unit_state(state_context, work_unit_id, plan, attempt=3),
        ),
        "attempt regression": (
            units.build_unit_state(state_context, work_unit_id, plan, attempt=2, status=units.STATUS_FAILED),
            failed,
        ),
        "dirty next attempt": (
            failed,
            _forced(
                units.build_unit_state(state_context, work_unit_id, plan, attempt=2),
                request_sha256="a" * 64,
            ),
        ),
        "decreased trials": (
            _linked(state_context, work_unit_id, plan, status=units.STATUS_FAILED,
                    result_path=None, result_sha256=None, trials_completed=9),
            _linked(state_context, work_unit_id, plan, status=units.STATUS_FAILED,
                    result_path=None, result_sha256=None, trials_completed=3),
        ),
        "changed request sha": (
            _linked(state_context, work_unit_id, plan, status=units.STATUS_FAILED,
                    result_path=None, result_sha256=None, trials_completed=9),
            _linked(state_context, work_unit_id, plan, status=units.STATUS_FAILED,
                    result_path=None, result_sha256=None, trials_completed=9,
                    request_sha256="c" * 64),
        ),
        "request sha cleared": (
            _linked(state_context, work_unit_id, plan, status=units.STATUS_FAILED,
                    result_path=None, result_sha256=None, trials_completed=9),
            _forced(
                _linked(state_context, work_unit_id, plan, status=units.STATUS_FAILED,
                        result_path=None, result_sha256=None, trials_completed=9),
                request_sha256=None,
            ),
        ),
        "scientific flag demoted": (
            _linked(state_context, work_unit_id, plan, status=units.STATUS_FAILED,
                    result_path=None, result_sha256=None, trials_completed=9),
            _forced(
                _linked(state_context, work_unit_id, plan, status=units.STATUS_FAILED,
                        result_path=None, result_sha256=None, trials_completed=9),
                scientific_execution_performed=False,
            ),
        ),
    }
    for label, (previous, proposed) in cases.items():
        with pytest.raises(units.StateConflictError):
            units.validate_state_transition(previous, proposed)
            pytest.fail(f"{label} was accepted")


# ---------------------------------------------------------------------------
# Crash-atomic first publication
# ---------------------------------------------------------------------------


def test_first_publication_never_opens_the_final_pathname_for_writing() -> None:
    source = (Path(__file__).parents[1] / "src/baseline/g8_bler_work_units.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    creator = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "create_unit_state_exclusive"
    )
    published = {
        node.func.attr
        for node in ast.walk(creator)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "link" not in published  # publication is delegated, not inlined
    assert "_publish_without_replace" in {
        node.func.id
        for node in ast.walk(creator)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "os.link(" in source and "follow_symlinks=False" in source


def test_exclusive_create_installs_complete_canonical_bytes(
    state_context: units.AuthenticatedUnitStateContext,
    tmp_path: Path,
) -> None:
    state, path = _claim(state_context, tmp_path)
    sha = units.create_unit_state_exclusive(state_context, state, root=tmp_path)
    body = path.read_bytes()
    assert sha == hashlib.sha256(body).hexdigest()
    assert body == units.canonical_state_bytes(state_context, state)
    assert not list(path.parent.glob(f"*{units.STAGING_FILENAME_SUFFIX}"))
    assert units.read_unit_state(state_context, path, root=tmp_path) == state


def test_eight_simultaneous_creator_processes_have_exactly_one_winner(
    state_context: units.AuthenticatedUnitStateContext,
    tmp_path: Path,
) -> None:
    state, path = _claim(state_context, tmp_path)
    path.parent.mkdir(parents=True)

    def create() -> None:  # pragma: no cover - child process
        units.create_unit_state_exclusive(state_context, state, root=tmp_path)

    codes = _run_children(RACERS, create)
    assert codes.count(0) == 1
    assert codes.count(4) == RACERS - 1
    assert units.read_unit_state(state_context, path, root=tmp_path) == state


def test_threads_in_one_process_cannot_both_create(
    state_context: units.AuthenticatedUnitStateContext,
    tmp_path: Path,
) -> None:
    state, path = _claim(state_context, tmp_path)

    def attempt() -> str:
        return units.create_unit_state_exclusive(state_context, state, root=tmp_path)

    with ThreadPoolExecutor(max_workers=RACERS) as executor:
        futures = [executor.submit(attempt) for _ in range(RACERS)]
    successes = 0
    conflicts = 0
    for future in futures:
        try:
            future.result()
            successes += 1
        except units.StateConflictError:
            conflicts += 1
    assert successes == 1
    assert conflicts == RACERS - 1


@pytest.mark.parametrize("occupant", ["regular", "dangling", "symlink", "directory"])
def test_preexisting_object_at_the_final_name_is_rejected(
    state_context: units.AuthenticatedUnitStateContext,
    tmp_path: Path,
    occupant: str,
) -> None:
    state, path = _claim(state_context, tmp_path)
    path.parent.mkdir(parents=True)
    if occupant == "regular":
        path.write_bytes(b"squatter")
    elif occupant == "dangling":
        path.symlink_to(tmp_path / "missing")
    elif occupant == "symlink":
        (tmp_path / "real").write_bytes(b"x")
        path.symlink_to(tmp_path / "real")
    else:
        path.mkdir()
    with pytest.raises((units.StateConflictError, units.UnsafeUnitStatePathError)):
        units.create_unit_state_exclusive(state_context, state, root=tmp_path)


def test_hard_exit_before_publication_leaves_the_final_path_absent(
    state_context: units.AuthenticatedUnitStateContext,
    tmp_path: Path,
) -> None:
    state, path = _claim(state_context, tmp_path)

    def die() -> None:  # pragma: no cover - child process
        units._publish_without_replace = lambda *a, **k: os._exit(9)
        units.create_unit_state_exclusive(state_context, state, root=tmp_path)

    assert _run_child(die) == 9
    assert not path.exists() and not path.is_symlink()
    # The retry succeeds despite any staging artifact the dead writer left.
    units.create_unit_state_exclusive(state_context, state, root=tmp_path)
    assert units.read_unit_state(state_context, path, root=tmp_path) == state


def test_hard_exit_after_publication_leaves_complete_canonical_bytes(
    state_context: units.AuthenticatedUnitStateContext,
    tmp_path: Path,
) -> None:
    state, path = _claim(state_context, tmp_path)

    def die() -> None:  # pragma: no cover - child process
        units._fsync_directory_descriptor = lambda *a, **k: os._exit(9)
        units.create_unit_state_exclusive(state_context, state, root=tmp_path)

    assert _run_child(die) == 9
    body = path.read_bytes()
    assert json.loads(body)  # never partial JSON
    assert units.read_unit_state(state_context, path, root=tmp_path) == state


def test_orphan_staging_artifacts_are_not_state_and_do_not_block_a_retry(
    state_context: units.AuthenticatedUnitStateContext,
    tmp_path: Path,
) -> None:
    state, path = _claim(state_context, tmp_path)
    path.parent.mkdir(parents=True)
    orphan = path.parent / f".{path.name}.1234.abcdef{units.STAGING_FILENAME_SUFFIX}"
    orphan.write_bytes(b'{"partial":')
    units.create_unit_state_exclusive(state_context, state, root=tmp_path)
    assert units.read_unit_state(state_context, path, root=tmp_path) == state
    assert orphan.exists()
    with pytest.raises(units.UnsafeUnitStatePathError):
        units.read_unit_state(state_context, orphan, root=tmp_path)


# ---------------------------------------------------------------------------
# Linearizable replacement
# ---------------------------------------------------------------------------


def test_replacement_is_serialised_under_an_exclusive_per_unit_lock(
    state_context: units.AuthenticatedUnitStateContext,
    tmp_path: Path,
) -> None:
    state, path = _claim(state_context, tmp_path)
    units.create_unit_state_exclusive(state_context, state, root=tmp_path)
    lock_dir = tmp_path / units.LOCK_DIRECTORY_NAME
    digest = path.name.removesuffix(".state.json")
    with units.unit_state_lock(tmp_path, digest):
        pass
    assert (lock_dir / f"{digest}{units.LOCK_FILENAME_SUFFIX}").exists()
    assert not lock_dir.name[:2].isalnum() or lock_dir.name.startswith(".")


@pytest.mark.parametrize("round_index", range(STRESS_ROUNDS))
def test_two_processes_with_the_same_predecessor_cannot_both_succeed(
    state_context: units.AuthenticatedUnitStateContext,
    tmp_path: Path,
    round_index: int,
) -> None:
    root = tmp_path / f"round{round_index}"
    state, path = _claim(state_context, root)
    previous = units.create_unit_state_exclusive(state_context, state, root=root)
    work_unit_id = state["identity"]["work_unit_id"]
    plan = _plan(state_context)
    proposals = [
        units.build_unit_state(state_context, work_unit_id, plan, status=units.STATUS_FAILED),
        _linked(state_context, work_unit_id, plan),
    ]
    bodies = [units.canonical_state_bytes(state_context, proposal) for proposal in proposals]

    read_fd, write_fd = os.pipe()
    children = []
    for proposal in proposals:
        with _forking():
            pid = os.fork()
        if pid == 0:  # pragma: no cover - child process
            status = 0
            try:
                os.read(read_fd, 1)
                units.replace_unit_state(state_context, path, proposal, previous, root=root)
            except units.StaleWriterError:
                status = 3
            except units.StateConflictError:
                status = 4
            except BaseException:
                status = 1
            os._exit(status)
        children.append(pid)
    os.write(write_fd, b"gg")
    codes = []
    for pid in children:
        _pid, raw = os.waitpid(pid, 0)
        codes.append(os.waitstatus_to_exitcode(raw))
    os.close(read_fd)
    os.close(write_fd)

    assert codes.count(0) == 1, f"exactly one writer must win, got {codes}"
    assert codes.count(3) == 1, f"the loser must observe the winner, got {codes}"
    assert path.read_bytes() == bodies[codes.index(0)]


def test_stale_writer_cannot_overwrite_newer_state(
    state_context: units.AuthenticatedUnitStateContext,
    tmp_path: Path,
) -> None:
    state, path = _claim(state_context, tmp_path)
    old_sha = units.create_unit_state_exclusive(state_context, state, root=tmp_path)
    plan = _plan(state_context)
    work_unit_id = state["identity"]["work_unit_id"]
    newer = units.build_unit_state(state_context, work_unit_id, plan, status=units.STATUS_FAILED)
    new_sha = units.replace_unit_state(state_context, path, newer, old_sha, root=tmp_path)
    before = path.read_bytes()
    with pytest.raises(units.StaleWriterError):
        units.replace_unit_state(state_context, path, state, old_sha, root=tmp_path)
    assert path.read_bytes() == before
    assert new_sha == hashlib.sha256(before).hexdigest()


def test_result_linked_state_is_terminal_even_with_the_current_sha(
    state_context: units.AuthenticatedUnitStateContext,
    tmp_path: Path,
) -> None:
    state, path = _claim(state_context, tmp_path)
    previous = units.create_unit_state_exclusive(state_context, state, root=tmp_path)
    plan = _plan(state_context)
    work_unit_id = state["identity"]["work_unit_id"]
    linked = _linked(state_context, work_unit_id, plan)
    linked_sha = units.replace_unit_state(state_context, path, linked, previous, root=tmp_path)
    before = path.read_bytes()

    # Exact canonical-byte idempotence is the only permitted operation.
    assert units.replace_unit_state(state_context, path, linked, linked_sha, root=tmp_path) == linked_sha
    assert path.read_bytes() == before

    for changed in (
        _linked(state_context, work_unit_id, plan, trials_completed=6),
        _linked(state_context, work_unit_id, plan, result_sha256="c" * 64),
        _linked(state_context, work_unit_id, plan, result_path="results/other.json"),
        _linked(state_context, work_unit_id, plan, request_sha256="c" * 64),
        units.build_unit_state(state_context, work_unit_id, plan, status=units.STATUS_FAILED),
        units.build_unit_state(
            state_context,
            work_unit_id,
            plan,
            runtime_metadata={
                "hostname": "h",
                "process_id": 1,
                "device": "cpu",
                "wall_clock_annotation": "a",
                "update_annotation": "a",
            },
        ),
    ):
        with pytest.raises(units.StateConflictError):
            units.replace_unit_state(state_context, path, changed, linked_sha, root=tmp_path)
        assert path.read_bytes() == before


def test_invalid_proposed_state_leaves_original_bytes_unchanged(
    state_context: units.AuthenticatedUnitStateContext,
    tmp_path: Path,
) -> None:
    state, path = _claim(state_context, tmp_path)
    previous_sha = units.create_unit_state_exclusive(state_context, state, root=tmp_path)
    invalid = copy.deepcopy(state)
    invalid["identity"]["campaign_id"] = "wrong"
    before = path.read_bytes()
    with pytest.raises(units.UnitStateError):
        units.replace_unit_state(state_context, path, invalid, previous_sha, root=tmp_path)
    assert path.read_bytes() == before


def test_process_exit_before_and_after_replace_leaves_canonical_state(
    state_context: units.AuthenticatedUnitStateContext,
    tmp_path: Path,
) -> None:
    state, path = _claim(state_context, tmp_path)
    previous = units.create_unit_state_exclusive(state_context, state, root=tmp_path)
    plan = _plan(state_context)
    proposed = units.build_unit_state(
        state_context, state["identity"]["work_unit_id"], plan, status=units.STATUS_FAILED
    )
    old_bytes = path.read_bytes()

    def die_before() -> None:  # pragma: no cover - child process
        real_replace = os.replace
        os.replace = lambda *a, **k: os._exit(9)
        try:
            units.replace_unit_state(state_context, path, proposed, previous, root=tmp_path)
        finally:
            os.replace = real_replace

    assert _run_child(die_before) == 9
    assert path.read_bytes() == old_bytes

    def die_after() -> None:  # pragma: no cover - child process
        units._fsync_directory_descriptor = lambda *a, **k: os._exit(9)
        units.replace_unit_state(state_context, path, proposed, previous, root=tmp_path)

    assert _run_child(die_after) == 9
    assert units.read_unit_state(state_context, path, root=tmp_path) == proposed


def test_replacement_with_identical_bytes_is_deterministic(
    state_context: units.AuthenticatedUnitStateContext,
    tmp_path: Path,
) -> None:
    state, path = _claim(state_context, tmp_path)
    first_sha = units.create_unit_state_exclusive(state_context, state, root=tmp_path)
    first_bytes = path.read_bytes()
    second_sha = units.replace_unit_state(state_context, path, state, first_sha, root=tmp_path)
    assert second_sha == first_sha
    assert path.read_bytes() == first_bytes


def test_malformed_existing_state_fails_without_repair(
    state_context: units.AuthenticatedUnitStateContext,
    tmp_path: Path,
) -> None:
    state, path = _claim(state_context, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"{")
    with pytest.raises(units.UnitStateError):
        units.replace_unit_state(state_context, path, state, "0" * 64, root=tmp_path)
    assert path.read_bytes() == b"{"


def test_cross_unit_and_cross_shard_replacements_fail(
    state_context: units.AuthenticatedUnitStateContext,
    tmp_path: Path,
) -> None:
    state, path = _claim(state_context, tmp_path, shard_count=3, shard_index=0)
    old_sha = units.create_unit_state_exclusive(state_context, state, root=tmp_path)
    other_plan = _plan(state_context, shard_count=3, shard_index=1)
    other_id = other_plan["assigned_work_unit_ids"][0]
    other_state = units.build_unit_state(
        state_context, other_id, other_plan, status=units.STATUS_FAILED
    )
    with pytest.raises(units.G8BlerWorkUnitError):
        units.replace_unit_state(state_context, path, other_state, old_sha, root=tmp_path)
    wrong_shard_state = units.build_unit_state(
        state_context,
        state["identity"]["work_unit_id"],
        _plan(state_context, shard_count=1, shard_index=0),
        status=units.STATUS_FAILED,
    )
    with pytest.raises(units.StateConflictError):
        units.replace_unit_state(state_context, path, wrong_shard_state, old_sha, root=tmp_path)


def test_state_schema_rejects_noncanonical_json_on_read(
    state_context: units.AuthenticatedUnitStateContext,
    tmp_path: Path,
) -> None:
    state, path = _claim(state_context, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(units.UnitStateError):
        units.read_unit_state(state_context, path, root=tmp_path)


# ---------------------------------------------------------------------------
# Descriptor lifecycle and forced filesystem failures
# ---------------------------------------------------------------------------


class _FailingStream:
    def __init__(self, descriptor: int, failure: str) -> None:
        self._stream = os.fdopen(descriptor, "wb")
        self._failure = failure

    def write(self, data: bytes) -> int:
        if self._failure == "write":
            raise OSError(errno.EIO, "simulated write failure")
        return self._stream.write(data)

    def flush(self) -> None:
        if self._failure == "flush":
            raise OSError(errno.EIO, "simulated flush failure")
        self._stream.flush()

    def fileno(self) -> int:
        return self._stream.fileno()

    def __enter__(self) -> "_FailingStream":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._stream.close()


@pytest.mark.parametrize(
    "failure",
    ["open", "fdopen", "write", "flush", "fsync", "link", "dir_fsync", "unlink", "reread"],
)
def test_forced_failures_raise_a_domain_error_and_close_descriptors_once(
    state_context: units.AuthenticatedUnitStateContext,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    state, path = _claim(state_context, tmp_path)
    tracker = _CloseTracker(monkeypatch)
    real_open = os.open
    real_fdopen = os.fdopen

    if failure == "open":
        def failing_open(name: Any, *args: Any, **kwargs: Any) -> int:
            if isinstance(name, str) and name.endswith(units.STAGING_FILENAME_SUFFIX):
                raise OSError(errno.EIO, "simulated open failure")
            return real_open(name, *args, **kwargs)

        monkeypatch.setattr(os, "open", failing_open)
    elif failure == "fdopen":
        def failing_fdopen(descriptor: int, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("simulated fdopen failure")

        monkeypatch.setattr(os, "fdopen", failing_fdopen)
    elif failure in {"write", "flush"}:
        def wrapping_fdopen(descriptor: int, *args: Any, **kwargs: Any) -> Any:
            return _FailingStream(descriptor, failure)

        monkeypatch.setattr(os, "fdopen", wrapping_fdopen)
    elif failure == "fsync":
        monkeypatch.setattr(
            os, "fsync", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EIO, "simulated fsync"))
        )
    elif failure == "link":
        monkeypatch.setattr(
            os, "link", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EIO, "simulated link"))
        )
    elif failure == "dir_fsync":
        monkeypatch.setattr(
            units,
            "_fsync_directory_descriptor",
            lambda *a, **k: (_ for _ in ()).throw(units.AtomicStateError("simulated dir fsync")),
        )
    elif failure == "unlink":
        monkeypatch.setattr(
            os, "unlink", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EIO, "simulated unlink"))
        )
    elif failure == "reread":
        calls = {"n": 0}
        real_reader = units._read_state_bytes

        def failing_reader(target: Path) -> bytes:
            calls["n"] += 1
            raise OSError(errno.EIO, "simulated reread failure")

        monkeypatch.setattr(units, "_read_state_bytes", failing_reader)
        assert real_reader is not failing_reader

    if failure == "unlink":
        # Staging cleanup happens only after a successful publication; a
        # cleanup failure must not turn a published state into a failure.
        units.create_unit_state_exclusive(state_context, state, root=tmp_path)
        monkeypatch.setattr(os, "unlink", os.unlink)
    else:
        with pytest.raises(units.UnitStateError) as excinfo:
            units.create_unit_state_exclusive(state_context, state, root=tmp_path)
        assert not isinstance(excinfo.value, units.StateConflictError) or failure == "link"
        assert "Bad file descriptor" not in str(excinfo.value)
        cause = excinfo.value.__cause__
        assert cause is None or not (
            isinstance(cause, OSError) and cause.errno == errno.EBADF
        )

    assert tracker.double_closed == [], f"descriptors closed twice: {tracker.double_closed}"
    monkeypatch.setattr(os, "fdopen", real_fdopen)


def test_directory_fsync_failure_is_never_reported_as_success(
    state_context: units.AuthenticatedUnitStateContext,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, _path = _claim(state_context, tmp_path)
    real_fsync = os.fsync

    def failing_fsync(descriptor: int) -> None:
        if os.fstat(descriptor).st_mode & 0o170000 == 0o040000:
            raise OSError(errno.EACCES, "simulated directory fsync permission failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", failing_fsync)
    with pytest.raises(units.AtomicStateError, match="crash-durable publication is unavailable"):
        units.create_unit_state_exclusive(state_context, state, root=tmp_path)


def test_missing_no_replace_primitive_fails_closed(
    state_context: units.AuthenticatedUnitStateContext,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, path = _claim(state_context, tmp_path)

    def unsupported(*args: Any, **kwargs: Any) -> None:
        raise OSError(errno.EOPNOTSUPP, "no hard links on this filesystem")

    monkeypatch.setattr(os, "link", unsupported)
    with pytest.raises(units.AtomicStateError, match="no atomic no-replace primitive"):
        units.create_unit_state_exclusive(state_context, state, root=tmp_path)
    assert not path.exists()


def test_dangling_symlink_at_the_lock_path_is_rejected(
    state_context: units.AuthenticatedUnitStateContext,
    tmp_path: Path,
) -> None:
    state, path = _claim(state_context, tmp_path)
    previous = units.create_unit_state_exclusive(state_context, state, root=tmp_path)
    digest = path.name.removesuffix(".state.json")
    lock_dir = tmp_path / units.LOCK_DIRECTORY_NAME
    lock_dir.mkdir(exist_ok=True)
    (lock_dir / f"{digest}{units.LOCK_FILENAME_SUFFIX}").symlink_to(tmp_path / "missing")
    plan = _plan(state_context)
    proposed = units.build_unit_state(
        state_context, state["identity"]["work_unit_id"], plan, status=units.STATUS_FAILED
    )
    with pytest.raises(units.UnsafeUnitStatePathError):
        units.replace_unit_state(state_context, path, proposed, previous, root=tmp_path)


def test_parent_directory_symlink_swap_cannot_redirect_publication(
    state_context: units.AuthenticatedUnitStateContext,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    state, path = _claim(state_context, root)
    bucket = path.parent
    bucket.mkdir(parents=True)

    real_open = os.open
    swapped = {"done": False}

    def swapping_open(name: Any, *args: Any, **kwargs: Any) -> int:
        # Swap the bucket for a symlink to another directory at the moment the
        # publication path would otherwise reopen it by name.
        if not swapped["done"] and name == bucket.name:
            swapped["done"] = True
            descriptor = real_open(name, *args, **kwargs)
            bucket.rmdir()
            bucket.symlink_to(elsewhere, target_is_directory=True)
            return descriptor
        return real_open(name, *args, **kwargs)

    monkeypatch.setattr(os, "open", swapping_open)
    with contextlib.suppress(units.G8BlerWorkUnitError):
        units.create_unit_state_exclusive(state_context, state, root=root)
    monkeypatch.setattr(os, "open", real_open)

    # Every operation after the bucket open is descriptor-relative, so the swap
    # cannot redirect the write.  Publication either lands in the directory
    # that was validated or fails closed with a domain error; what it must
    # never do is write into the attacker's replacement directory.
    assert swapped["done"]
    assert not list(elsewhere.iterdir())
    assert not (elsewhere / path.name).exists()


# ---------------------------------------------------------------------------
# Scope guards and independent verification
# ---------------------------------------------------------------------------


def test_scope_guard_inspects_ast_not_strings_or_comments() -> None:
    source_path = Path(__file__).parents[1] / "src/baseline/g8_bler_work_units.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    imported = set()
    called_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)
    assert not any(name == "baseline.ldpc" or name.startswith("baseline.ldpc.") for name in imported)
    assert not any(
        name in {"information_bit_stream", "normal_stream", "G8Authorization", "BlerTable"}
        for name in called_names
    )
    assert "checkpoint" in ast.get_docstring(tree)


def test_generated_b2c_contract_is_canonical_and_independently_verified() -> None:
    payload = state_verifier.verify()
    assert payload["contract_id"] == state_generator.contract_identifier(payload)
    assert payload["checkpoint"] == "B2C"
    assert payload["schema_version"] == 2
    assert payload["authority_bindings"]["required_work_unit_count"] == 3213
    assert payload["scope"]["scientific_execution_performed"] is False
    assert payload["scope"]["test_split_access"] == 0
    assert state_generator.main(["--check"]) == 0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.__setitem__("phase", "G8_C"),
        lambda p: p.__setitem__("checkpoint", "B2"),
        lambda p: p.__setitem__("schema_version", 1),
        lambda p: p["supersedes"].__setitem__("contract_id", "g8state-" + "0" * 64),
        lambda p: p["supersedes"].__setitem__("contract_bytes", 1),
        lambda p: p["authority_bindings"].__setitem__("required_work_unit_count", 1),
        lambda p: p["sharding"].__setitem__("formula", "ordinal % 2 == shard_index"),
        lambda p: p["unit_state_schema"].__setitem__("identity_digest_rule", "runtime too"),
        lambda p: p["unit_state_schema"].__setitem__("schema_version", 1),
        lambda p: p["unit_state_schema"]["identity_fields"].remove("bler_state_contract_id"),
        lambda p: p["publication"]["exclusive_creation"].__setitem__("silent_overwrite", True),
        lambda p: p["publication"]["atomic_replacement"].__setitem__(
            "operation", "optimistic compare-and-swap"
        ),
        lambda p: p["scope"].__setitem__("simulation_started", True),
    ],
)
def test_independent_verifier_rejects_contract_mutations(
    tmp_path: Path,
    mutation: Any,
) -> None:
    payload = state_generator.build()
    mutation(payload)
    payload["contract_id"] = state_generator.contract_identifier(payload)
    contract_path, state_path = _staged_contract(tmp_path, payload)
    with pytest.raises(
        (state_verifier.G8BlerStateContractError, units.G8BlerWorkUnitError)
    ):
        state_verifier.verify(contract_path, campaign_state_path=state_path)


def test_verifier_does_not_import_expected_values_from_the_module_under_test() -> None:
    source_path = Path(__file__).parents[1] / "tools/verify_g8_bler_state_contract.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module != "baseline.g8_bler_work_units", (
                "the verifier must not import expected constants from the module "
                "under test; import it as a module instead"
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "gen_g8_bler_state_contract"
                assert alias.name != "baseline.g8_bler_work_units"
    # It may import the module under test only through the whole-module form.
    assert "from baseline import g8_bler_work_units as units" in source_path.read_text(
        encoding="utf-8"
    )


def test_verifier_defines_its_own_expected_constants() -> None:
    """The immutable authority must be re-declared, not re-exported."""

    assert state_verifier.EXPECTED_CAMPAIGN_ID == units.EXPECTED_CAMPAIGN_ID
    assert state_verifier.EXPECTED_SUPERSEDED_CONTRACT_ID == units.SUPERSEDED_STATE_CONTRACT_ID
    assert (
        state_verifier.EXPECTED_UNIT_STATE_IDENTITY_FIELDS
        == units.UNIT_STATE_IDENTITY_FIELDS
    )
    source = (
        Path(__file__).parents[1] / "tools/verify_g8_bler_state_contract.py"
    ).read_text(encoding="utf-8")
    # Each expected value is a literal in the verifier, not an alias.
    assert f'"{units.EXPECTED_CAMPAIGN_ID}"' in source
    assert f'"{units.SUPERSEDED_STATE_CONTRACT_ID}"' in source


def test_require_no_live_state_is_an_explicit_option(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A legitimate untracked runtime tree must not fail ordinary verification."""

    live_root = tmp_path / "work_units"
    live_root.mkdir()
    (live_root / "ab").mkdir()
    monkeypatch.setattr(state_verifier, "LIVE_STATE_TREE_PATH", live_root)

    # Default mode tolerates the runtime tree; B3 and B4 depend on this.
    assert state_verifier.verify()["checkpoint"] == "B2C"

    # The explicit closeout option rejects it.
    with pytest.raises(state_verifier.G8BlerStateContractError, match="live work-unit state tree"):
        state_verifier.verify(require_no_live_state=True)


def test_no_tracked_unit_state_or_lock_files_exist() -> None:
    """Tracked state is always rejected.

    There is deliberately no permanent assertion that the *runtime* tree never
    exists: that is true today and is scheduled to become false the moment B3
    or B4 legitimately executes.  Runtime absence is asserted only through the
    explicit ``--require-no-live-state`` closeout option.
    """

    state_verifier._verify_no_tracked_state()
