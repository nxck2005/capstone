"""B2 mutation, partition, path, and crash-safety tests.

All publication tests use a temporary work-unit root.  This file deliberately
contains no runner, channel, codec, dataset, classifier, or test-split path.
"""

from __future__ import annotations

import ast
import copy
import errno
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from baseline import g8_bler_contract
from baseline import g8_bler_work_units as units
import gen_g8_bler_state_contract as state_generator
import verify_g8_bler_state_contract as state_verifier
from baseline.g8_campaign import rendered_json


@pytest.fixture(scope="module")
def context() -> units.AuthenticatedExecutionContext:
    return units.AuthenticatedExecutionContext()


def _plan(
    context: units.AuthenticatedExecutionContext,
    *,
    shard_count: int = 7,
    shard_index: int = 0,
) -> dict[str, Any]:
    return units.build_shard_plan(context, shard_count, shard_index)


def _claim(
    context: units.AuthenticatedExecutionContext,
    root: Path,
    *,
    shard_count: int = 7,
    shard_index: int = 0,
) -> tuple[dict[str, Any], Path]:
    plan = _plan(context, shard_count=shard_count, shard_index=shard_index)
    work_unit_id = plan["assigned_work_unit_ids"][0]
    state = units.build_unit_state(context, work_unit_id, plan)
    return state, units.unit_state_path(context, work_unit_id, root=root)


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
        "tooling_schema_version": 2,
        "request_schema_version": 2,
        "result_schema_version": 2,
        "required_work_unit_count": 3213,
    }
    assert len(context.ordered_work_unit_ids) == 3213
    assert context.work_unit_record(context.work_unit_ids[0])["work_unit_id"] == context.work_unit_ids[0]


def test_context_exposes_fresh_mutation_safe_records(
    context: units.AuthenticatedExecutionContext,
) -> None:
    binding = context.authority_binding()
    binding["campaign_id"] = "corrupted"
    record = context.work_unit_record(context.work_unit_ids[0])
    record["identity"]["k_and_n"][0] = 1
    record["source_packet_config_ids"].append("corrupted")
    index = context.work_unit_index()
    index[context.work_unit_ids[0]]["identity"]["k_and_n"][1] = 1
    index[context.work_unit_ids[0]]["source_packet_config_ids"].clear()
    assert context.campaign_id == units.EXPECTED_CAMPAIGN_ID
    fresh = context.work_unit_record(context.work_unit_ids[0])
    assert fresh["identity"]["k_and_n"] == [7128, 8534]
    assert fresh["source_packet_config_ids"]
    assert context.ordered_work_unit_ids[0] == "bler-0020cd25150d4f59a8fbb7c0"


def test_context_order_and_cached_authority_are_not_reparsed_per_unit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    original = g8_bler_contract.load_required_bler_identities

    def counted(path: Path = g8_bler_contract.REQUIRED_BLER_IDENTITIES) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return original(path)

    g8_bler_contract._campaign_binding_bytes.cache_clear()
    g8_bler_contract._required_work_unit_bytes.cache_clear()
    monkeypatch.setattr(g8_bler_contract, "load_required_bler_identities", counted)
    try:
        fresh = units.AuthenticatedExecutionContext()
        for work_unit_id in fresh.work_unit_ids[::97]:
            fresh.work_unit_record(work_unit_id)
        for shard_index in range(11):
            units.build_shard_plan(fresh, 11, shard_index)
        assert calls == 1
    finally:
        g8_bler_contract._campaign_binding_bytes.cache_clear()
        g8_bler_contract._required_work_unit_bytes.cache_clear()


@pytest.mark.parametrize("field", ["contract_id", "schema_version"])
def test_wrong_b1c_tooling_binding_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    original = g8_bler_contract.load_bler_tooling_contract

    def altered(path: Path | None = None) -> dict[str, Any]:
        payload = copy.deepcopy(original(path))
        payload[field] = "wrong" if field == "contract_id" else 1
        return payload

    monkeypatch.setattr(g8_bler_contract, "load_bler_tooling_contract", altered)
    with pytest.raises(units.AuthorityAuthenticationError):
        units.AuthenticatedExecutionContext()


def test_wrong_b1c_artifact_sha_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(units, "sha256_file", lambda _path: "0" * 64)
    with pytest.raises(units.AuthorityAuthenticationError):
        units.AuthenticatedExecutionContext()


def test_wrong_required_artifact_binding_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    original = g8_bler_contract.campaign_bindings

    def altered() -> dict[str, str]:
        payload = original()
        payload["required_bler_artifact_sha256"] = "0" * 64
        return payload

    monkeypatch.setattr(g8_bler_contract, "campaign_bindings", altered)
    with pytest.raises(units.AuthorityAuthenticationError):
        units.AuthenticatedExecutionContext()


def test_wrong_required_work_unit_count_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    original = g8_bler_contract.load_bler_tooling_contract

    def altered(path: Path | None = None) -> dict[str, Any]:
        payload = copy.deepcopy(original(path))
        payload["campaign_bindings"]["required_work_unit_count"] = 1
        return payload

    monkeypatch.setattr(g8_bler_contract, "load_bler_tooling_contract", altered)
    with pytest.raises(units.AuthorityAuthenticationError):
        units.AuthenticatedExecutionContext()


def test_one_shard_preserves_all_canonical_ids(context: units.AuthenticatedExecutionContext) -> None:
    plan = _plan(context, shard_count=1, shard_index=0)
    assert plan["assigned_work_unit_ids"] == list(context.ordered_work_unit_ids)
    assert units.validate_shard_plan(context, plan) == plan


@pytest.mark.parametrize("shard_count", [2, 3, 4, 7, 11, 32, 64, 127, 3213])
def test_ordinal_modulo_partition_is_complete_disjoint_and_balanced(
    context: units.AuthenticatedExecutionContext,
    shard_count: int,
) -> None:
    plans = [_plan(context, shard_count=shard_count, shard_index=i) for i in range(shard_count)]
    partitions = [plan["assigned_work_unit_ids"] for plan in plans]
    flattened = [work_unit_id for part in partitions for work_unit_id in part]
    assert len(flattened) == len(set(flattened)) == 3213
    assert set(flattened) == set(context.ordered_work_unit_ids)
    assert max(map(len, partitions)) - min(map(len, partitions)) <= 1
    for index, part in enumerate(partitions):
        assert part == list(context.ordered_work_unit_ids[index::shard_count])


def test_shard_plan_is_byte_identical_and_does_not_depend_on_mapping_order(
    context: units.AuthenticatedExecutionContext,
) -> None:
    first = _plan(context, shard_count=13, shard_index=4)
    second = _plan(context, shard_count=13, shard_index=4)
    assert units.shard_plan_bytes(first) == units.shard_plan_bytes(second)
    reordered = {key: first[key] for key in reversed(list(first))}
    assert units.shard_plan_bytes(reordered) == units.shard_plan_bytes(first)
    assert units.validate_shard_plan(context, reordered) == first


@pytest.mark.parametrize("value", [True, False, 0, -1, 2.0, "2", None])
def test_invalid_shard_counts_are_rejected(
    context: units.AuthenticatedExecutionContext,
    value: Any,
) -> None:
    with pytest.raises(units.ShardPlanError):
        units.build_shard_plan(context, value, 0)


@pytest.mark.parametrize("value", [True, False, -1, 2.0, "0", None])
def test_invalid_shard_indices_are_rejected(
    context: units.AuthenticatedExecutionContext,
    value: Any,
) -> None:
    with pytest.raises(units.ShardPlanError):
        units.build_shard_plan(context, 2, value)


def test_out_of_range_shard_index_is_rejected(context: units.AuthenticatedExecutionContext) -> None:
    with pytest.raises(units.ShardPlanError):
        units.build_shard_plan(context, 2, 2)


@pytest.mark.parametrize("field", ["assigned_work_unit_ids", "plan_digest", "shard_index", "sharding_algorithm"])
def test_shard_plan_mutations_fail_closed(
    context: units.AuthenticatedExecutionContext,
    field: str,
) -> None:
    plan = _plan(context, shard_count=5, shard_index=1)
    mutated = copy.deepcopy(plan)
    if field == "assigned_work_unit_ids":
        mutated[field] = list(reversed(mutated[field]))
    elif field == "plan_digest":
        mutated[field] = "0" * 64
    elif field == "shard_index":
        mutated[field] = 0
    else:
        mutated[field] = "other_algorithm"
    with pytest.raises(units.ShardPlanError):
        units.validate_shard_plan(context, mutated)


def test_mutating_returned_plan_does_not_corrupt_context(context: units.AuthenticatedExecutionContext) -> None:
    plan = _plan(context, shard_count=3, shard_index=0)
    plan["assigned_work_unit_ids"].clear()
    rebuilt = _plan(context, shard_count=3, shard_index=0)
    assert rebuilt["assigned_work_unit_count"] == 1071
    assert units.validate_shard_plan(context, rebuilt) == rebuilt


def test_shard_layout_never_enters_seed_derivation(context: units.AuthenticatedExecutionContext) -> None:
    work_unit_id = context.work_unit_ids[100]
    seeds = {purpose: context.seed(work_unit_id, purpose) for purpose in g8_bler_contract.SEED_PURPOSES}
    assert seeds == {
        purpose: g8_bler_contract.derive_seed(context.campaign_id, work_unit_id, purpose)
        for purpose in g8_bler_contract.SEED_PURPOSES
    }
    assert _plan(context, shard_count=1, shard_index=0)["assigned_work_unit_ids"][100] == work_unit_id
    assert _plan(context, shard_count=17, shard_index=100 % 17)["assigned_work_unit_ids"]
    assert seeds == {purpose: context.seed(work_unit_id, purpose) for purpose in g8_bler_contract.SEED_PURPOSES}


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
        units.validate_unit_state_path(context, str(path.relative_to(tmp_path)), work_unit_id, root=tmp_path)
    with pytest.raises(units.UnsafeUnitStatePathError):
        units.validate_unit_state_path(context, str(path).replace(str(tmp_path), str(tmp_path / "missing")), work_unit_id, root=tmp_path)


def test_symlink_escape_is_rejected(
    context: units.AuthenticatedExecutionContext,
    tmp_path: Path,
) -> None:
    work_unit_id = context.work_unit_ids[0]
    path = units.unit_state_path(context, work_unit_id, root=tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    try:
        path.parent.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("filesystem does not permit symlink creation")
    with pytest.raises(units.UnsafeUnitStatePathError):
        units.validate_unit_state_path(context, path, work_unit_id, root=tmp_path)


def test_claim_failed_and_result_linked_states_are_closed(
    context: units.AuthenticatedExecutionContext,
) -> None:
    plan = _plan(context, shard_count=3, shard_index=0)
    work_unit_id = plan["assigned_work_unit_ids"][0]
    claim = units.build_unit_state(context, work_unit_id, plan)
    failed = units.build_unit_state(context, work_unit_id, plan, status=units.STATUS_FAILED)
    linked = units.build_unit_state(
        context,
        work_unit_id,
        plan,
        status=units.STATUS_RESULT_LINKED,
        request_sha256="a" * 64,
        result_path="results/baseline/g8/results/example.json",
        result_sha256="b" * 64,
        scientific_execution_performed=True,
        trials_completed=1,
    )
    assert claim["identity"]["status"] == units.STATUS_CLAIMED
    assert failed["identity"]["status"] == units.STATUS_FAILED
    assert linked["identity"]["status"] == units.STATUS_RESULT_LINKED
    for state in (claim, failed, linked):
        assert units.validate_unit_state(context, state) == state


def test_state_identity_digest_excludes_runtime_metadata(
    context: units.AuthenticatedExecutionContext,
) -> None:
    plan = _plan(context, shard_count=3, shard_index=0)
    work_unit_id = plan["assigned_work_unit_ids"][0]
    bare = units.build_unit_state(context, work_unit_id, plan)
    annotated = units.build_unit_state(
        context,
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
    assert units.unit_state_sha256(context, bare) != units.unit_state_sha256(context, annotated)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.pop("identity_sha256"),
        lambda s: s.__setitem__("unexpected", True),
        lambda s: s.__setitem__("schema_version", 2),
        lambda s: s.__setitem__("artifact_role", "other"),
        lambda s: s["identity"].__setitem__("campaign_id", "other-campaign"),
        lambda s: s["identity"].__setitem__("request_schema_version", 1),
        lambda s: s["identity"].__setitem__("result_schema_version", 1),
        lambda s: s["identity"].__setitem__("canonical_ordinal", 1),
        lambda s: s["identity"].__setitem__("required_work_unit_record_sha256", "0" * 64),
        lambda s: s["identity"].__setitem__("shard_plan_digest", "0" * 64),
        lambda s: s["identity"].__setitem__("identity_sha256", "0" * 64),
    ],
)
def test_state_identity_mutations_fail_closed(
    context: units.AuthenticatedExecutionContext,
    mutation: Any,
) -> None:
    plan = _plan(context, shard_count=3, shard_index=0)
    state = units.build_unit_state(context, plan["assigned_work_unit_ids"][0], plan)
    mutated = copy.deepcopy(state)
    mutation(mutated)
    with pytest.raises(units.UnitStateError):
        units.validate_unit_state(context, mutated)


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
    context: units.AuthenticatedExecutionContext,
    field: str,
    value: Any,
) -> None:
    plan = _plan(context, shard_count=3, shard_index=0)
    state = units.build_unit_state(context, plan["assigned_work_unit_ids"][0], plan)
    mutated = copy.deepcopy(state)
    target = mutated["runtime_metadata"] if field == "process_id" else mutated["identity"]
    target[field] = value
    with pytest.raises(units.UnitStateError):
        units.validate_unit_state(context, mutated)


def test_status_contradictions_fail_closed(context: units.AuthenticatedExecutionContext) -> None:
    plan = _plan(context, shard_count=3, shard_index=0)
    work_unit_id = plan["assigned_work_unit_ids"][0]
    claim = units.build_unit_state(context, work_unit_id, plan)
    bad_claim = copy.deepcopy(claim)
    bad_claim["identity"]["trials_completed"] = 1
    bad_claim["identity_sha256"] = units.state_identity_digest(bad_claim)
    with pytest.raises(units.UnitStateError):
        units.validate_unit_state(context, bad_claim)
    bad_failed = copy.deepcopy(units.build_unit_state(context, work_unit_id, plan, status=units.STATUS_FAILED))
    bad_failed["identity"]["result_path"] = "results/x.json"
    bad_failed["identity"]["result_sha256"] = "a" * 64
    bad_failed["identity_sha256"] = units.state_identity_digest(bad_failed)
    with pytest.raises(units.UnitStateError):
        units.validate_unit_state(context, bad_failed)
    bad_linked = copy.deepcopy(claim)
    bad_linked["identity"]["status"] = units.STATUS_RESULT_LINKED
    bad_linked["identity_sha256"] = units.state_identity_digest(bad_linked)
    with pytest.raises(units.UnitStateError):
        units.validate_unit_state(context, bad_linked)


def test_state_schema_rejects_noncanonical_json_on_read(
    context: units.AuthenticatedExecutionContext,
    tmp_path: Path,
) -> None:
    state, path = _claim(context, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(units.UnitStateError):
        units.read_unit_state(context, path, root=tmp_path)


def test_exclusive_create_has_one_winner_under_race(
    context: units.AuthenticatedExecutionContext,
    tmp_path: Path,
) -> None:
    state, path = _claim(context, tmp_path)

    def attempt() -> str:
        return units.create_unit_state_exclusive(context, state, root=tmp_path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(attempt) for _ in range(8)]
    successes = []
    conflicts = []
    for future in futures:
        try:
            successes.append(future.result())
        except units.StateConflictError as exc:
            conflicts.append(exc)
    assert len(successes) == 1
    assert len(conflicts) == 7
    assert units.read_unit_state(context, path, root=tmp_path) == state


def test_stale_writer_cannot_overwrite_newer_state(
    context: units.AuthenticatedExecutionContext,
    tmp_path: Path,
) -> None:
    state, path = _claim(context, tmp_path)
    old_sha = units.create_unit_state_exclusive(context, state, root=tmp_path)
    plan = _plan(context)
    work_unit_id = state["identity"]["work_unit_id"]
    newer = units.build_unit_state(context, work_unit_id, plan, status=units.STATUS_FAILED)
    new_sha = units.replace_unit_state(context, path, newer, old_sha, root=tmp_path)
    before = path.read_bytes()
    with pytest.raises(units.StaleWriterError):
        units.replace_unit_state(context, path, state, old_sha, root=tmp_path)
    assert path.read_bytes() == before
    assert new_sha == hashlib.sha256(before).hexdigest()


def test_invalid_proposed_state_leaves_original_bytes_unchanged(
    context: units.AuthenticatedExecutionContext,
    tmp_path: Path,
) -> None:
    state, path = _claim(context, tmp_path)
    previous_sha = units.create_unit_state_exclusive(context, state, root=tmp_path)
    invalid = copy.deepcopy(state)
    invalid["identity"]["campaign_id"] = "wrong"
    before = path.read_bytes()
    with pytest.raises(units.UnitStateError):
        units.replace_unit_state(context, path, invalid, previous_sha, root=tmp_path)
    assert path.read_bytes() == before


def test_interruption_before_replace_leaves_original_and_cleans_partial(
    context: units.AuthenticatedExecutionContext,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, path = _claim(context, tmp_path)
    old_sha = units.create_unit_state_exclusive(context, state, root=tmp_path)
    proposed = units.build_unit_state(context, state["identity"]["work_unit_id"], _plan(context), status=units.STATUS_FAILED)
    before = path.read_bytes()

    def interrupted(*_args: Any, **_kwargs: Any) -> None:
        raise OSError(errno.EIO, "simulated interruption")

    monkeypatch.setattr(units.os, "replace", interrupted)
    with pytest.raises(units.AtomicStateError):
        units.replace_unit_state(context, path, proposed, old_sha, root=tmp_path)
    assert path.read_bytes() == before
    assert not list(path.parent.glob("*.partial"))


def test_interruption_after_replace_is_recoverable(
    context: units.AuthenticatedExecutionContext,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, path = _claim(context, tmp_path)
    old_sha = units.create_unit_state_exclusive(context, state, root=tmp_path)
    proposed = units.build_unit_state(context, state["identity"]["work_unit_id"], _plan(context), status=units.STATUS_FAILED)
    original_fsync_directory = units._fsync_directory

    def after_replace(_parent: Path) -> bool:
        raise OSError(errno.EIO, "simulated post-replace interruption")

    monkeypatch.setattr(units, "_fsync_directory", after_replace)
    with pytest.raises(units.AtomicStateError, match="installed bytes"):
        units.replace_unit_state(context, path, proposed, old_sha, root=tmp_path)
    monkeypatch.setattr(units, "_fsync_directory", original_fsync_directory)
    assert units.read_unit_state(context, path, root=tmp_path) == proposed
    assert not list(path.parent.glob("*.partial"))


def test_replacement_with_identical_bytes_is_deterministic(
    context: units.AuthenticatedExecutionContext,
    tmp_path: Path,
) -> None:
    state, path = _claim(context, tmp_path)
    first_sha = units.create_unit_state_exclusive(context, state, root=tmp_path)
    first_bytes = path.read_bytes()
    second_sha = units.replace_unit_state(context, path, state, first_sha, root=tmp_path)
    assert second_sha == first_sha
    assert path.read_bytes() == first_bytes


def test_malformed_existing_state_fails_without_repair(
    context: units.AuthenticatedExecutionContext,
    tmp_path: Path,
) -> None:
    state, path = _claim(context, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"{")
    with pytest.raises(units.UnitStateError):
        units.replace_unit_state(context, path, state, "0" * 64, root=tmp_path)
    assert path.read_bytes() == b"{"


def test_cross_unit_and_cross_shard_replacements_fail(
    context: units.AuthenticatedExecutionContext,
    tmp_path: Path,
) -> None:
    state, path = _claim(context, tmp_path, shard_count=3, shard_index=0)
    old_sha = units.create_unit_state_exclusive(context, state, root=tmp_path)
    other_plan = _plan(context, shard_count=3, shard_index=1)
    other_id = other_plan["assigned_work_unit_ids"][0]
    other_state = units.build_unit_state(context, other_id, other_plan, status=units.STATUS_FAILED)
    with pytest.raises(units.G8BlerWorkUnitError):
        units.replace_unit_state(context, path, other_state, old_sha, root=tmp_path)
    wrong_shard_state = units.build_unit_state(
        context,
        state["identity"]["work_unit_id"],
        _plan(context, shard_count=1, shard_index=0),
        status=units.STATUS_FAILED,
    )
    with pytest.raises(units.UnitStateError):
        units.replace_unit_state(context, path, wrong_shard_state, old_sha, root=tmp_path)


def test_attempt_regression_fails(
    context: units.AuthenticatedExecutionContext,
    tmp_path: Path,
) -> None:
    state, path = _claim(context, tmp_path)
    old_sha = units.create_unit_state_exclusive(context, state, root=tmp_path)
    plan = _plan(context)
    failed_attempt_two = units.build_unit_state(
        context,
        state["identity"]["work_unit_id"],
        plan,
        attempt=2,
        status=units.STATUS_FAILED,
    )
    new_sha = units.replace_unit_state(context, path, failed_attempt_two, old_sha, root=tmp_path)
    regressed = units.build_unit_state(
        context,
        state["identity"]["work_unit_id"],
        plan,
        attempt=1,
        status=units.STATUS_FAILED,
    )
    with pytest.raises(units.StateConflictError):
        units.replace_unit_state(context, path, regressed, new_sha, root=tmp_path)


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
    assert not any(name in {"information_bit_stream", "normal_stream", "G8Authorization", "BlerTable"} for name in called_names)
    # A comment/string containing a forbidden word is not an execution path.
    assert "checkpoint" in ast.get_docstring(tree)


def test_b2_never_creates_live_work_unit_tree() -> None:
    assert not units.DEFAULT_WORK_UNIT_ROOT.exists()


def test_generated_b2_contract_is_canonical_and_independently_verified() -> None:
    payload = state_verifier.verify()
    assert payload["contract_id"] == state_generator.contract_identifier(payload)
    assert payload["authority_bindings"]["required_work_unit_count"] == 3213
    assert payload["scope"]["scientific_execution_performed"] is False
    assert payload["scope"]["test_split_access"] == 0
    assert state_generator.main(["--check"]) == 0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.__setitem__("phase", "G8_C"),
        lambda p: p["authority_bindings"].__setitem__("required_work_unit_count", 1),
        lambda p: p["sharding"].__setitem__("formula", "ordinal % 2 == shard_index"),
        lambda p: p["unit_state_schema"].__setitem__("identity_digest_rule", "runtime too"),
        lambda p: p["publication"]["exclusive_creation"].__setitem__("silent_overwrite", True),
        lambda p: p["scope"].__setitem__("simulation_started", True),
    ],
)
def test_independent_b2_verifier_rejects_contract_mutations(
    tmp_path: Path,
    mutation: Any,
) -> None:
    payload = state_generator.build()
    mutation(payload)
    payload["contract_id"] = state_generator.contract_identifier(payload)
    path = tmp_path / "bler_state_contract.json"
    path.write_bytes(rendered_json(payload))
    with pytest.raises(state_verifier.G8BlerStateContractError):
        state_verifier.verify(path)


def test_contract_verifier_does_not_import_or_call_the_generator() -> None:
    source_path = Path(__file__).parents[1] / "tools/verify_g8_bler_state_contract.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    imported.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert "gen_g8_bler_state_contract" not in imported
