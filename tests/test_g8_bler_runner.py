"""B4 runner, gate, RNG, publication and isolated transaction tests."""

from __future__ import annotations

import json
import os
import ast
import copy
import io
from pathlib import Path
import subprocess
import sys
import tokenize

import numpy as np
import pytest

from baseline import g8_bler_contract as bler_contract
from baseline import g8_bler_runner as runner
from baseline import g8_bler_work_units as work_units
from baseline.ldpc import adapter as adapter_module
from baseline.g8_campaign import canonical_json, rendered_json, CAMPAIGN_STATE, REPO_ROOT, sha256_bytes
from config.params import get
import migrate_g8_bler_runner_contract as runner_migration
import verify_g8_bounded_smoke as smoke_verifier
import gen_g8_bler_runner_contract as runner_generator


@pytest.fixture(scope="module")
def auth_context(tmp_path_factory) -> runner.AuthenticatedRunnerContext:
    # Before the live v2 -> v3 migration, all execution tests use a generated
    # candidate and an isolated state projection.  The registered v2 artifact
    # remains untouched until the complete B5 production test block is green.
    root = tmp_path_factory.mktemp("runner-candidate")
    candidate_path = root / "bler_runner_contract.json"
    candidate = runner_generator.build()
    candidate_path.write_bytes(rendered_json(candidate))
    state = json.loads(CAMPAIGN_STATE.read_bytes())
    binding = {
        "path": runner.RUNNER_CONTRACT_REPO_RELATIVE_PATH,
        "sha256": sha256_bytes(candidate_path.read_bytes()),
        "bytes": candidate_path.stat().st_size,
    }
    for entry in state["identity"]["produced_artifacts"]:
        if entry["path"] == runner.RUNNER_CONTRACT_REPO_RELATIVE_PATH:
            entry.update(binding)
    state_path = root / "campaign_state.json"
    state_path.write_bytes(rendered_json(state))
    from baseline.g8_bler_resume import AuthenticatedResumeContext

    resume_context = AuthenticatedResumeContext(
        campaign_state_path=state_path,
        require_resume_contract=True,
    )
    return runner.AuthenticatedRunnerContext(
        resume_context=resume_context,
        runner_contract_path=candidate_path,
        require_registered_runner_contract=True,
    )


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


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    repo = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = f"{repo / 'src'}:{repo / 'tools'}"
    return env


def test_process_hard_exit_before_publication_leaves_no_final_json(tmp_path):
    root = tmp_path / "before-publish"
    target = root / "aa" / "request.json"
    script = r'''
import os, sys
from pathlib import Path
from baseline import g8_bler_runner as r
root = Path(sys.argv[1]); root.mkdir(); target = root / "aa" / "request.json"
def die(*args, **kwargs): os._exit(71)
r._publish_without_replace = die
r._publish_immutable_json(target, {"value": 1}, root=root)
'''
    completed = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        env=_child_env(),
        check=False,
    )
    assert completed.returncode == 71
    assert not target.exists()
    assert list((root / "aa").glob("*.staging"))


def test_process_hard_exit_after_publication_leaves_complete_json(tmp_path):
    root = tmp_path / "after-publish"
    target = root / "aa" / "request.json"
    body = canonical_json({"value": 2})
    script = r'''
import os, sys
from pathlib import Path
from baseline import g8_bler_runner as r
root = Path(sys.argv[1]); root.mkdir(); target = root / "aa" / "request.json"
original = r._publish_without_replace
def publish(*args, **kwargs):
    original(*args, **kwargs)
    os._exit(72)
r._publish_without_replace = publish
r._publish_immutable_json(target, {"value": 2}, root=root)
'''
    completed = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        env=_child_env(),
        check=False,
    )
    assert completed.returncode == 72
    assert target.read_bytes() == body


def test_concurrent_immutable_creators_have_one_exact_installed_result(tmp_path):
    root = tmp_path / "concurrent"
    root.mkdir()
    target = root / "aa" / "request.json"
    script = r'''
import sys
from pathlib import Path
from baseline import g8_bler_runner as r
root = Path(sys.argv[1]); target = root / "aa" / "request.json"
r._publish_immutable_json(target, {"same": True}, root=root)
'''
    first = subprocess.Popen([sys.executable, "-c", script, str(root)], env=_child_env())
    second = subprocess.Popen([sys.executable, "-c", script, str(root)], env=_child_env())
    assert first.wait(timeout=30) == 0
    assert second.wait(timeout=30) == 0
    assert target.read_bytes() == canonical_json({"same": True})
    assert not list(root.rglob("*.staging"))


def test_uncertain_publication_accepts_only_exact_installed_bytes(tmp_path, monkeypatch):
    root = tmp_path / "uncertain"
    root.mkdir()
    target = root / "aa" / "request.json"
    original_fsync = runner.os.fsync
    calls = 0

    def fail_directory_fsync(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected uncertain directory fsync")
        return original_fsync(fd)

    monkeypatch.setattr(runner.os, "fsync", fail_directory_fsync)
    payload = {"exact": True}
    digest = runner._publish_immutable_json(target, payload, root=root)
    assert digest == bler_contract.sha256_bytes(canonical_json(payload))
    assert target.read_bytes() == canonical_json(payload)


def test_uncertain_publication_rejects_nonexact_installed_bytes(tmp_path, monkeypatch):
    root = tmp_path / "uncertain-conflict"
    root.mkdir()
    target = root / "aa" / "request.json"
    original_read = runner._read_installed
    reads = 0

    def wrong_after_install(fd, name):
        nonlocal reads
        reads += 1
        observed = original_read(fd, name)
        return b"wrong" if reads >= 2 else observed

    monkeypatch.setattr(runner, "_read_installed", wrong_after_install)
    with pytest.raises(runner.RunnerPublicationError):
        runner._publish_immutable_json(target, {"exact": False}, root=root)


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


def _parse_without_comments(source: str) -> ast.AST:
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            token = tokenize.TokenInfo(
                token.type,
                "",
                token.start,
                token.end,
                token.line,
            )
        tokens.append(token)
    return ast.parse(tokenize.untokenize(tokens))


def test_staging_entropy_annotation_preserves_executable_ast():
    source_path = REPO_ROOT / "src/baseline/g8_bler_runner.py"
    before = subprocess.run(
        ["git", "show", "16377bd613ee89c1091688ad59cd527665757e33:src/baseline/g8_bler_runner.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")
    corrected_line = (
        '    staging = f".{target.name}.{os.getpid()}.{secrets.token_hex(12)}{_STAGING_SUFFIX}"  '
        "# literal-ok: cryptographic staging-name entropy bytes; filesystem uniqueness only, not a scientific parameter\n"
    )
    after = before.replace(
        '    staging = f".{target.name}.{os.getpid()}.{secrets.token_hex(12)}{_STAGING_SUFFIX}"\n',
        corrected_line,
    )
    assert after != before
    assert ast.dump(_parse_without_comments(before), include_attributes=False) == ast.dump(
        _parse_without_comments(after), include_attributes=False
    )
    assert corrected_line.rstrip() in source_path.read_text(encoding="utf-8")


def test_both_staging_entropy_literals_have_reasoned_annotations_and_no_global_exemption():
    source_path = REPO_ROOT / "src/baseline/g8_bler_runner.py"
    lines = [line for line in source_path.read_text(encoding="utf-8").splitlines() if "token_hex(12)" in line]
    assert len(lines) == 2
    assert all("literal-ok:" in line and line.split("literal-ok:", 1)[1].strip() for line in lines)
    assert 12 not in get("config.literal_lint_exempt_values")


@pytest.mark.parametrize("line_index", [0, 1])
def test_removing_either_staging_entropy_annotation_fails_literal_lint(tmp_path, monkeypatch, line_index):
    import check_literals as literals

    relative = Path("src/baseline/g8_bler_runner.py")
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    lines = (REPO_ROOT / relative).read_text(encoding="utf-8").splitlines(keepends=True)
    staging_lines = [index for index, line in enumerate(lines) if "token_hex(12)" in line]
    lines[staging_lines[line_index]] = lines[staging_lines[line_index]].split("  # literal-ok:", 1)[0].rstrip() + "\n"
    target.write_text("".join(lines), encoding="utf-8")
    monkeypatch.setattr(literals, "REPO", tmp_path)
    result = literals.check()
    assert any(finding.path == relative and finding.line == staging_lines[line_index] + 1 for finding in result.findings)


def test_runner_contract_verifier_is_independent_and_rejects_mutations(tmp_path):
    verifier_path = Path(__file__).resolve().parents[1] / "tools/verify_g8_bler_runner_contract.py"
    tree = ast.parse(verifier_path.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert not any("g8_bler_runner" in name or "gen_g8_bler_runner_contract" in name for name in imported)

    contract_path = Path(__file__).resolve().parents[1] / "results/baseline/g8/bler_runner_contract.json"
    original = json.loads(contract_path.read_text(encoding="utf-8"))
    for mutation in (
        "contract_id",
        "supersedes",
        "supersession_history",
        "authority_bindings",
        "dependencies",
        "authorization",
        "bounded_smoke",
        "campaign",
        "physical_layer",
        "contract_sources",
    ):
        mutated = json.loads(json.dumps(original))
        if mutation == "contract_id":
            mutated[mutation] = "g8runner-" + "0" * 64
        elif mutation == "supersedes":
            mutated[mutation]["contract_sha256"] = "0" * 64
        elif mutation == "supersession_history":
            mutated[mutation][0]["contract_id"] = "g8runner-" + "0" * 64
        elif mutation == "authority_bindings":
            mutated[mutation]["required_work_unit_count"] = 3212
        elif mutation == "dependencies":
            mutated[mutation]["configured_decoder_offset"] = 0.25
        elif mutation == "authorization":
            mutated[mutation]["bounded_smoke"]["max_work_units"] = 2
        elif mutation == "bounded_smoke":
            mutated[mutation]["trials_per_unit"] = 15
        elif mutation == "campaign":
            mutated[mutation] = "G-7"
        elif mutation == "physical_layer":
            mutated[mutation]["complex_noise_scale"] = "sqrt(N0)"
        else:
            mutated[mutation][0]["sha256"] = "0" * 64
        candidate = tmp_path / f"mutated-{mutation}.json"
        candidate.write_bytes(rendered_json(mutated))
        completed = subprocess.run(
            [sys.executable, "tools/verify_g8_bler_runner_contract.py", "--path", str(candidate)],
            cwd=Path(__file__).resolve().parents[1],
            env=_child_env(),
            check=False,
        )
        assert completed.returncode != 0, mutation


def test_candidate_v3_runner_contract_verifies_independently_before_registration(tmp_path):
    candidate = tmp_path / "bler_runner_contract-v3.json"
    candidate.write_bytes(rendered_json(runner_generator.build()))
    completed = subprocess.run(
        [sys.executable, "tools/verify_g8_bler_runner_contract.py", "--path", str(candidate)],
        cwd=REPO_ROOT,
        env=_child_env(),
        check=False,
    )
    assert completed.returncode == 0
    payload = json.loads(candidate.read_bytes())
    assert payload["schema_version"] == 3
    assert payload["supersedes"] == {
        "contract_id": runner.V2_RUNNER_CONTRACT_ID,
        "contract_sha256": runner.V2_RUNNER_CONTRACT_SHA256,
        "contract_bytes": runner.V2_RUNNER_CONTRACT_BYTES,
        "reason": runner.RUNNER_CONTRACT_SUPERSESSION_REASON,
    }
    assert payload["supersession_history"][0]["schema_version"] == 2
    assert payload["supersession_history"][1]["schema_version"] == 1


def test_candidate_runner_contract_registers_against_isolated_campaign_state(tmp_path):
    from baseline.g8_bler_resume import AuthenticatedResumeContext

    candidate_path = tmp_path / "bler_runner_contract.json"
    candidate_path.write_bytes(rendered_json(runner_generator.build()))
    state = json.loads(CAMPAIGN_STATE.read_bytes())
    for entry in state["identity"]["produced_artifacts"]:
        if entry["path"] == runner.RUNNER_CONTRACT_REPO_RELATIVE_PATH:
            entry.update(
                bytes=candidate_path.stat().st_size,
                sha256=sha256_bytes(candidate_path.read_bytes()),
            )
    state_path = tmp_path / "campaign_state.json"
    state_path.write_bytes(rendered_json(state))
    resume_context = AuthenticatedResumeContext(
        campaign_state_path=state_path,
        require_resume_contract=True,
    )
    context = runner.AuthenticatedRunnerContext(
        resume_context=resume_context,
        runner_contract_path=candidate_path,
        require_registered_runner_contract=True,
    )
    assert context.runner_contract_binding()["bler_runner_contract_id"].startswith("g8runner-")


def _old_runner_state_payload() -> dict:
    payload = json.loads(CAMPAIGN_STATE.read_bytes())
    for entry in payload["identity"]["produced_artifacts"]:
        if entry["path"] == runner.RUNNER_CONTRACT_REPO_RELATIVE_PATH:
            entry.update(
                bytes=runner.V2_RUNNER_CONTRACT_BYTES,
                sha256=runner.V2_RUNNER_CONTRACT_SHA256,
            )
        elif entry["path"] == runner.SMOKE_RECORD_REPO_RELATIVE_PATH:
            entry.update(
                bytes=runner_migration.OLD_SMOKE_BYTES,
                sha256=runner_migration.OLD_SMOKE_SHA256,
            )
    return payload


def _isolated_old_runner_state(tmp_path: Path) -> Path:
    path = tmp_path / "campaign_state.json"
    path.write_bytes(rendered_json(_old_runner_state_payload()))
    return path


def _isolated_old_smoke(tmp_path: Path) -> Path:
    path = tmp_path / "bounded_smoke_record-v2.json"
    path.write_bytes(
        subprocess.run(
            [
                "git",
                "show",
                "16377bd613ee89c1091688ad59cd527665757e33:results/baseline/g8/bounded_smoke_record.json",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    )
    return path


def _candidate_v3_runner(tmp_path: Path) -> Path:
    path = tmp_path / "bler_runner_contract-v3.json"
    path.write_bytes(rendered_json(runner_generator.build()))
    return path


def test_campaign_state_identity_remains_closed_after_runner_correction():
    payload = json.loads(CAMPAIGN_STATE.read_bytes())
    assert set(payload["identity"]) == {
        "campaign_id",
        "campaign_manifest_sha256",
        "phase",
        "stage",
        "completed_work_unit_ids",
        "in_progress_work_unit_id",
        "produced_artifacts",
        "restart_command",
        "seed_derivation_identity",
        "counters",
    }
    assert "required_bler_artifact_sha256" not in payload["identity"]
    assert "selection_policy_sha256" not in payload["identity"]


def test_runner_contract_migration_replaces_exactly_one_binding(tmp_path):
    state_path = _isolated_old_runner_state(tmp_path)
    before = json.loads(state_path.read_bytes())
    candidate_path = _candidate_v3_runner(tmp_path)
    installed = runner_migration.migrate(
        contract_path=candidate_path,
        state_path=state_path,
        smoke_path=_isolated_old_smoke(tmp_path),
    )
    after = installed["identity"]
    assert len(after["produced_artifacts"]) == 7
    assert sum(entry["path"] == runner.RUNNER_CONTRACT_REPO_RELATIVE_PATH for entry in after["produced_artifacts"]) == 1
    assert after["produced_artifacts"] != before["identity"]["produced_artifacts"]
    assert [entry for entry in after["produced_artifacts"] if entry["path"] != runner.RUNNER_CONTRACT_REPO_RELATIVE_PATH] == [
        entry for entry in before["identity"]["produced_artifacts"]
        if entry["path"] != runner.RUNNER_CONTRACT_REPO_RELATIVE_PATH
    ]
    new_binding = next(entry for entry in after["produced_artifacts"] if entry["path"] == runner.RUNNER_CONTRACT_REPO_RELATIVE_PATH)
    assert new_binding["sha256"] == sha256_bytes(candidate_path.read_bytes())


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda payload: next(entry for entry in payload["identity"]["produced_artifacts"] if entry["path"] == runner.RUNNER_CONTRACT_REPO_RELATIVE_PATH).__setitem__("sha256", "0" * 64), "state runner binding"),
        (lambda payload: payload["identity"]["produced_artifacts"].append({"path": runner.RUNNER_CONTRACT_REPO_RELATIVE_PATH, "sha256": runner.V2_RUNNER_CONTRACT_SHA256, "bytes": runner.V2_RUNNER_CONTRACT_BYTES}), "exactly seven"),
        (lambda payload: payload["identity"].__setitem__("stage", "tooling_smoke_complete"), "tooling_open"),
        (lambda payload: payload["identity"]["counters"].__setitem__("inference", 1), "zero counters"),
        (lambda payload: payload["identity"].__setitem__("completed_work_unit_ids", ["u"]), "no completed"),
        (lambda payload: payload["identity"].__setitem__("in_progress_work_unit_id", "u"), "no in-progress"),
    ],
)
def test_runner_contract_migration_rejects_state_mutations(tmp_path, mutation, match):
    payload = _old_runner_state_payload()
    mutation(payload)
    state_path = tmp_path / "campaign_state.json"
    state_path.write_bytes(rendered_json(payload))
    with pytest.raises(runner_migration.RunnerContractMigrationError, match=match):
        runner_migration.migrate(
            contract_path=_candidate_v3_runner(tmp_path),
            state_path=state_path,
            smoke_path=_isolated_old_smoke(tmp_path),
        )


def test_runner_contract_migration_rejects_unrelated_artifact_mutation(tmp_path):
    payload = _old_runner_state_payload()
    next(entry for entry in payload["identity"]["produced_artifacts"] if entry["path"] == "results/baseline/g8/bler_tooling_contract.json")["sha256"] = "0" * 64
    state_path = tmp_path / "campaign_state.json"
    state_path.write_bytes(rendered_json(payload))
    with pytest.raises(runner_migration.RunnerContractMigrationError, match="strict projected validation"):
        runner_migration.migrate(
            contract_path=_candidate_v3_runner(tmp_path),
            state_path=state_path,
            smoke_path=_isolated_old_smoke(tmp_path),
        )


def test_runner_contract_migration_rejects_malformed_supersession(tmp_path):
    candidate = runner_generator.build()
    candidate["supersedes"]["contract_id"] = "g8runner-" + "0" * 64
    candidate_path = tmp_path / "bler_runner_contract.json"
    candidate_path.write_bytes(rendered_json(candidate))
    state_path = _isolated_old_runner_state(tmp_path)
    with pytest.raises(runner_migration.RunnerContractMigrationError, match="independent verification"):
        runner_migration.migrate(
            contract_path=candidate_path,
            state_path=state_path,
            smoke_path=_isolated_old_smoke(tmp_path),
        )


def test_runner_contract_migration_preserves_state_on_interrupted_publication(tmp_path, monkeypatch):
    state_path = _isolated_old_runner_state(tmp_path)
    before = state_path.read_bytes()
    candidate_path = _candidate_v3_runner(tmp_path)

    def interrupted(*args, **kwargs):
        raise OSError("injected interrupted publication")

    monkeypatch.setattr(runner_migration, "_publish_state", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        runner_migration.migrate(
            contract_path=candidate_path,
            state_path=state_path,
            smoke_path=_isolated_old_smoke(tmp_path),
        )
    assert state_path.read_bytes() == before


def test_runner_contract_migration_recovers_the_complete_v2_v3_matrix(tmp_path):
    v2_contract = tmp_path / "bler_runner_contract-v2.json"
    v2_contract.write_bytes(
        subprocess.run(
            [
                "git",
                "show",
                "16377bd613ee89c1091688ad59cd527665757e33:results/baseline/g8/bler_runner_contract.json",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    )
    v2_smoke = _isolated_old_smoke(tmp_path)
    v3_contract = _candidate_v3_runner(tmp_path)
    v3_smoke = tmp_path / "bounded_smoke_record-v3.json"
    v3_smoke.write_bytes((REPO_ROOT / runner.SMOKE_RECORD_REPO_RELATIVE_PATH).read_bytes())

    # 1: v2 runner + v2 state binding + v2 smoke + v2 smoke binding is a no-op.
    state_path = tmp_path / "campaign_state.json"
    state_path.write_bytes(rendered_json(_old_runner_state_payload()))
    before = state_path.read_bytes()
    runner_migration.migrate(
        contract_path=v2_contract,
        state_path=state_path,
        smoke_path=v2_smoke,
    )
    assert state_path.read_bytes() == before

    # 2: install v3 runner while the state and smoke remain v2.
    runner_migration.migrate(
        contract_path=v3_contract,
        state_path=state_path,
        smoke_path=v2_smoke,
    )
    state_v3_runner = json.loads(state_path.read_bytes())
    assert next(
        entry for entry in state_v3_runner["identity"]["produced_artifacts"]
        if entry["path"] == runner.RUNNER_CONTRACT_REPO_RELATIVE_PATH
    )["sha256"] == sha256_bytes(v3_contract.read_bytes())
    assert next(
        entry for entry in state_v3_runner["identity"]["produced_artifacts"]
        if entry["path"] == runner.SMOKE_RECORD_REPO_RELATIVE_PATH
    )["sha256"] == runner_migration.OLD_SMOKE_SHA256

    # 3: v3 runner + v3 state binding + v2 smoke remains an idempotent no-op.
    before = state_path.read_bytes()
    runner_migration.migrate(
        contract_path=v3_contract,
        state_path=state_path,
        smoke_path=v2_smoke,
    )
    assert state_path.read_bytes() == before

    # 4: v3 smoke has been installed, but the state still binds v2 smoke.
    runner_migration.migrate(
        contract_path=v3_contract,
        state_path=state_path,
        smoke_path=v3_smoke,
    )
    state_v3 = json.loads(state_path.read_bytes())
    assert next(
        entry for entry in state_v3["identity"]["produced_artifacts"]
        if entry["path"] == runner.SMOKE_RECORD_REPO_RELATIVE_PATH
    )["sha256"] == sha256_bytes(v3_smoke.read_bytes())

    # 5: v3 runner + v3 state + v3 smoke binding is an idempotent no-op.
    before = state_path.read_bytes()
    runner_migration.migrate(
        contract_path=v3_contract,
        state_path=state_path,
        smoke_path=v3_smoke,
    )
    assert state_path.read_bytes() == before


def test_runner_contract_migration_recovers_after_state_publication_interrupt(tmp_path, monkeypatch):
    state_path = _isolated_old_runner_state(tmp_path)
    v2_smoke = _isolated_old_smoke(tmp_path)
    v3_contract = _candidate_v3_runner(tmp_path)
    real_publish = runner_migration._publish_state

    def publish_then_interrupt(*args, **kwargs):
        real_publish(*args, **kwargs)
        raise KeyboardInterrupt("simulated interruption after state publication")

    monkeypatch.setattr(runner_migration, "_publish_state", publish_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="after state publication"):
        runner_migration.migrate(
            contract_path=v3_contract,
            state_path=state_path,
            smoke_path=v2_smoke,
        )
    installed = json.loads(state_path.read_bytes())
    assert next(
        entry for entry in installed["identity"]["produced_artifacts"]
        if entry["path"] == runner.RUNNER_CONTRACT_REPO_RELATIVE_PATH
    )["sha256"] == sha256_bytes(v3_contract.read_bytes())
    monkeypatch.setattr(runner_migration, "_publish_state", real_publish)
    before = state_path.read_bytes()
    runner_migration.migrate(
        contract_path=v3_contract,
        state_path=state_path,
        smoke_path=v2_smoke,
    )
    assert state_path.read_bytes() == before


def test_old_runner_contract_is_rejected_after_supersession(tmp_path):
    old_bytes = subprocess.run(
        ["git", "show", "d4042bce2bcb3142c9a0b6e39fa3fa93a6fbb94a:results/baseline/g8/bler_runner_contract.json"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    old_path = tmp_path / "old-runner-contract.json"
    old_path.write_bytes(old_bytes)
    with pytest.raises(runner.RunnerAuthorizationError):
        runner.AuthenticatedRunnerContext(runner_contract_path=old_path, require_registered_runner_contract=True)


def test_cached_request_and_result_validation_does_not_reauthenticate_large_artifacts(auth_context, monkeypatch):
    request = _bounded_request(auth_context, trials=7)
    result = runner._build_result(
        auth_context,
        request=request,
        status=bler_contract.STATUS_COMPLETE,
        trials_completed=7,
        bit_errors=3,
        block_errors=1,
        execution_metadata={
            "wall_time_s": None,
            "hostname": None,
            "device": "cpu",
            "shard_index": 0,
            "shard_count": 1,
            "attempt": 1,
        },
    )
    counts = {"tooling": 0, "required": 0}
    old_tooling = bler_contract.load_bler_tooling_contract
    old_required = bler_contract._required_work_unit_bytes

    def tooling(*args, **kwargs):
        counts["tooling"] += 1
        return old_tooling(*args, **kwargs)

    def required(*args, **kwargs):
        counts["required"] += 1
        return old_required(*args, **kwargs)

    monkeypatch.setattr(bler_contract, "load_bler_tooling_contract", tooling)
    monkeypatch.setattr(bler_contract, "_required_work_unit_bytes", required)
    for _ in range(200):
        assert auth_context.validate_request(request)["work_unit_id"] == request["work_unit_id"]
        assert auth_context.validate_result(result, request=request)["status"] == bler_contract.STATUS_COMPLETE
    assert counts == {"tooling": 0, "required": 0}


def test_smoke_record_publisher_rejects_conflict_and_allows_guarded_provisional_replacement(tmp_path, monkeypatch):
    live_root = REPO_ROOT
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    target = tmp_path / "results" / "baseline" / "g8" / "bounded_smoke_record.json"
    target.parent.mkdir(parents=True)
    old_body = subprocess.run(
        ["git", "show", "d4042bce2bcb3142c9a0b6e39fa3fa93a6fbb94a:results/baseline/g8/bounded_smoke_record.json"],
        cwd=live_root,
        check=True,
        capture_output=True,
    ).stdout
    target.write_bytes(old_body)
    new = {"schema_version": 2, "artifact_role": "g8_bounded_smoke_record", "label": runner.BOUNDED_SMOKE_LABEL}
    digest = runner.publish_smoke_record_atomic(
        target,
        new,
        expected_provisional_sha256=sha256_bytes(old_body),
    )
    assert target.read_bytes() == rendered_json(new)
    assert digest == sha256_bytes(rendered_json(new))
    with pytest.raises(runner.RunnerConflictError):
        runner.publish_smoke_record_atomic(target, {"different": True})


def test_smoke_record_publisher_guarded_schema2_exchange_is_exact_and_alias_safe(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    parent = tmp_path / "results" / "baseline" / "g8"
    parent.mkdir(parents=True)
    old_body = _isolated_old_smoke(tmp_path).read_bytes()
    new_body = (REPO_ROOT / runner.SMOKE_RECORD_REPO_RELATIVE_PATH).read_bytes()
    new_payload = json.loads(new_body)
    target = parent / "bounded_smoke_record.json"
    target.write_bytes(old_body)
    digest = runner.publish_smoke_record_atomic(
        target,
        new_payload,
        expected_existing_sha256=runner_migration.OLD_SMOKE_SHA256,
        expected_existing_runner_contract_id=runner_migration.OLD_SMOKE_RUNNER_ID,
        expected_existing_runner_contract_sha256=runner_migration.OLD_SMOKE_RUNNER_SHA256,
    )
    assert target.read_bytes() == new_body
    assert digest == sha256_bytes(new_body)
    assert runner.publish_smoke_record_atomic(target, new_payload) == digest
    target.write_bytes(old_body)
    with pytest.raises(runner.RunnerConflictError):
        runner.publish_smoke_record_atomic(
            target,
            new_payload,
            expected_existing_sha256=runner_migration.OLD_SMOKE_SHA256,
            expected_existing_runner_contract_id="g8runner-" + "0" * 64,
            expected_existing_runner_contract_sha256=runner_migration.OLD_SMOKE_RUNNER_SHA256,
        )

    for kind in ("symlink", "dangling_symlink", "hard_link"):
        alias = parent / f"{kind}.json"
        foreign = tmp_path / f"{kind}-foreign.json"
        foreign.write_bytes(old_body)
        if kind == "symlink":
            alias.symlink_to(foreign)
        elif kind == "dangling_symlink":
            alias.symlink_to(tmp_path / f"{kind}-missing.json")
        else:
            alias.hardlink_to(foreign)
        with pytest.raises(runner.RunnerConflictError):
            runner.publish_smoke_record_atomic(
                alias,
                new_payload,
                expected_existing_sha256=runner_migration.OLD_SMOKE_SHA256,
                expected_existing_runner_contract_id=runner_migration.OLD_SMOKE_RUNNER_ID,
                expected_existing_runner_contract_sha256=runner_migration.OLD_SMOKE_RUNNER_SHA256,
            )


def test_smoke_record_guarded_exchange_recovers_directory_fsync_uncertainty(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    parent = tmp_path / "results" / "baseline" / "g8"
    parent.mkdir(parents=True)
    target = parent / "bounded_smoke_record.json"
    old_body = _isolated_old_smoke(tmp_path).read_bytes()
    target.write_bytes(old_body)
    new_payload = json.loads((REPO_ROOT / runner.SMOKE_RECORD_REPO_RELATIVE_PATH).read_bytes())
    original_fsync = runner.os.fsync
    calls = 0

    def fail_directory_fsync(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected smoke directory fsync uncertainty")
        return original_fsync(fd)

    monkeypatch.setattr(runner.os, "fsync", fail_directory_fsync)
    assert runner.publish_smoke_record_atomic(
        target,
        new_payload,
        expected_existing_sha256=runner_migration.OLD_SMOKE_SHA256,
        expected_existing_runner_contract_id=runner_migration.OLD_SMOKE_RUNNER_ID,
        expected_existing_runner_contract_sha256=runner_migration.OLD_SMOKE_RUNNER_SHA256,
    ) == sha256_bytes(rendered_json(new_payload))
    assert target.read_bytes() == rendered_json(new_payload)


def test_bounded_smoke_cli_rejects_diagnostic_max_units_without_touching_record(tmp_path):
    from run_g8_bler import main

    record = REPO_ROOT / "results/baseline/g8/bounded_smoke_record.json"
    before = record.read_bytes()
    assert main([
        "--execution-class", "bounded_smoke",
        "--root", str(tmp_path / "diagnostic"),
        "--device", "cpu",
        "--shard-count", "1",
        "--shard-index", "0",
        "--batch-size", "1",
        "--max-units", "1",
    ]) == 4
    assert record.read_bytes() == before


def test_smoke_verifier_does_not_import_runner_or_expect_campaign_identity_fields():
    verifier_path = REPO_ROOT / "tools/verify_g8_bounded_smoke.py"
    tree = ast.parse(verifier_path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any("g8_bler_runner" in name or "gen_g8_bler_runner_contract" in name for name in imports)
    source = verifier_path.read_text(encoding="utf-8")
    assert 'identity["required_bler_artifact_sha256"]' not in source
    assert 'identity["selection_policy_sha256"]' not in source


def test_smoke_record_publication_survives_hard_exit_before_and_after_install(tmp_path):
    env = _child_env()
    script_before = (
        "import os,sys\n"
        "from pathlib import Path\n"
        "from baseline import g8_bler_runner as r\n"
        "root=Path(sys.argv[1]); target=root/'results'/'baseline'/'g8'/'bounded_smoke_record.json'\n"
        "target.parent.mkdir(parents=True); r.REPO_ROOT=root\n"
        "def die(*args,**kwargs): os._exit(73)\n"
        "r._renameat2=die\n"
        "r.publish_smoke_record_atomic(target, {'schema_version':2})\n"
    )
    before_root = tmp_path / "before"
    completed = subprocess.run([sys.executable, "-c", script_before, str(before_root)], env=env, check=False)
    assert completed.returncode == 73
    before_dir = before_root / "results" / "baseline" / "g8"
    assert not (before_dir / "bounded_smoke_record.json").exists()
    assert list(before_dir.glob("*.staging"))

    script_after = (
        "import os,sys\n"
        "from pathlib import Path\n"
        "from baseline import g8_bler_runner as r\n"
        "root=Path(sys.argv[1]); target=root/'results'/'baseline'/'g8'/'bounded_smoke_record.json'\n"
        "target.parent.mkdir(parents=True); r.REPO_ROOT=root\n"
        "original=r._renameat2\n"
        "def install_then_die(*args,**kwargs):\n"
        "    original(*args,**kwargs); os._exit(74)\n"
        "r._renameat2=install_then_die\n"
        "r.publish_smoke_record_atomic(target, {'schema_version':2})\n"
    )
    after_root = tmp_path / "after"
    completed = subprocess.run([sys.executable, "-c", script_after, str(after_root)], env=env, check=False)
    assert completed.returncode == 74
    assert (after_root / "results" / "baseline" / "g8" / "bounded_smoke_record.json").read_bytes() == rendered_json({"schema_version": 2})


def test_registered_schema2_smoke_exchange_survives_hard_exit_boundaries(tmp_path):
    env = _child_env()
    old_path = _isolated_old_smoke(tmp_path)
    new_path = tmp_path / "bounded_smoke_record-v3.json"
    new_path.write_bytes((REPO_ROOT / runner.SMOKE_RECORD_REPO_RELATIVE_PATH).read_bytes())
    script = r'''
import json, os, sys
from pathlib import Path
from baseline import g8_bler_runner as r
root = Path(sys.argv[1]); old = Path(sys.argv[2]); new = Path(sys.argv[3])
target = root / "results" / "baseline" / "g8" / "bounded_smoke_record.json"
target.parent.mkdir(parents=True); target.write_bytes(old.read_bytes()); r.REPO_ROOT = root
kwargs = {
    "expected_existing_sha256": "cff4fb75835c4a010baed285103c3ba425b7b44b226186ce9969dcb17537763e",
    "expected_existing_runner_contract_id": "g8runner-3e4c870966837d255829dbca6afc4d1e3ce5ccf4754618460c939607d9c1c7e5",
    "expected_existing_runner_contract_sha256": "21ec8ae9c3c0787fa0a43bfdc12b4362bd26534a4774ee682070d94449e11268",
}
def die(*args, **kwargs): os._exit(75)
r._renameat2 = die
r.publish_smoke_record_atomic(target, json.loads(new.read_bytes()), **kwargs)
'''
    before_root = tmp_path / "guard-before"
    completed = subprocess.run(
        [sys.executable, "-c", script, str(before_root), str(old_path), str(new_path)],
        env=env,
        check=False,
    )
    assert completed.returncode == 75
    before_target = before_root / "results" / "baseline" / "g8" / "bounded_smoke_record.json"
    assert before_target.read_bytes() == old_path.read_bytes()
    assert list(before_target.parent.glob("*.staging"))

    after_script = script.replace(
        "def die(*args, **kwargs): os._exit(75)\nr._renameat2 = die",
        "original = r._renameat2\ndef die(*args, **kwargs):\n    original(*args, **kwargs)\n    os._exit(76)\nr._renameat2 = die",
    )
    after_root = tmp_path / "guard-after"
    completed = subprocess.run(
        [sys.executable, "-c", after_script, str(after_root), str(old_path), str(new_path)],
        env=env,
        check=False,
    )
    assert completed.returncode == 76
    after_target = after_root / "results" / "baseline" / "g8" / "bounded_smoke_record.json"
    assert after_target.read_bytes() == new_path.read_bytes()


def test_cli_returns_hold_when_post_publication_verifier_fails(tmp_path, monkeypatch):
    import run_g8_bler

    root = tmp_path / "smoke-root"
    monkeypatch.setattr(run_g8_bler.runner, "AuthenticatedRunnerContext", lambda: object())
    monkeypatch.setattr(run_g8_bler.runner, "run_bounded_smoke", lambda *args, **kwargs: ([], root))
    monkeypatch.setattr(run_g8_bler.runner, "build_bounded_smoke_record", lambda *args, **kwargs: {"schema_version": 2})
    monkeypatch.setattr(run_g8_bler, "_remove_isolated_root", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_g8_bler, "_write_smoke_record", lambda record: (tmp_path / "record.json", "0" * 64))
    monkeypatch.setattr(
        run_g8_bler.smoke_verifier,
        "verify",
        lambda *args, **kwargs: (_ for _ in ()).throw(smoke_verifier.SmokeVerificationError("injected verifier failure")),
    )
    assert run_g8_bler.main([
        "--execution-class", "bounded_smoke",
        "--root", str(root),
        "--device", "cpu",
        "--shard-count", "1",
        "--shard-index", "0",
        "--batch-size", "1",
        "--max-units", "3",
    ]) == run_g8_bler.EXIT_HOLD
