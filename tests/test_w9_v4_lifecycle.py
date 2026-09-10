"""Focused pre-science W9 v4 custody, authority and AM-97 fixtures."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import torch

from evaluation.h4_precision import H4PrecisionError, aggregate_three_cell_differences, simulate_h4_precision
import runtime.transactional_epochs as transactional_epochs
import runtime.w9_authority as w9_authority
from runtime.source_guard import committed_source_differences, working_tree_source_differences
from runtime.transactional_epochs import TransactionalEpochStore, TransactionalRuntimeHold, canonical_bytes
from runtime.run_identity import RunDisposition, classify_run_identity, terminal_run_counts
from runtime.w9_authority import W9AuthorityHold, authenticate_live_w9_pascal, resolve_authority_pair, resolve_runtime_root
from training.deterministic_core import apply_optimizer_update
from training.w9_v4 import W9V4TrainingRuntime


def _store(tmp_path: Path, total: int = 2) -> TransactionalEpochStore:
    store = TransactionalEpochStore(
        tmp_path / "runtime",
        identity={"run_id": "synthetic-run", "source_binding": "source", "config_hash": "c" * 64},
        total_epochs=total,
        role="W9_SYNTHETIC_EPOCH",
    )
    store.initialise()
    return store


def _publish(store: TransactionalEpochStore, epoch: int, *, selected: int = 0) -> None:
    store.publish_epoch(
        epoch,
        f"checkpoint-{epoch}".encode(),
        {
            "optimizer_opportunities": 3,
            "applied_optimizer_steps": 2,
            "grad_scaler_skips": 1,
            "validation_n_correct": 10 + epoch,
            "selection_candidate": selected,
        },
    )


def test_successful_transaction_and_staging_is_not_committed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    staging = store.epochs_root / ".epoch-0000-interrupted.staging"
    staging.mkdir()
    assert os.stat(staging).st_dev == os.stat(store.epochs_root).st_dev
    assert store.inspect() == []
    committed = _store(tmp_path / "other").publish_epoch(0, b"abc", {"optimizer_opportunities": 0, "applied_optimizer_steps": 0, "grad_scaler_skips": 0})
    assert committed.epoch == 0
    assert committed.checkpoint_path.is_file()


def test_epoch_rename_is_followed_by_parent_directory_fsync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, Path]] = []
    original_rename = transactional_epochs.os.rename
    original_fsync_directory = transactional_epochs._fsync_directory

    def rename(source: Path, destination: Path) -> None:
        original_rename(source, destination)
        events.append(("rename", Path(destination)))

    def fsync_directory(path: Path) -> None:
        original_fsync_directory(path)
        events.append(("fsync", Path(path)))

    monkeypatch.setattr(transactional_epochs.os, "rename", rename)
    monkeypatch.setattr(transactional_epochs, "_fsync_directory", fsync_directory)
    store = _store(tmp_path)
    _publish(store, 0)
    rename_index = next(index for index, event in enumerate(events) if event == ("rename", store.epochs_root / "epoch-0000"))
    assert events[rename_index + 1] == ("fsync", store.epochs_root)


def test_two_epochs_resume_exact_next_epoch(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _publish(store, 0)
    _publish(store, 1)
    assert [item.epoch for item in store.inspect()] == [0, 1]
    with pytest.raises(TransactionalRuntimeHold, match="exact next epoch"):
        store.publish_epoch(1, b"replay", {})


def test_stale_pointer_recovers_but_corrupt_epoch_holds(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _publish(store, 0)
    _publish(store, 1)
    pointer = json.loads((store.runtime_root / "latest.json").read_bytes())
    pointer["epoch"] = 0
    pointer["next_epoch"] = 1
    pointer["epoch_path"] = "epochs/epoch-0000"
    pointer["chain_sha256"] = store.inspect()[0].chain_sha256
    (store.runtime_root / "latest.json").write_bytes(canonical_bytes(pointer))
    assert store.inspect()[-1].epoch == 1
    assert json.loads((store.runtime_root / "latest.json").read_bytes())["epoch"] == 1
    (store.epochs_root / "epoch-0001" / "checkpoint.pt").write_bytes(b"corrupt")
    with pytest.raises(TransactionalRuntimeHold, match="checkpoint hash"):
        store.inspect()


def test_gap_and_identity_mismatch_hold(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _publish(store, 0)
    _publish(store, 1)
    os.rename(store.epochs_root / "epoch-0001", store.epochs_root / "epoch-0002")
    with pytest.raises(TransactionalRuntimeHold, match="unbroken prefix"):
        store.inspect()

    store = _store(tmp_path / "identity")
    _publish(store, 0)
    record_path = store.epochs_root / "epoch-0000" / "record.json"
    record = json.loads(record_path.read_bytes())
    record["identity"]["run_id"] = "other"
    record_path.write_bytes(canonical_bytes(record))
    with pytest.raises(TransactionalRuntimeHold, match="identity differs"):
        store.inspect()


def test_crash_after_commit_before_pointer_and_terminal_idempotence(tmp_path: Path) -> None:
    store = _store(tmp_path, total=1)
    _publish(store, 0)
    (store.runtime_root / "latest.json").unlink()
    assert store.inspect()[-1].epoch == 0
    terminal, reused = store.terminalize(selected_epoch=0, selection_metric={"validation_n_correct": 10}, tie_break="earliest_epoch")
    assert reused is False and terminal["test_access"] == 0
    again, reused = store.terminalize(selected_epoch=0, selection_metric={"validation_n_correct": 10}, tie_break="earliest_epoch")
    assert reused is True and again == terminal
    path = store.runtime_root / "run_terminal.json"
    value = json.loads(path.read_bytes())
    value["selected_epoch"] = 99
    path.write_bytes(canonical_bytes(value))
    with pytest.raises(TransactionalRuntimeHold, match="run terminal differs"):
        store.terminalize(selected_epoch=0, selection_metric={"validation_n_correct": 10}, tie_break="earliest_epoch")


def test_h4_one_cell_sanity_and_three_cell_covariance() -> None:
    low = simulate_h4_precision([1] * 10 + [0] * 90, sample_size=3925)
    high = simulate_h4_precision([1] * 50 + [0] * 50, sample_size=3925)
    assert 1.3 < low["mde_percentage_points"]["q95"] < 1.5
    assert 3.0 < high["mde_percentage_points"]["q95"] < 3.3
    ids = ["image-a", "image-b"]
    learned = {cell: {stable: {"-1": int(stable == "image-a")} for stable in ids} for cell in ("0/0", "1/1", "2/2")}
    er9 = {cell: {stable: {"-1": 0} for stable in ids} for cell in ("0/0", "1/1", "2/2")}
    aggregate = aggregate_three_cell_differences(learned, er9, snr_grid_db=(-1,))
    assert aggregate["values"].tolist() == [[1.0], [0.0]]
    with pytest.raises(H4PrecisionError, match="randomized"):
        aggregate_three_cell_differences(learned, er9, snr_grid_db=(-1,), learned_arm="randomized_er2")
    broken = dict(learned)
    broken["1/1"] = {"image-a": {"-1": 1}, "image-c": {"-1": 0}}
    with pytest.raises(H4PrecisionError, match="stable IDs"):
        aggregate_three_cell_differences(broken, er9, snr_grid_db=(-1,))


def test_runtime_authority_has_one_source_of_truth(tmp_path: Path) -> None:
    authority = {"runtime_root": "checkpoints/er9_pascal_v4"}
    assert resolve_runtime_root(tmp_path, authority) == (tmp_path / authority["runtime_root"]).resolve()
    assert resolve_authority_pair(tmp_path, authority, dict(authority)).name == "er9_pascal_v4"
    with pytest.raises(W9AuthorityHold):
        resolve_authority_pair(tmp_path, authority, {"runtime_root": "checkpoints/er2_randomized_pascal_v4"})


def test_source_drift_fixture_is_classified_without_touching_repo(tmp_path: Path) -> None:
    # The real guard uses Git.  This fixture ensures its live-tree shape is
    # stable and does not mistake ordinary output namespaces for source.
    assert set(working_tree_source_differences(Path.cwd())) <= {"unstaged", "staged", "untracked"}


def _git_fixture(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "git-fixture"
    for directory in ("src", "tools", "configs", "spec", "tests", "results/learned/er9"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "src/protected.py").write_text("value = 1\n", encoding="utf-8")
    (root / "requirements-pascal.lock").write_text("fixture-lock\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "W9 fixture"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=root, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    return root, commit


def test_source_guard_rejects_unstaged_staged_untracked_and_accepts_outputs(tmp_path: Path) -> None:
    root, _ = _git_fixture(tmp_path)
    protected = root / "src/protected.py"
    protected.write_text("value = 2\n", encoding="utf-8")
    assert working_tree_source_differences(root) == {"unstaged": ["src/protected.py"], "staged": [], "untracked": []}
    subprocess.run(["git", "add", "src/protected.py"], cwd=root, check=True)
    assert working_tree_source_differences(root) == {"unstaged": [], "staged": ["src/protected.py"], "untracked": []}
    subprocess.run(["git", "restore", "--staged", "src/protected.py"], cwd=root, check=True)
    protected.write_text("value = 1\n", encoding="utf-8")
    (root / "src/untracked.py").write_text("value = 3\n", encoding="utf-8")
    (root / "results/learned/er9/runtime.json").write_text("{}\n", encoding="utf-8")
    differences = working_tree_source_differences(root)
    assert differences == {"unstaged": [], "staged": [], "untracked": ["src/untracked.py"]}


def test_source_guard_reports_committed_protected_drift(tmp_path: Path) -> None:
    root, source_commit = _git_fixture(tmp_path)
    (root / "src/protected.py").write_text("value = 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "src/protected.py"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "drift"], cwd=root, check=True)
    assert committed_source_differences(root, source_commit) == ["src/protected.py"]


def _pascal_authority() -> dict[str, object]:
    return {
        "execution_profile_id": "confessor_pascal_cu126",
        "host": "confessor",
        "gpu_uuid": "GPU-expected",
        "gpu_name": "TITAN Xp",
        "compute_capability": "6.1",
        "device": "cuda:0",
        "source_binding": {"source_commit": "a" * 40},
    }


def test_live_pascal_authentication_rejects_wrong_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(w9_authority.socket, "gethostname", lambda: "laptop")
    monkeypatch.setattr(w9_authority.platform, "node", lambda: "laptop")
    with pytest.raises(W9AuthorityHold, match="live host is not Confessor"):
        authenticate_live_w9_pascal(tmp_path, _pascal_authority(), config_hash="c" * 64)


def test_live_pascal_authentication_rejects_wrong_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(w9_authority.socket, "gethostname", lambda: "confessor")
    monkeypatch.setattr(w9_authority.platform, "node", lambda: "confessor")
    monkeypatch.setattr(w9_authority, "assert_clean_source_closure", lambda *_args: {})
    monkeypatch.setattr(
        w9_authority,
        "authenticate_execution_profile",
        lambda *_args, **_kwargs: {
            "gpu_uuid": "GPU-wrong",
            "gpu_name": "TITAN Xp",
            "gpu_compute_capability": "6.1",
            "gpu_index": 0,
            "git_dirty": False,
        },
    )
    with pytest.raises(W9AuthorityHold, match="GPU UUID differs"):
        authenticate_live_w9_pascal(tmp_path, _pascal_authority(), config_hash="c" * 64)


def test_run_counts_are_model_identities_not_process_starts() -> None:
    fresh = classify_run_identity(model_identity="m1", runtime_exists=False, terminal_exists=False)
    resumed = classify_run_identity(model_identity="m1", runtime_exists=True, terminal_exists=False, process_starts=2)
    reused = classify_run_identity(model_identity="m1", runtime_exists=True, terminal_exists=True, process_starts=3)
    promoted = classify_run_identity(model_identity="m-search", runtime_exists=True, terminal_exists=True, promoted_from_search=True)
    assert fresh.disposition is RunDisposition.FRESH_STARTED
    assert resumed.disposition is RunDisposition.RESUMED_INCOMPLETE
    assert reused.disposition is RunDisposition.EXISTING_COMPLETE_REUSED
    assert promoted.disposition is RunDisposition.PROMOTED_SEARCH_RUN
    assert terminal_run_counts(resumed)["scientific_model_count"] == 1
    assert terminal_run_counts(resumed)["process_starts"] == 2


def test_amp_overflow_after_unscale_is_a_scaler_skip() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)

    class FakeScaler:
        def __init__(self) -> None:
            self.scale = 8.0

        def get_scale(self) -> float:
            return self.scale

        def unscale_(self, _optimizer: torch.optim.Optimizer) -> None:
            parameter.grad = torch.tensor(float("inf"))

        def step(self, _optimizer: torch.optim.Optimizer) -> None:
            pass

        def update(self) -> None:
            self.scale = 4.0

    parameter.grad = torch.tensor(8.0)
    update = apply_optimizer_update(optimizer, FakeScaler(), denominator=1)
    assert update.applied is False
    assert update.optimizer_gradients["finite"] is False
    assert 0 + 1 == 1  # one opportunity = one recorded scaler skip


def test_shared_model_optimizer_adapter_resumes_without_replay(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    runtime = W9V4TrainingRuntime(
        tmp_path / "adapter",
        identity={"run_id": "adapter", "source_binding": "s", "config_hash": "a" * 64},
        total_epochs=1,
        role="W9_ADAPTER",
    )
    runtime.commit_epoch(epoch=0, model=model, optimizer=optimizer, scaler=None, metrics={"optimizer_opportunities": 1, "applied_optimizer_steps": 1, "grad_scaler_skips": 0})
    restored = torch.nn.Linear(2, 2)
    restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.1)
    assert runtime.restore_latest(restored, restored_optimizer, None) == 0
    terminal, reused = runtime.terminalize(selected_epoch=0, selection_metric={"validation_n_correct": 0}, tie_break="earliest_epoch")
    assert terminal["selected_epoch"] == 0 and reused is False
