from __future__ import annotations

from pathlib import Path

import pytest

import run_post_g10_historical_check as adapter


def _touch(root: Path, relative: Path) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_adapter_requires_safe_completion_and_reconciliation(tmp_path: Path) -> None:
    with pytest.raises(adapter.PostG10HistoricalCheckHold):
        adapter._require_terminal(tmp_path)
    _touch(tmp_path, adapter.COMPLETION_PATH)
    with pytest.raises(adapter.PostG10HistoricalCheckHold):
        adapter._require_terminal(tmp_path)
    _touch(tmp_path, adapter.RECONCILIATION_PATH)
    adapter._require_terminal(tmp_path)


def test_adapter_rejects_symlink_terminal_sentinels(tmp_path: Path) -> None:
    completion = tmp_path / adapter.COMPLETION_PATH
    completion.parent.mkdir(parents=True, exist_ok=True)
    completion.symlink_to(tmp_path / "missing-completion.json")
    _touch(tmp_path, adapter.RECONCILIATION_PATH)

    with pytest.raises(adapter.PostG10HistoricalCheckHold):
        adapter._require_terminal(tmp_path)


def test_adapter_has_only_explicit_historical_targets() -> None:
    expected = {
        "g8_f_sampler_plan_check",
        "g8_f1_closeout",
        "w5_training_system",
        "g8_campaign_manifest_check",
        "w4_baseline_integration",
        "w6_classical_build_check",
        "w6_classical_verify",
        "w6_complete",
        "w7_g4",
        "w8_a",
    }
    assert set(adapter.TARGETS) == expected
    assert "g10_w9_terminal" not in adapter.TARGETS
    forbidden = {
        "tools/run_g10_campaign.py",
        "tools/verify_g10_w9.py",
        "tools/train",
        "tools/inference",
    }
    assert not any(path in forbidden for target in adapter.TARGETS.values() for path in target)


def test_adapter_rejects_arbitrary_target() -> None:
    with pytest.raises(adapter.PostG10HistoricalCheckHold):
        adapter._execute_target("arbitrary.py", Path("/tmp"))


def test_additive_am94_verification_is_required_before_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(adapter, "_verify_terminal", lambda root: calls.append("terminal"))

    def reject_additive(root: Path) -> None:
        calls.append("am94")
        raise adapter.PostG10HistoricalCheckHold("additive verification failed")

    monkeypatch.setattr(adapter, "_verify_additive_am94", reject_additive)
    monkeypatch.setattr(adapter, "_execute_target", lambda target, root: calls.append("target"))

    with pytest.raises(adapter.PostG10HistoricalCheckHold):
        adapter.run("g8_f_sampler_plan_check", tmp_path)
    assert calls == ["terminal", "am94"]


def test_adapter_patches_only_in_memory_before_target_and_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = object()
    original_load = adapter.g10_spec_compatibility.load
    monkeypatch.setattr(adapter, "_verify_terminal", lambda root: {})
    monkeypatch.setattr(adapter, "_verify_additive_am94", lambda root: {})

    def inspect_target(target: str, root: Path) -> None:
        assert adapter.g10_spec_compatibility.load(tmp_path) is sentinel

    monkeypatch.setattr(adapter, "verify_am94_boundary", lambda root, outcomes_allowed: sentinel)
    monkeypatch.setattr(adapter, "_execute_target", inspect_target)
    adapter.run("g8_f_sampler_plan_check", tmp_path)
    assert adapter.g10_spec_compatibility.load is original_load


def test_target_failure_propagates_as_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter, "_verify_terminal", lambda root: {})
    monkeypatch.setattr(adapter, "_verify_additive_am94", lambda root: {})

    def fail_target(target: str, root: Path) -> None:
        raise SystemExit(7)

    monkeypatch.setattr(adapter, "_execute_target", fail_target)
    assert adapter.main(["g8_f_sampler_plan_check"]) == 7


def test_main_rejects_arbitrary_python_script() -> None:
    with pytest.raises(SystemExit):
        adapter.main(["/tmp/arbitrary.py"])
