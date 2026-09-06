from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import run_quality_gate as gate


_CONFTEST_SPEC = importlib.util.spec_from_file_location(
    "capstone_test_conftest", Path(__file__).with_name("conftest.py")
)
assert _CONFTEST_SPEC is not None and _CONFTEST_SPEC.loader is not None
test_conftest = importlib.util.module_from_spec(_CONFTEST_SPEC)
_CONFTEST_SPEC.loader.exec_module(test_conftest)


def _joined(profile: str) -> str:
    return "\n".join(" ".join(command) for command in gate.profile_commands(profile))


def test_evidence_profile_has_no_execution_entrypoints():
    commands = _joined("evidence")
    assert "run_one_unit" not in commands
    assert "run_g8_bler_characterization" not in commands
    assert "verify_g8_evidence_readonly.py" in commands
    assert "tests/test_g8_bler_characterization_v2.py" in commands


def test_software_profile_uses_offline_runner_verification():
    commands = _joined("static")
    assert "verify_g8_bler_runner_contract_offline.py" in commands
    assert "verify_g8_bler_characterization_manifest_v2.py" in commands
    assert "run_post_g10_historical_check.py w5_training_system" in commands
    assert "verify_w5_training_system.py --pre-source" not in commands
    assert "gen_g8_bler_runner_contract.py" not in commands


def test_cpu_profile_excludes_only_audited_nonportable_categories(monkeypatch):
    monkeypatch.delenv("CAPSTONE_INCLUDE_EXTERNAL_LDPC_FIXTURE", raising=False)
    commands = _joined("ci-cpu")
    assert "not primary_runtime and not external_ldpc_fixture" in commands
    assert "not external_dataset and not frozen_checkpoint" in commands
    assert "not external_codec_runtime and not historical_profile_artifact" in commands
    assert "and not historical_pre_g10" in commands


def test_weekly_cpu_profile_adds_only_the_external_ldpc_fixture(monkeypatch):
    monkeypatch.setenv("CAPSTONE_INCLUDE_EXTERNAL_LDPC_FIXTURE", "1")
    commands = _joined("ci-cpu")
    assert "not primary_runtime and not external_dataset" in commands
    assert "not external_codec_runtime and not historical_profile_artifact" in commands
    assert "and not historical_pre_g10" in commands
    assert "external_ldpc_fixture" not in commands


def _g10_tool(commands: tuple[list[str], ...]) -> str:
    assert len(commands) == 1
    return Path(commands[0][1]).name


def _mark(tmp_path: Path, relative: Path) -> None:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_g10_phase_selector_uses_historical_freeze_only_without_authority(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(gate, "REPO", tmp_path)

    assert _g10_tool(gate._g10_commands()) == "verify_g10_semantics_freeze.py"


def test_g10_phase_selector_uses_read_only_authority_verifier_pre_execution(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(gate, "REPO", tmp_path)
    _mark(tmp_path, gate.G10_AUTHORIZATION_V2)

    selected = _g10_tool(gate._g10_commands())
    assert selected == "verify_g10_authority.py"
    assert selected != "verify_g10_semantics_freeze.py"
    assert selected != "verify_g10_w9.py"


def test_g10_phase_selector_uses_terminal_verifier_for_completion(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(gate, "REPO", tmp_path)
    _mark(tmp_path, gate.G10_AUTHORIZATION_V2)
    _mark(tmp_path, gate.G10_COMPLETION)

    assert _g10_tool(gate._g10_commands()) == "verify_g10_w9.py"


def test_g10_phase_selector_uses_terminal_verifier_for_reconciliation(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(gate, "REPO", tmp_path)
    _mark(tmp_path, gate.G10_AUTHORIZATION_V2)
    _mark(tmp_path, gate.G10_COMPLETION)
    _mark(tmp_path, gate.G10_RECONCILIATION)

    assert _g10_tool(gate._g10_commands()) == "verify_g10_w9.py"


def test_g10_phase_selector_fails_closed_on_reconciliation_without_completion(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(gate, "REPO", tmp_path)
    _mark(tmp_path, gate.G10_AUTHORIZATION_V2)
    _mark(tmp_path, gate.G10_RECONCILIATION)

    assert _g10_tool(gate._g10_commands()) == "verify_g10_w9.py"


def test_g10_phase_selector_treats_unsafe_symlink_sentinels_as_present(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(gate, "REPO", tmp_path)
    terminal = tmp_path / gate.G10_COMPLETION
    terminal.parent.mkdir(parents=True, exist_ok=True)
    terminal.symlink_to(tmp_path / "missing-completion.json")
    assert _g10_tool(gate._g10_commands()) == "verify_g10_w9.py"

    monkeypatch.setattr(gate, "REPO", tmp_path / "authority-only")
    authority = gate.REPO / gate.G10_AUTHORIZATION_V2
    authority.parent.mkdir(parents=True, exist_ok=True)
    authority.symlink_to(gate.REPO / "missing-authority.json")
    assert _g10_tool(gate._g10_commands()) == "verify_g10_authority.py"


def test_current_terminal_checkout_selects_only_terminal_g10_verifier():
    selected = _joined("static")
    assert "tools/verify_g10_w9.py" in selected
    assert "tools/verify_g10_semantics_freeze.py" not in selected
    assert "tools/verify_g10_authority.py" not in selected


def test_affected_historical_check_is_direct_before_terminal_g10(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "REPO", tmp_path)
    direct = ["python", "tools/verify_w5_training_system.py"]

    assert gate._historical_command("w5_training_system", direct) == direct


def test_affected_historical_check_uses_adapter_after_terminal_g10(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "REPO", tmp_path)
    _mark(tmp_path, gate.G10_COMPLETION)
    direct = ["python", "tools/verify_w5_training_system.py"]

    selected = gate._historical_command("w5_training_system", direct)
    assert Path(selected[1]).name == Path(gate.POST_G10_HISTORICAL_ADAPTER).name
    assert selected[2] == "w5_training_system"


def test_terminal_collection_marks_only_exact_pre_science_g10_tests(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(test_conftest, "REPO", tmp_path)
    terminal = tmp_path / test_conftest.G10_COMPLETION
    terminal.parent.mkdir(parents=True, exist_ok=True)
    terminal.write_bytes(b"terminal sentinel")

    ordinary = SimpleNamespace(
        fspath=tmp_path / "tests/test_g8_d_contract.py",
        nodeid="tests/test_g8_d_contract.py::test_table_identity_round_trips_wrapped_schema",
        markers=[],
    )
    pre_science = SimpleNamespace(
        fspath=tmp_path / "tests/test_g10_protocol.py",
        nodeid="tests/test_g10_protocol.py::test_am94_boundary_remains_pre_science",
        markers=[],
    )
    ordinary.add_marker = ordinary.markers.append
    pre_science.add_marker = pre_science.markers.append

    test_conftest.pytest_collection_modifyitems(None, [ordinary, pre_science])

    assert ordinary.markers == []
    assert len(pre_science.markers) == 1


def test_pre_science_tests_are_not_marked_before_terminal_g10(tmp_path, monkeypatch):
    monkeypatch.setattr(test_conftest, "REPO", tmp_path)
    item = SimpleNamespace(
        fspath=tmp_path / "tests/test_g10_protocol.py",
        nodeid="tests/test_g10_protocol.py::test_no_outcome_files_exist_before_authority",
        markers=[],
    )
    item.add_marker = item.markers.append

    test_conftest.pytest_collection_modifyitems(None, [item])

    assert item.markers == []


def test_unsafe_terminal_symlink_still_activates_terminal_phase(tmp_path, monkeypatch):
    monkeypatch.setattr(test_conftest, "REPO", tmp_path)
    terminal = tmp_path / test_conftest.G10_RECONCILIATION
    terminal.parent.mkdir(parents=True, exist_ok=True)
    terminal.symlink_to(tmp_path / "missing-reconciliation.json")
    item = SimpleNamespace(
        fspath=tmp_path / "tests/test_g10_protocol.py",
        nodeid="tests/test_g10_protocol.py::test_am94_boundary_remains_pre_science",
        markers=[],
    )
    item.add_marker = item.markers.append

    test_conftest.pytest_collection_modifyitems(None, [item])

    assert len(item.markers) == 1


def test_full_local_excludes_exact_phase_marker_only_at_terminal_boundary(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(gate, "REPO", tmp_path)
    before = "\n".join(" ".join(command) for command in gate.profile_commands("full-local"))
    assert "-m not historical_pre_g10" not in before

    completion = tmp_path / gate.G10_COMPLETION
    completion.parent.mkdir(parents=True, exist_ok=True)
    completion.symlink_to(tmp_path / "missing-completion.json")
    after = "\n".join(" ".join(command) for command in gate.profile_commands("full-local"))
    assert "-m not historical_pre_g10" in after
    assert "not primary_runtime" not in after


def test_post_g10_test_context_restores_strict_compatibility_loader():
    original = test_conftest.g10_spec_compatibility.load
    with test_conftest._post_g10_am94_context():
        assert test_conftest.g10_spec_compatibility.load is not original
    assert test_conftest.g10_spec_compatibility.load is original
