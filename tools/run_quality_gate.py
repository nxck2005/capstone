#!/usr/bin/env python3
"""Run the repository-owned CI profiles locally or in GitHub Actions.

Profiles are intentionally explicit.  The evidence profile calls only the
offline evidence verifier and its focused regression tests; it never starts a
characterization coordinator.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
CPU_TEST_SELECTION = (
    "not primary_runtime and not external_ldpc_fixture "
    "and not external_dataset and not frozen_checkpoint "
    "and not external_codec_runtime and not historical_profile_artifact "
    "and not historical_pre_g10"
)
G10_AUTHORIZATION_V2 = Path("results/learned/w9/g10_execution_authorization_v2.json")
G10_COMPLETION = Path("results/learned/w9/w9a_completion.json")
G10_RECONCILIATION = Path("results/learned/w9/w9a_reconciliation.json")
POST_G10_HISTORICAL_ADAPTER = "tools/run_post_g10_historical_check.py"
FULL_LOCAL_TEST_MARKER = "not historical_pre_g10"


def _python_tool(path: str, *args: str) -> list[str]:
    return [PYTHON, str(REPO / path), *args]


def _present(path: Path) -> bool:
    """Treat symlinks as present so unsafe sentinels cannot weaken routing."""

    return path.exists() or path.is_symlink()


def _g10_commands() -> tuple[list[str], ...]:
    """Select the strict verifier for the repository's G-10 lifecycle phase."""

    if _present(REPO / G10_COMPLETION) or _present(REPO / G10_RECONCILIATION):
        return (_python_tool("tools/verify_g10_w9.py"),)
    if _present(REPO / G10_AUTHORIZATION_V2):
        return (_python_tool("tools/verify_g10_authority.py"),)
    return (_python_tool("tools/verify_g10_semantics_freeze.py"),)


def _g10_terminal() -> bool:
    return _present(REPO / G10_COMPLETION) or _present(REPO / G10_RECONCILIATION)


def _historical_command(target: str, direct: list[str]) -> list[str]:
    if _g10_terminal():
        return _python_tool(POST_G10_HISTORICAL_ADAPTER, target)
    return direct


def _full_local_test_command() -> list[str]:
    """Keep full-local broad while omitting only impossible terminal tests."""

    command = [PYTHON, "-m", "pytest", "-q"]
    if _g10_terminal():
        command.extend(["-m", FULL_LOCAL_TEST_MARKER])
    return command


def _w8_a_commands() -> tuple[list[str], ...]:
    """Run the carrier-only W8-A verifier when its immutable records exist.

    The scientific source epoch intentionally has no W8 authority files, so it
    cannot run this carrier check.  Once the later carrier contains the six
    immutable pre-execution records, every software lane authenticates them in
    source-only mode without requiring the ignored dataset archive.
    """

    authority_root = REPO / "results/learned/w8"
    required = (
        authority_root / "w8_source_manifest.json",
        authority_root / "w8_execution_authorization.json",
        authority_root / "w8_a_smoke.json",
        authority_root / "w8_data_verification.json",
        authority_root / "w8_runtime_estimate.json",
        authority_root / "w8_a_completion.json",
    )
    if all(path.is_file() and not path.is_symlink() for path in required):
        return (
            _historical_command(
                "w8_a",
                _python_tool("tools/verify_w8_a.py", "--skip-data"),
            ),
        )
    return ()


def _w8_c_commands() -> tuple[list[str], ...]:
    """Run the compact, read-only W8-C reconciliation checks when published.

    Full worker custody is intentionally not part of a normal checkout.  The
    compact verifier is therefore the carrier-side gate; ``verify-live``
    remains the explicit worker-custody command used at reconciliation time.
    """

    evidence_root = REPO / "results/learned/w8"
    reconciliation = evidence_root / "w8_c_reconciliation.json"
    inventory = evidence_root / "w8_c_root_inventory.jsonl"
    completion = evidence_root / "w8_completion.json"
    authority_files = (
        evidence_root / "w8_r1_source_manifest.json",
        evidence_root / "w8_r1_execution_authorization.json",
        evidence_root / "w8_r1_launch_authorization.json",
        evidence_root / "w8_r1_successor_lineage.json",
    )
    if (
        reconciliation.is_file()
        and inventory.is_file()
        and completion.is_file()
        and all(path.is_file() and not path.is_symlink() for path in authority_files)
    ):
        return (
            _python_tool(
                "tools/verify_w8_c.py",
                "verify-compact",
                "--reconciliation",
                str(reconciliation),
                "--inventory",
                str(inventory),
                "--authority-dir",
                str(evidence_root),
            ),
            _python_tool(
                "tools/verify_w8_c.py",
                "verify-terminal",
                "--completion",
                str(completion),
                "--reconciliation",
                str(reconciliation),
            ),
            [PYTHON, "-m", "pytest", "-q", "tests/test_w8_c_reconciliation.py"],
        )
    return ()


def _static_commands() -> tuple[list[str], ...]:
    return (
        _python_tool("tools/gen_spec_views.py", "--check"),
        *_g10_commands(),
        _python_tool("tools/check_doc_consistency.py", "-v"),
        _python_tool("tools/check_literals.py", "-v"),
        _python_tool("tools/gen_g8_f_corpus_plan.py", "--check"),
        _python_tool("tools/verify_g8_f_corpus_plan.py"),
        _historical_command(
            "g8_f_sampler_plan_check",
            _python_tool("tools/gen_g8_f_sampler_plan.py", "--check"),
        ),
        _python_tool("tools/verify_g8_f_sampler_plan.py"),
        _historical_command(
            "g8_f0_authorization",
            [
                PYTHON,
                "-c",
                "from baseline.g8_f_f0 import verify_f0_authorization; "
                "value = verify_f0_authorization(require_zero_prefix=False); "
                "print('G8_F F0 offline authentication PASS:', value['authorization_id'])",
            ],
        ),
        _historical_command(
            "g8_f1_closeout",
            _python_tool("tools/closeout_g8_f_f1.py", "verify"),
        ),
        _python_tool("tools/closeout_g8_f_f2.py", "verify"),
        _python_tool("tools/closeout_g8.py", "verify"),
        _historical_command(
            "w5_training_system",
            _python_tool("tools/verify_w5_training_system.py"),
        ),
        _historical_command(
            "g8_campaign_manifest_check",
            _python_tool("tools/gen_g8_campaign_manifest.py", "--check"),
        ),
        _python_tool("tools/gen_g8_bler_tooling_contract.py", "--check"),
        _python_tool("tools/verify_g8_bler_tooling_contract.py"),
        _python_tool("tools/gen_g8_bler_state_contract.py", "--check"),
        _python_tool("tools/gen_g8_bler_resume_contract.py", "--check"),
        _python_tool("tools/verify_g8_bler_resume_contract.py"),
        _python_tool("tools/verify_g8_bler_runner_contract_offline.py"),
        _python_tool("tools/verify_g8_bler_characterization_manifest_v2.py"),
        _historical_command(
            "w4_baseline_integration",
            _python_tool("tools/verify_w4_baseline_integration.py"),
        ),
        _python_tool("tools/verify_g2_adjudication.py"),
        _historical_command(
            "w6_classical_build_check",
            _python_tool("tools/build_w6_classical_evidence.py", "--check"),
        ),
        _historical_command(
            "w6_classical_verify",
            _python_tool("tools/verify_w6_classical_evidence.py", "--no-upstream"),
        ),
        _historical_command(
            "w6_complete",
            _python_tool("tools/verify_w6_complete.py"),
        ),
        _python_tool("tools/verify_w7_b1.py", "verify"),
        # W5/W6/B1 are authenticated immediately above; retain the standalone
        # B2R and terminal G-4 checks so every software lane sees the exact
        # upstream boundary that W8 preflight invokes.
        _python_tool("tools/verify_w7_b2r.py", "verify", "--skip-upstream"),
        _historical_command(
            "w7_g4",
            _python_tool("tools/verify_w7_g4.py", "verify"),
        ),
        *_w8_a_commands(),
        *_w8_c_commands(),
        ["git", "diff", "--check"],
    )


def profile_commands(profile: str) -> tuple[list[str], ...]:
    if profile == "static":
        return _static_commands()
    if profile == "ci-cpu":
        selection = CPU_TEST_SELECTION
        if os.environ.get("CAPSTONE_INCLUDE_EXTERNAL_LDPC_FIXTURE") == "1":
            selection = (
                "not primary_runtime and not external_dataset "
                "and not frozen_checkpoint and not external_codec_runtime "
                "and not historical_profile_artifact "
                "and not historical_pre_g10"
            )
        return (*_static_commands(), [PYTHON, "-m", "pytest", "-q", "-m", selection])
    if profile == "evidence":
        return (
            _python_tool("tools/verify_g8_evidence_readonly.py"),
            [PYTHON, "-m", "pytest", "-q", "tests/test_g8_bler_characterization_v2.py"],
            ["git", "diff", "--check"],
        )
    if profile == "full-local":
        return (*_static_commands(), _full_local_test_command())
    raise ValueError(f"unknown quality-gate profile: {profile}")


def _check_clean_checkout() -> None:
    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    if result.stdout.strip():
        raise RuntimeError(f"quality gate requires a clean checkout:\n{result.stdout}")


def run(profile: str) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), str(REPO / "tools"), environment.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    for command in profile_commands(profile):
        print("$ " + " ".join(command), flush=True)
        subprocess.run(command, cwd=REPO, env=environment, check=True)
    if profile in {"static", "ci-cpu", "evidence", "full-local"}:
        _check_clean_checkout()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("static", "ci-cpu", "evidence", "full-local"))
    args = parser.parse_args(argv)
    run(args.profile)
    print(f"quality gate PASS: {args.profile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
