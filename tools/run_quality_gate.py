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
    "and not external_codec_runtime and not historical_profile_artifact"
)


def _python_tool(path: str, *args: str) -> list[str]:
    return [PYTHON, str(REPO / path), *args]


def _static_commands() -> tuple[list[str], ...]:
    return (
        _python_tool("tools/gen_spec_views.py", "--check"),
        _python_tool("tools/check_doc_consistency.py", "-v"),
        _python_tool("tools/check_literals.py", "-v"),
        _python_tool("tools/gen_g8_f_corpus_plan.py", "--check"),
        _python_tool("tools/verify_g8_f_corpus_plan.py"),
        _python_tool("tools/gen_g8_f_sampler_plan.py", "--check"),
        _python_tool("tools/verify_g8_f_sampler_plan.py"),
        [
            PYTHON,
            "-c",
            "from baseline.g8_f_f0 import verify_f0_authorization; "
            "value = verify_f0_authorization(require_zero_prefix=False); "
            "print('G8_F F0 offline authentication PASS:', value['authorization_id'])",
        ],
        _python_tool("tools/closeout_g8_f_f1.py", "verify"),
        _python_tool("tools/closeout_g8_f_f2.py", "verify"),
        _python_tool("tools/closeout_g8.py", "verify"),
        _python_tool("tools/verify_w5_training_system.py"),
        _python_tool("tools/gen_g8_campaign_manifest.py", "--check"),
        _python_tool("tools/gen_g8_bler_tooling_contract.py", "--check"),
        _python_tool("tools/verify_g8_bler_tooling_contract.py"),
        _python_tool("tools/gen_g8_bler_state_contract.py", "--check"),
        _python_tool("tools/gen_g8_bler_resume_contract.py", "--check"),
        _python_tool("tools/verify_g8_bler_resume_contract.py"),
        _python_tool("tools/verify_g8_bler_runner_contract_offline.py"),
        _python_tool("tools/verify_g8_bler_characterization_manifest_v2.py"),
        _python_tool("tools/verify_w4_baseline_integration.py"),
        _python_tool("tools/verify_g2_adjudication.py"),
        _python_tool("tools/build_w6_classical_evidence.py", "--check"),
        _python_tool("tools/verify_w6_classical_evidence.py", "--no-upstream"),
        _python_tool("tools/verify_w6_complete.py"),
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
                "and not historical_profile_artifact"
            )
        return (*_static_commands(), [PYTHON, "-m", "pytest", "-q", "-m", selection])
    if profile == "evidence":
        return (
            _python_tool("tools/verify_g8_evidence_readonly.py"),
            [PYTHON, "-m", "pytest", "-q", "tests/test_g8_bler_characterization_v2.py"],
            ["git", "diff", "--check"],
        )
    if profile == "full-local":
        return (*_static_commands(), [PYTHON, "-m", "pytest", "-q"])
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
