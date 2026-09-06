#!/usr/bin/env python3
"""Run an allowlisted historical read-only check after terminal G-10."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from evaluation import g10_spec_compatibility  # noqa: E402
from evaluation.g10_protocol import (  # noqa: E402
    G10ProtocolHold,
    RECONCILIATION_PATH,
    COMPLETION_PATH,
    verify_am94_boundary,
)
from verify_g10_w9 import verify as verify_g10_terminal  # noqa: E402


class PostG10HistoricalCheckHold(RuntimeError):
    """The terminal evidence or target allowlist is not valid."""


TARGETS: dict[str, tuple[str, ...]] = {
    "g8_f_sampler_plan_check": ("tools/gen_g8_f_sampler_plan.py", "--check"),
    "g8_f1_closeout": ("tools/closeout_g8_f_f1.py", "verify"),
    "w5_training_system": ("tools/verify_w5_training_system.py",),
    "g8_campaign_manifest_check": ("tools/gen_g8_campaign_manifest.py", "--check"),
    "w4_baseline_integration": ("tools/verify_w4_baseline_integration.py",),
    "w6_classical_build_check": ("tools/build_w6_classical_evidence.py", "--check"),
    "w6_classical_verify": ("tools/verify_w6_classical_evidence.py", "--no-upstream"),
    "w6_complete": ("tools/verify_w6_complete.py",),
    "w7_g4": ("tools/verify_w7_g4.py", "verify"),
    "w8_a": ("tools/verify_w8_a.py", "--skip-data"),
}


def _require_terminal(root: Path = REPO) -> None:
    for relative in (COMPLETION_PATH, RECONCILIATION_PATH):
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise PostG10HistoricalCheckHold(f"unsafe or missing terminal sentinel: {relative}")


def _verify_terminal(root: Path = REPO) -> dict[str, Any]:
    _require_terminal(root)
    return verify_g10_terminal(root)


def _verify_additive_am94(root: Path = REPO) -> dict[str, Any]:
    return verify_am94_boundary(root, outcomes_allowed=True)


def _run_f0_authorization() -> None:
    from baseline.g8_f_f0 import verify_f0_authorization

    value = verify_f0_authorization(require_zero_prefix=False)
    print("G8_F F0 offline authentication PASS:", value["authorization_id"])


def _run_w6_complete(root: Path) -> None:
    """Run W6 in-process, adapting only its nested historical W4 check."""

    script_path = root / "tools/verify_w6_complete.py"
    sys.argv = [str(script_path)]
    namespace = runpy.run_path(str(script_path), run_name="_post_g10_w6_complete")
    target_globals = namespace["main"].__globals__
    original_run_tool = target_globals["_run_tool"]

    def run_tool(path: Path, *arguments: str) -> str:
        if path.resolve() == (root / "tools/verify_w4_baseline_integration.py").resolve():
            _execute_target("w4_baseline_integration", root)
            return ""
        return original_run_tool(path, *arguments)

    target_globals["_run_tool"] = run_tool
    try:
        result = namespace["main"]()
    finally:
        target_globals["_run_tool"] = original_run_tool
    if isinstance(result, int) and result != 0:
        raise SystemExit(result)


def _run_w8_a(root: Path) -> None:
    """Run W8-A while adapting its nested historical W7-G4 subprocess."""

    script_path = root / "tools/verify_w8_a.py"
    sys.argv = [str(script_path), "--skip-data"]
    namespace = runpy.run_path(str(script_path), run_name="_post_g10_w8_a")
    target_globals = namespace["main"].__globals__
    original_w7_verifier = target_globals["_run_w7_g4_verifier"]

    def run_w7_verifier(repo: Path) -> None:
        _execute_target("w7_g4", root)

    target_globals["_run_w7_g4_verifier"] = run_w7_verifier
    try:
        result = namespace["main"]()
    finally:
        target_globals["_run_w7_g4_verifier"] = original_w7_verifier
    if isinstance(result, int) and result != 0:
        raise SystemExit(result)


def _execute_target(target: str, root: Path = REPO) -> None:
    if target == "g8_f0_authorization":
        _run_f0_authorization()
        return
    if target == "w6_complete":
        _run_w6_complete(root)
        return
    if target == "w8_a":
        _run_w8_a(root)
        return
    target_args = TARGETS.get(target)
    if target_args is None:
        raise PostG10HistoricalCheckHold(f"target is not allowlisted: {target}")
    script, *arguments = target_args
    script_path = root / script
    sys.argv = [str(script_path), *arguments]
    runpy.run_path(str(script_path), run_name="__main__")


def run(target: str, root: Path = REPO) -> None:
    if target != "g8_f0_authorization" and target not in TARGETS:
        raise PostG10HistoricalCheckHold(f"target is not allowlisted: {target}")
    _verify_terminal(root)
    _verify_additive_am94(root)
    original_load = g10_spec_compatibility.load
    additive_load = lambda load_root=root: verify_am94_boundary(  # noqa: E731
        Path(load_root), outcomes_allowed=True
    )
    try:
        g10_spec_compatibility.load = additive_load
        _execute_target(target, root)
    finally:
        g10_spec_compatibility.load = original_load


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("g8_f0_authorization", *TARGETS))
    args = parser.parse_args(argv)
    try:
        run(args.target)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        return code
    except (G10ProtocolHold, OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"post-G10 historical check HOLD — {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
