#!/usr/bin/env python3
"""Run an allowlisted historical read-only check after terminal G-10."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from evaluation import am96_spec_compatibility, am97_spec_compatibility, g10_spec_compatibility  # noqa: E402
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
    w6_module = target_globals["w6"]
    original_w8_normative_hashes = dict(w6_module._W8_CURRENT_NORMATIVE_SHA256)
    w6_module._W8_CURRENT_NORMATIVE_SHA256 = {
        "normative_spec": am97_spec_compatibility.VIEW_HASHES[0][4],
        "resolved_params": am97_spec_compatibility.VIEW_HASHES[1][4],
    }

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
        w6_module._W8_CURRENT_NORMATIVE_SHA256 = original_w8_normative_hashes
    if isinstance(result, int) and result != 0:
        raise SystemExit(result)


def _run_w6_evidence_tool(target: str, root: Path) -> None:
    """Run a W6-A reader with an in-memory AM-96 normative projection."""

    script, *arguments = TARGETS[target]
    script_path = root / script
    sys.argv = [str(script_path), *arguments]
    namespace = runpy.run_path(str(script_path), run_name=f"_post_g10_{target}")
    w6_module = sys.modules["baseline.w6_evidence"]
    original_hashes = dict(w6_module._W8_CURRENT_NORMATIVE_SHA256)
    w6_module._W8_CURRENT_NORMATIVE_SHA256 = {
        "normative_spec": am97_spec_compatibility.VIEW_HASHES[0][4],
        "resolved_params": am97_spec_compatibility.VIEW_HASHES[1][4],
    }
    try:
        result = namespace["main"]()
    finally:
        w6_module._W8_CURRENT_NORMATIVE_SHA256 = original_hashes
    if isinstance(result, int) and result != 0:
        raise SystemExit(result)


def _run_w8_a(root: Path) -> None:
    """Run W8-A while adapting its nested historical W7-G4 subprocess."""

    script_path = root / "tools/verify_w8_a.py"
    sys.argv = [str(script_path), "--skip-data"]
    namespace = runpy.run_path(str(script_path), run_name="_post_g10_w8_a")
    target_globals = namespace["main"].__globals__
    original_w7_verifier = target_globals["_run_w7_g4_verifier"]
    authorization_module = sys.modules["gen_w8_execution_authorization"]
    original_authorization_predecessor = authorization_module._am94_predecessor_config_bindings

    def predecessor_config_bindings(*, role: str = authorization_module.W8_CORE_ROLE) -> list[dict[str, Any]]:
        """Project current W8 configs back through AM-94..AM-97 in memory."""

        names = []
        for path in (*authorization_module.AM94_ALLOWED_PARAMETER_PATHS, *am97_spec_compatibility.ALLOWED_PARAMETER_PATHS):
            prefix = "evaluation."
            if not path.startswith(prefix) or "." in path[len(prefix):]:
                raise ValueError("AM-94 compatibility contains a non-evaluation leaf")
            names.append(path[len(prefix):])
        values: list[dict[str, Any]] = []
        for cell in authorization_module.run_cells():
            config = authorization_module.load_w8_config(
                cell.ratio, cell.train_seed, cell.channel_seed, role=role
            )
            historical = config.to_dict()
            evaluation = historical["parameters"]["evaluation"]
            for name in names:
                if name not in evaluation:
                    raise ValueError(f"AM-94 parameter is absent from W8 config: {name}")
                del evaluation[name]
            channel = historical["parameters"]["channel"]
            for name in (
                "train_snr_randomisation_distribution",
                "train_snr_randomisation_rng_purpose",
                "train_snr_randomisation_unit",
            ):
                channel.pop(name, None)
            digital = historical["parameters"]["digital_semantic_control"]
            for name in am96_spec_compatibility.ALLOWED_PARAMETER_PATHS:
                prefix = "digital_semantic_control."
                if not name.startswith(prefix):
                    continue
                digital.pop(name[len(prefix):], None)
            artifacts = historical["parameters"]["artifacts"]
            artifacts["rng_purposes"] = [
                purpose for purpose in artifacts["rng_purposes"]
                if purpose != "er2_snr_randomised_v1"
            ]
            artifacts["rng_identity_fields"].pop("er2_snr_randomised_v1", None)
            values.append(
                {
                    "run_index": cell.run_index,
                    "config_hash": authorization_module.run_config_canonical_sha256(
                        {
                            "fingerprint_schema_version": historical["fingerprint_schema_version"],
                            "resolved": historical["resolved"],
                            "parameters": historical["parameters"],
                        }
                    ),
                    "protocol_config_hash": authorization_module.run_config_canonical_sha256(
                        {"protocol": authorization_module.protocol_descriptor(), "config": historical}
                    ),
                }
            )
        return values

    authorization_module._am94_predecessor_config_bindings = predecessor_config_bindings
    target_globals["_am94_predecessor_config_bindings"] = predecessor_config_bindings
    run_w8_module = sys.modules.get("run_w8_campaign")
    original_run_w8_predecessor = None
    if run_w8_module is not None and hasattr(run_w8_module, "_am94_predecessor_config_bindings"):
        original_run_w8_predecessor = run_w8_module._am94_predecessor_config_bindings
        run_w8_module._am94_predecessor_config_bindings = predecessor_config_bindings

    def run_w7_verifier(repo: Path) -> None:
        _execute_target("w7_g4", root)

    target_globals["_run_w7_g4_verifier"] = run_w7_verifier
    try:
        result = namespace["main"]()
    finally:
        target_globals["_run_w7_g4_verifier"] = original_w7_verifier
        authorization_module._am94_predecessor_config_bindings = original_authorization_predecessor
        if run_w8_module is not None and original_run_w8_predecessor is not None:
            run_w8_module._am94_predecessor_config_bindings = original_run_w8_predecessor
    if isinstance(result, int) and result != 0:
        raise SystemExit(result)


def _execute_target(target: str, root: Path = REPO) -> None:
    if target == "g8_f0_authorization":
        _run_f0_authorization()
        return
    if target == "w6_complete":
        _run_w6_complete(root)
        return
    if target in {"w6_classical_build_check", "w6_classical_verify"}:
        _run_w6_evidence_tool(target, root)
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
    def additive_load(load_root: Path = root) -> dict[str, Any]:
        load_root = Path(load_root)
        historical = verify_am94_boundary(load_root, outcomes_allowed=True)
        # Keep the test seam's sentinel/stand-in behavior intact.  The normal
        # repository path returns the AM-94 mapping and is then projected to
        # the authenticated AM-96 current bytes below.
        if not isinstance(historical, dict):
            return historical
        successor = am96_spec_compatibility.load(load_root)
        successor_entries = {entry["path"]: entry for entry in successor["entries"]}
        projection = json.loads(json.dumps(historical))
        for entry in projection["entries"]:
            successor = successor_entries[entry["path"]]
            entry["current_bytes"] = successor["current_bytes"]
            entry["current_sha256"] = successor["current_sha256"]
        return projection
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
