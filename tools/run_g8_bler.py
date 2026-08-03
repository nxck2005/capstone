#!/usr/bin/env python3
"""Run one bounded G8_B smoke transaction or authorize a future full run."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from baseline import g8_bler_contract as bler_contract  # noqa: E402
from baseline import g8_bler_runner as runner  # noqa: E402
from baseline.g8_campaign import CAMPAIGN_STATE, REPO_ROOT, rendered_json  # noqa: E402
import verify_g8_bounded_smoke as smoke_verifier  # noqa: E402
import migrate_g8_bler_runner_contract as runner_migration  # noqa: E402


EXIT_SUCCESS = 0
EXIT_INCOMPLETE = 2
EXIT_CONFLICT = 3
EXIT_HOLD = 4


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execution-class",
        required=True,
        choices=(runner.EXECUTION_CLASS_BOUNDED_SMOKE, runner.EXECUTION_CLASS_FULL_STRENGTH),
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repair-recoverable", action="store_true")
    parser.add_argument("--max-units", type=int, default=runner.BOUNDED_SMOKE_MAX_WORK_UNITS)
    parser.add_argument("--work-unit-id")
    return parser


def _write_smoke_record(record: dict) -> tuple[Path, str]:
    path = REPO_ROOT / "results/baseline/g8/bounded_smoke_record.json"
    expected_existing_sha256 = None
    expected_existing_runner_contract_id = None
    expected_existing_runner_contract_sha256 = None
    if path.exists():
        existing = path.read_bytes()
        existing_sha256 = bler_contract.sha256_bytes(existing)
        try:
            provisional = json.loads(existing)
        except (TypeError, ValueError):
            raise runner.RunnerConflictError("existing smoke record is not decodable JSON")
        if not isinstance(provisional, dict):
            raise runner.RunnerConflictError("existing smoke record is not an object")
        if provisional.get("schema_version") == 1:
            # The old provisional path is retained for historical tests only;
            # the official registered replacement below is schema-2 -> v3.
            expected_existing_sha256 = existing_sha256
        elif (
            provisional.get("schema_version") == runner.SMOKE_RECORD_SCHEMA_VERSION
            and existing_sha256 == runner_migration.OLD_SMOKE_SHA256
            and len(existing) == runner_migration.OLD_SMOKE_BYTES
            and provisional.get("bler_runner_contract_id") == runner_migration.OLD_SMOKE_RUNNER_ID
            and provisional.get("bler_runner_contract_sha256") == runner_migration.OLD_SMOKE_RUNNER_SHA256
        ):
            expected_existing_sha256 = runner_migration.OLD_SMOKE_SHA256
            expected_existing_runner_contract_id = runner_migration.OLD_SMOKE_RUNNER_ID
            expected_existing_runner_contract_sha256 = runner_migration.OLD_SMOKE_RUNNER_SHA256
        else:
            raise runner.RunnerConflictError("existing smoke record is not the exact guarded v2 record")
    body = rendered_json(record)
    digest = runner.publish_smoke_record_atomic(
        path,
        record,
        expected_provisional_sha256=expected_existing_sha256 if expected_existing_runner_contract_id is None else None,
        expected_existing_sha256=expected_existing_sha256 if expected_existing_runner_contract_id is not None else None,
        expected_existing_runner_contract_id=expected_existing_runner_contract_id,
        expected_existing_runner_contract_sha256=expected_existing_runner_contract_sha256,
    )
    return path, digest


def _verify_candidate_smoke(record: dict, context: runner.AuthenticatedRunnerContext) -> None:
    """Verify a complete candidate against a projected v3-bound state."""

    runner_path = context.runner_contract_path
    runner_raw = runner_path.read_bytes()
    candidate_binding = {
        "path": runner.RUNNER_CONTRACT_REPO_RELATIVE_PATH,
        "sha256": bler_contract.sha256_bytes(runner_raw),
        "bytes": len(runner_raw),
    }
    record_raw = rendered_json(record)
    smoke_binding = {
        "path": runner.SMOKE_RECORD_REPO_RELATIVE_PATH,
        "sha256": bler_contract.sha256_bytes(record_raw),
        "bytes": len(record_raw),
    }
    with tempfile.TemporaryDirectory(prefix="g8-b5-smoke-candidate-") as directory:
        candidate_root = Path(directory)
        candidate_record = candidate_root / "bounded_smoke_record.json"
        candidate_record.write_bytes(record_raw)
        state = json.loads(CAMPAIGN_STATE.read_bytes())
        for entry in state["identity"]["produced_artifacts"]:
            if entry["path"] == runner.RUNNER_CONTRACT_REPO_RELATIVE_PATH:
                entry.update(candidate_binding)
            elif entry["path"] == runner.SMOKE_RECORD_REPO_RELATIVE_PATH:
                entry.update(smoke_binding)
        projected_state = candidate_root / "campaign_state.json"
        projected_state.write_bytes(rendered_json(state))
        smoke_verifier.verify(
            candidate_record,
            campaign_state_path=projected_state,
            runner_contract_path=runner_path,
        )


def _finish_installed_v3_smoke() -> bool:
    """Recover the post-replacement/pre-state-publication interruption."""

    record_path = REPO_ROOT / runner.SMOKE_RECORD_REPO_RELATIVE_PATH
    contract_path = runner.DEFAULT_RUNNER_CONTRACT_PATH
    if not record_path.exists() or not contract_path.exists():
        return False
    try:
        contract = json.loads(contract_path.read_bytes())
        record = json.loads(record_path.read_bytes())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(contract, dict) or not isinstance(record, dict):
        return False
    if contract.get("schema_version") != runner.RUNNER_CONTRACT_SCHEMA_VERSION:
        return False
    if record.get("schema_version") != runner.SMOKE_RECORD_SCHEMA_VERSION:
        return False
    if record.get("bler_runner_contract_id") != contract.get("contract_id") or record.get("bler_runner_contract_sha256") != bler_contract.sha256_bytes(contract_path.read_bytes()):
        return False
    try:
        runner_migration.migrate()
        verified = smoke_verifier.verify(record_path)
    except Exception as exc:
        raise runner.RunnerAuthorizationError(f"installed v3 smoke recovery failed: {exc}") from exc
    print(
        "G8 bounded smoke PASS: "
        f"units={len(verified['selected_work_units'])} trials_per_unit={runner.BOUNDED_SMOKE_MAX_TRIALS} "
        f"record={record_path.relative_to(REPO_ROOT)} sha256={bler_contract.sha256_bytes(record_path.read_bytes())} recovered_existing=true"
    )
    return True


def _remove_isolated_root(root: Path) -> None:
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise runner.RunnerPublicationError(
                f"refusing to remove a non-directory smoke root: {root}"
            )
        if runner._root_is_production_alias(root):
            raise runner.RunnerAuthorizationError(
                "refusing to remove the production runtime root"
            )
        shutil.rmtree(root)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.root.is_absolute():
        print("G8 B4 runner HOLD: --root must be an absolute path", file=sys.stderr)
        return EXIT_HOLD
    if args.execution_class == runner.EXECUTION_CLASS_BOUNDED_SMOKE:
        if args.device != "cpu":
            print("G8 B4 runner HOLD: bounded smoke is CPU-only", file=sys.stderr)
            return EXIT_HOLD
        if args.work_unit_id is not None:
            print("G8 B4 runner HOLD: --work-unit-id is full-strength-only", file=sys.stderr)
            return EXIT_HOLD
        if args.max_units != runner.official_smoke_unit_count():
            print(
                "G8 B4 runner HOLD: official bounded smoke requires exactly "
                f"{runner.official_smoke_unit_count()} units",
                file=sys.stderr,
            )
            return EXIT_HOLD
        try:
            if _finish_installed_v3_smoke():
                return EXIT_SUCCESS
        except runner.G8BlerRunnerError as exc:
            print(f"G8 B5 runner HOLD: {exc}", file=sys.stderr)
            return EXIT_HOLD
    elif args.max_units != runner.BOUNDED_SMOKE_MAX_WORK_UNITS:
        print("G8 B4 runner HOLD: --max-units is smoke-only", file=sys.stderr)
        return EXIT_HOLD

    root_was_absent = not args.root.exists()
    runtime_root: Path | None = (
        args.root if args.execution_class == runner.EXECUTION_CLASS_BOUNDED_SMOKE else None
    )
    try:
        context = runner.AuthenticatedRunnerContext()
        if args.execution_class == runner.EXECUTION_CLASS_FULL_STRENGTH:
            # This gate is intentionally before any root creation, request
            # publication, adapter construction, bit generation or decoding.
            runner.authorize_execution(
                context,
                runner.EXECUTION_CLASS_FULL_STRENGTH,
                root=args.root,
            )
            if args.work_unit_id is None:
                raise runner.RunnerAuthorizationError(
                    "full-strength mode requires an exact --work-unit-id"
                )
            outcome = runner.run_one_unit(
                context,
                execution_class=runner.EXECUTION_CLASS_FULL_STRENGTH,
                root=args.root,
                work_unit_id=args.work_unit_id,
                shard_count=args.shard_count,
                shard_index=args.shard_index,
                batch_size=args.batch_size,
                device=args.device,
            )
            print(
                "G8 full-strength transaction PASS: "
                f"work_unit_id={outcome['work_unit_id']} "
                f"attempt={outcome['attempt']}"
            )
            return EXIT_SUCCESS

        outcomes, runtime_root = runner.run_bounded_smoke(
            context,
            root=args.root,
            device=args.device,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
            batch_size=args.batch_size,
            max_units=args.max_units,
            repair_recoverable=args.repair_recoverable,
        )
        record = runner.build_bounded_smoke_record(
            context,
            outcomes,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
            batch_size=args.batch_size,
            production_root_used=False,
            temporary_root_removed=False,
        )
        _remove_isolated_root(runtime_root)
        record["temporary_root_removed"] = True
        try:
            _verify_candidate_smoke(record, context)
        except Exception as exc:
            print(f"G8 B5 runner HOLD: candidate smoke failed independent verification: {exc}", file=sys.stderr)
            return EXIT_HOLD
        path, digest = _write_smoke_record(record)
        try:
            runner_migration.migrate()
        except Exception as exc:
            print(f"G8 B5 runner HOLD: smoke installed but campaign binding migration failed: {exc}", file=sys.stderr)
            return EXIT_HOLD
        try:
            smoke_verifier.verify(path)
        except smoke_verifier.SmokeVerificationError as exc:
            print(f"G8 B4 runner HOLD: installed smoke record failed independent verification: {exc}", file=sys.stderr)
            return EXIT_HOLD
        print(
            "G8 bounded smoke PASS: "
            f"units={len(outcomes)} trials_per_unit={runner.BOUNDED_SMOKE_MAX_TRIALS} "
            f"record={path.relative_to(REPO_ROOT)} sha256={digest}"
        )
        return EXIT_SUCCESS
    except runner.RunnerConflictError as exc:
        print(f"G8 B4 runner CONFLICT: {exc}", file=sys.stderr)
        return EXIT_CONFLICT
    except runner.RunnerExecutionError as exc:
        print(f"G8 B4 runner INCOMPLETE: {exc}", file=sys.stderr)
        return EXIT_INCOMPLETE
    except runner.G8BlerRunnerError as exc:
        print(f"G8 B4 runner HOLD: {exc}", file=sys.stderr)
        return EXIT_HOLD
    finally:
        # Cleanup is limited to a root that this invocation was explicitly
        # given and that did not exist at authorization time.  A pre-existing
        # path is never deleted after a failed authorization.
        if runtime_root is not None and root_was_absent and runtime_root.exists():
            _remove_isolated_root(runtime_root)


if __name__ == "__main__":
    raise SystemExit(main())
