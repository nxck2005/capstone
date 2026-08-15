#!/usr/bin/env python3
"""Run the explicit two-process Pascal G8_C successor coordinator.

``--plan-only`` authenticates the live profile and prints the complete
partition without creating a runtime root.  ``--execute --dry-run`` enters
the same launch gate and child-command construction but never starts a
worker.  A real ``--execute`` additionally requires the owner's detached
launch-authorization record; this task never creates that record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline.g8_pascal_production import (  # noqa: E402
    PRODUCTION_CONTRACT,
    ProductionContractError,
    RecoveryError,
    STATUS_ACCEPTED,
    STATUS_FAILED,
    STATUS_TERMINAL_INVALID,
    authenticate_worker_profile,
    audit_campaign,
    ensure_runtime_root,
    exact_shard_partition,
    inspect_unit,
    reconcile_campaign,
    run_unit,
    successor_bindings,
    validate_production_contracts,
)
from baseline.g8_pascal_successor import (  # noqa: E402
    REQUIRED_COUNT,
    SUCCESSOR_MANIFEST,
    SUCCESSOR_PROFILE_ID,
    SUCCESSOR_ROOT,
    SUCCESSOR_STATE,
    authority_shard,
    canonical_json,
    load_json,
    validate_successor_manifest,
    validate_successor_state,
)
from config.execution_profiles import authenticate_execution_profile, profile_definition  # noqa: E402


GPU_QUERY = ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader,nounits"]
DEVICE_RE = re.compile(r"^cuda:[0-9]+$")
LAUNCH_AUTHORIZATION_ROLE = "g8_c_pascal_successor_launch_authorization"


def _positive_max_units(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--max-units must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--max-units must be a positive integer")
    return parsed


def _validate_max_units(max_units: int | None) -> None:
    if max_units is not None and (type(max_units) is not int or max_units <= 0):
        raise RuntimeError("--max-units must be a positive integer")


def _run_worker_batch(
    *,
    root: Path,
    partition: list[int],
    shard_index: int,
    shard_count: int,
    device: str,
    gpu_uuid: str,
    profile: Mapping[str, Any],
    batch_size: int,
    max_units: int | None,
) -> int:
    """Run at most ``max_units`` attempted units and stop the batch on failure."""

    _validate_max_units(max_units)
    attempted = 0
    for ordinal in partition:
        classification = inspect_unit(root, ordinal)["classification"]
        if classification == STATUS_ACCEPTED:
            continue
        if classification == STATUS_TERMINAL_INVALID:
            raise RecoveryError(f"successor unit {ordinal} is terminal-invalid")
        if max_units is not None and attempted >= max_units:
            break
        attempted += 1
        report = run_unit(
            root,
            ordinal=ordinal,
            shard_index=shard_index,
            shard_count=shard_count,
            device=device,
            gpu_uuid=gpu_uuid,
            profile=profile,
            batch_size=batch_size,
        )
        if report["status"] == STATUS_TERMINAL_INVALID:
            raise RecoveryError(f"successor unit {ordinal} became terminal-invalid")
        if report["status"] == STATUS_FAILED:
            break
        if report["status"] != STATUS_ACCEPTED:
            raise RecoveryError(f"successor unit {ordinal} returned unexpected status {report['status']!r}")
    return attempted


def _summary_is_nonpass(summary: Mapping[str, Any]) -> bool:
    """A worker/coordinator invocation is not PASS with unresolved evidence."""

    return any(summary.get(key) for key in (
        "failed_authority_ordinals",
        "terminal_invalid_authority_ordinals",
        "in_progress_authority_ordinals",
    ))


def _inventory() -> list[dict[str, str | int]]:
    try:
        output = subprocess.run(
            GPU_QUERY,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,  # literal-ok: subprocess safety timeout
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"cannot query NVIDIA inventory: {exc}") from exc
    result: list[dict[str, str | int]] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise RuntimeError("unexpected nvidia-smi inventory format")
        result.append({
            "gpu_index": int(fields[0]),
            "gpu_uuid": fields[1],
            "gpu_name": fields[2],
            "driver_version": fields[3],
        })
    if len({item["gpu_uuid"] for item in result}) != len(result):
        raise RuntimeError("NVIDIA inventory contains duplicate UUIDs")
    return result


def _config_hash() -> str:
    return hashlib.sha256(PRODUCTION_CONTRACT.read_bytes()).hexdigest()


def _validate_partition(workers: list[dict[str, Any]]) -> dict[int, list[int]]:
    partitions = exact_shard_partition()
    seen: list[int] = []
    for worker in workers:
        assigned = worker["assigned_authority_ordinals"]
        if assigned != partitions[worker["shard_index"]]:
            raise RuntimeError("worker assignment differs from the authenticated authority partition")
        seen.extend(assigned)
    if len(seen) != REQUIRED_COUNT or len(set(seen)) != REQUIRED_COUNT or set(seen) != set(range(REQUIRED_COUNT)):
        raise RuntimeError("successor shard partition has overlap or omission")
    return partitions


def build_plan() -> dict[str, object]:
    """Authenticate both logical CUDA devices and return the full plan."""

    validate_successor_manifest(load_json(SUCCESSOR_MANIFEST))
    validate_successor_state(load_json(SUCCESSOR_STATE))
    production = validate_production_contracts()
    bindings = successor_bindings()
    profile = profile_definition(SUCCESSOR_PROFILE_ID)
    inventory = _inventory()
    by_uuid = {str(item["gpu_uuid"]): item for item in inventory}
    workers: list[dict[str, Any]] = []
    for worker_contract in production["workers"]:
        device = str(worker_contract["device"])
        gpu_uuid = str(worker_contract["gpu_uuid"])
        if DEVICE_RE.fullmatch(device) is None:
            raise RuntimeError("successor coordinator requires explicit cuda:N devices")
        gpu = by_uuid.get(gpu_uuid)
        if gpu is None:
            raise RuntimeError(f"required GPU UUID is missing: {gpu_uuid}")
        if gpu["gpu_name"] != worker_contract["gpu_name"]:
            raise RuntimeError(f"GPU name differs for {gpu_uuid}")
        if gpu_uuid not in profile["allowed_gpu_uuids"] or str(gpu["gpu_name"]) not in profile["allowed_gpu_names"]:
            raise RuntimeError(f"GPU {gpu_uuid} is not admitted by {SUCCESSOR_PROFILE_ID}")
        environment = authenticate_execution_profile(
            SUCCESSOR_PROFILE_ID,
            device=device,
            config_hash=_config_hash(),
            require_openjpeg=False,
        )
        if environment["gpu_uuid"] != gpu_uuid or environment["gpu_name"] != worker_contract["gpu_name"]:
            raise RuntimeError(f"Torch logical UUID/name mismatch on {device}")
        if environment["gpu_compute_capability"] != worker_contract["gpu_compute_capability"]:
            raise RuntimeError(f"compute capability mismatch on {device}")
        assigned = [ordinal for ordinal in range(REQUIRED_COUNT) if authority_shard(ordinal) == worker_contract["shard_index"]]
        workers.append({
            **dict(worker_contract),
            **gpu,
            "environment": environment,
            "assigned_authority_ordinals": assigned,
        })
    _validate_partition(workers)
    return {
        "status": "NON_SCIENTIFIC_PLAN_ONLY",
        "campaign_id": bindings["campaign_id"],
        "execution_profile_id": SUCCESSOR_PROFILE_ID,
        "successor_root": str(SUCCESSOR_ROOT.relative_to(REPO)),
        "old_root": "results/baseline/g8/work_units",
        "required_identity_count": REQUIRED_COUNT,
        "accepted_count": 0,
        "workers": workers,
        "duplicate_work_units": 0,
        "missing_work_units": 0,
        "partition_union_count": REQUIRED_COUNT,
        "production_contract_sha256": production["production_contract_sha256"],
        "lock_file_sha256": bindings["lock_file_sha256"],
        "driver_versions": sorted({str(worker["environment"]["driver_version"]) for worker in workers}),
        "test_access": 0,
        "launch_gate": "MERGED_MAIN_FINAL_QUALIFICATION_WRITER_AUTH_AND_EXPLICIT_OWNER_AUTHORIZATION",
    }


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def _validate_authorization(path: Path, bindings: dict[str, Any]) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"launch authorization cannot be read: {exc}") from exc
    if not isinstance(payload, dict) or raw != json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n":
        raise RuntimeError("launch authorization is not canonical rendered JSON")
    required = {"schema_version", "artifact_role", "campaign_id", "execution_profile_id", "production_contract_sha256", "authorized_by", "authorization_sha256"}
    if set(payload) != required or payload["schema_version"] != 1 or payload["artifact_role"] != LAUNCH_AUTHORIZATION_ROLE:
        raise RuntimeError("launch authorization schema differs")
    if payload["campaign_id"] != bindings["campaign_id"] or payload["execution_profile_id"] != SUCCESSOR_PROFILE_ID or payload["production_contract_sha256"] != bindings["production_contract_sha256"]:
        raise RuntimeError("launch authorization campaign/profile/contract differs")
    if not isinstance(payload["authorized_by"], str) or not payload["authorized_by"].strip():
        raise RuntimeError("launch authorization owner is missing")
    body = dict(payload)
    body.pop("authorization_sha256")
    if payload["authorization_sha256"] != hashlib.sha256(canonical_json(body)).hexdigest():
        raise RuntimeError("launch authorization digest differs")
    return payload


def _existing_runtime_summary(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {"accepted_count": 0, "in_progress_count": 0, "failed_count": 0, "runtime_exists": False}
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("successor runtime root is not a real directory")
    summary = audit_campaign(root)
    return {
        "accepted_count": summary.get("accepted_count", 0),
        "in_progress_count": len(summary.get("in_progress_authority_ordinals", [])),
        "failed_count": len(summary.get("failed_authority_ordinals", [])),
        "runtime_exists": True,
    }


def _validate_launch_gate(root: Path, *, authorization: Path | None, dry_run: bool) -> dict[str, Any]:
    if not root.is_absolute():
        raise RuntimeError("production execution requires an absolute --root")
    bindings = successor_bindings()
    if _git("rev-parse", "HEAD") != _git("rev-parse", "origin/main"):
        raise RuntimeError("production execution requires exact parity with origin/main")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("production execution requires a clean worktree")
    summary = _existing_runtime_summary(root)
    if dry_run:
        return {"status": "DRY_RUN_LAUNCH_PATH_READY", "runtime": summary, "authorization": None}
    if authorization is None:
        raise RuntimeError("real successor execution requires an explicit owner launch authorization file")
    record = _validate_authorization(authorization, bindings)
    return {"status": "AUTHORIZED", "runtime": summary, "authorization": record["authorized_by"]}


def _worker(args: argparse.Namespace) -> int:
    validate_successor_manifest(load_json(SUCCESSOR_MANIFEST))
    validate_successor_state(load_json(SUCCESSOR_STATE))
    production = validate_production_contracts()
    worker = next((item for item in production["workers"] if item["device"] == args.device), None)
    if worker is None or worker["gpu_uuid"] != args.gpu_uuid or worker["shard_index"] != args.shard_index:
        raise RuntimeError("worker command is not an exact registered device/shard mapping")
    if args.shard_count != 2 or DEVICE_RE.fullmatch(args.device) is None:
        raise RuntimeError("worker requires the frozen two-shard explicit CUDA mapping")
    profile = authenticate_worker_profile(
        device=args.device,
        expected_gpu_uuid=args.gpu_uuid,
        config_hash=_config_hash(),
    )
    root = ensure_runtime_root(args.root)
    reconcile_campaign(root)
    partition = exact_shard_partition()[args.shard_index]
    attempted = _run_worker_batch(
        root=root,
        partition=partition,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        device=args.device,
        gpu_uuid=args.gpu_uuid,
        profile=profile,
        batch_size=args.batch_size,
        max_units=args.max_units,
    )
    summary = reconcile_campaign(root)
    status = "FAIL" if _summary_is_nonpass(summary) else "PASS"
    print(json.dumps({"status": status, "worker": args.device, "shard_index": args.shard_index, "units_attempted": attempted, "reconciliation": summary}, sort_keys=True))
    return 1 if status != "PASS" else 0


def _child_command(worker: Mapping[str, Any], *, root: Path, batch_size: int, max_units: int | None) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--root",
        str(root),
        "--device",
        str(worker["device"]),
        "--gpu-uuid",
        str(worker["gpu_uuid"]),
        "--shard-index",
        str(worker["shard_index"]),
        "--shard-count",
        "2",
        "--batch-size",
        str(batch_size),
    ]
    if max_units is not None:
        command.extend(["--max-units", str(max_units)])
    return command


def launch_children(
    plan: Mapping[str, Any],
    *,
    root: Path,
    batch_size: int,
    max_units: int | None,
) -> list[dict[str, Any]]:
    """Launch one independent child per registered Pascal GPU and collect exits."""

    _validate_max_units(max_units)
    children: list[tuple[Mapping[str, Any], subprocess.Popen[str]]] = []
    failures: list[dict[str, Any]] = []
    for worker in plan["workers"]:
        command = _child_command(worker, root=root, batch_size=batch_size, max_units=max_units)
        try:
            child = subprocess.Popen(command, cwd=REPO, text=True)
        except OSError as exc:
            failures.append({"device": worker["device"], "return_code": 127, "error": str(exc)})
            continue
        children.append((worker, child))
    for worker, child in children:
        return_code = child.wait()
        if return_code != 0:
            failures.append({"device": worker["device"], "return_code": return_code})
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--authorization-file", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-units", type=_positive_max_units)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--device")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=2)
    args = parser.parse_args()

    if args.worker:
        if args.root is None or args.device is None or args.gpu_uuid is None or args.shard_index is None:
            parser.error("worker requires --root, --device, --gpu-uuid and --shard-index")
        return _worker(args)
    if args.plan_only and (args.execute or args.dry_run):
        parser.error("--plan-only cannot be combined with execution flags")
    if not args.plan_only and not args.execute:
        parser.error("choose --plan-only or --execute")
    plan = build_plan()
    if args.plan_only:
        print(json.dumps(plan, sort_keys=True))
        return 0
    if args.root is None:
        parser.error("--execute requires an explicit absolute --root")
    gate = _validate_launch_gate(args.root, authorization=args.authorization_file, dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps({"plan": plan, "launch_gate": gate, "children": [
            {"device": worker["device"], "gpu_uuid": worker["gpu_uuid"], "shard_index": worker["shard_index"], "shard_count": worker["shard_count"]}
            for worker in plan["workers"]
        ]}, sort_keys=True))
        return 0

    root = args.root
    ensure_runtime_root(root)
    reconcile_campaign(root)
    failures = launch_children(
        plan,
        root=root,
        batch_size=args.batch_size,
        max_units=args.max_units,
    )
    reconciliation = reconcile_campaign(root)
    status = "FAIL" if failures or _summary_is_nonpass(reconciliation) else "PASS"
    result = {"status": status, "workers_failed": failures, "reconciliation": reconciliation}
    print(json.dumps(result, sort_keys=True))
    return 1 if status != "PASS" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ProductionContractError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
