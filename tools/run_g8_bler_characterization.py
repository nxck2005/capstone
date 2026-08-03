#!/usr/bin/env python3
"""Coordinate resumable G8_C full-strength BLER worker batches.

The coordinator owns campaign-state reconciliation.  A worker process owns
only its assigned per-unit transactions and creates one authenticated runner
context for its whole batch.  This command deliberately performs at most one
durable worker batch; the caller commits and pushes the resulting raw evidence
before launching the next batch.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from baseline import g8_bler_characterization as characterization  # noqa: E402
from baseline import g8_bler_resume as resume  # noqa: E402
from baseline import g8_bler_runner as runner  # noqa: E402
from baseline.g8_campaign import CAMPAIGN_STATE, load_campaign_state  # noqa: E402


EXIT_COMPLETE_BATCH = 0
EXIT_INCOMPLETE = 2
EXIT_HOLD = 4


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), required=True)
    parser.add_argument("--shard-count", required=True)
    parser.add_argument("--shard-index")
    parser.add_argument("--batch-size", required=True)
    parser.add_argument("--max-units-per-worker-batch", required=True)
    parser.add_argument("--repair-recoverable", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--reconcile-only", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    return parser


def _positive_token(value: str, label: str) -> int:
    if not isinstance(value, str) or not value or not value.isdecimal():
        raise ValueError(f"{label} must be auto or a positive decimal integer")
    number = int(value)
    if number <= 0:
        raise ValueError(f"{label} must be positive")
    return number


def _validate_args(args: argparse.Namespace) -> Path:
    """Validate every CLI argument before state/root inspection."""

    modes = sum(bool(value) for value in (args.plan_only, args.reconcile_only, args.merge_only))
    if modes > 1:
        raise ValueError("--plan-only, --reconcile-only and --merge-only are mutually exclusive")
    if args.shard_count != "auto":
        _positive_token(args.shard_count, "--shard-count")
    if args.shard_index is not None:
        if not args.shard_index.isdecimal():
            raise ValueError("--shard-index must be a non-negative decimal integer")
    if args.batch_size != "auto":
        _positive_token(args.batch_size, "--batch-size")
    _positive_token(args.max_units_per_worker_batch, "--max-units-per-worker-batch")
    return characterization.validate_production_root(Path(args.root))


def _resolve_topology(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    if args.device == "auto":
        device = "cuda" if cuda_available and device_count > 0 else "cpu"
    else:
        device = args.device
    if device == "cuda" and (not cuda_available or device_count == 0):
        raise RuntimeError("CUDA was explicitly requested but no supported CUDA device is available")
    if args.shard_count == "auto":
        shard_count = device_count if device == "cuda" else 1
    else:
        shard_count = int(args.shard_count)
    if shard_count <= 0:
        raise ValueError("resolved shard count must be positive")
    shard_index = 0 if args.shard_index is None else int(args.shard_index)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard index is outside the resolved shard count")
    # The runner has already proved batch-partition determinism.  This is an
    # execution choice only; it never enters a request, seed, or table ID.
    batch_size = 64 if args.batch_size == "auto" else int(args.batch_size)  # literal-ok: conservative RTX-4060 auto batch
    if batch_size <= 0:
        raise ValueError("resolved batch size must be positive")
    gpu_names = []
    if cuda_available:
        gpu_names = [str(torch.cuda.get_device_name(index)) for index in range(device_count)]
    return {
        "requested_device": args.device,
        "resolved_device": device,
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "gpu_names": gpu_names,
        "shard_count": shard_count,
        "shard_index": shard_index,
        "batch_size": batch_size,
        "batch_size_reason": "auto conservative default" if args.batch_size == "auto" else "explicit",
        "workers": device_count if device == "cuda" else 1,
    }


def _require_c1_registration() -> None:
    source = characterization.SOURCE_MANIFEST_PATH
    payload = json.loads(source.read_bytes())
    characterization.validate_source_manifest(payload, require_registered=True)
    state = load_campaign_state(CAMPAIGN_STATE)
    identity = state["identity"]
    if identity["phase"] != "G8_C" or identity["stage"] not in {"characterization_open", "characterization_complete"}:
        raise RuntimeError("G8_C source-manifest registration is not live")


def _worker_main(
    root_string: str,
    device: str,
    shard_count: int,
    shard_index: int,
    batch_size: int,
    work_unit_ids: list[str],
    result_queue: Any,
) -> None:
    """Run one process-scoped authenticated worker batch."""

    try:
        context = runner.AuthenticatedRunnerContext()
        root = Path(root_string)
        outcomes: list[dict[str, Any]] = []
        for work_unit_id in work_unit_ids:
            # A worker never trusts its parent plan after another transaction;
            # the frozen B3 plan is rebuilt before every unit.
            plan = resume.build_resume_plan(
                context.resume_context,
                root=root,
                shard_count=shard_count,
                shard_index=shard_index,
                scan_mode=resume.SCAN_MODE_PRODUCTION_MERGE,
            )
            if work_unit_id not in plan["remaining_work_unit_ids"]:
                raise RuntimeError(f"planned work unit is no longer remaining: {work_unit_id}")
            outcome = runner.run_one_unit(
                context,
                execution_class=runner.EXECUTION_CLASS_FULL_STRENGTH,
                root=root,
                work_unit_id=work_unit_id,
                shard_count=shard_count,
                shard_index=shard_index,
                batch_size=batch_size,
                device=device,
            )
            result = outcome["result"]
            outcomes.append(
                {
                    "work_unit_id": work_unit_id,
                    "attempt": outcome["attempt"],
                    "status": result["status"],
                    "trials_completed": result["measurement"]["trials_completed"],
                    "request_sha256": outcome["request_sha256"],
                    "result_sha256": outcome["result_sha256"],
                    "state_sha256": outcome["state_sha256"],
                }
            )
            if result["status"] != "complete":
                break
        result_queue.put({"ok": True, "outcomes": outcomes})
    except BaseException as exc:
        result_queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _run_worker_batch(root: Path, topology: dict[str, Any], work_unit_ids: list[str]) -> dict[str, Any]:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_worker_main,
        args=(
            str(root),
            topology["resolved_device"],
            topology["shard_count"],
            topology["shard_index"],
            topology["batch_size"],
            work_unit_ids,
            queue,
        ),
    )
    process.start()
    process.join()
    try:
        result = queue.get(timeout=5)
    except Exception as exc:
        raise RuntimeError(f"worker exited without a durable summary (exitcode={process.exitcode})") from exc
    if process.exitcode not in (0, None) and result.get("ok"):
        raise RuntimeError(f"worker exited unexpectedly with code {process.exitcode}")
    return result


def _summary(plan: dict[str, Any], topology: dict[str, Any], worker: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "plan_digest": plan["plan_digest"],
        "assigned_count": len(plan["assigned_work_unit_ids"]),
        "completed_count": len(plan["completed_work_unit_ids"]),
        "recoverable_count": len(plan["recoverable_work_unit_ids"]),
        "remaining_count": len(plan["remaining_work_unit_ids"]),
        "terminal_nonmergeable_count": len(plan["terminal_nonmergeable_work_unit_ids"]),
        "topology": topology,
        "worker": worker,
        "test_split_access": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        root = _validate_args(args)
        topology = _resolve_topology(args)
        _require_c1_registration()
        context = runner.AuthenticatedRunnerContext()
        if args.merge_only:
            report = resume.build_merge_report(
                context.resume_context,
                root=root,
                scan_mode=resume.SCAN_MODE_PRODUCTION_MERGE,
            )
            print(json.dumps(report, sort_keys=True))
            return EXIT_COMPLETE_BATCH
        if args.reconcile_only:
            result = characterization.reconcile_characterization_campaign(
                context.resume_context,
                root=root,
                repair_recoverable=args.repair_recoverable,
            )
            print(json.dumps(result, sort_keys=True))
            return EXIT_COMPLETE_BATCH

        if args.repair_recoverable:
            characterization.reconcile_characterization_campaign(
                context.resume_context,
                root=root,
                repair_recoverable=True,
            )
        plan = resume.build_resume_plan(
            context.resume_context,
            root=root,
            shard_count=topology["shard_count"],
            shard_index=topology["shard_index"],
            scan_mode=resume.SCAN_MODE_PRODUCTION_MERGE,
        )
        if args.plan_only:
            print(json.dumps(_summary(plan, topology), sort_keys=True))
            return EXIT_COMPLETE_BATCH
        if plan["recoverable_work_unit_ids"]:
            raise RuntimeError(
                "recoverable work exists; rerun with --repair-recoverable before execution"
            )
        maximum = int(args.max_units_per_worker_batch)
        selected = plan["remaining_work_unit_ids"][:maximum]
        if not selected:
            reconciled = characterization.reconcile_characterization_campaign(
                context.resume_context,
                root=root,
                repair_recoverable=False,
            )
            print(json.dumps({"complete": reconciled["counts"]["remaining"] == 0, "reconciled": reconciled, "topology": topology}, sort_keys=True))
            return EXIT_COMPLETE_BATCH

        worker_result = _run_worker_batch(root, topology, selected)
        reconciled = characterization.reconcile_characterization_campaign(
            context.resume_context,
            root=root,
            repair_recoverable=False,
        )
        output = {
            "selected_work_unit_ids": selected,
            "worker": worker_result,
            "reconciled": reconciled,
            "topology": topology,
            "test_split_access": 0,
        }
        print(json.dumps(output, sort_keys=True))
        if not worker_result.get("ok"):
            return EXIT_INCOMPLETE
        if any(item["status"] != "complete" for item in worker_result.get("outcomes", [])):
            return EXIT_INCOMPLETE
        return EXIT_COMPLETE_BATCH
    except Exception as exc:
        print(f"G8_C characterization HOLD: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_HOLD


if __name__ == "__main__":
    raise SystemExit(main())
