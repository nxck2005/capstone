#!/usr/bin/env python3
"""Preflight or separately owner-launch the exact frozen G8_F/F1 assignment.

F0-v3 alone is intentionally insufficient for ``--start``. This command requires
an additive owner-issued F1 launch authorization whose identity binds the active
Pascal F0-v3 file and repaired source commit. F0-v3 runs only ``--preflight`` and
never calls this command's production loop.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline.g8_f_f0 import AUTHORIZATION_PATH, rendered_json, verify_f0_authorization  # noqa: E402
from baseline.g8_f_materializer import (  # noqa: E402
    F1Materializer,
    G8FMaterializationHold,
    canonical_json,
    load_frozen_assignments,
    validate_exact_result_prefix,
)
from baseline.j2k import J2KCodec  # noqa: E402
from data.registry import load_dataset  # noqa: E402


class F1LaunchAuthorizationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise F1LaunchAuthorizationError(message)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def verify_separate_f1_launch(path: Path, f0_path: Path, f0: dict[str, Any]) -> dict[str, Any]:
    """Authenticate a later owner action; F0 never creates this artifact."""

    raw = path.read_bytes()
    value = json.loads(raw)
    _require(raw == rendered_json(value), "F1 launch authorization is not canonical rendered JSON")
    required = {
        "schema_version", "artifact_role", "status", "scope", "issued_at",
        "f0_authorization_id", "f0_file_sha256", "intended_f1_source_commit",
        "owner_statement", "launch_id",
    }
    _require(set(value) == required, "F1 launch authorization schema differs")
    _require(value["schema_version"] == 1, "F1 launch authorization schema version differs")
    _require(value["artifact_role"] == "g8_f_f1_owner_launch_authorization", "F1 launch authorization role differs")
    _require(value["status"] == "OWNER_AUTHORIZED_F1_LAUNCH" and value["scope"] == "G8_F_F1_ONLY", "owner did not authorize F1 only")
    _require(value["owner_statement"] == "DELIBERATE_SEPARATE_OWNER_ACTION_AFTER_F0_FREEZE", "F1 owner statement differs")
    _require(value["f0_authorization_id"] == f0["authorization_id"], "F1 launch references another F0 authorization")
    _require(value["f0_file_sha256"] == _sha(f0_path.read_bytes()), "F1 launch references other F0 bytes")
    _require(value["intended_f1_source_commit"] == f0["source"]["intended_f1_source_commit"], "F1 launch source commit differs")
    body = dict(value)
    launch_id = body.pop("launch_id")
    _require(launch_id == "g8ff1launch-" + _sha(canonical_json(body)), "F1 launch authorization ID differs")
    return value


def run_f1(f0_path: Path, launch_path: Path, runtime_root: Path) -> None:
    f0 = verify_f0_authorization(f0_path, live_runtime=True, require_zero_prefix=False)
    _require(runtime_root.resolve() == (REPO / f0["execution"]["runtime_root"]).resolve(), "F1 runtime destination differs from F0")
    launch = verify_separate_f1_launch(launch_path, f0_path, f0)
    del launch
    assignments = load_frozen_assignments()

    runtime_root.mkdir(parents=True, exist_ok=True)
    lock_path = runtime_root / "f1.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise F1LaunchAuthorizationError("another F1 writer holds the runtime lock") from None
        completed = validate_exact_result_prefix(runtime_root, assignments)
        dataset = load_dataset("imagenette160", "train")
        source_by_id = {
            dataset.source_sample(index).stable_sample_id: dataset.source_sample(index)
            for index in range(len(dataset))
        }
        _require(set(source_by_id) == {assignment.stable_sample_id for assignment in assignments}, "loaded train source membership differs from AM-88")
        backend = J2KCodec(runtime_root / "backend_j2k_cache")
        materializer = F1Materializer(runtime_root, backend, scientific=True)
        for assignment in assignments[completed:]:
            materializer.materialize(assignment, source_by_id[assignment.stable_sample_id], split="train")
            completed += 1
            print(json.dumps({"completed": completed, "total": len(assignments), "assignment_id": assignment.assignment_id}), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="authenticate F0 only; never start F1")
    mode.add_argument("--start", action="store_true", help="require a separate owner F1 authorization and run/resume exact AM-88")
    parser.add_argument("--f0-authorization", type=Path, default=AUTHORIZATION_PATH)
    parser.add_argument("--f1-launch-authorization", type=Path)
    parser.add_argument("--runtime-root", type=Path, default=REPO / "results/baseline/g8_f/runtime")
    args = parser.parse_args(argv)
    try:
        if args.preflight:
            _require(args.f1_launch_authorization is None, "F0 preflight does not accept or launch F1")
            value = verify_f0_authorization(args.f0_authorization, live_runtime=True)
            print(json.dumps({
                "status": "PASS",
                "authorization_id": value["authorization_id"],
                "f1_started": False,
                "verdict": "F0-V3 PASCAL GREEN - F1 REQUIRES SEPARATE OWNER/OPERATOR LAUNCH",
            }, sort_keys=True))
            return 0
        _require(args.f1_launch_authorization is not None, "--start requires a separate owner-issued F1 launch authorization")
        run_f1(args.f0_authorization, args.f1_launch_authorization, args.runtime_root)
        return 0
    except (F1LaunchAuthorizationError, G8FMaterializationHold, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "HOLD", "reason": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
