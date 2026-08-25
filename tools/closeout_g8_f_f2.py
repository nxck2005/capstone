#!/usr/bin/env python3
"""Generate or verify compact G8_F/F2 completion and classifier-freeze evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.g8_f_f2 import atomic_bytes, sha256_bytes
from training.g8_f_f2_closeout import (
    COMPLETION_PATH,
    FREEZE_PATH,
    MONITOR_PATH,
    audit_runtime,
    build_freeze,
    rendered_json,
    verify_checkpoint,
    verify_compact,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--runtime-root", type=Path, default=ROOT / "results/baseline/g8_f/f2_runtime")
    generate.add_argument("--f1-runtime", type=Path, default=ROOT / "results/baseline/g8_f/runtime")
    generate.add_argument("--ops-root", type=Path, default=Path("/home/nick/g8-f-f2-ops"))
    generate.add_argument("--skip-object-authentication", action="store_true")
    commands.add_parser("verify")
    checkpoint = commands.add_parser("verify-checkpoint")
    checkpoint.add_argument("checkpoint", type=Path)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.command == "generate":
        completion = audit_runtime(
            runtime_root=args.runtime_root.resolve(),
            f1_runtime=args.f1_runtime.resolve(),
            ops_root=args.ops_root.resolve(),
            authenticate_objects=not args.skip_object_authentication,
        )
        completion_raw = rendered_json(completion)
        freeze = build_freeze(completion, sha256_bytes(completion_raw))
        atomic_bytes(COMPLETION_PATH, completion_raw)
        atomic_bytes(FREEZE_PATH, rendered_json(freeze))
        verify_compact(COMPLETION_PATH, FREEZE_PATH, monitor_path=None)
        print(
            "F2 closeout generation PASS:",
            completion["completion_id"],
            freeze["freeze_id"],
            f"best_epoch={completion['selection']['best_epoch']}",
            f"best_top1={completion['selection']['best_validation_top1']}",
        )
        return 0
    if args.command == "verify":
        completion, freeze = verify_compact(COMPLETION_PATH, FREEZE_PATH, MONITOR_PATH)
        print(
            "F2 compact closeout PASS:",
            completion["completion_id"],
            freeze["freeze_id"],
            "F3_rescoring=0 pass_two=0 test=0",
        )
        return 0
    verify_checkpoint(args.checkpoint.resolve())
    print("F2 artifact-classifier checkpoint loadability PASS:", args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
