#!/usr/bin/env python3
"""Run or close out validation-only G8_F/F3 cached-artifact rescoring."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baseline.g8_f_f3 import (
    AGGREGATE_PATH,
    DEFAULT_BATCH_SIZE,
    RUNTIME_ROOT,
    atomic_bytes,
    build_aggregate,
    rendered_json,
    run_f3,
    verify_aggregate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--runtime-root", type=Path, default=RUNTIME_ROOT)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    close = commands.add_parser("closeout")
    close.add_argument("--runtime-root", type=Path, default=RUNTIME_ROOT)
    verify = commands.add_parser("verify")
    verify.add_argument("--live-runtime", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run_f3(runtime_root=args.runtime_root.resolve(), checkpoint_path=args.checkpoint.resolve(), device=args.device, batch_size=args.batch_size)
        print("F3 scoring units COMPLETE; run closeout before pass two")
        return 0
    if args.command == "closeout":
        value = build_aggregate(args.runtime_root.resolve())
        atomic_bytes(AGGREGATE_PATH, rendered_json(value), refuse_existing=True)
        verify_aggregate(AGGREGATE_PATH, live_runtime=args.runtime_root.resolve())
        print("F3 exact closeout PASS:", value["aggregate_id"], f"rows={value['row_count']}", f"delivered={value['outcomes']['delivered']}", f"outage={value['outcomes']['codec_infeasibility']}")
        return 0
    value = verify_aggregate(AGGREGATE_PATH, live_runtime=args.live_runtime)
    print("F3 aggregate verification PASS:", value["aggregate_id"], f"rows={value['row_count']}", "pass_two=0 test=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
