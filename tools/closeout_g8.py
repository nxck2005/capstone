#!/usr/bin/env python3
"""Build or verify deterministic validation-only G8 adjudication evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baseline.g8_f_f3 import atomic_bytes, rendered_json
from baseline.g8_g_closeout import (
    CLOSEOUT_PATH,
    INPUT_PATH,
    SOURCE_MANIFEST_PATH,
    TERMINAL_BINDING_CORRECTION_PATH,
    build_adjudication_input,
    build_closeout,
    build_source_manifest,
    verify_adjudication_input,
    verify_closeout,
    verify_source_manifest,
    verify_terminal_binding_correction,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inputs = commands.add_parser("build-input")
    inputs.add_argument("--runtime-root", type=Path, required=True)
    commands.add_parser("verify-input")
    commands.add_parser("freeze")
    commands.add_parser("verify")
    args = parser.parse_args()
    if args.command == "build-input":
        value = build_adjudication_input(runtime_root=args.runtime_root)
        atomic_bytes(INPUT_PATH, rendered_json(value), refuse_existing=True)
        verify_adjudication_input()
        print("G8 adjudication input frozen:", value["input_id"], "qualities=120 structural=288 test=0")
        return 0
    if args.command == "verify-input":
        value = verify_adjudication_input(); print("G8 adjudication input PASS:", value["input_id"]); return 0
    if args.command == "freeze":
        closeout = build_closeout()
        atomic_bytes(CLOSEOUT_PATH, rendered_json(closeout), refuse_existing=True)
        verify_closeout()
        manifest = build_source_manifest()
        atomic_bytes(SOURCE_MANIFEST_PATH, rendered_json(manifest), refuse_existing=True)
        verify_source_manifest()
        print("G8 CLOSED:", closeout["closeout_id"], json.dumps(closeout["operating_points"], sort_keys=True))
        return 0
    closeout = verify_closeout(); manifest = verify_source_manifest(); correction = verify_terminal_binding_correction()
    print(json.dumps({"status": "PASS", "closeout_id": closeout["closeout_id"], "source_manifest_id": manifest["manifest_id"], "terminal_binding_correction_id": correction["correction_id"], "terminal_binding_correction_path": str(TERMINAL_BINDING_CORRECTION_PATH.relative_to(ROOT)), "operating_points": closeout["operating_points"], "pass_two": 1, "pass_three": 0, "learned_training": 0, "test": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
