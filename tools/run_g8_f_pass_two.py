#!/usr/bin/env python3
"""Freeze authorization, execute, or verify exact-once BR-4 pass two."""

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
from baseline.g8_f_pass_two import (
    AUTHORIZATION_PATH,
    COMPARISON_PATH,
    STATE_PATH,
    build_authorization,
    build_comparison,
    run_pass_two,
    verify_authorization,
    verify_comparison,
    verify_state,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    authorize = commands.add_parser("authorize")
    authorize.add_argument("--source-commit", required=True)
    authorize.add_argument("--github-run-id", type=int, required=True)
    authorize.add_argument("--github-run-url", required=True)
    commands.add_parser("verify-authorization")
    commands.add_parser("execute")
    commands.add_parser("verify")
    args = parser.parse_args()
    if args.command == "authorize":
        value = build_authorization(source_commit=args.source_commit, github_actions={"sha": args.source_commit, "run_id": args.github_run_id, "url": args.github_run_url, "conclusion": "success"})
        atomic_bytes(AUTHORIZATION_PATH, rendered_json(value), refuse_existing=True)
        verify_authorization()
        print("pass-two authorization frozen:", value["authorization_id"], "pre_count=0")
        return 0
    if args.command == "verify-authorization":
        value = verify_authorization(); print("pass-two authorization PASS:", value["authorization_id"], "pass_two=0 pass_three=0 test=0"); return 0
    if args.command == "execute":
        value = run_pass_two()
        comparison = build_comparison(); atomic_bytes(COMPARISON_PATH, rendered_json(comparison), refuse_existing=True)
        verify_state(); verify_comparison()
        print("PASS TWO COMPLETE:", value["completion_id"], f"calls={value['call_count']}", f"candidates={value['totals']['candidates_evaluated']}", f"cells={value['totals']['snr_cells_with_selection']}", f"ties={value['totals']['tie_breaks_applied']}", f"changed={comparison['changed_cells']}")
        return 0
    state = verify_state(); comparison = verify_comparison()
    print(json.dumps({"status": "PASS", "completion_id": state["completion_id"], "calls": state["call_count"], "candidates": state["totals"]["candidates_evaluated"], "cells": state["totals"]["snr_cells_with_selection"], "ties": state["totals"]["tie_breaks_applied"], "changed": comparison["changed_cells"], "pass_two": 1, "pass_three": 0, "test": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
