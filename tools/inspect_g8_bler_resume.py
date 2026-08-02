#!/usr/bin/env python3
"""Inspect the authenticated G8_B runtime tree without running science."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from baseline import g8_bler_resume as resume  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--scan-mode",
        choices=resume.SCAN_MODES,
        default=resume.SCAN_MODE_PRODUCTION_MERGE,
    )
    parser.add_argument(
        "--repair-mode",
        choices=resume.REPAIR_MODES,
        default=resume.REPAIR_MODE_READ_ONLY,
    )
    parser.add_argument("--campaign-state-path", type=Path, default=None)
    parser.add_argument("--state-contract-path", type=Path, default=None)
    parser.add_argument("--resume-contract-path", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        context = resume.AuthenticatedResumeContext(
            campaign_state_path=args.campaign_state_path,
            state_contract_path=args.state_contract_path,
            resume_contract_path=args.resume_contract_path,
            require_resume_contract=True,
        )
        report = resume.inspect_runtime_root(
            context,
            root=args.root,
            scan_mode=args.scan_mode,
            repair_mode=args.repair_mode,
        )
    except resume.G8BlerResumeError as exc:
        raise SystemExit(f"G8 B3 resume inspection HOLD: {exc}") from exc
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
