#!/usr/bin/env python3
"""Generate or verify compact G8_F/F1 completion and corpus evidence.

Generation is live-worker-only and hashes every request/result/object. Offline
verification consumes only tracked compact evidence. Neither mode invokes a
codec, classifier, optimizer, selection pass, or guarded test loader.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from baseline.g8_f_closeout import (  # noqa: E402
    COMPLETION_PATH,
    MANIFEST_PATH,
    RUNTIME_PATH,
    G8FF1CloseoutHold,
    build_closeout,
    verify_closeout,
    verify_monitor_closeout,
)
from baseline.g8_f_materializer import G8FMaterializationHold  # noqa: E402
from baseline.g8_f_f0 import rendered_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="authenticate live runtime and freeze compact evidence")
    generate.add_argument("--runtime-root", type=Path, default=RUNTIME_PATH)
    generate.add_argument("--ops-root", type=Path, required=True)
    generate.add_argument("--monitor-source", type=Path, required=True)
    generate.add_argument("--monitor-log", type=Path, required=True)
    generate.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    generate.add_argument("--completion", type=Path, default=COMPLETION_PATH)
    verify = subparsers.add_parser("verify", help="verify tracked evidence, optionally against live corpus bytes")
    verify.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    verify.add_argument("--completion", type=Path, default=COMPLETION_PATH)
    verify.add_argument("--runtime-root", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.command == "generate":
            manifest, completion = build_closeout(
                args.runtime_root,
                args.ops_root,
                args.monitor_source,
                args.monitor_log,
                manifest_path=args.manifest,
            )
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.completion.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_bytes(manifest)
            args.completion.write_bytes(rendered_json(completion))
        value = verify_closeout(
            args.completion,
            args.manifest,
            runtime_root=args.runtime_root if args.command == "verify" else args.runtime_root,
        )
        monitor = verify_monitor_closeout(completion=value) if args.command == "verify" else None
        print(json.dumps({
            "status": "PASS",
            "completion_id": value["completion_id"],
            "corpus_id": value["corpus_id"],
            "authenticated_prefix": value["coverage"]["authenticated_prefix"],
            "outcomes": value["outcomes"],
            "live_full_object_verification": args.runtime_root is not None,
            "discord_monitor_closeout": None if monitor is None else monitor["monitor_closeout_id"],
            "verdict": value["terminal_statement"],
        }, sort_keys=True))
        return 0
    except (G8FF1CloseoutHold, G8FMaterializationHold, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "HOLD", "reason": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
