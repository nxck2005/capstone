#!/usr/bin/env python3
"""Generate, check, and register the additive G8_C epoch-2 source manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from baseline.g8_bler_characterization_v2 import (  # noqa: E402
    SOURCE_MANIFEST_PATH,
    build_source_manifest,
    register_source_manifest,
    validate_source_manifest,
)
from baseline.classical.outage import write_json_atomically  # noqa: E402
from baseline.g8_campaign import rendered_json, sha256_bytes  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--require-registered", action="store_true")
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args(argv)
    if args.check and args.register:
        parser.error("--check and --register are mutually exclusive")
    try:
        if args.check:
            payload = json.loads(SOURCE_MANIFEST_PATH.read_bytes())
            validate_source_manifest(payload, require_registered=args.require_registered)
            expected = rendered_json(payload)
            actual = SOURCE_MANIFEST_PATH.read_bytes()
            if actual != expected:
                raise SystemExit("G8_C characterization source manifest is stale")
            print(
                "ok: G8_C epoch-2 source manifest matches registered source bytes "
                f"manifest_id={payload['manifest_id']} sha256={sha256_bytes(actual)} bytes={len(actual)}"
            )
            return 0
        payload = build_source_manifest()
        # Registration performs the final validation against the raw
        # pre-science state as well.  This order permits one guarded
        # pre-science rebind when a source file was corrected after a
        # provisional registration; campaign state rejects the same path
        # once epoch-2 science has begun.
        if not args.register:
            validate_source_manifest(payload)
        expected = rendered_json(payload)
        digest = write_json_atomically(SOURCE_MANIFEST_PATH, payload)
        print(
            "wrote G8_C epoch-2 source manifest "
            f"manifest_id={payload['manifest_id']} sha256={digest} bytes={len(expected)}"
        )
        if args.register:
            state_digest = register_source_manifest()
            print(f"registered G8_C epoch-2 source manifest state_sha256={state_digest}")
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(f"G8_C source-manifest HOLD: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
