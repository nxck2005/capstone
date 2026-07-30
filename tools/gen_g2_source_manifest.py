#!/usr/bin/env python3
"""Generate the G-2 execution-source manifest from the measurement commit.

`verify_g2_adjudication.py` already bound the G-2 evidence *files*, the measurement
commit and the ancestry measurement -> evidence. It did not bind the bytes of the
LDPC implementation that produced the measurement, so an edit to
`src/baseline/ldpc/` left every recorded number verifying against a different
implementation than the one that produced it.

This writes down, for each source that participated in the measurement, the path,
the Git blob object id at the measurement commit, the SHA-256 of those exact bytes,
and a role. The roles are the load-bearing part: the current checkout legitimately
differs from the measurement commit for some of these files, because
`82f6c569f792bf17ff28acd80ed1d516adfc06fa` added CLI import bootstrapping to the
measurement runners after the campaign ran. Only the `runtime` role is asserted to
be byte-identical today; everything else is bound as a historical record.

The expected path set and the role policy live in `verify_g2_adjudication.py`, not
here, so the manifest cannot define its own completeness.

Usage:
    python tools/gen_g2_source_manifest.py            # write the manifest
    python tools/gen_g2_source_manifest.py --check     # regenerate in memory, compare bytes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from verify_g2_adjudication import (  # noqa: E402  (path bootstrap must precede)
    EXPECTED_SOURCES,
    ROLE_POLICY,
    SOURCE_MANIFEST,
    blob_bytes,
    blob_id,
    git,
    manifest_path,
    sha256_bytes,
)


def build(measurement: str) -> dict:
    sources = []
    for path, role in sorted(EXPECTED_SOURCES.items()):
        content = blob_bytes(measurement, path)
        sources.append({
            "path": path,
            "role": role,
            "measurement_blob": blob_id(measurement, path),
            "measurement_sha256": sha256_bytes(content),
            "measurement_bytes": len(content),
        })
    return {
        "schema_version": 1,
        "kind": "g2_execution_source_manifest",
        "gate": "G-2",
        "generated_by": "tools/gen_g2_source_manifest.py",
        "measurement_commit": measurement,
        "git_object_format": git("rev-parse", "--show-object-format"),
        # The evidence commit is resolved through Git path history rather than
        # recorded, because a file cannot contain the hash of the commit that adds
        # it. See the verifier's docstring.
        "evidence_commit_resolution":
            "git log -1 --format=%H -- results/baseline/g2/g2_adjudication.json",
        "roles": ROLE_POLICY,
        # An entry here is the only thing that permits the current
        # `src/baseline/ldpc/` bytes to differ from the adjudicated ones. Adding one
        # means a new G-2 campaign really ran; it is not a way to silence the check.
        "readjudications": [],
        "sources": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="regenerate in memory and compare the committed bytes")
    args = parser.parse_args()

    adjudication = json.loads(
        (manifest_path().parent / "g2_adjudication.json").read_text()
    )
    manifest = build(adjudication["measurement_commit"])
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    target = manifest_path()

    if args.check:
        if not target.exists():
            print(f"{SOURCE_MANIFEST} is absent; run without --check", file=sys.stderr)
            return 1
        if target.read_text() != rendered:
            print(f"{SOURCE_MANIFEST} is stale; run without --check", file=sys.stderr)
            return 1
        print(f"ok: {SOURCE_MANIFEST} matches the measurement commit "
              f"({len(manifest['sources'])} sources)")
        return 0

    target.write_text(rendered)
    roles = sorted({source["role"] for source in manifest["sources"]})
    print(f"wrote {target.relative_to(REPO)}: {len(manifest['sources'])} sources, "
          f"roles {', '.join(roles)}, measurement {manifest['measurement_commit'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
