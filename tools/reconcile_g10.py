#!/usr/bin/env python3
"""Fresh-process, read-only terminal reconciliation for W9-A/G-10."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from evaluation.g10_protocol import (  # noqa: E402
    RECONCILIATION_PATH,
    canonical_sha256,
    rendered_json,
    sha256_file,
)
from verify_g10_w9 import verify  # noqa: E402


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> int:
    try:
        verified = verify(REPO)
        body = {
            "schema_version": 1,
            "artifact_role": "W9A_G10_FRESH_PROCESS_RECONCILIATION",
            "status": "W9A_G10_RECONCILED_GREEN_STOP",
            "terminal_commit_before_reconciliation": _head(),
            "authority_id": verified["authorization"]["authorization_id"],
            "source_commit": verified["authorization"]["scientific_source"]["commit"],
            "completion_id": verified["completion"]["completion_id"],
            "completion_sha256": sha256_file(REPO / "results/learned/w9/w9a_completion.json"),
            "adjudication_id": verified["adjudication"]["adjudication_id"],
            "adjudication_sha256": sha256_file(REPO / "results/learned/w9/g10_adjudication.json"),
            "cell_index_id": verified["index"]["index_id"],
            "cell_index_sha256": sha256_file(REPO / "results/learned/w9/g10_cell_index.json"),
            "runtime_manifest_id": verified["runtime"]["runtime_manifest_id"],
            "runtime_manifest_sha256": verified["runtime_manifest_sha256"],
            "classification": verified["adjudication"]["classification"],
            "grid_db": verified["index"]["snr_grid_db"],
            "complete_learned_evaluations": 63,
            "headline_comparator": "G8/F3 adaptive/oracle classical r_1_6",
            "protected_counters": verified["adjudication"]["protected_counters"],
            "test": "SEALED",
            "next_action": "STOP; separate owner authority required for ER-9, randomized ER-2, G-11 or W10",
        }
        value = dict(body)
        value["reconciliation_id"] = "w9areconcile-" + canonical_sha256(body)
        value["artifact_content_sha256"] = canonical_sha256(value)
        path = REPO / RECONCILIATION_PATH
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"refusing to replace immutable reconciliation: {path}")
        path.write_bytes(rendered_json(value))
    except Exception as exc:  # this is a terminal verifier CLI; preserve a concise HOLD
        print(f"G-10 RECONCILIATION HOLD — {exc}", file=sys.stderr)
        return 1
    print(
        "W9-A/G-10 fresh-process reconciliation PASS: "
        f"{value['reconciliation_id']} classification={value['classification']} test=SEALED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
