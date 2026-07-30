#!/usr/bin/env python
"""Freeze (or re-verify) the W4 constant-class outage-policy artifact.

The selection is a label-frequency calculation over the *entire* committed
validation manifest: no image is decoded, no classifier runs, and no test row is
read.  ``--check`` regenerates the record in memory and compares it field by
field against the committed bytes, ignoring only the two fields that are
explicitly allowed to move — the generation timestamp and the source commit that
produced it.

Usage:
    .venv/bin/python tools/gen_w4_outage_policy.py
    .venv/bin/python tools/gen_w4_outage_policy.py --check
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from baseline.classical.outage import (  # noqa: E402
    build_outage_policy_record,
    load_outage_policy,
    write_json_atomically,
)

OUTAGE_POLICY_PATH = Path("results/baseline/w4/outage_policy.json")

#: The classifier that scores real W4 task rows is the adjudicated G-1
#: Imagenette-160 model, so the frozen outage class must come from the same
#: label vocabulary.  This is a semantic constraint, not a convenience default.
OUTAGE_DATASET = "imagenette160"

#: Fields whose value legitimately depends on *when* and *where* the artifact
#: was generated rather than on what was measured.
_PROVENANCE_FIELDS = ("generated_at", "selection_source_commit")


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise SystemExit(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _record(*, source_commit: str, generated_at: str) -> dict:
    return build_outage_policy_record(
        OUTAGE_DATASET,
        selection_source_commit=source_commit,
        generated_at=generated_at,
        repo_root=REPO,
    )


def _comparable(record: dict) -> dict:
    return {key: value for key, value in record.items() if key not in _PROVENANCE_FIELDS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate in memory and compare against the committed artifact",
    )
    arguments = parser.parse_args()
    target = REPO / OUTAGE_POLICY_PATH

    if arguments.check:
        if not target.is_file():
            print(f"missing committed outage policy artifact: {OUTAGE_POLICY_PATH}")
            return 1
        committed = json.loads(target.read_text(encoding="utf-8"))
        regenerated = _record(
            source_commit=str(committed.get("selection_source_commit", "")),
            generated_at=str(committed.get("generated_at", "")),
        )
        if _comparable(committed) != _comparable(regenerated):
            differing = sorted(
                key
                for key in set(_comparable(committed)) | set(_comparable(regenerated))
                if _comparable(committed).get(key) != _comparable(regenerated).get(key)
            )
            print(f"outage policy artifact differs from regeneration: {differing}")
            return 1
        recorded_commit = str(committed.get("selection_source_commit", ""))
        if not recorded_commit:
            print("outage policy artifact records no selection_source_commit")
            return 1
        try:
            _git("cat-file", "-e", f"{recorded_commit}^{{commit}}")
        except SystemExit:
            print(f"selection_source_commit is not reachable: {recorded_commit}")
            return 1
        policy = load_outage_policy(target, expected_dataset=OUTAGE_DATASET)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        print(
            f"ok: {OUTAGE_POLICY_PATH} matches regeneration "
            f"(dataset={policy.dataset}, selected_class={policy.selected_class}, "
            f"accuracy={policy.selected_count}/{policy.validation_count}, "
            f"sha256={digest})"
        )
        return 0

    source_commit = _git("rev-parse", "HEAD")
    generated_at = dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds")
    record = _record(source_commit=source_commit, generated_at=generated_at)
    write_json_atomically(target, record)
    policy = load_outage_policy(target, expected_dataset=OUTAGE_DATASET)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    print(
        f"wrote {OUTAGE_POLICY_PATH}: dataset={policy.dataset}, "
        f"class_counts={list(policy.class_counts)}, "
        f"tied_maximum_classes={list(policy.tied_maximum_classes)}, "
        f"selected_class={policy.selected_class}, "
        f"accuracy={policy.selected_count}/{policy.validation_count}"
        f"={policy.measured_validation_accuracy}, sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
