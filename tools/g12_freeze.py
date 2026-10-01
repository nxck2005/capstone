#!/usr/bin/env python3
"""Generate or verify the G-12 freeze manifest (SR-22, DEC-12).

    python tools/g12_freeze.py generate   # writes results/freeze_manifest.json
    python tools/g12_freeze.py verify     # recomputes it and compares bytes

The manifest covers every field in ``params.evaluation.freeze_manifest_covers``
and binds the exact code commit it was generated from.  Generation refuses a
dirty protected tree, an existing manifest, or any recorded test access.  The
manifest is then committed on its own: the campaign runner requires that the
only difference between ``code_commit`` and the commit that opens test is this
file.  Nothing here reads a test sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402

ARTIFACT_ROLE = "G12_FREEZE_MANIFEST"
SCHEMA_VERSION = 1
PROTECTED = ("src/", "tools/", "configs/", "spec/", "tests/", "requirements-pascal.lock")
ANALYSIS_SOURCES = ("src/evaluation/g12_analysis.py", "tools/run_g12_campaign.py")
CONFIGS = (
    "configs/er9-digital-pascal-v4.yaml",
    "configs/learned-er2-randomized-pascal-v4.yaml",
)
G8_CLOSEOUT = "results/baseline/g8/g8_closeout.json"
AUTHORIZATION = (
    "owner decision 2026-10-02: final test numbers for the paper; Option B (AM-100) "
    "added before the freeze; the owner approves this freeze before it is committed"
)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def _sha(relative: str) -> dict[str, Any]:
    raw = (REPO / relative).read_bytes()
    return {"path": relative, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"G-12 freeze HOLD: {message}")


def _h2_window() -> dict[str, Any]:
    closeout = json.loads((REPO / G8_CLOSEOUT).read_bytes())
    window = closeout["br16_h2_validation_freeze"]
    _require(window["system"] == get("evaluation.cliff_reference_system"), "H2 window names a different reference system")
    _require(float(window["window_width_db"]) == float(get("evaluation.cliff_window_db")), "H2 window width differs")
    return {
        "source": _sha(G8_CLOSEOUT),
        "system": window["system"],
        "ratio": window["ratio"],
        "low_snr_db": float(window["low_snr_db"]),
        "high_snr_db": float(window["high_snr_db"]),
        "selected_on": "validation_split_at_G-8",
        "fixed_configuration": window["fixed_configuration"],
    }


def build(*, code_commit: str) -> dict[str, Any]:
    from evaluation import g12_analysis
    from evaluation.g12_bindings import resolve
    from evaluation.g12_scope import scope_identity, scope_sha256
    from training.deterministic_core import canonical_sha256

    bindings = resolve(REPO)
    checkpoints = [
        {"role": item["arm"]["role"], "cell": item["cell"], "binding_id": item["binding_id"], "checkpoint": item["checkpoint"]}
        for item in bindings
    ]
    operating_points = [
        {"role": item["arm"]["role"], "cell": item["cell"], "selection": item["selection"],
         "per_snr_phy": item["checkpoint"].get("per_snr_phy"),
         "fixed_phy": {key: item["checkpoint"].get(key) for key in ("modulation", "ldpc_rate", "payload_bits")} if item["arm"]["backend"] == "er9_low_rate" else None}
        for item in bindings
    ]
    dataset = "imagenette160"
    manifest_file = f"data/manifests/{dataset}.csv"
    body = {
        "artifact_role": ARTIFACT_ROLE,
        "schema_version": SCHEMA_VERSION,
        "gate": get("evaluation.test_access_gate"),
        "authorization": AUTHORIZATION,
        "code_commit": code_commit,
        "resolved_config": {
            "spec": _sha("spec/SPEC.md"),
            "params": _sha("spec/params.generated.yaml"),
            "configs": [_sha(path) for path in CONFIGS],
            "requirements_pascal_lock": _sha("requirements-pascal.lock"),
        },
        "split_manifest": {
            **_sha(manifest_file),
            "pinned_manifest_sha256": get(f"datasets.{dataset}.manifest_sha256"),
            "test_images": int(get(f"datasets.{dataset}.test_images")),
        },
        "checkpoints": checkpoints,
        "classifier_variant": {
            "headline_scorer": get("reference_classifier.headline_scorer"),
            "classical_variants": ["artifact_finetuned", "clean"],
            "learned_scorer": "own_task_head",
            "sources": [_sha("results/baseline/g8_f/artifact_classifier_freeze.json"), _sha("results/reference_classifier/best_checkpoint.json")],
        },
        "operating_points": operating_points,
        "h2_window": _h2_window(),
        "analysis_version": {
            "config_analysis_version": int(get("config.analysis_version")),
            "implementation": g12_analysis.ANALYSIS_IMPLEMENTATION,
            "choices": g12_analysis.ANALYSIS_CHOICES,
            "sources": [_sha(path) for path in ANALYSIS_SOURCES],
            "bootstrap_resamples": int(get("evaluation.bootstrap_resamples")),
            "calibration_resamples": int(get("evaluation.h1_run_permutation_resamples")),
        },
        "scope": {**scope_identity(), "sha256": scope_sha256()},
        "test_campaign_is_single": bool(get("evaluation.test_campaign_is_single")),
        "test_access_before_freeze": 0,
    }
    _require(body["test_campaign_is_single"], "params.evaluation.test_campaign_is_single must be true")
    missing = [field for field in get("evaluation.freeze_manifest_covers") if not body.get(field)]
    _require(not missing, f"freeze manifest does not cover {missing}")
    body["freeze_id"] = "g12freeze-" + canonical_sha256(body)
    return body


def _encode(value: dict[str, Any]) -> bytes:
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False).encode() + b"\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("generate", "verify"))
    args = parser.parse_args()
    path = REPO / str(get("artifacts.freeze_manifest_file"))
    if args.command == "generate":
        _require(not path.exists(), f"{path.relative_to(REPO)} already exists; G-12 freezes once")
        dirty = _git("status", "--porcelain", "--untracked-files=all", "--", *PROTECTED)
        _require(not dirty, f"protected tree is dirty:\n{dirty}")
        value = build(code_commit=_git("rev-parse", "HEAD"))
        path.write_bytes(_encode(value))
        print(f"wrote {path.relative_to(REPO)}: {value['freeze_id']}")
        return 0
    value = json.loads(path.read_bytes())
    rebuilt = build(code_commit=value["code_commit"])
    _require(_encode(rebuilt) == path.read_bytes(), "freeze manifest does not reproduce from the current tree")
    changed = _git("diff", "--name-only", value["code_commit"], "HEAD", "--", *PROTECTED)
    _require(not changed, f"protected source changed after the freeze commit: {changed}")
    print(f"G-12 freeze manifest PASS: {value['freeze_id']} at code commit {value['code_commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
