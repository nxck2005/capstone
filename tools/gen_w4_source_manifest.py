#!/usr/bin/env python3
"""Generate the W4 bounded-evidence execution-source manifest.

The bounded W4 evidence is only worth what its provenance is worth. This binds
every source that participated in producing it — runtime modules, the PB_1
transport dependencies actually used, the outage and record layers, the frozen
classifier loader, preprocessing, identity and RNG utilities, the dataset
loader, the runner, and the committed configuration and frozen artifacts — to
the clean runner-ready commit the evidence declares.

For each path the manifest records the repository-relative path, a role, the Git
blob object id at the execution commit, the SHA-256 of those exact bytes, and
the byte length. `--check` regenerates in memory and compares against the
committed bytes.

Unlike G-2, there is no re-adjudication mechanism here and there should not be
one: the W4 evidence is a *bounded* run that takes about a minute, so the honest
response to a changed runtime source is to rerun it, not to write a justification
for why the change did not matter.

Usage:
    .venv/bin/python tools/gen_w4_source_manifest.py
    .venv/bin/python tools/gen_w4_source_manifest.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SOURCE_MANIFEST = Path("results/baseline/w4/execution_source_manifest.json")
SMOKE_SUMMARY = Path("results/baseline/w4/smoke_summary.json")

#: Every source whose bytes could change a recorded W4 number, with its role.
#: The set is declared here and enforced by the verifier, so the manifest cannot
#: define its own completeness by quietly omitting a file.
EXPECTED_SOURCES: dict[str, str] = {
    # The classical arm itself.
    "src/baseline/classical/__init__.py": "runtime",
    "src/baseline/classical/channel_transport.py": "runtime",
    "src/baseline/classical/pipeline.py": "runtime",
    "src/baseline/classical/outage.py": "runtime",
    "src/baseline/classical/records.py": "runtime",
    # PB_1 transport dependencies actually exercised by the bounded run.
    "src/baseline/ldpc/adapter.py": "runtime",
    "src/baseline/ldpc/crc.py": "runtime",
    "src/baseline/ldpc/modulation.py": "runtime",
    "src/baseline/ldpc/rate_matching.py": "runtime",
    "src/baseline/ldpc/segmentation.py": "runtime",
    "src/baseline/ldpc/transport.py": "runtime",
    "src/baseline/j2k.py": "runtime",
    "src/channels/awgn.py": "runtime",
    "src/channels/power.py": "runtime",
    "src/channels/registry.py": "runtime",
    # Frozen classifier, preprocessing and the shared identity utilities.
    "src/models/frozen_reference_classifier.py": "runtime",
    "src/models/reference_classifier.py": "runtime",
    "src/data/preprocessing.py": "runtime",
    "src/data/classifier.py": "runtime",
    "src/artifacts/ids.py": "runtime",
    "src/artifacts/rng.py": "runtime",
    # Dataset loading and manifest identity.
    "src/data/registry.py": "runtime",
    "src/data/adapters.py": "runtime",
    "src/data/manifests.py": "runtime",
    "src/data/provenance.py": "runtime",
    "src/data/identity.py": "runtime",
    "src/config/params.py": "runtime",
    "src/config/run_config.py": "runtime",
    # The runner and the frozen-artifact generator.
    "tools/run_classical_baseline_w4_smoke.py": "measurement_runner",
    "tools/gen_w4_outage_policy.py": "measurement_runner",
    # Committed configuration.
    "configs/classical-baseline-w4-smoke.yaml": "configuration",
    "configs/classical-baseline-w4-smoke-plan.yaml": "configuration",
    "spec/params.generated.yaml": "configuration",
    "data/manifests/imagenette160.csv": "configuration",
    "data/manifests/cifar10.csv": "configuration",
    # Frozen artifacts that exist *before* the run and feed into it.
    #
    # The produced records — per_image.csv and aggregate.csv — are deliberately
    # NOT bound here: they are outputs of the execution commit, so they cannot
    # exist at it. They are bound instead by SHA-256 inside smoke_summary.json,
    # which the verifier recomputes from the files on disk.
    "results/baseline/w4/outage_policy.json": "record",
    "results/reference_classifier/g1_adjudication.json": "record",
}

ROLE_POLICY = {
    "runtime": (
        "asserted byte-identical at HEAD: a change means the recorded numbers "
        "describe a different implementation, and the bounded run must be rerun"
    ),
    "measurement_runner": "asserted byte-identical at HEAD",
    "configuration": "asserted byte-identical at HEAD",
    "record": "bound as the exact evidence bytes produced by the execution commit",
}


class ManifestError(RuntimeError):
    """A W4 source-manifest contract violation."""


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise ManifestError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def git_bytes(commit: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ManifestError(
            f"{path} is not present at {commit}: "
            f"{result.stderr.decode('utf-8', 'replace').strip()}"
        )
    return result.stdout


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def manifest_path(repo_root: Path = REPO) -> Path:
    return repo_root / SOURCE_MANIFEST


def execution_commit(repo_root: Path = REPO) -> str:
    summary = json.loads((repo_root / SMOKE_SUMMARY).read_text(encoding="utf-8"))
    commit = summary.get("execution_source_commit")
    if not isinstance(commit, str) or not commit:
        raise ManifestError("smoke_summary.json records no execution_source_commit")
    return commit


def build(commit: str) -> dict:
    sources = []
    for path, role in sorted(EXPECTED_SOURCES.items()):
        content = git_bytes(commit, path)
        sources.append(
            {
                "path": path,
                "role": role,
                "git_blob_sha": git("rev-parse", f"{commit}:{path}"),
                "sha256": sha256_bytes(content),
                "bytes": len(content),
                "source_commit": commit,
            }
        )
    return {
        "schema_version": 1,
        "kind": "w4_execution_source_manifest",
        "phase": "W4 bounded classical-baseline integration",
        "generated_by": "tools/gen_w4_source_manifest.py",
        "execution_source_commit": commit,
        "git_object_format": git("rev-parse", "--show-object-format"),
        "roles": ROLE_POLICY,
        "readjudications_permitted": False,
        "readjudication_policy": (
            "None. The bounded W4 run is cheap enough to rerun, so a changed "
            "runtime source is answered by rerunning the evidence, never by "
            "recording an exception."
        ),
        "sources": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="regenerate in memory and compare the committed bytes")
    arguments = parser.parse_args()

    try:
        commit = execution_commit()
        manifest = build(commit)
    except ManifestError as exc:
        print(f"W4 source manifest FAIL: {exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    target = manifest_path()

    if arguments.check:
        if not target.exists():
            print(f"{SOURCE_MANIFEST} is absent; run without --check", file=sys.stderr)
            return 1
        if target.read_text(encoding="utf-8") != rendered:
            print(f"{SOURCE_MANIFEST} is stale; run without --check", file=sys.stderr)
            return 1
        print(
            f"ok: {SOURCE_MANIFEST} matches the execution commit "
            f"{commit[:12]} ({len(manifest['sources'])} sources)"
        )
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered, encoding="utf-8")
    print(f"wrote {SOURCE_MANIFEST}: {len(manifest['sources'])} sources at {commit[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
