#!/usr/bin/env python3
"""Package the bounded offline exhibit without the multi-GB research runtime.

Dependencies must be installed on the target laptop before disconnecting. This
archive contains verified frozen weights, the built static UI, and only the
scientific source and immutable input records used by the demo adapter.
"""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from provision_assets import DEST, bindings, digest


ROOT = Path(__file__).resolve().parents[2]
FILES = (
    "requirements-cpu.lock",
    "spec/params.generated.yaml",
    "data/manifests/imagenette160.csv",
    "results/learned/w8/w8_c_reconciliation.json",
    "results/learned/w10/w10_rehearsal_authorization.json",
    "results/learned/w10/w10_continuation_closeout_v11.json",
    "results/learned/w10/w10_continuation_units_v11.json",
    "results/learned/w10/w10_continuation_unit_manifest_v11.json",
    "results/learned/w10/w10_continuation_per_image_manifest_v11.json",
    "results/baseline/g8_e/candidate_authority.json",
    "results/baseline/g8_f/pass_two_state.json",
    "results/baseline/g8_f/artifact_classifier_freeze.json",
    "results/baseline/w4/outage_policy.json",
    "presentation-results/data/w10_primary_252.csv",
    "presentation-results/data/w10_scorers_357.csv",
)
TREES = ("src", "configs", "demo/frontend/dist", "demo/assets/examples")
DEMO_FILES = (
    "demo/README.md", "demo/backend/app.py", "demo/backend/API.md",
    "demo/backend/requirements.txt", "demo/scripts/start.sh",
    "demo/scripts/serve.py", "demo/scripts/provision_assets.py",
    "demo/scripts/smoke_api.py", "demo/scripts/package_offline.py",
    "demo/assets/checkpoints/learned-r1-6.pt",
    "demo/assets/checkpoints/artifact-classifier.pt",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="destination .tar.gz file (outside the repo recommended)")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.suffixes[-2:] != [".tar", ".gz"]:
        parser.error("output must end in .tar.gz")
    b = bindings()
    required = {
        "learned-r1-6.pt": b["learned"]["checkpoint"]["checkpoint_id"],
        "artifact-classifier.pt": b["classical_adaptive"]["scorer"]["checkpoint_file_sha256"],
    }
    for name, sha in required.items():
        asset = DEST / name
        if not asset.is_file() or asset.is_symlink() or digest(asset) != sha:
            raise ValueError(f"{name}: missing or not the authority-bound frozen weight")
    prefix = "capstone-exhibit"
    output.parent.mkdir(parents=True, exist_ok=True)

    def safe(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if any(part in {"__pycache__", ".cache", "node_modules"} for part in Path(info.name).parts):
            return None
        if not info.isfile() and not info.isdir():
            raise ValueError(f"refusing unsafe archive entry: {info.name}")
        return info

    with tarfile.open(output, "w:gz") as archive:
        for relative in (*FILES, *TREES, *DEMO_FILES):
            path = ROOT / relative
            if not path.exists() or path.is_symlink():
                raise FileNotFoundError(f"required portable exhibit input missing: {relative}")
            archive.add(path, arcname=f"{prefix}/{relative}", recursive=True, filter=safe)
    h = hashlib.sha256()
    with output.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    print(json.dumps({"archive": str(output), "bytes": output.stat().st_size, "sha256": h.hexdigest(), "extract_to": prefix}))


if __name__ == "__main__":
    main()
