#!/usr/bin/env python3
"""Copy two frozen W10-bound weights into an ignored, portable local bundle.

This script neither downloads nor loads models. It verifies their published
identities before atomically installing them; no smoke/pilot substitutions.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "demo/assets/checkpoints"
AUTH = ROOT / "results/learned/w10/w10_rehearsal_authorization.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bindings() -> dict[str, dict]:
    authority = json.loads(AUTH.read_text())
    if authority["test"] != "SEALED" or authority["test_access"] != 0:
        raise ValueError("W10 authority does not preserve sealed test")
    return {b["scope"]["system"]: b for b in authority["bindings"] if b["scope"]["bw_ratio"] == "r_1_6"}


def install(source: Path, name: str, expected: str) -> None:
    if digest(source) != expected:
        raise ValueError(f"{source}: SHA-256 differs from frozen W10 authority ({expected})")
    DEST.mkdir(parents=True, exist_ok=True)
    target = DEST / name
    with tempfile.NamedTemporaryFile(dir=DEST, prefix=".weight-", delete=False) as temporary:
        staged = Path(temporary.name)
        try:
            with source.open("rb") as original:
                shutil.copyfileobj(original, temporary)
        except BaseException:
            staged.unlink(missing_ok=True)
            raise
    try:
        if digest(staged) != expected:
            raise ValueError("copied bytes differ from frozen authority")
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)
    print(f"{target.relative_to(ROOT)}: verified {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned", type=Path, help="W8 train0/channel0 r_1_6 selected epoch-0092.pt")
    parser.add_argument("--artifact", type=Path, help="G8_F artifact classifier epoch-17.pt")
    parser.add_argument("--verify", action="store_true", help="verify existing local assets without copying")
    args = parser.parse_args()
    b = bindings()
    expected = {
        "learned-r1-6.pt": b["learned"]["checkpoint"]["checkpoint_id"],
        "artifact-classifier.pt": b["classical_adaptive"]["scorer"]["checkpoint_file_sha256"],
    }
    if args.verify:
        all_verified = True
        for name, sha in expected.items():
            path = DEST / name
            verified = path.is_file() and not path.is_symlink() and digest(path) == sha
            print(f"{name}: {'VERIFIED' if verified else 'UNAVAILABLE / WRONG SHA'}")
            all_verified &= verified
        if not all_verified:
            raise SystemExit(1)
        return
    if not args.learned and not args.artifact:
        parser.error("provide --learned and/or --artifact, or --verify")
    if args.learned:
        install(args.learned, "learned-r1-6.pt", expected["learned-r1-6.pt"])
    if args.artifact:
        install(args.artifact, "artifact-classifier.pt", expected["artifact-classifier.pt"])


if __name__ == "__main__":
    main()
