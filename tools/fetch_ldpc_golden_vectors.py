#!/usr/bin/env python3
"""Fetch, authenticate and convert the pinned srsRAN rung-2 LDPC fixture."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from config.params import REPO_ROOT, get
from baseline.ldpc.modulation import bits_per_symbol, interleave

FILLER_MARKER = 254
MIN_RATE_DENOMINATOR = {1: 3, 2: 5}
ENCODER_ARCHIVE = "ldpc_encoder_test_data.tar.gz"


def _digest(read) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return _digest(handle.read)


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output)


def _extract_encoder_archive(asset: Path, destination: Path) -> dict[str, str]:
    """Authenticate every pinned inner vector archive, then extract the encoder one.

    All of `baseline.ldpc_golden_vector_sha256` is checked, not just the encoder archive
    this fixture consumes: a substituted release asset that carries the right encoder
    bytes but tampered rate-matcher or segmenter bytes must not pass authentication
    merely because the converter happens not to read those members.
    """
    pinned = dict(get("baseline.ldpc_golden_vector_sha256"))
    if ENCODER_ARCHIVE not in pinned:
        raise RuntimeError(f"{ENCODER_ARCHIVE} is not pinned in baseline.ldpc_golden_vector_sha256")
    verified: dict[str, str] = {}
    inner_bytes = b""
    with tarfile.open(asset) as outer:
        for item in outer.getmembers():
            name = PurePosixPath(item.name).name
            if name not in pinned or name in verified:
                continue
            extracted = outer.extractfile(item)
            if extracted is None:
                raise RuntimeError(f"could not read pinned vector archive {name}")
            if name == ENCODER_ARCHIVE:
                inner_bytes = extracted.read()
                actual = hashlib.sha256(inner_bytes).hexdigest()
            else:
                actual = _digest(extracted.read)
            if actual != pinned[name]:
                raise RuntimeError(
                    f"inner vector-archive checksum mismatch for {name}: {actual} != {pinned[name]}"
                )
            verified[name] = actual
    missing = sorted(set(pinned) - set(verified))
    if missing:
        raise RuntimeError(
            f"authenticated release asset lacks pinned vector archives: {', '.join(missing)}"
        )
    with tarfile.open(fileobj=io.BytesIO(inner_bytes), mode="r:gz") as inner:
        inner.extractall(destination, filter="data")
    return verified


def convert(vectors: Path, output: Path) -> dict:
    arrays: dict[str, np.ndarray] = {}
    cases = []
    for case in get("baseline.ldpc_golden_vector_cases"):
        index, bg, z = int(case["index"]), int(case["base_graph"]), int(case["lifting_size"])
        q_m = bits_per_symbol(case["modulation"])
        k_columns = 22 if bg == 1 else 10
        n_columns = int(get("baseline.ldpc_mother_code_columns")[f"bg{bg}"])
        message_width = k_columns * z
        stored_width = (n_columns - int(get("baseline.ldpc_punctured_systematic_columns"))) * z
        raw_message = np.fromfile(vectors / f"ldpc_encoder_test_input{index}.dat", dtype=np.uint8)
        raw_codeword = np.fromfile(vectors / f"ldpc_encoder_test_output{index}.dat", dtype=np.uint8)
        if raw_message.size % message_width or raw_codeword.size % stored_width:
            raise RuntimeError(f"upstream case {index} has an unexpected layout")
        messages = raw_message.reshape(-1, message_width)
        codewords = raw_codeword.reshape(-1, stored_width)
        k_true = int(np.count_nonzero(messages[0] != FILLER_MARKER))
        if any(np.count_nonzero(row != FILLER_MARKER) != k_true for row in messages):
            raise RuntimeError(f"upstream case {index} has ragged filler")
        inputs = np.stack([row[row != FILLER_MARKER] for row in messages]).astype(np.uint8)
        aligned = np.stack([row[row != FILLER_MARKER] for row in codewords]).astype(np.uint8)
        n = min(aligned.shape[1], MIN_RATE_DENOMINATOR[bg] * k_true - 1)
        n -= n % q_m
        encoder_reference = aligned[:, :n]
        rate_matched_reference = interleave(encoder_reference, q_m)
        prefix = f"case_{index}"
        arrays[f"{prefix}_input"] = inputs
        arrays[f"{prefix}_encoder"] = encoder_reference
        arrays[f"{prefix}_rate_matched"] = rate_matched_reference
        cases.append({
            **case,
            "k": k_true,
            "n": n,
            "messages": int(inputs.shape[0]),
            "alignment": "drop_filler_marker_254_only_already_2Z_punctured",
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return {"cases": cases, "fixture_sha256": sha256(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset",
        type=Path,
        default=REPO_ROOT / "data" / "archives" / get("baseline.ldpc_golden_vector_upstream_asset"),
    )
    parser.add_argument("--vectors-dir", type=Path)
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / get("baseline.ldpc_golden_vector_file")
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-materialize even if the fixture is already present",
    )
    args = parser.parse_args()
    expected = get("baseline.ldpc_golden_vector_asset_sha256")
    record = {
        "source_rung": int(get("baseline.ldpc_golden_vector_source_rung")),
        "release": get("baseline.ldpc_golden_vector_upstream_release"),
        "asset": get("baseline.ldpc_golden_vector_upstream_asset"),
        "asset_sha256_expected": expected,
        "fixture": str(args.output.relative_to(REPO_ROOT) if args.output.is_relative_to(REPO_ROOT)
                       else args.output),
    }
    # An existing fixture is authoritative, so this path is a network-free no-op: running
    # the tool on a clone that already materialized it must not re-fetch the release asset.
    if args.output.exists() and not args.force:
        print(json.dumps({
            **record,
            "already_present": True,
            "downloaded": False,
            "asset_sha256_verified": None,
            "inner_archives_verified": None,
            "cases": None,
            "fixture_sha256": sha256(args.output),
        }, indent=2))
        return 0
    downloaded = False
    inner = None
    if args.vectors_dir is None:
        if not args.asset.exists():
            _download(get("baseline.ldpc_golden_vector_upstream_url"), args.asset)
            downloaded = True
        actual = sha256(args.asset)
        if actual != expected:
            raise RuntimeError(f"release asset checksum mismatch: {actual} != {expected}")
        with tempfile.TemporaryDirectory(prefix="capstone-ldpc-vectors-") as directory:
            vectors = Path(directory)
            inner = _extract_encoder_archive(args.asset, vectors)
            summary = convert(vectors, args.output)
    else:
        summary = convert(args.vectors_dir, args.output)
        actual = None
    print(json.dumps({
        **record,
        "already_present": False,
        "downloaded": downloaded,
        "asset_sha256_verified": actual,
        "inner_archives_verified": inner,
        **summary,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
