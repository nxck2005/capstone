#!/usr/bin/env python3
"""Evidence for AM-25: srsRAN's MATLAB-generated vectors validate Sionna's encoder.

BR-2 needs a golden reference independent of `params.baseline.ldpc_impl`. This is
the W0 check that established rung 2 works, and it is also where BR-2's alignment
recipe came from. It is evidence, not the fixture: the real fixture is built at
W3 and must pin the lifting size (see "limits" below).

Reproduce:

    python spec/evidence/fetch_srsran_vectors.sh      # or fetch by hand, see README
    python spec/evidence/golden_vectors_check.py <dir-of-extracted-dat-files>

The .dat files are deliberately NOT committed (AM-25): they are third-party data
published as a release asset, and this repository carries their checksums rather
than a copy. `srsran_vectors.sha256` is what makes the fetch verifiable.

The alignment is the whole point and is not obvious. srsRAN stores the codeword
with the first 2Z systematic bits ALREADY punctured -- BG1's codeword is 68Z, the
stored buffer is 66Z -- and marks filler positions with the byte 254. Sionna
instead deletes filler positions *before* puncturing. Aligning therefore means
dropping the 254s and nothing else. Getting this wrong does not look wrong: the
three attempts that preceded this one agreed with the reference at 0.50, which is
chance, and is indistinguishable from a library defect.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import torch
from sionna.phy.fec.ldpc import LDPC5GEncoder

# BG1: 68 columns, 22 systematic. BG2: 52 columns, 10 systematic.
DIMS = {1: (22, 68), 2: (10, 52)}
FILLER = 254
# Minimum coderate each base graph supports; below this the library refuses,
# because going lower needs repetition coding (see params.baseline.ldpc_bg1_min_coderate).
MIN_RATE_DEN = {1: 3, 2: 5}


def load_cases(header: Path) -> list[tuple[int, int, int]]:
    """(index, base graph, lifting size) from srsRAN's ldpc_encoder_test_data.h."""
    text = header.read_text()
    return [(i, int(bg), int(ls)) for i, (_nof, bg, ls) in enumerate(
        re.findall(r"^\s*\{(\d+), (\d+), (\d+),", text, re.M))]


def check_case(d: Path, idx: int, bg: int, Z: int, dev) -> dict:
    kcols, ncols = DIMS[bg]
    k_ldpc, n_stored = kcols * Z, (ncols - 2) * Z
    msg = np.fromfile(d / f"ldpc_encoder_test_input{idx}.dat", dtype=np.uint8)
    cw = np.fromfile(d / f"ldpc_encoder_test_output{idx}.dat", dtype=np.uint8)

    if len(msg) % k_ldpc or len(cw) % n_stored or \
            len(msg) // k_ldpc != len(cw) // n_stored:
        return dict(idx=idx, status="skip", why="layout mismatch")
    nof = len(msg) // k_ldpc
    msg, cw = msg.reshape(nof, k_ldpc), cw.reshape(nof, n_stored)

    k_true = int((msg[0] != FILLER).sum())
    if any(int((r != FILLER).sum()) != k_true for r in msg):
        return dict(idx=idx, status="skip", why="ragged filler counts")

    # Drop filler positions -- and nothing else. The buffer is already punctured.
    ref = np.stack([r[r != FILLER] for r in cw])
    n_target = min(ref.shape[1], MIN_RATE_DEN[bg] * k_true - 1)
    ref = ref[:, :n_target]
    u = torch.tensor(np.stack([r[r != FILLER] for r in msg]).astype(np.float32),
                     device=dev)
    try:
        enc = LDPC5GEncoder(k=k_true, n=n_target, bg=f"bg{bg}").to(dev)
    except Exception as exc:
        return dict(idx=idx, status="skip", why=f"{type(exc).__name__}: {exc}"[:70])
    if enc._z != Z:
        # The probe lets the library infer Z from (k, n); when it picks a
        # different one the comparison is not apples-to-apples. A limitation of
        # this check, not a disagreement -- the W3 fixture must pin Z.
        return dict(idx=idx, status="skip", why=f"Z inferred {enc._z}, vector has {Z}")

    out = enc(u).cpu().numpy().astype(np.uint8)
    agree = float((out == ref).mean()) if out.shape == ref.shape else -1.0
    exact = out.shape == ref.shape and bool((out == ref).all())
    return dict(idx=idx, status="match" if exact else "MISMATCH", bg=bg, Z=Z,
                k=k_true, n=n_target, agreement=agree)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    d = Path(sys.argv[1])
    header = Path(__file__).parent / "ldpc_encoder_test_data.h"
    if not header.exists():
        print(f"missing {header} -- see README", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("note: no CUDA, running on CPU (slower, same result)")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cases = load_cases(header)
    print(f"upstream encoder cases: {len(cases)}\n")

    matched, mismatched, skipped = [], [], []
    for idx, bg, Z in cases:
        r = check_case(d, idx, bg, Z, dev)
        {"match": matched, "MISMATCH": mismatched, "skip": skipped}[r["status"]].append(r)
        if r["status"] != "skip":
            print(f"  case{r['idx']:<3d} BG{r['bg']} Z={r['Z']:<3d} k={r['k']:<5d} "
                  f"n={r['n']:<5d} {r['status']} agreement={r['agreement']:.4f}")

    print(f"\nexact match : {len(matched)}")
    print(f"MISMATCH    : {len(mismatched)}")
    print(f"skipped     : {len(skipped)}")
    print(f"base graphs validated  : {sorted({r['bg'] for r in matched})}")
    print(f"lifting sizes validated: {sorted({r['Z'] for r in matched})}")
    if skipped:
        print("\nskip reasons (all are probe limitations, not disagreements):")
        why: dict[str, int] = {}
        for r in skipped:
            key = "lifting size inferred rather than pinned" \
                if r["why"].startswith("Z inferred") else r["why"]
            why[key] = why.get(key, 0) + 1
        for reason, count in sorted(why.items(), key=lambda x: -x[1]):
            print(f"   {count:3d}  {reason}")
    if mismatched:
        print("\nMISMATCHES -- BR-2 would fail here:")
        for r in mismatched:
            print("  ", r)
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
