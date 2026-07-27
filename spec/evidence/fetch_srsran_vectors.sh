#!/usr/bin/env bash
# Fetch and verify srsRAN's LDPC test vectors (BR-2 rung 2, AM-25).
#
# The vectors are NOT committed to this repository. They are third-party data
# published as a release asset of an AGPLv3 project, so this repo carries their
# checksums and this script rather than a copy -- which keeps the fixture
# byte-exactly reproducible without redistributing the data. Checksums are facts
# about a file, not copies of one.
#
# The upstream repository is ARCHIVED: srsRAN became OCUDU in December 2025
# (https://gitlab.com/ocudu/ocudu). Tags and release assets remain, and the URL
# below pins an immutable release, but this can disappear -- which is why BR-2
# also requires a committed, always-run hand-derived floor case that needs no
# network at all.
#
#   ./fetch_srsran_vectors.sh [dest-dir]     default: ./srsran_vectors
set -euo pipefail

REL="release_25_10"
URL="https://github.com/srsran/srsRAN_Project/releases/download/${REL}/phy_testvectors.tar"
HDR="https://raw.githubusercontent.com/srsran/srsRAN_Project/${REL}/tests/unittests/phy/upper/channel_coding/ldpc/ldpc_encoder_test_data.h"
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-$HERE/srsran_vectors}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$DEST"

echo "== fetching $REL (250 MB; only the ~1 MB of LDPC sets is kept)"
# Stream and extract only the LDPC members, so the 250 MB never hits disk.
# -C is positional in tar: it must precede the member pattern, or it is applied
# after the pattern is processed and nothing lands in $WORK.
curl -sL "$URL" | tar -x -C "$WORK" --wildcards '*ldpc*'
INNER="$WORK/tests/unittests/phy/upper/channel_coding/ldpc"

echo "== verifying checksums"
( cd "$INNER" && sha256sum -c "$HERE/srsran_vectors.sha256" )

echo "== extracting .dat files to $DEST"
for f in "$INNER"/*.tar.gz; do tar -xzf "$f" -C "$DEST"; done

echo "== fetching the case table (ldpc_encoder_test_data.h)"
curl -sf "$HDR" -o "$HERE/ldpc_encoder_test_data.h"

echo
echo "done: $(find "$DEST" -name '*.dat' | wc -l) vector files in $DEST"
echo "next: python $HERE/golden_vectors_check.py $DEST"
