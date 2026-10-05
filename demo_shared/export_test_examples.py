#!/usr/bin/env python3
"""Export the demos' fixed test-split gallery once, through the guarded test loader (AM-101).

Selection is outcome-blind and fixed by rule: for each of the ten classes, the
test image whose stable sample ID sorts first in the committed split manifest.
The original source JPEG bytes are read through ``data.test_access`` (which
refuses unless the committed G-12 freeze manifest is present) and written to
demo_shared/test_examples/. Alongside each image the manifest records what
G-12 measured for it in seed cell 0 at r = 1/6, at every SNR, for DJSCC and
for adaptive JPEG 2000 + LDPC, copied from the hash-checked per-image streams
in results/per_image/. The demos compare their live CPU output against those
records; nothing here runs a model.

    .venv/bin/python demo_shared/export_test_examples.py
    .venv/bin/python demo_shared/export_test_examples.py --full-cache

``--full-cache`` instead writes every test image's source bytes to the ignored
demo_shared/.cache/test_split/ for the demos' "browse all" panel. The demos run
it themselves, in a child process, on first use; later starts read the cache
(about 1 s) instead of re-verifying the extracted dataset (about 4 s).
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.test_access import load_test_dataset  # noqa: E402

DATASET = "imagenette160"
OUT = ROOT / "demo_shared/test_examples"
FULL_CACHE = ROOT / "demo_shared/.cache/test_split"
SPLIT_MANIFEST = ROOT / "data/manifests/imagenette160.csv"
FREEZE = ROOT / "results/freeze_manifest.json"
PER_IMAGE_MANIFEST = ROOT / "results/per_image_manifest.csv"
CLASS_NAMES = (
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute",
)
# The two systems the live receivers run, in the G-12 cell the demo reproduces.
RECORDED = {"learned": "own_task_head", "classical_adaptive": "artifact_finetuned"}
CELL, RATIO = 0, "r_1_6"
SELECTION_RULE = "per class, the test row with the lexicographically smallest stable_sample_id in data/manifests/imagenette160.csv"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def select() -> dict[int, str]:
    """Class index → chosen stable ID, by SELECTION_RULE; reads IDs and labels only."""
    chosen: dict[int, str] = {}
    with SPLIT_MANIFEST.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test":
                label = int(row["label"])
                if label not in chosen or row["stable_sample_id"] < chosen[label]:
                    chosen[label] = row["stable_sample_id"]
    if sorted(chosen) != list(range(len(CLASS_NAMES))):
        raise SystemExit("the test split does not cover all ten classes")
    return chosen


def recorded(ids: set[str]) -> dict[str, dict[str, dict[str, dict]]]:
    """stable ID → system → SNR → the G-12 per-image record fields the demo shows."""
    out: dict[str, dict[str, dict[str, dict]]] = {i: {s: {} for s in RECORDED} for i in ids}
    files = 0
    with PER_IMAGE_MANIFEST.open(newline="", encoding="utf-8") as handle:
        for entry in csv.DictReader(handle):
            if (RECORDED.get(entry["system"]) != entry["classifier_variant"] or entry["bw_ratio"] != RATIO
                    or int(entry["train_seed"]) != CELL):
                continue
            raw = gzip.open(ROOT / "results/per_image" / entry["file"]).read()
            if _sha256(raw) != entry["rows_sha256"]:
                raise SystemExit(f"per-image stream differs from its manifest: {entry['file']}")
            files += 1
            for line in raw.splitlines():
                row = json.loads(line)
                if row["stable_sample_id"] in ids:
                    if row["split"] != "test" or row["run_id"] != entry["run_id"]:
                        raise SystemExit(f"unexpected per-image row in {entry['file']}")
                    out[row["stable_sample_id"]][entry["system"]][f"{float(entry['test_snr_db']):g}"] = {
                        "pred_label": row["pred_label"], "correct": bool(row["correct"]),
                        "outage": bool(row["outage"]), "outage_reason": row["outage_reason"],
                        "noise_id": row["noise_id"], "run_id": row["run_id"],
                    }
    if files != 2 * 21 or any(len(snrs) != 21 for systems in out.values() for snrs in systems.values()):
        raise SystemExit("per-image records do not cover both systems at all 21 SNRs")
    return out


def full_cache() -> int:
    """Every test image's source bytes, named by stable ID, plus an index bound to the freeze."""
    dataset = load_test_dataset(DATASET)  # refuses without the committed G-12 freeze
    FULL_CACHE.mkdir(parents=True, exist_ok=True)
    ids = []
    for i in range(len(dataset)):
        sample = dataset.source_sample(i)
        (FULL_CACHE / f"{sample.stable_sample_id}.jpeg").write_bytes(sample.source_bytes)
        ids.append(sample.stable_sample_id)
    index = {"freeze_id": json.loads(FREEZE.read_text(encoding="utf-8"))["freeze_id"],
             "source_manifest_sha256": _sha256(SPLIT_MANIFEST.read_bytes()), "stable_sample_ids": ids}
    (FULL_CACHE / "index.json").write_text(json.dumps(index) + "\n", encoding="utf-8")
    print(f"cached {len(ids)} test images in {FULL_CACHE.relative_to(ROOT)}")
    return 0


def main() -> int:
    if sys.argv[1:] == ["--full-cache"]:
        return full_cache()
    chosen = select()
    ids = set(chosen.values())
    dataset = load_test_dataset(DATASET)  # refuses without the committed G-12 freeze
    sources = {s.stable_sample_id: s for s in (dataset.source_sample(i) for i in range(len(dataset))) if s.stable_sample_id in ids}
    records = recorded(ids)

    OUT.mkdir(parents=True, exist_ok=True)
    images = []
    for label, stable_id in sorted(chosen.items()):
        sample = sources[stable_id]
        if sample.label != label:
            raise SystemExit(f"{stable_id}: loaded label {sample.label} differs from the manifest")
        slug = re.sub(r"[^a-z]+", "-", CLASS_NAMES[label].lower()).strip("-")
        (OUT / f"{slug}.jpeg").write_bytes(sample.source_bytes)
        images.append({
            "id": slug, "file": f"{slug}.jpeg", "split": "test", "stable_sample_id": stable_id,
            "sha256": _sha256(sample.source_bytes), "class_index": label, "label": CLASS_NAMES[label],
            "recorded": records[stable_id],
        })
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    manifest = {
        "dataset": DATASET, "split": "test", "amendment": "AM-101", "selection_rule": SELECTION_RULE,
        "source_manifest_sha256": _sha256(SPLIT_MANIFEST.read_bytes()),
        "freeze_id": freeze["freeze_id"],
        "per_image_manifest_sha256": _sha256(PER_IMAGE_MANIFEST.read_bytes()),
        "recorded_cell": {"train_seed": CELL, "channel_seed": CELL, "bw_ratio": RATIO, "scorers": RECORDED},
        "images": images,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(images)} test examples to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
