"""The demos' test-split gallery and its recorded G-12 outcomes (AM-101).

Two sources, both read through the guarded test loader (``data.test_access``,
which refuses without the G-12 freeze manifest):

- **Featured:** the ten images in demo_shared/test_examples/, exported once by
  export_test_examples.py, one per class by a fixed outcome-blind rule, with
  their G-12 records bundled. Checked at start-up against the committed split,
  freeze and per-image manifests; this is what a portable laptop always has.
- **Full split (optional):** all 3,925 test images, loaded on first use from the
  extracted dataset with ``load_test_dataset``, and their G-12 records read from
  the hash-checked per-image streams in results/per_image/ (published as the
  ``g12-test-per-image-2026-10-02`` release). Where either is missing,
  browsing reports itself unavailable; the featured ten still work.

``transmit`` runs the frozen r = 1/6 models on one image (via the original
demo backend's verified ``LiveLearned`` and ``LiveClassical``) and sets each
live outcome beside what G-12 recorded for that image in seed cell 0. The
demo noise keys equal G-12's, so the two should agree; any disagreement is
reported, never hidden. These are display outputs, not reported metrics.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GALLERY = ROOT / "demo_shared/test_examples"
FULL_CACHE = ROOT / "demo_shared/.cache/test_split"
SPLIT_MANIFEST = ROOT / "data/manifests/imagenette160.csv"
FREEZE = ROOT / "results/freeze_manifest.json"
PER_IMAGE_MANIFEST = ROOT / "results/per_image_manifest.csv"
CLASS_NAMES = (
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute",
)
SYSTEMS = ("learned", "classical_adaptive")
OUTAGES = {"decode_failure", "codec_infeasibility", "structural_infeasibility"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TestGallery:
    def __init__(self, root: Path = ROOT) -> None:
        from data.identity import stable_sample_id
        from data.test_access import load_test_sample

        gallery = root / GALLERY.relative_to(ROOT)
        manifest = json.loads((gallery / "manifest.json").read_text(encoding="utf-8"))
        freeze = json.loads((root / FREEZE.relative_to(ROOT)).read_text(encoding="utf-8"))
        split_bytes = (root / SPLIT_MANIFEST.relative_to(ROOT)).read_bytes()
        if (manifest.get("dataset"), manifest.get("split")) != ("imagenette160", "test"):
            raise RuntimeError("test gallery manifest is not an Imagenette-160 test manifest")
        if manifest["source_manifest_sha256"] != _sha256(split_bytes):
            raise RuntimeError("test gallery disagrees with the committed split manifest")
        if manifest["freeze_id"] != freeze["freeze_id"]:
            raise RuntimeError("test gallery was not exported under the committed G-12 freeze")
        if manifest["per_image_manifest_sha256"] != _sha256((root / PER_IMAGE_MANIFEST.relative_to(ROOT)).read_bytes()):
            raise RuntimeError("test gallery records disagree with the committed per-image manifest")

        rows = list(csv.DictReader(io.StringIO(split_bytes.decode("utf-8"))))
        test_labels = {r["stable_sample_id"]: int(r["label"]) for r in rows if r["split"] == "test"}
        # Re-derive the outcome-blind selection: first stable ID per class.
        expected: dict[int, str] = {}
        for stable_id, label in test_labels.items():
            if label not in expected or stable_id < expected[label]:
                expected[label] = stable_id

        self.images: dict[str, dict[str, Any]] = {}
        for row in manifest["images"]:
            slug, stable_id = row["id"], row["stable_sample_id"]
            if not isinstance(slug, str) or not slug.replace("-", "").isalpha() or row["file"] != f"{slug}.jpeg":
                raise RuntimeError("test gallery has an unsafe file name")
            path = gallery / row["file"]
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"test gallery image {row['file']} is missing or unsafe")
            source = load_test_sample(lambda _sid, path=path: path.read_bytes(), stable_id)  # guarded read
            if (_sha256(source) != row["sha256"] or stable_sample_id(source) != stable_id
                    or test_labels.get(stable_id) != row["class_index"] or expected.get(row["class_index"]) != stable_id
                    or row["label"] != CLASS_NAMES[row["class_index"]]):
                raise RuntimeError(f"test gallery image {slug} fails source, label or selection-rule checks")
            if any(len(row["recorded"][system]) != 21 for system in SYSTEMS):
                raise RuntimeError(f"test gallery image {slug} lacks a recorded result at every SNR")
            self.images[stable_id] = {**row, "source": source}
        if sorted(r["class_index"] for r in self.images.values()) != list(range(10)):
            raise RuntimeError("test gallery must hold exactly one image per class")
        self.root = root
        self.test_labels = test_labels
        self.test_ids = [r["stable_sample_id"] for r in rows if r["split"] == "test"]  # manifest order
        self._png: dict[str, bytes] = {}
        self._full: dict[str, Path] | None = None
        self._full_records: dict[str, dict[float, dict[str, tuple[int, bool, bool, str]]]] | None = None
        self._full_error: str | None = None
        self._lock = threading.Lock()

    def by_slug(self, slug: str) -> str | None:
        return next((i for i, row in self.images.items() if row["id"] == slug), None)

    def full_split(self) -> str | None:
        """Make the whole test split browsable; return a reason if it is unavailable.

        The first call fills the ignored cache in a child process (see
        export_test_examples.py --full-cache) and loads the seed-cell-0 records.
        Images are then read one at a time, through the guard, when chosen.
        """
        with self._lock:
            if self._full is not None or self._full_error is not None:
                return self._full_error
            try:
                cache = self.root / FULL_CACHE.relative_to(ROOT)
                if not self._cache_ok(cache):
                    subprocess.run([sys.executable, str(self.root / "demo_shared/export_test_examples.py"), "--full-cache"],
                                   check=True, capture_output=True, timeout=600)
                    if not self._cache_ok(cache):
                        raise RuntimeError("the test-split cache is incomplete after export")
                self._full_records = _full_records(self.root)
                self._full = {stable_id: cache / f"{stable_id}.jpeg" for stable_id in self.test_ids}
            except Exception as exc:  # the featured ten stay available
                self._full_error = (f"Browsing the full test split needs the extracted Imagenette-160 dataset and the "
                                    f"G-12 per-image records ({type(exc).__name__}).")
            return self._full_error

    def _cache_ok(self, cache: Path) -> bool:
        try:
            index = json.loads((cache / "index.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        freeze = json.loads((self.root / FREEZE.relative_to(ROOT)).read_text(encoding="utf-8"))
        return (index.get("freeze_id") == freeze["freeze_id"] and index.get("stable_sample_ids") == self.test_ids
                and all((cache / f"{i}.jpeg").is_file() for i in self.test_ids))

    def known(self, image_id: str) -> bool:
        return image_id in self.images or (self._full is not None and image_id in self._full)

    def label(self, image_id: str) -> int:
        return self.test_labels[image_id]

    def product(self, image_id: str) -> Any:
        from data.preprocessing import canonicalize_source

        if image_id in self.images:
            return canonicalize_source(self.images[image_id]["source"], "imagenette160")
        from data.identity import stable_sample_id
        from data.test_access import load_test_sample

        path = self._full[image_id]  # type: ignore[index]
        source = load_test_sample(lambda _sid: path.read_bytes(), image_id)  # guarded read
        if stable_sample_id(source) != image_id:
            raise RuntimeError(f"cached test image {image_id} fails its identity check")
        return canonicalize_source(source, "imagenette160")

    def png(self, image_id: str) -> bytes:
        if image_id in self._png:
            return self._png[image_id]
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(self.product(image_id).canonical_image, mode="RGB").save(buffer, format="PNG")
        if image_id in self.images or len(self._png) < 512:  # bound the cache while browsing
            self._png[image_id] = buffer.getvalue()
        return buffer.getvalue()

    def recorded(self, image_id: str, system: str, snr_db: float) -> dict[str, Any]:
        if image_id in self.images:
            record = self.images[image_id]["recorded"][system][f"{float(snr_db):g}"]
            pred, correct, outage, noise_id = record["pred_label"], record["correct"], record["outage"], record["noise_id"]
        else:
            pred, correct, outage, noise_id = self._full_records[system][float(snr_db)][image_id]  # type: ignore[index]
        return {"predicted_label": CLASS_NAMES[pred], "correct": correct, "outage": outage, "noise_id": noise_id}


def _full_records(root: Path) -> dict[str, dict[float, dict[str, tuple[int, bool, bool, str]]]]:
    """Seed-cell-0, r = 1/6 G-12 records for every test image, from the hash-checked streams."""
    import gzip

    scorers = {"learned": "own_task_head", "classical_adaptive": "artifact_finetuned"}
    out: dict[str, dict[float, dict[str, tuple[int, bool, bool, str]]]] = {system: {} for system in scorers}
    with (root / PER_IMAGE_MANIFEST.relative_to(ROOT)).open(newline="", encoding="utf-8") as handle:
        for entry in csv.DictReader(handle):
            if scorers.get(entry["system"]) != entry["classifier_variant"] or entry["bw_ratio"] != "r_1_6" or entry["train_seed"] != "0":
                continue
            raw = gzip.open(root / "results/per_image" / entry["file"]).read()
            if _sha256(raw) != entry["rows_sha256"]:
                raise RuntimeError(f"per-image stream differs from its manifest: {entry['file']}")
            rows = (json.loads(line) for line in raw.splitlines())
            out[entry["system"]][float(entry["test_snr_db"])] = {
                r["stable_sample_id"]: (r["pred_label"], bool(r["correct"]), bool(r["outage"]), r["noise_id"]) for r in rows}
    if any(len(snrs) != 21 or any(len(ids) != 3925 for ids in snrs.values()) for snrs in out.values()):
        raise RuntimeError("per-image records do not cover every test image at every SNR")
    return out


def _with_record(arm: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Attach the G-12 record and whether the live outcome reproduces it."""
    if arm["status"] == "unavailable":
        return {**arm, "recorded": {**record, "matches": None}}
    live_outage = arm["status"] in OUTAGES
    matches = (arm["predicted_label"] == record["predicted_label"] and live_outage == record["outage"]
               and arm.get("noise_id") == record["noise_id"])
    return {**arm, "recorded": {**record, "matches": matches}}


def transmit(evidence: Any, gallery: TestGallery, image_id: str, snr_db: int) -> dict[str, Any]:
    """Both live arms for one gallery image at r = 1/6, each beside its G-12 record."""
    product = gallery.product(image_id)
    label = gallery.label(image_id)

    learned: dict[str, Any] = {"status": "unavailable", "predicted_label": None, "confidence": None, "image_url": None,
                               "detail": "Frozen learned checkpoint is absent or fails SHA-256."}
    if evidence.live.available:
        try:
            live = evidence.live.infer(product, snr_db)
            learned = {"status": "delivered", "predicted_label": CLASS_NAMES[live["label_index"]],
                       "confidence": live["confidence"], "image_url": live["reconstruction_png_data_url"],
                       "detail": "Fresh CPU inference · uncalibrated softmax; reconstruction is visualization only.",
                       "noise_id": live["noise_id"], "checkpoint_sha256": live["checkpoint_sha256"]}
        except Exception as exc:  # fail closed: no substitute prediction
            learned["detail"] = f"Learned inference failed closed ({type(exc).__name__}); no prediction was produced."

    classical: dict[str, Any] = {"status": "unavailable", "predicted_label": None, "confidence": None, "image_url": None,
                                 "detail": "Frozen artifact scorer or OpenJPEG 2.5.4 is unavailable."}
    if evidence.classical.available:
        try:
            classical = dict(evidence.classical.infer(product, label=label, snr_db=snr_db))
        except Exception as exc:
            classical["detail"] = f"Classical inference failed closed ({type(exc).__name__}); no prediction was produced."

    return {
        "learned": _with_record(learned, gallery.recorded(image_id, "learned", snr_db)),
        "classical": _with_record(classical, gallery.recorded(image_id, "classical_adaptive", snr_db)),
    }


def data_url_bytes(url: str | None) -> bytes | None:
    return base64.b64decode(url.split(",", 1)[1]) if url else None
