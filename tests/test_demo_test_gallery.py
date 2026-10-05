"""AM-101 demo test gallery: bundled images are guarded, authentic, rule-selected and bound to G-12 records."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import data.test_access as test_access
from demo_shared.test_gallery import FREEZE, GALLERY, PER_IMAGE_MANIFEST, ROOT, SPLIT_MANIFEST, TestGallery


def test_gallery_is_one_rule_selected_test_image_per_class() -> None:
    gallery = TestGallery()
    rows = list(gallery.images.values())
    assert sorted(r["class_index"] for r in rows) == list(range(10))
    assert all(r["split"] == "test" for r in rows)
    first = next(r for r in rows if r["label"] == "tench")
    assert first["stable_sample_id"] == min(i for i, label in gallery.test_labels.items() if label == 0)


def _copy(tmp_path: Path) -> Path:
    for path in (FREEZE, PER_IMAGE_MANIFEST, SPLIT_MANIFEST):
        target = tmp_path / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, target)
    shutil.copytree(GALLERY, tmp_path / GALLERY.relative_to(ROOT))
    return tmp_path


def test_a_swapped_image_is_refused(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    folder = root / GALLERY.relative_to(ROOT)
    shutil.copy(folder / "church.jpeg", folder / "tench.jpeg")
    with pytest.raises(RuntimeError, match="fails source, label or selection-rule checks"):
        TestGallery(root)


def test_an_image_outside_the_selection_rule_is_refused(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    path = root / GALLERY.relative_to(ROOT) / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["images"][0]["stable_sample_id"] = "f" * 16
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError):
        TestGallery(root)


def test_a_different_freeze_is_refused(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    path = root / FREEZE.relative_to(ROOT)
    freeze = json.loads(path.read_text())
    freeze["freeze_id"] = "g12freeze-other"
    path.write_text(json.dumps(freeze))
    with pytest.raises(RuntimeError, match="not exported under the committed G-12 freeze"):
        TestGallery(root)


def test_reading_the_gallery_goes_through_the_test_access_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    real = test_access.load_test_sample

    def spy(loader, stable_sample_id):
        calls.append(stable_sample_id)
        return real(loader, stable_sample_id)

    monkeypatch.setattr(test_access, "load_test_sample", spy)
    gallery = TestGallery()
    assert sorted(calls) == sorted(gallery.images)
