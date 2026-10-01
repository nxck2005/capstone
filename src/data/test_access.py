"""The sole guarded boundary through which a test sample may be loaded (SR-22)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from config.params import REPO_ROOT, get


class TestAccessError(RuntimeError):
    """Raised before any loader call when the G-12 freeze is absent or invalid."""


Sample = TypeVar("Sample")


def _freeze_manifest_path() -> Path:
    return REPO_ROOT / get("artifacts.freeze_manifest_file")


def _validated_freeze_manifest() -> Mapping[str, Any]:
    path = _freeze_manifest_path()
    if not path.is_file():
        raise TestAccessError(
            f"test access remains sealed until {path} exists "
            f"at params.evaluation.test_access_gate={get('evaluation.test_access_gate')}"
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TestAccessError(f"invalid test freeze manifest {path}: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise TestAccessError(f"test freeze manifest must contain an object: {path}")

    required = get("evaluation.freeze_manifest_covers")
    missing = [
        field
        for field in required
        if field not in manifest or manifest[field] in (None, "", [], {})
    ]
    if missing:
        raise TestAccessError(
            f"test freeze manifest {path} does not cover required fields: {missing}"
        )
    return manifest


def load_test_sample(
    loader: Callable[[str], Sample],
    stable_sample_id: str,
) -> Sample:
    """Validate the committed freeze before invoking a synthetic or real loader."""

    _validated_freeze_manifest()
    return loader(stable_sample_id)


def _committed_freeze_manifest() -> Mapping[str, Any]:
    """The freeze manifest must be complete *and* committed exactly as on disk."""

    import subprocess  # noqa: PLC0415

    manifest = _validated_freeze_manifest()
    relative = str(get("artifacts.freeze_manifest_file"))
    try:
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TestAccessError(f"test freeze manifest {relative} is not committed at HEAD") from exc
    if committed != _freeze_manifest_path().read_bytes():
        raise TestAccessError(f"test freeze manifest {relative} differs from its committed bytes")
    return manifest


def load_test_dataset(dataset: str) -> Any:
    """The model-facing test split, released only after the committed G-12 freeze.

    This is the single model-facing test loader (SR-22).  It reuses the
    registry's manifest, provenance and label checks and differs only in reading
    the manifest's ``test`` rows from the published partition they come from.
    """

    _committed_freeze_manifest()
    from data.adapters import _adapter  # noqa: PLC0415
    from data.manifests import manifest_path, manifest_sha256, validate_manifest_bytes  # noqa: PLC0415
    from data.provenance import dataset_root, verify_extracted_dataset  # noqa: PLC0415
    from data.registry import CanonicalDataset, DatasetRegistryError, _verify_manifest_pin  # noqa: PLC0415

    verify_extracted_dataset(dataset, REPO_ROOT)
    rows = validate_manifest_bytes(dataset, manifest_path(dataset, REPO_ROOT).read_bytes())
    _verify_manifest_pin(dataset, manifest_sha256(dataset, REPO_ROOT))
    adapter = _adapter(dataset, dataset_root(dataset, REPO_ROOT))
    samples = {sample.stable_sample_id: sample for sample in adapter.iter_source_samples("test")}
    chosen = []
    for row in rows:
        if row.split != "test":
            continue
        sample = samples.get(row.stable_sample_id)
        if sample is None or sample.label != row.label:
            raise DatasetRegistryError(f"{dataset}: test manifest row {row.stable_sample_id} differs from the published source")
        chosen.append(sample)
    expected = int(get(f"datasets.{dataset}.test_images"))
    if len(chosen) != expected or len({sample.stable_sample_id for sample in chosen}) != expected:
        raise DatasetRegistryError(f"{dataset}/test: loaded {len(chosen)} unique samples, expected {expected}")
    return CanonicalDataset(dataset, "test", tuple(chosen))
