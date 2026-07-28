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
