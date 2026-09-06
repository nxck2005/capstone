"""Shared network-free synthetic dataset fixtures."""

from __future__ import annotations

import hashlib
import pickle
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml
from PIL import Image
from torchvision.datasets import CIFAR10, STL10, Imagenette

import config.params as config_params
from baseline import g8_campaign, w8_spec_compatibility
from evaluation import g10_spec_compatibility
from evaluation.g10_protocol import verify_am94_boundary


REPO = Path(__file__).resolve().parents[1]
G10_COMPLETION = Path("results/learned/w9/w9a_completion.json")
G10_RECONCILIATION = Path("results/learned/w9/w9a_reconciliation.json")

# Keep this compatibility name empty for callers that imported the old
# inventory. Phase routing is deliberately node-specific now: a module is
# never excluded merely because one of its tests describes an older phase.
HISTORICAL_PRE_G10_TEST_MODULES = frozenset()
HISTORICAL_PRE_G10_TESTS = frozenset(
    {
        "tests/test_g10_protocol.py::test_am94_boundary_remains_pre_science",
        "tests/test_g10_protocol.py::test_no_outcome_files_exist_before_authority",
    }
)
POST_G10_CONTEXT_TESTS = frozenset(
    {
        "tests/test_w6_classical_evidence.py::test_resigned_malicious_index_fails_inner_or_reproduction",
        "tests/test_w6_classical_evidence.py::test_false_future_classification_is_rejected_even_when_resigned",
        "tests/test_w6_classical_evidence.py::test_current_index_and_matrix_are_deterministic",
    }
)
_POST_G10_CONTEXTS: dict[str, object] = {}


def _present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    del config
    if not (
        _present(REPO / G10_COMPLETION)
        or _present(REPO / G10_RECONCILIATION)
    ):
        return
    marker = pytest.mark.historical_pre_g10
    for item in items:
        relative = Path(item.fspath).resolve().relative_to(REPO).as_posix()
        if (
            relative in HISTORICAL_PRE_G10_TEST_MODULES
            or item.nodeid.split("[", 1)[0] in HISTORICAL_PRE_G10_TESTS
        ):
            item.add_marker(marker)


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Open the additive adapter only around exact historical test calls."""

    base_nodeid = item.nodeid.split("[", 1)[0]
    if base_nodeid not in POST_G10_CONTEXT_TESTS:
        return
    context = _post_g10_am94_context()
    context.__enter__()
    _POST_G10_CONTEXTS[item.nodeid] = context


def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> None:
    del nextitem
    context = _POST_G10_CONTEXTS.pop(item.nodeid, None)
    if context is not None:
        context.__exit__(None, None, None)


@contextmanager
def _post_g10_am94_context():
    """Adapt one historical test call to terminal AM-94 in memory only.

    Historical G8/W5/W6/W7/W8 verifiers intentionally authenticate their
    source image as pre-G-10. The command-line carrier adapts those calls in
    the same narrow way. Tests opt into this context explicitly; the strict
    ``g10_spec_compatibility.load`` implementation and all scientific files
    remain untouched, and every patch is restored on exit.
    """

    if not (
        _present(REPO / G10_COMPLETION)
        or _present(REPO / G10_RECONCILIATION)
    ):
        yield
        return

    verify_am94_boundary(REPO, outcomes_allowed=True)

    def additive_load(root: Path = REPO):
        return verify_am94_boundary(Path(root), outcomes_allowed=True)

    with ExitStack() as stack:
        stack.enter_context(patch.object(g10_spec_compatibility, "load", additive_load))
        stack.enter_context(
            patch.object(g8_campaign, "load_am94_spec_compatibility", additive_load)
        )
        stack.enter_context(
            patch.object(w8_spec_compatibility, "load_am94_spec_compatibility", additive_load)
        )
        # A few historical verifiers import the strict loader directly rather
        # than through one of the two package modules above.  Patch only those
        # already-loaded verifier aliases for the duration of an opted-in test;
        # never change the production compatibility implementation.
        for module_name in (
            "baseline.w6_evidence",
            "verify_w7_g4",
            "gen_w8_execution_authorization",
            "verify_w8_a",
        ):
            module = sys.modules.get(module_name)
            if module is not None and hasattr(module, "load_am94_spec_compatibility"):
                stack.enter_context(
                    patch.object(module, "load_am94_spec_compatibility", additive_load)
                )
        yield


@pytest.fixture
def post_g10_am94():
    """Opt-in, function-scoped AM-94 compatibility for one historical test."""

    with _post_g10_am94_context():
        yield


@pytest.fixture(scope="module")
def post_g10_am94_module():
    """Module-scoped form for module-scoped historical artifact fixtures."""

    with _post_g10_am94_context():
        yield


def _record(width: int, token: int) -> bytes:
    values = (np.arange(width, dtype=np.uint32) + token) % 256
    return values.astype(np.uint8).tobytes()


@pytest.fixture
def synthetic_dataset_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    params = yaml.safe_load(config_params.PARAMS_PATH.read_text(encoding="utf-8"))
    for config in params["datasets"].values():
        if isinstance(config, dict) and "loader" in config:
            config["train_images"] = 10
            config["val_images"] = 10
            config["test_images"] = 10
            config["manifest_sha256"] = "pending_manifest_materialization_at_W1"

    params_path = tmp_path / "params.yaml"
    params_path.write_text(
        yaml.safe_dump(params, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    monkeypatch.setattr(config_params, "PARAMS_PATH", params_path)
    config_params.load_params.cache_clear()

    archive_dir = tmp_path / params["datasets"]["archive_dir"]
    archive_dir.mkdir(parents=True)
    for name in ("imagenette160", "stl10", "cifar10"):
        config = params["datasets"][name]
        payload = f"synthetic archive for {name}".encode()
        (archive_dir / config["archive_filename"]).write_bytes(payload)
        config["archive_bytes"] = len(payload)
        config["archive_sha256"] = hashlib.sha256(payload).hexdigest()

    extracted = tmp_path / params["datasets"]["extracted_dir"]
    imagenette_root = extracted / "imagenette160" / "imagenette2-160"
    identifiers = sorted(Imagenette._WNID_TO_CLASS)
    for split, per_class in (("train", 2), ("val", 1)):
        split_offset = 0 if split == "train" else 101
        for label, identifier in enumerate(identifiers):
            class_dir = imagenette_root / split / identifier
            class_dir.mkdir(parents=True)
            for index in range(per_class):
                array = np.full(
                    (7, 9, 3),
                    (label * 17 + index * 3 + split_offset) % 256,
                    dtype=np.uint8,
                )
                array[0, 0] = (
                    label,
                    index,
                    (label + index + split_offset) % 256,
                )
                Image.fromarray(array).save(
                    class_dir / f"{label}-{index}.JPEG",
                    format="JPEG",
                    quality=91,
                )

    stl_root = extracted / "stl10" / STL10.base_folder
    stl_root.mkdir(parents=True)
    (stl_root / STL10.class_names_file).write_text(
        "\n".join(f"class-{index}" for index in range(10)) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    stl_width = int(params["datasets"]["stl10"]["n"])
    (stl_root / STL10.train_list[0][0]).write_bytes(
        b"".join(_record(stl_width, index + 1) for index in range(20))
    )
    (stl_root / STL10.train_list[1][0]).write_bytes(
        bytes((index % 10) + 1 for index in range(20))
    )
    (stl_root / STL10.test_list[0][0]).write_bytes(
        b"".join(_record(stl_width, index + 101) for index in range(10))
    )
    (stl_root / STL10.test_list[1][0]).write_bytes(
        bytes((index % 10) + 1 for index in range(10))
    )

    cifar_root = extracted / "cifar10" / CIFAR10.base_folder
    cifar_root.mkdir(parents=True)
    cifar_width = int(params["datasets"]["cifar10"]["n"])
    with (cifar_root / CIFAR10.meta["filename"]).open("wb") as stream:
        pickle.dump(
            {CIFAR10.meta["key"]: [f"class-{index}" for index in range(10)]},
            stream,
        )
    token = 201
    for filename, _md5 in CIFAR10.train_list:
        rows = np.stack([np.frombuffer(_record(cifar_width, token + i), dtype=np.uint8) for i in range(4)])
        labels = [((token + i) - 201) % 10 for i in range(4)]
        with (cifar_root / filename).open("wb") as stream:
            pickle.dump({"data": rows, "labels": labels}, stream)
        token += 4
    test_rows = np.stack(
        [
            np.frombuffer(_record(cifar_width, 301 + index), dtype=np.uint8)
            for index in range(10)
        ]
    )
    with (cifar_root / CIFAR10.test_list[0][0]).open("wb") as stream:
        pickle.dump(
            {"data": test_rows, "labels": list(range(10))},
            stream,
        )

    params_path.write_text(
        yaml.safe_dump(params, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    config_params.load_params.cache_clear()

    from data.manifests import write_manifest

    loaded = config_params.load_params()
    for name in ("imagenette160", "stl10", "cifar10"):
        root = extracted / name
        (root / ".archive-sha256").write_text(
            f"{params['datasets'][name]['archive_sha256']}\n",
            encoding="ascii",
            newline="\n",
        )
        _path, sha256 = write_manifest(name, tmp_path)
        loaded["datasets"][name]["manifest_sha256"] = sha256

    yield tmp_path
    config_params.load_params.cache_clear()


@pytest.fixture
def run_config_factory():
    """Build resolved learned configs for synthetic dataset/ratio unit cases."""

    from config.params import get
    from config.run_config import RunConfig, load_experiment

    first_train_seed = get("evaluation.train_seeds")[0]
    base = load_experiment(
        "configs/learned-headline.yaml",
        train_seed=first_train_seed,
        channel_seed=get("evaluation.channel_seeds")[0],
        test_snr_db=get("channel.test_snr_grid_db")[0],
    )

    def factory(
        dataset: str = "cifar10",
        ratio: str = "r_1_48",
        *,
        train_seed: int = first_train_seed,
        reconstruction_weight: float | None = None,
    ) -> RunConfig:
        value = base.to_dict()
        resolved = value["resolved"]
        resolved["dataset"] = dataset
        resolved["bw_ratio"] = ratio
        resolved["train_seed"] = train_seed
        resolved["dataset_version"] = get(
            f"datasets.{dataset}.{get('config.dataset_version_rule')}"
        )
        resolved["k"] = get(f"bandwidth.k_symbols.{dataset}.{ratio}")
        if reconstruction_weight is not None:
            resolved["lambda"] = reconstruction_weight
        return RunConfig.from_dict(value)

    return factory
