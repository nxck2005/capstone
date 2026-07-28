"""Shared network-free synthetic dataset fixtures."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image
from torchvision.datasets import CIFAR10, STL10, Imagenette

import config.params as config_params


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
        _path, sha256 = write_manifest(name, tmp_path)
        loaded["datasets"][name]["manifest_sha256"] = sha256

    yield tmp_path
    config_params.load_params.cache_clear()
