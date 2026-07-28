#!/usr/bin/env python3
"""Verify that the SR-21 CPU lock is genuinely CUDA-free (AM-73)."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
PARAMS = REPO / "spec" / "params.generated.yaml"
REQUIREMENT_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)")


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _forbidden(names: set[str]) -> list[str]:
    return sorted(
        name
        for name in names
        if name.startswith("cuda-")
        or name.startswith("nvidia-")
        or name == "triton"
        or name.startswith("triton-")
    )


def _locked_requirements(lock: Path) -> dict[str, str]:
    requirements: dict[str, str] = {}
    for line in lock.read_text(encoding="utf-8").splitlines():
        if match := REQUIREMENT_RE.match(line):
            requirements[_normalise(match.group(1))] = match.group(2)
    return requirements


def verify_structure(lock: Path, params: dict) -> None:
    body = lock.read_text(encoding="utf-8")
    requirements = _locked_requirements(lock)
    expected = {
        "torch": str(params["environment"]["torch_cpu"]),
        "torchvision": str(params["environment"]["torchvision_cpu"]),
    }
    for name, version in expected.items():
        if requirements.get(name) != version:
            raise RuntimeError(
                f"{lock.name} has {name}=={requirements.get(name)!r}, "
                f"expected {version!r}"
            )
    index = str(params["environment"]["torch_cpu_index_url"])
    if f"--index-url {index}" not in body and f"--extra-index-url {index}" not in body:
        raise RuntimeError(f"{lock.name} does not embed the official CPU index {index}")
    forbidden = _forbidden(set(requirements))
    if forbidden:
        raise RuntimeError(
            f"{lock.name} contains CUDA distributions: {', '.join(forbidden)}"
        )


def verify_clean_install(lock: Path, params: dict) -> None:
    with tempfile.TemporaryDirectory(prefix="capstone-cpu-lock-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True, clear=True).create(environment)
        python = environment / "bin" / "python"
        pip = environment / "bin" / "pip"
        subprocess.run(
            [
                str(pip),
                "install",
                "--require-hashes",
                "--disable-pip-version-check",
                "-r",
                str(lock),
            ],
            check=True,
        )
        probe = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import importlib.metadata,json,torch,torchvision;"
                    "print(json.dumps({"
                    "'cuda':torch.version.cuda,"
                    "'torch':torch.__version__,"
                    "'torchvision':torchvision.__version__,"
                    "'names':sorted(d.metadata['Name'] for d in "
                    "importlib.metadata.distributions())"
                    "}))"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        record = json.loads(probe.stdout)
    if record["cuda"] is not None:
        raise RuntimeError(
            f"clean CPU-lock install reports torch.version.cuda={record['cuda']!r}"
        )
    expected_torch = str(params["environment"]["torch_cpu"])
    expected_torchvision = str(params["environment"]["torchvision_cpu"])
    if record["torch"] != expected_torch or record["torchvision"] != expected_torchvision:
        raise RuntimeError(
            "clean install versions differ: "
            f"torch={record['torch']!r}, torchvision={record['torchvision']!r}"
        )
    forbidden = _forbidden({_normalise(name) for name in record["names"]})
    if forbidden:
        raise RuntimeError(
            f"clean CPU-lock install contains CUDA distributions: {', '.join(forbidden)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-install",
        action="store_true",
        help="also install into a temporary venv with plain pip --require-hashes",
    )
    args = parser.parse_args()
    params = yaml.safe_load(PARAMS.read_text(encoding="utf-8"))
    lock = REPO / params["environment"]["cpu_lock_file"]
    verify_structure(lock, params)
    if args.clean_install:
        verify_clean_install(lock, params)
    print(
        f"ok: {lock.name} is structurally CUDA-free"
        + (" and its clean install is CUDA-free" if args.clean_install else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
