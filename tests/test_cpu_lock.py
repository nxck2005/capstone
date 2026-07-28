"""Network-free structural tests for the AM-73 CPU lock verifier."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import verify_cpu_lock
from config.params import REPO_ROOT


def _params() -> dict:
    return yaml.safe_load(
        (REPO_ROOT / "spec/params.generated.yaml").read_text(encoding="utf-8")
    )


def test_committed_cpu_lock_is_structurally_cuda_free():
    params = _params()
    lock = REPO_ROOT / params["environment"]["cpu_lock_file"]

    verify_cpu_lock.verify_structure(lock, params)


@pytest.mark.parametrize("name", ("cuda-toolkit", "nvidia-cublas", "triton"))
def test_verifier_rejects_forbidden_distribution_names(
    tmp_path: Path,
    name: str,
):
    params = _params()
    index = params["environment"]["torch_cpu_index_url"]
    lock = tmp_path / "requirements-cpu.lock"
    lock.write_text(
        f"--index-url {index}\n"
        f"torch=={params['environment']['torch_cpu']}\n"
        f"torchvision=={params['environment']['torchvision_cpu']}\n"
        f"{name}==1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="CUDA distributions"):
        verify_cpu_lock.verify_structure(lock, params)
