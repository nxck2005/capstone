"""Environment contract tests (SR-21, SR-12, SR-1).

The first test is the reason this file exists. Everything AM-23 warned about is
invisible without it: a bare resolve gives you the CPU build of torch, the import
succeeds, tensors work, and the failure surfaces weeks later as "training is
mysteriously slow" -- by which point the compute budget in `params.compute` is
fiction.

There is deliberately no skip marker, no `CAPSTONE_CPU_ONLY` env var and no
`pytest.importorskip` guarding it. An escape hatch would be exported once in a
shell profile and would then silently disarm the only check that catches a CPU
build on the machine that actually trains. The CPU-only install path
(`requirements-cpu.lock`, SR-21) exists for analysis and demo and is not expected
to run this suite -- see AGENTS.md.
"""

from __future__ import annotations

import hashlib
import platform

import pytest

import env
from config.params import REPO_ROOT, get


def test_cuda_build():
    """`params.environment.cuda_assertion` -- the AM-23 trap.

    A successful `import torch` is NOT this check.
    """
    import torch

    assert torch.version.cuda is not None, (
        f"CPU BUILD: torch {torch.__version__} has no CUDA. It was resolved from "
        "PyPI rather than the cu130 index. See requirements.in."
    )


def test_assert_cuda_helper_agrees():
    """The helper the training entrypoints call must agree with the raw check."""
    env.assert_cuda()


def test_cuda_assertion_matches_params():
    """The mirrored expression in `src/env.py` still matches the spec.

    `env.CUDA_ASSERTION` is code rather than an `eval` of the parameter, so this
    is what keeps the two honest.
    """
    assert env.CUDA_ASSERTION == get("environment.cuda_assertion")


def test_torch_version_matches_params():
    """Catches a resolve that quietly took a different build or version."""
    import torch

    assert torch.__version__ == get("environment.torch")


def test_torchvision_version_matches_params():
    import torchvision

    assert torchvision.__version__ == get("environment.torchvision")


def test_python_version_matches_params():
    assert platform.python_version() == get("environment.python_version")


def test_cuda_is_available():
    """A CUDA build with no visible device would still pass `test_cuda_build`."""
    import torch

    assert torch.cuda.is_available()


# --- determinism (SR-12) -----------------------------------------------------


def test_deterministic_backend_applied():
    import torch

    env.set_deterministic_backend()
    settings = get("environment.deterministic_backend")
    assert torch.backends.cudnn.deterministic == settings["cudnn_deterministic"]
    assert torch.backends.cudnn.benchmark == settings["cudnn_benchmark"]


def test_unknown_determinism_key_raises(monkeypatch):
    """A setting added to the spec must not be silently unapplied.

    SR-12's promise rests on these being set; a parameter the code skips would
    leave the promise resting on nothing while every test still passed.
    """
    monkeypatch.setattr(env, "get", lambda path: {"cudnn_allow_tf32": True})
    with pytest.raises(NotImplementedError, match="cudnn_allow_tf32"):
        env.set_deterministic_backend()


def test_determinism_values_must_be_boolean(monkeypatch):
    monkeypatch.setattr(
        env,
        "get",
        lambda path: {"cudnn_deterministic": "yes"},
    )
    with pytest.raises(TypeError, match="must be boolean"):
        env.set_deterministic_backend()


# --- run metadata (SR-21, SR-13) ---------------------------------------------


def test_environment_record_keys_match_params():
    """Set equality, both directions: no missing field, no extra field."""
    record = env.environment_record()
    assert set(record) == set(get("environment.record_in_run_metadata"))
    assert record["openjpeg_version"] == get("environment.openjpeg")


def test_environment_record_is_fully_populated():
    """Every field resolves to something on the primary device.

    `device_name` and `driver_version` return None off-GPU by design, so this
    asserts the GPU path actually produced them rather than falling through.
    """
    record = env.environment_record()
    empty = [k for k, v in record.items() if v is None or v == ""]
    assert not empty, f"unpopulated run-metadata fields: {empty}"


def test_unknown_metadata_field_raises(monkeypatch):
    monkeypatch.setattr(env, "get", lambda path: ["python_version", "phase_of_moon"])
    with pytest.raises(NotImplementedError, match="phase_of_moon"):
        env.environment_record()


def test_lock_file_hash_matches():
    """The recorded hash is of the lockfile actually in the checkout.

    SR-21's reproducibility claim is only as good as this: a run whose metadata
    names a lockfile hash that no longer exists cannot be rebuilt.
    """
    lock = REPO_ROOT / get("environment.lock_file")
    assert lock.is_file(), f"{lock} is missing"
    expected = hashlib.sha256(lock.read_bytes()).hexdigest()
    assert env.environment_record()["lock_file_sha256"] == expected


def test_cpu_lock_file_exists():
    """The CPU-only install path is a MUST, not an aspiration (SR-21, AM-66)."""
    cpu_lock = REPO_ROOT / get("environment.cpu_lock_file")
    assert cpu_lock.is_file(), f"{cpu_lock} is missing"


def test_openjpeg_may_be_unavailable_for_non_j2k_metadata(monkeypatch):
    def unavailable():
        raise ImportError("fixture has no OpenJPEG")

    monkeypatch.setattr(env, "_query_openjpeg_version", unavailable)

    assert env.loaded_openjpeg_version(required=False) is None


def test_openjpeg_mismatch_fails_even_when_metadata_is_optional(monkeypatch):
    monkeypatch.setattr(env, "_query_openjpeg_version", lambda: "9.9.9")

    with pytest.raises(RuntimeError, match="version mismatch"):
        env.loaded_openjpeg_version(required=False)


def test_j2k_preflight_fails_before_artifact_creation(tmp_path, monkeypatch):
    artifact = tmp_path / "results" / "j2k-output.jp2"

    def unavailable():
        raise OSError("fixture loader unavailable")

    monkeypatch.setattr(env, "_query_openjpeg_version", unavailable)

    with pytest.raises(RuntimeError, match="required.*unavailable"):
        env.assert_j2k_runtime()
    assert not artifact.parent.exists()
