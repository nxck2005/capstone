"""Environment assertion, determinism and the run-metadata record (SR-21, SR-12).

Three jobs, and the first one is the reason this module exists at all:

  * `assert_cuda()` -- catch the AM-23 trap. A bare `pip install torch` resolves
    to the CPU build, and it fails *silently*: the import succeeds, the tensors
    work, and you find out weeks later when a training run is thirty times slower
    than the compute budget assumed. `torch.version.cuda is not None` is the only
    check that sees it. `tests/test_env.py::test_cuda_build` runs it with no skip
    marker and no env-var escape hatch, on purpose.

  * `set_deterministic_backend()` -- apply `params.environment.deterministic_backend`.
    SR-12 promises the same seed and config reproduce reported metrics within
    0.5pp on the pinned environment; cuDNN autotuning breaks that promise.

  * `environment_record()` -- emit exactly `params.environment.record_in_run_metadata`
    for SR-13 to attach to every result row. Without it, "reproducible from a
    clean checkout" is a claim with nothing behind it.

Every one of the three is driven off `params.environment` rather than off literals
here (SR-1), and each raises on a key it does not recognise rather than skipping
it -- a parameter added to the spec and silently ignored by the code is the exact
failure mode this project keeps finding in its own audits.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from collections.abc import Mapping
from typing import Any, Callable

from config.params import REPO_ROOT, get

# The assertion expression named by `params.environment.cuda_assertion`, mirrored
# here as code rather than `eval`-ed from the spec. `test_cuda_assertion_matches_params`
# fails if the two ever drift, which keeps the parameter honest without executing
# a string from a config file.
CUDA_ASSERTION = "torch.version.cuda is not None"


def assert_cuda() -> None:
    """Fail unless torch is a CUDA build (`params.environment.cuda_assertion`).

    A successful `import torch` is NOT this check (AM-23).
    """
    import torch

    if torch.version.cuda is None:
        raise RuntimeError(
            "CPU BUILD: torch.version.cuda is None. "
            f"torch {torch.__version__} was resolved from PyPI rather than "
            f"{get('environment.torch_index_url')}. Recompile the lockfile with "
            "`--index-strategy unsafe-best-match` and re-sync; see requirements.in."
        )


def set_deterministic_backend() -> None:
    """Apply every key in `params.environment.deterministic_backend`.

    Only the two audited mappings are supported. Adding a similarly named torch
    attribute is not enough: a new result-affecting setting needs an explicit
    implementation and test.
    """
    import torch

    settings = get("environment.deterministic_backend")
    if not isinstance(settings, Mapping):
        raise TypeError("params.environment.deterministic_backend must be a mapping")
    handlers = {
        "cudnn_deterministic": (torch.backends.cudnn, "deterministic"),
        "cudnn_benchmark": (torch.backends.cudnn, "benchmark"),
    }
    for key, value in settings.items():
        if key not in handlers:
            raise NotImplementedError(
                f"params.environment.deterministic_backend.{key} has no handler in "
                "src/env.py. Add one rather than letting the setting go unapplied."
            )
        if not isinstance(value, bool):
            raise TypeError(
                f"params.environment.deterministic_backend.{key} must be boolean"
            )
        owner, attribute = handlers[key]
        setattr(owner, attribute, value)
        if getattr(owner, attribute) is not value:
            raise RuntimeError(
                f"failed to apply params.environment.deterministic_backend.{key}"
            )


def _query_openjpeg_version() -> str:
    from glymur.version import openjpeg_version

    return str(openjpeg_version)


def loaded_openjpeg_version(*, required: bool = False) -> str | None:
    """Return the verified loaded OpenJPEG version, or ``None`` when optional."""

    expected = str(get("environment.openjpeg"))
    try:
        loaded = _query_openjpeg_version()
    except (ImportError, OSError):
        if required:
            raise RuntimeError(
                f"OpenJPEG {expected} is required for this JPEG 2000 path "
                "but is unavailable"
            ) from None
        return None
    if loaded != expected:
        raise RuntimeError(
            f"OpenJPEG version mismatch: loaded {loaded!r}, expected {expected!r}"
        )
    return loaded


def assert_j2k_runtime() -> None:
    """Fail before a JPEG 2000 path creates result directories or artifacts."""

    loaded_openjpeg_version(required=True)


# --- run metadata ------------------------------------------------------------


def _python_version() -> str:
    return platform.python_version()


def _torch_version() -> str:
    import torch

    return torch.__version__


def _cuda_version() -> str | None:
    import torch

    return torch.version.cuda


def _device_name() -> str | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(0).name


def _driver_version() -> str | None:
    """The NVIDIA *display driver* version.

    Deliberately not `torch.version.cuda`, which is the toolkit the wheel was
    built against, and deliberately not `torch._C._cuda_getDriverVersion()`,
    which returns the CUDA driver API version (e.g. 13000). Neither identifies
    the driver actually installed, which is what a reproduction attempt needs.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,  # literal-ok: subprocess safety timeout, not an experiment setting
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip().splitlines()[0].strip() or None


def _lock_file_sha256() -> str:
    path = REPO_ROOT / get("environment.lock_file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _openjpeg_version() -> str | None:
    return loaded_openjpeg_version(required=False)


_PROVIDERS: dict[str, Callable[[], Any]] = {
    "python_version": _python_version,
    "torch_version": _torch_version,
    "cuda_version": _cuda_version,
    "driver_version": _driver_version,
    "device_name": _device_name,
    "lock_file_sha256": _lock_file_sha256,
    "openjpeg_version": _openjpeg_version,
}


def environment_record() -> dict[str, Any]:
    """Exactly the fields in `params.environment.record_in_run_metadata`.

    Built by iterating the parameter list rather than by returning a hand-written
    dict, so a field added to the spec with no provider here fails loudly instead
    of being quietly absent from every result row.
    """
    fields = get("environment.record_in_run_metadata")
    missing = [f for f in fields if f not in _PROVIDERS]
    if missing:
        raise NotImplementedError(
            f"params.environment.record_in_run_metadata names {missing}, which "
            "src/env.py cannot provide. Add a provider to _PROVIDERS."
        )
    return {f: _PROVIDERS[f]() for f in fields}


def profile_environment_record(
    execution_profile_id: str,
    *,
    device: str,
    config_hash: str,
    require_openjpeg: bool = False,
    allow_pending_qualification: bool = False,
) -> dict[str, Any]:
    """Fail-closed metadata for new profile-aware scientific execution.

    ``environment_record`` above remains the historical local-profile surface
    used by completed artifacts.  New science calls this function and therefore
    authenticates exact Torch/CUDA/package/lock/GPU identity rather than merely
    checking that Torch has some CUDA build.
    """

    from config.execution_profiles import authenticate_execution_profile

    return authenticate_execution_profile(
        execution_profile_id,
        device=device,
        config_hash=config_hash,
        require_openjpeg=require_openjpeg,
        allow_pending_qualification=allow_pending_qualification,
    )
