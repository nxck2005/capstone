"""G8_B BLER execution engine and bounded-smoke transaction.

The runner is deliberately downstream of the B3 resume authority.  It never
enumerates a new grid, and it refuses full-strength execution until the
campaign is in G8_C.  The only execution permitted while this module is being
completed is the small, explicitly labelled CPU smoke used to exercise the
same transaction and physical-layer path.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import secrets
import stat
import time
import ctypes
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from baseline import g8_bler_contract as bler_contract
from baseline import g8_bler_resume as resume
from baseline import g8_bler_work_units as work_units
from baseline.g8_campaign import canonical_json, rendered_json, sha256_bytes
from baseline.g8_bler_resume import AuthenticatedResumeContext
from config.params import REPO_ROOT, get


PHASE = "G8_B"
CHECKPOINT = "B4"
RUNNER_CONTRACT_SCHEMA_VERSION = 1
RUNNER_CONTRACT_ARTIFACT_ROLE = "g8_bler_runner_contract"
RUNNER_CONTRACT_ID_PREFIX = "g8runner"
RUNNER_CONTRACT_REPO_RELATIVE_PATH = "results/baseline/g8/bler_runner_contract.json"
RUNNER_CONTRACT_SOURCE_ROLE = "g8b_b4_runner_contract_source"
RUNNER_CONTRACT_SOURCE_PATHS = (
    "src/baseline/g8_bler_runner.py",
    "tools/run_g8_bler.py",
    "tools/gen_g8_bler_runner_contract.py",
    "tools/verify_g8_bler_runner_contract.py",
    "tools/verify_g8_bounded_smoke.py",
)
DEFAULT_RUNNER_CONTRACT_PATH = REPO_ROOT / RUNNER_CONTRACT_REPO_RELATIVE_PATH

EXECUTION_CLASS_FULL_STRENGTH = bler_contract.EXECUTION_CLASS_FULL_STRENGTH
EXECUTION_CLASS_BOUNDED_SMOKE = bler_contract.EXECUTION_CLASS_BOUNDED_SMOKE
BOUNDED_SMOKE_LABEL = bler_contract.BOUNDED_SMOKE_LABEL
BOUNDED_SMOKE_MAX_WORK_UNITS = bler_contract.BOUNDED_SMOKE_MAX_WORK_UNITS
BOUNDED_SMOKE_MAX_TRIALS = bler_contract.BOUNDED_SMOKE_MAX_TRIALS_PER_UNIT

SMOKE_RECORD_SCHEMA_VERSION = 1
SMOKE_RECORD_ARTIFACT_ROLE = "g8_bounded_smoke_record"

_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_STAGING_SUFFIX = ".staging"
_RENAME_NOREPLACE = 1


class G8BlerRunnerError(RuntimeError):
    """Base class for fail-closed runner errors."""


class RunnerAuthorizationError(G8BlerRunnerError):
    """The requested execution class is not authorized by campaign state."""


class RunnerConflictError(G8BlerRunnerError):
    """An immutable request/result publication conflicts with existing bytes."""


class RunnerPublicationError(G8BlerRunnerError):
    """An immutable artifact could not be published or verified exactly."""


class RunnerExecutionError(G8BlerRunnerError):
    """A physical-layer attempt failed and was recorded as failed evidence."""


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise G8BlerRunnerError(f"{label} must be a positive integer")
    return value


def _require_absolute_root(root: Path | str | None) -> Path:
    if root is None:
        raise RunnerAuthorizationError("runner requires an explicit absolute runtime root")
    path = Path(root)
    if not path.is_absolute():
        raise RunnerAuthorizationError("runner runtime root must be absolute")
    return path


def runner_contract_identifier(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("contract_id", None)
    return f"{RUNNER_CONTRACT_ID_PREFIX}-{sha256_bytes(canonical_json(body))}"


def _source_bindings() -> list[dict[str, Any]]:
    bindings = []
    for relative in RUNNER_CONTRACT_SOURCE_PATHS:
        try:
            body = (REPO_ROOT / relative).read_bytes()
        except OSError as exc:
            raise G8BlerRunnerError(f"cannot read runner contract source {relative}: {exc}") from exc
        bindings.append(
            {
                "path": relative,
                "role": RUNNER_CONTRACT_SOURCE_ROLE,
                "bytes": len(body),
                "sha256": sha256_bytes(body),
            }
        )
    return bindings


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RunnerAuthorizationError(f"cannot read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RunnerAuthorizationError(f"{label} is not a JSON object")
    return payload, raw


class AuthenticatedRunnerContext:
    """Authenticate B1C, B2C, B3, and the candidate/registered B4 contract once."""

    __slots__ = (
        "_resume_context",
        "_runner_contract_path",
        "_runner_contract",
        "_runner_binding",
        "_dependency_binding",
    )

    def __init__(
        self,
        resume_context: AuthenticatedResumeContext | None = None,
        *,
        runner_contract_path: Path | str | None = None,
        require_runner_contract: bool = True,
        require_registered_runner_contract: bool = False,
    ) -> None:
        if resume_context is None:
            resume_context = AuthenticatedResumeContext(require_resume_contract=True)
        if not isinstance(resume_context, AuthenticatedResumeContext):
            raise TypeError("resume_context must be an AuthenticatedResumeContext")
        resume_context.require_resume_contract_binding()
        self._resume_context = resume_context
        contract_path = (
            DEFAULT_RUNNER_CONTRACT_PATH
            if runner_contract_path is None
            else Path(runner_contract_path)
        )
        self._runner_contract_path = contract_path
        registered = self._registered_runner_binding(resume_context.campaign_state_path)
        if registered is None and require_registered_runner_contract:
            raise RunnerAuthorizationError("campaign state does not register the B4 runner contract")
        if not contract_path.exists():
            if require_runner_contract:
                raise RunnerAuthorizationError(f"B4 runner contract is missing: {contract_path}")
            self._runner_contract = None
            self._runner_binding = None
        else:
            payload, raw = self._authenticate_runner_contract(contract_path, registered)
            if registered is not None and (
                len(raw) != registered["bytes"] or sha256_bytes(raw) != registered["sha256"]
            ):
                raise RunnerAuthorizationError("registered B4 runner contract bytes changed")
            self._runner_contract = {
                "contract_id": payload["contract_id"],
                "schema_version": payload["schema_version"],
                "artifact_role": payload["artifact_role"],
                "phase": payload["phase"],
                "checkpoint": payload["checkpoint"],
            }
            self._runner_binding = {
                "bler_runner_contract_id": payload["contract_id"],
                "bler_runner_contract_sha256": sha256_bytes(raw),
            }
        self._dependency_binding = self._authenticate_dependencies()

    @staticmethod
    def _registered_runner_binding(state_path: Path) -> dict[str, Any] | None:
        state, _raw = _read_json(state_path, "campaign state")
        artifacts = state.get("identity", {}).get("produced_artifacts")
        if not isinstance(artifacts, list):
            raise RunnerAuthorizationError("campaign state produced-artifact list is malformed")
        matches = [
            dict(entry)
            for entry in artifacts
            if isinstance(entry, Mapping)
            and entry.get("path") == RUNNER_CONTRACT_REPO_RELATIVE_PATH
        ]
        if not matches:
            return None
        if len(matches) != 1 or set(matches[0]) != {"path", "sha256", "bytes"}:
            raise RunnerAuthorizationError("campaign state has an invalid B4 runner binding")
        if not isinstance(matches[0]["bytes"], int) or isinstance(matches[0]["bytes"], bool):
            raise RunnerAuthorizationError("B4 runner binding byte count is malformed")
        return matches[0]

    @staticmethod
    def _authenticate_runner_contract(
        path: Path,
        registered: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], bytes]:
        payload, raw = _read_json(path, "B4 runner contract")
        if raw != rendered_json(payload):
            raise RunnerAuthorizationError("B4 runner contract is not canonical rendered JSON")
        if payload.get("schema_version") != RUNNER_CONTRACT_SCHEMA_VERSION:
            raise RunnerAuthorizationError("B4 runner contract schema version changed")
        if payload.get("artifact_role") != RUNNER_CONTRACT_ARTIFACT_ROLE:
            raise RunnerAuthorizationError("B4 runner contract artifact role changed")
        if payload.get("phase") != PHASE or payload.get("checkpoint") != CHECKPOINT:
            raise RunnerAuthorizationError("B4 runner contract phase/checkpoint changed")
        contract_id = payload.get("contract_id")
        if not isinstance(contract_id, str) or contract_id != runner_contract_identifier(payload):
            raise RunnerAuthorizationError("B4 runner contract ID does not reproduce")
        sources = payload.get("contract_sources")
        if not isinstance(sources, list) or [entry.get("path") for entry in sources] != list(
            RUNNER_CONTRACT_SOURCE_PATHS
        ):
            raise RunnerAuthorizationError("B4 runner source path list changed")
        for entry in sources:
            if not isinstance(entry, Mapping) or entry.get("path") == RUNNER_CONTRACT_REPO_RELATIVE_PATH:
                raise RunnerAuthorizationError("B4 runner contract binds its own output path")
            body = (REPO_ROOT / entry["path"]).read_bytes()
            if (
                entry.get("role") != RUNNER_CONTRACT_SOURCE_ROLE
                or entry.get("bytes") != len(body)
                or entry.get("sha256") != sha256_bytes(body)
            ):
                raise RunnerAuthorizationError(f"B4 runner source changed: {entry.get('path')}")
        if sha256_bytes(raw).encode("ascii") in raw:
            raise RunnerAuthorizationError("B4 runner contract binds its own SHA-256")
        if registered is not None and (
            registered.get("path") != RUNNER_CONTRACT_REPO_RELATIVE_PATH
            or registered.get("sha256") != sha256_bytes(raw)
            or registered.get("bytes") != len(raw)
        ):
            raise RunnerAuthorizationError("registered B4 runner binding does not match bytes")
        return payload, raw

    @staticmethod
    def _authenticate_dependencies() -> dict[str, Any]:
        import sionna
        import torch

        if not bler_contract.installed_rng_version_matches():
            raise RunnerAuthorizationError(
                f"NumPy {np.__version__} != frozen {bler_contract.RNG_LIBRARY_VERSION}"
            )
        if str(sionna.__version__) != str(get("baseline.ldpc_impl_version")):
            raise RunnerAuthorizationError("installed Sionna version does not match params")
        if torch.version.cuda is None:
            raise RunnerAuthorizationError("runner requires the configured CUDA torch build")
        return {
            "numpy_version": str(np.__version__),
            "sionna_version": str(sionna.__version__),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "torch_device_available": bool(torch.cuda.is_available()),
        }

    @property
    def resume_context(self) -> AuthenticatedResumeContext:
        return self._resume_context

    @property
    def campaign_state_path(self) -> Path:
        return self._resume_context.campaign_state_path

    @property
    def runner_contract_path(self) -> Path:
        return self._runner_contract_path

    def runner_contract_binding(self) -> dict[str, str]:
        if self._runner_binding is None:
            raise RunnerAuthorizationError("B4 runner contract is not authenticated")
        return dict(self._runner_binding)

    def dependency_binding(self) -> dict[str, Any]:
        return dict(self._dependency_binding)

    def authority_binding(self) -> dict[str, Any]:
        return self._resume_context.authority_binding()

    def ordered_work_unit_ids(self) -> tuple[str, ...]:
        return self._resume_context.ordered_work_unit_ids

    def work_unit_record(self, work_unit_id: str) -> dict[str, Any]:
        return self._resume_context.work_unit_record(work_unit_id)


def _root_is_production_alias(root: Path) -> bool:
    production = Path(work_units.DEFAULT_WORK_UNIT_ROOT)
    if os.path.normpath(os.fspath(root)) == os.path.normpath(os.fspath(production)):
        return True
    try:
        return Path(os.path.realpath(root)) == Path(os.path.realpath(production))
    except OSError:
        return False


def authorize_execution(
    context: AuthenticatedRunnerContext,
    execution_class: str,
    *,
    root: Path | str | None,
    require_fresh_root: bool = True,
) -> Path:
    """Authorize before any runtime-root, adapter, bit, or decoder operation."""

    root_path = _require_absolute_root(root)
    state, _raw = _read_json(context.campaign_state_path, "campaign state")
    identity = state.get("identity", {})
    phase = identity.get("phase")
    stage = identity.get("stage")
    if execution_class == EXECUTION_CLASS_BOUNDED_SMOKE:
        if phase != "G8_B" or stage != "tooling_open":
            raise RunnerAuthorizationError("bounded smoke requires G8_B/tooling_open")
        if _root_is_production_alias(root_path):
            raise RunnerAuthorizationError("bounded smoke may not use the production root or an alias")
        if require_fresh_root and root_path.exists():
            raise RunnerAuthorizationError(
                "bounded smoke requires a fresh isolated root; recoverable evidence must be repaired first"
            )
        if not require_fresh_root and not root_path.is_dir():
            raise RunnerAuthorizationError("prepared bounded-smoke root is not an existing directory")
        return root_path
    if execution_class == EXECUTION_CLASS_FULL_STRENGTH:
        if phase != "G8_C" or stage != "characterization_open":
            raise RunnerAuthorizationError(
                "full-strength execution is not authorized before G8_C/characterization_open"
            )
        return root_path
    raise RunnerAuthorizationError(f"unknown execution class {execution_class!r}")


def _ensure_root(root: Path) -> None:
    if not root.parent.is_dir():
        raise RunnerAuthorizationError(
            f"runtime-root parent must already exist: {root.parent}"
        )
    with resume.reconciliation_lock(
        root,
        mode=resume.LOCK_MODE_EXCLUSIVE,
        create_missing_root=True,
    ):
        pass


def _open_bucket(root: Path, bucket: str) -> tuple[int, int]:
    root_fd = os.open(root, _DIRECTORY_FLAGS)
    try:
        try:
            os.mkdir(bucket, 0o700, dir_fd=root_fd)
        except FileExistsError:
            pass
        entry = os.lstat(bucket, dir_fd=root_fd)
        if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
            raise RunnerPublicationError(f"runtime bucket is not a real directory: {bucket}")
        bucket_fd = os.open(bucket, _DIRECTORY_FLAGS, dir_fd=root_fd)
        return root_fd, bucket_fd
    except Exception:
        os.close(root_fd)
        raise


def _read_installed(bucket_fd: int, final_name: str) -> bytes | None:
    try:
        entry = os.lstat(final_name, dir_fd=bucket_fd)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode) or entry.st_nlink != 1:
        raise RunnerConflictError(f"immutable artifact target is an alias or non-regular object: {final_name}")
    descriptor = os.open(final_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=bucket_fd)
    try:
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            return stream.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _publish_without_replace(bucket_fd: int, staging: str, final_name: str) -> None:
    """Use Linux ``renameat2(RENAME_NOREPLACE)`` without a fallback.

    Unlike a hard-link publication, the successful rename leaves no second
    hard link behind.  Thus a hard exit after publication leaves an ordinary
    one-link immutable file that B3 can validate on the next scan.
    """

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RunnerPublicationError(
            "crash-durable no-replace publication is unavailable; refusing fallback"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        bucket_fd,
        os.fsencode(staging),
        bucket_fd,
        os.fsencode(final_name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(errno.EEXIST, os.strerror(error_number), final_name)
    if error_number in {errno.ENOSYS, errno.EPERM, errno.EOPNOTSUPP, errno.EXDEV}:
        raise RunnerPublicationError(
            "crash-durable no-replace publication is unavailable; refusing fallback"
        ) from OSError(error_number, os.strerror(error_number))
    raise RunnerPublicationError(
        f"descriptor-relative no-replace publication failed at {final_name}: "
        f"{os.strerror(error_number)}"
    ) from OSError(error_number, os.strerror(error_number))


def _publish_immutable_json(path: Path, payload: Mapping[str, Any], *, root: Path) -> str:
    """Publish exact canonical request/result bytes without replacing a target."""

    body = canonical_json(dict(payload))
    digest = sha256_bytes(body)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise RunnerPublicationError("immutable artifact path is outside the runtime root") from exc
    if len(relative.parts) != 2:
        raise RunnerPublicationError("immutable artifact path has the wrong bucket layout")
    bucket, final_name = relative.parts
    root_fd, bucket_fd = _open_bucket(root, bucket)
    staging = f".{final_name}.{os.getpid()}.{secrets.token_hex(12)}{_STAGING_SUFFIX}"
    try:
        existing = _read_installed(bucket_fd, final_name)
        if existing is not None:
            if existing == body:
                return digest
            raise RunnerConflictError(f"immutable artifact conflicts at {path}")
        descriptor: int | None = None
        stream = None
        try:
            descriptor = os.open(
                staging,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                _FILE_MODE,
                dir_fd=bucket_fd,
            )
            stream = os.fdopen(descriptor, "wb")
            descriptor = None
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        except Exception as exc:
            raise RunnerPublicationError(f"cannot stage immutable artifact {path}: {exc}") from exc
        finally:
            if stream is not None:
                stream.close()
            if descriptor is not None:
                os.close(descriptor)
        published = False
        try:
            _publish_without_replace(bucket_fd, staging, final_name)
            published = True
            staging = None
        except FileExistsError as exc:
            installed = _read_installed(bucket_fd, final_name)
            if installed == body:
                return digest
            raise RunnerConflictError(f"immutable artifact publication conflicts at {path}") from exc
        except (NotImplementedError, OSError) as exc:
            if isinstance(exc, OSError) and exc.errno not in {
                errno.EPERM,
                errno.ENOSYS,
                errno.EOPNOTSUPP,
                errno.EXDEV,
            }:
                raise RunnerPublicationError(f"immutable artifact publication failed at {path}: {exc}") from exc
            raise RunnerPublicationError(
                "crash-durable immutable publication is unavailable; refusing fallback"
            ) from exc
        try:
            os.fsync(bucket_fd)
            os.fsync(root_fd)
        except OSError as exc:
            installed = _read_installed(bucket_fd, final_name)
            if installed == body:
                return digest
            raise RunnerPublicationError(
                f"directory durability failed and exact publication was not proven: {path}: {exc}"
            ) from exc
        installed = _read_installed(bucket_fd, final_name)
        if installed != body:
            raise RunnerPublicationError(f"installed immutable artifact bytes differ at {path}")
        return digest
    finally:
        if staging is not None:
            try:
                os.unlink(staging, dir_fd=bucket_fd)
            except FileNotFoundError:
                pass
            except OSError:
                # A published final file is authoritative; an orphan staging file
                # is explicitly handled by B3 census and never interpreted as data.
                pass
        os.close(bucket_fd)
        os.close(root_fd)


def _batch_ranges(total: int, batch_size: int) -> Iterator[tuple[int, int]]:
    start = 0
    while start < total:
        count = min(batch_size, total - start)
        yield start, count
        start += count


def _execute_measurement(
    request: Mapping[str, Any],
    *,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    """Execute one request from trial zero with bounded memory."""

    _require_positive_int(batch_size, "batch_size")
    from baseline.ldpc.adapter import SionnaLDPCAdapter
    from baseline.ldpc.modulation import map_bits, max_log_llr, n0_from_esn0_db, bits_per_symbol

    request = bler_contract.validate_work_unit_request(request)
    identity = request["bler_identity"]
    k, n = (int(value) for value in identity["k_and_n"])
    q_m = bits_per_symbol(identity["modulation"])
    if n % q_m:
        raise RunnerExecutionError("codeword length is not divisible by modulation order")
    trials_requested = int(request["trials_requested"])
    info_seed = int(request["stream_seeds"][bler_contract.PURPOSE_INFORMATION_BITS]["seed_uint64"])
    real_seed = int(request["stream_seeds"][bler_contract.PURPOSE_AWGN_REAL]["seed_uint64"])
    imag_seed = int(request["stream_seeds"][bler_contract.PURPOSE_AWGN_IMAG]["seed_uint64"])
    n0 = n0_from_esn0_db(float(request["snr_db"]))
    noise_scale = math.sqrt(n0 / 2.0)
    adapter = SionnaLDPCAdapter(k, n, q_m, int(identity["base_graph"]), device=device)
    if adapter.lifting_size != int(identity["lifting_size"]):
        raise RunnerExecutionError("LDPC adapter lifting size differs from authenticated identity")
    real_rng = np.random.Generator(np.random.Philox(key=real_seed))
    imag_rng = np.random.Generator(np.random.Philox(key=imag_seed))
    bit_errors = 0
    block_errors = 0
    trials_completed = 0
    symbols_per_trial = n // q_m
    try:
        for start, count in _batch_ranges(trials_requested, batch_size):
            information = bler_contract.information_bit_stream(
                info_seed,
                start * k,
                count * k,
            ).reshape(count, k)
            encoded = np.asarray(adapter.encode(information), dtype=np.uint8)
            if encoded.shape != (count, n) or np.any((encoded != 0) & (encoded != 1)):
                raise RunnerExecutionError("LDPC encoder returned an incompatible or nonbinary array")
            symbols = map_bits(encoded, identity["modulation"])
            if symbols.shape != (count, symbols_per_trial):
                raise RunnerExecutionError("modulator returned an incompatible symbol shape")
            real = real_rng.standard_normal((count, symbols_per_trial))
            imag = imag_rng.standard_normal((count, symbols_per_trial))
            received = symbols + noise_scale * (real + 1j * imag)
            llr = np.asarray(max_log_llr(received, identity["modulation"], n0), dtype=np.float32)
            if llr.shape != (count, n) or not np.isfinite(llr).all():
                raise RunnerExecutionError("demapper returned nonfinite or incompatible LLRs")
            decoded = np.asarray(adapter.decode(llr), dtype=np.uint8)
            if decoded.shape != (count, k) or np.any((decoded != 0) & (decoded != 1)):
                raise RunnerExecutionError("LDPC decoder returned an incompatible or nonbinary array")
            differences = np.not_equal(decoded, information)
            bit_errors += int(np.count_nonzero(differences))
            block_errors += int(np.count_nonzero(np.any(differences, axis=1)))
            trials_completed += count
    except RunnerExecutionError:
        raise
    except Exception as exc:
        raise RunnerExecutionError(f"physical-layer execution failed: {exc}") from exc
    return {
        "status": bler_contract.STATUS_COMPLETE,
        "trials_completed": trials_completed,
        "bit_errors": bit_errors,
        "block_errors": block_errors,
    }


def _failed_measurement(error: Exception, *, trials_completed: int, bit_errors: int, block_errors: int) -> dict[str, Any]:
    return {
        "status": bler_contract.STATUS_FAILED,
        "trials_completed": trials_completed,
        "bit_errors": bit_errors,
        "block_errors": block_errors,
        "error": str(error),
    }


def _select_smoke_ids(context: AuthenticatedRunnerContext, max_units: int) -> list[str]:
    configured = list(get("baseline.modulations"))
    selected: list[str] = []
    for modulation in configured:
        for work_unit_id in context.ordered_work_unit_ids():
            record = context.work_unit_record(work_unit_id)
            if record["identity"]["modulation"] == modulation:
                selected.append(work_unit_id)
                break
        else:
            raise RunnerAuthorizationError(f"no required work unit exists for modulation {modulation}")
    return selected[: min(max_units, BOUNDED_SMOKE_MAX_WORK_UNITS)]


def _state_claim(
    context: AuthenticatedRunnerContext,
    work_unit_id: str,
    plan: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[dict[str, Any], str, int]:
    record = next(item for item in plan["assigned_unit_records"] if item["work_unit_id"] == work_unit_id)
    classification = record["classification"]
    if classification not in resume.REMAINING_CLASSIFICATIONS:
        raise RunnerAuthorizationError(
            f"work unit {work_unit_id} is {classification}, not remaining work"
        )
    attempt = int(record["proposed_attempt"])
    shard_plan = work_units.build_shard_plan(
        context.resume_context.state_context,
        plan["shard_count"],
        plan["shard_index"],
    )
    claim = work_units.build_unit_state(
        context.resume_context.state_context,
        work_unit_id,
        shard_plan,
        attempt=attempt,
        status=work_units.STATUS_CLAIMED,
    )
    target = resume.state_path(context.resume_context, work_unit_id, root=root)
    if record["state_sha256"] is None:
        claim_sha = work_units.create_unit_state_exclusive(
            context.resume_context.state_context,
            claim,
            root=root,
            path=target,
        )
    else:
        claim_sha = work_units.replace_unit_state(
            context.resume_context.state_context,
            target,
            claim,
            record["state_sha256"],
            root=root,
        )
    return claim, claim_sha, attempt


def run_one_unit(
    context: AuthenticatedRunnerContext,
    *,
    execution_class: str,
    root: Path | str,
    work_unit_id: str,
    shard_count: int,
    shard_index: int,
    batch_size: int,
    device: str,
    _root_prepared: bool = False,
) -> dict[str, Any]:
    """Run exactly one unit through claim/request/execute/result/link."""

    root_path = _require_absolute_root(root)
    _require_positive_int(batch_size, "batch_size")
    _require_positive_int(shard_count, "shard_count")
    if shard_index < 0 or shard_index >= shard_count:
        raise RunnerAuthorizationError("shard_index must be within shard_count")
    if execution_class not in {
        EXECUTION_CLASS_FULL_STRENGTH,
        EXECUTION_CLASS_BOUNDED_SMOKE,
    }:
        raise RunnerAuthorizationError(f"unknown execution class {execution_class!r}")

    # The gate is deliberately enforced at the per-unit boundary.  It is
    # reached before root creation, state publication, adapter construction or
    # any random stream is touched for an unauthorized full-strength call.
    if execution_class == EXECUTION_CLASS_BOUNDED_SMOKE and _root_prepared:
        if _root_is_production_alias(root_path) or not root_path.is_dir():
            raise RunnerAuthorizationError("prepared bounded-smoke root is not an isolated directory")
        authorize_execution(
            context,
            execution_class,
            root=root_path,
            require_fresh_root=False,
        )
    else:
        authorize_execution(context, execution_class, root=root_path)
        if execution_class == EXECUTION_CLASS_BOUNDED_SMOKE:
            _ensure_root(root_path)

    plan = resume.build_resume_plan(
        context.resume_context,
        root=root_path,
        shard_count=shard_count,
        shard_index=shard_index,
        scan_mode=(
            resume.SCAN_MODE_BOUNDED_SMOKE_INSPECTION
            if execution_class == EXECUTION_CLASS_BOUNDED_SMOKE
            else resume.SCAN_MODE_PRODUCTION_MERGE
        ),
    )
    if work_unit_id not in plan["assigned_work_unit_ids"]:
        raise RunnerAuthorizationError(f"work unit {work_unit_id} is not assigned to this shard")
    if work_unit_id not in plan["remaining_work_unit_ids"]:
        raise RunnerAuthorizationError(
            f"work unit {work_unit_id} is not remaining work in the validated B3 plan"
        )

    with resume.reconciliation_lock(root_path, mode=resume.LOCK_MODE_SHARED) as lease:
        # A runner's full transaction is shared-global then B2C per-unit.  The
        # state primitive owns the latter lock and its compare-and-swap.
        lease._assert_usable(root_path, resume.LOCK_MODE_SHARED)
        claim, claim_sha, attempt = _state_claim(context, work_unit_id, plan, root=root_path)
        request = (
            bler_contract.build_full_strength_request(work_unit_id)
            if execution_class == EXECUTION_CLASS_FULL_STRENGTH
            else bler_contract.build_bounded_smoke_request(
                work_unit_id=work_unit_id,
                bler_identity=context.work_unit_record(work_unit_id)["identity"],
                snr_db=context.work_unit_record(work_unit_id)["snr_db"],
                source_packet_config_ids=context.work_unit_record(work_unit_id)["source_packet_config_ids"],
                trials_requested=BOUNDED_SMOKE_MAX_TRIALS,
            )
        )
        request_path = resume.request_path(context.resume_context, work_unit_id, attempt, root=root_path)
        request_sha = _publish_immutable_json(request_path, request, root=root_path)
        try:
            measurement = _execute_measurement(request, device=device, batch_size=batch_size)
        except Exception as exc:
            measurement = _failed_measurement(exc, trials_completed=0, bit_errors=0, block_errors=0)
        result = bler_contract.build_work_unit_result(
            request=request,
            status=measurement["status"],
            trials_completed=measurement["trials_completed"],
            bit_errors=measurement["bit_errors"],
            block_errors=measurement["block_errors"],
            execution_metadata={
                "wall_time_s": None,
                "hostname": None,
                "device": device,
                "shard_index": shard_index,
                "shard_count": shard_count,
                "attempt": attempt,
            },
        )
        result_path = resume.result_path(context.resume_context, work_unit_id, attempt, root=root_path)
        result_sha = _publish_immutable_json(result_path, result, root=root_path)
        if measurement["status"] == bler_contract.STATUS_COMPLETE:
            proposed = work_units.build_unit_state(
                context.resume_context.state_context,
                work_unit_id,
                work_units.build_shard_plan(
                    context.resume_context.state_context, shard_count, shard_index
                ),
                attempt=attempt,
                status=work_units.STATUS_RESULT_LINKED,
                request_sha256=request_sha,
                result_path=resume.logical_result_path(context.resume_context, work_unit_id, attempt),
                result_sha256=result_sha,
                scientific_execution_performed=True,
                trials_completed=measurement["trials_completed"],
            )
        else:
            proposed = work_units.build_unit_state(
                context.resume_context.state_context,
                work_unit_id,
                work_units.build_shard_plan(
                    context.resume_context.state_context, shard_count, shard_index
                ),
                attempt=attempt,
                status=work_units.STATUS_FAILED,
                request_sha256=request_sha,
                scientific_execution_performed=True,
                trials_completed=measurement["trials_completed"],
            )
        state_path = resume.state_path(context.resume_context, work_unit_id, root=root_path)
        state_sha = work_units.replace_unit_state(
            context.resume_context.state_context,
            state_path,
            proposed,
            claim_sha,
            root=root_path,
        )
        # Validate the exact current chain before releasing the shared lease.
        request_record = resume.validate_request_file(
            context.resume_context,
            work_unit_id,
            attempt,
            root=root_path,
            require_full_strength=execution_class == EXECUTION_CLASS_FULL_STRENGTH,
        )
        result_record = resume.validate_result_file(
            context.resume_context,
            work_unit_id,
            attempt,
            root=root_path,
            request_record=request_record,
            shard_index=shard_index,
            shard_count=shard_count,
            scan_mode=(
                resume.SCAN_MODE_BOUNDED_SMOKE_INSPECTION
                if execution_class == EXECUTION_CLASS_BOUNDED_SMOKE
                else resume.SCAN_MODE_PRODUCTION_MERGE
            ),
        )
        installed_state = work_units.read_unit_state(
            context.resume_context.state_context,
            state_path,
            root=root_path,
        )
        return {
            "work_unit_id": work_unit_id,
            "attempt": attempt,
            "request": request_record,
            "result": result_record,
            "request_sha256": request_sha,
            "result_sha256": result_sha,
            "state_sha256": state_sha,
            "state": installed_state,
            "measurement": measurement,
        }


def build_bounded_smoke_record(
    context: AuthenticatedRunnerContext,
    outcomes: Sequence[Mapping[str, Any]],
    *,
    shard_count: int,
    shard_index: int,
    batch_size: int,
    production_root_used: bool,
    temporary_root_removed: bool,
) -> dict[str, Any]:
    """Build the deterministic tracked smoke record from validated outcomes."""

    runner_binding = context.runner_contract_binding()
    authority = context.authority_binding()
    selected = []
    for outcome in outcomes:
        request = outcome["request"]["request"]
        result = outcome["result"]["result"]
        identity = context.work_unit_record(outcome["work_unit_id"])
        selected.append(
            {
                "work_unit_id": outcome["work_unit_id"],
                "identity_sha256": sha256_bytes(canonical_json(identity)),
                "seed_records": request["stream_seeds"],
                "request_sha256": outcome["request_sha256"],
                "result_sha256": outcome["result_sha256"],
                "terminal_state_sha256": outcome["state_sha256"],
                "trials_requested": request["trials_requested"],
                "trials_completed": result["measurement"]["trials_completed"],
                "information_bits": result["measurement"]["information_bits"],
                "bit_errors": result["measurement"]["bit_errors"],
                "block_errors": result["measurement"]["block_errors"],
                "ber": result["measurement"]["ber"],
                "bler": result["measurement"]["bler"],
                "wilson_low": result["measurement"]["bler_confidence_low"],
                "wilson_high": result["measurement"]["bler_confidence_high"],
                "classification": resume.CLASSIFICATION_TERMINAL_NONMERGEABLE,
                "required_coverage_contribution": 0,
                "test_split_access": 0,
            }
        )
    return {
        "schema_version": SMOKE_RECORD_SCHEMA_VERSION,
        "artifact_role": SMOKE_RECORD_ARTIFACT_ROLE,
        "label": BOUNDED_SMOKE_LABEL,
        "campaign_id": authority["campaign_id"],
        "campaign_manifest_sha256": authority["campaign_manifest_sha256"],
        "required_bler_artifact_sha256": authority["required_bler_artifact_sha256"],
        "selection_policy_sha256": authority["selection_policy_sha256"],
        "bler_tooling_contract_id": authority["bler_tooling_contract_id"],
        "bler_tooling_contract_sha256": authority["bler_tooling_contract_sha256"],
        "bler_state_contract_id": context.resume_context.state_contract_binding()["bler_state_contract_id"],
        "bler_state_contract_sha256": context.resume_context.state_contract_binding()["bler_state_contract_sha256"],
        "bler_resume_contract_id": context.resume_context.resume_contract_binding()["bler_resume_contract_id"],
        "bler_resume_contract_sha256": context.resume_context.resume_contract_binding()["bler_resume_contract_sha256"],
        "bler_runner_contract_id": runner_binding["bler_runner_contract_id"],
        "bler_runner_contract_sha256": runner_binding["bler_runner_contract_sha256"],
        "execution_class": EXECUTION_CLASS_BOUNDED_SMOKE,
        "selection_rule": bler_contract.BOUNDED_SMOKE_SELECTION_RULE,
        "maximum_work_units": BOUNDED_SMOKE_MAX_WORK_UNITS,
        "maximum_trials_per_unit": BOUNDED_SMOKE_MAX_TRIALS,
        "selected_work_units": selected,
        "shard_count": shard_count,
        "shard_index": shard_index,
        "batch_size": batch_size,
        "non_scientific": True,
        "merge_eligible": False,
        "required_coverage_contribution": 0,
        "test_split_access": 0,
        "production_root_used": production_root_used,
        "temporary_root_removed": temporary_root_removed,
        "characterization_started": False,
        "scientific_execution_performed": False,
    }


def run_bounded_smoke(
    context: AuthenticatedRunnerContext,
    *,
    root: Path | str,
    device: str = "cpu",
    shard_count: int = 1,
    shard_index: int = 0,
    batch_size: int = 1,
    max_units: int = BOUNDED_SMOKE_MAX_WORK_UNITS,
    repair_recoverable: bool = False,
) -> tuple[list[dict[str, Any]], Path]:
    """Execute the actual bounded CPU smoke; caller owns final root removal."""

    root_path = authorize_execution(context, EXECUTION_CLASS_BOUNDED_SMOKE, root=root)
    _require_positive_int(batch_size, "batch_size")
    _require_positive_int(max_units, "max_units")
    if device != "cpu":
        raise RunnerAuthorizationError("G8_B bounded smoke is CPU-only")
    _require_positive_int(shard_count, "shard_count")
    if shard_index < 0 or shard_index >= shard_count:
        raise RunnerAuthorizationError("shard_index must be within shard_count")
    _ensure_root(root_path)
    scan_mode = resume.SCAN_MODE_BOUNDED_SMOKE_INSPECTION
    if repair_recoverable:
        resume.inspect_runtime_root(
            context.resume_context,
            root=root_path,
            scan_mode=scan_mode,
            repair_mode=resume.REPAIR_MODE_REPAIR_RECOVERABLE,
        )
    selected = _select_smoke_ids(context, max_units)
    outcomes = []
    for work_unit_id in selected:
        outcomes.append(
            run_one_unit(
                context,
                execution_class=EXECUTION_CLASS_BOUNDED_SMOKE,
                root=root_path,
                work_unit_id=work_unit_id,
                shard_count=shard_count,
                shard_index=shard_index,
                batch_size=batch_size,
                device=device,
                _root_prepared=True,
            )
        )
    inspected = resume.inspect_runtime_root(
        context.resume_context,
        root=root_path,
        scan_mode=scan_mode,
        repair_mode=resume.REPAIR_MODE_READ_ONLY,
    )
    for record in inspected["classifications"]:
        if record["work_unit_id"] in selected:
            if record["classification"] != resume.CLASSIFICATION_TERMINAL_NONMERGEABLE:
                raise RunnerExecutionError(
                    f"bounded smoke unit {record['work_unit_id']} did not settle as terminal_nonmergeable"
                )
            if record["required_coverage_contribution"] != 0:
                raise RunnerExecutionError("bounded smoke contributed required coverage")
    return outcomes, root_path


__all__ = [
    "AuthenticatedRunnerContext",
    "BOUNDED_SMOKE_LABEL",
    "BOUNDED_SMOKE_MAX_TRIALS",
    "BOUNDED_SMOKE_MAX_WORK_UNITS",
    "CHECKPOINT",
    "DEFAULT_RUNNER_CONTRACT_PATH",
    "EXECUTION_CLASS_BOUNDED_SMOKE",
    "EXECUTION_CLASS_FULL_STRENGTH",
    "G8BlerRunnerError",
    "RunnerAuthorizationError",
    "RunnerConflictError",
    "RunnerExecutionError",
    "RunnerPublicationError",
    "RUNNER_CONTRACT_ARTIFACT_ROLE",
    "RUNNER_CONTRACT_ID_PREFIX",
    "RUNNER_CONTRACT_REPO_RELATIVE_PATH",
    "RUNNER_CONTRACT_SCHEMA_VERSION",
    "RUNNER_CONTRACT_SOURCE_PATHS",
    "RUNNER_CONTRACT_SOURCE_ROLE",
    "SMOKE_RECORD_ARTIFACT_ROLE",
    "SMOKE_RECORD_SCHEMA_VERSION",
    "authorize_execution",
    "build_bounded_smoke_record",
    "run_bounded_smoke",
    "run_one_unit",
    "runner_contract_identifier",
]
