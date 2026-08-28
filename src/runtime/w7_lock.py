"""Kernel-enforced sole-writer lock for the complete W7 campaign."""

from __future__ import annotations

import fcntl
import json
import os
import socket
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# This is intentionally one host-global path, not a candidate/runtime path and
# not configurable per GPU.  A stale inode or metadata file is never ownership.
W7_GLOBAL_LOCK_PATH = Path("/tmp/capstone-w7-g4-global.lock")
_METADATA_SUFFIX = ".metadata.json"


class W7LockBusy(RuntimeError):
    """Another process owns the kernel lock."""


class W7LockHold(RuntimeError):
    """The lock could not be safely acquired or released."""


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def metadata_path(lock_path: Path = W7_GLOBAL_LOCK_PATH) -> Path:
    return Path(lock_path).with_name(Path(lock_path).name + _METADATA_SUFFIX)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_metadata(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class W7CampaignLock:
    """A held ``flock`` whose file descriptor lifetime spans the campaign."""

    def __init__(
        self,
        *,
        campaign_id: str,
        source_commit: str,
        execution_image: str,
        gpu_uuid: str,
        lock_path: Path = W7_GLOBAL_LOCK_PATH,
    ) -> None:
        values = {
            "campaign_id": campaign_id,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "source_commit": source_commit,
            "execution_image": execution_image,
            "gpu_uuid": gpu_uuid,
            "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if any(value in (None, "") for value in values.values()):
            raise ValueError("W7 lock metadata fields must be non-empty")
        if not isinstance(values["pid"], int) or values["pid"] <= 0:
            raise ValueError("W7 lock PID is invalid")
        self.lock_path = Path(lock_path)
        self._metadata = values
        self._fd: int | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self) -> dict[str, Any]:
        if self._fd is not None:
            raise W7LockHold("W7 campaign lock is already held by this object")
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(self.lock_path, flags, 0o666)
        except OSError as exc:
            raise W7LockHold(f"cannot open W7 campaign lock safely: {exc}") from None
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError) as exc:
                if isinstance(exc, BlockingIOError) or getattr(exc, "errno", None) in {11, 35}:  # literal-ok: Linux EAGAIN/EDEADLK errno values
                    raise W7LockBusy("another W7 campaign owns the global kernel lock") from None
                raise W7LockHold(f"cannot acquire W7 campaign flock: {exc}") from None
            try:
                _write_metadata(metadata_path(self.lock_path), self._metadata)
            except BaseException as exc:
                fcntl.flock(fd, fcntl.LOCK_UN)
                raise W7LockHold(f"cannot publish W7 lock metadata: {exc}") from None
            self._fd = fd
            return self.metadata
        except BaseException:
            os.close(fd)
            raise

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self) -> W7CampaignLock:
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback
        self.release()

    def __del__(self) -> None:
        # Closing the descriptor is the kernel's release mechanism.  This is a
        # last-resort guard for tests/process teardown; normal code uses context.
        try:
            self.release()
        except Exception:
            pass


def acquire_w7_campaign_lock(**kwargs: Any) -> W7CampaignLock:
    """Construct and acquire the one global W7 campaign lock."""

    lock = W7CampaignLock(**kwargs)
    lock.acquire()
    return lock
