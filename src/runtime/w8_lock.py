"""Kernel-enforced sole-writer lock for the complete W8 campaign.

The lock is deliberately independent from the W7 lock.  Its file descriptor is
held by the one detached W8 process for the whole six-run campaign; metadata is
only an observable description and never ownership evidence.
"""

from __future__ import annotations

import fcntl
import json
import os
import socket
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


W8_GLOBAL_LOCK_PATH = Path("/tmp/capstone-w8-final-global.lock")
_METADATA_SUFFIX = ".metadata.json"


class W8LockBusy(RuntimeError):
    """Another process owns the W8 kernel lock."""


class W8LockHold(RuntimeError):
    """The W8 lock could not be acquired or safely described."""


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def metadata_path(lock_path: Path = W8_GLOBAL_LOCK_PATH) -> Path:
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


class W8CampaignLock:
    """A non-blocking ``flock`` whose lifetime spans the entire campaign."""

    def __init__(
        self,
        *,
        campaign_id: str,
        source_commit: str,
        execution_image: str,
        gpu_uuid: str,
        lock_path: Path = W8_GLOBAL_LOCK_PATH,
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
            raise ValueError("W8 lock metadata fields must be non-empty")
        if not isinstance(values["pid"], int) or values["pid"] <= 0:
            raise ValueError("W8 lock PID is invalid")
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
            raise W8LockHold("W8 campaign lock is already held by this object")
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.lock_path, flags, 0o666)
        except OSError as exc:
            raise W8LockHold(f"cannot open W8 campaign lock safely: {exc}") from None
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError) as exc:
                if isinstance(exc, BlockingIOError) or getattr(exc, "errno", None) in {11, 35}:  # literal-ok: Linux EAGAIN/EDEADLK errno values
                    raise W8LockBusy("another W8 campaign owns the global kernel lock") from None
                raise W8LockHold(f"cannot acquire W8 campaign flock: {exc}") from None
            try:
                _write_metadata(metadata_path(self.lock_path), self._metadata)
            except BaseException as exc:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                raise W8LockHold(f"cannot publish W8 lock metadata: {exc}") from None
            self._fd = descriptor
            return self.metadata
        except BaseException:
            os.close(descriptor)
            raise

    def release(self) -> None:
        descriptor, self._fd = self._fd, None
        if descriptor is None:
            return
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __enter__(self) -> W8CampaignLock:
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


def acquire_w8_campaign_lock(**kwargs: Any) -> W8CampaignLock:
    """Construct and acquire the one global W8 campaign lock."""

    lock = W8CampaignLock(**kwargs)
    lock.acquire()
    return lock
