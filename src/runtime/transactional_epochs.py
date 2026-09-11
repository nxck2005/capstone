"""Crash-safe committed-epoch custody for W9 v4 runtimes.

The store is intentionally independent of a model or dataset.  A caller hands
it already-serialised checkpoint bytes and a finite epoch record; the store
publishes a complete same-filesystem directory or nothing.  Existing committed
epochs are never replaced.  Corruption is a hold, while unpublished staging
directories are ignored as deterministic interruption residue.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class TransactionalRuntimeHold(RuntimeError):
    """Committed runtime evidence cannot be authenticated safely."""


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TransactionalRuntimeHold(message)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError as exc:
        raise TransactionalRuntimeHold(f"cannot open directory for fsync: {path}: {exc}") from None
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise TransactionalRuntimeHold(f"directory fsync failed: {path}: {exc}") from None
    finally:
        os.close(descriptor)


def _write_new(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)  # literal-ok: owner-only checkpoint metadata creation
    try:
        written = 0
        while written < len(data):
            written += os.write(descriptor, data[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class CommittedEpoch:
    epoch: int
    record: dict[str, Any]
    sidecar: dict[str, Any]
    checkpoint_path: Path
    checkpoint_sha256: str
    chain_sha256: str


class TransactionalEpochStore:
    """Authenticate and publish one v4 run's committed epoch chain."""

    EPOCH_RE = re.compile(r"^epoch-(\d{4})$")
    STAGING_PREFIX = ".epoch-"
    SCHEMA_VERSION = 1
    GENESIS_CHAIN = "0" * 64  # literal-ok: fixed-width genesis chain sentinel

    def __init__(self, runtime_root: Path, *, identity: Mapping[str, Any], total_epochs: int, role: str) -> None:
        self.runtime_root = Path(runtime_root)
        _require(not self.runtime_root.is_symlink(), "runtime root is a symlink")
        self.identity = dict(identity)
        _require(self.identity and all(isinstance(key, str) for key in self.identity), "runtime identity is empty or malformed")
        _require(isinstance(total_epochs, int) and total_epochs > 0, "total_epochs must be positive")
        self.total_epochs = total_epochs
        self.role = str(role)
        _require(self.role, "runtime role is empty")
        self.epochs_root = self.runtime_root / "epochs"
        self._authenticated: list[CommittedEpoch] | None = None
        self._authenticated_signatures: dict[int, tuple[Any, ...]] = {}
        self.last_inspect_authenticated_count = 0

    def initialise(self) -> None:
        if self.runtime_root.exists():
            _require(self.runtime_root.is_dir(), "runtime root is not a directory")
        else:
            self.runtime_root.mkdir(parents=True, exist_ok=False)
        _require(not self.runtime_root.is_symlink(), "runtime root is a symlink")
        if self.epochs_root.exists() or self.epochs_root.is_symlink():
            _require(self.epochs_root.is_dir() and not self.epochs_root.is_symlink(), "epochs root is not a directory")
        else:
            self.epochs_root.mkdir(exist_ok=False)
        _fsync_directory(self.epochs_root)

    def _identity(self, value: Mapping[str, Any], label: str) -> None:
        _require(dict(value) == self.identity, f"{label} runtime identity differs")

    def _read_json(self, path: Path, label: str) -> tuple[dict[str, Any], bytes]:
        _require(path.is_file() and not path.is_symlink(), f"{label} is missing or unsafe")
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TransactionalRuntimeHold(f"{label} is corrupt: {exc}") from None
        _require(isinstance(value, dict), f"{label} is not an object")
        _require(raw == canonical_bytes(value), f"{label} is not canonical")
        return value, raw

    @staticmethod
    def _signature(path: Path, label: str) -> tuple[int, ...]:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise TransactionalRuntimeHold(f"{label} cannot be inspected: {exc}") from None
        _require(not os.path.islink(path), f"{label} is a symlink")
        _require(stat.S_ISREG(metadata.st_mode), f"{label} is not a regular file")
        # ctime catches same-size in-place replacement in the active process;
        # inode/device also catches replacement by a different file.  This is
        # a safe cache invalidation check and does not reread old checkpoints.
        return (
            int(metadata.st_dev),
            int(metadata.st_ino),
            int(metadata.st_size),
            int(metadata.st_mtime_ns),
            int(metadata.st_ctime_ns),
            int(metadata.st_mode),
        )

    def _epoch_signature(self, epoch: int) -> tuple[Any, ...]:
        final = self.epochs_root / f"epoch-{epoch:04d}"
        try:
            children = tuple(sorted(child.name for child in final.iterdir()))
        except OSError as exc:
            raise TransactionalRuntimeHold(f"epoch {epoch} directory cannot be inspected: {exc}") from None
        return (
            children,
            *(self._signature(final / name, f"epoch {epoch} {name}") for name in ("checkpoint.pt", "record.json", "sidecar.json")),
        )

    def _cache_signatures(self, epochs: list[CommittedEpoch]) -> None:
        self._authenticated_signatures = {
            item.epoch: self._epoch_signature(item.epoch) for item in epochs
        }

    def _authenticate_epoch(
        self,
        epoch: int,
        *,
        predecessor_chain: str | None = None,
    ) -> CommittedEpoch:
        self.last_inspect_authenticated_count += 1
        final = self.epochs_root / f"epoch-{epoch:04d}"
        _require(final.is_dir() and not final.is_symlink(), f"committed epoch directory is missing: {epoch}")
        try:
            children = {child.name for child in final.iterdir()}
        except OSError as exc:
            raise TransactionalRuntimeHold(f"epoch {epoch} directory cannot be inspected: {exc}") from None
        _require(
            children == {"checkpoint.pt", "record.json", "sidecar.json"},
            f"epoch {epoch} contains an unexpected path",
        )
        checkpoint = final / "checkpoint.pt"
        record_path = final / "record.json"
        sidecar_path = final / "sidecar.json"
        _require(checkpoint.is_file() and not checkpoint.is_symlink(), f"epoch {epoch} checkpoint is missing")
        record, record_raw = self._read_json(record_path, f"epoch {epoch} record")
        sidecar, _ = self._read_json(sidecar_path, f"epoch {epoch} sidecar")
        try:
            checkpoint_raw = checkpoint.read_bytes()
        except OSError as exc:
            raise TransactionalRuntimeHold(f"epoch {epoch} checkpoint cannot be read: {exc}") from None
        checkpoint_sha = sha256_bytes(checkpoint_raw)
        _require(record.get("schema_version") == self.SCHEMA_VERSION, f"epoch {epoch} record schema differs")
        _require(record.get("artifact_role") == self.role, f"epoch {epoch} role differs")
        _require(record.get("epoch") == epoch and record.get("next_epoch") == epoch + 1, f"epoch {epoch} sequence differs")
        self._identity(record.get("identity", {}), f"epoch {epoch} record")
        _require(sidecar.get("schema_version") == self.SCHEMA_VERSION and sidecar.get("artifact_role") == self.role, f"epoch {epoch} sidecar schema differs")
        _require(sidecar.get("epoch") == epoch and sidecar.get("next_epoch") == epoch + 1, f"epoch {epoch} sidecar sequence differs")
        self._identity(sidecar.get("identity", {}), f"epoch {epoch} sidecar")
        _require(sidecar.get("checkpoint_sha256") == checkpoint_sha, f"epoch {epoch} checkpoint hash differs")
        _require(sidecar.get("record_sha256") == sha256_bytes(record_raw), f"epoch {epoch} record hash differs")
        _require(record.get("checkpoint_sha256") == checkpoint_sha, f"epoch {epoch} record checkpoint hash differs")
        if predecessor_chain is None:
            _require(epoch == 0, "non-genesis epoch authentication requires its predecessor chain")
            predecessor_chain = self.GENESIS_CHAIN
        predecessor = predecessor_chain
        _require(sidecar.get("predecessor_chain_sha256") == predecessor, f"epoch {epoch} predecessor chain differs")
        chain_body = {
            "artifact_role": self.role,
            "checkpoint_sha256": checkpoint_sha,
            "epoch": epoch,
            "identity": self.identity,
            "predecessor_chain_sha256": predecessor,
            "record_sha256": sha256_bytes(record_raw),
        }
        chain = canonical_sha256(chain_body)
        _require(sidecar.get("chain_sha256") == chain, f"epoch {epoch} chain digest differs")
        return CommittedEpoch(epoch, record, sidecar, checkpoint, checkpoint_sha, chain)

    def _pointer_body(self, committed: CommittedEpoch) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "artifact_role": self.role,
            "identity": self.identity,
            "epoch": committed.epoch,
            "next_epoch": committed.epoch + 1,
            "epoch_path": str(committed.checkpoint_path.parent.relative_to(self.runtime_root)),
            "chain_sha256": committed.chain_sha256,
        }

    def _publish_pointer(self, committed: CommittedEpoch) -> None:
        pointer = self._pointer_body(committed)
        target = self.runtime_root / "latest.json"
        _require(not target.is_symlink() and (not target.exists() or target.is_file()), "latest pointer is unsafe")
        staging = self.runtime_root / f".latest-{uuid.uuid4().hex}.staging"
        _write_new(staging, canonical_bytes(pointer))
        os.replace(staging, target)
        _fsync_directory(self.runtime_root)

    def inspect(self, *, repair_pointer: bool = True) -> list[CommittedEpoch]:
        self.initialise()
        self.last_inspect_authenticated_count = 0
        epochs: list[int] = []
        for child in self.epochs_root.iterdir():
            match = self.EPOCH_RE.fullmatch(child.name)
            if match:
                _require(child.is_dir() and not child.is_symlink(), f"epoch path is not a directory: {child}")
                epochs.append(int(match.group(1)))
            elif child.name.startswith(self.STAGING_PREFIX):
                # Unpublished staging is intentionally not accepted as a
                # committed epoch; it may safely remain for custody review.
                continue
            else:
                raise TransactionalRuntimeHold(
                    f"unexpected path in committed epochs root: {child.name}"
                )
        epochs.sort()
        _require(epochs == list(range(len(epochs))), "committed epochs do not form an unbroken prefix")

        committed: list[CommittedEpoch]
        cached = self._authenticated
        cache_usable = cached is not None and len(epochs) >= len(cached)
        if cache_usable:
            for item in cached:
                try:
                    signature = self._epoch_signature(item.epoch)
                except TransactionalRuntimeHold:
                    cache_usable = False
                    break
                if signature != self._authenticated_signatures.get(item.epoch):
                    cache_usable = False
                    break
        if cache_usable:
            committed = list(cached)
            predecessor = committed[-1].chain_sha256 if committed else self.GENESIS_CHAIN
            for epoch in range(len(committed), len(epochs)):
                item = self._authenticate_epoch(epoch, predecessor_chain=predecessor)
                committed.append(item)
                predecessor = item.chain_sha256
        else:
            committed = []
            predecessor = self.GENESIS_CHAIN
            for epoch in epochs:
                item = self._authenticate_epoch(epoch, predecessor_chain=predecessor)
                committed.append(item)
                predecessor = item.chain_sha256
        self._authenticated = committed
        self._cache_signatures(committed)
        pointer_path = self.runtime_root / "latest.json"
        if committed:
            if pointer_path.exists() or pointer_path.is_symlink():
                pointer, _ = self._read_json(pointer_path, "latest pointer")
                _require(pointer.get("schema_version") == self.SCHEMA_VERSION and pointer.get("artifact_role") == self.role, "latest pointer schema differs")
                self._identity(pointer.get("identity", {}), "latest pointer")
                pointer_epoch = pointer.get("epoch")
                _require(isinstance(pointer_epoch, int) and 0 <= pointer_epoch <= committed[-1].epoch, "latest pointer names an impossible epoch")
                _require(pointer == self._pointer_body(committed[pointer_epoch]), "latest pointer differs from its committed epoch")
            if repair_pointer and (not pointer_path.exists() or pointer.get("epoch") != committed[-1].epoch):
                self._publish_pointer(committed[-1])
        elif pointer_path.exists() or pointer_path.is_symlink():
            raise TransactionalRuntimeHold("latest pointer exists without a committed epoch")
        return committed

    def publish_epoch(self, epoch: int, checkpoint_bytes: bytes, record: Mapping[str, Any]) -> CommittedEpoch:
        committed = self.inspect()
        expected_epoch = len(committed)
        _require(0 <= epoch < self.total_epochs, f"epoch {epoch} is outside the configured run")
        _require(epoch == expected_epoch, f"epoch {epoch} is not the exact next epoch {expected_epoch}")
        final = self.epochs_root / f"epoch-{epoch:04d}"
        _require(not final.exists() and not final.is_symlink(), f"committed epoch {epoch} already exists and cannot be replaced")
        body = dict(record)
        body.update({
            "schema_version": self.SCHEMA_VERSION,
            "artifact_role": self.role,
            "identity": self.identity,
            "epoch": epoch,
            "next_epoch": epoch + 1,
            "checkpoint_sha256": sha256_bytes(checkpoint_bytes),
        })
        predecessor = self.GENESIS_CHAIN if epoch == 0 else committed[-1].chain_sha256
        record_raw = canonical_bytes(body)
        chain_body = {
            "artifact_role": self.role,
            "checkpoint_sha256": body["checkpoint_sha256"],
            "epoch": epoch,
            "identity": self.identity,
            "predecessor_chain_sha256": predecessor,
            "record_sha256": sha256_bytes(record_raw),
        }
        sidecar = {
            "schema_version": self.SCHEMA_VERSION,
            "artifact_role": self.role,
            "identity": self.identity,
            "epoch": epoch,
            "next_epoch": epoch + 1,
            "checkpoint_sha256": body["checkpoint_sha256"],
            "record_sha256": sha256_bytes(record_raw),
            "predecessor_chain_sha256": predecessor,
            "chain_sha256": canonical_sha256(chain_body),
        }
        staging = self.epochs_root / f"{self.STAGING_PREFIX}{epoch:04d}-{uuid.uuid4().hex}.staging"
        staging.mkdir(exist_ok=False)
        try:
            _write_new(staging / "checkpoint.pt", checkpoint_bytes)
            _write_new(staging / "record.json", record_raw)
            _write_new(staging / "sidecar.json", canonical_bytes(sidecar))
            _fsync_directory(staging)
            try:
                os.rename(staging, final)
            except FileExistsError:
                raise TransactionalRuntimeHold(f"committed epoch {epoch} appeared during publication") from None
            _fsync_directory(self.epochs_root)
        except Exception:
            # The staging directory is deliberately left for forensic custody
            # if publication failed after bytes were written.
            raise
        published = self._authenticate_epoch(epoch, predecessor_chain=predecessor)
        self._authenticated = [*committed, published]
        self._cache_signatures(self._authenticated)
        self._publish_pointer(published)
        return published

    def terminalize(
        self,
        *,
        selected_epoch: int,
        selection_metric: Mapping[str, Any],
        tie_break: str,
        test: str = "SEALED",
    ) -> tuple[dict[str, Any], bool]:
        committed = self.inspect()
        _require(len(committed) == self.total_epochs, "terminalization requires every configured epoch")
        _require(0 <= selected_epoch < self.total_epochs, "selected epoch is outside the committed run")
        selected = committed[selected_epoch]
        opportunities = sum(int(item.record.get("optimizer_opportunities", 0)) for item in committed)
        applied = sum(int(item.record.get("applied_optimizer_steps", 0)) for item in committed)
        skips = sum(int(item.record.get("grad_scaler_skips", 0)) for item in committed)
        _require(applied + skips == opportunities, "terminal optimizer counters do not reconcile")
        body = {
            "schema_version": 1,
            "artifact_role": f"{self.role}_TERMINAL",
            "run_identity": self.identity,
            "source_binding": self.identity.get("source_binding"),
            "config_hash": self.identity.get("config_hash"),
            "total_epochs": self.total_epochs,
            "selected_epoch": selected_epoch,
            "selected_checkpoint_path": str(selected.checkpoint_path.relative_to(self.runtime_root)),
            "selected_checkpoint_sha256": selected.checkpoint_sha256,
            "selection_metric": dict(selection_metric),
            "tie_break": tie_break,
            "optimizer_opportunities": opportunities,
            "applied_optimizer_steps": applied,
            "grad_scaler_skips": skips,
            "committed_epoch_chain_sha256": committed[-1].chain_sha256,
            "test_access": 0,
            "test": test,
        }
        expected = dict(body)
        expected["terminal_id"] = f"w9terminal-{canonical_sha256(body)}"
        terminal_path = self.runtime_root / "run_terminal.json"
        if terminal_path.exists() or terminal_path.is_symlink():
            actual, raw = self._read_json(terminal_path, "run terminal")
            _require(actual == expected, "run terminal differs from authenticated recomputation")
            return actual, True
        staging = self.runtime_root / f".run-terminal-{uuid.uuid4().hex}.staging"
        _write_new(staging, canonical_bytes(expected))
        try:
            os.rename(staging, terminal_path)
        except FileExistsError:
            # A concurrent publisher may have won; authenticate the winner.
            actual, _ = self._read_json(terminal_path, "run terminal")
            _require(actual == expected, "concurrent run terminal differs")
            return actual, True
        _fsync_directory(self.runtime_root)
        return expected, False


__all__ = ["CommittedEpoch", "TransactionalEpochStore", "TransactionalRuntimeHold", "canonical_bytes", "canonical_sha256", "sha256_bytes"]
