"""Checkout-portable authentication for the frozen Pascal successor evidence.

The original C3--C7 closeout used a normalized GNU tar stream as a runtime
tree digest.  That digest is useful historical provenance, but it includes
filesystem modes and local coordination files that Git does not preserve.
This module adds the scientific evidence identity used by all strict loader
callers after the repair:

* only the campaign state and request/result/state JSON namespace is included;
* every included path, byte length and exact-byte SHA-256 is bound;
* the complete runtime is audited independently before the frozen table is
  returned; and
* the old merge, table and C6 artifacts remain exact historical artifacts.

No function here starts a worker, opens a selection gate, decodes validation
data, trains a model or reads the test split.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from baseline import g8_pascal_merge as legacy
from baseline.classical import composition
from baseline.g8_campaign import canonical_json, rendered_json, sha256_bytes
from baseline.g8_pascal_production import (
    REQUIRED_COUNT,
    SUCCESSOR_LOGICAL_RUNTIME_ROOT,
    SUCCESSOR_PROFILE_ID,
    SUCCESSOR_ROOT,
    TRIALS_PER_IDENTITY,
    audit_campaign,
    successor_bindings,
    validate_production_contracts,
    validate_runtime_namespace,
)
from config.params import REPO_ROOT


class PortableVerificationError(legacy.SuccessorMergeError):
    """Portable scientific evidence is not an eligible frozen input."""


PORTABLE_MANIFEST_PATH = SUCCESSOR_ROOT / "portable_scientific_runtime_manifest.json"
PORTABLE_PROVENANCE_PATH = SUCCESSOR_ROOT / "portable_verification_provenance.json"
PORTABLE_MANIFEST_SCHEMA_VERSION = 1
PORTABLE_PROVENANCE_SCHEMA_VERSION = 1
PORTABLE_MANIFEST_ARTIFACT_ROLE = "g8_c_pascal_successor_portable_scientific_runtime_manifest"
PORTABLE_PROVENANCE_ARTIFACT_ROLE = "g8_c_pascal_successor_portable_verification_provenance"
PORTABLE_MANIFEST_ID_PREFIX = "g8pportable-"
PORTABLE_PROVENANCE_ID_PREFIX = "g8pportableprov-"
HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
CONTROL_NAMES = frozenset({".locks", ".campaign.lock"})

# These are the sources of the corrective verification epoch.  The list is
# itself part of the provenance artifact so later edits cannot silently become
# the bytes claimed by this repair.
PORTABLE_SOURCE_PATHS = (
    ("src/baseline/g8_pascal_merge.py", "historical-loader-compatibility"),
    ("src/baseline/g8_pascal_portable.py", "portable-scientific-runtime-verifier"),
    ("tools/gen_g8_pascal_portable_manifest.py", "portable-manifest-generator"),
    ("tools/gen_g8_pascal_portable_provenance.py", "portable-provenance-generator"),
    ("tools/verify_g8_pascal_portable.py", "portable-verifier-entrypoint"),
    ("tools/verify_g8_pascal_closeout.py", "historical-c6-compatibility"),
)

MANIFEST_FIELDS = (
    "schema_version",
    "artifact_role",
    "manifest_id",
    "runtime_relative_path",
    "scientific_runtime_sha256",
    "bindings",
    "required_identity_count",
    "trials_per_identity",
    "files",
)
FILE_FIELDS = ("path", "bytes", "sha256")
PROVENANCE_FIELDS = (
    "schema_version",
    "artifact_role",
    "epoch",
    "provenance_id",
    "classification",
    "historical_g8_c",
    "portable_evidence",
    "repair_sources",
    "repair_source_digest",
    "repair_commit",
    "scientific_values_unchanged",
    "safety",
)


def _fail(message: str) -> None:
    raise PortableVerificationError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _digest(value: Any, label: str) -> None:
    _require(isinstance(value, str) and HEX_DIGEST.fullmatch(value) is not None, f"{label} is not a SHA-256 digest")


def _strict_object(value: Any, fields: Sequence[str], label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping) and set(value) == set(fields), f"{label} schema differs")
    return dict(value)


def _read_rendered(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read {label}: {exc}")
    _require(isinstance(payload, dict), f"{label} is not an object")
    _require(raw == rendered_json(payload), f"{label} is not canonical rendered JSON")
    return payload, raw


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        _fail(f"cannot hash {path}: {exc}")


def _binding_snapshot() -> dict[str, Any]:
    bindings = successor_bindings()
    production = validate_production_contracts()
    _authority, authority_set_digest, authority_file_digest = legacy.load_required_authority()
    _require(authority_file_digest == bindings["required_bler_artifact_sha256"], "portable authority file binding differs")
    expected = {
        "campaign_id": bindings["campaign_id"],
        "execution_profile_id": SUCCESSOR_PROFILE_ID,
        "campaign_manifest_sha256": bindings["campaign_manifest_sha256"],
        "source_manifest_sha256": bindings["source_manifest_sha256"],
        "runner_contract_sha256": bindings["runner_contract_sha256"],
        "production_contract_sha256": production["production_contract_sha256"],
        "production_source_manifest_sha256": production["production_source_manifest_sha256"],
        "production_runner_contract_sha256": production["production_runner_contract_sha256"],
        "required_bler_artifact_sha256": bindings["required_bler_artifact_sha256"],
        "required_authority_identity_set_sha256": authority_set_digest,
    }
    return expected


def _scientific_paths(root: Path) -> list[Path]:
    """Enumerate only the scientific runtime namespace, never coordination."""

    _require(root.is_dir() and not root.is_symlink(), f"portable runtime root is not a real directory: {root}")
    paths: list[Path] = []
    campaign_state = root / "campaign_state.json"
    _require(campaign_state.is_file() and not campaign_state.is_symlink(), "portable runtime campaign_state.json is missing")
    paths.append(campaign_state)
    try:
        entries = sorted(root.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        _fail(f"cannot enumerate portable runtime: {exc}")
    for entry in entries:
        if entry.name == "campaign_state.json" or entry.name in CONTROL_NAMES:
            continue
        _require(entry.is_dir() and not entry.is_symlink(), f"foreign portable runtime path: {entry.name}")
        try:
            children = sorted(entry.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            _fail(f"cannot enumerate portable runtime bucket {entry.name}: {exc}")
        for child in children:
            _require(child.is_file() and not child.is_symlink(), f"foreign portable runtime artifact: {child}")
            paths.append(child)
    return sorted(paths, key=lambda item: item.relative_to(root).as_posix())


def _file_entries(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in _scientific_paths(root):
        relative = path.relative_to(root).as_posix()
        try:
            raw = path.read_bytes()
        except OSError as exc:
            _fail(f"cannot read portable scientific file {relative}: {exc}")
        entries.append({"path": relative, "bytes": len(raw), "sha256": sha256_bytes(raw)})
    return entries


def _manifest_digest(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body["manifest_id"] = None
    body["scientific_runtime_sha256"] = None
    return sha256_bytes(canonical_json(body))


def _manifest_id(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("manifest_id", None)
    return PORTABLE_MANIFEST_ID_PREFIX + sha256_bytes(canonical_json(body))


def build_scientific_manifest(root: Path | str, *, validate_namespace: bool = True) -> dict[str, Any]:
    """Build the portable manifest without consulting filesystem metadata."""

    root_path = Path(root).resolve()
    if validate_namespace:
        try:
            validate_runtime_namespace(root_path)
        except Exception as exc:
            _fail(f"portable runtime namespace is invalid: {exc}")
    files = _file_entries(root_path)
    body: dict[str, Any] = {
        "schema_version": PORTABLE_MANIFEST_SCHEMA_VERSION,
        "artifact_role": PORTABLE_MANIFEST_ARTIFACT_ROLE,
        "manifest_id": None,
        "runtime_relative_path": SUCCESSOR_LOGICAL_RUNTIME_ROOT,
        "scientific_runtime_sha256": None,
        "bindings": _binding_snapshot(),
        "required_identity_count": REQUIRED_COUNT,
        "trials_per_identity": TRIALS_PER_IDENTITY,
        "files": files,
    }
    body["scientific_runtime_sha256"] = _manifest_digest(body)
    body["manifest_id"] = _manifest_id(body)
    return _validate_manifest_shape(body)


def _validate_manifest_shape(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = _strict_object(payload, MANIFEST_FIELDS, "portable scientific runtime manifest")
    _require(value["schema_version"] == PORTABLE_MANIFEST_SCHEMA_VERSION and value["artifact_role"] == PORTABLE_MANIFEST_ARTIFACT_ROLE, "unsupported portable runtime manifest")
    _require(value["runtime_relative_path"] == SUCCESSOR_LOGICAL_RUNTIME_ROOT, "portable runtime logical path differs")
    _require(value["required_identity_count"] == REQUIRED_COUNT and value["trials_per_identity"] == TRIALS_PER_IDENTITY, "portable runtime coverage/trial contract differs")
    _digest(value["scientific_runtime_sha256"], "portable scientific runtime digest")
    _require(value["scientific_runtime_sha256"] == _manifest_digest(value), "portable scientific runtime digest does not reproduce")
    _require(value["manifest_id"] == _manifest_id(value), "portable scientific runtime manifest ID does not reproduce")
    _require(value["bindings"] == _binding_snapshot(), "portable runtime bindings differ")
    files = value["files"]
    _require(isinstance(files, list), "portable runtime file list is not a list")
    previous = None
    for item in files:
        entry = _strict_object(item, FILE_FIELDS, "portable runtime file entry")
        path = entry["path"]
        _require(isinstance(path, str) and path and not PurePosixPath(path).is_absolute() and ".." not in PurePosixPath(path).parts, "portable runtime path is not canonical")
        _require(previous is None or previous < path, "portable runtime file list is not strictly sorted")
        previous = path
        _require(type(entry["bytes"]) is int and entry["bytes"] >= 0, f"portable runtime byte length is invalid: {path}")
        _digest(entry["sha256"], f"portable runtime file digest {path}")
    return value


def verify_portable_scientific_manifest(
    manifest: Mapping[str, Any],
    root: Path | str,
    *,
    validate_namespace: bool = True,
    require_repository_root: bool = False,
) -> dict[str, Any]:
    """Verify the manifest against exact scientific bytes at *root*."""

    value = _validate_manifest_shape(manifest)
    root_path = Path(root).resolve()
    if require_repository_root:
        _require(root_path == (REPO_ROOT / SUCCESSOR_LOGICAL_RUNTIME_ROOT).resolve(), "portable loader refuses a non-successor runtime path")
    if validate_namespace:
        try:
            validate_runtime_namespace(root_path)
        except Exception as exc:
            _fail(f"portable runtime namespace is invalid: {exc}")
    observed = _file_entries(root_path)
    _require(observed == value["files"], "portable scientific runtime files, paths, lengths or bytes differ")
    return value


def _read_portable_manifest(root: Path) -> tuple[dict[str, Any], bytes, str]:
    payload, raw = _read_rendered(PORTABLE_MANIFEST_PATH, "portable scientific runtime manifest")
    value = verify_portable_scientific_manifest(payload, root, require_repository_root=True)
    return value, raw, sha256_bytes(raw)


def _read_exact_artifact(path: Path, expected_sha256: str, label: str) -> tuple[dict[str, Any], bytes, str]:
    payload, raw = _read_rendered(path, label)
    digest = sha256_bytes(raw)
    _require(digest == expected_sha256, f"{label} bytes are not the frozen historical artifact")
    return payload, raw, digest


def _verify_historical_artifacts() -> tuple[dict[str, Any], dict[str, Any], bytes, bytes]:
    merge, merge_raw, merge_sha = _read_exact_artifact(legacy.MERGE_REPORT_PATH, legacy.HISTORICAL_MERGE_REPORT_SHA256, "successor merge report")
    table, table_raw, table_sha = _read_exact_artifact(legacy.TABLE_PATH, legacy.HISTORICAL_TABLE_SHA256, "successor BLER table")
    _require(merge["report_id"] == legacy.HISTORICAL_MERGE_REPORT_ID, "frozen successor merge ID changed")
    _require(table["table_id"] == legacy.HISTORICAL_TABLE_ID, "frozen successor table ID changed")
    _require(table["merge_report_id"] == merge["report_id"] and table["merge_report_sha256"] == merge_sha, "frozen table/merge binding differs")
    legacy.validate_successor_merge_report(merge)
    legacy.validate_successor_bler_table(table, merge_report=merge, merge_report_sha256=merge_sha)
    provenance, provenance_raw, provenance_sha = _read_exact_artifact(legacy.PROVENANCE_PATH, legacy.HISTORICAL_CLOSEOUT_SHA256, "successor C6 provenance")
    _require(provenance["closure_id"] == legacy.HISTORICAL_CLOSEOUT_ID, "frozen successor C6 ID changed")
    _require(provenance["closeout_source"]["source_digest"] == legacy.HISTORICAL_CLOSEOUT_SOURCE_DIGEST, "historical C6 source epoch was rewritten")
    _require(provenance["measurement_source"]["runtime_tree_sha256"] == legacy.HISTORICAL_RUNTIME_TREE_SHA256, "historical runtime tree provenance changed")
    _require(provenance["artifacts"]["merge_report"]["sha256"] == merge_sha and provenance["artifacts"]["bler_table"]["sha256"] == table_sha, "historical C6 artifact bindings differ")
    _require(provenance_sha == legacy.HISTORICAL_CLOSEOUT_SHA256, "historical C6 provenance hash read failed")
    return merge, table, merge_raw, table_raw


def _compare_reconstructed_science(frozen: Mapping[str, Any], reconstructed: Mapping[str, Any], table: Mapping[str, Any]) -> None:
    metadata = {"report_id", "runtime_tree_sha256", "closeout_source_digest"}
    for field in legacy.MERGE_FIELDS:
        if field in metadata:
            continue
        _require(frozen[field] == reconstructed[field], f"portable reconstruction differs from frozen merge: {field}")
    _require(frozen["units"] == reconstructed["units"], "portable reconstruction differs in per-unit scientific evidence")
    _require(legacy._table_curves(reconstructed["units"]) == table["curves"], "portable reconstruction differs from frozen BLER curves")
    _require(frozen["runtime_tree_sha256"] == legacy.HISTORICAL_RUNTIME_TREE_SHA256, "frozen merge legacy tree digest changed")


def _source_entries() -> list[dict[str, Any]]:
    entries = []
    for relative, role in PORTABLE_SOURCE_PATHS:
        path = REPO_ROOT / relative
        _require(path.is_file(), f"portable repair source is missing: {relative}")
        raw = path.read_bytes()
        entries.append({"path": relative, "role": role, "bytes": len(raw), "sha256": sha256_bytes(raw)})
    return entries


def _provenance_id(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("provenance_id", None)
    return PORTABLE_PROVENANCE_ID_PREFIX + sha256_bytes(canonical_json(body))


def build_portable_verification_provenance(*, repair_commit: str) -> dict[str, Any]:
    """Build the additive repair record after the manifest is published."""

    manifest, manifest_raw = _read_rendered(PORTABLE_MANIFEST_PATH, "portable scientific runtime manifest")
    manifest = _validate_manifest_shape(manifest)
    sources = _source_entries()
    body: dict[str, Any] = {
        "schema_version": PORTABLE_PROVENANCE_SCHEMA_VERSION,
        "artifact_role": PORTABLE_PROVENANCE_ARTIFACT_ROLE,
        "epoch": "g8-c-portable-scientific-runtime-v1",
        "provenance_id": None,
        "classification": {
            "kind": "implementation_clean_checkout_reproducibility_provenance_verifier_defect",
            "measurement_defect": False,
            "bler_characterization_defect": False,
            "metadata_only": True,
        },
        "historical_g8_c": {
            "merge_report_id": legacy.HISTORICAL_MERGE_REPORT_ID,
            "merge_report_sha256": legacy.HISTORICAL_MERGE_REPORT_SHA256,
            "table_id": legacy.HISTORICAL_TABLE_ID,
            "table_sha256": legacy.HISTORICAL_TABLE_SHA256,
            "closeout_id": legacy.HISTORICAL_CLOSEOUT_ID,
            "closeout_sha256": legacy.HISTORICAL_CLOSEOUT_SHA256,
            "legacy_runtime_tree_sha256": legacy.HISTORICAL_RUNTIME_TREE_SHA256,
            "legacy_closeout_source_digest": legacy.HISTORICAL_CLOSEOUT_SOURCE_DIGEST,
            "old_closeout_remains_historical_valid_evidence": True,
        },
        "portable_evidence": {
            "manifest_path": str(PORTABLE_MANIFEST_PATH.relative_to(REPO_ROOT)),
            "manifest_sha256": sha256_bytes(manifest_raw),
            "manifest_id": manifest["manifest_id"],
            "scientific_runtime_sha256": manifest["scientific_runtime_sha256"],
            "namespace": "campaign_state.json plus every request/result/state JSON; coordination paths excluded",
            "file_fields": ["path", "bytes", "sha256"],
            "exact_authority_count": REQUIRED_COUNT,
            "trials_per_identity": TRIALS_PER_IDENTITY,
        },
        "repair_sources": sources,
        "repair_source_digest": sha256_bytes(canonical_json(sources)),
        "repair_commit": repair_commit,
        "scientific_values_unchanged": {
            "request_bytes": True,
            "result_bytes": True,
            "state_bytes": True,
            "campaign_state_bytes": True,
            "authority": True,
            "trials": True,
            "raw_counts": True,
            "bler_values": True,
            "table_curves": True,
            "measurement_rerun": False,
        },
        "safety": {
            "protected_counters": legacy.PROTECTED_COUNTERS,
            "test_access": 0,
            "old_result_ingest": False,
            "predecessor_table_contribution": "none",
            "g8_d_scientific_semantics_changed": False,
            "validation_campaign_started": False,
            "g8_e_started": False,
        },
    }
    body["provenance_id"] = _provenance_id(body)
    return _strict_object(body, PROVENANCE_FIELDS, "portable verification provenance")


def _verify_git_source_binding(commit: str, entries: Sequence[Mapping[str, Any]]) -> None:
    _require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40,64}", commit) is not None, "portable repair commit is unresolved")
    try:
        subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=REPO_ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as exc:
        _fail(f"portable repair commit is not available: {commit}: {exc}")
    for item in entries:
        try:
            completed = subprocess.run(["git", "show", f"{commit}:{item['path']}"], cwd=REPO_ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (OSError, subprocess.CalledProcessError) as exc:
            _fail(f"portable repair source is absent at {commit}: {item['path']}: {exc}")
        _require(len(completed.stdout) == item["bytes"] and sha256_bytes(completed.stdout) == item["sha256"], f"portable repair source changed after its bound commit: {item['path']}")


def verify_portable_verification_provenance(manifest_raw: bytes) -> dict[str, Any]:
    payload, raw = _read_rendered(PORTABLE_PROVENANCE_PATH, "portable verification provenance")
    value = _strict_object(payload, PROVENANCE_FIELDS, "portable verification provenance")
    _require(value["schema_version"] == PORTABLE_PROVENANCE_SCHEMA_VERSION and value["artifact_role"] == PORTABLE_PROVENANCE_ARTIFACT_ROLE and value["epoch"] == "g8-c-portable-scientific-runtime-v1", "unsupported portable verification provenance")
    _require(value["provenance_id"] == _provenance_id(value), "portable verification provenance ID does not reproduce")
    _require(value["classification"] == {
        "kind": "implementation_clean_checkout_reproducibility_provenance_verifier_defect",
        "measurement_defect": False,
        "bler_characterization_defect": False,
        "metadata_only": True,
    }, "portable repair classification differs")
    historical = value["historical_g8_c"]
    _require(historical == {
        "merge_report_id": legacy.HISTORICAL_MERGE_REPORT_ID,
        "merge_report_sha256": legacy.HISTORICAL_MERGE_REPORT_SHA256,
        "table_id": legacy.HISTORICAL_TABLE_ID,
        "table_sha256": legacy.HISTORICAL_TABLE_SHA256,
        "closeout_id": legacy.HISTORICAL_CLOSEOUT_ID,
        "closeout_sha256": legacy.HISTORICAL_CLOSEOUT_SHA256,
        "legacy_runtime_tree_sha256": legacy.HISTORICAL_RUNTIME_TREE_SHA256,
        "legacy_closeout_source_digest": legacy.HISTORICAL_CLOSEOUT_SOURCE_DIGEST,
        "old_closeout_remains_historical_valid_evidence": True,
    }, "historical G8_C bindings differ")
    manifest, _ = _read_rendered(PORTABLE_MANIFEST_PATH, "portable scientific runtime manifest")
    portable = value["portable_evidence"]
    _require(portable["manifest_path"] == str(PORTABLE_MANIFEST_PATH.relative_to(REPO_ROOT)), "portable manifest path differs")
    _require(portable["manifest_sha256"] == sha256_bytes(manifest_raw) == _file_sha256(PORTABLE_MANIFEST_PATH), "portable manifest hash differs")
    _require(portable["manifest_id"] == manifest["manifest_id"] and portable["scientific_runtime_sha256"] == manifest["scientific_runtime_sha256"], "portable manifest identity differs")
    _require(portable["file_fields"] == ["path", "bytes", "sha256"] and portable["exact_authority_count"] == REQUIRED_COUNT and portable["trials_per_identity"] == TRIALS_PER_IDENTITY, "portable evidence contract differs")
    sources = value["repair_sources"]
    _require(sources == _source_entries(), "portable repair source bytes differ")
    _require(value["repair_source_digest"] == sha256_bytes(canonical_json(sources)), "portable repair source digest differs")
    _verify_git_source_binding(value["repair_commit"], sources)
    _require(value["scientific_values_unchanged"] == {
        "request_bytes": True,
        "result_bytes": True,
        "state_bytes": True,
        "campaign_state_bytes": True,
        "authority": True,
        "trials": True,
        "raw_counts": True,
        "bler_values": True,
        "table_curves": True,
        "measurement_rerun": False,
    }, "portable scientific-integrity declaration differs")
    _require(value["safety"] == {
        "protected_counters": legacy.PROTECTED_COUNTERS,
        "test_access": 0,
        "old_result_ingest": False,
        "predecessor_table_contribution": "none",
        "g8_d_scientific_semantics_changed": False,
        "validation_campaign_started": False,
        "g8_e_started": False,
    }, "portable repair safety declaration differs")
    _require(raw == rendered_json(value), "portable verification provenance rendering differs")
    return value


def verify_portable_successor(*, runtime_root: Path | str = legacy.SUCCESSOR_RUNTIME_ROOT) -> dict[str, Any]:
    """Authenticate all frozen G8_C artifacts through portable runtime bytes."""

    root = Path(runtime_root).resolve()
    merge, table, merge_raw, table_raw = _verify_historical_artifacts()
    manifest, manifest_raw, manifest_sha = _read_portable_manifest(root)
    provenance = verify_portable_verification_provenance(manifest_raw)
    try:
        audit = audit_campaign(root)
    except Exception as exc:
        _fail(f"portable successor audit failed: {exc}")
    _require(audit["campaign_id"] == successor_bindings()["campaign_id"], "portable successor campaign differs")
    _require(audit["accepted_authority_ordinals"] == list(range(REQUIRED_COUNT)), "portable successor authority coverage is incomplete")
    _require(audit["failed_authority_ordinals"] == audit["terminal_invalid_authority_ordinals"] == audit["in_progress_authority_ordinals"] == [], "portable successor has unresolved units")
    try:
        reconstructed = legacy.build_successor_merge_report(root)
    except Exception as exc:
        _fail(f"portable successor scientific reconstruction failed: {exc}")
    _compare_reconstructed_science(merge, reconstructed, table)
    _require(manifest["required_identity_count"] == REQUIRED_COUNT and manifest["trials_per_identity"] == TRIALS_PER_IDENTITY, "portable successor manifest coverage differs")
    _require(table["test_access"] == 0 and table["protected_counters"] == legacy.PROTECTED_COUNTERS and table["old_result_ingest"] is False, "portable successor table claims protected activity")
    _require(table["predecessor_table_contribution"] == "none", "portable successor admitted predecessor contribution")
    return {
        "status": "PASS",
        "runtime_root": str(root),
        "manifest_id": manifest["manifest_id"],
        "scientific_runtime_sha256": manifest["scientific_runtime_sha256"],
        "manifest_sha256": manifest_sha,
        "accepted_count": REQUIRED_COUNT,
        "measured_point_count": table["measured_point_count"],
        "trials_per_point": table["trials_per_point"],
        "merge_report_sha256": sha256_bytes(merge_raw),
        "table_sha256": sha256_bytes(table_raw),
        "portable_provenance_id": provenance["provenance_id"],
    }


def load_portable_successor_bler_table(
    path: Path | str = legacy.TABLE_PATH,
    *,
    merge_path: Path | str = legacy.MERGE_REPORT_PATH,
    runtime_root: Path | str = legacy.SUCCESSOR_RUNTIME_ROOT,
) -> composition.BlerTable:
    """Run strict portable verification, then expose the frozen table."""

    _require(Path(path).resolve() == legacy.TABLE_PATH.resolve(), "successor table loader refuses a non-successor table path")
    _require(Path(merge_path).resolve() == legacy.MERGE_REPORT_PATH.resolve(), "successor table loader refuses a non-successor merge path")
    verify_portable_successor(runtime_root=runtime_root)
    table, _raw, _sha = _read_exact_artifact(legacy.TABLE_PATH, legacy.HISTORICAL_TABLE_SHA256, "successor BLER table")
    curves: dict[composition.BlerIdentity, Any] = {}
    for curve in table["curves"]:
        identity = composition.BlerIdentity.from_mapping(curve["identity"])
        points = curve["points"]
        curves[identity] = composition._Curve(
            snr_db=tuple(float(point["snr_db"]) for point in points),
            bler=tuple(float(point["bler"]) for point in points),
            trials=TRIALS_PER_IDENTITY,
        )
    return composition.BlerTable(curves, provenance=str(legacy.TABLE_PATH.relative_to(REPO_ROOT)))


__all__ = [
    "PORTABLE_MANIFEST_PATH",
    "PORTABLE_PROVENANCE_PATH",
    "PORTABLE_SOURCE_PATHS",
    "PortableVerificationError",
    "build_scientific_manifest",
    "build_portable_verification_provenance",
    "verify_portable_scientific_manifest",
    "verify_portable_verification_provenance",
    "verify_portable_successor",
    "load_portable_successor_bler_table",
]
