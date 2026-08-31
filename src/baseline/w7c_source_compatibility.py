"""Fail-closed source compatibility for the additive W7-C λ transition.

W7-C changes the current normative λ state after the historical G8 and W6
artifacts were closed.  Those artifacts remain byte-identical; this small
compatibility record admits only the exact successor bytes needed to replay
historical, read-only verifiers.  It contains no scientific execution or
selection logic.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from config.params import REPO_ROOT
from baseline.w8_spec_compatibility import load as load_w8_spec_compatibility

COMPATIBILITY_RELATIVE_PATH = "results/learned/w7/w7_c_normative_source_compatibility.json"
AM91_RELATIVE_PATH = "results/baseline/g8/g8_am91_source_compatibility.json"
AM91_COMPATIBILITY_ID = "g8postsource-f92b28458d9462a4d722c065906e753e0b1edf38f4a2775ad4cf3bce02032935"
AM91_COMPATIBILITY_SHA256 = "f919907712a2aa9d1be17851249f02bf413b0b8b1cfe752591f4580e91f21a34"
AM91_PRIOR_COMMIT = "e9e273b1665e90f4244e59b785f71384d1efa008"
AM91_CURRENT_COMMIT = "94518919cc2f7603eb8dd35f41b8aef9a4c49e9d"
W7C_BASE_COMMIT = "002bc698e2059d941cc279ee6d700a646f56573f"

# Filled after the immutable record is materialized.  Keeping the expected
# identity here makes a resigned or substituted compatibility record fail
# closed rather than becoming a new accepted source epoch.
W7C_COMPATIBILITY_ID = "w7csource-bc37eecfb1b3dfddff04a850e9ffa8988305c1e1473b4080f2c622c96b5afa6c"
W7C_COMPATIBILITY_SHA256 = "506dc8ccaa7c9cc706b623ddccc0a1b48114248aa68980c0aa12397d64b4bcbe"
W8_G8_SOURCE_COMPATIBILITY_RELATIVE_PATH = "results/learned/w7/w8_g8_campaign_source_compatibility.json"
W8_G8_SOURCE_COMPATIBILITY_ID = "w8g8source-94456a6e684987148cc099108984e40cc644b38a752177a0086fb2d9210ff079"
W8_G8_SOURCE_COMPATIBILITY_SHA256 = "cb80c87fda984a68a1597fb6eb020cdc7818cff23980be80a2e52431545c0c47"
W8_G8_SOURCE_PATH = "src/baseline/g8_campaign.py"
W8_G8_BASE_BYTES = 57523
W8_G8_BASE_SHA256 = "6979875d351682c54547a8dde509499ea37a2e1549a771fa9234f0309f2c05af"

W7C_ALLOWED_PARAMETER_PATHS = (
    "learned_system.lambda_core",
    "learned_system.lambda_status",
)
W7C_PATHS = (
    "spec/SPEC.md",
    "spec/params.generated.yaml",
    "src/baseline/g8_campaign.py",
)
W7C_BOUNDARY = {
    "g4_adjudication_run": 1,
    "g8_scientific_change": 0,
    "training": 0,
    "model_inference": 0,
    "test_access": 0,
    "w8_final_training_runs": 0,
}
W7C_SCIENTIFIC_SOURCE = {
    "source_commit": "cc704fcacec706719bc2791ae14a6c9d71dd4032",
    "adjudicator_path": "src/adjudication/w7_g4.py",
    "adjudicator_git_blob_sha1": "f1071971bce8dc6a48ddf504e5743e3faea5edfa",
    "selected_lambda": 3.0,
}


class W7CSourceCompatibilityError(RuntimeError):
    """The exact additive W7-C source transition is absent or has drifted."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def rendered(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise W7CSourceCompatibilityError(message)


def _read_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise W7CSourceCompatibilityError(f"cannot read {label}: {exc}") from None
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    return value, raw


def _leaf_difference_paths(old: Any, new: Any, prefix: str = "") -> set[str]:
    if isinstance(old, Mapping) and isinstance(new, Mapping):
        result: set[str] = set()
        for key in set(old) | set(new):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in old or key not in new:
                result.add(child)
            else:
                result.update(_leaf_difference_paths(old[key], new[key], child))
        return result
    return set() if old == new else {prefix}


def _git_bytes(root: Path, commit: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    _require(result.returncode == 0, f"cannot resolve historical bytes for {path}")
    return result.stdout


def load_w8_g8_campaign_source_compatibility(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Authenticate the one exact AM-93 successor image for G8 history."""

    root = Path(root).resolve()
    path = root / W8_G8_SOURCE_COMPATIBILITY_RELATIVE_PATH
    value, raw = _read_object(path, "W8 G8-campaign source compatibility")
    _require(raw == rendered(value), "W8 G8-campaign source compatibility is not canonical rendered JSON")
    _require(sha256_bytes(raw) == W8_G8_SOURCE_COMPATIBILITY_SHA256, "W8 G8-campaign source compatibility bytes differ")
    body = {key: child for key, child in value.items() if key != "compatibility_id"}
    _require(
        value.get("compatibility_id") == W8_G8_SOURCE_COMPATIBILITY_ID
        and value.get("compatibility_id") == "w8g8source-" + sha256_bytes(canonical(body)),
        "W8 G8-campaign source compatibility ID differs",
    )
    _require(
        set(value)
        == {
            "schema_version", "artifact_role", "status", "amendment", "timing",
            "base_compatibility", "allowed_parameter_paths", "protected_boundary",
            "scientific_boundary", "entries", "compatibility_id",
        },
        "W8 G8-campaign source compatibility schema differs",
    )
    _require(
        value.get("schema_version") == 1
        and value.get("artifact_role") == "W8_HISTORICAL_G8_CAMPAIGN_SOURCE_COMPATIBILITY"
        and value.get("status") == "ADDITIVE_FAIL_CLOSED"
        and value.get("amendment") == "AM-93"
        and value.get("timing") == "after_w7_g4_before_w8_source_freeze"
        and value.get("base_compatibility") == {
            "path": COMPATIBILITY_RELATIVE_PATH,
            "compatibility_id": W7C_COMPATIBILITY_ID,
            "sha256": W7C_COMPATIBILITY_SHA256,
        }
        and value.get("allowed_parameter_paths") == []
        and value.get("protected_boundary") == {
            "g8_scientific_change": 0,
            "g8_campaign_measurement": 0,
            "w7_result_changed": False,
            "g4_result_changed": False,
            "w8_science_performed": False,
            "test_access": 0,
        }
        and value.get("scientific_boundary") == {
            "source_only_historical_verifier_repair": True,
            "g8_result_changed": False,
            "w7_result_changed": False,
            "g4_result_changed": False,
            "w8_science_performed": False,
            "test_access": 0,
        },
        "W8 G8-campaign source compatibility boundary differs",
    )
    entries = value.get("entries")
    _require(
        isinstance(entries, list) and len(entries) == 1
        and isinstance(entries[0], Mapping)
        and set(entries[0]) == {"path", "archived_bytes", "archived_sha256", "current_bytes", "current_sha256"}
        and entries[0]["path"] == W8_G8_SOURCE_PATH
        and entries[0]["archived_bytes"] == W8_G8_BASE_BYTES
        and entries[0]["archived_sha256"] == W8_G8_BASE_SHA256,
        "W8 G8-campaign source compatibility entries differ",
    )
    current_path = root / W8_G8_SOURCE_PATH
    _require(current_path.is_file() and not current_path.is_symlink(), "W8 G8-campaign successor source is missing")
    current = current_path.read_bytes()
    _require(
        entries[0]["current_bytes"] == len(current)
        and entries[0]["current_sha256"] == sha256_bytes(current),
        "W8 G8-campaign successor source bytes differ",
    )
    return value


def load(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Authenticate the one exact W7-C successor of the AM-91 source image."""

    root = Path(root).resolve()
    path = root / COMPATIBILITY_RELATIVE_PATH
    value, raw = _read_object(path, "W7-C source compatibility")
    _require(raw == rendered(value), "W7-C source compatibility is not canonical rendered JSON")
    _require(sha256_bytes(raw) == W7C_COMPATIBILITY_SHA256, "W7-C source compatibility file bytes differ")
    body = {key: child for key, child in value.items() if key != "compatibility_id"}
    _require(
        value.get("compatibility_id") == W7C_COMPATIBILITY_ID
        and value.get("compatibility_id") == "w7csource-" + sha256_bytes(canonical(body)),
        "W7-C source compatibility identity differs",
    )
    _require(
        set(value)
        == {
            "schema_version",
            "artifact_role",
            "compatibility_id",
            "amendment",
            "timing",
            "base_commit",
            "prior_compatibility",
            "allowed_parameter_paths",
            "protected_boundary",
            "scientific_source",
            "entries",
        },
        "W7-C source compatibility schema differs",
    )
    _require(
        value.get("schema_version") == 1
        and value.get("artifact_role") == "w7_c_normative_lambda_transition_source_compatibility"
        and value.get("amendment") == "W7-C"
        and value.get("timing") == "post_g4_selection_before_w8_authorization"
        and value.get("base_commit") == W7C_BASE_COMMIT
        and value.get("allowed_parameter_paths") == list(W7C_ALLOWED_PARAMETER_PATHS)
        and value.get("protected_boundary") == W7C_BOUNDARY
        and value.get("scientific_source") == W7C_SCIENTIFIC_SOURCE,
        "W7-C source compatibility boundary differs",
    )

    prior_path = root / AM91_RELATIVE_PATH
    prior, prior_raw = _read_object(prior_path, "AM-91 source compatibility")
    _require(
        sha256_bytes(prior_raw) == AM91_COMPATIBILITY_SHA256
        and prior.get("compatibility_id") == AM91_COMPATIBILITY_ID,
        "AM-91 prior compatibility identity differs",
    )
    _require(
        value.get("prior_compatibility")
        == {
            "path": AM91_RELATIVE_PATH,
            "compatibility_id": AM91_COMPATIBILITY_ID,
            "sha256": AM91_COMPATIBILITY_SHA256,
        },
        "W7-C prior compatibility binding differs",
    )

    entries = value.get("entries")
    _require(
        isinstance(entries, list)
        and [entry.get("path") for entry in entries if isinstance(entry, Mapping)] == list(W7C_PATHS),
        "W7-C source compatibility entries differ",
    )
    prior_entries = {
        entry["path"]: entry
        for entry in prior.get("entries", [])
        if isinstance(entry, Mapping) and isinstance(entry.get("path"), str)
    }
    _require(set(prior_entries) >= set(W7C_PATHS), "AM-91 prior source entries are incomplete")

    w8_spec: dict[str, Any] | None = None
    w8_g8_source: dict[str, Any] | None = None
    for entry in entries:
        _require(isinstance(entry, Mapping), "W7-C source entry is malformed")
        _require(
            set(entry) == {"path", "archived_bytes", "archived_sha256", "current_bytes", "current_sha256"},
            f"W7-C source entry schema differs: {entry.get('path')!r}",
        )
        path_text = str(entry["path"])
        old = prior_entries[path_text]
        _require(
            entry["archived_bytes"] == old.get("current_bytes")
            and entry["archived_sha256"] == old.get("current_sha256"),
            f"W7-C source transition does not start at AM-91: {path_text}",
        )
        current_path = root / path_text
        _require(current_path.is_file() and not current_path.is_symlink(), f"cannot read W7-C source {path_text}")
        current = current_path.read_bytes()
        if entry["current_bytes"] == len(current) and entry["current_sha256"] == sha256_bytes(current):
            continue
        if path_text in {"spec/SPEC.md", "spec/params.generated.yaml"}:
            if w8_spec is None:
                w8_spec = load_w8_spec_compatibility(root)
            successor = next(item for item in w8_spec["entries"] if item["path"] == path_text)
            _require(
                successor["base_bytes"] == entry["current_bytes"]
                and successor["base_sha256"] == entry["current_sha256"]
                and successor["current_bytes"] == len(current)
                and successor["current_sha256"] == sha256_bytes(current),
                f"W8 successor is not chained from W7-C: {path_text}",
            )
            continue
        if path_text == W8_G8_SOURCE_PATH:
            if w8_g8_source is None:
                w8_g8_source = load_w8_g8_campaign_source_compatibility(root)
            successor = w8_g8_source["entries"][0]
            _require(
                successor["archived_bytes"] == entry["current_bytes"]
                and successor["archived_sha256"] == entry["current_sha256"]
                and successor["current_bytes"] == len(current)
                and successor["current_sha256"] == sha256_bytes(current),
                "W8 G8-campaign source successor is not chained from W7-C",
            )
            continue
        _require(False, f"W7-C current source bytes differ: {path_text}")

    params_entry = next(entry for entry in entries if entry["path"] == "spec/params.generated.yaml")
    old_params = _git_bytes(root, AM91_CURRENT_COMMIT, "spec/params.generated.yaml")
    current_params = (root / "spec/params.generated.yaml").read_bytes()
    _require(
        len(old_params) == params_entry["archived_bytes"]
        and sha256_bytes(old_params) == params_entry["archived_sha256"],
        "W7-C historical generated-parameter bytes differ",
    )
    try:
        old_yaml = yaml.safe_load(old_params)
        current_yaml = yaml.safe_load(current_params)
    except yaml.YAMLError as exc:
        raise W7CSourceCompatibilityError(f"W7-C generated parameters are not YAML: {exc}") from None
    allowed_parameter_paths = set(W7C_ALLOWED_PARAMETER_PATHS)
    if w8_spec is not None:
        # AM-93 is a chained successor of this exact W7-C image.  Admit only
        # its one additional generated-parameter leaf; the W7-C λ transition
        # remains independently constrained above.
        allowed_parameter_paths.add("learned_system.checkpoint_selection_snr_db")
    _require(
        _leaf_difference_paths(old_yaml, current_yaml) == allowed_parameter_paths,
        "W7-C generated-parameter drift exceeds the authenticated successor leaves",
    )
    if w8_spec is None and w8_g8_source is None:
        return value
    # Consumers of this historical verifier need the authenticated *current*
    # byte frontier as well as the original W7-C record.  Return a read-only
    # projection whose entries are advanced only for the exact AM-93
    # normative/source paths; the published record itself remains byte-identical
    # and was authenticated above.
    projection = json.loads(json.dumps(value))
    successor_entries: dict[str, Mapping[str, Any]] = {}
    if w8_spec is not None:
        successor_entries.update({entry["path"]: entry for entry in w8_spec["entries"]})
    if w8_g8_source is not None:
        successor_entries.update({entry["path"]: entry for entry in w8_g8_source["entries"]})
    for entry in projection["entries"]:
        successor = successor_entries.get(entry["path"])
        if successor is not None:
            entry["current_bytes"] = successor["current_bytes"]
            entry["current_sha256"] = successor["current_sha256"]
    projection["allowed_parameter_paths"] = sorted(allowed_parameter_paths)
    return projection
