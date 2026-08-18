"""Pre-data G8_E contract primitives.

This module contains only phase-opening and contract-validation code.  It does
not load a dataset, decode an image, invoke a classifier, construct a
``G8Authorization`` or start a scientific campaign.  The scientific runner
belongs to the later E2 checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from config.params import REPO_ROOT


class G8EContractError(ValueError):
    """A fail-closed G8_E opening-contract error."""


E0_PATH = REPO_ROOT / "results/baseline/g8_e/e0_open.json"
E0_SCHEMA_VERSION = 1
E0_ARTIFACT_ROLE = "g8_e_pre_data_opening"
E0_ARTIFACT_PREFIX = "g8e0-"

UPSTREAM_BINDING_PATHS: tuple[tuple[str, str], ...] = (
    (
        "results/baseline/g8_pascal_successor/portable_scientific_runtime_manifest.json",
        "g8_c_portable_scientific_runtime_manifest",
    ),
    (
        "results/baseline/g8_pascal_successor/portable_verification_provenance.json",
        "g8_c_portable_verification_provenance",
    ),
    ("src/baseline/g8_pascal_portable.py", "g8_c_portable_verifier_source"),
    ("src/baseline/g8_pascal_merge.py", "g8_c_legacy_loader_wrapper_source"),
    (
        "results/baseline/g8_pascal_successor/successor_bler_merge_report.json",
        "g8_c_frozen_successor_merge",
    ),
    (
        "results/baseline/g8_pascal_successor/successor_bler_table.json",
        "g8_c_frozen_successor_table",
    ),
    (
        "results/baseline/g8_pascal_successor/successor_closeout_provenance.json",
        "g8_c_historical_c6_closeout",
    ),
    ("results/baseline/g8_d/measurement_contract.json", "g8_d_current_contract"),
    ("results/baseline/g8_d/d7_handoff.json", "g8_d_current_d7_handoff"),
    (
        "results/baseline/g8_d/portable_rebind_provenance.json",
        "g8_d_portable_rebind_provenance",
    ),
    ("src/baseline/g8_d.py", "g8_d_identity_cache_source"),
)


def canonical_json(value: Any) -> bytes:
    """Return the repository's compact, deterministic JSON identity bytes."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def rendered_json(value: Any) -> bytes:
    """Return canonical human-readable JSON bytes used by committed artifacts."""

    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
        "ascii"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G8EContractError(message)


def _digest(value: object, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64, f"{label} is not a SHA-256 digest")  # literal-ok: SHA-256 hex width
    try:
        int(value, 16)  # literal-ok: hexadecimal digest radix
    except ValueError:
        raise G8EContractError(f"{label} is not a hexadecimal SHA-256 digest") from None
    return value


def _read_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise G8EContractError(f"cannot read {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def _binding(path_text: str, role: str) -> dict[str, Any]:
    path = REPO_ROOT / path_text
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise G8EContractError(f"cannot bind upstream path {path_text}: {exc}") from exc
    return {"path": path_text, "role": role, "bytes": len(payload), "sha256": sha256_bytes(payload)}


def _git_head() -> str:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,  # literal-ok: bounded provenance command
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise G8EContractError(f"cannot resolve opening Git commit: {exc}") from exc
    _require(len(value) == 40, "opening Git commit is not a full object ID")  # literal-ok: full Git object ID width
    return value


def _portable_epoch(provenance: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    portable = provenance.get("portable_evidence")
    classification = provenance.get("classification")
    _require(isinstance(portable, Mapping), "portable provenance has no portable evidence")
    _require(isinstance(classification, Mapping), "portable provenance has no defect classification")
    _require(provenance.get("epoch") == "g8-c-portable-scientific-runtime-v1", "portable verification epoch differs")
    _require(provenance.get("provenance_id") == "g8pportableprov-" + sha256_bytes(
        canonical_json({key: value for key, value in provenance.items() if key != "provenance_id"})
    ), "portable provenance ID does not reproduce")
    _require(manifest.get("manifest_id") == portable.get("manifest_id"), "portable manifest/provenance ID differs")
    _require(manifest.get("scientific_runtime_sha256") == portable.get("scientific_runtime_sha256"), "portable runtime digest differs")
    return {
        "epoch": provenance["epoch"],
        "classification": dict(classification),
        "repair_commit": provenance["repair_commit"],
        "repair_source_digest": provenance["repair_source_digest"],
        "portable_manifest_id": portable["manifest_id"],
        "portable_manifest_sha256": sha256_file(
            REPO_ROOT / "results/baseline/g8_pascal_successor/portable_scientific_runtime_manifest.json"
        ),
        "scientific_runtime_sha256": portable["scientific_runtime_sha256"],
        "exact_authority_count": portable["exact_authority_count"],
        "trials_per_identity": portable["trials_per_identity"],
        "legacy_runtime_tree_sha256": provenance["historical_g8_c"]["legacy_runtime_tree_sha256"],
        "legacy_tree_digest_is_historical_only": True,
    }


def build_e0_opening(*, opening_commit: str | None = None) -> dict[str, Any]:
    """Build the deterministic E0 opening witness from current upstream bytes."""

    from baseline.g8_pascal_portable import verify_portable_successor

    # This is a strict upstream read/verification.  It does not touch any
    # validation image or invoke a model-facing data path.
    portable_result = verify_portable_successor()
    _require(portable_result["status"] == "PASS", "portable G8_C verification did not pass")
    manifest = _read_object(
        REPO_ROOT / "results/baseline/g8_pascal_successor/portable_scientific_runtime_manifest.json"
    )
    provenance = _read_object(
        REPO_ROOT / "results/baseline/g8_pascal_successor/portable_verification_provenance.json"
    )
    merge = _read_object(
        REPO_ROOT / "results/baseline/g8_pascal_successor/successor_bler_merge_report.json"
    )
    table = _read_object(
        REPO_ROOT / "results/baseline/g8_pascal_successor/successor_bler_table.json"
    )
    closeout = _read_object(
        REPO_ROOT / "results/baseline/g8_pascal_successor/successor_closeout_provenance.json"
    )
    d_contract = _read_object(REPO_ROOT / "results/baseline/g8_d/measurement_contract.json")
    d_handoff = _read_object(REPO_ROOT / "results/baseline/g8_d/d7_handoff.json")
    d_rebind = _read_object(REPO_ROOT / "results/baseline/g8_d/portable_rebind_provenance.json")

    _require(merge.get("report_id") == "g8pmerge-" + sha256_bytes(canonical_json({
        key: value for key, value in merge.items() if key != "report_id"
    })), "frozen successor merge ID does not reproduce")
    _require(table.get("table_id") == "g8pblertable-" + sha256_bytes(canonical_json({
        key: value for key, value in table.items() if key != "table_id"
    })), "frozen successor table ID does not reproduce")
    _require(closeout.get("closure_id") == "g8pcloseout-" + sha256_bytes(canonical_json({
        key: value for key, value in closeout.items() if key != "closure_id"
    })), "historical C6 ID does not reproduce")
    _require(d_handoff.get("contract_id") == d_contract.get("contract_id"), "G8_D handoff is not bound to current contract")
    _require(d_contract.get("next_gate") == "G8_E/E0", "G8_D does not release E0")
    _require(d_handoff.get("g8_e_released") is True, "G8_D handoff does not release E0")
    _require(d_handoff.get("full_campaign_not_started") is True, "G8_D full campaign is not unopened")
    _require(d_rebind.get("current_contract", {}).get("contract_id") == d_contract.get("contract_id"), "G8_D rebind does not name current contract")
    _require(d_rebind.get("current_handoff", {}).get("artifact_id") == d_handoff.get("artifact_id"), "G8_D rebind does not name current handoff")

    upstream = sorted(
        (_binding(path, role) for path, role in UPSTREAM_BINDING_PATHS),
        key=lambda item: item["path"],
    )
    body: dict[str, Any] = {
        "schema_version": E0_SCHEMA_VERSION,
        "artifact_role": E0_ARTIFACT_ROLE,
        "phase": "G8_E",
        "checkpoint": "E0",
        "status": "OPEN",
        "opening_commit": opening_commit or _git_head(),
        "upstream_verification": {
            "g8_c_successor_verifier": "PASS",
            "g8_c_portable_verifier": "PASS",
            "g8_c_closeout_compatibility": "PASS",
            "exact_required_identities": 3213,
            "exact_accepted_identities": portable_result["accepted_count"],
            "exact_frozen_points": portable_result["measured_point_count"],
            "trials_per_identity": portable_result["trials_per_point"],
            "retry_history_ordinals": [0, 1],
            "predecessor_table_contribution": "none",
            "protected_counters": {
                "inference": 0,
                "training": 0,
                "validation_decoding": 0,
                "test_access": 0,
            },
        },
        "g8_c": {
            "campaign_id": merge["campaign_id"],
            "merge_report_id": merge["report_id"],
            "merge_report_sha256": sha256_file(REPO_ROOT / "results/baseline/g8_pascal_successor/successor_bler_merge_report.json"),
            "table_id": table["table_id"],
            "table_sha256": sha256_file(REPO_ROOT / "results/baseline/g8_pascal_successor/successor_bler_table.json"),
            "historical_c6_id": closeout["closure_id"],
            "historical_c6_sha256": sha256_file(REPO_ROOT / "results/baseline/g8_pascal_successor/successor_closeout_provenance.json"),
            "portable_manifest_id": manifest["manifest_id"],
            "portable_manifest_sha256": sha256_file(REPO_ROOT / "results/baseline/g8_pascal_successor/portable_scientific_runtime_manifest.json"),
            "portable_provenance_id": provenance["provenance_id"],
            "portable_provenance_sha256": sha256_file(REPO_ROOT / "results/baseline/g8_pascal_successor/portable_verification_provenance.json"),
            "portable_scientific_runtime_sha256": manifest["scientific_runtime_sha256"],
            "portable_verification_epoch": _portable_epoch(provenance, manifest),
        },
        "g8_d": {
            "contract_id": d_contract["contract_id"],
            "contract_sha256": sha256_file(REPO_ROOT / "results/baseline/g8_d/measurement_contract.json"),
            "handoff_id": d_handoff["artifact_id"],
            "handoff_sha256": sha256_file(REPO_ROOT / "results/baseline/g8_d/d7_handoff.json"),
            "portable_rebind_provenance_id": d_rebind["provenance_id"],
            "portable_rebind_provenance_sha256": sha256_file(REPO_ROOT / "results/baseline/g8_d/portable_rebind_provenance.json"),
            "d4_scientific_evidence": False,
            "d4_merge_eligible": False,
            "d6_scientific_evidence": False,
            "d6_merge_eligible": False,
        },
        "upstream_bindings": upstream,
        "safety": {
            "g8_e_measurement_coverage": 0,
            "e2_started": False,
            "pass_one_started": False,
            "pass_two_started": False,
            "training_started": False,
            "fallback_invoked": False,
            "ratio_adjudicated": False,
            "test_access": 0,
            "inference": 0,
            "training": 0,
            "validation_decoding": 0,
        },
        "declarations": {
            "g8_c_scientific_state_frozen": True,
            "portable_verification_pass": True,
            "g8_d_green": True,
            "g8_e_measurement_coverage": 0,
            "e2_not_started": True,
            "pass_one_not_started": True,
            "training_forbidden": True,
            "test_forbidden": True,
            "validation_image_decoding_required_to_open": False,
        },
    }
    body["artifact_id"] = E0_ARTIFACT_PREFIX + sha256_bytes(canonical_json(body))
    return body


def validate_e0_opening(value: Mapping[str, Any], *, expected_commit: str | None = None) -> dict[str, Any]:
    """Validate E0's exact schema and all current upstream bindings."""

    _require(isinstance(value, Mapping), "E0 artifact is not an object")
    required = {
        "schema_version", "artifact_role", "phase", "checkpoint", "status", "opening_commit",
        "upstream_verification", "g8_c", "g8_d", "upstream_bindings", "safety", "declarations", "artifact_id",
    }
    _require(set(value) == required, "E0 artifact schema differs")
    _require(value["schema_version"] == E0_SCHEMA_VERSION, "E0 schema version differs")
    _require((value["artifact_role"], value["phase"], value["checkpoint"], value["status"]) == (E0_ARTIFACT_ROLE, "G8_E", "E0", "OPEN"), "E0 header differs")
    _require(isinstance(value["opening_commit"], str) and len(value["opening_commit"]) == 40, "E0 opening commit is not a full SHA")  # literal-ok: full Git object ID width
    if expected_commit is not None:
        _require(value["opening_commit"] == expected_commit, "E0 opening commit differs")
    body = dict(value)
    artifact_id = body.pop("artifact_id")
    _require(artifact_id == E0_ARTIFACT_PREFIX + sha256_bytes(canonical_json(body)), "E0 artifact ID differs")

    verification = value["upstream_verification"]
    _require(isinstance(verification, Mapping), "E0 upstream verification is not an object")
    _require(all(verification[field] == "PASS" for field in ("g8_c_successor_verifier", "g8_c_portable_verifier", "g8_c_closeout_compatibility")), "E0 upstream verifier status is not PASS")
    _require((verification["exact_required_identities"], verification["exact_accepted_identities"], verification["exact_frozen_points"], verification["trials_per_identity"]) == (3213, 3213, 3213, 5000), "E0 G8_C coverage differs")  # literal-ok: frozen G8_C authority/trial contract
    _require(verification["retry_history_ordinals"] == [0, 1] and verification["predecessor_table_contribution"] == "none", "E0 retry/predecessor boundary differs")
    _require(verification["protected_counters"] == {"inference": 0, "training": 0, "validation_decoding": 0, "test_access": 0}, "E0 upstream protected counters are nonzero")

    g8c = value["g8_c"]
    _require(g8c["merge_report_id"] == "g8pmerge-2e861c39d8981af0e2d57dc8ded5828b9ed56a1459491e04929b5e9c3418de89", "E0 merge ID differs")
    _require(g8c["table_id"] == "g8pblertable-69ecc729f3b7dc3d67c0a3a5d8cf071cab927ad0a1e0cd5b18a6bbe674b9126f", "E0 table ID differs")
    _require(g8c["historical_c6_id"] == "g8pcloseout-8cc5be86e6bbb350ca35c1806686e751f4528ad32aa7083e9a754c4849feba70", "E0 C6 ID differs")
    for field in ("merge_report_sha256", "table_sha256", "historical_c6_sha256", "portable_manifest_sha256", "portable_provenance_sha256", "portable_scientific_runtime_sha256"):
        _digest(g8c[field], f"E0 G8_C {field}")
    epoch = g8c["portable_verification_epoch"]
    _require(epoch["epoch"] == "g8-c-portable-scientific-runtime-v1" and epoch["legacy_tree_digest_is_historical_only"] is True, "E0 portable epoch differs")
    _digest(epoch["scientific_runtime_sha256"], "E0 portable runtime digest")

    g8d = value["g8_d"]
    _require(g8d["contract_id"] == "g8dcontract-c1ebf0b23e0e5725d387f447e633b37f123688d2595695f92e86a1c663db7889", "E0 G8_D contract ID differs")
    _require(g8d["handoff_id"] == "g8dhandoff-31c48fcabe765a0e70bcd7bcfec5f4bd705b88ee3cb0d140ba9aecf67e1dfd4c", "E0 D7 handoff ID differs")
    _require(g8d["d4_scientific_evidence"] is False and g8d["d4_merge_eligible"] is False and g8d["d6_scientific_evidence"] is False and g8d["d6_merge_eligible"] is False, "E0 binds a scientific/mergeable D record")

    bindings = value["upstream_bindings"]
    expected_bindings = sorted((path, role) for path, role in UPSTREAM_BINDING_PATHS)
    _require(isinstance(bindings, list) and [item["path"] for item in bindings] == [path for path, _role in expected_bindings], "E0 upstream path set/order differs")
    for item, (_path, role) in zip(bindings, expected_bindings):
        _require(set(item) == {"path", "role", "bytes", "sha256"}, "E0 upstream binding schema differs")
        _require(item["role"] == role, f"E0 upstream role differs for {item['path']}")
        path = REPO_ROOT / item["path"]
        _require(path.is_file(), f"E0 upstream binding is missing: {item['path']}")
        _require(item["bytes"] == path.stat().st_size and item["sha256"] == sha256_file(path), f"E0 upstream bytes changed: {item['path']}")

    safety = value["safety"]
    _require(safety == {
        "g8_e_measurement_coverage": 0,
        "e2_started": False,
        "pass_one_started": False,
        "pass_two_started": False,
        "training_started": False,
        "fallback_invoked": False,
        "ratio_adjudicated": False,
        "test_access": 0,
        "inference": 0,
        "training": 0,
        "validation_decoding": 0,
    }, "E0 safety state is nonzero")
    declarations = value["declarations"]
    _require(all(declarations[field] is True for field in ("g8_c_scientific_state_frozen", "portable_verification_pass", "g8_d_green", "e2_not_started", "pass_one_not_started", "training_forbidden", "test_forbidden")), "E0 declarations are not closed")
    _require(declarations["g8_e_measurement_coverage"] == 0 and declarations["validation_image_decoding_required_to_open"] is False, "E0 opening boundary differs")
    return dict(value)


def verify_e0_file(path: Path = E0_PATH, *, expected_commit: str | None = None) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise G8EContractError(f"cannot read E0 artifact: {exc}") from exc
    _require(raw == rendered_json(value), "E0 artifact is not canonical rendered JSON")
    return validate_e0_opening(value, expected_commit=expected_commit)
