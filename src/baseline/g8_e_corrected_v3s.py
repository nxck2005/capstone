"""G8_E corrected-v3 confessor worker-successor epoch (additive E2-E4 only).

The owner aborted the partial corrected-v3 E2 campaign on the development
laptop at an exact clean durable prefix and relocated authoritative E2-E4
execution to the qualified ``confessor_pascal_cu126`` profile.  Classification
of the predecessor: ``PARTIAL_OWNER_ABORTED_PROFILE_RELOCATION`` — real
validation measurements, preserved intact, contributing zero successor
coverage and never silently treated as superseded-before-data.

Every scientific field of this epoch is copied byte-identically from the
frozen corrected-v3 contract: the measurement authority, logical/structural
mapping, scientific data identity, codec configuration, classifier identity,
outage policy, clean measurement semantics, transaction semantics, E3/E4
transformations and all G8_C/G8_D/G-1 upstream bindings.  Only execution
relocation changes: the profile/device binding, the campaign/contract/source
identities, the runtime root, the owner authorization artifact and the
operational compute/storage plan measured on the worker host.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from baseline import g8_e_corrected_v2 as v2
from baseline import g8_e_corrected_v3 as v3
from config.params import REPO_ROOT, get


V3S_SCHEMA_VERSION = v3.V3_SCHEMA_VERSION
V3S_ROOT = REPO_ROOT / "results/baseline/g8_e/e2_confessor_successor"
V3S_CONTRACT_PATH = V3S_ROOT / "measurement_contract.json"
V3S_SOURCE_MANIFEST_PATH = V3S_ROOT / "execution_source_manifest.json"
AM87_SOURCE_COMPATIBILITY_PATH = REPO_ROOT / "results/baseline/g8_f/am87_g8e_source_compatibility.json"
V3S_STORAGE_PLAN_PATH = V3S_ROOT / "compute_storage_plan.json"
V3S_RELOCATION_PROVENANCE_PATH = V3S_ROOT / "relocation_provenance.json"
V3S_SYNTHETIC_PROOF_PATH = V3S_ROOT / "synthetic_lifecycle_proof.json"
V3S_RUNTIME_ROOT = V3S_ROOT / "runtime"
V3S_AUTHORIZATION_PATH = V3S_ROOT / "e2_execution_authorization.json"
V3S_E2_COMPLETION_PATH = V3S_RUNTIME_ROOT / "e2_completion.json"
V3S_E3_PATH = V3S_RUNTIME_ROOT / "e3_exact_set_closure.json"
V3S_E4_PATH = V3S_RUNTIME_ROOT / "e4_count_derived.json"

V3S_CONTRACT_PREFIX = "g8econtractcorrectedv3s-"
V3S_CAMPAIGN_PREFIX = "g8e-v3s-"
V3S_SOURCE_PREFIX = "g8esourcecorrectedv3s-"

SUCCESSOR_PROFILE_ID = "confessor_pascal_cu126"

G8EV3SError = v3.G8EV3Error
FatalExecutionError = v3.FatalExecutionError
CampaignHoldError = v3.CampaignHoldError
ScientificDecodeFailure = v3.ScientificDecodeFailure
SyntheticSample = v3.SyntheticSample
MeasurementRecordV3S = v3.MeasurementRecordV3
PhysicalCacheKey = v3.PhysicalCacheKey
AtomicE2CampaignV3S = v3.AtomicE2CampaignV3
MeasurementExecutorV3S = v3.MeasurementExecutorV3

canonical_json = v3.canonical_json
rendered_json = v3.rendered_json
sha256_bytes = v3.sha256_bytes
sha256_file = v3.sha256_file
_id = v3._id
_digest = v3._digest
_copy = v3._copy
_strict = v3._strict
_rendered_object = v3._rendered_object
_atomic_publish = v3._atomic_publish
_relative = v3._relative

expected_work_units = v3.expected_work_units
load_measurement_authority = v3.load_measurement_authority
storage_preflight = v3.storage_preflight
verify_runtime_prefix_readonly = v3.verify_runtime_prefix_readonly
verify_scientific_data_identity = v3.verify_scientific_data_identity
verify_live_validation_identity = v3.verify_live_validation_identity
frozen_validation_metadata = v3.frozen_validation_metadata
frozen_validation_ids = v3.frozen_validation_ids
build_e2_completion = v3.build_e2_completion
publish_e2_completion = v3.publish_e2_completion
verify_e2_completion_artifact = v3.verify_e2_completion_artifact
build_e3_artifact = v3.build_e3_artifact
publish_e3_artifact = v3.publish_e3_artifact
verify_e3_artifact = v3.verify_e3_artifact
build_e4_artifact = v3.build_e4_artifact
publish_e4_artifact = v3.publish_e4_artifact
verify_e4_artifact = v3.verify_e4_artifact
_state_for_runtime = v3._state_for_runtime


def _aborted_local_campaign_binding() -> tuple[dict[str, Any], str]:
    """Read and authenticate the frozen local predecessor contract."""

    value, raw = _rendered_object(v3.V3_CONTRACT_PATH, "corrected-v3 predecessor contract")
    if value.get("contract_id") != (
        "g8econtractcorrectedv3-da3e1d32d5b826a5bfa06f0d7b7a9e3c1809026843633648d28a70a9437986a4"
    ):
        raise G8EV3SError("the frozen corrected-v3 predecessor contract identity differs")
    return dict(value), sha256_bytes(raw)


def aborted_local_campaign_id() -> str:
    contract, _ = _aborted_local_campaign_binding()
    return str(contract["campaign_id"])


def superseded_campaign_ids() -> set[str]:
    ids = {
        v2.ORIGINAL_CAMPAIGN_ID,
        v2.FIRST_CORRECTED_CAMPAIGN_ID,
    }
    value, _ = _rendered_object(v2.V2_CONTRACT_PATH, "v2 historical contract")
    ids.add(str(value["campaign_id"]))
    ids.add(aborted_local_campaign_id())
    return ids


def reject_superseded_campaign(campaign_id: str) -> None:
    v3.reject_superseded_campaign(campaign_id)
    if campaign_id == aborted_local_campaign_id():
        raise G8EV3SError(
            "the PARTIAL_OWNER_ABORTED_PROFILE_RELOCATION local campaign cannot execute again"
        )


def _successor_source_paths() -> tuple[tuple[str, str], ...]:
    return v3._source_paths() + (
        ("src/baseline/g8_e_corrected_v3s.py", "v3s_worker_successor_lifecycle"),
        ("tools/freeze_g8_e_v3s.py", "v3s_pre_data_freezer"),
        ("tools/run_g8_e_corrected_v3s.py", "v3s_owner_gated_runner"),
        ("tools/merge_g8_e_corrected_v3s.py", "v3s_e3_cli"),
        ("tools/aggregate_g8_e_corrected_v3s.py", "v3s_e4_cli"),
        ("tools/verify_g8_e_corrected_v3s.py", "v3s_lifecycle_verifier"),
        ("requirements-pascal.lock", "worker_python_cuda_dependency_lock"),
    )


def build_source_manifest(source_commit: str) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for path_text, role in _successor_source_paths():
        path = REPO_ROOT / path_text
        if not path.is_file():
            raise G8EV3SError(f"v3s source binding is missing: {path_text}")
        raw = path.read_bytes()
        entries.append({"path": path_text, "role": role, "bytes": len(raw), "sha256": sha256_bytes(raw)})
    seen = {entry["path"] for entry in entries}
    for entry in v2._direct_upstream_bindings() + v2._g1_bindings():
        if entry["path"] not in seen:
            entries.append(entry)
    body: dict[str, Any] = {
        "schema_version": V3S_SCHEMA_VERSION,
        "artifact_role": "g8_e_v3s_execution_source_manifest",
        "status": "FROZEN_PRE_DATA",
        "source_commit": source_commit,
        "source_entries": entries,
        "source_classes": {
            "code": [entry["path"] for entry in entries if entry["path"].startswith(("src/", "tools/"))],
            "scientific_data_identity": [
                "results/baseline/g8_e/e1_corrected/measurement_authority.json",
                "results/baseline/g8_e/e1_corrected/logical_measurement_mapping.json",
                "data/manifests/imagenette160.csv",
            ],
            "environment": [
                "requirements-pascal.lock",
                "src/config/execution_profiles.py",
                "src/env.py",
            ],
        },
        "runtime_outputs_excluded": [
            "results/baseline/g8_e/e2_confessor_successor/runtime/",
            "results/baseline/g8_e/e2_confessor_successor/e2_execution_authorization.json",
            "results/baseline/g8_e/e1_corrected_v3/runtime/",
        ],
    }
    body["source_manifest_id"] = _id(V3S_SOURCE_PREFIX, body)
    return body


def _load_am87_source_compatibility(source_entries: Sequence[Mapping[str, Any]]) -> set[str]:
    """Authenticate the exact post-campaign verifier/builder source pair."""

    try:
        raw = AM87_SOURCE_COMPATIBILITY_PATH.read_bytes()
        compatibility = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise G8EV3SError(f"cannot load AM-87 G8_E source compatibility: {exc}") from None
    required = {
        "schema_version", "artifact_role", "amendment", "timing", "classification",
        "entries", "linked_protocol_compatibility", "protected_boundary", "compatibility_id",
    }
    if not isinstance(compatibility, Mapping) or set(compatibility) != required:
        raise G8EV3SError("AM-87 G8_E source-compatibility schema differs")
    body = {key: child for key, child in compatibility.items() if key != "compatibility_id"}
    if compatibility["compatibility_id"] != "g8esourcecompat-" + sha256_bytes(canonical_json(body)):
        raise G8EV3SError("AM-87 G8_E source-compatibility ID differs")
    if (
        compatibility["schema_version"] != 1
        or compatibility["artifact_role"] != "g8_e_am87_post_campaign_source_compatibility"
        or compatibility["amendment"] != "AM-87"
        or compatibility["timing"] != "post_g8e_e7_pre_g8f_execution"
        or compatibility["classification"] != "post_campaign_historical_builder_and_verifier_only"
        or compatibility["protected_boundary"] != {
            "g8_c_changed": False,
            "g8_d_changed": False,
            "g8_e_changed": False,
            "g8_f_execution": 0,
            "pass_one_rerun": False,
            "pass_two": 0,
            "test_access": 0,
            "training": 0,
        }
    ):
        raise G8EV3SError("AM-87 G8_E source-compatibility boundary differs")
    linked = compatibility["linked_protocol_compatibility"]
    protocol_path = REPO_ROOT / "results/baseline/g8_f/am87_post_campaign_source_compatibility.json"
    try:
        protocol = json.loads(protocol_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise G8EV3SError(f"cannot load linked AM-87 protocol compatibility: {exc}") from None
    if linked != {
        "path": "results/baseline/g8_f/am87_post_campaign_source_compatibility.json",
        "compatibility_id": protocol.get("compatibility_id"),
        "sha256": sha256_file(protocol_path),
    }:
        raise G8EV3SError("AM-87 linked protocol compatibility differs")
    entries = compatibility["entries"]
    if not isinstance(entries, list) or len(entries) != 2:  # literal-ok: exact historical builder/verifier pair
        raise G8EV3SError("AM-87 G8_E source-compatibility entries differ")
    frozen = {str(entry.get("path")): entry for entry in source_entries}
    expected_paths = {"src/baseline/g8_d.py", "src/baseline/g8_e_corrected_v3s.py"}
    admitted: set[str] = set()
    for item in entries:
        fields = {
            "path", "kind", "archived_bytes", "archived_sha256", "current_bytes",
            "current_sha256", "scientific_execution_reachable", "justification",
        }
        if not isinstance(item, Mapping) or set(item) != fields:
            raise G8EV3SError("AM-87 G8_E source entry schema differs")
        path_text = item["path"]
        if path_text in admitted or path_text not in expected_paths or path_text not in frozen:
            raise G8EV3SError("AM-87 G8_E source path is duplicate or foreign")
        prior = frozen[path_text]
        current_path = REPO_ROOT / path_text
        if (
            item["archived_bytes"] != prior.get("bytes")
            or item["archived_sha256"] != prior.get("sha256")
            or not current_path.is_file()
            or item["current_bytes"] != current_path.stat().st_size
            or item["current_sha256"] != sha256_file(current_path)
            or item["scientific_execution_reachable"] is not False
            or not isinstance(item["justification"], str)
            or not item["justification"]
        ):
            raise G8EV3SError("AM-87 G8_E source entry bytes or scope differ")
        admitted.add(path_text)
    if admitted != expected_paths:
        raise G8EV3SError("AM-87 G8_E source compatibility is incomplete")
    kinds = {item["path"]: item["kind"] for item in entries}
    if kinds != {
        "src/baseline/g8_d.py": "post_d7_historical_contract_builder_only",
        "src/baseline/g8_e_corrected_v3s.py": "post_e7_source_verifier_compatibility_only",
    }:
        raise G8EV3SError("AM-87 G8_E source compatibility kinds differ")
    return admitted


def validate_source_manifest(value: Mapping[str, Any], *, verify_live_sources: bool = True) -> dict[str, Any]:
    required = {
        "schema_version", "artifact_role", "status", "source_commit", "source_entries",
        "source_classes", "runtime_outputs_excluded", "source_manifest_id",
    }
    if set(value) != required or value["schema_version"] != V3S_SCHEMA_VERSION:
        raise G8EV3SError("v3s source manifest schema differs")
    if value["artifact_role"] != "g8_e_v3s_execution_source_manifest" or value["status"] != "FROZEN_PRE_DATA":
        raise G8EV3SError("v3s source manifest role/status differs")
    body = {key: child for key, child in value.items() if key != "source_manifest_id"}
    if value["source_manifest_id"] != _id(V3S_SOURCE_PREFIX, body):
        raise G8EV3SError("v3s source manifest ID differs")
    source_commit = str(value["source_commit"])
    resolved = subprocess.run(
        ["git", "rev-parse", "--verify", f"{source_commit}^{{commit}}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if resolved.returncode != 0 or resolved.stdout.strip() != source_commit:
        raise G8EV3SError("v3s source commit is not an exact available Git commit")
    if verify_live_sources:
        compatibility: set[str] | None = None
        for item in value["source_entries"]:
            entry = _strict(item, ("path", "role", "bytes", "sha256"), "v3s source entry")
            path = REPO_ROOT / entry["path"]
            exact = path.is_file() and path.stat().st_size == entry["bytes"] and sha256_file(path) == entry["sha256"]
            if not exact:
                if compatibility is None:
                    compatibility = _load_am87_source_compatibility(value["source_entries"])
                if entry["path"] not in compatibility:
                    raise G8EV3SError(f"v3s frozen source drift: {entry['path']}")
            historical = subprocess.run(
                ["git", "show", f"{source_commit}:{entry['path']}"],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
            )
            if historical.returncode != 0 or len(historical.stdout) != entry["bytes"] or sha256_bytes(historical.stdout) != entry["sha256"]:
                raise G8EV3SError(f"v3s source entry is not bound to source_commit: {entry['path']}")
    return dict(value)


def build_storage_plan() -> dict[str, Any]:
    """Worker-measured plan reusing every frozen v3 planning number.

    Must be executed ON the worker host so the recorded free bytes, inode
    counts and atomic-publication peak describe the filesystem that will hold
    the production runtime.
    """

    plan = dict(v3.build_storage_plan())
    plan["artifact_role"] = "g8_e_v3s_compute_storage_plan"
    basis = dict(plan.get("basis", {}))
    basis["worker_successor_epoch"] = "g8_e_v3s"
    basis["complexity_evidence_reused_path"] = _relative(v3.V3_COMPLEXITY_PATH)
    plan["basis"] = basis
    return plan


def _validate_worker_profile_block(profile: Mapping[str, Any]) -> None:
    expected_keys = {
        "profile_id", "device", "config_hash", "lock_file", "lock_file_sha256",
        "opportunistic_host_change_forbidden", "profile_frozen_before_first_measurement",
        "sole_writer",
    }
    if set(profile) != expected_keys:
        raise G8EV3SError("v3s execution profile block schema differs")
    if profile["profile_id"] != SUCCESSOR_PROFILE_ID:
        raise G8EV3SError("v3s contract does not bind the qualified worker profile")
    configured = get(f"environment.execution_profiles.{SUCCESSOR_PROFILE_ID}")
    if profile["lock_file"] != str(configured["lock_file"]) or profile["lock_file_sha256"] != str(configured["lock_file_sha256"]):
        raise G8EV3SError("v3s profile lock binding differs from the qualified registry")
    device = str(profile["device"])
    if not device.startswith("cuda:") or not device[5:].isdigit():  # literal-ok: cuda device prefix length
        raise G8EV3SError("v3s profile device must be explicit cuda:N")


def build_relocation_provenance(*, worker_device: str | None = None, verify_preserved_runtime: bool = True) -> dict[str, Any]:
    """Freeze the custody record of the aborted local partial campaign.

    Executed ON the laptop host while the preserved runtime is attached; the
    frozen artifact then binds the chosen worker profile/device for the
    successor contract.  ``verify_preserved_runtime=False`` re-derives only the
    state-independent fields (worker side, where the preserved runtime is
    absent and this artifact is authenticated by its contract SHA-256 binding).
    """

    predecessor, predecessor_raw = _aborted_local_campaign_binding()
    if not isinstance(worker_device, str) or not worker_device.startswith("cuda:") or not worker_device[5:].isdigit():  # literal-ok: cuda device prefix length
        raise G8EV3SError("relocation provenance requires an explicit cuda:N worker device")
    configured = get(f"environment.execution_profiles.{SUCCESSOR_PROFILE_ID}")
    body: dict[str, Any] = {
        "schema_version": V3S_SCHEMA_VERSION,
        "artifact_role": "g8_e_v3s_relocation_provenance",
        "classification": "PARTIAL_OWNER_ABORTED_PROFILE_RELOCATION",
        "reason": (
            "Owner elected to relocate authoritative E2-E4 execution from the active "
            "development laptop to the dedicated qualified worker server confessor."
        ),
        "scientific_defect": "NONE",
        "selection_contribution": "ZERO",
        "superseded_before_data": False,
        "accepted_evidence_preserved": True,
        "used_by_worker_successor": False,
        "used_by_e3_e4_e5": False,
        "g8_c_rerun": False,
        "g8_d_rerun": False,
        "g1_retrain": False,
        "worker": {
            "host": str(configured["scientific_writer_host"]),
            "profile_id": SUCCESSOR_PROFILE_ID,
            "device": worker_device,
            "lock_file": str(configured["lock_file"]),
            "lock_file_sha256": str(configured["lock_file_sha256"]),
        },
        "worker_successor_rejection_rule": (
            "successor verification rejects any record whose campaign/profile identity is "
            "not exactly the successor campaign on confessor_pascal_cu126"
        ),
    }
    if verify_preserved_runtime:
        runtime_root = v3.V3_RUNTIME_ROOT
        state_path = runtime_root / "campaign_state.json"
        if not state_path.is_file():
            raise G8EV3SError("the preserved local partial runtime is missing its campaign state")
        state, state_raw = _rendered_object(state_path, "preserved local partial campaign state")
        authorization_path = v3.V3_AUTHORIZATION_PATH
        if not authorization_path.is_file():
            raise G8EV3SError("the preserved local campaign lacks its owner authorization")
        records_dir = runtime_root / "records"
        record_count = sum(1 for item in records_dir.iterdir()) if records_dir.is_dir() else 0
        forbidden = [v3.V3_E3_PATH, v3.V3_E4_PATH]
        present = [str(path) for path in forbidden if path.exists()]
        if present:
            raise G8EV3SError(f"the preserved local campaign unexpectedly contains lifecycle output(s): {present}")
        counters = {key: int(value) for key, value in state["counters"].items()}
        if counters.get("training", 0) or counters.get("test_access", 0):
            raise G8EV3SError("the preserved local campaign records a forbidden nonzero counter")

        def _tree_bytes(root: Path) -> int:
            total = 0
            stack = [str(root)]
            while stack:
                current = stack.pop()
                for entry in os.scandir(current):
                    if entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                    elif entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
            return total

        body["predecessor"] = {
            "campaign_id": predecessor["campaign_id"],
            "contract_id": predecessor["contract_id"],
            "contract_sha256": predecessor_raw,
            "profile_id": predecessor["execution_profile"]["profile_id"],
            "authorization_sha256": sha256_file(authorization_path),
            "runtime_path": _relative(runtime_root),
            "record_count": record_count,
            "completion_artifact_absent": not (runtime_root / "e2_completion.json").exists(),
            "e3_present": False,
            "e4_present": False,
            "runtime_bytes_at_stop": _tree_bytes(runtime_root),
            "state": {
                "schema_version": state["schema_version"],
                "completed_prefix_count": state["completed_prefix_count"],
                "total_required": state["total_required"],
                "authority_order_sha256": state["authority_order_sha256"],
                "rolling_prefix_digest": state["rolling_prefix_digest"],
                "last_completed_work_unit_id": state["last_completed_work_unit_id"],
                "counters": counters,
                "state_sha256": sha256_bytes(state_raw),
            },
        }
    else:
        body["predecessor_state_binding"] = "authenticated_by_contract_sha256_on_worker"
    body["relocation_provenance_id"] = _id("g8erelocationv3s-", {key: child for key, child in body.items() if key != "relocation_provenance_id"})
    return body


def verify_relocation_provenance(expected: Mapping[str, Any], *, verify_preserved_runtime: bool = True) -> dict[str, Any]:
    """Re-derive the relocation facts; laptop mode also re-reads the runtime."""

    body = build_relocation_provenance(
        worker_device=expected.get("worker", {}).get("device"),
        verify_preserved_runtime=verify_preserved_runtime,
    )
    if verify_preserved_runtime and dict(body) != dict(expected):
        raise G8EV3SError("v3s relocation provenance does not reproduce from the preserved runtime")
    if not verify_preserved_runtime:
        stripped = {key: child for key, child in body.items() if key != "predecessor_state_binding"}
        expected_stripped = {key: child for key, child in expected.items() if key != "predecessor"}
        if stripped != expected_stripped:
            raise G8EV3SError("v3s relocation provenance worker-side fields differ")
    return body


def build_contract(source: Mapping[str, Any], storage: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the frozen v3 science byte-identically; change relocation identity only."""

    v3_contract, v3_contract_sha = _aborted_local_campaign_binding()
    data_identity_path = REPO_ROOT / str(v3_contract["scientific_data_identity"]["path"])
    data_raw = data_identity_path.read_bytes()
    data_identity, _ = _rendered_object(data_identity_path, "v3 scientific data identity")
    if (
        data_identity.get("data_identity_id") != v3_contract["scientific_data_identity"]["id"]
        or sha256_bytes(data_raw) != v3_contract["scientific_data_identity"]["sha256"]
    ):
        raise G8EV3SError("the reused v3 scientific data identity does not match its frozen binding")
    relocation_path = REPO_ROOT / str(_relative(V3S_RELOCATION_PROVENANCE_PATH))
    if not relocation_path.is_file():
        raise G8EV3SError("v3s relocation provenance must be frozen before the successor contract")
    relocation_raw = relocation_path.read_bytes()
    relocation, _ = _rendered_object(relocation_path, "v3s relocation provenance")
    if relocation.get("relocation_provenance_id") != _id(
        "g8erelocationv3s-", {key: child for key, child in relocation.items() if key != "relocation_provenance_id"}
    ):
        raise G8EV3SError("v3s relocation provenance ID differs")

    seed = {
        "schema_version": V3S_SCHEMA_VERSION,
        "semantics_epoch": "g8_e_v3s_confessor_worker_relocation",
        "predecessor_campaign_id": v3_contract["campaign_id"],
        "predecessor_contract_sha256": v3_contract_sha,
        "predecessor_accepted_prefix_count": relocation["predecessor"]["state"]["completed_prefix_count"],
        "authority_id": v3_contract["authority"]["authority_id"],
        "mapping_id": v3_contract["mapping"]["mapping_id"],
        "data_identity_id": data_identity["data_identity_id"],
        "data_identity_sha256": sha256_bytes(data_raw),
        "source_manifest_id": source["source_manifest_id"],
        "source_manifest_sha256": sha256_bytes(rendered_json(source)),
        "storage_plan_sha256": sha256_bytes(rendered_json(storage)),
    }
    campaign_id = _id(V3S_CAMPAIGN_PREFIX, seed)

    body = copy.deepcopy({key: child for key, child in v3_contract.items() if key != "contract_id"})
    body["artifact_role"] = "g8_e_v3s_executable_pre_data_worker_successor_contract"
    body["checkpoint"] = "E1_corrected_v3_confessor_successor"
    body["status"] = "FROZEN_PRE_DATA_EXECUTABLE"
    body["campaign_id"] = campaign_id
    body["campaign_seed"] = seed
    supersedes = copy.deepcopy(body.get("supersedes_before_data", {}))
    supersedes["corrected_v3_local_partial"] = {
        "classification": "PARTIAL_OWNER_ABORTED_PROFILE_RELOCATION",
        "campaign_id": v3_contract["campaign_id"],
        "contract_id": v3_contract["contract_id"],
        "contract_sha256": v3_contract_sha,
        "accepted_prefix_count": relocation["predecessor"]["state"]["completed_prefix_count"],
        "total_required": relocation["predecessor"]["state"]["total_required"],
        "accepted_evidence_preserved": True,
        "used_by_successor": False,
        "successor_coverage_contribution": 0,
        "scientific_invalidation": "NONE",
        "relocation_provenance_id": relocation["relocation_provenance_id"],
        "relocation_provenance_sha256": sha256_bytes(relocation_raw),
    }
    body["supersedes_before_data"] = supersedes
    body["execution_profile"] = _copy(v3_contract["execution_profile"])
    body["execution_profile"]["profile_id"] = SUCCESSOR_PROFILE_ID
    configured = get(f"environment.execution_profiles.{SUCCESSOR_PROFILE_ID}")
    worker_device = relocation.get("worker", {}).get("device")
    if not isinstance(worker_device, str) or not worker_device.startswith("cuda:") or not worker_device[5:].isdigit():  # literal-ok: cuda device prefix length
        raise G8EV3SError("the worker device must be frozen in the relocation provenance first")
    if relocation.get("worker", {}).get("profile_id") != SUCCESSOR_PROFILE_ID:
        raise G8EV3SError("relocation provenance does not bind the qualified worker profile")
    body["execution_profile"]["device"] = worker_device
    body["execution_profile"]["lock_file"] = str(configured["lock_file"])
    body["execution_profile"]["lock_file_sha256"] = str(configured["lock_file_sha256"])
    body["execution_profile"]["sole_writer"] = str(configured["scientific_writer_host"])
    body["source_manifest"] = {
        "path": _relative(V3S_SOURCE_MANIFEST_PATH),
        "id": source["source_manifest_id"],
        "sha256": sha256_bytes(rendered_json(source)),
        "source_commit": source["source_commit"],
    }
    body["compute_plan"]["storage_path"] = _relative(V3S_STORAGE_PLAN_PATH)
    body["compute_plan"]["storage_sha256"] = sha256_bytes(rendered_json(storage))
    body["authorization"]["path"] = _relative(V3S_AUTHORIZATION_PATH)
    body["e3"]["artifact_path"] = _relative(V3S_E3_PATH)
    body["e4"]["artifact_path"] = _relative(V3S_E4_PATH)
    body["contract_id"] = _id(V3S_CONTRACT_PREFIX, body)
    return body


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    if (
        value.get("schema_version") != V3S_SCHEMA_VERSION
        or value.get("checkpoint") != "E1_corrected_v3_confessor_successor"
        or value.get("status") != "FROZEN_PRE_DATA_EXECUTABLE"
        or value.get("artifact_role") != "g8_e_v3s_executable_pre_data_worker_successor_contract"
    ):
        raise G8EV3SError("v3s contract is not the frozen worker-successor epoch")
    body = {key: child for key, child in value.items() if key != "contract_id"}
    if value.get("contract_id") != _id(V3S_CONTRACT_PREFIX, body):
        raise G8EV3SError("v3s contract ID differs")
    if value.get("campaign_id") in superseded_campaign_ids():
        raise G8EV3SError("a superseded or aborted campaign remains current")
    if value.get("authorization", {}).get("issued") is not False or value.get("safety", {}).get("measurement_coverage") != 0:
        raise G8EV3SError("the immutable v3s contract itself must remain pre-data")
    _validate_worker_profile_block(value.get("execution_profile", {}))
    return dict(value)


def verify_frozen_contract(*, verify_live_sources: bool = True, verify_live_data: bool = True) -> dict[str, Any]:
    """Phase-invariant successor verification valid before, during and after E2-E4."""

    contract, contract_raw = _rendered_object(V3S_CONTRACT_PATH, "v3s measurement contract")
    source, source_raw = _rendered_object(V3S_SOURCE_MANIFEST_PATH, "v3s source manifest")
    storage, storage_raw = _rendered_object(V3S_STORAGE_PLAN_PATH, "v3s storage plan")
    relocation, relocation_raw = _rendered_object(V3S_RELOCATION_PROVENANCE_PATH, "v3s relocation provenance")
    validate_source_manifest(source, verify_live_sources=verify_live_sources)
    validate_contract(contract)
    if contract["source_manifest"] != {
        "path": _relative(V3S_SOURCE_MANIFEST_PATH),
        "id": source["source_manifest_id"],
        "sha256": sha256_bytes(source_raw),
        "source_commit": source["source_commit"],
    }:
        raise G8EV3SError("v3s contract/source manifest binding differs")
    v3_contract, v3_contract_sha = _aborted_local_campaign_binding()
    supersedes = contract["supersedes_before_data"]["corrected_v3_local_partial"]
    if (
        supersedes["campaign_id"] != v3_contract["campaign_id"]
        or supersedes["contract_sha256"] != v3_contract_sha
        or supersedes["relocation_provenance_sha256"] != sha256_bytes(relocation_raw)
    ):
        raise G8EV3SError("v3s contract/predecessor relocation binding differs")
    data_identity_path = REPO_ROOT / str(contract["scientific_data_identity"]["path"])
    if data_identity_path.resolve() != v3.V3_DATA_IDENTITY_PATH.resolve():
        raise G8EV3SError("v3s contract must reuse the frozen v3 scientific data identity file")
    stored_data_identity, _ = _rendered_object(data_identity_path, "v3 data identity")
    data_body = {key: child for key, child in stored_data_identity.items() if key != "data_identity_id"}
    if stored_data_identity.get("data_identity_id") != _id(v3.V3_DATA_PREFIX, data_body):
        raise G8EV3SError("the reused scientific data identity ID differs")
    if contract["scientific_data_identity"]["sha256"] != sha256_file(data_identity_path):
        raise G8EV3SError("v3s contract/data identity binding differs")
    if verify_live_data:
        live = v3.build_scientific_data_identity(verify_archive_bytes=True)
        if live != stored_data_identity:
            raise G8EV3SError("v3s reused live scientific data identity drifted")
    required = int(storage.get("estimated_bytes", {}).get("required_with_safety_margin", -1))
    subtotal = int(storage.get("estimated_bytes", {}).get("subtotal", -1))
    margin = int(storage.get("estimated_bytes", {}).get("safety_margin_25_percent", -1))
    if required != subtotal + margin or margin != (subtotal + 3) // 4:  # literal-ok: exact integer ceiling for frozen 25% margin
        raise G8EV3SError("v3s storage plan arithmetic differs")
    runtime_estimate = storage.get("production_runtime_estimate", {})
    runtime_components = (
        "jpeg2000_physical_work_seconds", "reconstruction_seconds", "classifier_observations_seconds",
        "e2_transaction_seconds", "checkpoints_seconds", "resume_reconciliation_allowance_seconds",
        "final_e3_seconds", "e4_seconds", "filesystem_cache_allowance_seconds",
    )
    if (
        runtime_estimate.get("status") != "PASS_PLANNING_ESTIMATE_ONLY"
        or runtime_estimate.get("practical_on_frozen_local_profile") is not True
        or abs(sum(float(runtime_estimate.get(key, -1)) for key in runtime_components) - float(runtime_estimate.get("total_seconds", -2))) > 1e-6  # literal-ok: arithmetic serialization tolerance only
        or runtime_estimate.get("basis", {}).get("complexity_evidence_sha256") != sha256_file(v3.V3_COMPLEXITY_PATH)
    ):
        raise G8EV3SError("v3s production runtime estimate differs")
    if contract.get("compute_plan", {}).get("storage_sha256") != sha256_bytes(storage_raw):
        raise G8EV3SError("v3s contract/storage plan binding differs")
    if contract != build_contract(source, storage):
        raise G8EV3SError("v3s contract does not independently reproduce from frozen inputs")
    storage_preflight(storage, V3S_RUNTIME_ROOT)
    return {
        "contract": contract,
        "source_manifest": source,
        "storage_plan": storage,
        "relocation_provenance": relocation,
        "contract_sha256": sha256_bytes(contract_raw),
        "source_manifest_sha256": sha256_bytes(source_raw),
        "storage_plan_sha256": sha256_bytes(storage_raw),
        "relocation_provenance_sha256": sha256_bytes(relocation_raw),
    }


def verify_predata_zero_state(**kwargs: Any) -> dict[str, Any]:
    bundle = verify_frozen_contract(**kwargs)
    forbidden = [V3S_AUTHORIZATION_PATH, V3S_RUNTIME_ROOT, V3S_E2_COMPLETION_PATH, V3S_E3_PATH, V3S_E4_PATH]
    present = [str(path) for path in forbidden if path.exists()]
    if present:
        raise G8EV3SError(f"v3s pre-data zero state is closed by legitimate/foreign lifecycle artifacts: {present}")
    return {**bundle, "phase": "PRE_DATA_ZERO", "production_e2_records": 0, "production_e2_completed_units": 0}


def authenticate_owner_authorization(
    path: Path,
    contract: Mapping[str, Any],
    data_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if data_identity is None:
        data_identity_path = REPO_ROOT / str(contract["scientific_data_identity"]["path"])
        data_identity, _ = _rendered_object(data_identity_path, "v3 reused data identity")
    return v3.authenticate_owner_authorization_v3(Path(path), contract, data_identity)


def verify_active_e2(*, runtime_root: Path = V3S_RUNTIME_ROOT, authorization_path: Path = V3S_AUTHORIZATION_PATH, **kwargs: Any) -> dict[str, Any]:
    bundle = verify_frozen_contract(**kwargs)
    authorization = authenticate_owner_authorization(authorization_path, bundle["contract"])
    authority = load_measurement_authority()
    sample_ids, _ = frozen_validation_metadata(bundle["contract"]["scientific_data_identity"])
    state = verify_runtime_prefix_readonly(
        runtime_root=runtime_root,
        contract=bundle["contract"],
        authority=authority,
        sample_ids=sample_ids,
    )
    return {**bundle, "authorization": authorization, "state": state, "phase": "ACTIVE_E2"}


def verify_e2_complete(*, runtime_root: Path = V3S_RUNTIME_ROOT, **kwargs: Any) -> dict[str, Any]:
    active = verify_active_e2(runtime_root=runtime_root, **kwargs)
    observed, digest = verify_e2_completion_artifact(
        runtime_root=runtime_root,
        contract=active["contract"],
        authority=load_measurement_authority(),
        production=True,
    )
    return {**active, "completion": observed, "completion_sha256": digest, "phase": "E2_COMPLETE"}


def verify_e3_complete(*, e3_path: Path = V3S_E3_PATH, e3_sha256: str | None = None, **kwargs: Any) -> dict[str, Any]:
    runtime_root = Path(e3_path).parent
    complete = verify_e2_complete(runtime_root=runtime_root, **kwargs)
    value = verify_e3_artifact(e3_path, contract=complete["contract"], expected_sha256=e3_sha256)
    return {**complete, "e3": value, "e3_sha256": sha256_file(e3_path), "phase": "E3_COMPLETE"}


def verify_e4_complete(*, e4_path: Path = V3S_E4_PATH, e3_path: Path = V3S_E3_PATH, e3_sha256: str | None = None, **kwargs: Any) -> dict[str, Any]:
    e3_complete = verify_e3_complete(e3_path=e3_path, e3_sha256=e3_sha256, **kwargs)
    bound_e3_sha = e3_sha256 or sha256_file(e3_path)
    value = verify_e4_artifact(e4_path, contract=e3_complete["contract"], e3_path=e3_path, e3_sha256=bound_e3_sha)
    return {**e3_complete, "e4": value, "e4_sha256": sha256_file(e4_path), "phase": "E4_COMPLETE"}
