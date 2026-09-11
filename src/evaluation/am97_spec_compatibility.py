"""Authenticate the narrow prospective AM-97 H4 precision freeze.

AM-97 is deliberately a semantic successor to AM-96.  It changes only the
interpretation and reproducible inputs of the validation-only H4 precision
diagnostic.  The loader has explicit pre-science and downstream/history modes:
the former rejects later science, while the latter still authenticates the
immutable AM-96/AM-97 views and delegates downstream result validation to its
own verifiers.
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
from evaluation import am96_spec_compatibility as am96


PREDECESSOR_COMMIT = "62762fe"
FREEZE_RELATIVE_PATH = "results/learned/w9/am97_pre_science_freeze.json"
AUDIT_RELATIVE_PATH = "audit/er9-am97-h4-precision-diagnostic.md"
ALLOWED_PARAMETER_PATHS = (
    "evaluation.h4_precision_diagnostic_role",
    "evaluation.h4_precision_seed_aggregation",
    "evaluation.h4_precision_bootstrap_unit",
    "evaluation.h4_precision_evaluation_region",
    "evaluation.h4_precision_bootstrap_resamples",
    "evaluation.h4_precision_reference_pp",
    "evaluation.h4_precision_interpretation",
)

_CURRENT_VIEW_HASHES: dict[str, tuple[int, str]] = {
    "spec/SPEC.md": (422329, "0b13ec55d8bcd2e74350d8e9992c3c480a543328f55e9be4a6b65a895aeb53a8"),
    "spec/params.generated.yaml": (49256, "229b16889d18f1341d6a724b2f87454d99e32badb2b9079d7030334bd398ba90"),
    "spec/DATASHEET.md": (91378, "739726a1d506e82fdfc8b624518d2330fd511620470e42e2c0a7d3b65dbc6238"),
    "spec/concerns/amendments.md": (152253, "220bba39e562444dee4abbad9e6e4d03e7de1de9c63eaef7590ec1d7d1dcb00f"),
    "spec/concerns/baseline.md": (56342, "4c8d63c900a5dbf7b7adfee0fc3d9b67218fb555897bff03f73f37474b8d827f"),
    "spec/concerns/demo.md": (1803, "5c1cd9e80f6d5e8d1334169f64c5b1f84593a3a002fec4def40d687855128be3"),
    "spec/concerns/experiments.md": (35073, "4893dedf63493443ae82f1c0aa694c1daba5cce3c2effb424200fe1cc68f2d50"),
    "spec/concerns/hardware.md": (3051, "5092f014751415746029bbb66fe94aa6c396ffa6a4b7e1b5dd02a0eceb5b7f43"),
    "spec/concerns/programme.md": (11927, "0cca702a96b541540b5ba23d24e10adfd39e04329424dd967a9135254be6ac16"),
    "spec/concerns/roadmap.md": (43452, "bc8c8c40a307d6f9c16de854a33b5804c13026c7357b9b7c02fa67a049170b91"),
    "spec/concerns/system.md": (37338, "aecc3b1806cc163543212d7970d0f9bb2bbb68e939309c597706401e72bcbde5"),
}

VIEW_HASHES: tuple[tuple[str, int, str, int, str], ...] = tuple(
    (
        relative,
        next(item[3] for item in am96.VIEW_HASHES if item[0] == relative),
        next(item[4] for item in am96.VIEW_HASHES if item[0] == relative),  # literal-ok: historical view tuple field index
        current_bytes,
        current_sha,
    )
    for relative, (current_bytes, current_sha) in _CURRENT_VIEW_HASHES.items()
)


class AM97SpecCompatibilityError(RuntimeError):
    """The AM-97 semantic freeze or its lifecycle boundary differs."""


V4_SOURCE_MANIFEST_RELATIVE_PATH = "er_execution_source_manifest_v4.json"
V4_STAGE1_AUTHORITY_RELATIVE_PATH = "er9_stage1_execution_authorization_v4.json"
V4_PRE_SCIENCE_AUTHORITY_STATUS = "FROZEN_STAGE1_ONLY_PRE_SCIENCE"


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def rendered(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AM97SpecCompatibilityError(message)


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or unsafe")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AM97SpecCompatibilityError(f"cannot read {label}: {exc}") from None
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    return value, raw


def _authenticate_v4_pre_science_custody(root: Path) -> None:
    """Classify v4 preparation records without counting planned work as work.

    The Stage-1 authority records the six *authorized future* candidates.  It
    is not evidence that those candidates have trained.  AM-97's strict mode
    therefore admits the pair only when both records authenticate as the
    frozen pre-science contract, all execution counters are zero, and the
    authority-derived scientific runtime is absent.  Actual epoch terminals
    remain outside this boundary and are rejected by the surrounding path
    allow-list.
    """

    er9_root = root / "results/learned/er9"
    source_path = er9_root / V4_SOURCE_MANIFEST_RELATIVE_PATH
    authority_path = er9_root / V4_STAGE1_AUTHORITY_RELATIVE_PATH
    source_present = source_path.exists() or source_path.is_symlink()
    authority_present = authority_path.exists() or authority_path.is_symlink()
    if not source_present and not authority_present:
        return
    _require(source_present and authority_present, "incomplete W9 v4 pre-science custody")

    source, source_raw = _read_json(source_path, "W9 v4 source manifest")
    _require(not source_path.is_symlink(), "W9 v4 source manifest is unsafe")
    source_body = dict(source)
    source_id = source_body.pop("manifest_id", None)
    _require(
        source_id == "er9sourcev4-" + sha256_bytes(canonical(source_body)),
        "W9 v4 source manifest identity differs",
    )
    _require(
        source.get("schema_version") == 2
        and source.get("manifest_kind") == "W9_V4_FULL_SCIENTIFIC_SOURCE_CLOSURE"
        and source.get("source_commit_comparison") == "exact_clean_HEAD_at_freeze",
        "W9 v4 source manifest is not a pre-science closure",
    )
    _require(source_raw == rendered(source), "W9 v4 source manifest is not canonical rendered JSON")

    authority, authority_raw = _read_json(authority_path, "W9 v4 Stage-1 authority")
    _require(not authority_path.is_symlink(), "W9 v4 Stage-1 authority is unsafe")
    authority_body = dict(authority)
    authority_id = authority_body.pop("authority_id", None)
    _require(
        authority_id == "w9er9stage1v4auth-" + sha256_bytes(canonical(authority_body)),
        "W9 v4 Stage-1 authority identity differs",
    )
    _require(authority_raw == rendered(authority), "W9 v4 Stage-1 authority is not canonical rendered JSON")
    _require(
        authority.get("schema_version") == 1
        and authority.get("authority_kind") == "W9_ER9_STAGE1_EXECUTION_AUTHORITY_V4"
        and authority.get("status") == V4_PRE_SCIENCE_AUTHORITY_STATUS
        and authority.get("authorization_scope") == "W9_ER9_STAGE1_ONLY",
        "W9 v4 Stage-1 authority is not a pre-science authority",
    )
    _require(
        authority.get("source_manifest")
        == {
            "path": "results/learned/er9/er_execution_source_manifest_v4.json",
            "manifest_id": source_id,
            "sha256": sha256_bytes(source_raw),
        },
        "W9 v4 Stage-1 source manifest binding differs",
    )
    _require(
        authority.get("source_commit") == source.get("source_commit")
        and authority.get("source_binding") == source,
        "W9 v4 Stage-1 source closure binding differs",
    )
    _require(
        authority.get("runtime_root") == "checkpoints/er9_pascal_v4"
        and authority.get("execution_profile_id") == "confessor_pascal_cu126"
        and authority.get("host") == "confessor"
        and authority.get("device") == "cuda:0"
        and authority.get("sole_writer") is True
        and authority.get("fresh_initialization_required") is True,
        "W9 v4 Stage-1 execution boundary differs",
    )
    _require(
        isinstance(authority.get("stage1_candidates"), list)
        and isinstance(authority.get("stage1_candidate_count"), int)
        and authority.get("stage1_candidate_count") > 0
        and authority.get("stage1_training_count") == authority.get("stage1_candidate_count")
        and len(authority["stage1_candidates"]) == authority["stage1_candidate_count"],
        "W9 v4 Stage-1 planned candidate count differs",
    )
    _require(
        authority.get("stage2_authorized") is False
        and authority.get("production_authorized") is False
        and authority.get("randomized_er2_authorized") is False,
        "W9 v4 authority scope is wider than Stage-1",
    )
    _require(
        authority.get("test") == "SEALED" and authority.get("test_access") == 0,
        "W9 v4 Stage-1 test boundary differs",
    )
    _require(
        authority.get("pre_execution_counters")
        == {
            "new_er9_v4_training": 0,
            "randomized_er2_scientific_training": 0,
            "g11": 0,
            "w10": 0,
            "learned_test_inference": 0,
            "model_facing_test_access": 0,
        },
        "W9 v4 pre-execution counters are not zero",
    )
    runtime = root / str(authority["runtime_root"])
    _require(not runtime.exists() and not runtime.is_symlink(), "W9 v4 Stage-1 runtime exists at the AM-97 boundary")


def _git_bytes(root: Path, commit: str, relative: str) -> bytes:
    result = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=root, capture_output=True, check=False)
    _require(result.returncode == 0, f"cannot resolve historical bytes: {relative}")
    return result.stdout


def _leaf_differences(old: Any, new: Any, prefix: str = "") -> set[str]:
    if isinstance(old, Mapping) and isinstance(new, Mapping):
        result: set[str] = set()
        for key in set(old) | set(new):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in old or key not in new:
                result.add(child)
            else:
                result.update(_leaf_differences(old[key], new[key], child))
        return result
    return set() if old == new else {prefix}


def expected_entries() -> list[dict[str, Any]]:
    return [
        {"path": relative, "base_bytes": base_bytes, "base_sha256": base_sha, "current_bytes": current_bytes, "current_sha256": current_sha}
        for relative, base_bytes, base_sha, current_bytes, current_sha in VIEW_HASHES
    ]


def _boundary() -> dict[str, Any]:
    return {
        "g10_model_facing_evaluations": 63,
        "g10_reruns": 0,
        "new_er9_v4_training": 0,
        "er9_training": 0,
        "er2_randomized_training": 0,
        "g11": 0,
        "w10": 0,
        "learned_test_inference": 0,
        "model_facing_test_access": 0,
        "test_split": "SEALED",
    }


def load(root: Path = REPO_ROOT, *, allow_downstream: bool = False) -> dict[str, Any]:
    root = Path(root).resolve()
    value, raw = _read_json(root / FREEZE_RELATIVE_PATH, "AM-97 pre-science freeze")
    _require(raw == rendered(value), "AM-97 freeze is not canonical rendered JSON")
    body = dict(value)
    identifier = body.pop("freeze_id", None)
    _require(isinstance(identifier, str) and identifier == "er9precision-" + sha256_bytes(canonical(body)), "AM-97 freeze identity differs")
    _require(value.get("schema_version") == 1 and value.get("artifact_role") == "ER9_AM97_H4_POINTWISE_PRECISION_PRE_SCIENCE_FREEZE", "AM-97 role differs")
    _require(value.get("status") == "SEMANTICS_ONLY_FROZEN" and value.get("amendment") == "AM-97" and value.get("timing") == "post_am96_pre_er9_real_chain_observation", "AM-97 timing differs")
    _require(value.get("predecessor_commit") == PREDECESSOR_COMMIT, "AM-97 predecessor commit differs")
    _require(value.get("predecessor") == {"path": am96.FREEZE_RELATIVE_PATH, "freeze_id": am96.FREEZE_ID, "sha256": am96.FREEZE_SHA256}, "AM-97 predecessor binding differs")
    _require(value.get("allowed_parameter_paths") == list(ALLOWED_PARAMETER_PATHS), "AM-97 parameter allow-list differs")
    _require(value.get("entries") == expected_entries(), "AM-97 source-view entries differ")
    _require(value.get("scientific_boundary") == _boundary(), "AM-97 scientific boundary differs")
    _require(value.get("semantic_contract") == {
        "role": "pointwise_precision_diagnostic_only",
        "learned_arm": "ordinary_learned_w8_g10",
        "comparator": "er9_digital",
        "cells": [[0, 0], [1, 1], [2, 2]],
        "aggregation": "signed_correctness_difference_then_mean_within_stable_image",
        "bootstrap_unit": "stable_image_complete_three_cell_trajectory",
        "bootstrap_resamples": 10000,  # literal-ok: AM-97 freezes the prospective resample count
        "reference_pp": 2,
        "full_h4_power_certified": False,
        "h4_run_calibration_represented": False,
        "correlated_snr_probabilities_may_be_multiplied": False,
        "negative_h4_exclusion": False,
        "randomized_er2_comparator": False,
        "test_access": 0,
    }, "AM-97 semantic contract differs")
    _require(value.get("entries") == expected_entries(), "AM-97 entries differ")
    # Authenticate AM-96 as a historical predecessor.  Its downstream mode
    # permits this additive AM-97 successor but retains all AM-96 hashes.
    am96.load(root, allow_downstream=True)
    old_params = yaml.safe_load(_git_bytes(root, PREDECESSOR_COMMIT, "spec/params.generated.yaml"))
    new_params = yaml.safe_load((root / "spec/params.generated.yaml").read_bytes())
    _require(_leaf_differences(old_params, new_params) == set(ALLOWED_PARAMETER_PATHS), "AM-97 parameter drift exceeds its named leaves")
    for relative, base_bytes, base_sha, current_bytes, current_sha in VIEW_HASHES:
        predecessor = _git_bytes(root, PREDECESSOR_COMMIT, relative)
        _require(len(predecessor) == base_bytes and sha256_bytes(predecessor) == base_sha, f"AM-97 predecessor view differs: {relative}")
        path = root / relative
        _require(path.is_file() and not path.is_symlink(), f"AM-97 view missing: {relative}")
        current = path.read_bytes()
        _require(len(current) == current_bytes and sha256_bytes(current) == current_sha, f"AM-97 current view differs: {relative}")
    audit_path = root / AUDIT_RELATIVE_PATH
    _require(audit_path.is_file() and sha256_bytes(audit_path.read_bytes()) == value["audit"]["sha256"], "AM-97 audit differs")
    _require(value.get("g10_terminal_evidence") == {
        "source": "9515c490aed4439f7ced2c163abef61557654ddf",
        "evaluations": 63,
        "reruns": 0,
        "classification": "expected_crossover_observed",
        "headline_bracket_db": [-5, -4],  # literal-ok: frozen G-10 headline bracket
    }, "AM-97 G-10 binding differs")
    w9_root = root / "results/learned/w9"
    allowed_w9 = {"am94_pre_science_freeze.json", "am95_pre_science_freeze.json", "am96_pre_science_freeze.json", "am97_pre_science_freeze.json", "g10_adjudication.json", "g10_cell_index.json", "g10_classical_adaptive_r1_6_extract.json", "g10_execution_authorization.json", "g10_execution_authorization_v2.json", "g10_headline_curve.json", "g10_runtime_manifest.json", "g10_source_manifest.json", "g10_source_manifest_v2.json", "w9a_completion.json", "w9a_reconciliation.json", "w9_pascal_v4_lifecycle_smoke.json"}
    actual_w9 = {path.relative_to(w9_root).as_posix() for path in w9_root.glob("**/*") if path.is_file()}
    _require(actual_w9 <= allowed_w9, f"unexpected W9 artifact at AM-97 boundary: {sorted(actual_w9 - allowed_w9)}")
    er9_root = root / "results/learned/er9"
    allowed_pre = {
        "er_execution_source_manifest.json",
        "er9_stage1_execution_authorization.json",
        "er_execution_source_manifest_v2.json",
        "er9_stage1_execution_authorization_v2.json",
        "er_execution_source_manifest_v3.json",
        "er9_stage1_execution_authorization_v3.json",
        V4_SOURCE_MANIFEST_RELATIVE_PATH,
        V4_STAGE1_AUTHORITY_RELATIVE_PATH,
        "incidents/v3_local_profile_owner_supersession.json",
    }
    actual_er9 = {path.relative_to(er9_root).as_posix() for path in er9_root.glob("**/*") if path.is_file()} if er9_root.exists() else set()
    if not allow_downstream:
        _authenticate_v4_pre_science_custody(root)
    _require(allow_downstream or actual_er9 <= allowed_pre, "downstream ER-9 result exists at strict AM-97 boundary")
    if not allow_downstream:
        _require(not (root / "results/learned/er2_randomized").exists(), "randomized ER-2 exists at strict AM-97 boundary")
        _require(not (root / "results/learned/g11").exists(), "G-11 exists at strict AM-97 boundary")
    return value


__all__ = ["ALLOWED_PARAMETER_PATHS", "FREEZE_RELATIVE_PATH", "load", "canonical", "rendered", "sha256_bytes"]
