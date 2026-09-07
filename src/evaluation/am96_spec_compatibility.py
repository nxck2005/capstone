"""Authenticate the prospective AM-96 ER-9 semantic freeze.

AM-96 is an additive source-view successor to AM-95.  It closes the complete
known P1-7 ER-9 reproducibility inventory before any ER-9 optimizer step or
randomized ER-2 run.  The compatibility boundary is intentionally strict:
the predecessor AM-95 image, generated views, parameter leaf set, audit note,
G-10 terminal evidence and zero-science counters all have to match.
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
from evaluation import am95_spec_compatibility as am95


PREDECESSOR_COMMIT = "2096b1f7e1572448dca26ca2d9f4c8ba9d2bec32"
FREEZE_RELATIVE_PATH = "results/learned/w9/am96_pre_science_freeze.json"
FREEZE_ID = "er9control-5572ebb2670ab3434afbf068156e3553cc1a04076bd7be2b8ff14d482d1d4f4d"
FREEZE_SHA256 = "7f8a8ab929120b24d6e5a1658ebadf962467a4f299217a96d0ebebe95dcb8699"
AUDIT_RELATIVE_PATH = "audit/er9-p1-7-am96.md"
AUDIT_SHA256 = "40d1b73e2ff23f58cf1fc04b939601c0202119d5a3e0bf1cc44037bcc5fb30ae"

ALLOWED_PARAMETER_PATHS = (
    "digital_semantic_control.admissibility_budget_floor",
    "digital_semantic_control.admissibility_metadata_bits",
    "digital_semantic_control.dimension_scope",
    "digital_semantic_control.entropy_padding_rule",
    "digital_semantic_control.entropy_stream_length_field",
    "digital_semantic_control.factorisation",
    "digital_semantic_control.factorisation_channel_rule",
    "digital_semantic_control.factorisation_flatten_order",
    "digital_semantic_control.factorisation_pool_height",
    "digital_semantic_control.factorisation_pool_width",
    "digital_semantic_control.final_seed_scope",
    "digital_semantic_control.post_interface_task_path",
    "digital_semantic_control.pre_interface_tap",
    "digital_semantic_control.quantiser_clipping",
    "digital_semantic_control.quantiser_levels",
    "digital_semantic_control.quantiser_range",
    "digital_semantic_control.quantiser_scale",
    "digital_semantic_control.quantiser_scale_side_information_bits",
    "digital_semantic_control.raw_bit_order",
    "digital_semantic_control.reconstruction_head",
    "digital_semantic_control.search_seed_cell",
    "digital_semantic_control.stage1_dimension_order",
    "digital_semantic_control.stage1_quantiser_bits",
    "digital_semantic_control.stage1_selection_metric",
    "digital_semantic_control.stage1_tie_break",
    "digital_semantic_control.stage2_bits_order",
    "digital_semantic_control.stage2_reuse_rule",
    "digital_semantic_control.stage2_selection_metric",
    "digital_semantic_control.stage2_tie_break",
    "digital_semantic_control.training_lambda",
    "digital_semantic_control.training_loss",
    "digital_semantic_control.training_transport",
)

_CURRENT_VIEW_HASHES: dict[str, tuple[int, str]] = {
    "spec/SPEC.md": (417758, "5be83f36265947999db5e73b3cf40b55d6ee92922da251918a09af483bc67fa5"),
    "spec/params.generated.yaml": (48735, "0faee4aa473a2c3ac1ea9bfaa7f25216ac643878756efb8a81e1d8d281810ae8"),
    "spec/DATASHEET.md": (90717, "5f0f22aefc6f04d934c9f96662486ebff717b10fb56933f45d70902471a9c0a4"),
    "spec/concerns/amendments.md": (150297, "ed7ee05d2159bff4044de9fa4fd376aee6b69743a410420193d89d3f0906bb40"),
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
        next(item[3] for item in am95.VIEW_HASHES if item[0] == relative),
        next(item[4] for item in am95.VIEW_HASHES if item[0] == relative),  # literal-ok: authenticated AM-95 current-view tuple field
        current_bytes,
        current_sha,
    )
    for relative, (current_bytes, current_sha) in _CURRENT_VIEW_HASHES.items()
)

SEMANTIC_CONTRACT: dict[str, Any] = {
    "pre_interface_tap": "encoder_residual_trunk_output_before_complex_projection",
    "post_interface_task_path": "dequantized_feature_tensor_to_image_classification_head",
    "reconstruction_head": "none",
    "training_loss": "cross_entropy",
    "training_lambda": "not_applicable",
    "factorisation": {
        "mechanism": "output_channel_count_and_adaptive_pooling",
        "channel_rule": "D_div_64",
        "pool_height": 8,  # literal-ok: AM-96 canonical adaptive-pool geometry
        "pool_width": 8,  # literal-ok: AM-96 canonical adaptive-pool geometry
        "flatten_order": "channel_major_row_major_column_major_nchw_contiguous",
        "inverse_reshape": "C_x_8_x_8_contiguous_nchw",
        "exact_for_configured_grid": True,
        "trunk_channel_capacity": 128,  # literal-ok: AM-96 djscc_residual_v1 trunk capacity
    },
    "dimension_scope": "one_global_pair_per_bandwidth_ratio_frozen_across_snr",
    "quantiser": {
        "family": "uniform_scalar",
        "range_operation": "tanh",
        "range": [-1, 1],
        "levels": "2_to_b_inclusive_endpoint_uniform",
        "training": "straight_through_estimator_identity_rounding_true_tanh_derivative",
        "scale": "fixed_protocol_no_image_adaptive_scale",
        "clipping": "tanh_to_protocol_range",
        "scale_side_information_bits": 0,
    },
    "training_transport": "ste_quantise_dequantise_task_path_without_digital_channel_backpropagation",
    "search": {
        "seed_cell": "first_zipped_seed_pair",
        "checkpoint_selection": {
            "split": "validation",
            "metric": "top1_accuracy",
            "mode": "max",
            "tie_break": "earliest_epoch",
        },
        "admissibility": {
            "payload_floor": "exact_bpsk_rate_1_3_packetisation_payload_at_matched_k",
            "metadata_bits": "framing_selector_bits_plus_am96_required_metadata",
        },
        "stage1": {
            "bits": 2,
            "candidate_order": "ascending_numeric_dimension",
            "metric": "exact_full_validation_n_correct_at_fixed_7_db_real_digital_chain",
            "tie_break": "smallest_transmit_dim",
        },
        "stage2": {
            "candidate_order": "ascending_numeric_bits",
            "reuse": "stage1_selected_dimension_at_stage1_bits",
            "metric": "exact_full_validation_n_correct_at_fixed_7_db_real_digital_chain",
            "tie_break": "smallest_quantiser_bits",
            "new_training_count": "feasible_width_count_minus_one",
        },
        "cross_product": False,
        "counts": "stage1_authorized_before_training_stage2_authorized_after_stage1_selection",
    },
    "entropy_and_framing": {
        "coder": "static_range_coder",
        "model": "fitted_offline_on_train_split",
        "learned_model_permitted": False,
        "table_bytes_counted": False,
        "raw_escape_required": True,
        "branch": "shorter_of_range_coded_and_fixed_width_raw",
        "selector_bits": 1,
        "stream_length_field": "none_decode_exact_D_symbols",
        "padding": "ignored_after_exact_symbol_count",
        "raw_bit_order": "most_significant_bit_first",
        "all_per_message_metadata_counted": True,
    },
    "production": {
        "ratio": "r_1_6",
        "seed_scope": "all_three_existing_zipped_seed_cells",
        "search_run_promotion": "exact_source_config_seed_epoch_checkpoint_identity_only",
    },
}


class AM96SpecCompatibilityError(RuntimeError):
    """The exact AM-96 successor or its pre-science boundary differs."""


def canonical(value: Any) -> bytes:
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
        raise AM96SpecCompatibilityError(message)


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or unsafe")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AM96SpecCompatibilityError(f"cannot read {label}: {exc}") from None
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    return value, raw


def _git_bytes(root: Path, commit: str, relative: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=root,
        capture_output=True,
        check=False,
    )
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
        {
            "path": relative,
            "base_bytes": base_bytes,
            "base_sha256": base_sha,
            "current_bytes": current_bytes,
            "current_sha256": current_sha,
        }
        for relative, base_bytes, base_sha, current_bytes, current_sha in VIEW_HASHES
    ]


def _verify_am95_predecessor(root: Path) -> dict[str, Any]:
    return am95.load(root, current_commit=PREDECESSOR_COMMIT)


def load(root: Path = REPO_ROOT, *, allow_downstream: bool = False) -> dict[str, Any]:
    """Verify AM-96 and its pre-science boundary.

    The immutable source-preparation records are allowed to appear below
    ``results/learned/er9`` before the first optimizer step.  Once scientific
    records exist, callers that are explicitly verifying the post-freeze
    project state pass ``allow_downstream=True``; the AM-96 bytes and its
    protected counters remain unchanged either way.
    """

    root = Path(root).resolve()
    value, raw = _read_json(root / FREEZE_RELATIVE_PATH, "AM-96 pre-science freeze")
    _require(raw == rendered(value), "AM-96 freeze is not canonical rendered JSON")
    _require(FREEZE_SHA256 and sha256_bytes(raw) == FREEZE_SHA256, "AM-96 freeze file bytes differ")
    body = dict(value)
    identifier = body.pop("freeze_id", None)
    _require(
        FREEZE_ID
        and identifier == FREEZE_ID
        and identifier == "er9control-" + sha256_bytes(canonical(body)),
        "AM-96 freeze identity differs",
    )
    _require(
        set(value)
        == {
            "schema_version",
            "artifact_role",
            "status",
            "amendment",
            "timing",
            "predecessor_commit",
            "predecessor",
            "allowed_parameter_paths",
            "entries",
            "audit",
            "semantic_contract",
            "g10_terminal_evidence",
            "scientific_boundary",
            "freeze_id",
        },
        "AM-96 freeze schema differs",
    )
    _require(
        value.get("schema_version") == 1
        and value.get("artifact_role") == "ER9_EXECUTABLE_SEMANTICS_PRE_SCIENCE_FREEZE"
        and value.get("status") == "SEMANTICS_ONLY_FROZEN"
        and value.get("amendment") == "AM-96"
        and value.get("timing") == "post_am95_pre_er9_er2_training"
        and value.get("predecessor_commit") == PREDECESSOR_COMMIT
        and value.get("predecessor")
        == {
            "path": am95.FREEZE_RELATIVE_PATH,
            "freeze_id": am95.FREEZE_ID,
            "sha256": am95.FREEZE_SHA256,
        }
        and value.get("allowed_parameter_paths") == list(ALLOWED_PARAMETER_PATHS),
        "AM-96 freeze boundary differs",
    )
    predecessor = _verify_am95_predecessor(root)
    _require(value.get("entries") == expected_entries(), "AM-96 source-view entries differ")
    for relative, base_bytes, base_sha, current_bytes, current_sha in VIEW_HASHES:
        predecessor_bytes = _git_bytes(root, PREDECESSOR_COMMIT, relative)
        _require(
            len(predecessor_bytes) == base_bytes and sha256_bytes(predecessor_bytes) == base_sha,
            f"AM-96 predecessor view differs: {relative}",
        )
        current_path = root / relative
        _require(current_path.is_file() and not current_path.is_symlink(), f"AM-96 current view is missing: {relative}")
        current = current_path.read_bytes()
        _require(
            len(current) == current_bytes and sha256_bytes(current) == current_sha,
            f"AM-96 current view differs: {relative}",
        )
    old_params = yaml.safe_load(_git_bytes(root, PREDECESSOR_COMMIT, "spec/params.generated.yaml"))
    new_params = yaml.safe_load((root / "spec/params.generated.yaml").read_bytes())
    _require(
        _leaf_differences(old_params, new_params) == set(ALLOWED_PARAMETER_PATHS),
        "AM-96 parameter drift exceeds the named ER-9 semantic leaves",
    )
    audit_path = root / AUDIT_RELATIVE_PATH
    _require(
        audit_path.is_file() and not audit_path.is_symlink() and sha256_bytes(audit_path.read_bytes()) == AUDIT_SHA256,
        "AM-96 P1-7 audit bytes differ",
    )
    _require(value.get("audit") == {"path": AUDIT_RELATIVE_PATH, "sha256": AUDIT_SHA256, "closed_items": 16}, "AM-96 audit binding differs")  # literal-ok: P1-7 inventory has sixteen named items
    _require(value.get("semantic_contract") == SEMANTIC_CONTRACT, "AM-96 semantic contract differs")
    _require(
        value.get("scientific_boundary")
        == {
            "g10_model_facing_evaluations": 63,
            "g10_reruns": 0,
            "er9_training": 0,
            "er2_randomized_training": 0,
            "g11": 0,
            "w10": 0,
            "learned_test_inference": 0,
            "model_facing_test_access": 0,
            "test_split": "SEALED",
        },
        "AM-96 scientific boundary differs",
    )
    _require(
        value.get("g10_terminal_evidence")
        == {
            "completion_path": "results/learned/w9/w9a_completion.json",
            "completion_id": predecessor["g10_terminal_evidence"]["completion_id"],
            "completion_sha256": predecessor["g10_terminal_evidence"]["completion_sha256"],
            "reconciliation_path": "results/learned/w9/w9a_reconciliation.json",
            "reconciliation_id": predecessor["g10_terminal_evidence"]["reconciliation_id"],
            "reconciliation_sha256": predecessor["g10_terminal_evidence"]["reconciliation_sha256"],
            "scientific_source_commit": "9515c490aed4439f7ced2c163abef61557654ddf",
            "g10_source_sha256": "e332478bcac3d873b482ac97525a89ad4bb8a60219835e2a4b3c1938c23e24f4",
            "evaluations": 63,
            "reruns": 0,
            "classification": "expected_crossover_observed",
            "headline_bracket_db": [-5, -4],  # literal-ok: frozen G-10 measured bracket
        },
        "AM-96 G-10 terminal binding differs",
    )
    evidence = value["g10_terminal_evidence"]
    completion, completion_raw = _read_json(root / evidence["completion_path"], "G-10 completion")
    reconciliation, reconciliation_raw = _read_json(root / evidence["reconciliation_path"], "G-10 reconciliation")
    _require(
        completion.get("completion_id") == evidence["completion_id"]
        and sha256_bytes(completion_raw) == evidence["completion_sha256"]
        and reconciliation.get("reconciliation_id") == evidence["reconciliation_id"]
        and sha256_bytes(reconciliation_raw) == evidence["reconciliation_sha256"],
        "AM-96 G-10 evidence bytes differ",
    )
    _require(not (root / "results/freeze_manifest.json").exists(), "test freeze manifest exists at AM-96")
    w9_root = root / "results/learned/w9"
    allowed_w9 = {
        "am94_pre_science_freeze.json",
        "am95_pre_science_freeze.json",
        "am96_pre_science_freeze.json",
        "g10_adjudication.json",
        "g10_cell_index.json",
        "g10_classical_adaptive_r1_6_extract.json",
        "g10_execution_authorization.json",
        "g10_execution_authorization_v2.json",
        "g10_headline_curve.json",
        "g10_runtime_manifest.json",
        "g10_source_manifest.json",
        "g10_source_manifest_v2.json",
        "w9a_completion.json",
        "w9a_reconciliation.json",
    }
    actual_w9 = {
        path.relative_to(w9_root).as_posix()
        for path in w9_root.glob("**/*")
        if path.is_file()
    }
    _require(actual_w9 <= allowed_w9, f"ER-9/G-11 artifact exists before AM-96: {sorted(actual_w9 - allowed_w9)}")
    er9_root = root / "results/learned/er9"
    if er9_root.exists():
        allowed_pre_science = {
            "er_execution_source_manifest.json",
            "er9_stage1_execution_authorization.json",
            "er_execution_source_manifest_v2.json",
            "er9_stage1_execution_authorization_v2.json",
            "er_execution_source_manifest_v3.json",
            "er9_stage1_execution_authorization_v3.json",
        }
        actual_er9 = {
            path.relative_to(er9_root).as_posix()
            for path in er9_root.glob("**/*")
            if path.is_file()
        }
        _require(
            allow_downstream or actual_er9 <= allowed_pre_science,
            "scientific ER-9 result directory exists at AM-96 pre-science boundary",
        )
    _require(not (root / "results/learned/er2_randomized").exists(), "randomized ER-2 result directory exists at AM-96")
    return value
