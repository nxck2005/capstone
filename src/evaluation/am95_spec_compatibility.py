"""Authenticate the additive AM-95 semantic freeze after terminal G-10.

AM-94 is historical evidence and must remain byte-identical.  AM-95 changes
only the executable description of the not-yet-run randomized ER-2 variant.
This module admits one exact successor view, chained from the AM-94 current
bytes, so historical W6--G-8 readers can continue to authenticate their own
records without treating the new semantics as a rewrite of old science.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from config.params import REPO_ROOT, get
from evaluation import g10_spec_compatibility as am94


PREDECESSOR_COMMIT = "342b85b653dbd3f237ea732a57f29e53528171b2"
FREEZE_RELATIVE_PATH = "results/learned/w9/am95_pre_science_freeze.json"
FREEZE_ID = "er2snrsemantics-2a3ca648c52fc318ec05ad5c4fe58c9530762dfd45d26ebfb4d68aea3d9715bd"
FREEZE_SHA256 = "b1b4702f1101dedb62a58e1a4e5827e41e7d235a9f8de8608410d208c21f79fd"

ALLOWED_PARAMETER_PATHS = (
    "artifacts.rng_identity_fields.er2_snr_randomised_v1",
    "artifacts.rng_purposes",
    "channel.train_snr_randomisation_distribution",
    "channel.train_snr_randomisation_rng_purpose",
    "channel.train_snr_randomisation_unit",
)

RANDOMISATION_DOMAIN = tuple(get("channel.train_snr_db_set"))
RNG_IDENTITY_FIELDS = (
    "dataset_version",
    "split_manifest_hash",
    "stable_sample_id",
    "train_seed",
    "epoch",
)

# Entries are (path, AM-94 current bytes, AM-94 current SHA, AM-95 bytes,
# AM-95 SHA).  The AM-94 values are imported from its immutable source module;
# the AM-95 values are the only new view frontier admitted here.
_CURRENT_VIEW_HASHES: dict[str, tuple[int, str]] = {
    "spec/SPEC.md": (410328, "d3bc33b591b48bf387650017eda45643085a5b49140879b408e423aa3167cc9c"),
    "spec/params.generated.yaml": (46914, "6f44c7e981dcb179984c5737d844c1b8fc0ac39c0fa29fb84b1a6c67c35cbeb1"),
    "spec/DATASHEET.md": (87808, "7933ef979817d8def28bdc4d094da2b0f21a53272a461adbc7b2c69e0f43433c"),
    "spec/concerns/amendments.md": (148178, "bf53321198e854996cbfc3e1270c5256352e8039d7884182ce9be8b09c8d005e"),
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
        next(item[3] for item in am94.VIEW_HASHES if item[0] == relative),  # literal-ok: authenticated view tuple field
        next(item[4] for item in am94.VIEW_HASHES if item[0] == relative),  # literal-ok: authenticated view tuple field
        current_bytes,
        current_sha,
    )
    for relative, (current_bytes, current_sha) in _CURRENT_VIEW_HASHES.items()
)


class AM95SpecCompatibilityError(RuntimeError):
    """The exact AM-95 successor or its pre-science boundary differs."""


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
        raise AM95SpecCompatibilityError(message)


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or unsafe")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AM95SpecCompatibilityError(f"cannot read {label}: {exc}") from None
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


def _verify_am94_predecessor(root: Path) -> None:
    value, raw = _read_json(
        root / am94.FREEZE_RELATIVE_PATH,
        "AM-94 predecessor freeze",
    )
    _require(raw == am94.rendered(value), "AM-94 predecessor is not canonical rendered JSON")
    _require(sha256_bytes(raw) == am94.FREEZE_SHA256, "AM-94 predecessor bytes differ")
    body = dict(value)
    identifier = body.pop("freeze_id", None)
    _require(
        identifier == am94.FREEZE_ID
        and identifier == "g10semantics-" + sha256_bytes(am94.canonical(body)),
        "AM-94 predecessor identity differs",
    )
    _require(value.get("entries") == am94.expected_entries(), "AM-94 predecessor entries differ")


def load(root: Path = REPO_ROOT, *, current_commit: str | None = None) -> dict[str, Any]:
    """Verify AM-95, optionally against its historical current source image.

    The optional commit is used only by the immutable AM-95 verifier after a
    later additive semantic successor has advanced the live source views.
    """

    root = Path(root).resolve()
    value, raw = _read_json(root / FREEZE_RELATIVE_PATH, "AM-95 pre-science freeze")
    _require(raw == rendered(value), "AM-95 freeze is not canonical rendered JSON")
    _require(FREEZE_SHA256 and sha256_bytes(raw) == FREEZE_SHA256, "AM-95 freeze file bytes differ")
    body = dict(value)
    identifier = body.pop("freeze_id", None)
    _require(
        FREEZE_ID
        and identifier == FREEZE_ID
        and identifier == "er2snrsemantics-" + sha256_bytes(canonical(body)),
        "AM-95 freeze identity differs",
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
            "sampling_contract",
            "g10_terminal_evidence",
            "scientific_boundary",
            "freeze_id",
        },
        "AM-95 freeze schema differs",
    )
    _require(
        value.get("schema_version") == 1
        and value.get("artifact_role") == "ER2_RANDOMISED_SNR_PRE_SCIENCE_SEMANTICS_FREEZE"
        and value.get("status") == "SEMANTICS_ONLY_FROZEN"
        and value.get("amendment") == "AM-95"
        and value.get("timing") == "post_g10_pre_er9_er2_training"
        and value.get("predecessor_commit") == PREDECESSOR_COMMIT
        and value.get("predecessor")
        == {
            "path": am94.FREEZE_RELATIVE_PATH,
            "freeze_id": am94.FREEZE_ID,
            "sha256": am94.FREEZE_SHA256,
        }
        and value.get("allowed_parameter_paths") == list(ALLOWED_PARAMETER_PATHS),
        "AM-95 freeze boundary differs",
    )
    _verify_am94_predecessor(root)
    _require(value.get("entries") == expected_entries(), "AM-95 source-view entries differ")
    for relative, base_bytes, base_sha, current_bytes, current_sha in VIEW_HASHES:
        predecessor = _git_bytes(root, PREDECESSOR_COMMIT, relative)
        current_path = root / relative
        if current_commit is None:
            _require(current_path.is_file() and not current_path.is_symlink(), f"AM-95 current view is missing: {relative}")
            current = current_path.read_bytes()
        else:
            current = _git_bytes(root, current_commit, relative)
        _require(
            len(predecessor) == base_bytes and sha256_bytes(predecessor) == base_sha,
            f"AM-95 predecessor view differs: {relative}",
        )
        _require(
            len(current) == current_bytes and sha256_bytes(current) == current_sha,
            f"AM-95 current view differs: {relative}",
        )
    old_params = yaml.safe_load(_git_bytes(root, PREDECESSOR_COMMIT, "spec/params.generated.yaml"))
    new_params_raw = (
        _git_bytes(root, current_commit, "spec/params.generated.yaml")
        if current_commit is not None
        else (root / "spec/params.generated.yaml").read_bytes()
    )
    new_params = yaml.safe_load(new_params_raw)
    _require(
        _leaf_differences(old_params, new_params) == set(ALLOWED_PARAMETER_PATHS),
        "AM-95 parameter drift exceeds the five named semantic leaves",
    )
    _require(
        value.get("sampling_contract")
        == {
            "domain_parameter": "params.channel.train_snr_db_set",
            "domain_db": list(RANDOMISATION_DOMAIN),
            "distribution": "discrete_uniform",
            "unit": "per_sample_per_epoch",
            "rng_purpose": "er2_snr_randomised_v1",
            "rng_identity_fields": list(RNG_IDENTITY_FIELDS),
            "channel_noise_is_separate_keyed_stream": True,
            "continuous_interpolation": False,
            "weighting": False,
            "curriculum": False,
            "batch_order_invariant": True,
            "batch_size_invariant": True,
            "worker_count_invariant": True,
            "gradient_accumulation_invariant": True,
            "resume_segmentation_invariant": True,
            "fixed_training_snr_unchanged": True,
        },
        "AM-95 sampling contract differs",
    )
    _require(
        value.get("g10_terminal_evidence")
        == {
            "completion_path": "results/learned/w9/w9a_completion.json",
            "completion_id": "w9acompletion-533df59159a9748b7c33d92ee7b3a87d2bfc066cb30ba3cb931cdf17f302bfc8",
            "completion_sha256": "ceeb790c5d910fb107e3b1081194abf97b0d9cb7182404b3b59a801e5effebd5",
            "reconciliation_path": "results/learned/w9/w9a_reconciliation.json",
            "reconciliation_id": "w9areconcile-fef8473e3ce676813d8b3212182d89171abe8ae75ee38e8a9957347d0bc92d1a",
            "reconciliation_sha256": "c6b805a0631df0e37a2fcd35e24026af8885772c44ad28a25a1dec73582fbae8",
            "scientific_source_commit": "9515c490aed4439f7ced2c163abef61557654ddf",
            "g10_source_sha256": "e332478bcac3d873b482ac97525a89ad4bb8a60219835e2a4b3c1938c23e24f4",
            "evaluations": 63,
            "reruns": 0,
            "classification": "expected_crossover_observed",
            "headline_bracket_db": [-5, -4],  # literal-ok: frozen G-10 measured bracket
        },
        "AM-95 G-10 terminal binding differs",
    )
    evidence = value["g10_terminal_evidence"]
    for key in ("completion", "reconciliation"):
        path = root / evidence[f"{key}_path"]
        actual, actual_raw = _read_json(path, f"G-10 {key}")
        _require(
            sha256_bytes(actual_raw) == evidence[f"{key}_sha256"]
            and actual.get(f"{key}_id" if key == "completion" else "reconciliation_id")
            == evidence[f"{key}_id"],
            f"AM-95 G-10 {key} evidence differs",
        )
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
        "AM-95 scientific boundary differs",
    )
    _require(not (root / "results/freeze_manifest.json").exists(), "test freeze manifest exists at AM-95")
    return value
