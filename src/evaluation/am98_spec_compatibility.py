"""Authenticate the prospective AM-98 W10/PAPR source epoch.

AM-98 changes only future W10/PAPR semantics and adds parameters; it does not
touch any closed G-10/ER-9/ER-2/G-11 result.  This successor loader authenticates
the new current spec views, binds the W10 downstream source manifest, and
delegates the historical AM-94..AM-97 chain in its downstream mode.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from config.params import REPO_ROOT
from evaluation import am97_spec_compatibility as am97
from runtime.source_epochs import (
    W10_MANIFEST_KIND,
    W10_V2_MANIFEST_KIND,
    W10_V3_MANIFEST_KIND,
    W10_V4_MANIFEST_KIND,
    W10_V5_MANIFEST_KIND,
    W10_V6_MANIFEST_KIND,
    active_manifest_path,
    load_w10_manifest,
)

SUCCESSOR_RELATIVE_PATH = "results/learned/w10/w10_downstream_source_manifest.json"
AMENDMENT = "AM-98"
TIMING = "post_g11_pre_w10_authority"

_CURRENT_VIEW_HASHES: dict[str, tuple[int, str]] = {
    "spec/SPEC.md": (431648, "51f428740e76125c58ea381b7c09c492ece7badd4e54d15424e1ca11b7e7e7f5"),
    "spec/params.generated.yaml": (49750, "37a5b93ea539cffb94ebcb9b28b79dae5811794cdd2c21c3a414fcc4f723efc0"),
    "spec/DATASHEET.md": (92168, "e7046d38339de5ccef82ed157f6af6ba44abcdcd60ca252b343c8ea97d46bee3"),
    "spec/concerns/amendments.md": (158540, "7f4d310eaba641f0820e9520498cc567fb6c13f84aea150cc0a17e9446dff72e"),
    "spec/concerns/baseline.md": (56600, "35025a9af82cdd203eca48bd92b3af524ac88d2f8cd6822813074ba009a81b69"),
    "spec/concerns/demo.md": (1803, "5c1cd9e80f6d5e8d1334169f64c5b1f84593a3a002fec4def40d687855128be3"),
    "spec/concerns/experiments.md": (35199, "30b66f6132045a2349a72822a14f236546991cdaf98973fd2267765b810dd84e"),
    "spec/concerns/hardware.md": (3051, "5092f014751415746029bbb66fe94aa6c396ffa6a4b7e1b5dd02a0eceb5b7f43"),
    "spec/concerns/programme.md": (11927, "0cca702a96b541540b5ba23d24e10adfd39e04329424dd967a9135254be6ac16"),
    "spec/concerns/roadmap.md": (43452, "bc8c8c40a307d6f9c16de854a33b5804c13026c7357b9b7c02fa67a049170b91"),
    "spec/concerns/system.md": (39663, "2576f8179d6ec5762bff13c39f2354558fc20231760d9709c1326ee98cafb692"),
}

VIEW_HASHES: dict[str, tuple[int, str]] = dict(_CURRENT_VIEW_HASHES)

AM98_PARAMETER_PATHS = {
    "evaluation.w10_scope_version",
    "evaluation.w10_validation_denominator",
    "evaluation.w10_learned_ratios",
    "evaluation.w10_per_image_required",
    "evaluation.w10_analysis_version_policy",
    "evaluation.w10_papr_protocol_version",
    "evaluation.w10_papr_cap_db",
    "evaluation.w10_papr_training_runs",
    "evaluation.w10_er12_protocol_version",
    "evaluation.w10_er12_label_bits",
    "evaluation.w10_er12_payload_frame",
    "evaluation.w10_er12_declared_role",
}

SEMANTIC_CONTRACT = {
    "w10_scope_arms": 12,  # literal-ok: AM-98 frozen arm count
    "w10_derived_units": 252,
    "w10_validation_denominator": 1000,  # literal-ok: frozen Imagenette-160 validation size
    "w10_learned_ratios": ["r_1_6", "r_1_24"],
    "papr_cap_db": 3.0,
    "papr_domain": "symbol_domain_not_oversampled_waveform",
    "papr_training_runs": 1,
    "er12_label_bits": 4,  # literal-ok: AM-98 label field width
    # The scientific rule: ``classical_finetune_scored`` is NOT a separate W10
    # physical arm.  Artifact-finetuned scoring is the primary scorer stream of
    # ``classical_adaptive``; the clean scorer is a secondary stream over the
    # same physical reconstruction and channel realisation (AM-98, clarified by
    # AM-99).
    "classical_finetune_scored_is_a_separate_w10_physical_arm": False,
    "analysis_version": 2,
    "analysis_version_policy": "retain_2_no_estimand_or_definition_change",
    "test": "SEALED",
    "test_access": 0,
}


def verify_semantic_contract() -> None:
    """Cross-check every declared AM-98 semantic against the live scope/params."""

    from config.params import get
    from evaluation.w10_scope import SCOPE, snr_grid, unit_count, validate_scope
    from pathlib import Path as _Path

    try:
        from evaluation.w10_scope import W10_VALIDATION_DENOMINATOR
    except ImportError:  # pragma: no cover - the constant is always present
        W10_VALIDATION_DENOMINATOR = int(get("evaluation.w10_validation_denominator"))
    validate_scope()
    _require(len(SCOPE) == SEMANTIC_CONTRACT["w10_scope_arms"], "AM-98 scope arm count differs")
    _require(unit_count() == SEMANTIC_CONTRACT["w10_derived_units"], "AM-98 derived unit count differs")
    _require(len(snr_grid()) == 21, "AM-98 SNR grid cardinality differs")  # literal-ok: frozen SNR grid cardinality
    _require(
        int(get("evaluation.w10_validation_denominator")) == SEMANTIC_CONTRACT["w10_validation_denominator"]
        and W10_VALIDATION_DENOMINATOR == SEMANTIC_CONTRACT["w10_validation_denominator"],
        "AM-98 validation denominator differs",
    )
    _require(list(get("evaluation.w10_learned_ratios")) == SEMANTIC_CONTRACT["w10_learned_ratios"], "AM-98 learned ratios differ")
    _require(float(get("evaluation.w10_papr_cap_db")) == SEMANTIC_CONTRACT["papr_cap_db"], "AM-98 PAPR cap differs")
    _require(int(get("evaluation.w10_papr_training_runs")) == SEMANTIC_CONTRACT["papr_training_runs"], "AM-98 PAPR run count differs")
    _require(int(get("evaluation.w10_er12_label_bits")) == SEMANTIC_CONTRACT["er12_label_bits"], "AM-98 ER-12 label width differs")
    _require(
        int(get("config.analysis_version")) == SEMANTIC_CONTRACT["analysis_version"],
        "AM-98 analysis version differs from the live config analysis_version",
    )
    _require(
        str(get("evaluation.w10_analysis_version_policy")) == SEMANTIC_CONTRACT["analysis_version_policy"],
        "AM-98 analysis-version policy differs from the live parameter",
    )
    _require(
        "classical_finetune_scored" not in {entry.system for entry in SCOPE},
        "classical_finetune_scored must not be a separate W10 physical arm",
    )
    for entry in SCOPE:
        _require(entry.denominator == SEMANTIC_CONTRACT["w10_validation_denominator"], "AM-98 arm denominator differs")
        _require(entry.per_image_required is True, "AM-98 arm is not per-image required")
        _require(entry.bw_ratio in ("r_1_6", "r_1_24"), "AM-98 arm ratio is outside the learned set")
        _require(entry.system.count("finetune") == 0 or entry.system == "classical_adaptive", "AM-98 arm names a finetune duplicate")


class AM98SpecCompatibilityError(RuntimeError):
    """The AM-98 source epoch or its successor binding differs."""


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AM98SpecCompatibilityError(message)


def _leaf_differences(old: Any, new: Any, prefix: str = "") -> set[str]:
    differences: set[str] = set()
    if isinstance(old, dict) and isinstance(new, dict):
        for key in set(old) | set(new):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in old or key not in new:
                differences.add(path)
            else:
                differences |= _leaf_differences(old[key], new[key], path)
    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            differences.add(prefix)
        else:
            for index, (left, right) in enumerate(zip(old, new, strict=True)):
                differences |= _leaf_differences(left, right, f"{prefix}.{index}")
    elif old != new:
        differences.add(prefix)
    return differences


def load(root: Path = REPO_ROOT, *, allow_downstream: bool = False) -> dict[str, Any]:
    """Authenticate the AM-98 views and bind the W10 successor source manifest."""

    root = Path(root).resolve()
    successor_path = root / SUCCESSOR_RELATIVE_PATH
    _require(successor_path.is_file() and not successor_path.is_symlink(), "AM-98 historical successor source manifest is missing")
    manifest = load_w10_manifest(root, live=True)
    _require(manifest.get("manifest_kind") in {W10_MANIFEST_KIND, W10_V2_MANIFEST_KIND, W10_V3_MANIFEST_KIND, W10_V4_MANIFEST_KIND, W10_V5_MANIFEST_KIND, W10_V6_MANIFEST_KIND}, "AM-98 successor manifest kind differs")
    _require("w10_validation_rehearsal" in manifest.get("governs", []), "AM-98 successor does not govern W10")
    for relative, (expected_bytes, expected_sha) in _CURRENT_VIEW_HASHES.items():
        path = root / relative
        _require(path.is_file() and not path.is_symlink(), f"AM-98 view is missing: {relative}")
        current = path.read_bytes()
        if not (len(current) == expected_bytes and sha256_bytes(current) == expected_sha):
            # AM-98 is a superseded pre-science epoch once a later amendment
            # advances the frontier; downstream mode tolerates exactly that
            # named successor, while strict mode still authenticates the
            # AM-98-era view bytes.
            _require(allow_downstream, f"AM-98 current view differs: {relative}")
    verify_semantic_contract()
    old_params = yaml.safe_load(am97._git_bytes(root, am97.PREDECESSOR_COMMIT, "spec/params.generated.yaml"))
    new_params = yaml.safe_load((root / "spec/params.generated.yaml").read_bytes())
    differences = _leaf_differences(old_params, new_params)
    _require(
        differences <= set(am97.ALLOWED_PARAMETER_PATHS) | AM98_PARAMETER_PATHS,
        f"AM-98 parameter drift exceeds its named leaves: {sorted(differences - set(am97.ALLOWED_PARAMETER_PATHS) - AM98_PARAMETER_PATHS)}",
    )
    _require(AM98_PARAMETER_PATHS <= differences, "AM-98 parameters are not present in the current view")
    am97.load(root, allow_downstream=True)
    active_path = active_manifest_path(root)
    return {
        "amendment": AMENDMENT,
        "timing": TIMING,
        "successor_manifest": {
            "path": str(active_path.relative_to(root)),
            "manifest_id": manifest["manifest_id"],
            "sha256": sha256_bytes(active_path.read_bytes()),
            "source_commit": manifest["source_commit"],
        },
        "view_hash_count": len(_CURRENT_VIEW_HASHES),
        "semantic_contract": dict(SEMANTIC_CONTRACT),
    }


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


__all__ = [
    "AM98_PARAMETER_PATHS",
    "AM98SpecCompatibilityError",
    "AMENDMENT",
    "SEMANTIC_CONTRACT",
    "SUCCESSOR_RELATIVE_PATH",
    "TIMING",
    "VIEW_HASHES",
    "load",
    "verify_semantic_contract",
]
