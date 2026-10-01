"""Authenticate the AM-100 low-rate secondary-variant specification epoch.

AM-100 adds one descriptive ER-9 variant (the closed Stage-1 D1024/b2 run sent
at BPSK rate 1/5) before G-12.  It adds parameter leaves and one system value
and changes no closed result, so this loader authenticates the current views,
then delegates every historical check to AM-99 in downstream mode.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from config.params import REPO_ROOT

AMENDMENT = "AM-100"
TIMING = "post_second_review_pre_g12_freeze"
PREDECESSOR_COMMIT = "aac5edebe0a85270441e54b913cf7c46a9a49789"

AM100_PARAMETER_PATHS = frozenset({
    "artifacts.system_values",
    "digital_semantic_control.low_rate_variant_checkpoint_epoch",
    "digital_semantic_control.low_rate_variant_checkpoint_run",
    "digital_semantic_control.low_rate_variant_checkpoint_sha256",
    "digital_semantic_control.low_rate_variant_entropy_model",
    "digital_semantic_control.low_rate_variant_ldpc_rate",
    "digital_semantic_control.low_rate_variant_modulation",
    "digital_semantic_control.low_rate_variant_payload_bits",
    "digital_semantic_control.low_rate_variant_quantiser_bits",
    "digital_semantic_control.low_rate_variant_retraining",
    "digital_semantic_control.low_rate_variant_role",
    "digital_semantic_control.low_rate_variant_seed_cell",
    "digital_semantic_control.low_rate_variant_system",
    "digital_semantic_control.low_rate_variant_test_scope",
    "digital_semantic_control.low_rate_variant_transmit_dim",
    "digital_semantic_control.low_rate_variant_transport_selection",
    "digital_semantic_control.low_rate_variant_validation_scope",
    "digital_semantic_control.low_rate_variant_width_rule",
})

_CURRENT_VIEW_HASHES: dict[str, tuple[int, str]] = {
    "spec/SPEC.md": (442279, "74b2cc3e5f18603b02168504f49312f7585452b3830543752ed51a74fe7fabf7"),
    "spec/params.generated.yaml": (50795, "1471825821cb3f5b51f615ed318e80fd882356bf73e234adf9d4b8ab1d2475a6"),
    "spec/DATASHEET.md": (93918, "ea461fee5c4154bcab76949467979f0b1517fc8ca3f4cf4423eb2345575fca25"),
    "spec/concerns/amendments.md": (167161, "4c9236465c5db047e5921b44eaacec378a3eb633a8ceb00e47ad1ba1e743afec"),
    "spec/concerns/baseline.md": (56600, "35025a9af82cdd203eca48bd92b3af524ac88d2f8cd6822813074ba009a81b69"),
    "spec/concerns/demo.md": (1803, "5c1cd9e80f6d5e8d1334169f64c5b1f84593a3a002fec4def40d687855128be3"),
    "spec/concerns/experiments.md": (37737, "830aa411123f5d20766822bf8abbe2545e4251156f90aebe68c445476249b2c4"),
    "spec/concerns/hardware.md": (3051, "5092f014751415746029bbb66fe94aa6c396ffa6a4b7e1b5dd02a0eceb5b7f43"),
    "spec/concerns/programme.md": (11927, "0cca702a96b541540b5ba23d24e10adfd39e04329424dd967a9135254be6ac16"),
    "spec/concerns/roadmap.md": (43452, "bc8c8c40a307d6f9c16de854a33b5804c13026c7357b9b7c02fa67a049170b91"),
    "spec/concerns/system.md": (39663, "2576f8179d6ec5762bff13c39f2354558fc20231760d9709c1326ee98cafb692"),
}

VIEW_HASHES: dict[str, tuple[int, str]] = dict(_CURRENT_VIEW_HASHES)


class AM100SpecCompatibilityError(RuntimeError):
    """The AM-100 epoch or its parameter delta differs."""


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AM100SpecCompatibilityError(message)


def current_views_match(root: Path = REPO_ROOT) -> bool:
    """True when every spec view is exactly the AM-100 image."""

    for relative, (expected_bytes, expected_sha) in _CURRENT_VIEW_HASHES.items():
        path = Path(root) / relative
        if not path.is_file() or path.is_symlink():
            return False
        raw = path.read_bytes()
        if len(raw) != expected_bytes or sha256_bytes(raw) != expected_sha:
            return False
    return True


def load(root: Path = REPO_ROOT, *, allow_downstream: bool = False) -> dict[str, Any]:
    """Authenticate the AM-100 views, its exact parameter delta and AM-99."""

    from evaluation import am97_spec_compatibility as am97  # noqa: PLC0415
    from evaluation import am98_spec_compatibility as am98  # noqa: PLC0415
    from evaluation import am99_spec_compatibility as am99  # noqa: PLC0415

    root = Path(root).resolve()
    if not current_views_match(root):
        _require(allow_downstream, "AM-100 current views differ")
    old_params = yaml.safe_load(am97._git_bytes(root, PREDECESSOR_COMMIT, "spec/params.generated.yaml"))
    new_params = yaml.safe_load((root / "spec/params.generated.yaml").read_bytes())
    differences = am98._leaf_differences(old_params, new_params)
    if allow_downstream:
        _require(differences >= AM100_PARAMETER_PATHS, "AM-100 parameters are not present in the current view")
    else:
        _require(
            differences == set(AM100_PARAMETER_PATHS),
            f"AM-100 parameter drift differs from its named leaves: {sorted(differences ^ set(AM100_PARAMETER_PATHS))}",
        )
    system_values = new_params["artifacts"]["system_values"]
    _require(
        [value for value in system_values if value != "er9_digital_low_rate"]
        == list(old_params["artifacts"]["system_values"]),
        "AM-100 may only add er9_digital_low_rate to artifacts.system_values",
    )
    am99.load(root, allow_downstream=True)
    return {
        "amendment": AMENDMENT,
        "timing": TIMING,
        "predecessor_commit": PREDECESSOR_COMMIT,
        "parameter_paths": sorted(AM100_PARAMETER_PATHS),
        "view_hash_count": len(_CURRENT_VIEW_HASHES),
    }


__all__ = [
    "AM100_PARAMETER_PATHS",
    "AM100SpecCompatibilityError",
    "AMENDMENT",
    "PREDECESSOR_COMMIT",
    "TIMING",
    "VIEW_HASHES",
    "current_views_match",
    "load",
]
