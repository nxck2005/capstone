"""Authenticate the corrected AM-99 W10/PAPR pre-science source epoch.

AM-99 corrects the AM-98 analysis-version statement and the prospective
W10/PAPR implementation without changing any closed result.  This loader is the
strict current-epoch view authenticator; AM-98 remains the historical record
and is authenticated through its own downstream mode.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from config.params import REPO_ROOT
from evaluation import am98_spec_compatibility as am98
from runtime.source_epochs import (
    W10_V2_MANIFEST_KIND,
    W10_V3_MANIFEST_KIND,
    W10_V4_MANIFEST_KIND,
    W10_V5_MANIFEST_KIND,
    W10_V6_MANIFEST_KIND,
    W10_V7_MANIFEST_KIND,
    W10_V8_MANIFEST_KIND,
    W10_V9_MANIFEST_KIND,
    active_manifest_path,
    assert_successor_lineage,
    load_w10_manifest,
)

AMENDMENT = "AM-99"
TIMING = "post_am98_pre_papr_authority_corrective"
PREDECESSOR_COMMIT = "8b90a8b95b0f600199b0e83f5136e743f6d389a8"
AM99_ANALYSIS_VERSION = 2

_CURRENT_VIEW_HASHES: dict[str, tuple[int, str]] = {
    "spec/SPEC.md": (437035, "14e97bb11e8a596514dd06f11d7a0d56c2a431ba1945eca7b1548290213edb83"),
    "spec/params.generated.yaml": (49750, "45165a333775c6e089cb22cd70b1e08ef0bca920ac40ec8b59b3f5c374c09c69"),
    "spec/DATASHEET.md": (92182, "8b39745134854c7c1349f40b2569c64ad3be309ca27790a30bb19ca8fe3cf237"),
    "spec/concerns/amendments.md": (163927, "909e1c092ea935a54a01398361ba908c840bdbc8354dbf30d25da64b960de8fb"),
    "spec/concerns/baseline.md": (56600, "35025a9af82cdd203eca48bd92b3af524ac88d2f8cd6822813074ba009a81b69"),
    "spec/concerns/demo.md": (1803, "5c1cd9e80f6d5e8d1334169f64c5b1f84593a3a002fec4def40d687855128be3"),
    "spec/concerns/experiments.md": (35199, "30b66f6132045a2349a72822a14f236546991cdaf98973fd2267765b810dd84e"),
    "spec/concerns/hardware.md": (3051, "5092f014751415746029bbb66fe94aa6c396ffa6a4b7e1b5dd02a0eceb5b7f43"),
    "spec/concerns/programme.md": (11927, "0cca702a96b541540b5ba23d24e10adfd39e04329424dd967a9135254be6ac16"),
    "spec/concerns/roadmap.md": (43452, "bc8c8c40a307d6f9c16de854a33b5804c13026c7357b9b7c02fa67a049170b91"),
    "spec/concerns/system.md": (39663, "2576f8179d6ec5762bff13c39f2354558fc20231760d9709c1326ee98cafb692"),
}

VIEW_HASHES: dict[str, tuple[int, str]] = dict(_CURRENT_VIEW_HASHES)


class AM99SpecCompatibilityError(RuntimeError):
    """The AM-99 corrected epoch or its live semantics differ."""


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AM99SpecCompatibilityError(message)


def load(root: Path = REPO_ROOT, *, allow_downstream: bool = False) -> dict[str, Any]:
    """Authenticate the corrected AM-99 views and their live semantic contract."""

    root = Path(root).resolve()
    manifest = load_w10_manifest(root, live=True)
    if not allow_downstream:
        _require(
            manifest.get("manifest_kind") in {W10_V2_MANIFEST_KIND, W10_V3_MANIFEST_KIND, W10_V4_MANIFEST_KIND, W10_V5_MANIFEST_KIND, W10_V6_MANIFEST_KIND, W10_V7_MANIFEST_KIND, W10_V8_MANIFEST_KIND, W10_V9_MANIFEST_KIND},
            "AM-99 requires the active successor-v2 through successor-v9 source epoch",
        )
    if manifest.get("manifest_kind") == W10_V9_MANIFEST_KIND:
        historical_v8 = load_w10_manifest(root, live=False, epoch="v8")
        assert_successor_lineage(root, historical_v8, successor=manifest)
    _require("w10_validation_rehearsal" in manifest.get("governs", []), "AM-99 successor does not govern W10")
    for relative, (expected_bytes, expected_sha) in _CURRENT_VIEW_HASHES.items():
        path = root / relative
        _require(path.is_file() and not path.is_symlink(), f"AM-99 view is missing: {relative}")
        current = path.read_bytes()
        if not (len(current) == expected_bytes and sha256_bytes(current) == expected_sha):
            _require(allow_downstream, f"AM-99 current view differs: {relative}")
    am98.verify_semantic_contract()
    _require(int(am98.SEMANTIC_CONTRACT["analysis_version"]) == AM99_ANALYSIS_VERSION, "AM-99 analysis version contract differs")
    am98.load(root, allow_downstream=True)
    old_params = yaml.safe_load(am98.am97._git_bytes(root, am98.am97.PREDECESSOR_COMMIT, "spec/params.generated.yaml"))
    new_params = yaml.safe_load((root / "spec/params.generated.yaml").read_bytes())
    differences = am98._leaf_differences(old_params, new_params)
    from evaluation.am97_spec_compatibility import ALLOWED_PARAMETER_PATHS as AM97_PARAMETER_PATHS

    allowed = set(am98.AM98_PARAMETER_PATHS) | set(AM97_PARAMETER_PATHS)
    _require(differences <= allowed, "AM-99 parameter drift exceeds the AM-97/AM-98 named leaves")
    active_path = active_manifest_path(root)
    return {
        "amendment": AMENDMENT,
        "timing": TIMING,
        "predecessor_commit": PREDECESSOR_COMMIT,
        "successor_manifest": {
            "path": str(active_path.relative_to(root)),
            "manifest_id": manifest["manifest_id"],
            "sha256": sha256_bytes(active_path.read_bytes()),
            "source_commit": manifest["source_commit"],
            "kind": manifest["manifest_kind"],
        },
        "analysis_version": AM99_ANALYSIS_VERSION,
        "view_hash_count": len(_CURRENT_VIEW_HASHES),
        "semantic_contract": dict(am98.SEMANTIC_CONTRACT),
    }


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


__all__ = [
    "AM99_ANALYSIS_VERSION",
    "AM99SpecCompatibilityError",
    "AMENDMENT",
    "TIMING",
    "VIEW_HASHES",
    "load",
]
