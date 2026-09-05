"""Authenticate the additive AM-94 pre-science normative source transition."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from config.params import REPO_ROOT


PREDECESSOR_COMMIT = "7fc415e2debeda61e1cc95049596c2eac46062b1"
FREEZE_RELATIVE_PATH = "results/learned/w9/am94_pre_science_freeze.json"
FREEZE_ID = "g10semantics-e55bacf7372135cb0b49500d1def4ee0037b67169e8aed775c0c51552e33f6ff"
FREEZE_SHA256 = "b3d7239d4887d100707633daf5c7ee769ee225c2a78bc97900031316c1a64563"
PRIOR_COMPATIBILITY_PATH = "results/learned/w7/w8_spec_additive_compatibility.json"
PRIOR_COMPATIBILITY_ID = "w8speccompat-0d2184356d3adf0a339b68ab9d1435e45f8105f1b0e8614ee37470d34dbb3436"
PRIOR_COMPATIBILITY_SHA256 = "94eea47b7f4e244ee4d97a1207fb49e7c123d6d0c244e1a214d135bcb8c6a05c"

ALLOWED_PARAMETER_PATHS = (
    "evaluation.g10_classical_comparator",
    "evaluation.g10_exact_arithmetic",
    "evaluation.g10_expected_direction",
    "evaluation.g10_fixed_profile_role",
    "evaluation.g10_gap_orientation",
    "evaluation.g10_headline_event",
    "evaluation.g10_interpolation",
    "evaluation.g10_learned_aggregation",
    "evaluation.g10_learned_cells",
    "evaluation.g10_multiple_event_rule",
    "evaluation.g10_ratio",
    "evaluation.g10_seed_spread",
    "evaluation.g10_zero_rule",
)

VIEW_HASHES: tuple[tuple[str, int, str, int, str], ...] = (
    ("spec/SPEC.md", 401661, "b05a2f04d6b3fa0e8110d0544900c29f823c96c424a75090a092433bb72cc68b", 407696, "75af7748f17245cb7771fd8e7078b506ce557bf920c7986bd49bd034aab8b6ad"),
    ("spec/params.generated.yaml", 45731, "c5f598e4e9292f831279bed5584cd399c42cc769b6550f8921a7bf94d7e20234", 46590, "48e5fee874fe4b6c72ee60be3b92e0cd9da51edab87e74bc5f3af964c6f9e534"),
    ("spec/DATASHEET.md", 86257, "c5b9b78270099bd909a8248cf490ff00307ecb61672a98e5b1d29894340f7e68", 87382, "1e8cdef23ce80598af806d53ffc95e1b7b1b3057982feb3ce4133c7c0af38617"),
    ("spec/concerns/amendments.md", 144957, "12191834c4a76005243f7e0846d446e55cf934a44728fbac93dc0aeebab7b697", 146618, "8920d40befd8140e7671570dffc6f3af5680482e7370826c3045d6453faaa8b1"),
    ("spec/concerns/baseline.md", 56342, "4c8d63c900a5dbf7b7adfee0fc3d9b67218fb555897bff03f73f37474b8d827f", 56342, "4c8d63c900a5dbf7b7adfee0fc3d9b67218fb555897bff03f73f37474b8d827f"),
    ("spec/concerns/demo.md", 1803, "5c1cd9e80f6d5e8d1334169f64c5b1f84593a3a002fec4def40d687855128be3", 1803, "5c1cd9e80f6d5e8d1334169f64c5b1f84593a3a002fec4def40d687855128be3"),
    ("spec/concerns/experiments.md", 34282, "a9bda259769d9fba3ef581610ef3c00ee3d9684358f31f2d55e732b1f95b31ff", 34282, "a9bda259769d9fba3ef581610ef3c00ee3d9684358f31f2d55e732b1f95b31ff"),
    ("spec/concerns/hardware.md", 3051, "5092f014751415746029bbb66fe94aa6c396ffa6a4b7e1b5dd02a0eceb5b7f43", 3051, "5092f014751415746029bbb66fe94aa6c396ffa6a4b7e1b5dd02a0eceb5b7f43"),
    ("spec/concerns/programme.md", 11927, "0cca702a96b541540b5ba23d24e10adfd39e04329424dd967a9135254be6ac16", 11927, "0cca702a96b541540b5ba23d24e10adfd39e04329424dd967a9135254be6ac16"),
    ("spec/concerns/roadmap.md", 40175, "a6c79a0a2bfc1f7edaef5f7e739cbf6fc14188cdaaf539894922546ec08c2ee2", 43452, "bc8c8c40a307d6f9c16de854a33b5804c13026c7357b9b7c02fa67a049170b91"),
    ("spec/concerns/system.md", 37315, "7db78c26ff492d54b4bd434f5007b94dc2aa88ae79c5c4c7f789dc501461208d", 37315, "7db78c26ff492d54b4bd434f5007b94dc2aa88ae79c5c4c7f789dc501461208d"),
)

# Exact W8 source-manifest entries whose carrier code/view bytes changed only
# to admit and verify the additive AM-94 transition.  The frozen W8 source
# manifest remains immutable; this projection never authorizes execution.
W8_CARRIER_SOURCE_HASHES: tuple[tuple[str, int, str, int, str], ...] = (
    ("spec/SPEC.md", 401661, "b05a2f04d6b3fa0e8110d0544900c29f823c96c424a75090a092433bb72cc68b", 407696, "75af7748f17245cb7771fd8e7078b506ce557bf920c7986bd49bd034aab8b6ad"),
    ("spec/params.generated.yaml", 45731, "c5f598e4e9292f831279bed5584cd399c42cc769b6550f8921a7bf94d7e20234", 46590, "48e5fee874fe4b6c72ee60be3b92e0cd9da51edab87e74bc5f3af964c6f9e534"),
    ("tools/run_w8_campaign.py", 60143, "9e724fd6acb4a8b80063d382f361dd6b07e9b51b35da740a1b5322414b035eda", 60452, "3d611e4c4f9fb300b47dd09d798da280d05b245df26af8dab69760335e50c89b"),
    ("tools/verify_w8_a.py", 15306, "6560163a00647cc50a10a00d969ca5404300a612e2ee54e186093e39caf23325", 15748, "3c4e039ab420c5422562f8c9961237b3daead8eac498b69cd921a430b8e986dc"),
    ("tools/gen_w8_execution_authorization.py", 26959, "3b949c16e258f30dd793c7863defd21fe668b3ab191138b355cda21672d383fc", 29078, "228329ad26e3a5972b4039952818fccbb4f1615e746247d084703699f624d934"),
    ("src/baseline/w8_spec_compatibility.py", 8577, "0e9655e6198a66751fa5ed2d8e18033d4bde26e1197a9e449a0182a1616135cf", 9258, "5006f0a7f1467cf943ffeeb6a29fdd885a7bb49133f111c5d1faf7a7d1d3d0c9"),
    ("src/baseline/w7c_source_compatibility.py", 15999, "eb769bf68e2bee3ef88d394c8397b986faec7678d79e24da380c2528a9ec4b42", 16781, "bb38c951aa358bab9c55d466662b6c872a2112e76202da9af4f85c84d57d4e9d"),
    ("src/baseline/g8_campaign.py", 58091, "ddfd95cbd910cb898034c833929f4c1b1832c4d048c4dbfa0df87d49e0eea302", 58680, "2317ad14be2ba3c081f2a1bb4ff55d5b9a102a16acecf60b4412198de7f5cc69"),
    ("src/baseline/w6_evidence.py", 30102, "eac17c4007f9b7c828caa4fdfc498ce01f5cf6f655855b773a9700e6f81039b1", 30631, "6ba1932d485dc1ebd2dc8454c453492c214c48f168b51e67605f028bfc67d4fc"),
    ("tools/verify_w7_g4.py", 36169, "ac49919cc82f0bcd396b58308247a2ad65c31336dfad27cfc62211b176ce5f77", 36691, "82602f1ab9f3bb7adf089bde7e58d463ec3f71d92a503822de0f038d586a0291"),
    ("tools/verify_w6_complete.py", 52840, "4a9ba3025d2f4040ac13fa98ce2b56d8c07516c980240307bd62dbbaf02b3f6d", 53091, "e90a2677a6bc7f903c065639adaae1c75e6be0c1ed56049126266f740425b057"),
)


class G10SemanticsFreezeError(RuntimeError):
    """The exact AM-94 transition or its zero-science boundary differs."""


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def rendered(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G10SemanticsFreezeError(message)


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or unsafe")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G10SemanticsFreezeError(f"cannot read {label}: {exc}") from None
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    return value, raw


def _git_bytes(root: Path, commit: str, relative: str) -> bytes:
    result = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=root, capture_output=True, check=False)
    _require(result.returncode == 0, f"cannot resolve AM-94 predecessor bytes: {relative}")
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


def load(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Verify AM-94, its exact source bytes, and its protected zero counters."""

    root = Path(root).resolve()
    value, raw = _read_json(root / FREEZE_RELATIVE_PATH, "AM-94 pre-science freeze")
    _require(raw == rendered(value), "AM-94 freeze is not canonical rendered JSON")
    _require(sha256_bytes(raw) == FREEZE_SHA256, "AM-94 freeze file bytes differ")
    body = dict(value)
    identifier = body.pop("freeze_id", None)
    _require(identifier == FREEZE_ID and identifier == "g10semantics-" + sha256_bytes(canonical(body)), "AM-94 freeze identity differs")
    _require(
        set(value) == {
            "schema_version", "artifact_role", "status", "amendment", "timing",
            "predecessor_commit", "prior_compatibility", "allowed_parameter_paths",
            "entries", "scientific_boundary", "w8_terminal_evidence", "freeze_id",
        },
        "AM-94 freeze schema differs",
    )
    _require(
        value["schema_version"] == 1
        and value["artifact_role"] == "G10_PRE_SCIENCE_NORMATIVE_SEMANTICS_FREEZE"
        and value["status"] == "SEMANTICS_ONLY_FROZEN"
        and value["amendment"] == "AM-94"
        and value["timing"] == "post_w8_c_pre_g10_model_facing_evaluation"
        and value["predecessor_commit"] == PREDECESSOR_COMMIT
        and value["allowed_parameter_paths"] == list(ALLOWED_PARAMETER_PATHS),
        "AM-94 freeze boundary differs",
    )
    prior, prior_raw = _read_json(root / PRIOR_COMPATIBILITY_PATH, "AM-93 compatibility")
    _require(
        sha256_bytes(prior_raw) == PRIOR_COMPATIBILITY_SHA256
        and prior.get("compatibility_id") == PRIOR_COMPATIBILITY_ID
        and value["prior_compatibility"] == {
            "path": PRIOR_COMPATIBILITY_PATH,
            "compatibility_id": PRIOR_COMPATIBILITY_ID,
            "sha256": PRIOR_COMPATIBILITY_SHA256,
        },
        "AM-94 prior compatibility differs",
    )
    _require(value["entries"] == expected_entries(), "AM-94 source-view entries differ")
    for relative, base_bytes, base_sha, current_bytes, current_sha in VIEW_HASHES:
        base = _git_bytes(root, PREDECESSOR_COMMIT, relative)
        _require(len(base) == base_bytes and sha256_bytes(base) == base_sha, f"AM-94 predecessor bytes differ: {relative}")
        current = (root / relative).read_bytes()
        _require(len(current) == current_bytes and sha256_bytes(current) == current_sha, f"AM-94 current bytes differ: {relative}")
    old_params = yaml.safe_load(_git_bytes(root, PREDECESSOR_COMMIT, "spec/params.generated.yaml"))
    new_params = yaml.safe_load((root / "spec/params.generated.yaml").read_bytes())
    _require(_leaf_differences(old_params, new_params) == set(ALLOWED_PARAMETER_PATHS), "AM-94 parameter drift exceeds G-10 semantics")

    boundary = value["scientific_boundary"]
    _require(
        boundary == {
            "er2_randomized_training": 0,
            "er9_training": 0,
            "g10_learned_outcomes_observed": 0,
            "g10_model_facing_evaluations": 0,
            "g11": 0,
            "learned_test_inference": 0,
            "test_model_facing_access": 0,
            "test_split": "SEALED",
            "w10": 0,
        },
        "AM-94 protected counters differ",
    )
    completion_path = root / "results/learned/w8/w8_completion.json"
    completion, completion_raw = _read_json(completion_path, "W8 terminal completion")
    reconciliation_path = root / "results/learned/w8/w8_c_reconciliation.json"
    reconciliation, reconciliation_raw = _read_json(reconciliation_path, "W8-C reconciliation")
    evidence = value["w8_terminal_evidence"]
    _require(
        evidence == {
            "completion_id": completion.get("completion_id"),
            "completion_sha256": sha256_bytes(completion_raw),
            "reconciliation_id": reconciliation.get("reconciliation_id"),
            "reconciliation_sha256": sha256_bytes(reconciliation_raw),
        },
        "AM-94 W8 terminal evidence binding differs",
    )
    _require(
        completion.get("g10") == "NOT_EXECUTED"
        and completion.get("g10_count") == 0
        and completion.get("er2") == 0
        and completion.get("er9") == 0
        and completion.get("test") == "SEALED"
        and completion.get("test_model_facing_access") == 0
        and completion.get("learned_test_inference") == 0,
        "W8 terminal protected counters moved before AM-94",
    )
    _require(
        reconciliation.get("protected_boundaries") == {
            "er2": 0,
            "er9": 0,
            "g10": 0,
            "learned_test_inference": 0,
            "papr_constrained_training": 0,
            "test_model_facing_access": 0,
        },
        "W8-C protected boundaries moved before AM-94",
    )
    _require(not (root / "results/freeze_manifest.json").exists(), "test freeze manifest exists before G-12")
    w9_files = sorted(path.relative_to(root).as_posix() for path in (root / "results/learned/w9").glob("**/*") if path.is_file())
    _require(w9_files == [FREEZE_RELATIVE_PATH], "scientific W9/G-10 file exists at AM-94 freeze")
    _require(not (root / "results/learned/g10").exists(), "G-10 outcome directory exists at AM-94 freeze")
    _require(new_params["evaluation"]["test_access_gate"] == "G-12", "test access gate is not G-12")
    return value


def verify_w8_carrier_source_transition(
    manifest: Mapping[str, Any], root: Path = REPO_ROOT
) -> frozenset[str]:
    """Authenticate exact AM-94-only successors of frozen W8 carrier entries."""

    root = Path(root).resolve()
    load(root)
    entries = manifest.get("entries")
    _require(isinstance(entries, list), "W8 source manifest entries are malformed")
    by_path = {
        str(entry.get("path")): entry
        for entry in entries
        if isinstance(entry, Mapping)
    }
    compatible: set[str] = set()
    for path, base_bytes, base_sha, current_bytes, current_sha in W8_CARRIER_SOURCE_HASHES:
        entry = by_path.get(path)
        _require(
            isinstance(entry, Mapping)
            and entry.get("bytes") == base_bytes
            and entry.get("sha256") == base_sha,
            f"W8 frozen source-manifest base differs: {path}",
        )
        raw = (root / path).read_bytes()
        _require(
            len(raw) == current_bytes and sha256_bytes(raw) == current_sha,
            f"W8 AM-94 carrier source differs: {path}",
        )
        compatible.add(path)
    return frozenset(compatible)
