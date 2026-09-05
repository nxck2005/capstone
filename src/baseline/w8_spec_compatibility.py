"""Fail-closed compatibility for the additive pre-W8 AM-93 spec change.

The W6/W5 evidence indexes predate AM-93 and must not be regenerated against
its new normative bytes.  This record admits exactly one later byte image for
the normative views, after W7/G-4 and before W8 source freeze.  It is a
read-only compatibility layer: it contains no training, selection, or result
logic.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from config.params import REPO_ROOT
from evaluation.g10_spec_compatibility import (
    PREDECESSOR_COMMIT as AM94_PREDECESSOR_COMMIT,
    load as load_am94_spec_compatibility,
)

COMPATIBILITY_RELATIVE_PATH = "results/learned/w7/w8_spec_additive_compatibility.json"
COMPATIBILITY_ID = "w8speccompat-0d2184356d3adf0a339b68ab9d1435e45f8105f1b0e8614ee37470d34dbb3436"
COMPATIBILITY_SHA256 = "94eea47b7f4e244ee4d97a1207fb49e7c123d6d0c244e1a214d135bcb8c6a05c"
BASE_COMPATIBILITY_ID = "w7speccompat-3d51f1fe7bda993d7ad4b69edb5bed2e6b736b0fdb9e708faf2be926a90ca4f8"
BASE_COMPATIBILITY_SHA256 = "a4fdd5f8f355e89da156c6abe7b7171edf26f2c1e7bb0b7da2f70b18909b7850"
BASE_COMPATIBILITY_PATH = "results/learned/w7/w7_spec_additive_compatibility.json"
BASE_COMMIT = "11be6d6f519094fe37ada347bdc678c99d066521"

# These are the exact W7-terminal-era bytes and the exact post-AM-93 bytes.
# A later edit cannot broaden the compatibility exception by editing only the
# JSON record: the expected identities remain frozen in this source module.
VIEW_HASHES: tuple[tuple[str, int, str, int, str], ...] = (
    ("spec/SPEC.md", 399552, "15279f60bd50b00f0d07bc6a5c4355c02d3071b55f087cda412097a8191e466c", 401661, "b05a2f04d6b3fa0e8110d0544900c29f823c96c424a75090a092433bb72cc68b"),
    ("spec/params.generated.yaml", 45681, "53bb0aff29e999869780a5516f3302f3358170d7980aba0742ce5da8a87b01c5", 45731, "c5f598e4e9292f831279bed5584cd399c42cc769b6550f8921a7bf94d7e20234"),
    ("spec/DATASHEET.md", 86137, "addc9ce9d88f9b90dd5f2c88b22e63fea597111e9e08aa44a7e922bb15fd5af6", 86257, "c5b9b78270099bd909a8248cf490ff00307ecb61672a98e5b1d29894340f7e68"),
    ("spec/concerns/amendments.md", 143268, "a91dc02f20bd3a14500a994bcafaa8db8a0a94c50b69e80635896ad5874890df", 144957, "12191834c4a76005243f7e0846d446e55cf934a44728fbac93dc0aeebab7b697"),
    ("spec/concerns/baseline.md", 56342, "4c8d63c900a5dbf7b7adfee0fc3d9b67218fb555897bff03f73f37474b8d827f", 56342, "4c8d63c900a5dbf7b7adfee0fc3d9b67218fb555897bff03f73f37474b8d827f"),
    ("spec/concerns/demo.md", 1803, "5c1cd9e80f6d5e8d1334169f64c5b1f84593a3a002fec4def40d687855128be3", 1803, "5c1cd9e80f6d5e8d1334169f64c5b1f84593a3a002fec4def40d687855128be3"),
    ("spec/concerns/experiments.md", 34282, "a9bda259769d9fba3ef581610ef3c00ee3d9684358f31f2d55e732b1f95b31ff", 34282, "a9bda259769d9fba3ef581610ef3c00ee3d9684358f31f2d55e732b1f95b31ff"),
    ("spec/concerns/hardware.md", 3051, "5092f014751415746029bbb66fe94aa6c396ffa6a4b7e1b5dd02a0eceb5b7f43", 3051, "5092f014751415746029bbb66fe94aa6c396ffa6a4b7e1b5dd02a0eceb5b7f43"),
    ("spec/concerns/programme.md", 11927, "0cca702a96b541540b5ba23d24e10adfd39e04329424dd967a9135254be6ac16", 11927, "0cca702a96b541540b5ba23d24e10adfd39e04329424dd967a9135254be6ac16"),
    ("spec/concerns/roadmap.md", 40173, "ad5681a9c8ad61b17132482547fe5401fa0437985a98ee2b40a15f18681e065a", 40175, "a6c79a0a2bfc1f7edaef5f7e739cbf6fc14188cdaaf539894922546ec08c2ee2"),
    ("spec/concerns/system.md", 36981, "e3aa96b4595ad082ee565d059ee02c73d06f78d2aae8e2f23d2ef4021e448034", 37315, "7db78c26ff492d54b4bd434f5007b94dc2aa88ae79c5c4c7f789dc501461208d"),
)


class W8SpecCompatibilityError(RuntimeError):
    """The exact AM-93 compatibility record is absent or has drifted."""


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise W8SpecCompatibilityError(message)


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or unsafe")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise W8SpecCompatibilityError(f"cannot read {label}: {exc}") from None
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    return value, raw


def _git_bytes(root: Path, commit: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    _require(result.returncode == 0, f"cannot resolve historical AM-93 base bytes: {path}")
    return result.stdout


def _expected_entries() -> list[dict[str, Any]]:
    return [
        {
            "path": path,
            "base_bytes": base_bytes,
            "base_sha256": base_sha256,
            "current_bytes": current_bytes,
            "current_sha256": current_sha256,
        }
        for path, base_bytes, base_sha256, current_bytes, current_sha256 in VIEW_HASHES
    ]


def load(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Authenticate AM-93 and project through the exact AM-94 successor."""

    root = Path(root).resolve()
    path = root / COMPATIBILITY_RELATIVE_PATH
    value, raw = _read_json(path, "W8 spec compatibility")
    _require(raw == canonical(value), "W8 spec compatibility is not canonical JSON")
    _require(sha256_bytes(raw) == COMPATIBILITY_SHA256, "W8 spec compatibility file bytes differ")
    body = dict(value)
    identifier = body.pop("compatibility_id", None)
    _require(identifier == COMPATIBILITY_ID and identifier == "w8speccompat-" + sha256_bytes(canonical(body)), "W8 spec compatibility ID differs")
    _require(
        set(value) == {
            "schema_version", "artifact_role", "status", "timing", "amendment",
            "base_commit", "base_compatibility", "allowed_change", "entries",
            "scientific_boundary", "compatibility_id",
        },
        "W8 spec compatibility schema differs",
    )
    _require(
        value["schema_version"] == 1
        and value["artifact_role"] == "W8_HISTORICAL_SPEC_ADDITIVE_COMPATIBILITY"
        and value["status"] == "ADDITIVE_FAIL_CLOSED"
        and value["timing"] == "after_w7_g4_before_w8_source_freeze"
        and value["amendment"] == "AM-93"
        and value["base_commit"] == BASE_COMMIT
        and value["base_compatibility"] == {
            "path": BASE_COMPATIBILITY_PATH,
            "compatibility_id": BASE_COMPATIBILITY_ID,
            "sha256": BASE_COMPATIBILITY_SHA256,
        }
        and value["allowed_change"] == {
            "parameter": "params.learned_system.checkpoint_selection_snr_db",
            "resolution": "params.channel.train_snr_db_fixed",
            "paths": [item[0] for item in VIEW_HASHES],
            "schedule": "G-4 -> W8 -> W9/G-10/G-11",
        }
        and value["scientific_boundary"] == {
            "w7_result_changed": False,
            "g4_result_changed": False,
            "w8_science_performed": False,
            "test_access": 0,
        },
        "W8 spec compatibility boundary differs",
    )
    base_record, base_raw = _read_json(root / BASE_COMPATIBILITY_PATH, "W7 spec compatibility")
    _require(sha256_bytes(base_raw) == BASE_COMPATIBILITY_SHA256, "W7 spec compatibility bytes differ")
    _require(base_record.get("compatibility_id") == BASE_COMPATIBILITY_ID, "W7 spec compatibility ID differs")
    entries = value["entries"]
    expected = _expected_entries()
    _require(entries == expected, "W8 spec compatibility entries differ")
    for path_text, base_bytes, base_sha256, current_bytes, current_sha256 in VIEW_HASHES:
        historical = _git_bytes(root, BASE_COMMIT, path_text)
        _require(len(historical) == base_bytes and sha256_bytes(historical) == base_sha256, f"AM-93 base bytes differ: {path_text}")
        am93 = _git_bytes(root, AM94_PREDECESSOR_COMMIT, path_text)
        _require(len(am93) == current_bytes and sha256_bytes(am93) == current_sha256, f"AM-93 historical current bytes differ: {path_text}")

    try:
        successor = load_am94_spec_compatibility(root)
    except Exception as exc:
        raise W8SpecCompatibilityError(f"AM-94 successor differs: {exc}") from None
    successor_entries = {entry["path"]: entry for entry in successor["entries"]}
    projection = dict(value)
    projection["entries"] = []
    for entry in value["entries"]:
        later = successor_entries[entry["path"]]
        _require(
            later["base_bytes"] == entry["current_bytes"]
            and later["base_sha256"] == entry["current_sha256"],
            f"AM-94 successor is not chained from AM-93: {entry['path']}",
        )
        projected = dict(entry)
        projected["current_bytes"] = later["current_bytes"]
        projected["current_sha256"] = later["current_sha256"]
        projection["entries"].append(projected)
    return projection
