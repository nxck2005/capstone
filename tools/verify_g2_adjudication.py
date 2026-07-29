#!/usr/bin/env python3
"""Fail-closed, network-free verification of the W3 / G-2 evidence set."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

from baseline.ldpc.modulation import bits_per_symbol
from config.params import REPO_ROOT, get

DEFAULT_EVIDENCE = REPO_ROOT / "results" / "baseline" / "g2"
REQUIRED_FILES = {
    "resolved_config.json",
    "golden_vector_provenance.json",
    "golden_vector_summary.json",
    "known_answer_summary.json",
    "bler_reference.json",
    "bler_results.csv",
    "packetisation_runtime_check.json",
    "g2_adjudication.json",
}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True
    )
    if result.returncode:
        raise VerificationError(f"git object/ancestry verification failed: {' '.join(args)}")
    return result.stdout.strip()


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid evidence JSON: {path.name}") from exc
    require(isinstance(value, dict), f"{path.name} must contain an object")
    return value


def _crossing(rows: list[dict], system: str, modulation: str, target: float) -> float:
    points = sorted(
        (float(row["esn0_db"]), float(row["bler"]))
        for row in rows
        if row["system"] == system and row["modulation"] == modulation
    )
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        if y0 >= target >= y1 and y0 > 0 and y1 > 0:
            fraction = (math.log10(target) - math.log10(y0)) / (
                math.log10(y1) - math.log10(y0)
            )
            return x0 + fraction * (x1 - x0)
    raise VerificationError(f"{system}/{modulation}: insufficient simulation cells")


def verify(
    evidence_dir: Path = DEFAULT_EVIDENCE, *, require_evidence_commit: bool = True
) -> dict:
    evidence_dir = evidence_dir.resolve()
    actual_files = {path.name for path in evidence_dir.iterdir() if path.is_file()}
    require(actual_files == REQUIRED_FILES,
            f"G-2 evidence file set differs: missing={sorted(REQUIRED_FILES-actual_files)}, "
            f"unexpected={sorted(actual_files-REQUIRED_FILES)}")
    resolved = load_json(evidence_dir / "resolved_config.json")
    golden_provenance = load_json(evidence_dir / "golden_vector_provenance.json")
    golden = load_json(evidence_dir / "golden_vector_summary.json")
    known = load_json(evidence_dir / "known_answer_summary.json")
    reference = load_json(evidence_dir / "bler_reference.json")
    packet = load_json(evidence_dir / "packetisation_runtime_check.json")
    adjudication = load_json(evidence_dir / "g2_adjudication.json")

    require(adjudication.get("schema_version") == 1 and adjudication.get("gate") == "G-2",
            "wrong G-2 adjudication schema or gate")
    measurement = adjudication.get("measurement_commit")
    require(isinstance(measurement, str) and len(measurement) == 40,
            "dirty or unreachable measurement commit")
    git("cat-file", "-e", f"{measurement}^{{commit}}")
    require(resolved.get("measurement_commit") == measurement
            and resolved.get("measurement_dirty") is False
            and adjudication.get("measurement_dirty") is False,
            "dirty or inconsistent measurement commit")
    if require_evidence_commit:
        relative = (evidence_dir / "g2_adjudication.json").relative_to(REPO_ROOT)
        evidence_commit = git("log", "-1", "--format=%H", "--", str(relative))
        require(bool(evidence_commit), "G-2 evidence commit is unreachable")
        git("cat-file", "-e", f"{evidence_commit}^{{commit}}")
        git("merge-base", "--is-ancestor", measurement, evidence_commit)

    expected_hashes = adjudication.get("evidence_files")
    require(isinstance(expected_hashes, dict)
            and set(expected_hashes) == REQUIRED_FILES - {"g2_adjudication.json"},
            "evidence hash manifest differs")
    for name, expected in expected_hashes.items():
        require(sha256(evidence_dir / name) == expected, f"{name} hash mismatch")

    require(resolved.get("standard") == get("baseline.ldpc_standard")
            and resolved.get("standard_version") == get("baseline.ldpc_standard_version"),
            "wrong standards version")
    require(resolved.get("sionna_version") == str(get("baseline.ldpc_impl_version")),
            "wrong Sionna version")
    require(not any(resolved.get("test_split_access", {}).values()), "test-split access")

    require(golden_provenance.get("source_rung")
            == golden.get("source_rung")
            == int(get("baseline.ldpc_golden_vector_source_rung")) == 2,
            "wrong golden-vector source rung")
    require(golden_provenance.get("asset_sha256")
            == golden.get("asset_sha256")
            == get("baseline.ldpc_golden_vector_asset_sha256"),
            "wrong golden-vector source checksum")
    expected_alignment = "remove_filler_marker_254_only_input_already_2Z_punctured"
    require(golden_provenance.get("alignment") == golden.get("alignment") == expected_alignment,
            "wrong vector alignment")
    require(golden.get("pass") is True and golden["offline_floor"]["mismatches"] == 0,
            "golden or offline fixture mismatch")
    require(all(
        case["encoder_mismatches"] == case["rate_matched_mismatches"] == 0
        and case["selected_lifting_size"] == case["lifting_size"]
        for case in golden["cases"]
    ), "wrong base graph or lifting size")

    cfg = get("baseline.ldpc_bler_reference")
    require(reference.get("name") == get("baseline.ldpc_bler_reference_source")
            and reference.get("commit") == cfg["commit"]
            and reference.get("archive_sha256") == cfg["archive_sha256"],
            "wrong BLER reference checksum or identity")
    require(reference.get("licence") == cfg["licence"]
            and reference.get("reconstruction") == cfg["reconstruction"],
            "wrong BLER reference source rung or reconstruction")
    settings = reference.get("settings", {})
    for key in ("k", "n", "base_graph", "lifting_size", "rate", "source_snr_convention"):
        require(settings.get(key) == cfg[key], f"wrong reference {key}")
    require(reference.get("decoder_offset") == get("baseline.ldpc_decoder_offset"),
            "wrong decoder offset")
    require(reference.get("iterations") == get("baseline.ldpc_max_iters"),
            "wrong decoder iteration count")

    require(known.get("pass") is True, "wrong mapper, interleaver, CRC, or LLR sign")
    require(set(known.get("crc", {})) == set(get("baseline.crc_spec")), "missing CRC")
    require(all(
        item.get("pass") is True and item.get("actual") == item.get("expected")
        for item in known["crc"].values()
    ), "wrong CRC")
    require(set(known.get("modulation", {})) == set(get("baseline.modulations")),
            "missing modulation")
    for modulation, result in known["modulation"].items():
        require(result["labels_recovered"] and result["sign_flip_detected"],
                f"{modulation}: LLR-sign error")
        require(result["disabled_interleaver_detected"],
                f"{modulation}: wrong mapper or interleaver")

    with (evidence_dir / "bler_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_cells = {
        (system, modulation, float(snr))
        for system in ("reference", "sionna")
        for modulation in get("baseline.modulations")
        for snr in cfg["snr_grid_ebn0_db"][modulation]
    }
    actual_cells = {
        (row["system"], row["modulation"], float(row["ebn0_db"])) for row in rows
    }
    require(actual_cells == expected_cells and len(rows) == len(expected_cells),
            "missing modulation or insufficient simulation cells")
    for row in rows:
        blocks = int(row["blocks"])
        k = int(row["k"])
        bit_errors, block_errors = int(row["bit_errors"]), int(row["block_errors"])
        require(blocks == int(cfg["blocks_per_snr"]), "insufficient simulation blocks")
        require(float(row["ber"]) == bit_errors / (blocks * k)
                and float(row["bler"]) == block_errors / blocks,
                "BER/BLER arithmetic mismatch")
        q_m = bits_per_symbol(row["modulation"])
        expected_es = float(row["ebn0_db"]) + 10 * math.log10(float(cfg["rate"]) * q_m)
        require(abs(float(row["esn0_db"]) - expected_es) < 1e-12,
                "wrong SNR conversion")
        require(int(row["n"]) == int(cfg["n"])
                and int(row["base_graph"]) == int(cfg["base_graph"])
                and int(row["lifting_size"]) == int(cfg["lifting_size"]),
                "wrong base graph or lifting size")
        require(float(row["offset"]) == get("baseline.ldpc_decoder_offset"),
                "wrong decoder offset")
        require(int(row["iterations"]) == get("baseline.ldpc_max_iters"),
                "wrong decoder iteration count")

    target = float(cfg["waterfall_target_bler"])
    tolerance = float(get("evaluation.ber_match_tolerance_db"))
    require(adjudication.get("statistic") == get("evaluation.ber_match_statistic")
            and adjudication.get("target_bler") == target
            and adjudication.get("tolerance_db") == tolerance,
            "wrong comparison statistic")
    require(set(adjudication.get("waterfalls", {})) == set(get("baseline.modulations")),
            "missing modulation waterfall")
    for modulation, recorded in adjudication["waterfalls"].items():
        ref = _crossing(rows, "reference", modulation, target)
        dut = _crossing(rows, "sionna", modulation, target)
        displacement = dut - ref
        require(abs(recorded["reference_waterfall_esn0_db"] - ref) < 1e-12
                and abs(recorded["sionna_waterfall_esn0_db"] - dut) < 1e-12
                and abs(recorded["displacement_db"] - displacement) < 1e-12,
                f"{modulation}: waterfall displacement arithmetic mismatch")
        require(abs(displacement) <= tolerance and recorded["pass"] is True,
                f"{modulation}: BLER displacement above 0.5 dB")
        conversion = adjudication["snr_conversion"][modulation]
        require(
            abs(conversion["additive_db"]
                - 10 * math.log10(float(cfg["rate"]) * bits_per_symbol(modulation))) < 1e-12,
            "wrong SNR conversion",
        )

    require(packet.get("pass") is True and not packet.get("mismatches"),
            "packetisation mismatch")
    require(packet.get("configurations") == 216 and packet.get("proof_obligations") == 144
            and packet.get("feasible") == 215,
            "packetisation grid count mismatch")
    require(packet.get("expected_structural_infeasibility") == [{
        "tag": "cifar10/r_1_48/bpsk/1/3",
        "reason": "no_legal_byte_aligned_A_within_nominal_budget",
    }], "unexplained structural infeasibility")
    require(packet.get("progressive_design") == get("baseline.progressive_packetisation_sensitivity")
            and packet.get("progressive_design_executed") is False,
            "progressive packetisation design mismatch")

    components = adjudication.get("components", {})
    require(components and all(components.values()) and adjudication.get("verdict") == "PASS",
            "G-2 component verdict is not a consistent PASS")
    return {
        "verdict": "PASS",
        "measurement_commit": measurement,
        "waterfalls": adjudication["waterfalls"],
        "rows": len(rows),
        "test_split_access": False,
    }


def main() -> int:
    try:
        result = verify()
    except VerificationError as exc:
        print(f"G-2 adjudication verification FAILED: {exc}", file=sys.stderr)
        return 1
    print(
        f"G-2 adjudication verification PASS: measurement={result['measurement_commit'][:12]}, "
        f"rows={result['rows']}, test_split_access=0"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
