#!/usr/bin/env python3
"""Run the frozen synthetic W3 BER/BLER campaign and emit G-2 evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch
import yaml
from sionna import __version__ as sionna_version
from sionna.phy.fec.ldpc import LDPC5GEncoder

from baseline.ldpc.adapter import SionnaLDPCAdapter
from baseline.ldpc.modulation import (
    bits_per_symbol,
    esn0_from_ebn0_db,
    map_bits,
    max_log_llr,
    n0_from_esn0_db,
    realised_symbol_energy,
)
from baseline.ldpc.reference import IndependentFloodingOMS
from baseline.ldpc.transport import build_packet_plan
from config.params import PARAMS_PATH, REPO_ROOT, get

CONFIG = REPO_ROOT / "configs" / "ldpc-g2.yaml"
RESULTS = REPO_ROOT / "results" / "baseline" / "g2"
FIXTURE = REPO_ROOT / get("baseline.ldpc_golden_vector_file")
OFFLINE_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "ldpc_offline_floor.json"
PACKET_RECORD = REPO_ROOT / "spec" / "evidence" / "packetisation_record.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _fetch_reference() -> tuple[Path, dict]:
    cfg = get("baseline.ldpc_bler_reference")
    archive = REPO_ROOT / "data" / "archives" / (
        f"lcrypto-5g-ldpc-{cfg['commit']}.tar.gz"
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        with urllib.request.urlopen(cfg["archive_url"]) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
    actual = sha256(archive)
    if actual != cfg["archive_sha256"]:
        raise RuntimeError(f"BLER reference checksum mismatch: {actual}")
    extracted = Path(tempfile.mkdtemp(prefix="capstone-g2-reference-"))
    with tarfile.open(archive) as source:
        source.extractall(extracted, filter="data")
    root = next(path for path in extracted.iterdir() if path.is_dir())
    decoder = root / cfg["decoder_source"]
    graph = root / cfg["graph_source"]
    licence = root / "LICENSE"
    for path in (decoder, graph, licence):
        if not path.is_file():
            raise RuntimeError(f"BLER reference is missing {path.relative_to(root)}")
    return root, {
        "name": get("baseline.ldpc_bler_reference_source"),
        "repository": cfg["repository"],
        "commit": cfg["commit"],
        "archive_url": cfg["archive_url"],
        "archive_sha256": actual,
        "licence": cfg["licence"],
        "decoder_source": cfg["decoder_source"],
        "decoder_sha256": sha256(decoder),
        "graph_source": cfg["graph_source"],
        "graph_sha256": sha256(graph),
        "reconstruction": cfg["reconstruction"],
    }


def _golden_summary() -> tuple[dict, dict]:
    if not FIXTURE.exists():
        raise RuntimeError("rung-2 NPZ is absent; run tools/fetch_ldpc_golden_vectors.py")
    fixture = np.load(FIXTURE)
    case_results = []
    for case in get("baseline.ldpc_golden_vector_cases"):
        index, bg, z = int(case["index"]), int(case["base_graph"]), int(case["lifting_size"])
        inputs = fixture[f"case_{index}_input"]
        expected_encoder = fixture[f"case_{index}_encoder"]
        expected_rate_matched = fixture[f"case_{index}_rate_matched"]
        raw = LDPC5GEncoder(
            inputs.shape[1], expected_encoder.shape[1], bg=f"bg{bg}", device="cpu"
        )
        actual_encoder = raw(torch.tensor(inputs, dtype=torch.float32)).numpy().astype(np.uint8)
        adapter = SionnaLDPCAdapter(
            inputs.shape[1], expected_encoder.shape[1],
            bits_per_symbol(case["modulation"]), bg, "cpu",
        )
        actual_rate_matched = adapter.encode(inputs)
        case_results.append({
            **case,
            "k": int(inputs.shape[1]),
            "n": int(expected_encoder.shape[1]),
            "messages": int(inputs.shape[0]),
            "encoder_mismatches": int(np.count_nonzero(actual_encoder != expected_encoder)),
            "rate_matched_mismatches": int(
                np.count_nonzero(actual_rate_matched != expected_rate_matched)
            ),
            "selected_lifting_size": adapter.lifting_size,
            "alignment": "remove_filler_marker_254_only_input_already_2Z_punctured",
        })
    offline = json.loads(OFFLINE_FIXTURE.read_text())
    source = np.fromiter((int(bit) for bit in offline["input_bits"]), dtype=np.uint8)
    expected = np.fromiter((int(bit) for bit in offline["rate_matched_bits"]), dtype=np.uint8)
    encoder = LDPC5GEncoder(
        offline["k"], offline["n"], bg=f"bg{offline['base_graph']}", device="cpu"
    )
    actual = encoder(torch.tensor(source[None], dtype=torch.float32)).numpy().astype(np.uint8)[0]
    offline_result = {
        "fixture": str(OFFLINE_FIXTURE.relative_to(REPO_ROOT)),
        "fixture_sha256": sha256(OFFLINE_FIXTURE),
        "mismatches": int(np.count_nonzero(actual != expected)),
        "syndrome_weight": offline["full_syndrome_weight"],
        "selected_lifting_size": int(encoder.z),
        "expected_lifting_size": offline["lifting_size"],
    }
    summary = {
        "source_rung": int(get("baseline.ldpc_golden_vector_source_rung")),
        "release": get("baseline.ldpc_golden_vector_upstream_release"),
        "asset": get("baseline.ldpc_golden_vector_upstream_asset"),
        "asset_url": get("baseline.ldpc_golden_vector_upstream_url"),
        "asset_sha256": get("baseline.ldpc_golden_vector_asset_sha256"),
        "fixture": str(FIXTURE.relative_to(REPO_ROOT)),
        "fixture_sha256": sha256(FIXTURE),
        "alignment": "remove_filler_marker_254_only_input_already_2Z_punctured",
        "cases": case_results,
        "offline_floor": offline_result,
        "pass": all(
            item["encoder_mismatches"] == item["rate_matched_mismatches"] == 0
            and item["selected_lifting_size"] == item["lifting_size"]
            for item in case_results
        ) and offline_result["mismatches"] == offline_result["syndrome_weight"] == 0,
    }
    provenance = {
        key: summary[key] for key in (
            "source_rung", "release", "asset", "asset_url", "asset_sha256",
            "fixture", "fixture_sha256", "alignment",
        )
    }
    return summary, provenance


def _known_answer_summary() -> dict:
    from baseline.ldpc.crc import remainder
    from baseline.ldpc.modulation import constellation, interleaver_indices

    source = np.unpackbits(np.frombuffer(b"123456789", dtype=np.uint8))
    crc_expected = {
        "crc16": "0011000111000011",
        "crc24a": "110011011110011100000011",
        "crc24b": "001000111110111101010010",
    }
    crc_results = {
        name: "".join(str(int(bit)) for bit in remainder(source, name))
        for name in crc_expected
    }
    modulation = {}
    for name in get("baseline.modulations"):
        labels, points = constellation(name)
        llr = max_log_llr(points[None, :], name, 0.01).reshape(labels.shape)
        q_m = bits_per_symbol(name)
        modulation[name] = {
            "q_m": q_m,
            "labels_recovered": bool(np.array_equal((llr > 0).astype(np.uint8), labels)),
            "mean_constellation_energy": float(np.mean(np.abs(points) ** 2)),
            "interleaver_indices_n8": interleaver_indices(8, q_m).tolist(),
            "sign_flip_detected": not np.array_equal(
                ((-llr) > 0).astype(np.uint8), labels
            ),
            "disabled_interleaver_detected": q_m == 1
            or not np.array_equal(interleaver_indices(8, q_m), np.arange(8)),
        }
    return {
        "crc": {
            name: {"expected": expected, "actual": crc_results[name],
                   "pass": crc_results[name] == expected}
            for name, expected in crc_expected.items()
        },
        "modulation": modulation,
        "pass": crc_results == crc_expected and all(
            item["labels_recovered"] and item["sign_flip_detected"]
            and item["disabled_interleaver_detected"]
            and abs(item["mean_constellation_energy"] - 1.0) < 1e-6
            for item in modulation.values()
        ),
    }


def _wilson(errors: int, trials: int, confidence_percent: float) -> tuple[float, float]:
    z = NormalDist().inv_cdf(0.5 + confidence_percent / 200.0)
    p = errors / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _seed(root: int, modulation: str, snr: float) -> int:
    material = f"{root}|{modulation}|{snr:.8f}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _simulate(graph: Path) -> tuple[list[dict], dict]:
    cfg = get("baseline.ldpc_bler_reference")
    k, n, bg, z = (int(cfg[key]) for key in ("k", "n", "base_graph", "lifting_size"))
    rate, blocks = float(cfg["rate"]), int(cfg["blocks_per_snr"])
    confidence = float(cfg["confidence_percent"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    energy_sum = {name: [0.0, 0] for name in get("baseline.modulations")}
    for modulation in get("baseline.modulations"):
        q_m = bits_per_symbol(modulation)
        dut = SionnaLDPCAdapter(k, n, q_m, bg, device)
        if dut.lifting_size != z:
            raise RuntimeError(f"{modulation}: Sionna Z={dut.lifting_size}, expected {z}")
        reference = IndependentFloodingOMS(graph, z, device)
        for ebn0_db in cfg["snr_grid_ebn0_db"][modulation]:
            started = time.perf_counter()
            rng = np.random.default_rng(_seed(int(cfg["simulation_seed"]), modulation, ebn0_db))
            bit_errors = {"reference": 0, "sionna": 0}
            block_errors = {"reference": 0, "sionna": 0}
            processed = 0
            while processed < blocks:
                batch = min(250, blocks - processed)
                information = rng.integers(0, 2, size=(batch, k), dtype=np.uint8)
                coded = dut.encode(information)
                symbols = map_bits(coded, modulation)
                esn0_db = esn0_from_ebn0_db(float(ebn0_db), rate, q_m)
                n0 = n0_from_esn0_db(esn0_db)
                noise = (
                    rng.normal(size=symbols.shape) + 1j * rng.normal(size=symbols.shape)
                ) * math.sqrt(n0 / 2)
                llr = max_log_llr(symbols + noise, modulation, n0)
                decoded = {
                    "reference": reference.decode(llr, k, q_m),
                    "sionna": dut.decode(llr),
                }
                for system, output in decoded.items():
                    errors = output != information
                    bit_errors[system] += int(np.count_nonzero(errors))
                    block_errors[system] += int(np.count_nonzero(np.any(errors, axis=1)))
                energy_sum[modulation][0] += realised_symbol_energy(symbols) * batch
                energy_sum[modulation][1] += batch
                processed += batch
            elapsed = time.perf_counter() - started
            for system in ("reference", "sionna"):
                low, high = _wilson(block_errors[system], blocks, confidence)
                rows.append({
                    "system": system,
                    "modulation": modulation,
                    "q_m": q_m,
                    "ebn0_db": float(ebn0_db),
                    "esn0_db": esn0_from_ebn0_db(float(ebn0_db), rate, q_m),
                    "blocks": blocks,
                    "information_bits": blocks * k,
                    "bit_errors": bit_errors[system],
                    "block_errors": block_errors[system],
                    "ber": bit_errors[system] / (blocks * k),
                    "bler": block_errors[system] / blocks,
                    "bler_wilson_low": low,
                    "bler_wilson_high": high,
                    "wall_time_s_shared_realisation": elapsed,
                    "k": k,
                    "n": n,
                    "base_graph": bg,
                    "lifting_size": z,
                    "rate": rate,
                    "decoder": get("baseline.ldpc_decoder"),
                    "offset": float(get("baseline.ldpc_decoder_offset")),
                    "iterations": int(get("baseline.ldpc_max_iters")),
                    "device": device,
                })
    return rows, {
        modulation: {
            "packet_mean_energy": total / count,
            "symbols_accumulated_over_blocks": count * (int(cfg["n"]) // bits_per_symbol(modulation)),
        }
        for modulation, (total, count) in energy_sum.items()
    }


def _crossing(rows: list[dict], system: str, modulation: str, target: float) -> float:
    points = sorted(
        ((row["esn0_db"], row["bler"]) for row in rows
         if row["system"] == system and row["modulation"] == modulation)
    )
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        if y0 >= target >= y1 and y0 > 0 and y1 > 0:
            fraction = (math.log10(target) - math.log10(y0)) / (
                math.log10(y1) - math.log10(y0)
            )
            return x0 + fraction * (x1 - x0)
    raise RuntimeError(f"{system}/{modulation}: BLER grid does not bracket {target}")


def _packetisation_summary() -> dict:
    record = json.loads(PACKET_RECORD.read_text())
    mismatches, infeasible = [], []
    for row in record["configurations"]:
        packet = build_packet_plan(row["k"], row["modulation"], row["nominal_rate_str"])
        if not row["feasible"]:
            infeasible.append({"tag": row["tag"], "reason": row["reason"]})
            if packet.feasible or packet.reason != row["reason"]:
                mismatches.append(row["tag"])
            continue
        for key in (
            "A", "source_bytes", "tb_crc_type", "tb_crc_bits", "B", "base_graph",
            "num_codeblocks", "B_prime", "K_prime", "lifting_size", "K",
            "filler_bits_total", "E", "E_sum",
        ):
            if packet.metadata()[key] != row[key]:
                mismatches.append(f"{row['tag']}:{key}")
    return {
        "solver_record": str(PACKET_RECORD.relative_to(REPO_ROOT)),
        "solver_record_sha256": sha256(PACKET_RECORD),
        "configurations": len(record["configurations"]),
        "proof_obligations": sum(row["obligation"] for row in record["configurations"]),
        "feasible": sum(row["feasible"] for row in record["configurations"]),
        "expected_structural_infeasibility": infeasible,
        "mismatches": mismatches,
        "progressive_design": get("baseline.progressive_packetisation_sensitivity"),
        "progressive_design_executed": False,
        "pass": not mismatches
        and infeasible == [{
            "tag": "cifar10/r_1_48/bpsk/1/3",
            "reason": "no_legal_byte_aligned_A_within_nominal_budget",
        }],
    }


def main() -> int:
    if git("status", "--porcelain"):
        raise RuntimeError("G-2 measurement requires a clean checkout")
    commit = git("rev-parse", "HEAD")
    git("cat-file", "-e", f"{commit}^{{commit}}")
    if sionna_version != str(get("baseline.ldpc_impl_version")):
        raise RuntimeError("wrong Sionna version")
    RESULTS.mkdir(parents=True, exist_ok=True)
    root, reference_provenance = _fetch_reference()
    try:
        golden, golden_provenance = _golden_summary()
        known = _known_answer_summary()
        packetisation = _packetisation_summary()
        rows, energy = _simulate(root / get("baseline.ldpc_bler_reference")["graph_source"])
    finally:
        shutil.rmtree(root.parent)
    cfg = get("baseline.ldpc_bler_reference")
    target = float(cfg["waterfall_target_bler"])
    comparisons = {}
    tolerance = float(get("evaluation.ber_match_tolerance_db"))
    for modulation in get("baseline.modulations"):
        reference = _crossing(rows, "reference", modulation, target)
        measured = _crossing(rows, "sionna", modulation, target)
        displacement = measured - reference
        comparisons[modulation] = {
            "reference_waterfall_esn0_db": reference,
            "sionna_waterfall_esn0_db": measured,
            "displacement_db": displacement,
            "absolute_displacement_db": abs(displacement),
            "tolerance_db": tolerance,
            "pass": abs(displacement) <= tolerance,
        }
    minimum_errors = int(cfg["required_minimum_block_errors_below_waterfall"])
    sufficient = {}
    for modulation in get("baseline.modulations"):
        sufficient[modulation] = {}
        for system in ("reference", "sionna"):
            below = [
                row for row in rows
                if row["modulation"] == modulation and row["system"] == system
                and 0 < row["bler"] <= target
            ]
            sufficient[modulation][system] = bool(
                below and max(row["block_errors"] for row in below) >= minimum_errors
            )
    resolved = {
        "schema_version": 1,
        "config_file": str(CONFIG.relative_to(REPO_ROOT)),
        "config_sha256": sha256(CONFIG),
        "params_file": str(PARAMS_PATH.relative_to(REPO_ROOT)),
        "params_sha256": sha256(PARAMS_PATH),
        "measurement_commit": commit,
        "measurement_dirty": False,
        "standard": get("baseline.ldpc_standard"),
        "standard_version": get("baseline.ldpc_standard_version"),
        "sionna_version": sionna_version,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "test_split_access": {
            "decoder_calls": 0,
            "canonicalisation_calls": 0,
            "inference_calls": 0,
            "accuracy_calls": 0,
        },
    }
    _write_json(RESULTS / "resolved_config.json", resolved)
    _write_json(RESULTS / "golden_vector_provenance.json", golden_provenance)
    _write_json(RESULTS / "golden_vector_summary.json", golden)
    _write_json(RESULTS / "known_answer_summary.json", known)
    _write_json(RESULTS / "bler_reference.json", {
        **reference_provenance,
        "settings": cfg,
        "decoder_offset": get("baseline.ldpc_decoder_offset"),
        "iterations": get("baseline.ldpc_max_iters"),
        "snr_conversion": "Es/N0[dB] = Eb/N0[dB] + 10*log10(R*Qm)",
    })
    fieldnames = list(rows[0])
    with (RESULTS / "bler_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_json(RESULTS / "packetisation_runtime_check.json", packetisation)
    evidence_files = {
        name: sha256(RESULTS / name)
        for name in (
            "resolved_config.json",
            "golden_vector_provenance.json",
            "golden_vector_summary.json",
            "known_answer_summary.json",
            "bler_reference.json",
            "bler_results.csv",
            "packetisation_runtime_check.json",
        )
    }
    components = {
        "golden_vectors": golden["pass"],
        "known_answers": known["pass"],
        "bler_reference_provenance": True,
        "bler_simulation_sufficiency": all(
            all(systems.values()) for systems in sufficient.values()
        ),
        "bler_displacement": all(value["pass"] for value in comparisons.values()),
        "packetisation_runtime": packetisation["pass"],
        "progressive_design_frozen_not_run": (
            packetisation["progressive_design"]["status"] == "frozen_not_run_at_G2"
            and not packetisation["progressive_design_executed"]
        ),
        "clean_reachable_measurement_commit": True,
        "test_split_sealed": not any(resolved["test_split_access"].values()),
    }
    adjudication = {
        "schema_version": 1,
        "gate": "G-2",
        "measurement_commit": commit,
        "measurement_dirty": False,
        "evidence_commit": None,
        "statistic": get("evaluation.ber_match_statistic"),
        "target_bler": target,
        "tolerance_db": tolerance,
        "snr_conversion": {
            modulation: {
                "q_m": bits_per_symbol(modulation),
                "rate": float(cfg["rate"]),
                "additive_db": 10 * math.log10(float(cfg["rate"]) * bits_per_symbol(modulation)),
                "formula": "Es/N0 = Eb/N0 + 10 log10(R Qm)",
            }
            for modulation in get("baseline.modulations")
        },
        "waterfalls": comparisons,
        "simulation_sufficiency": sufficient,
        "realised_symbol_energy": energy,
        "evidence_files": evidence_files,
        "components": components,
        "verdict": "PASS" if all(components.values()) else "HOLD",
    }
    _write_json(RESULTS / "g2_adjudication.json", adjudication)
    print(json.dumps({
        "verdict": adjudication["verdict"],
        "measurement_commit": commit,
        "waterfalls": comparisons,
        "components": components,
    }, indent=2))
    return 0 if adjudication["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
