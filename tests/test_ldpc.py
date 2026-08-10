from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from sionna.phy.fec.ldpc import LDPC5GEncoder

from baseline.ldpc.adapter import SionnaLDPCAdapter
from baseline.ldpc.crc import attach, check, remainder, strip
from baseline.ldpc.modulation import (
    constellation,
    deinterleave,
    interleave,
    interleaver_indices,
    map_bits,
    max_log_llr,
    realised_symbol_energy,
)
from baseline.ldpc.rate_matching import distribute
from baseline.ldpc.segmentation import concatenate, plan, segment
from baseline.ldpc.transport import build_packet_plan

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("crc16", "0011000111000011"),
        ("crc24a", "110011011110011100000011"),
        ("crc24b", "001000111110111101010010"),
    ],
)
def test_crc_known_answers(name: str, expected: str):
    bits = np.unpackbits(np.frombuffer(b"123456789", dtype=np.uint8))
    actual = "".join(str(int(value)) for value in remainder(bits, name))
    assert actual == expected
    framed = attach(bits, name)
    assert check(framed, name)
    assert np.array_equal(strip(framed, name), bits)
    framed[0] ^= 1
    assert not check(framed, name)


@pytest.mark.parametrize(
    ("modulation", "labels", "points"),
    [
        ("bpsk", [[0], [1]], [1 + 0j, -1 + 0j]),
        (
            "qpsk",
            [[0, 0], [0, 1], [1, 0], [1, 1]],
            [(1 + 1j) / np.sqrt(2), (1 - 1j) / np.sqrt(2),
             (-1 + 1j) / np.sqrt(2), (-1 - 1j) / np.sqrt(2)],
        ),
        (
            "qam16",
            [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1]],
            [(1 + 1j) / np.sqrt(10), (1 + 3j) / np.sqrt(10),
             (3 + 1j) / np.sqrt(10), (3 + 3j) / np.sqrt(10)],
        ),
    ],
)
def test_mapper_hand_derived_known_answers(modulation, labels, points):
    actual = map_bits(np.asarray(labels, dtype=np.uint8).reshape(-1), modulation)
    assert np.allclose(actual, points)
    all_labels, all_points = constellation(modulation)
    assert np.isclose(np.mean(np.abs(all_points) ** 2), 1.0)
    assert np.array_equal(all_labels[: len(labels)], labels)


@pytest.mark.parametrize(
    ("q_m", "expected"),
    [
        (1, [0, 1, 2, 3, 4, 5, 6, 7]),
        (2, [0, 4, 1, 5, 2, 6, 3, 7]),
        (4, [0, 2, 4, 6, 1, 3, 5, 7]),
    ],
)
def test_interleaver_hand_derived_known_answers(q_m, expected):
    bits = np.arange(8)
    assert interleaver_indices(8, q_m).tolist() == expected
    assert interleave(bits, q_m).tolist() == expected
    assert np.array_equal(deinterleave(interleave(bits, q_m), q_m), bits)


@pytest.mark.parametrize("modulation", ["bpsk", "qpsk", "qam16"])
def test_demapper_known_labels_and_llr_sign(modulation):
    labels, points = constellation(modulation)
    llrs = max_log_llr(points[None, :], modulation, 0.01).reshape(-1, labels.shape[1])
    assert np.array_equal((llrs > 0).astype(np.uint8), labels)


def test_llr_sign_flip_mutation_is_detected():
    labels, points = constellation("qam16")
    correct = max_log_llr(points[None, :], "qam16", 0.01).reshape(-1, 4)
    assert not np.array_equal(((-correct) > 0).astype(np.uint8), labels)


def test_disabled_qam16_interleaver_mutation_is_detected():
    bits = np.arange(16)
    assert not np.array_equal(interleave(bits, 4), bits)


def test_rate_matching_distribution_partial_final_block():
    assert distribute(20, 3, 4) == [4, 8, 8]
    assert sum(distribute(42624, 6, 4)) == 42624
    with pytest.raises(ValueError):
        distribute(19, 3, 4)


def test_offline_floor_fixture_is_unconditional_and_exact():
    fixture = json.loads((ROOT / "tests/fixtures/ldpc_offline_floor.json").read_text())
    source = np.fromiter((int(bit) for bit in fixture["input_bits"]), dtype=np.uint8)
    expected = np.fromiter((int(bit) for bit in fixture["rate_matched_bits"]), dtype=np.uint8)
    encoder = LDPC5GEncoder(
        fixture["k"], fixture["n"], bg=f"bg{fixture['base_graph']}", device="cpu"
    )
    actual = encoder(torch.tensor(source[None], dtype=torch.float32)).numpy().astype(np.uint8)[0]
    assert int(encoder.z) == fixture["lifting_size"]
    assert fixture["full_syndrome_weight"] == 0
    assert np.array_equal(actual, expected)


@pytest.mark.external_ldpc_fixture
def test_srsran_encoder_and_rate_matched_fixture_exact():
    fixture_path = ROOT / "tests/fixtures/ldpc_ts38212_golden.npz"
    assert fixture_path.exists(), (
        "rung-2 fixture absent; run tools/fetch_ldpc_golden_vectors.py. "
        "The offline floor test above still runs independently."
    )
    fixture = np.load(fixture_path)
    for index, bg, z in ((23, 1, 36), (81, 2, 64)):
        inputs = fixture[f"case_{index}_input"]
        expected_encoder = fixture[f"case_{index}_encoder"]
        expected_rate_matched = fixture[f"case_{index}_rate_matched"]
        raw = LDPC5GEncoder(inputs.shape[1], expected_encoder.shape[1], bg=f"bg{bg}", device="cpu")
        actual_encoder = raw(torch.tensor(inputs, dtype=torch.float32)).numpy().astype(np.uint8)
        assert int(raw.z) == z
        assert np.array_equal(actual_encoder, expected_encoder)
        adapter = SionnaLDPCAdapter(inputs.shape[1], expected_encoder.shape[1], 4, bg, "cpu")
        assert adapter.lifting_size == z
        assert np.array_equal(adapter.encode(inputs), expected_rate_matched)


def test_segmentation_crc_filler_and_concatenation():
    layout = plan(42624, 0.5)
    assert layout is not None and layout.code_blocks > 1
    payload = np.arange(layout.payload_bits, dtype=np.uint8) % 2
    blocks = segment(payload, layout)
    assert all(block.size == layout.k for block in blocks)
    assert np.array_equal(concatenate(blocks, layout), payload)
    blocks[0][0] ^= 1
    with pytest.raises(ValueError, match="CRC24B"):
        concatenate(blocks, layout)


def test_runtime_packetisation_matches_every_solver_row():
    record = json.loads((ROOT / "spec/evidence/packetisation_record.json").read_text())
    exact_keys = {
        "A", "source_bytes", "tb_crc_type", "tb_crc_bits", "B", "base_graph",
        "code_block_max_bits", "num_codeblocks", "cb_crc_bits_total", "B_prime",
        "K_prime", "K_b_for_lifting", "lifting_size", "K", "filler_bits_per_block",
        "filler_bits_total", "E", "E_sum", "B_nominal", "effective_code_rate",
    }
    infeasible = []
    for row in record["configurations"]:
        runtime = build_packet_plan(row["k"], row["modulation"], row["nominal_rate_str"])
        if not row["feasible"]:
            infeasible.append(row["tag"])
            assert not runtime.feasible and runtime.reason == row["reason"]
            continue
        metadata = runtime.metadata()
        assert runtime.channel_bits == row["k"] * runtime.q_m
        assert {key: metadata[key] for key in exact_keys} == {
            key: row[key] for key in exact_keys
        }, row["tag"]
    assert infeasible == ["cifar10/r_1_48/bpsk/1/3"]


@pytest.mark.primary_runtime
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA conformance is enforced elsewhere")
@pytest.mark.parametrize(("modulation", "q_m"), [("bpsk", 1), ("qpsk", 2), ("qam16", 4)])
def test_adapter_clean_high_snr_all_modulations(modulation, q_m):
    rng = np.random.default_rng(7 + q_m)
    source = rng.integers(0, 2, size=(16, 128), dtype=np.uint8)
    adapter = SionnaLDPCAdapter(128, 256, q_m, 2, "cuda")
    coded = adapter.encode(source)
    symbols = map_bits(coded, modulation)
    llrs = max_log_llr(symbols, modulation, 1e-4)
    assert np.array_equal(adapter.decode(llrs), source)
    assert realised_symbol_energy(symbols) > 0
