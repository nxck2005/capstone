"""W10 classical/evaluator seams: J2K cache scope, exact JPEG quality, PAPR reporting."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import baseline.classical.jpeg_pipeline as jpeg_pipeline
import data.preprocessing as preprocessing
import evaluation.w10_classical as w10_classical
from baseline.classical.pipeline import CODEC_INFEASIBILITY, DECODE_FAILURE, DELIVERED, ChannelIdentity
from baseline.classical.records import score_result
from baseline.jpeg import JpegCodec
from baseline.j2k import J2KCodec
from config.params import get
from data.preprocessing import canonicalize_source
from evaluation.w10_backends import W10Execution, learned_unit
from training.papr_constrained import load_papr_config, papr_protocol_config_hash
from training.w8_protocol import load_w8_config


def _synthetic_rgb() -> np.ndarray:
    rows = np.arange(32)[:, None, None]
    columns = np.arange(32)[None, :, None]
    return np.concatenate(
        (
            (rows * 7 + columns * 3) % 256,
            (rows * 11 + columns * 5) % 256,
            (rows * 13 + columns * 17) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


@pytest.fixture(autouse=True)
def fixture_source_decoders(monkeypatch):
    decoders = {dataset: (lambda source_bytes: _synthetic_rgb()) for dataset in ("cifar10", "stl10", "imagenette160")}
    monkeypatch.setattr(preprocessing, "_SOURCE_DECODERS", decoders)


class _StubView:
    def __init__(self, count: int = 1) -> None:
        self.stable_ids = [f"val-{index:04d}" for index in range(count)]
        self.labels = {stable_id: index % 10 for index, stable_id in enumerate(self.stable_ids)}
        self.products = {stable_id: canonicalize_source(f"fixture/{stable_id}".encode(), "cifar10") for stable_id in self.stable_ids}

    def product(self, stable_id: str):
        return self.products[stable_id]

    def label(self, stable_id: str) -> int:
        return self.labels[stable_id]

    def canonical_tensor(self, stable_id: str) -> torch.Tensor:
        index = self.stable_ids.index(stable_id)
        return torch.zeros(3, 4, 4, dtype=torch.float32) + (index % 2)


class _StubTransport:
    def __init__(self, papr_db: float) -> None:
        self.papr_db = papr_db


class _StubClassifier(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(inputs.shape[0], 10, dtype=torch.float32)


class _StubPolicy:
    dataset = "imagenette160"

    def predict(self) -> int:
        return 0

    def is_correct(self, true_label: int) -> bool:
        return int(true_label) == 0


def test_j2k_route_constructs_cache_scoped_codec(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    real_codec = J2KCodec

    class _SpyCodec(real_codec):
        def __init__(self, cache_root: Path) -> None:
            captured["cache_root"] = Path(cache_root)
            super().__init__(cache_root)

    monkeypatch.setattr(w10_classical, "J2KCodec", _SpyCodec)
    monkeypatch.setattr(w10_classical, "_outage_policy", lambda _root: _StubPolicy())
    context = W10Execution(root=tmp_path, device="cpu")
    context.view = _StubView(1)
    context.classifiers["artifact_finetuned"] = _StubClassifier().eval()
    candidate_id = None
    import json

    table = json.loads((Path(".") / "results/baseline/g8_e/candidate_authority.json").read_bytes())
    candidate = table["candidates"][0]
    candidate_id = str(candidate["candidate_id"])
    point = {
        "snr_db": int(get("channel.test_snr_grid_db")[0]),
        "authority_candidate_id": candidate_id,
        "tie_break_applied": False,
    }
    binding = {"kind": "classical_pass_two", "selections": [point] * 21}
    unit = {
        "system": "classical_adaptive",
        "bw_ratio": "r_1_6",
        "snr_db": int(get("channel.test_snr_grid_db")[0]),
        "channel_seed": 0,
    }
    monkeypatch.setattr(
        w10_classical,
        "run_classical_pipeline",
        lambda *args, **kwargs: SimpleNamespace(
            verdict=DECODE_FAILURE,
            dataset="imagenette160",
            k_symbols=12800,
            modulation="qpsk",
            ldpc_rate="1/2",
            snr_db=-8.0,
            stable_sample_id="val-0000",
            noise_id=None,
            packet_feasible=True,
            structural_reason=None,
            accounting=None,
            source_coding=None,
            transport=_StubTransport(3.5),
            codestream_recovered_exactly=None,
            decoded_image=None,
        ),
    )
    aggregate = w10_classical.classical_unit(
        context,
        unit,
        root=Path("."),
        checkpoint_id="not_applicable_untrained_codec",
        binding=binding,
        codec_kind="jpeg2000",
    )
    assert captured["cache_root"] == Path(".") / w10_classical.W10_J2K_CACHE_DIR
    assert aggregate["mean_papr_db"] == 3.5
    assert aggregate["papr_measured_count"] == 1
    assert aggregate["papr_denominator"] == 1


def test_jpeg_exact_quality_never_substitutes(tmp_path: Path) -> None:
    codec = JpegCodec()
    image = _synthetic_rgb()
    low = min(codec.quality_grid)
    high = max(codec.quality_grid)
    low_bytes = len(codec.encode_at_quality(image, quality=low))
    high_bytes = len(codec.encode_at_quality(image, quality=high))
    assert low_bytes < high_bytes
    budget = low_bytes + 1  # the best (lowest) quality fits; the frozen high quality does not

    exact = codec.encode_exact_quality(
        image,
        canonical_pixels_sha256="a" * 64,
        budget_bytes=budget,
        encode_axis_px=32,
        quality=high,
    )
    assert exact.feasible is False and exact.codestream is None
    assert exact.search_points == (high,)
    search = codec.encode_to_budget(
        image, canonical_pixels_sha256="a" * 64, budget_bytes=budget, encode_axis_px=32
    )
    assert search.feasible is True and search.quality != high  # the selector's search is separate


def test_jpeg_pipeline_executes_the_frozen_quality_and_reports_infeasibility(
    monkeypatch, tmp_path: Path
) -> None:
    codec = JpegCodec()
    image = _synthetic_rgb()
    low = min(codec.quality_grid)
    high = max(codec.quality_grid)
    budget = len(codec.encode_at_quality(image, quality=low)) + 1

    attempted: list[int] = []
    original = JpegCodec.encode_exact_quality

    def spy(self, image, *, canonical_pixels_sha256, budget_bytes, encode_axis_px, quality):
        attempted.append(int(quality))
        return original(
            self,
            image,
            canonical_pixels_sha256=canonical_pixels_sha256,
            budget_bytes=budget_bytes,
            encode_axis_px=encode_axis_px,
            quality=quality,
        )

    monkeypatch.setattr(JpegCodec, "encode_exact_quality", spy)
    monkeypatch.setattr(
        jpeg_pipeline, "build_packet_plan", lambda *args, **kwargs: SimpleNamespace(feasible=True)
    )
    monkeypatch.setattr(
        jpeg_pipeline,
        "build_accounting",
        lambda packet: SimpleNamespace(
            payload_bytes=budget,
            payload_bits=budget * 8,
            k_symbols=512,
            modulation="qpsk",
            reconciles=True,
        ),
    )
    product = canonicalize_source(b"fixture/jpeg-exact", "cifar10")
    identity = ChannelIdentity(
        dataset_version=get("datasets.cifar10.archive_sha256"),
        split_manifest_hash=get("datasets.cifar10.manifest_sha256"),
        channel_seed=0,
    )
    result = jpeg_pipeline.run_jpeg_pipeline(
        product,
        dataset="cifar10",
        k_symbols=512,
        modulation="qpsk",
        ldpc_rate="1/2",
        snr_db=0.0,
        quality=high,
        codec=codec,
        channel_identity=identity,
        encode_axis_px=32,
    )
    assert result.verdict == CODEC_INFEASIBILITY
    assert attempted == [high]  # no fallback to the lower quality that would fit


def test_classical_qam16_transport_papr_is_measured_not_null(monkeypatch, tmp_path: Path) -> None:
    from baseline.classical.channel_transport import build_accounting, transport_round_trip
    from baseline.ldpc.transport import build_packet_plan

    packet = build_packet_plan(512, "qam16", "1/2")
    assert packet.feasible
    accounting = build_accounting(packet)
    rng = np.random.default_rng(20260920)
    payload_bits = rng.integers(0, 2, size=accounting.payload_bits, dtype=np.uint8)
    outcome = transport_round_trip(
        payload_bits,
        packet,
        snr_db=18.0,
        noise_id="n" * 64,
        device="cpu",
    )
    assert outcome.papr_db > 0  # QAM16 is not constant modulus
    assert outcome.realised_symbol_energy != 0

    monkeypatch.setattr(w10_classical, "_outage_policy", lambda _root: _StubPolicy())
    context = W10Execution(root=tmp_path, device="cpu")
    context.view = _StubView(1)
    context.classifiers["artifact_finetuned"] = _StubClassifier().eval()
    unit = {
        "system": "classical_adaptive",
        "bw_ratio": "r_1_6",
        "snr_db": int(get("channel.test_snr_grid_db")[0]),
        "channel_seed": 0,
    }
    binding = {
        "kind": "classical_pass_two",
        "selections": [
            {"snr_db": int(snr), "authority_candidate_id": "cand-missing", "tie_break_applied": False}
            for snr in get("channel.test_snr_grid_db")
        ],
    }
    import json

    candidate = json.loads((Path(".") / "results/baseline/g8_e/candidate_authority.json").read_bytes())["candidates"][0]
    binding["selections"] = [
        {"snr_db": int(snr), "authority_candidate_id": str(candidate["candidate_id"]), "tie_break_applied": False}
        for snr in get("channel.test_snr_grid_db")
    ]
    stub = SimpleNamespace(
        verdict=DELIVERED,
        dataset="imagenette160",
        k_symbols=12800,
        modulation=str(candidate["modulation"]),
        ldpc_rate=str(candidate["ldpc_rate"]),
        snr_db=float(unit["snr_db"]),
        stable_sample_id="val-0000",
        noise_id=None,
        packet_feasible=True,
        structural_reason=None,
        accounting=None,
        source_coding=None,
        transport=outcome,
        codestream_recovered_exactly=True,
        decoded_image=_synthetic_rgb(),
    )
    monkeypatch.setattr(w10_classical, "run_classical_pipeline", lambda *args, **kwargs: stub)
    aggregate = w10_classical.classical_unit(
        context,
        unit,
        root=Path("."),
        checkpoint_id="not_applicable_untrained_codec",
        binding=binding,
        codec_kind="jpeg2000",
    )
    assert aggregate["mean_papr_db"] == pytest.approx(outcome.papr_db)
    assert aggregate["papr_measured_count"] == 1


class _PaprOutput:
    def __init__(self, inputs: torch.Tensor, papr: float) -> None:
        self.logits = torch.zeros(inputs.shape[0], 10)
        self.logits[:, 0] = 5.0
        self.reconstruction = inputs
        self.papr_db = torch.full((inputs.shape[0],), papr)


class _PaprModel(torch.nn.Module):
    def __init__(self, papr: float) -> None:
        super().__init__()
        self.papr = papr

    def forward(self, inputs, snr_db, *, unit_noise=None):
        return _PaprOutput(inputs, self.papr)


def test_learned_papr_records_mean_max_and_cap_compliance() -> None:
    context = W10Execution(root=Path("."), device="cpu")
    context.view = _StubView(4)
    unit = {"system": "learned", "bw_ratio": "r_1_6", "snr_db": -8, "train_seed": 0, "channel_seed": 0}
    config = load_w8_config("r_1_6", 0, 0)
    plain = learned_unit(context, unit, model=_PaprModel(2.0), config=config, checkpoint_id="a" * 64)
    assert plain["mean_papr_db"] == 2.0 and plain["max_papr_db"] == 2.0
    assert plain["papr_measured_count"] == 4 and plain["papr_denominator"] == 4
    assert "papr_cap_compliant" not in plain

    papr_unit = {
        "system": "learned_papr_constrained",
        "bw_ratio": "r_1_6",
        "snr_db": -8,
        "train_seed": 0,
        "channel_seed": 0,
    }
    papr_config = load_papr_config()
    compliant = learned_unit(
        context,
        papr_unit,
        model=_PaprModel(2.0),
        config=papr_config,
        checkpoint_id="a" * 64,
        papr_cap_db=3.0,
        protocol_config_hash=papr_protocol_config_hash(papr_config),
    )
    assert compliant["papr_cap_compliant"] is True
    excessive = learned_unit(
        context,
        papr_unit,
        model=_PaprModel(3.5),
        config=papr_config,
        checkpoint_id="a" * 64,
        papr_cap_db=3.0,
        protocol_config_hash=papr_protocol_config_hash(papr_config),
    )
    assert excessive["papr_cap_compliant"] is False

def test_jpeg_execution_uses_frozen_quality_without_selection_digest_reuse(monkeypatch, tmp_path: Path) -> None:
    context = W10Execution(root=Path("."), device="cpu")
    context.view = _StubView(1)
    context.classifiers["artifact_finetuned"] = _StubClassifier().eval()
    context.classifiers["clean"] = _StubClassifier().eval()
    monkeypatch.setattr(w10_classical, "_outage_policy", lambda _root: _StubPolicy())
    stub = SimpleNamespace(
        verdict=DELIVERED,
        dataset="imagenette160",
        k_symbols=12800,
        modulation="qpsk",
        ldpc_rate="1/2",
        snr_db=-8.0,
        stable_sample_id="val-0000",
        noise_id=None,
        packet_feasible=True,
        structural_reason=None,
        accounting=None,
        source_coding=None,
        transport=_StubTransport(1.0),
        codestream_recovered_exactly=True,
        decoded_image=_synthetic_rgb(),
    )
    monkeypatch.setattr(w10_classical, "run_jpeg_pipeline", lambda *args, **kwargs: stub)
    unit = {
        "system": "classical_jpeg_secondary",
        "bw_ratio": "r_1_6",
        "snr_db": -8,
        "channel_seed": 0,
    }
    binding = {
        "kind": "w10_jpeg_validation_selection",
        "selection_id": "w10jpegselection-synthetic",
        "selections": [
            {
                "snr_db": -8,
                "modulation": "qpsk",
                "ldpc_rate": "1/2",
                "encode_axis_px": 32,
                "quality": 50,
                "per_image_correct_digest": "0" * 64,
            }
        ],
    }
    aggregate = w10_classical.classical_unit(
        context,
        unit,
        root=Path("."),
        checkpoint_id="not_applicable_untrained_codec",
        binding=binding,
        codec_kind="jpeg",
        quality=50,
    )
    assert aggregate["n_total"] == 1
    assert aggregate["binding"]["quality"] == 50
