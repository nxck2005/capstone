"""Synthetic-only checks for the AM-98 W10 successor evaluators and PAPR protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from baseline.jpeg import JpegCodec
from channels.power import PeakPowerConstraint
from config.params import get
from config.run_config import config_hash as run_config_hash
from evaluation import w10_dispatch
from evaluation.w10_backends import (
    DECODE_FAILURE,
    W10Execution,
    decode_label_payload,
    label_payload,
    learned_unit,
    recon_ablation_unit,
)
from evaluation.w10_bindings import pending_states, resolve_scope_bindings
from evaluation.w10_evidence import validate_per_image
from evaluation.w10_scope import entry_for
from training.papr_constrained import (
    PaprConstrainedTrainer,
    build_papr_model,
    load_papr_config,
    papr_protocol_config_hash,
    protected_counters,
)
from training.w8_protocol import load_w8_config


class _StubView:
    def __init__(self, count: int = 1000) -> None:
        self.stable_ids = [f"val-{index:04d}" for index in range(count)]
        self.labels = {stable_id: index % 10 for index, stable_id in enumerate(self.stable_ids)}

    def label(self, stable_id: str) -> int:
        return self.labels[stable_id]

    def canonical_tensor(self, stable_id: str) -> torch.Tensor:
        index = self.stable_ids.index(stable_id)
        return torch.zeros(3, 4, 4, dtype=torch.float32) + (index % 2)


class _StubOutput:
    def __init__(self, inputs: torch.Tensor, *, label: int) -> None:
        batch = inputs.shape[0]
        logits = torch.zeros(batch, 10, dtype=torch.float32)
        logits[:, label] = 5.0
        logits[:, (label + 1) % 10] = 4.0
        self.logits = logits
        self.reconstruction = inputs
        self.papr_db = torch.zeros(batch, dtype=torch.float32)


class _StubModel(torch.nn.Module):
    """Logits prefer class 0; reconstruction is a constant class 7 image."""

    def forward(self, inputs: torch.Tensor, snr_db: float, *, unit_noise: torch.Tensor) -> _StubOutput:
        return _StubOutput(inputs, label=0)


class _ReconstructionModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, snr_db: float, *, unit_noise: torch.Tensor) -> _StubOutput:
        return _StubOutput(torch.zeros_like(inputs), label=0)


class _StubClassifier(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(inputs.shape[0], 10, dtype=torch.float32)
        logits[:, 7] = 5.0
        return logits


def _context() -> W10Execution:
    context = W10Execution(root=Path("."), device="cpu")
    context.view = _StubView()
    return context


def test_label_payload_is_the_frozen_low_nibble_frame() -> None:
    for label in range(10):
        frame = label_payload(label)
        assert frame.shape == (1,)
        assert int(frame[0]) == label and int(frame[0]) >> 4 == 0
        assert decode_label_payload(np.unpackbits(frame), payload_bits=8) == label
    with pytest.raises(ValueError):
        label_payload(10)
    assert decode_label_payload(np.unpackbits(np.array([0x1A], dtype=np.uint8)), payload_bits=8) is None
    assert decode_label_payload(np.unpackbits(np.array([0xFF], dtype=np.uint8)), payload_bits=8) is None
    assert decode_label_payload(np.array([0, 0, 0], dtype=np.uint8), payload_bits=8) is None


def test_learned_unit_is_validation_only_with_a_1000_row_contract() -> None:
    context = _context()
    unit = {"system": "learned", "bw_ratio": "r_1_6", "snr_db": -8, "train_seed": 0, "channel_seed": 0}
    evidence = learned_unit(context, unit, model=_StubModel(), config=load_w8_config("r_1_6", 0, 0), checkpoint_id="a" * 64)
    rows = evidence["per_image"]
    assert len(rows) == 1000
    validate_per_image(rows, expected_stable_ids=context.view.stable_ids, system="learned", bw_ratio="r_1_6", snr_db=-8)
    assert evidence["n_correct"] == 100 and evidence["n_total"] == 1000
    assert all(row["source_bytes"] is None for row in rows)
    # The stub logits prefer class 0, which every true label would only match once per ten.
    assert len({row["pred_label"] for row in rows}) == 1


def test_recon_ablation_scores_the_reconstruction_not_the_logits() -> None:
    context = _context()
    context.classifiers["clean"] = _StubClassifier()
    unit = {"system": "semantic_recon_ablation", "bw_ratio": "r_1_6", "snr_db": -8, "train_seed": 0, "channel_seed": 0}
    evidence = recon_ablation_unit(context, unit, model=_ReconstructionModel(), config=load_w8_config("r_1_6", 0, 0), checkpoint_id="b" * 64)
    rows = evidence["per_image"]
    assert len(rows) == 1000
    assert {row["pred_label"] for row in rows} == {7}
    assert evidence["binding"]["uses_djscc_logits"] is False


def test_bindings_resolve_and_only_the_future_prerequisites_are_pending() -> None:
    bindings = resolve_scope_bindings(Path("."))
    assert len(bindings) == 12
    pending = pending_states(bindings)
    assert any("papr_training" in item for item in pending)
    assert any("w10_jpeg_validation_selection" in item for item in pending)
    assert any("w10_er12_validation_selection" in item for item in pending)
    learned = [item for item in bindings if item["scope"]["role"] == "er1_headline_learned"][0]
    assert learned["checkpoint"]["state"] == "FROZEN"
    er9 = [item for item in bindings if item["scope"]["role"] == "er9_control_h4"][0]
    assert len(er9["checkpoint"]["per_snr_phy"]) == 21


def test_papr_constraint_is_installed_only_on_the_constrained_build() -> None:
    config = load_papr_config()
    constrained = build_papr_model(config, "cpu")
    assert isinstance(constrained.encoder.peak_constraint, PeakPowerConstraint)
    assert float(constrained.encoder.peak_constraint.max_papr_db) == 3.0  # literal-ok: AM-98 frozen cap
    assert PaprConstrainedTrainer._build_model is not None
    assert protected_counters()["papr_constrained_training"] == 0


def test_w10_papr_dispatch_uses_projected_config_identity_not_ordinary_w8(monkeypatch) -> None:
    papr_config = load_papr_config()
    ordinary_config = load_w8_config("r_1_6", 0, 0)
    assert run_config_hash(papr_config) != run_config_hash(ordinary_config)
    captured: dict[str, object] = {}

    class _Model:
        def load_state_dict(self, state, strict=True):
            captured["state"] = state
            captured["strict"] = strict
            return type("Incompatible", (), {"missing_keys": [], "unexpected_keys": []})()

        def eval(self):
            captured["eval"] = True
            return self

    monkeypatch.setattr(w10_dispatch, "build_papr_model", lambda config, device: (captured.update(config=config, device=device) or _Model()))
    monkeypatch.setattr(w10_dispatch, "_load_checkpoint_state", lambda path, expected: {})
    monkeypatch.setattr(w10_dispatch, "load_papr_config", lambda: papr_config)
    monkeypatch.setattr(w10_dispatch, "load_w8_config", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ordinary W8 loader used")))
    checkpoint = {
        "config_hash": run_config_hash(papr_config),
        "protocol_config_hash": papr_protocol_config_hash(papr_config),
        "papr_cap_db": float(get("evaluation.w10_papr_cap_db")),
        "papr_cap_compliant": True,
        "authority_id": "paprtrainingauth-synthetic",
        "selection_id": "paprselected-synthetic",
        "completion_id": "paprcompletion-synthetic",
        "checkpoint_path": "checkpoints/synthetic.pt",
        "checkpoint_id": "a" * 64,
    }
    model, config = w10_dispatch._load_papr_model(W10Execution(root=Path("."), device="cpu"), checkpoint)
    assert model is not None and config is papr_config
    assert captured["config"] is papr_config
    assert checkpoint["config_hash"] != run_config_hash(ordinary_config)


def test_w10_papr_evaluation_binding_carries_projected_config_and_protocol() -> None:
    config = load_papr_config()
    context = _context()
    unit = {
        "system": "learned_papr_constrained",
        "bw_ratio": "r_1_6",
        "snr_db": -8,
        "train_seed": 0,
        "channel_seed": 0,
    }
    evidence = learned_unit(
        context,
        unit,
        model=_StubModel(),
        config=config,
        checkpoint_id="a" * 64,
        papr_cap_db=3.0,
        protocol_config_hash=papr_protocol_config_hash(config),
    )
    assert evidence["binding"]["config_hash"] == run_config_hash(config)
    assert evidence["binding"]["protocol_config_hash"] == papr_protocol_config_hash(config)
    assert evidence["binding"]["config_hash"] != run_config_hash(load_w8_config("r_1_6", 0, 0))


def test_jpeg_codec_search_stays_within_budget() -> None:
    codec = JpegCodec()
    image = (np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3) % 251)
    budget = 900
    result = codec.encode_to_budget(image, canonical_pixels_sha256="a" * 64, budget_bytes=budget, encode_axis_px=64)
    assert result.feasible and result.emitted_byte_count is not None
    assert result.emitted_byte_count <= budget
    assert result.quality in codec.quality_grid
    tight = codec.encode_to_budget(image, canonical_pixels_sha256="a" * 64, budget_bytes=8, encode_axis_px=64)
    assert tight.feasible is False
