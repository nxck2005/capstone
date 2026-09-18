"""Production W10 dispatcher: one explicit backend route per scope role (AM-98).

The dispatcher is the only place a W10 unit resolves a model, checkpoint,
classifier or operating-point selection.  It fails closed for an unsupported
system and never falls back to another evaluator.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from channels.power import PeakPowerConstraint
from config.params import get
from config.run_config import load_experiment
from evaluation.er9_search import configured_phy_candidates
from evaluation.w10_backends import (
    W10Execution,
    er2_unit,
    er9_unit,
    label_bound_unit,
    learned_unit,
    recon_ablation_unit,
)
from evaluation.w10_bindings import resolve_scope_bindings
from evaluation.w10_classical import classical_unit
from evaluation.w10_scope import entry_for
from models.djscc import build_djscc
from models.er9_digital import build_er9_model
from training.w8_protocol import load_w8_config

ER2_CONFIG = "configs/learned-er2-randomized-pascal-v4.yaml"
ER9_CONFIG = "configs/er9-digital-pascal-v4.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_checkpoint_state(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"W10 checkpoint is missing or unsafe: {path}")
    if _sha256(path) != expected_sha256:
        raise RuntimeError(f"W10 checkpoint bytes differ: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model_state" not in payload:
        raise RuntimeError("W10 checkpoint carries no model state")
    return dict(payload["model_state"])


def _load_w8_model(context: W10Execution, ratio: str, checkpoint: Mapping[str, Any], *, papr_cap_db: float | None) -> tuple[torch.nn.Module, Any]:
    config = load_w8_config(ratio, 0, 0)
    constraint = None if papr_cap_db is None else PeakPowerConstraint(float(papr_cap_db))
    model = build_djscc(config, peak_constraint=constraint, device=context.device)
    state = _load_checkpoint_state(Path(str(checkpoint["checkpoint_path"])), str(checkpoint["checkpoint_id"]))
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("W10 W8 checkpoint did not load strictly")
    model.eval()
    return model, config


def _load_er2_model(context: W10Execution, checkpoint: Mapping[str, Any]) -> tuple[torch.nn.Module, Any]:
    config = load_experiment(ER2_CONFIG, train_seed=0, channel_seed=0)
    model = build_djscc(config, device=context.device)
    state = _load_checkpoint_state(Path(str(checkpoint["checkpoint_path"])), str(checkpoint["checkpoint_sha256"]))
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("W10 ER-2 checkpoint did not load strictly")
    model.eval()
    return model, config


def _load_er9_assets(context: W10Execution, checkpoint: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    config = load_experiment(ER9_CONFIG, train_seed=0, channel_seed=0)
    dimension = 2048  # literal-ok: closed Stage-2 D2048/b2 selection
    quantiser_bits = 2
    model = build_er9_model(config, transmit_dim=dimension, quantiser_bits=quantiser_bits, device=context.device)
    runtime_root = Path(str(root)) / str(checkpoint["runtime_root"])
    epoch = int(checkpoint["selected_epoch"])
    epoch_dir = runtime_root / "epochs" / f"epoch-{epoch:04d}"
    state = _load_checkpoint_state(epoch_dir / "checkpoint.pt", str(checkpoint["selected_checkpoint_sha256"]))
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("W10 ER-9 checkpoint did not load strictly")
    model.eval()
    from evaluation.er9_campaign import fit_entropy_model

    entropy, entropy_record = fit_entropy_model(model, config, device=context.device, num_workers=0)
    candidates = [
        {"modulation": str(modulation), "ldpc_rate": str(rate), "packet": packet}
        for modulation, rate, packet in configured_phy_candidates(int(config.resolved["k"]))
    ]
    return {
        "model": model,
        "config": config,
        "entropy": entropy,
        "entropy_record": entropy_record,
        "dimension": dimension,
        "quantiser_bits": quantiser_bits,
        "checkpoint_id": str(checkpoint["selected_checkpoint_sha256"]),
        "phy_candidates": candidates,
    }


def dispatch(root: Path, *, device: torch.device | str, authority: Mapping[str, Any]) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    """Build the production backend callable for one frozen W10 authority."""

    bindings = {str(item["scope"]["role"]): item for item in authority["bindings"]}
    live = {str(item["scope"]["role"]): item for item in resolve_scope_bindings(root)}
    if set(bindings) != set(live):
        raise RuntimeError("W10 authority roles differ from the live scope")
    context = W10Execution(root=root, device=device)

    def backend(unit: Mapping[str, Any]) -> Mapping[str, Any]:
        entry = entry_for(str(unit["system"]), str(unit["bw_ratio"]))
        if entry.role not in bindings:
            raise RuntimeError(f"W10 unit has no authority binding: {entry.role}")
        bound = bindings[entry.role]
        current = live[entry.role]
        if bound["binding_id"] != current["binding_id"]:
            raise RuntimeError(f"W10 frozen binding drifted for {entry.role}")
        checkpoint = bound["checkpoint"]
        if entry.backend == "learned":
            papr_cap = None if entry.system != "learned_papr_constrained" else float(get("evaluation.w10_papr_cap_db"))
            key = f"learned:{entry.bw_ratio}:{entry.system}"
            model, config = context.model(key, lambda: _load_w8_model(context, entry.bw_ratio, checkpoint, papr_cap_db=papr_cap))
            return learned_unit(context, unit, model=model, config=config, checkpoint_id=str(checkpoint["checkpoint_id"]), papr_cap_db=papr_cap)
        if entry.backend == "recon_ablation":
            key = f"learned:{entry.bw_ratio}:recon"
            model, config = context.model(key, lambda: _load_w8_model(context, entry.bw_ratio, checkpoint, papr_cap_db=None))
            return recon_ablation_unit(context, unit, model=model, config=config, checkpoint_id=str(checkpoint["checkpoint_id"]))
        if entry.backend == "er2_randomised":
            model, config = context.model("er2", lambda: _load_er2_model(context, checkpoint))
            return er2_unit(context, unit, model=model, config=config, checkpoint_id=str(checkpoint["checkpoint_sha256"]), task_head_identity=str(checkpoint["task_head_identity"]))
        if entry.backend in ("er9_digital", "label_bound"):
            assets = context.er9 or _load_er9_assets(context, checkpoint, root=root)
            context.er9 = assets
            if entry.backend == "er9_digital":
                selection_phy = {float(item["snr_db"]): item for item in checkpoint["per_snr_phy"]}
                return er9_unit(context, unit, assets=assets, selection_phy=selection_phy)
            return label_bound_unit(context, unit, selection=bound["selection"], assets=assets)
        if entry.backend == "classical":
            return classical_unit(context, unit, root=root, checkpoint_id=str(checkpoint.get("kind", "untrained")), binding=bound["selection"], codec_kind="jpeg2000")
        if entry.backend == "jpeg_secondary":
            selection = bound["selection"]
            point = next(item for item in selection["selections"] if float(item["snr_db"]) == float(unit["snr_db"]))
            return classical_unit(context, unit, root=root, checkpoint_id="not_applicable_untrained_codec", binding={"kind": "w10_jpeg_validation_selection", "selection_id": selection["selection_id"], "selections": selection["selections"]}, codec_kind="jpeg", quality=int(point["quality"]))
        raise RuntimeError(f"unsupported W10 backend: {entry.backend}")

    return backend


__all__ = ["dispatch"]
