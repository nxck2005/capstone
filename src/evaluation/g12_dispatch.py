"""G-12 dispatcher: one frozen binding per arm × cell, through the W10 evaluators.

The evaluators are the W10 ones unchanged; G-12 changes only what they are
pointed at — the guarded test view, the unit's zipped seed cell, the cell's
checkpoint, and the batched classical transport (shown identical to the
per-image path on committed W10 units).  This module never opens the test split
itself: the caller supplies ``context.view`` from ``data.test_access``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from baseline.ldpc.transport import build_packet_plan
from config.params import get
from config.run_config import load_experiment
from evaluation.er9_search import configured_phy_candidates
from evaluation.g12_bindings import binding_for
from evaluation.g12_scope import arm_for
from evaluation.w10_backends import (
    W10Execution,
    er2_unit,
    er9_unit,
    label_bound_unit,
    learned_unit,
    recon_ablation_unit,
)
from evaluation.w10_classical import classical_unit
from evaluation.w10_dispatch import ER9_CONFIG, _load_checkpoint_state, _load_er2_model, _load_papr_model
from models.djscc import build_djscc
from models.er9_digital import build_er9_model
from training.papr_constrained import papr_protocol_config_hash  # noqa: F401  (bound through the PAPR checkpoint)
from training.w8_protocol import load_w8_config


class G12DispatchError(RuntimeError):
    """A G-12 unit could not be routed to exactly its frozen evaluator."""


def _path(root: Path, relative: str) -> Path:
    path = Path(str(relative))
    return path if path.is_absolute() else Path(root) / path


def _load_w8(root: Path, device: Any, ratio: str, cell: int, checkpoint: Mapping[str, Any]) -> tuple[torch.nn.Module, Any]:
    config = load_w8_config(ratio, cell, cell)
    model = build_djscc(config, device=device)
    state = _load_checkpoint_state(_path(root, checkpoint["checkpoint_path"]), str(checkpoint["checkpoint_id"]))
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise G12DispatchError(f"W8 {ratio} cell {cell} checkpoint did not load strictly")
    model.eval()
    return model, config


def _load_er9(root: Path, device: Any, cell: int, checkpoint: Mapping[str, Any], *, transmit_dim: int, quantiser_bits: int, epoch: int, sha256: str, runtime_root: str, entropy_table: Mapping[str, Any] | None) -> dict[str, Any]:
    from evaluation.er9_campaign import fit_entropy_model  # noqa: PLC0415

    config = load_experiment(ER9_CONFIG, train_seed=cell, channel_seed=cell)
    model = build_er9_model(config, transmit_dim=transmit_dim, quantiser_bits=quantiser_bits, device=device)
    state = _load_checkpoint_state(Path(root) / runtime_root / "epochs" / f"epoch-{epoch:04d}" / "checkpoint.pt", sha256)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise G12DispatchError(f"ER-9 cell {cell} checkpoint did not load strictly")
    model.eval()
    entropy, record = fit_entropy_model(model, config, device=device, num_workers=0)
    if entropy_table is not None and record["table"] != entropy_table:
        raise G12DispatchError(f"ER-9 cell {cell} refit entropy table differs from its closed record")
    return {
        "model": model,
        "config": config,
        "entropy": entropy,
        "entropy_record": record,
        "dimension": transmit_dim,
        "quantiser_bits": quantiser_bits,
        "checkpoint_id": sha256,
    }


def dispatch(root: Path, *, device: torch.device | str, bindings: list[Mapping[str, Any]], view: Any, split: str = "test", j2k_cache_dir: str = "checkpoints/g12_campaign/j2k_cache") -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    """Return ``backend(unit)`` for the frozen G-12 bindings over ``view``.

    ``split="val"`` with a validation view is the pre-freeze rehearsal of this
    exact routing; the campaign always uses the guarded test view.
    """

    root = Path(root)
    contexts: dict[int, W10Execution] = {}
    cache: dict[str, Any] = {}

    def context_for(cell: int) -> W10Execution:
        if cell not in contexts:
            contexts[cell] = W10Execution(
                root=root,
                device=device,
                view=view,
                split=split,
                train_seed=cell,
                channel_seed=cell,
                batched_classical=True,
                j2k_cache_dir=j2k_cache_dir,
            )
            # Classifiers are seed-free; share them across cell contexts.
            if contexts.get(0) is not None and cell != 0:
                contexts[cell].classifiers = contexts[0].classifiers
                contexts[cell].builders = contexts[0].builders
        return contexts[cell]

    def cached(key: str, loader: Callable[[], Any]) -> Any:
        if key not in cache:
            cache[key] = loader()
        return cache[key]

    def backend(unit: Mapping[str, Any]) -> Mapping[str, Any]:
        arm = arm_for(str(unit["role"]))
        cell = int(unit["train_seed"])
        if int(unit["channel_seed"]) != cell:
            raise G12DispatchError("G-12 cells are zipped; train and channel seed differ")
        bound = binding_for(bindings, arm.role, cell)
        checkpoint = bound["checkpoint"]
        context = context_for(cell)
        full_unit = dict(unit)
        if arm.backend == "learned":
            if arm.system == "learned_papr_constrained":
                model, config = cached("papr", lambda: _load_papr_model(context, checkpoint))
                return learned_unit(context, full_unit, model=model, config=config, checkpoint_id=str(checkpoint["checkpoint_id"]), papr_cap_db=float(get("evaluation.w10_papr_cap_db")), protocol_config_hash=str(checkpoint["protocol_config_hash"]))
            model, config = cached(f"w8:{arm.bw_ratio}:{cell}", lambda: _load_w8(root, device, arm.bw_ratio, cell, checkpoint))
            return learned_unit(context, full_unit, model=model, config=config, checkpoint_id=str(checkpoint["checkpoint_id"]), papr_cap_db=None)
        if arm.backend == "recon_ablation":
            model, config = cached(f"w8:{arm.bw_ratio}:{cell}", lambda: _load_w8(root, device, arm.bw_ratio, cell, checkpoint))
            return recon_ablation_unit(context, full_unit, model=model, config=config, checkpoint_id=str(checkpoint["checkpoint_id"]))
        if arm.backend == "er2_randomised":
            model, config = cached("er2", lambda: _load_er2_model(context, checkpoint))
            return er2_unit(context, full_unit, model=model, config=config, checkpoint_id=str(checkpoint["checkpoint_sha256"]), task_head_identity=str(checkpoint["task_head_identity"]))
        if arm.backend in ("er9_digital", "label_bound"):
            assets = cached(f"er9:{cell}", lambda: _load_er9(
                root, device, cell, checkpoint,
                transmit_dim=int(checkpoint.get("selected_pair", {}).get("transmit_dim", 2048)),  # literal-ok: closed Stage-2 D2048/b2
                quantiser_bits=int(checkpoint.get("selected_pair", {}).get("quantiser_bits", 2)),  # literal-ok: closed Stage-2 D2048/b2
                epoch=int(checkpoint["selected_epoch"]),
                sha256=str(checkpoint["selected_checkpoint_sha256"]),
                runtime_root=str(checkpoint["runtime_root"]),
                entropy_table=checkpoint.get("entropy_table"),
            ))
            if "phy_candidates" not in assets:
                assets["phy_candidates"] = [
                    {"modulation": str(modulation), "ldpc_rate": str(rate), "packet": packet}
                    for modulation, rate, packet in configured_phy_candidates(int(assets["config"].resolved["k"]))
                ]
            if arm.backend == "er9_digital":
                selection_phy = {float(item["snr_db"]): item for item in checkpoint["per_snr_phy"]}
                return er9_unit(context, full_unit, assets=assets, selection_phy=selection_phy)
            return label_bound_unit(context, full_unit, selection=bound["selection"], assets=assets)
        if arm.backend == "er9_low_rate":
            assets = cached("er9_low_rate", lambda: _load_er9(
                root, device, 0, checkpoint,
                transmit_dim=int(checkpoint["transmit_dim"]),
                quantiser_bits=int(checkpoint["quantiser_bits"]),
                epoch=int(checkpoint["checkpoint_epoch"]),
                sha256=str(checkpoint["checkpoint_sha256"]),
                runtime_root=str(checkpoint["runtime_root"]),
                entropy_table=None,
            ))
            packet = build_packet_plan(int(assets["config"].resolved["k"]), str(checkpoint["modulation"]), str(checkpoint["ldpc_rate"]))
            assets["phy_candidates"] = [{"modulation": str(checkpoint["modulation"]), "ldpc_rate": str(checkpoint["ldpc_rate"]), "packet": packet}]
            selection_phy = {float(unit["snr_db"]): {"modulation": str(checkpoint["modulation"]), "ldpc_rate": str(checkpoint["ldpc_rate"])}}
            return er9_unit(context, full_unit, assets=assets, selection_phy=selection_phy)
        if arm.backend == "classical":
            return classical_unit(context, full_unit, root=root, checkpoint_id=str(checkpoint.get("kind", "untrained")), binding=bound["selection"], codec_kind="jpeg2000")
        if arm.backend == "jpeg_secondary":
            selection = bound["selection"]
            point = next(item for item in selection["selections"] if float(item["snr_db"]) == float(unit["snr_db"]))
            return classical_unit(context, full_unit, root=root, checkpoint_id="not_applicable_untrained_codec", binding={"kind": "w10_jpeg_validation_selection", "selection_id": selection["selection_id"], "selections": selection["selections"]}, codec_kind="jpeg", quality=int(point["quality"]))
        raise G12DispatchError(f"unsupported G-12 backend: {arm.backend}")

    return backend


__all__ = ["G12DispatchError", "dispatch"]
