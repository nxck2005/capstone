"""Reusable deterministic training primitives for post-W5 learned runs.

The historical W5 engine is intentionally not edited: its source bytes and
artifact meaning are part of the W5 evidence boundary.  W7 composes these
primitives with its own policy and compact checkpoint layer instead of
silently changing W5's smoke semantics.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


class TrainingCoreHold(RuntimeError):
    """Fail-closed deterministic-core violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingCoreHold(message)


def canonical_bytes(value: Any) -> bytes:
    """Canonical finite JSON bytes used by compact companion records."""

    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def optimizer_parameters(optimizer: torch.optim.Optimizer) -> list[nn.Parameter]:
    """Return every parameter owned by an optimizer, exactly once."""

    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    identities = [id(parameter) for parameter in parameters]
    _require(
        len(identities) == len(set(identities)),
        "optimizer owns a parameter more than once",
    )
    return parameters


def gradient_status(parameters: Iterable[nn.Parameter]) -> dict[str, int | bool]:
    """Classify all gradients in one named module or optimizer ownership set."""

    parameter_list = list(parameters)
    gradients = [parameter.grad for parameter in parameter_list if parameter.grad is not None]
    return {
        "parameter_count": len(parameter_list),
        "gradient_count": len(gradients),
        "present": bool(gradients),
        "finite": bool(gradients)
        and all(torch.isfinite(gradient).all().item() for gradient in gradients),
        "nonzero": bool(gradients)
        and any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients),
    }


def divide_optimizer_gradients(
    optimizer: torch.optim.Optimizer,
    denominator: int,
) -> None:
    """Apply sample-weighted effective-batch normalisation in place."""

    _require(isinstance(denominator, int) and not isinstance(denominator, bool), "gradient denominator must be an integer")
    _require(denominator > 0, "gradient denominator must be positive")
    for parameter in optimizer_parameters(optimizer):
        if parameter.grad is not None:
            parameter.grad.div_(denominator)


@dataclass(frozen=True)
class OptimizerUpdate:
    """Authenticated result of one attempted optimizer update."""

    applied: bool
    scaler_scale_before: float | None
    scaler_scale_after: float | None
    optimizer_gradients: dict[str, int | bool]


def apply_optimizer_update(
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    *,
    denominator: int,
) -> OptimizerUpdate:
    """Unscale, normalise, classify and apply one optimizer update.

    GradScaler's own ``step`` remains the authority for CUDA updates, but the
    decision is recorded from the finiteness of *every* optimizer-owned
    gradient.  This is the W5 repair carried into W7 without copying W5's
    detailed trace into every W7 checkpoint.
    """

    scale_before: float | None = None
    scale_after: float | None = None
    if scaler is None:
        divide_optimizer_gradients(optimizer, denominator)
    else:
        scale_before = float(scaler.get_scale())
        scaler.unscale_(optimizer)
        divide_optimizer_gradients(optimizer, denominator)

    status = gradient_status(optimizer_parameters(optimizer))
    _require(status["present"], "optimizer update has no gradients")
    finite = bool(status["finite"])

    if scaler is None:
        _require(finite, "non-finite optimizer-owned gradient without GradScaler")
        optimizer.step()
        applied = True
    else:
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        if finite:
            _require(
                scale_after >= float(scale_before),
                "GradScaler backed off despite finite optimizer-owned gradients",
            )
            applied = True
        else:
            _require(
                scale_after < float(scale_before),
                "GradScaler did not skip/back off after a non-finite optimizer-owned gradient",
            )
            applied = False

    return OptimizerUpdate(
        applied=applied,
        scaler_scale_before=scale_before,
        scaler_scale_after=scale_after,
        optimizer_gradients=status,
    )


def module_gradient_status(model: nn.Module) -> dict[str, dict[str, int | bool]]:
    """Return the named-region checks used by W7's compact epoch record."""

    encoder = getattr(model, "encoder", None)
    decoder = getattr(model, "decoder", None)
    _require(encoder is not None and decoder is not None, "DJSCC model regions are missing")
    return {
        "encoder": gradient_status(encoder.parameters()),
        "reconstruction_head": gradient_status(decoder.reconstruction_head.parameters()),
        "task_head": gradient_status(decoder.task_head.parameters()),
    }


def state_tree_sha256(value: Any) -> str:
    """Stable hash for resume tests without serialising a cumulative history."""

    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            digest.update(b"tensor\0")
            digest.update(str(item.dtype).encode("ascii"))
            digest.update(str(tuple(item.shape)).encode("ascii"))
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda candidate: str(candidate)):
                visit(str(key))
                visit(item[key])
        elif isinstance(item, list | tuple):
            digest.update(b"sequence\0")
            for nested in item:
                visit(nested)
        else:
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(b"\0")
            digest.update(repr(item).encode("utf-8"))
            digest.update(b"\0")

    visit(value)
    return digest.hexdigest()


def model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
