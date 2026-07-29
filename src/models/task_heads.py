"""Registry and default downstream task head (SR-15)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn


TaskHeadFactory = Callable[..., nn.Module]
DEFAULT_TASK_HEAD = "image_classification"
_TASK_HEADS: dict[str, TaskHeadFactory] = {}


class ImageClassificationHead(nn.Module):
    """Global-average-pooling image classification head."""

    def __init__(self, input_channels: int, classes: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(input_channels, classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(torch.flatten(self.pool(features), 1))


def register_task_head(name: str, factory: TaskHeadFactory) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("task-head name must be a non-empty string")
    if not callable(factory):
        raise TypeError("task-head factory must be callable")
    if name in _TASK_HEADS:
        raise ValueError(f"task head already registered: {name}")
    _TASK_HEADS[name] = factory


def build_task_head(name: str, **kwargs: Any) -> nn.Module:
    try:
        factory = _TASK_HEADS[name]
    except KeyError:
        raise ValueError(
            f"unknown task head {name!r}; registered={sorted(_TASK_HEADS)}"
        ) from None
    head = factory(**kwargs)
    if not isinstance(head, nn.Module):
        raise TypeError(f"task-head factory {name!r} did not return torch.nn.Module")
    return head


def task_head_names() -> tuple[str, ...]:
    return tuple(sorted(_TASK_HEADS))


register_task_head(DEFAULT_TASK_HEAD, ImageClassificationHead)
