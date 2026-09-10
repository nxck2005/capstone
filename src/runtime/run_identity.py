"""Model-identity counting for fresh, resumed and promoted W9 runs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class RunDisposition(StrEnum):
    FRESH_STARTED = "fresh_started"
    RESUMED_INCOMPLETE = "resumed_incomplete"
    EXISTING_COMPLETE_REUSED = "existing_complete_reused"
    PROMOTED_SEARCH_RUN = "promoted_search_run"


@dataclass(frozen=True)
class RunIdentityAccounting:
    scientific_model_identity: str
    disposition: RunDisposition
    process_starts: int
    scientific_model_count: int


def classify_run_identity(
    *,
    model_identity: str,
    runtime_exists: bool,
    terminal_exists: bool,
    promoted_from_search: bool = False,
    process_starts: int = 1,
) -> RunIdentityAccounting:
    if not model_identity:
        raise ValueError("scientific model identity must be non-empty")
    if process_starts <= 0:
        raise ValueError("process_starts must be positive")
    if promoted_from_search:
        disposition = RunDisposition.PROMOTED_SEARCH_RUN
    elif not runtime_exists:
        disposition = RunDisposition.FRESH_STARTED
    elif terminal_exists:
        disposition = RunDisposition.EXISTING_COMPLETE_REUSED
    else:
        disposition = RunDisposition.RESUMED_INCOMPLETE
    return RunIdentityAccounting(
        scientific_model_identity=model_identity,
        disposition=disposition,
        process_starts=process_starts,
        scientific_model_count=1,
    )


def terminal_run_counts(accounting: RunIdentityAccounting) -> dict[str, Any]:
    """Machine-readable counters that count model identities, not launches."""

    return {
        "fresh_started": int(accounting.disposition is RunDisposition.FRESH_STARTED),
        "resumed_incomplete": int(accounting.disposition is RunDisposition.RESUMED_INCOMPLETE),
        "existing_complete_reused": int(accounting.disposition is RunDisposition.EXISTING_COMPLETE_REUSED),
        "promoted_search_run": int(accounting.disposition is RunDisposition.PROMOTED_SEARCH_RUN),
        "process_starts": accounting.process_starts,
        "scientific_model_count": accounting.scientific_model_count,
        "scientific_model_identity": accounting.scientific_model_identity,
    }


__all__ = ["RunDisposition", "RunIdentityAccounting", "classify_run_identity", "terminal_run_counts"]
