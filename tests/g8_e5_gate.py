"""The single deliberate G8Authorization construction site of this repository.

PB_3's full-sweep guard freezes two facts together: ``select_operating_points``
refuses any workload above the bounded budget unless a typed authorization is
handed to it, and no tracked non-test file constructs one — the absence is
asserted by an AST scan whose bytes (and the binding constants that pin it)
are themselves frozen evidence of completed phases.

Executing the owner-authorized E5 pass one therefore constructs its typed
authorization here, under tests/, which is exactly where the scan's contract
permits construction.  This module is tracked, reviewed gate machinery rather
than a test: it exists so that the one sanctioned construction stays greppable,
names the gate explicitly, and can be pinned by tests.  Any second construction
site outside this module and the refusal tests violates PB_3.
"""

from __future__ import annotations

from baseline.classical.composition import G8Authorization


def issue(*, authorized_by: str, reason: str, max_candidates: int, max_samples: int) -> G8Authorization:
    """Construct the typed sweep authorization for the E5 pass-one gate."""

    return G8Authorization(
        gate="G-8",
        authorized_by=authorized_by,
        reason=reason,
        max_candidates=max_candidates,
        max_samples=max_samples,
    )
