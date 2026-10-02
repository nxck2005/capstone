#!/usr/bin/env python3
"""``run_g12_campaign.py`` under recorded post-freeze execution corrections.

The freeze binds the runner's own bytes, so the runner is left unchanged and
this wrapper replaces only its freeze check.  Protected source may differ from
the frozen code commit solely as a committed correction note records it: the
exact set of changed paths, and for each its blob before (at the frozen code
commit, or absent) and after (at HEAD).  Every other check, the scope, the
bindings and the analysis are the runner's own.

    python tools/run_g12_campaign_corrected.py run --worker 0 --workers 4 --device cuda:0
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools")]

import g12_freeze  # noqa: E402
import run_g12_campaign as runner  # noqa: E402
from config.params import get  # noqa: E402

CORRECTIONS = "results/g12/post_freeze_corrections"


def _blob(commit: str, path: str) -> str | None:
    listed = runner._git("ls-tree", commit, "--", path)
    return listed.split()[2] if listed else None


def changed_paths(code_commit: str) -> list[str]:
    return sorted(filter(None, runner._git("diff", "--name-only", code_commit, "HEAD", "--", *g12_freeze.PROTECTED).splitlines()))


def correction_record(code_commit: str, head: str = "HEAD") -> list[dict[str, Any]]:
    return [{"path": path, "before_blob": _blob(code_commit, path), "after_blob": _blob(head, path)} for path in changed_paths(code_commit)]


def corrected_freeze(mode: runner.Mode) -> dict[str, Any]:
    if mode.rehearsal:
        return g12_freeze.build(code_commit=runner._git("rev-parse", "HEAD"))
    path = REPO / str(get("artifacts.freeze_manifest_file"))
    runner._require(path.is_file(), "no freeze manifest; G-12 has not been frozen")
    value = json.loads(path.read_bytes())
    rebuilt = g12_freeze.build(code_commit=value["code_commit"])
    runner._require(g12_freeze._encode(rebuilt) == path.read_bytes(), "freeze manifest does not reproduce from this tree")
    notes = sorted((REPO / CORRECTIONS).glob("*.json"))
    runner._require(bool(notes), "protected source differs from the freeze and no correction is recorded")
    runner._require(bool(runner._git("ls-files", "--", str(notes[-1].relative_to(REPO)))), "correction note is not committed")
    note = json.loads(notes[-1].read_bytes())
    runner._require(note["freeze_id"] == value["freeze_id"] and note["frozen_code_commit"] == value["code_commit"], "correction note belongs to another freeze")
    runner._require(note["changes"] == correction_record(value["code_commit"]), "protected source differs from the recorded correction")
    dirty = runner._git("status", "--porcelain", "--untracked-files=no", "--", *g12_freeze.PROTECTED, str(path.relative_to(REPO)), CORRECTIONS)
    runner._require(not dirty, f"working tree differs from the committed freeze and correction:\n{dirty}")
    return value


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """``g12_analysis.mcnemar_exact`` in exact integer arithmetic (correction 2).

    The frozen version divides by ``2.0**total``, which overflows a float once
    ``total`` exceeds 1023 discordant images — reachable on the 3925-image test
    split but not on the 1000-image validation split it was rehearsed on.
    """

    from fractions import Fraction  # noqa: PLC0415
    from math import comb  # noqa: PLC0415

    total = a_only + b_only
    if total == 0:
        return 1.0
    tail = Fraction(sum(comb(total, k) for k in range(0, min(a_only, b_only) + 1)), 2**total)
    return float(min(Fraction(1), 2 * tail))


def _install_corrections() -> None:
    import evaluation.g12_analysis as analysis  # noqa: PLC0415

    runner._freeze = corrected_freeze
    analysis.mcnemar_exact = mcnemar_exact


_install_corrections()

if __name__ == "__main__":
    raise SystemExit(runner.main())
