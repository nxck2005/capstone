"""Offline demo, G-12 edition: the exhibition dashboard over the published test-split curves.

Images, live inference and every safety check are the original demo backend's
(demo/backend/app.py), reused unchanged. Only two routes differ: /api/metadata
and /api/chart describe and serve the G-12 test curves
(presentation-results/data/g12_test_curves.csv) instead of the W10 validation
units. Every served point is checked against results/g12/results.csv at
start-up. Nothing here trains a model or recomputes a reported metric.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query

from demo.backend import app as base  # first: it puts src/ on sys.path
from demo_streamlit.results import TEST_IMAGES, Key, Point, load_curves

from config.params import get  # noqa: E402

RATIOS = ("r_1_6", "r_1_24")
# Paper and exhibition colours for the two systems the live receivers show.
SERIES = (
    ("learned", "own_task_head", "Semantic DJSCC", "#0072B2"),
    ("classical_adaptive", "artifact_finetuned", "Classical adaptive (artifact scorer)", "#D55E00"),
)
REPLACED = {"/api/metadata", "/api/chart"}


def load_g12() -> tuple[list[float], dict[Key, list[Point]]]:
    return load_curves(tuple((system, scorer, ratio) for ratio in RATIOS for system, scorer, *_ in SERIES))


def create_app(*, evidence: base.Evidence | None = None,
               g12: tuple[list[float], dict[Key, list[Point]]] | None = None) -> FastAPI:
    evidence = evidence or base.Evidence()
    grid, curves = g12 or load_g12()
    if [float(v) for v in get("channel.test_snr_grid_db")] != grid:
        raise RuntimeError("G-12 curves do not use the specification's SNR grid")
    app = base.create_app(evidence=evidence)
    app.title = "Capstone G-12 test demo"
    app.router.routes = [route for route in app.router.routes if getattr(route, "path", None) not in REPLACED]

    @app.get("/api/metadata")
    def metadata() -> dict[str, Any]:
        return {
            "dataset": "imagenette160", "split": "test", "n_images": TEST_IMAGES,
            "channel": "simulated AWGN", "snr_grid_db": grid, "default_snr_db": -8,
            "ratios": [
                {"id": ratio, "label": ratio.replace("r_1_", "1/"),
                 "channel_uses": int(get(f"bandwidth.k_symbols.imagenette160.{ratio}"))}
                for ratio in RATIOS
            ],
            "evidence_label": f"G-12 test split · {TEST_IMAGES:,} images",
            "systems": [{"system": system, "bw_ratio": ratio} for ratio in RATIOS for system, *_ in SERIES],
            "source": "G-12 closeout (results/g12/results.csv) via presentation-results/data/g12_test_curves.csv",
            "inference_available": evidence.live.available or evidence.classical.available,
            "inference_reason": None if evidence.live.available and evidence.classical.available else "One or more frozen inference checkpoints are unavailable; each arm reports its own status.",
            "live_inference_arms": [
                {"system": system, "ratio": "r_1_6"}
                for system, available in (("learned", evidence.live.available), ("classical_adaptive", evidence.classical.available))
                if available
            ],
            "classical_inference_available": evidence.classical.available,
            "portable_example_count": len(evidence.examples),
            "portable_examples_split": "train",
        }

    @app.get("/api/chart")
    def chart(ratio: str = Query(default="r_1_6")) -> dict[str, Any]:
        if ratio not in RATIOS:
            raise HTTPException(status_code=422, detail="ratio outside the G-12 chart")
        series = []
        for system, scorer, label, color in SERIES:
            points = curves[(system, scorer, ratio)]
            series.append({"id": system, "label": label, "color": color, "cells": points[0].cells, "points": [
                # Intervals are drawn only for curves averaged over more than one seed cell, as in the paper.
                {"snr_db": p.snr_db, "accuracy": p.accuracy, "coverage": p.coverage,
                 **({"ci_low": p.ci_low, "ci_high": p.ci_high} if p.cells > 1 else {})}
                for p in points
            ]})
        return {
            "ratio": ratio, "series": series, "split": "test", "n_images": TEST_IMAGES,
            "metric": "top-1 accuracy, mean over seed cells; 95% image-bootstrap interval",
            "source": "results/g12/results.csv",
        }

    return app


app = create_app()
