"""Offline demo, G-12 edition: the exhibition dashboard over the published test-split curves.

Built on the original demo backend (demo/backend/app.py), whose verified live
inference classes and input validation are reused unchanged. Four routes differ:

- /api/metadata and /api/chart describe and serve the G-12 test curves
  (presentation-results/data/g12_test_curves.csv), each point checked against
  results/g12/results.csv at start-up;
- /api/images and /api/infer serve the fixed ten-image test gallery
  (demo_shared/test_gallery.py, AM-101) instead of the four training examples,
  and every live outcome is returned beside what G-12 recorded for that image;
- /api/test-split and /api/test-images/{id} let a visitor browse all 3,925
  test images and pick any one, when the extracted dataset and per-image
  records are present.

Nothing here trains a model or recomputes a reported metric.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response

from demo.backend import app as base  # first: it puts src/ on sys.path
from demo_shared.results import TEST_IMAGES, Key, Point, load_curves
from demo_shared.test_gallery import TestGallery, transmit

from config.params import get  # noqa: E402

RATIOS = ("r_1_6", "r_1_24")
# Paper and exhibition colours for the two systems the live receivers show.
SERIES = (
    ("learned", "own_task_head", "Semantic DJSCC", "#0072B2"),
    ("classical_adaptive", "artifact_finetuned", "Classical adaptive (artifact scorer)", "#D55E00"),
)
REPLACED = {"/api/metadata", "/api/chart", "/api/images", "/api/infer", "/api/examples", "/api/examples/{name}"}


def load_g12() -> tuple[list[float], dict[Key, list[Point]]]:
    return load_curves(tuple((system, scorer, ratio) for ratio in RATIOS for system, scorer, *_ in SERIES))


def create_app(*, evidence: base.Evidence | None = None,
               g12: tuple[list[float], dict[Key, list[Point]]] | None = None,
               gallery: TestGallery | None = None) -> FastAPI:
    evidence = evidence or base.Evidence()
    gallery = gallery or TestGallery()
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
            "portable_example_count": len(gallery.images),
            "portable_examples_split": "test",
            "gallery_rule": "one test image per class: the smallest stable sample ID (AM-101)",
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

    @app.get("/api/images")
    def images() -> dict[str, Any]:
        return {"split": "test", "images": [
            {"id": image_id, "label": row["label"], "truth_label": row["label"],
             "thumbnail_url": f"/api/test-examples/{row['id']}", "split": "test"}
            for image_id, row in gallery.images.items()
        ]}

    @app.get("/api/test-examples/{name}")
    def test_example(name: str) -> Response:
        image_id = gallery.by_slug(name)
        if image_id is None:
            raise HTTPException(status_code=404, detail="unknown test-gallery image")
        return Response(content=gallery.png(image_id), media_type="image/png",
                        headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"})

    @app.get("/api/test-split")
    def test_split(page: int = Query(default=0, ge=0), size: int = Query(default=40, ge=1, le=100),
                   label: int | None = Query(default=None, ge=0, le=9)) -> dict[str, Any]:
        reason = gallery.full_split()
        if reason is not None:
            return {"available": False, "reason": reason, "total": 0, "page": 0, "pages": 0, "images": []}
        ids = [i for i in gallery.test_ids if label is None or gallery.label(i) == label]
        pages = max(1, -(-len(ids) // size))
        page = min(page, pages - 1)
        return {"available": True, "reason": None, "total": len(ids), "page": page, "pages": pages, "size": size,
                "images": [{"id": i, "label": base.CLASS_NAMES[gallery.label(i)], "truth_label": base.CLASS_NAMES[gallery.label(i)],
                            "thumbnail_url": f"/api/test-images/{i}", "split": "test"}
                           for i in ids[page * size:(page + 1) * size]]}

    @app.get("/api/test-images/{image_id}")
    def test_image(image_id: str) -> Response:
        if len(image_id) != 16 or any(c not in "0123456789abcdef" for c in image_id) or not gallery.known(image_id):
            raise HTTPException(status_code=404, detail="unknown test image")
        return Response(content=gallery.png(image_id), media_type="image/png",
                        headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"})

    @app.post("/api/infer")
    def infer(request: base.InferRequest) -> dict[str, Any]:
        if not gallery.known(request.image_id):
            raise HTTPException(status_code=404, detail="unknown test image")
        if float(request.snr_db) not in grid or request.ratio not in RATIOS:
            raise HTTPException(status_code=422, detail="SNR or bandwidth ratio outside the G-12 grid")
        if request.ratio != "r_1_6":
            unavailable = {"status": "unavailable", "predicted_label": None, "confidence": None, "image_url": None,
                           "detail": "Live inference runs at 1/6 only; G-12 per-image records are shown for 1/6."}
            arms = {"learned": unavailable, "classical": unavailable}
        else:
            arms = transmit(evidence, gallery, request.image_id, request.snr_db)
        return {
            "split": "test", "image_id": request.image_id, "snr_db": request.snr_db, "ratio": request.ratio,
            "input_image_url": f"/api/test-images/{request.image_id}",
            **arms, "inference_mode": "fresh_cpu_display_output_checked_against_g12_record",
        }

    return app


app = create_app()
