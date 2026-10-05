"""G-12 demo adapter: test-curve routes over the original demo's reused routes (not scientific evaluation)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from demo.backend.app import Evidence
from demo_g12.backend.app import create_app, load_g12
from demo_streamlit.results import TEST_IMAGES


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(evidence=Evidence(), g12=load_g12()))


def test_metadata_describes_the_test_split(client: TestClient) -> None:
    info = client.get("/api/metadata").json()
    assert (info["split"], info["n_images"], info["default_snr_db"]) == ("test", TEST_IMAGES, -8)
    assert "test" not in info  # no stale "SEALED" claim from the W10 demo
    assert len(info["snr_grid_db"]) == 21
    assert [r["id"] for r in info["ratios"]] == ["r_1_6", "r_1_24"]


def test_chart_serves_g12_curves_with_intervals_only_for_multi_seed_curves(client: TestClient) -> None:
    six = client.get("/api/chart?ratio=r_1_6").json()
    assert six["split"] == "test"
    assert [s["id"] for s in six["series"]] == ["learned", "classical_adaptive"]
    learned = six["series"][0]
    assert learned["cells"] == 3 and len(learned["points"]) == 21
    first = learned["points"][0]
    assert first["snr_db"] == -8.0 and first["ci_low"] < first["accuracy"] < first["ci_high"]
    assert six["series"][1]["points"][0]["coverage"] == 0.0

    quarter = client.get("/api/chart?ratio=r_1_24").json()
    assert all(s["cells"] == 1 for s in quarter["series"])
    assert all("ci_low" not in p for s in quarter["series"] for p in s["points"])
    assert client.get("/api/chart?ratio=unknown").status_code == 422


def test_image_and_inference_routes_are_the_original_demos(client: TestClient) -> None:
    paths = sorted(route.path for route in client.app.routes if route.path.startswith("/api/"))
    assert paths == ["/api/chart", "/api/examples", "/api/examples/{name}", "/api/images",
                     "/api/images/{image_id}", "/api/infer", "/api/metadata"]
    gallery = client.get("/api/images").json()
    assert gallery["split"] == "train" and len(gallery["images"]) == 4
    bad = {"image_id": "f" * 16, "snr_db": -8, "ratio": "r_1_6"}
    assert client.post("/api/infer", json=bad).status_code == 404
