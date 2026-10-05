"""G-12 demo adapter: test-curve routes over the original demo's reused routes (not scientific evaluation)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from demo.backend.app import Evidence
from demo_g12.backend.app import create_app, load_g12
from demo_shared.test_gallery import TestGallery
from demo_shared.results import TEST_IMAGES


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(evidence=Evidence(), g12=load_g12(), gallery=TestGallery()))


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


def test_gallery_routes_serve_the_ten_test_images(client: TestClient) -> None:
    paths = sorted(route.path for route in client.app.routes if route.path.startswith("/api/"))
    assert paths == ["/api/chart", "/api/images", "/api/images/{image_id}", "/api/infer", "/api/metadata",
                     "/api/test-examples/{name}", "/api/test-images/{image_id}", "/api/test-split"]
    gallery = client.get("/api/images").json()
    assert gallery["split"] == "test" and len(gallery["images"]) == 10
    assert len({image["label"] for image in gallery["images"]}) == 10
    thumb = client.get(gallery["images"][0]["thumbnail_url"])
    assert thumb.headers["content-type"] == "image/png" and thumb.content[:8] == b"\x89PNG\r\n\x1a\n"
    bad = {"image_id": "f" * 16, "snr_db": -8, "ratio": "r_1_6"}
    assert client.post("/api/infer", json=bad).status_code == 404
    assert client.get("/api/test-images/zzzz").status_code == 404


def test_live_outcomes_carry_and_reproduce_the_g12_record(client: TestClient) -> None:
    image = client.get("/api/images").json()["images"][7]  # gas pump: DJSCC is wrong here in G-12
    body = client.post("/api/infer", json={"image_id": image["id"], "snr_db": -8, "ratio": "r_1_6"}).json()
    assert body["split"] == "test"
    learned, classical = body["learned"], body["classical"]
    assert learned["recorded"]["correct"] is False and classical["recorded"]["outage"] is True
    if learned["status"] == "unavailable" or classical["status"] == "unavailable":
        pytest.skip("frozen weights are not provisioned on this machine")
    assert learned["recorded"]["matches"] is True and classical["recorded"]["matches"] is True
    assert learned["predicted_label"] == learned["recorded"]["predicted_label"]


def test_the_whole_test_split_can_be_browsed_and_chosen(client: TestClient) -> None:
    listing = client.get("/api/test-split?label=7&page=2&size=40").json()
    if not listing["available"]:
        pytest.skip("the extracted test split or per-image records are not on this machine")
    assert (listing["total"], listing["pages"], listing["page"], len(listing["images"])) == (419, 11, 2, 40)
    assert {image["label"] for image in listing["images"]} == {"gas pump"}
    everything = client.get("/api/test-split?size=100&page=1000").json()
    assert everything["total"] == TEST_IMAGES and everything["page"] == everything["pages"] - 1
    chosen = listing["images"][0]["id"]
    assert client.get(f"/api/test-images/{chosen}").status_code == 200
    body = client.post("/api/infer", json={"image_id": chosen, "snr_db": 18, "ratio": "r_1_6"}).json()
    assert body["image_id"] == chosen and "recorded" in body["learned"]
    assert client.get("/api/test-split?label=10").status_code == 422
