"""Offline, read-only demo adapter tests (not scientific evaluation)."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from demo.backend.app import Evidence, create_app


@pytest.fixture(scope="module")
def evidence() -> Evidence:
    return Evidence()


@pytest.fixture(scope="module")
def client(evidence: Evidence) -> TestClient:
    return TestClient(create_app(evidence=evidence))


def test_metadata_preserves_sealed_validation_boundary(client: TestClient, evidence: Evidence) -> None:
    response = client.get("/api/metadata")
    assert response.status_code == 200
    info = response.json()
    assert (info["split"], info["test"], info["test_access"]) == ("val", "SEALED", 0)
    assert info["inference_available"] == (evidence.live.available or evidence.classical.available)
    assert info["classical_inference_available"] == evidence.classical.available
    assert len(info["snr_grid_db"]) == 21
    assert len(info["systems"]) == 12
    assert info["default_snr_db"] == -8
    assert info["ratios"][0] == {"id": "r_1_6", "label": "1/6", "channel_uses": 12800}
    assert info["portable_examples_split"] == "train"


def test_chart_is_complete_and_count_derived(client: TestClient) -> None:
    data = client.get("/api/chart?ratio=r_1_6").json()
    assert data["split"] == "val"
    assert data["ratio"] == "r_1_6"
    assert [series["id"] for series in data["series"]] == ["learned", "classical_adaptive", "learned_snr_randomised"]
    assert all(
        len(series["points"]) == 21 for series in data["series"]
    )
    learned = data["series"][0]
    assert learned["points"][0]["snr_db"] == -8
    assert learned["points"][0]["accuracy"] == 0.728
    assert client.get("/api/chart?ratio=r_1_24").json()["ratio"] == "r_1_24"
    assert client.get("/api/chart?ratio=unknown").status_code == 422


def test_infer_rejects_untrusted_inputs(client: TestClient, evidence: Evidence) -> None:
    image_id = next(iter(evidence.examples))
    body = {"image_id": image_id, "snr_db": -8, "ratio": "r_1_6"}
    for bad, expected in (
        ({**body, "image_id": "../test/dataset"}, 422),
        ({**body, "image_id": "f" * 16}, 404),
        ({**body, "snr_db": 8}, 422),
        ({**body, "ratio": "r_1_12"}, 422),
        ({**body, "split": "test"}, 422),
        ({**body, "bw_ratio": "r_1_6"}, 422),
        ({**body, "snr_db": True}, 422),
    ):
        assert client.post("/api/infer", json=bad).status_code == expected


def test_gallery_is_portable_train_only(client: TestClient, evidence: Evidence) -> None:
    response = client.get("/api/images")
    assert response.status_code == 200
    assert response.json()["split"] == "train"
    assert len(response.json()["images"]) == 4
    assert [r["id"] for r in response.json()["images"]] == list(evidence.examples)
    assert client.get("/api/images?split=test").status_code == 422
    assert client.get("/api/images/../../test").status_code in (400, 404)
    assert client.get("/api/images/ffffffffffffffff").status_code == 404
    assert client.get("/api/examples/../test").status_code in (400, 404)
    assert client.get("/api/examples/not-bundled").status_code == 404


def test_png_is_local_canonical_bundled_image_and_cached(client: TestClient, evidence: Evidence) -> None:
    image_id = next(iter(evidence.examples))
    name = evidence.examples[image_id]["id"]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: client.get(f"/api/examples/{name}"), range(3)))
    assert all(response.status_code == 200 for response in results)
    assert all(response.headers["content-type"] == "image/png" for response in results)
    assert all(response.content == results[0].content for response in results)
    assert results[0].content.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(evidence._png_cache) == 1


def test_live_learned_and_classical_end_to_end(client: TestClient, evidence: Evidence) -> None:
    if not (evidence.live.available and evidence.classical.available):
        pytest.skip("live CPU integration requires two separately provisioned, SHA-verified frozen checkpoints")
    image_id = next(iter(evidence.examples))
    body = {"image_id": image_id, "snr_db": -8, "ratio": "r_1_6"}
    low = client.post("/api/infer", json=body)
    assert low.status_code == 200
    low_result = low.json()
    assert low_result["image_id"] == image_id and low_result["ratio"] == "r_1_6"
    assert low_result["split"] == "train" and low_result["input_image_url"].startswith("/api/examples/")
    assert low_result["learned"]["status"] == "delivered"
    assert low_result["learned"]["predicted_label"] == "English springer"
    assert 0 < low_result["learned"]["confidence"] < 1
    assert low_result["learned"]["image_url"].startswith("data:image/png;base64,")
    assert low_result["classical"]["status"] == "decode_failure"
    assert low_result["classical"]["predicted_label"] == "tench"  # BR-13 measured fallback, not a decoded score
    assert low_result["classical"]["confidence"] is None
    assert low_result["classical"]["image_url"] is None
    assert low_result["classical"]["noise_id"] == low_result["learned"]["noise_id"]
    again = client.post("/api/infer", json=body).json()
    assert again["learned"]["cache_hit"] and again["classical"]["cache_hit"]
    assert again["learned"]["image_url"] == low_result["learned"]["image_url"]
    high = client.post("/api/infer", json={**body, "snr_db": 7})
    assert high.status_code == 200
    assert high.json()["classical"]["status"] == "delivered"
    assert high.json()["classical"]["image_url"].startswith("data:image/png;base64,")
    assert 0 < high.json()["classical"]["confidence"] < 1
    assert high.json()["classical"]["noise_id"] == high.json()["learned"]["noise_id"]
    for snr in (-4, 18):
        point = client.post("/api/infer", json={**body, "snr_db": snr})
        assert point.status_code == 200
        assert point.json()["classical"]["status"] == "delivered"
        assert point.json()["learned"]["status"] == "delivered"
        assert point.json()["classical"]["noise_id"] == point.json()["learned"]["noise_id"]
    efficiency = client.post("/api/infer", json={**body, "ratio": "r_1_24"}).json()
    assert efficiency["classical"]["status"] == "unavailable"
    assert efficiency["learned"]["status"] == "unavailable"


@pytest.mark.parametrize("verdict,has_pixels", [
    ("decode_failure", False), ("codec_infeasibility", False),
    ("structural_infeasibility", False), ("delivered", True),
])
def test_classical_verdict_has_no_fabricated_image(monkeypatch: pytest.MonkeyPatch, evidence: Evidence, verdict: str, has_pixels: bool) -> None:
    """The typed result layer never turns an outage into a decoded image."""
    if not evidence.classical.available:
        pytest.skip("typed classical path requires the separately provisioned artifact checkpoint")
    from baseline.classical.pipeline import DELIVERED, DECODE_FAILURE, CODEC_INFEASIBILITY, STRUCTURAL_INFEASIBILITY

    assert verdict in {DELIVERED, DECODE_FAILURE, CODEC_INFEASIBILITY, STRUCTURAL_INFEASIBILITY}
    from demo.backend.app import create_app

    def stub(*args, **kwargs):
        return {"status": verdict, "predicted_label": "tench", "confidence": .9 if has_pixels else None,
                "image_url": "data:image/png;base64,AAAA" if has_pixels else None, "detail": "typed mocked transport"}

    monkeypatch.setattr(evidence.classical, "infer", stub)
    image_id = next(iter(evidence.examples))
    response = TestClient(create_app(evidence=evidence)).post("/api/infer", json={"image_id": image_id, "snr_db": -8, "ratio": "r_1_6"})
    assert response.status_code == 200
    assert (response.json()["classical"]["image_url"] is not None) is has_pixels


def test_evidence_fails_closed_on_tampered_manifest(tmp_path: Path) -> None:
    from demo.backend.app import ROOT

    base = tmp_path / "results/learned/w10"
    base.mkdir(parents=True)
    for name in (
        "w10_continuation_closeout_v11.json",
        "w10_continuation_units_v11.json",
        "w10_continuation_unit_manifest_v11.json",
        "w10_continuation_per_image_manifest_v11.json",
    ):
        (base / name).write_bytes((ROOT / "results/learned/w10" / name).read_bytes())
    payload = json.loads((base / "w10_continuation_unit_manifest_v11.json").read_text())
    payload["tamper"] = True
    (base / "w10_continuation_unit_manifest_v11.json").write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="differs from closeout"):
        Evidence(tmp_path)


def test_evidence_fails_closed_on_presentation_csv_drift(tmp_path: Path) -> None:
    from demo.backend.app import ROOT, PRIMARY_CSV

    base = tmp_path / "results/learned/w10"
    base.mkdir(parents=True)
    for name in (
        "w10_continuation_closeout_v11.json",
        "w10_continuation_units_v11.json",
        "w10_continuation_unit_manifest_v11.json",
        "w10_continuation_per_image_manifest_v11.json",
    ):
        (base / name).write_bytes((ROOT / "results/learned/w10" / name).read_bytes())
    target = tmp_path / PRIMARY_CSV
    target.parent.mkdir(parents=True)
    rows = (ROOT / PRIMARY_CSV).read_text()
    target.write_text(rows.replace("0,learned,", "0,classical_adaptive,", 1))
    with pytest.raises(RuntimeError, match="presentation CSV differs"):
        Evidence(tmp_path)


def test_arms_fail_independently_without_fabrication(monkeypatch: pytest.MonkeyPatch, evidence: Evidence) -> None:
    if not evidence.live.available:
        pytest.skip("independent-arm live integration requires the separately provisioned learned checkpoint")
    def broken(*args, **kwargs):
        raise RuntimeError("deliberate local loader failure")

    monkeypatch.setattr(evidence.classical, "infer", broken)
    image_id = next(iter(evidence.examples))
    response = TestClient(create_app(evidence=evidence)).post("/api/infer", json={"image_id": image_id, "snr_db": -8, "ratio": "r_1_6"})
    assert response.status_code == 200
    assert response.json()["classical"]["status"] == "unavailable"
    assert response.json()["classical"]["image_url"] is None
    assert response.json()["learned"]["status"] == "delivered"


def test_absent_frozen_assets_fail_closed(monkeypatch: pytest.MonkeyPatch, evidence: Evidence) -> None:
    monkeypatch.setattr(evidence.live, "checkpoint_bytes", None)
    monkeypatch.setattr(evidence.classical, "checkpoint_bytes", None)
    client = TestClient(create_app(evidence=evidence))
    assert client.get("/api/metadata").json()["inference_available"] is False
    image_id = next(iter(evidence.examples))
    result = client.post("/api/infer", json={"image_id": image_id, "snr_db": -8, "ratio": "r_1_6"}).json()
    for arm in ("learned", "classical"):
        assert result[arm]["status"] == "unavailable"
        assert result[arm]["predicted_label"] is None
        assert result[arm]["confidence"] is None
        assert result[arm]["image_url"] is None
