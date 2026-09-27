#!/usr/bin/env python3
"""Offline API journey: genuine example inference, chart parity and SNR changes."""

import csv
import json
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = "http://127.0.0.1:" + str(__import__("os").environ.get("DEMO_PORT", "8000"))


def get(route: str) -> dict:
    with urllib.request.urlopen(BASE + route, timeout=20) as result:
        return json.load(result)


def infer(image_id: str, snr: int) -> tuple[dict, float]:
    body = json.dumps({"image_id": image_id, "snr_db": snr, "ratio": "r_1_6"}).encode()
    request = urllib.request.Request(BASE + "/api/infer", data=body, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as result:
        return json.load(result), time.perf_counter() - start


def main() -> None:
    metadata = get("/api/metadata")
    assert metadata["default_snr_db"] == -8
    assert metadata["test"] == "SEALED" and metadata["test_access"] == 0
    grid = metadata["snr_grid_db"]
    assert grid == list(range(-8, 8)) + [9, 11, 13, 15, 18]
    chart = get("/api/chart?ratio=r_1_6")
    assert chart["ratio"] == "r_1_6"
    rows = list(csv.DictReader((ROOT / "presentation-results/data/w10_primary_252.csv").open()))
    for arm in ("learned", "classical_adaptive"):
        series = next(s for s in chart["series"] if s["id"] == arm)
        assert [p["snr_db"] for p in series["points"]] == grid
        for snr in (-8, -4, 18):
            measured = next(r for r in rows if r["system"] == arm and r["bw_ratio"] == "r_1_6" and float(r["snr_db"]) == snr)
            point = next(p for p in series["points"] if p["snr_db"] == snr)
            assert point["accuracy"] == int(measured["n_correct"]) / 1000
            print(f"chart {arm} {snr:+} dB: {100 * point['accuracy']:.1f}%")
    gallery = get("/api/images")
    assert gallery["split"] == "train" and gallery["images"]
    image = gallery["images"][0]
    with urllib.request.urlopen(BASE + image["thumbnail_url"], timeout=20) as result:
        assert result.headers.get_content_type() == "image/png" and result.read(8) == b"\x89PNG\r\n\x1a\n"
    for snr in (-8, -4, 18):
        response, seconds = infer(image["id"], snr)
        assert response["image_id"] == image["id"] and response["snr_db"] == snr and response["ratio"] == "r_1_6"
        for arm in ("learned", "classical"):
            outcome = response[arm]
            assert outcome["status"] in {"delivered", "decode_failure", "codec_infeasibility", "structural_infeasibility", "unavailable"}
            if outcome["status"] == "unavailable":
                assert outcome.get("detail") and not outcome.get("image_url")
        print(f"example {snr:+} dB: learned={response['learned']['status']}, classical={response['classical']['status']}, wall={seconds:.2f}s")
    print("PASS: exact published chart values, genuine-or-explicitly-unavailable example responses, no test access")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise
