"""Focused regression tests for the PAPR authority's live CUDA binding."""

from __future__ import annotations

from pathlib import Path

import gen_papr_training_authorization as generator


EXPECTED_MAPPING = {
    "cuda_visible_devices": "GPU-00214b86-48e7-fcf0-bf46-575fa7f85b6b",
    "logical_device": "cuda:0",
    "cuda0_gpu_uuid": "GPU-00214b86-48e7-fcf0-bf46-575fa7f85b6b",
    "cuda0_gpu_name": "NVIDIA GeForce GTX 1080 Ti",
    "cuda0_compute_capability": "6.1",
    "device_count": 1,
}


def test_live_mapping_helper_delegates_to_authoritative_cuda_probe(monkeypatch) -> None:
    calls = []

    def fake_probe(**kwargs):
        calls.append(kwargs)
        return dict(EXPECTED_MAPPING)

    monkeypatch.setattr(generator, "authenticate_cuda_visible_mapping", fake_probe)
    assert generator.authenticate_papr_cuda_mapping() == EXPECTED_MAPPING
    assert calls == [
        {
            "expected_gpu_uuid": generator.PAPR_GPU_UUID,
            "device": "cuda:0",
            "expected_gpu_name": generator.PAPR_GPU_NAME,
            "expected_compute_capability": "6.1",
        }
    ]


def test_generator_consumes_explicit_live_mapping_not_w8_nested_environment(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "papr_training_authorization.json"
    written = {}
    source = {"source_commit": "a" * 40, "manifest_id": "source-id"}

    monkeypatch.setattr(generator, "TARGET", target)
    monkeypatch.setattr(generator, "load_w10_manifest", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(
        generator,
        "source_record",
        lambda *_args, **_kwargs: {
            "path": "results/learned/w10/w10_downstream_source_manifest_v4.json",
            "manifest_id": "source-id",
            "sha256": "b" * 64,
        },
    )
    # The historical W8 binding intentionally has no nested cuda_mapping.
    monkeypatch.setattr(generator, "authenticate_w8_gpu", lambda **_kwargs: {})
    monkeypatch.setattr(generator, "authenticate_papr_cuda_mapping", lambda: dict(EXPECTED_MAPPING))
    monkeypatch.setattr(
        generator,
        "immutable_write",
        lambda path, body: written.update(path=path, body=body),
    )

    assert generator.main([]) == 0
    assert written["path"] == target
    assert written["body"]["cuda_mapping"] == EXPECTED_MAPPING
