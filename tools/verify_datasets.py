#!/usr/bin/env python3
"""Verify real archives, manifests, model-facing splits, and test-scan isolation."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402
from data import adapters, preprocessing  # noqa: E402
from data.manifests import check_manifest, materialize_manifest_bytes  # noqa: E402
from data.provenance import configured_datasets, verify_extracted_dataset  # noqa: E402
from data.registry import load_dataset  # noqa: E402


def _forbidden(name: str, calls: dict[str, int]):
    def fail(*_args, **_kwargs):
        calls[name] += 1
        raise RuntimeError(f"provenance-only scan called forbidden path {name}")

    return fail


def main() -> int:
    for dataset in configured_datasets():
        provenance = verify_extracted_dataset(dataset, REPO)
        manifest_hash = check_manifest(dataset, REPO)
        samples: list[str] = []
        for split in ("train", "val"):
            loaded = load_dataset(dataset, split, REPO)
            product, label = loaded[0]
            expected_shape = tuple(get(f"datasets.{dataset}.image_size"))
            if product.canonical_image.shape != expected_shape:
                raise RuntimeError(
                    f"{dataset}/{split}: canonical shape "
                    f"{product.canonical_image.shape} != {expected_shape}"
                )
            source = loaded.source_sample(0)
            if product.stable_sample_id != source.stable_sample_id:
                raise RuntimeError(
                    f"{dataset}/{split}: canonical/source stable IDs differ"
                )
            samples.append(
                f"{split}:{product.stable_sample_id}:label={label}:"
                f"shape={product.canonical_image.shape}"
            )

        calls = {"decoder": 0, "canonicalize_source": 0}
        original_decoders = adapters._DECODERS
        original_canonicalize = preprocessing.canonicalize_source
        try:
            adapters._DECODERS = {
                name: _forbidden("decoder", calls)
                for name in original_decoders
            }
            preprocessing.canonicalize_source = _forbidden(
                "canonicalize_source",
                calls,
            )
            regenerated = materialize_manifest_bytes(dataset, REPO)
            if not regenerated:
                raise RuntimeError(f"{dataset}: empty regenerated manifest")
        finally:
            adapters._DECODERS = original_decoders
            preprocessing.canonicalize_source = original_canonicalize
        if calls != {"decoder": 0, "canonicalize_source": 0}:
            raise RuntimeError(
                f"{dataset}: provenance-only test scan call counts {calls}"
            )

        print(
            f"{dataset}: archive_bytes={provenance.byte_length} "
            f"archive_sha256={provenance.sha256} "
            f"manifest_sha256={manifest_hash} "
            f"samples=[{'; '.join(samples)}] "
            "test_scan_decoder_calls=0 test_scan_canonicalization_calls=0"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
