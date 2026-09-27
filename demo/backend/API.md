# Offline demo backend: frontend contract

Install the project's CPU runtime lock and the separate `demo/backend/requirements.txt`; run from repository root:

```bash
.venv/bin/python -m uvicorn demo.backend.app:app --host 127.0.0.1 --port 8000
```

The four portable gallery examples are **training-split illustration inputs** authenticated against the committed Imagenette manifest and their exact source JPEG SHA-256s. They are **not held-out evaluation examples**. The chart independently uses the 252 **published validation** units (1,000 validation images per point), with parity checked against `presentation-results/data/w10_primary_252.csv`; it is not recomputed from gallery requests. Final test is sealed. No network download, training, uploaded image, dynamic filesystem path, or test loader exists in the API.

| Method | Path | Response |
| --- | --- | --- |
| GET | `/api/metadata` | `snr_grid_db` (the 21 measured SNRs), `default_snr_db: -8`, `ratios: [{id,label,channel_uses}]`, `evidence_label`, `split: val`, `test: SEALED`. Only `r_1_6` has live learned/classical support; `r_1_24` has chart data but live arms report unavailable. |
| GET | `/api/chart?ratio=r_1_6` | `{ratio,series:[{id,label,color,points:[{snr_db,accuracy,coverage}]}]}`. Ordered learned, adaptive classical, randomized-SNR learned at 1/6; learned and adaptive classical at 1/24. Colors match `presentation-results/plot_results.py` (`#0072B2`, `#D55E00`, `#009E73`). `accuracy` is authenticated `n_correct / n_total`, not an individual prediction. No interpolated measurements. |
| GET | `/api/images` | `{split:'train',images:[{id,label,truth_label,thumbnail_url,example_split:'train'}]}` for four portable train examples. `id` is the original source-byte stable ID. |
| GET | `/api/examples/{name}` | Canonical 160×160 RGB PNG from an authenticated bundled training JPEG. |
| GET | `/api/images/{image_id}` | Optional **validation-only** image from the verified full local dataset, when the extracted archive exists; portable gallery does **not** depend on it. |
| POST | `/api/infer` | Body `{image_id,snr_db,ratio}`; returns `{image_id,ratio,snr_db,input_image_url,classical,learned,split}`. Each arm has `{status,predicted_label,confidence,image_url,detail}`. Status is `delivered`, `decode_failure`, `codec_infeasibility`, `structural_infeasibility`, or `unavailable`. Invalid or non-gallery/non-validation IDs 404; unsupported ratio/SNR 422. |

At `r_1_6`, learned inference uses the SHA-256-verified frozen W8 epoch-92 checkpoint (train/channel seed 0) in `demo/assets/checkpoints/learned-r1-6.pt`; it calls the scientific `build_djscc`, `evaluation_input`, `scheduled_noise_id` and `keyed_complex_noise` on CPU. Its label comes from the task head. Confidence is **uncalibrated softmax**, and reconstruction PNG is a **visualization**, not a reported metric. The classical path uses the frozen W10 binding, G8/F3 pass-two candidate selection per SNR, `J2KCodec` into ignored `demo/backend/.cache/j2k`, the genuine JPEG 2000 + LDPC + AWGN `run_classical_pipeline`, frozen BR-13 outage policy and SHA-256-verified BR-12 artifact scorer in `demo/assets/checkpoints/artifact-classifier.pt`. Delivered classical confidence is uncalibrated softmax; on any outage, `predicted_label` is the frozen constant fallback, but **confidence and image are null**. An outage is not a delivered transmission. The local artifact scorer is constructed without fetching G1 weights. Both checkpoint snapshots are hashed before parsing; loading uses `torch.load(weights_only=True)`. Checkpoint mismatch makes that arm unavailable rather than downloading or fabricating outputs.

The chart and live example outputs have different roles: train gallery responses are fresh illustrative CPU inference, not published W10 per-image evidence, and cannot substantiate a held-out accuracy claim. Keyed noise is deterministic for the stable ID, ratio, SNR and frozen channel seed. A 64-entry per-arm in-memory result cache and a 32-entry canonical PNG cache use locks; J2K codestream cache is private and ignored by Git. Classical first request can take seconds; cached requests are milliseconds. The `inference_ms` field (in each arm) excludes checkpoint lazy loading and local image acquisition, so benchmark complete HTTP wall time separately for cold starts.

Measured on this development CPU with the bundled Springer example: first combined request at −8 dB ~4.1 s wall (learned checkpoint load + classical transport), warmed classical +7 dB ~3.7 s wall and delivered; repeated +7 dB request ~10 ms wall. CPU speed and OpenJPEG setup vary by laptop. The runtime requires OpenJPEG 2.5.4 and the project's CPU inference dependencies; no GPU or network is required.
