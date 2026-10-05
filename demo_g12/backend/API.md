# G-12 demo backend: frontend contract

The routes and response shapes are those of `demo/backend/API.md`, with two exceptions.

| Method | Path | Response |
| --- | --- | --- |
| GET | `/api/metadata` | `split: "test"`, `n_images: 3925`, `evidence_label: "G-12 test split · 3,925 images"`, `snr_grid_db` (the 21 measured SNRs), `default_snr_db: -8`, `ratios` for `r_1_6` and `r_1_24`, and the inference availability fields. There is no `test: "SEALED"` field. |
| GET | `/api/chart?ratio=r_1_6` | `{ratio, split: "test", series: [{id, label, color, cells, points: [{snr_db, accuracy, coverage, ci_low?, ci_high?}]}]}` for `learned` (own task head) and `classical_adaptive` (artifact-fine-tuned scorer). `accuracy` is the mean over seed cells on the 3,925 test images. `ci_low`/`ci_high` (95% image bootstrap) are present only when `cells > 1`. |

| GET | `/api/images` | `{split: "test", images: [{id, label, truth_label, thumbnail_url, split: "test"}]}`: the ten default test images (AM-101), one per class. |
| GET | `/api/test-examples/{name}` | Canonical 160×160 PNG of a default test image. |
| GET | `/api/test-split?page=0&size=40&label=7` | `{available, reason, total, page, pages, size, images}` over all 3,925 test images in split-manifest order, optionally one class. When the extracted dataset or per-image records are missing, `available: false` with a reason. |
| GET | `/api/test-images/{stable_id}` | Canonical PNG of any test image the gallery knows. |
| POST | `/api/infer` | As in `demo/`, for any test image. Each arm gains `recorded: {predicted_label, correct, outage, noise_id, matches}`, the G-12 seed-cell-0 record for that image and SNR, where `matches` says whether the live outcome reproduces it (`null` when the arm is unavailable). Ratio `r_1_24` returns both arms unavailable. |

`/api/images/{image_id}` (optional validation images) comes from the original backend unchanged; the training-example routes are removed. Start it from the repository root:

```bash
.venv/bin/python -m uvicorn demo_g12.backend.app:app --host 127.0.0.1 --port 8001
```
