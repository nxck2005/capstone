# G-12 demo backend: frontend contract

The routes and response shapes are those of `demo/backend/API.md`, with two exceptions.

| Method | Path | Response |
| --- | --- | --- |
| GET | `/api/metadata` | `split: "test"`, `n_images: 3925`, `evidence_label: "G-12 test split · 3,925 images"`, `snr_grid_db` (the 21 measured SNRs), `default_snr_db: -8`, `ratios` for `r_1_6` and `r_1_24`, and the inference availability fields. There is no `test: "SEALED"` field. |
| GET | `/api/chart?ratio=r_1_6` | `{ratio, split: "test", series: [{id, label, color, cells, points: [{snr_db, accuracy, coverage, ci_low?, ci_high?}]}]}` for `learned` (own task head) and `classical_adaptive` (artifact-fine-tuned scorer). `accuracy` is the mean over seed cells on the 3,925 test images. `ci_low`/`ci_high` (95% image bootstrap) are present only when `cells > 1`. |

`/api/images`, `/api/examples/{name}`, `/api/images/{image_id}` and `/api/infer` come from the original backend unchanged. Start it from the repository root:

```bash
.venv/bin/python -m uvicorn demo_g12.backend.app:app --host 127.0.0.1 --port 8001
```
