# Signal Lab · exhibition frontend

React 18 + TypeScript + Vite + Tailwind. **This directory owns presentation only.** All measured values come from the local API backed by frozen W10 validation evidence; per-image outputs come from separate, read-only inference. No fixtures, invented predictions, metric recomputation, training or test access are shipped in production. Development fonts are bundled, and the browser makes no CDN request.

## Run

```bash
cd demo/frontend
npm ci
npm run dev             # http://localhost:5173 ; /api proxies to 127.0.0.1:8000
npm test
npm run build
```

Set `DEMO_API_URL=http://127.0.0.1:PORT` before `npm run dev` to change the Vite proxy target. In production, serve `dist/` on the same origin as `/api` (or reverse-proxy `/api`); static files can be served without internet. API-provided media URLs should be same-origin local URLs. A missing API is reported explicitly, not replaced with synthetic results.

## Backend contract (JSON, same-origin)

- `GET /api/metadata` → `{ "dataset": "Imagenette-160 validation", "evidence_label": "W10 v11 closeout · validation only", "snr_grid_db": [-8, ..., 18], "default_snr_db": -8, "ratios": [{"id":"r_1_6","label":"1/6","channel_uses":12800},{"id":"r_1_24","label":"1/24","channel_uses":3200}], "inference_available": false, "inference_reason": "Optional honest reason" }`. The full **21-element measured grid** must be returned in increasing order. `source_commit`, `inference_available`, and `inference_reason` are optional, but the UI displays an unavailable state when provided. No test data. **Default SNR is −8 dB**, not −4 dB.
- `GET /api/chart?ratio=r_1_6` → `{ "ratio":"r_1_6", "series":[{"id":"learned","label":"Learned DJSCC","points":[{"snr_db":-8,"accuracy":0.728,"coverage":1.0}, ...]},{"id":"classical","label":"Adaptive classical","points":[...]}] }`. Each series has **exactly one point for every metadata grid entry, in identical order**, with accuracy expressed as **fraction 0–1** (`n_correct / 1000`), not percent. Put the headline learned series first, adaptive classical second; the UI supports up to four displayed series. Preserve real outage values and avoid interpolated points. The chart is aggregate published validation evidence; it does not change when an image changes.
- `GET /api/images` → `{ "split":"train", "images":[{"id":"opaque-stable-id","label":"Training example A","thumbnail_url":"/api/examples/...","truth_label":"optional train label","example_split":"train"}] }`. Return **four legitimate bundled TRAIN images**, not validation or test samples. Each row requires `example_split: "train"` or `split: "train"`. No raw host filesystem paths. Image content is served locally by the backend.
- `POST /api/infer`, `Content-Type: application/json`, body `{ "image_id":"opaque-stable-id", "snr_db":-8, "ratio":"r_1_6" }` → `{ "image_id":"opaque-stable-id", "snr_db":-8, "ratio":"r_1_6", "input_image_url":"/api/assets/...", "classical":{ "status":"unavailable", "predicted_label":null, "confidence":null, "image_url":null, "detail":"Checkpoint not present" }, "learned":{ "status":"unavailable", "predicted_label":null, "confidence":null, "image_url":null, "detail":"Checkpoint not present" }, "reason":"Image-level inference unavailable" }`. If real inference is available, use status `delivered`, genuine labels, confidence and local image URLs. `confidence` is fraction 0–1, or `null` if unavailable; label and image URL may also be `null`. Valid status: `delivered`, `decode_failure`, `codec_infeasibility`, `structural_infeasibility`, `unavailable`. For outages, return the real fallback prediction if applicable, with a non-delivered status; do not represent fallback as delivery. For unavailable, give an honest reason and no simulated fallback. Echo the exact requested ID/SNR/ratio so the frontend can reject stale/mismatched results. Request cancellation is supported via browser AbortController; server need not implement cancellation. HTTP errors should include `{"detail":"explanation"}`.

All fetched image URLs must remain within the offline local service. The frontend rejects non-grid curves and mismatched inference identities, cancels superseded requests, and shows service errors instead of invented content. The SNR control indexes the measured grid, not a continuous interval. The **Recharts numeric X-axis** uses the measured −8…+18 dB positions; its `ReferenceLine` marks the selected point. The chart uses the report's W10 colors: learned `#0072B2`, adaptive classical `#D55E00`, optional randomized `#009E73`.

**Evidence / compliance:** `presentation-results/report/findings.md` and `presentation-results/data/w10_primary_252.csv` document the 252 published validation units (12 arms × 21 SNRs); results are descriptive and test is sealed. This UI implements the visual interaction intent of DR-1–3, 5–6 but cannot by itself certify DR-4: current `spec/SPEC.md` names Streamlit and a shared Python thesis-figure style module. A React frontend is a user-requested presentation alternative, not a silent spec amendment. Backend availability and CPU-only offline end-to-end behavior require an integration test with that service.
