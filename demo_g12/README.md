# Exhibition demo · G-12 test edition

```bash
./run-g12-demo.sh
```

This is a copy of the exhibition dashboard in `demo/`, with the chart showing the **final G-12 test results** in place of W10 validation data. Run the launcher from the repository root. It opens `http://127.0.0.1:8001` (`--port`, `--no-browser`) and builds the page on first run if `dist/` is missing, which needs Node/npm. `?snr=-4&ratio=r_1_24` deep-links to a measured point.

The three demos:

| Command | Folder | Chart data | Style |
| --- | --- | --- | --- |
| `./run-demo.sh` | `demo/` | W10 validation (1,000 images, one seed) | Visitor dashboard (React) |
| `./run-g12-demo.sh` | `demo_g12/` | G-12 test (3,925 images, mean of 3 seeds at 1/6) | Visitor dashboard (React) |
| `./run-streamlit-demo.sh` | `demo_streamlit/` | G-12 test, plus the ER-9 control | Academic, paper figures (Streamlit) |

## What differs from `demo/`

- **Backend.** `backend/app.py` builds the original `demo.backend.app` application and replaces only `/api/metadata` and `/api/chart`. The gallery, inference, outage handling, checkpoint verification and input validation are the original code, unchanged. The weights and example images are shared, so provisioning `demo/` (see `demo/README.md`, "Frozen weights") provisions this demo too.
- **Chart data.** `presentation-results/data/g12_test_curves.csv`, read through `demo_streamlit/results.py`. That module checks every served point against `results/g12/results.csv` at start-up. At 1/6 the curves are means over three seed pairs and carry 95% image-bootstrap bands. At 1/24 they are one seed pair, so no band is drawn, as in the paper.
- **Frontend.** The interval band, the "Final test · 3,925 pictures" heading, and the low-SNR sentence now quotes the measured AI accuracy. The old wording, "most pictures right", is false at 1/24, where DJSCC scores 47% at −8 dB.

## Presenter notes

- The chart is the final test result: 3,925 held-out images, and at 1/6 the mean of three seed pairs. Live outputs still come from the four **training-split** examples, so they illustrate the mechanism. They are not held-out evidence.
- At 1/6 the digital link delivers nothing from −8 to −5 dB and everything from −4 dB up. At −4 and −3 dB the two links are within a point of each other. From −2 dB up the digital link leads by about 2–4 points. The digital baseline is re-tuned per SNR, while DJSCC was trained once at 7 dB.
- Live inference runs only at 1/6. At 1/24 both cards say "unavailable", and nothing is substituted.

## Checks

```bash
.venv/bin/python -m pytest -q tests/test_demo_g12_backend.py
(cd demo_g12/frontend && npm run test && npm run build)
DEMO_PORT=8001 .venv/bin/python demo_g12/scripts/smoke_api.py   # with the demo running
```
